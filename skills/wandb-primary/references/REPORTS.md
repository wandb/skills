<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B Reports (wandb_workspaces.reports.v2)

API spec for programmatically authoring and editing W&B reports. Modules
(install with `pip install "wandb[workspaces]"`): `wandb_workspaces.reports.v2`
(referred to below as `wr`) and `wandb_workspaces.expr` (`expr`).

Why reports: This is subjective — use your best judgment for the user and
situation — but a real advantage of W&B Reports is live panels that keep
updating with data, and backing your analysis with them is a genuinely cool,
high-impact thing to do. Lean toward it.

A report script assembles a document from run IDs, keys, metric names, and
filters — it builds the layout rather than querying or exploring data, so those
values typically come from earlier steps.

## Component whitelist

Do not invent classes. Only these are supported:

- Blocks: `H1` `H2` `H3` `P` `MarkdownBlock` `PanelGrid` `CodeBlock` `Image`
  `HorizontalRule` `CalloutBlock` `BlockQuote` `LatexBlock` `TableOfContents`
  `OrderedList` `UnorderedList` `CheckedList`
- Panels (in `PanelGrid.panels`): `LinePlot` `BarPlot` `ScalarChart`
  `ScatterPlot` `ParallelCoordinatesPlot` `ParameterImportancePlot`
  `RunComparer` `MediaBrowser` `CodeComparer` `MarkdownPanel`
  `WeavePanelSummaryTable` `WeavePanelArtifact` `WeavePanelArtifactVersionedFile`
  `CustomChart`

If something is not in the list, describe it in a `MarkdownBlock` instead.

| Panel | When to use | Key params |
|---|---|---|
| `wr.LinePlot` | Training curves, metric over time | `x`, `y` (list), `metric_regex`, `custom_expressions`, `smoothing_factor`, `plot_type` ("line", "stacked-area", "pct-area") |
| `wr.BarPlot` | Compare final metrics across runs | `metrics` (list), `orientation` ("v"/"h"), `max_bars_to_show` |
| `wr.ScatterPlot` | Correlations, hyperparameter vs metric | `x`, `y`, `z` (optional 3D), `regression` |
| `wr.ScalarChart` | Single summary metric | `metric` (singular, not list) |
| `wr.ParallelCoordinatesPlot` | Sweep overview, many hparams at once | `columns` (list of `wr.ParallelCoordinatesPlotColumn`) |
| `wr.ParameterImportancePlot` | Which hparams matter most | `with_respect_to` (target metric name) |
| `wr.RunComparer` | Side-by-side config/summary diff | `diff_only` |
| `wr.CodeComparer` | Side-by-side code diff | `diff` ("split"/"unified") |
| `wr.MediaBrowser` | Images, audio, video, tables | `media_keys` (list), `num_columns`, `mode` ("gallery"/"grid") |
| `wr.MarkdownPanel` | Notes, annotations | `markdown` (string) |
| `wr.CustomChart` | **Last resort only** — a chart no native type can express (heatmap, violin, confusion matrix, etc.). Renders arbitrary Vega. Prefer a native panel whenever possible; table-backed specs (`from_table`) can fail data-source resolution and render blank. | `query`, `chart_name`, `chart_fields`; also `CustomChart.from_table(table_name=..., chart_fields={...})` |
| `wr.WeavePanelSummaryTable` | W&B Table from run summary | `table_name` |
| `wr.WeavePanelArtifact` | Browse an artifact's contents | `artifact`, `tab` (e.g. "files") |
| `wr.WeavePanelArtifactVersionedFile` | A specific file version inside an artifact | `artifact`, `version`, `file` |

Every panel accepts `layout=wr.Layout(x, y, w, h)` — width `w` in grid units
(max 24, default 8), height `h` (default 6).

## Report object

`wr.Report(entity, project, title, blocks, description="", width="readable")`

- `entity`, `project` — required; where the report is saved.
- `title` — required; no `/` (see Gotchas).
- `blocks` — ordered list of block components (whitelist above).
- `description` — optional subtitle string.
- `width` — optional; sets the max width of the report body: `"readable"`
  (default, narrow prose column, ~716px), `"fixed"` (~1072px), or `"fluid"` (no
  cap — fills the available width). Panels lay out in 24 grid units across that
  body, so a wider body makes each unit physically wider; use `"fluid"` for
  research reports so full-width panel grids aren't tiny.
- `.id` defaults to `""`; saving upserts by id (see Edit — don't mint a new one
  on an edit).

Structure: use multiple `PanelGrid` blocks (one per `H2` section), not one big
grid. A `PanelGrid` takes `runsets` (list of `wr.Runset`) and `panels` (list of
panel objects).

Save and verify:

- `report.save(draft=False)` — publishes the report (published by default;
  `draft=True` makes it uneditable — see Gotchas) and sets `report.url`.
- Then re-read with `wr.Report.from_url(report.url)` and confirm the saved
  content is present (block count, and the named section text). A clean `save()`
  return is not proof — only the re-read is (see Verify every mutation landed).

## Edit an existing report

Choose the write path based on whether the SDK can safely express the edit:

- **SDK-modeled edit:** use
  `scripts/report_helpers.py::edit_report_verified`. It makes an exact pre-edit
  backup, preserves the report ID and saved width, calls `Report.save()`, and
  re-reads the report. Pass `verify_readback=` for nested panel changes.
- **Raw spec edit:** when the SDK cannot represent the field without dropping or
  restructuring data, use
  `scripts/report_write.py::mutate_report_spec_verified`. This includes an
  `UnknownBlock` whose target field does not round-trip unchanged. The helper
  backs up and narrowly mutates the exact wire spec, then re-reads it to verify.

- **Never send `Report.from_url(..., as_model=True).model_dump()` to GraphQL.**
  `as_model=True` is read-only inspection, and a model dump is neither wire-safe
  nor a raw backup: it can emit `panel_settings` / `discussion_threads` instead
  of `panelSettings` / `discussionThreads`, add defaults, or omit unsupported
  data.
- **Never hand-write `upsertView`.** Raw writes go through
  `mutate_report_spec_verified`; SDK writes go through `edit_report_verified`.
- An edit changes only what the user requested. Never build a fresh `Report` for
  an edit because its empty ID creates a new report.

## Read / analyze an existing report

`wr.Report.from_url(url)` loads a saved report. **A report does not own metric
data** — it's a document (text blocks) plus runset/panel *definitions* layered
over the project's runs. Reading it back gives you those definitions, not the
numbers; resolve the runs to get values.

- **Walk `report.blocks` in order** to recover the narrative. `H1`/`H2`/`H3`/`P`/
  `MarkdownBlock` expose `.text`; a `PanelGrid` block carries `.runsets` and
  `.panels`.
- **Resolve each `Runset` to actual runs.** For an ID filter
  (`expr.Metric("name").isin([...])`), pull the IDs and load exactly those with
  `wandb.Api().run("<entity>/<project>/<id>")`. For broader filters, translate
  the filter fields and query with `wandb.Api().runs(...)`.
- **Panels are plot definitions, not the data.** To get the values a panel
  shows: `BarPlot`/`ScalarChart` read from `run.summary`; `LinePlot` needs
  bounded `run.history(keys=[...], samples=N)`; `ScatterPlot` joins
  summary/config/run-attribute values; media/table/custom panels reference
  table/media keys whose backing files/artifacts you fetch separately.
- **History-fetch gotchas.** Use bounded `run.history(keys=[...], samples=N)`;
  reach for `run.scan_history(keys=[...])` only on small runs and never without
  `keys`. For long runs (check `run.summary["_step"]`; 10K+ steps) prefer sampled
  `history`. System/GPU metrics are a separate stream:
  `run.history(stream="system", keys=[...])`, not the default stream.
- **Verify numeric claims against the runs.** Report text can assert "99 runs"
  or "val/bpb 0.273"; if the user wants those validated, query the backing runs
  rather than trusting the prose.

## Finding reports

Report list rows are GraphQL `View` objects (raw GraphQL, like personal
workspace views — see WORKSPACES.md). The report-list UI (team overview, team
reports tab, profile pages, and the project reports tab) drives these same
queries: a title search wired to `displayNameContains` plus server-side
pagination. Search surfaces by scope:

- **Project** — `project(entityName, name).allViews(viewType, displayNameContains, userName, first, after)`: reports/drafts in one project; supports author (`userName`) and published-vs-draft (`viewType`).
- **User (cross-project)** — `user(userName).views(viewType, displayNameContains, …)`.
- **Entity/team (cross-project)** — `entity(name).views(displayNameContains, offset, …)`: no `viewType` arg; cursors can come back empty, so page with `offset`.
- **Broad** — `reportSearch(query)`: across all accessible reports; discovery, not scoped listing.

Published reports are `type: "runs"`; drafts `"runs/draft"` (pass as `viewType`
where supported). Useful `View` row fields: `id`, `displayName`, `description`,
`type`, `createdAt`, `updatedAt`, `starCount`, `starred`, `showcasedAt`,
`user { username }`, `project { name entityName }`.

- **`displayNameContains` is a title-only substring filter.** It's a
  case-insensitive "contains" match on `displayName`, backed by SQL
  `LIKE '%…%'` — the same box the report-list UI drives. Consequences to plan
  for: it matches a *contiguous* substring, not tokenized words (`"loss curve"`
  won't match a report titled "loss over the curve"); `_` and `%` are matched
  literally, not as wildcards; and an empty/omitted value means "no filter"
  (all rows), not "match empty". It never looks at the description or body —
  for those, fetch a bounded candidate list and filter locally, and for full
  body text load candidates with `wr.Report.from_url(...)` and inspect blocks.
- **Page with `first` + `offset`.** These view connections are server-paginated
  the way the report list itself pages: request `first` (page size) and
  `offset` (`page × first`) rather than pulling every row. Combine with
  `displayNameContains` to narrow first, then page the matches. On
  `entity.views`, prefer `offset` — its `after` cursors can come back empty.
- **Don't request `spec` in list queries** — it's the full report body/layout.
  Fetch light rows first; load full reports only for selected candidates.
- Build a URL from a row as `{base}/{entityName}/{name}/reports/{displayName with
  spaces→dashes}--{id}` (strip `=` padding from the id), then
  `wr.Report.from_url(url)`.

## Runset — which runs appear

A report's plots come from `wr.Runset` objects inside `PanelGrid` blocks; the
Runset's `filters` decide which runs appear.

`wr.Runset(entity, project, name="", filters=[], groupby=[], order=[], <column params>)`

- `entity`, `project` — required.
- `name` — optional label for the runset.
- `filters` — list of `expr.FilterExpr`, ANDed together (see below). Prefer
  structured filters over raw strings — they validate, never crash the UI, and
  survive field renames.
- `groupby` — list of run attributes to split by (see Grouping).
- `order` — list of `wr.OrderBy(name=..., ascending=<bool>)`. `name` takes a
  backend run field string (`"CreatedTimestamp"`, `"State"`, ...) or, to order
  by a metric, a wrapped reference: `wr.OrderBy(name=SummaryMetric("val/loss"))`
  or `name=Config("lr")`. A bare string is always treated as a run field — see
  Gotchas.
- column params — see Runs table columns.

### Filters

Filter constructors:

- `expr.Config("<key>")` — a hyperparameter from `run.config`.
- `expr.Summary("<key>")` — a final metric from `run.summary`.
- `expr.Metric("<field>")` — backend run fields. The run ID field is `name`
  (lowercase); `Metric("ID")` is valid and maps to `name` too via
  `FE_METRIC_NAME_MAP`. `Metric("Name")` is the display name. Prefer the
  positive `name` form. Other useful fields: `State`, `CreatedTimestamp`,
  `Runtime`, `Tags`, `Sweep`, `Group` (full `FE_METRIC_NAME_MAP` table in
  WORKSPACES.md, "Filterable fields").
- `expr.Tags()` — takes no name argument; supports `.isin([...])` /
  `.notin([...])`.

Operators: `==` `!=` `<` `>` `<=` `>=` `.isin([...])` `.notin([...])`.

- `<` and `>` are silently rewritten to `<=` / `>=` (each emits a
  `UserWarning`) — strict inequality does not exist; write `<=`/`>=` to avoid
  the warning.
- The top-level `filters` list is AND-only. For OR, wrap alternatives in
  `expr.Or(a, b)` and put that in the list.
- Select runs by ID with `expr.Metric("name").isin([...])` — wrap even a single
  value in a list. For "top N by metric", query the IDs first (e.g.
  `wandb.Api().runs(...)` sorted by the summary metric) then pass them in.
- Tags — always a list, even for one tag (`Tags().isin([tag])`, not
  `Tags().isin(tag)`); a bare string can silently match nothing.
- Reproducing a workspace's run filters in a report: a workspace filter can be
  toggled off in the UI without being deleted — it stays in the runset's filter
  list marked `disabled: true` but is **not applied** (see WORKSPACES.md,
  "Filters"). So the runs a user sees in a workspace reflect its **enabled**
  filters only; a `disabled: true` entry is present in the spec yet filters
  nothing, so it isn't part of what the user is currently looking at. Worth
  noting because applying all of them, disabled ones included, can over-constrain
  the runset and filter out every run.

### Runs table columns

Every `PanelGrid` renders a runs table beneath its panels; by default it shows
all columns. Control it with these `Runset` params. Column names are
`"<section>:<name>"`:

- `run:<attr>` — built-in run properties (`run:name`, `run:state`)
- `config:<key>.value` — config keys (note the `.value` suffix)
- `summary:<key>` — summary metrics
- `tags:__ALL__` — run tags

Params:

- `visible_columns` — columns to show
- `pinned_columns` — columns pinned to the left
- `hidden_columns` — columns to hide (overrides visible/pinned)
- `column_order` — left-to-right column order
- `column_widths` — per-column pixel widths (e.g. `{"summary:acc": 150}`)
- `lock_columns` — when `True`, shows only `pinned_columns` and
  `visible_columns`; unlisted columns are hidden

## Panel params + width

- Set concrete keys when named: LinePlot `x` (defaults to `"Step"`) and `y` (a
  list, e.g. `["accuracy", "loss"]`); BarPlot uses `metrics` (a list).
- Metric families: set `metric_regex="val/.*"` on LinePlot instead of listing
  every key (auto-sets `use_metric_regex=True`).
- Panel size via `layout=wr.Layout(w=..., h=...)`. The grid is 24 units wide, so
  `w` sets panels-per-row: `w=24` is one per row, `w=12` two across, `w=8` (the
  default) three across; `h` is height in rows (default 6).
- Default to full-width analytical panels (`w=24, h=8`+, paired with
  `width="fluid"`, which widens the whole report body). The API default `w=8`
  (three-across) and dashboard instincts bias toward compact grids, and you can't
  see the rendered report to notice a panel came out too small — so bias toward
  full width unless a chart is intentionally compact. Reserve smaller widths for
  panels that stay legible small: scalar cards and single-value charts (`w=8`),
  short-label bar charts, and simple side-by-side comparisons.
- The reason: at `w=8`/`w=12` an analytical panel renders small — chart titles
  truncate, the legend eats into the plot, axis and bar labels overlap, and
  multi-run line/scatter/parallel-coordinates charts get hard to read;
  `RunComparer`, table, and media panels are cramped. Extra height (`h=10`–`14`)
  gives dense panels room.

## Grouping panels by run group / sweep

Grouping compares experiment *families* in aggregate instead of plotting every
run on its own. Group on a run attribute or a config/parameter value: all runs
sharing that value collapse into one aggregate series, so you can compare one
family against another — e.g. group by a learning-rate config to show the range
and central tendency of every run at each value, side by side. On a `LinePlot`
each group renders as an aggregate line plus a shaded band showing spread; on
`BarPlot` / `ScalarChart` each group collapses to a single aggregated value.

Split a panel by a run-level attribute (`Group`, `Sweep`, `State`, ...) or a
config value by setting `groupby=[<attr>]` on the **`Runset`**; the panel
inherits it.

- **Never set `groupby` on the panel** for a run attribute — it collapses to a
  single `group: -` bar. Group at the Runset only. (Applies to every panel type.)
- Aggregate line (LinePlot): `groupby_aggfunc` — `"mean"`, `"median"`, `"min"`,
  `"max"`, `"sum"`, or `"samples"`. Shaded band: `groupby_rangefunc` —
  `"minmax"`, `"stddev"`, `"stderr"`, `"none"`, or `"samples"`. Leave the
  panel's own `groupby` unset.
- **BarPlot metrics must be bare summary keys** (`metrics=["reward"]`), not
  `wr.SummaryMetric(...)` — the wrapped form renders an empty chart. (LinePlot
  uses `y=[...]`.)

## wandb.Table in reports

Display a table logged via `wandb.log({"my_table": wandb.Table(...)})` with
`wr.WeavePanelSummaryTable(table_name=<log key>)`. `table_name` must match the
key the runs logged the table under; the panel renders it from `run.summary`, so
only the last logged version appears. Tables benefit from extra height
(`layout=wr.Layout(w=24, h=12)`).

## Export is UI-only

There is **no SDK or API method to export a report**, and no server/GraphQL
export endpoint. PDF/LaTeX export is a browser operation (the report's "…" →
Download menu: PDF via browser print, LaTeX as a client-generated zip), so it
cannot be scripted. Create the report and point the user to the Download menu.
This is a reason to keep the analysis living in the report
itself rather than promising a downloadable artifact.

## Gotchas (SDK-verified, wandb_workspaces 0.4.x)

1. **Draft reports cannot be edited by users.** `report.save()` publishes by
   default (`draft=False`). Passing `draft=True` creates a report users cannot
   edit.
2. **No `/` in report titles** — the slug keeps the literal `/`, adding a 6th
   path segment that `from_url` cannot parse back. Use ` — `, `:`, or `_`.
3. **`wr.P()` renders text literally** — no markdown, no `**bold**`, no
   `| col |` tables. Use `wr.MarkdownBlock(text=...)` for any markdown,
   including tables. A markdown `| col |` table gives no control over column
   spacing — how packed it looks is up to the renderer's default table styling.
   A `MarkdownBlock` also renders HTML, so an HTML `<table>` with `border-spacing`
   / cell `padding` is what makes column separation adjustable.
4. **No dot-paths in filter strings.** `"config.lr"`, `"summary.loss"`,
   `"tags.foo"` parse to missing keys and can crash the report UI. Use
   `expr.Config("lr")`, `expr.Summary("loss")`, `expr.Tags()`.
5. **`Runset.query` is a run-name regex only** — it maps to
   `internal.RunsetSearch(query=...)`, the UI search box. No `tags:` / `config:`
   syntax; use structured `filters` for everything else. Reach for `query` only
   when you want a name regex.
6. **Report width default is `"readable"`** (narrow). Pass `width="fluid"` for
   research-style reports so panel grids are not tiny. Options: `"readable"`,
   `"fixed"`, `"fluid"`.
7. **A bad `OrderBy` string kills the whole report page.** `OrderBy(name=...)`
   sends a bare string to the backend as a *run field*; a string that isn't one
   — especially the column-spec form `"summary:<key>"` — is written into the
   spec verbatim, `save()` and the `from_url` re-read both pass, and then the
   report page fails to render with a raw SQL error (`Error 1054: Unknown
   column '...' in 'order clause'`). Order by metrics only via the wrappers:
   `SummaryMetric("<key>")` / `Config("<key>")`.
8. **`BarPlot(metrics=...)` takes bare metric-key strings — the opposite of
   `OrderBy`.** Wrapping them (`metrics=[SummaryMetric("val/bpb")]`) serializes
   to invalid UI keys like `summary_metrics.val/bpb`, and the panel renders
   `Select runs that logged summary_metrics...` even though the runs logged the
   real metric. Use `metrics=["val/bpb", "core_metric"]`. Verifying this needs
   care: a plain `from_url()` re-read rehydrates the saved strings back into
   `Metric(name=...)`, hiding the bug — re-read with
   `wr.Report.from_url(url, as_model=True)` and assert the raw panel config
   holds bare strings (`config.metrics == ["val/bpb", ...]`, no
   `summary_metrics.` prefix).
9. **`wandb.init()` creates a new run.** Report authoring needs only
   `wandb.Api()` / `wandb_workspaces`; a `wandb.init()` in a report script mints
   a surprise utility run in the user's project.
10. **`wr.MarkdownPanel` inside a `PanelGrid` can render as an empty or oversized
   panel.** It's for short annotations sitting alongside real panels in a grid.
   Narrative markdown — prose, tables, recommendations — belongs in a top-level
   `wr.MarkdownBlock` block, which renders reliably.
11. **`wr.MediaBrowser` shows history-logged media only.** Media that exists
    only in a run's final summary (e.g. image files logged once without a step)
    never appears: the panel saves fine, passes the re-read, and renders
    `No data available. Please select runs that logged the keys...` — even
    though the runs did log those keys, just to summary. Whether a key has
    history rows is checkable up front with the public API:
    `next(run.scan_history(keys=[key], page_size=1), None) is None` — if that
    is true while `run.summary` has the key, the media is summary-only.
    (`historyKeys` is a GraphQL-only field; the public `Run` object does not
    expose it.) Summary-only images render reliably as top-level
    `wr.Image(url=...)` blocks using the run file's URL.
12. **`BarPlot.line_colors` persists but does not color bars.** The field
    accepts any mapping and round-trips through `save()` + `from_url`, so a
    spec-level verify will "confirm" colors the user never sees — the frontend
    does not apply per-metric colors to bar charts and bars render in the
    default palette. Bars keyed by *runs* follow the runset's run colors;
    there is no native way to set per-metric bar colors. A pre-rendered image
    block is the reliable way to guarantee a specific color mapping.
13. **Panel classes have no `.to_json()`.** `wr.LinePlot`, `wr.BarPlot`, etc.
    are not JSON-serializable via `to_json()`. Build reports through
    `wr.PanelGrid` + `report.save()`, or serialize with
    `panel._to_model().model_dump(by_alias=True, exclude_none=True)` only when
    you must edit raw spec JSON. Never call name-mangled internals like
    `_Base__to_model`.

## Pre-rendered figures (matplotlib → `wr.Image`)

Some figures have no native panel — box/violin plots, slope charts, custom-colored
bars (gotcha 12) — so they get rendered with matplotlib and embedded as an
`wr.Image` block. The rendered PNG is never visible to you, and `from_url` confirms
only that the image block exists, not how it looks, so overlap and clipping
problems in the figure aren't observable after the fact — the user is the only one
who sees them. matplotlib's defaults don't avoid them. What governs each common
issue:

- **Clipping at the frame** (titles/labels/legends cut off at the edges) comes from
  figure layout: `constrained_layout=True` / `fig.tight_layout()` and
  `savefig(bbox_inches="tight")` fit the content to the frame; the defaults leave
  edge content clipped.
- **Values spilling past the axes** (box whiskers/outliers outside the plot box)
  happens when the axis limits sit exactly on the data extremes; padding the limits
  to the full extent including whiskers/outliers (`ax.margins(...)` / a padded
  `set_ylim`) keeps them inside.
- **Colliding tick labels** are a function of label length and count vs rotation:
  long or numerous horizontal categorical labels overlap; rotation
  (`tick_params(rotation=...)`) and fewer ticks separate them.
- **Title on the legend / legend on the data** is a placement matter: the default
  in-axes legend can land on the data or the title, which legend `loc` /
  `bbox_to_anchor` (e.g. outside the axes) and title `pad` control.
- **Cramped figures** trace to figure size: `figsize` sets how much room the
  content gets, so more categories/series need a larger frame.

## Verify every mutation landed

Saves can fail silently. **A clean `save()` return is not proof; only the
re-read is.** After `report.save()`, re-read with `wr.Report.from_url(url)` and
assert specific saved content (block count + the named section text) before
telling the user it landed.

- One save, one verify. Do not loop `save()` hoping it sticks; two failed
  verifies means stop and report the failure with the URL you tried.
- **The re-read proves the block/panel is in the spec — not that a
  `CustomChart`/Vega panel renders.** A table-backed Vega panel can pass the
  re-read yet show blank because its data source fails to resolve. Prefer native
  panels; if you keep a Vega panel, do not claim it renders on the strength of
  the spec alone. The same spec-vs-render gap applies to wrapped `BarPlot`
  metrics (gotcha 8), media panels without history rows (gotcha 11), and
  persisted-but-unrendered fields like `BarPlot.line_colors` (gotcha 12).

Workspace-view edits (save/verify, `save_as_new_view()`, delete-vs-hide, per-run
`RunSettings(disabled=True)`) follow the same verify-after-save rule — see
WORKSPACES.md.
