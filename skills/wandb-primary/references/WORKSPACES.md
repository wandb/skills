<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Workspace Views & Panels (wandb_workspaces)

Mutation reference for programmatically creating and editing W&B workspace views
and panels. Install with `pip install "wandb[workspaces]"`. The workspace
tab includes a runs sidebar (run selector) that shares state with the runs
table — see RUNS_TABLE.md for runs-table-specific details (delta columns,
column management, filtering shortcuts, CSV export, etc.).

```python
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
```

## Panel types

Panel classes live in `wandb_workspaces.reports.v2`. Only these are supported in
workspace sections:

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
| `wr.CustomChart` | **Last resort only** — a chart no native type can express (heatmap, violin, confusion matrix, etc.). Renders arbitrary Vega. Table/artifact-backed specs (`from_table`) frequently fail data-source resolution in a workspace and are silently dropped at render — see caveat below. | `query`, `chart_name`, `chart_fields`; also `CustomChart.from_table(table_name=..., chart_fields={...})` |
| `wr.WeavePanelSummaryTable` | W&B Table from run summary | `table_name` |

Every panel accepts `layout=wr.Layout(x, y, w, h)` — width `w` in grid units
(max 24, default 8), height `h` (default 6).

**Prefer native panels over `CustomChart`.** If a visualization can be expressed
with a native panel type (a bar chart of final metrics → `BarPlot`, a metric over
time → `LinePlot`, a single value → `ScalarChart`, a correlation → `ScatterPlot`),
use that native type. Native panels resolve their data reliably at render time;
`workspace_ops.py add-panel` covers bar/line/scalar directly. Reach for
`CustomChart` only when no native panel fits, and read the render caveat under
"wandb.Tables in workspaces" first.

## Default for plot requests

When asked to plot or compare runs, prefer native workspace/report panels over
logged static artifacts whenever the chart can be built from run history,
summary, or config. Use direct native panels first (`wr.LinePlot`,
`wr.ScatterPlot`, `wr.BarPlot`); if no saved view exists, create one with
`ws.Workspace(...).save()` and share its URL.

For grouped or multi-dimensional charts, choose a supported primary axis and
encode the rest with grouping, legend fields, workspace coloring
(`semanticLegendSettings.key`), or titled panels/sections. Mention unsupported
exact layouts in the response. Do not create helper runs, tables/artifacts,
`wr.MediaBrowser`, Plotly/PNG files, or `wandb.Image` just to approximate
run-metric charts; reserve static artifacts for genuinely derived figures/media.

## Workspace modes: automated vs manual, reset & restoring panels

If a user **reset** their workspace (personal-workspace `•••` menu → **Reset
workspace**) and chose **Manual workspace**, they land on a blank slate ("No
visualizations yet") — every panel is gone and run grouping is cleared.

**One option worth mentioning: Undo.** If the reset was accidental and just
happened, the user can press the workspace **Undo** button (Cmd+Z, or the ↶
button top-right) **twice** in their open tab to revert the reset-to-manual and
restore the previous workspace *exactly* — all panels **and** run grouping — with
no regeneration. That is the only recovery that also brings grouping back, and
the same Undo reverts other just-made changes too (a hidden run, a deleted
panel). Undo has no server API, so it is something the user does in their own
tab. If the user would rather have the panels restored for
them — or Undo is no longer available (tab reloaded/closed, reset not recent) —
use the paths below; doing the restore for them is a perfectly good option, not
just a fallback.

**First: is manual mode intentional?** A manual workspace is often a deliberate
**performance** choice, not an accident. Projects logging many metrics (roughly
5k–100k keys) render slowly, and the documented fix is exactly this: reset to
manual mode, then add back only the panels you need. W&B still collects and
stores the unplotted metrics, so no data is lost. So before regenerating
*everything*, check whether the user wants **all** panels back or just a
**subset** — blindly turning auto-generate back on can reintroduce the slow load
the manual workspace was meant to avoid.

### Restore all panels
Regenerate every auto-panel at once (do **not** hand-rebuild a subset from logged
metrics as a "restore" — it is slow and misses keys):

- **UI:** **+ Add panels** → **Quick panel builder** → **"Create N panels"**
  (N = every logged key). Or re-run **Reset workspace** and pick **Automated
  workspace** instead of Manual.
- **Programmatic (personal `nwuser` view):** set
  `section.workspaceSettings.shouldAutoGeneratePanels = true` (the "Automated
  workspace" equivalent). The SDK's `from_url` rejects personal views, so mutate
  the raw spec via `mutate_view_with_retry` (see "Edit the raw spec"):

```python
def mutate(spec):
    spec.setdefault("section", {}).setdefault("workspaceSettings", {})[
        "shouldAutoGeneratePanels"] = True

mutate_view_with_retry(api, entity=ENTITY, project=PROJECT,
                       view_url="https://wandb.ai/ENTITY/PROJECT?nw=nwuserNAME",
                       view_token=None, display_contains=None, mutate=mutate)
```

### Restore a subset (keep it lean / fast)
Usually the right move when the workspace was manual for performance — add back
only the panels the user cares about, leaving auto-generate **off**:

- **UI:** **+ Add panels** → **Quick panel builder** → filter with **Search keys
  with regex** (e.g. `^val/`), then **"Create N panels"** — N updates to the
  filtered subset. The **Single-key** vs **Multi-metric** tabs choose
  one-key-per-panel vs combined panels.
- **Programmatic:** leave `shouldAutoGeneratePanels` off and add explicit panels
  only for the chosen keys (e.g. a `LinePlot` per metric) via the add-panel path
  (see "Adding a panel"). This keeps the workspace manual and fast.

> **Use the full namespaced key, not the display label.** The Quick panel
> builder shows short labels (`bpb_train`, `bpb`), but the logged keys are often
> namespaced (`eval/bpb_train`, `val/bpb`). A panel whose `metrics`/`y` uses the
> bare label **does not resolve** — the UI silently drops it (you get fewer
> panels than expected, with no error). Look up the exact key first (`historyKeys`
> / a run's summary keys — see "Discovering available keys") and build panels
> against that, e.g. `LinePlot(y=["eval/bpb_train"])`, not `["bpb_train"]`.

Auto-generated panels render **client-side and are not stored in the spec** (see
`shouldAutoGeneratePanels` under "Key spec paths"): a spec read-back confirms the
*setting* flipped, not that the panels rendered — flip it, then ask the user to
refresh to confirm. Run **grouping is not restored** by either path; re-apply it
separately (see "Run grouping").

Sections with `isPanelsAuto: true` are generated by the frontend. You cannot
see or count their visible panels from the saved spec, and changes to these
sections can disappear on the next render. Never mutate an auto-panel section;
add panels to a non-auto section, creating one if necessary. Likewise, do not
edit a materialized panel with `isAuto: true`; clone it with a new `__id__` and
`isAuto: false`, then edit the clone.

## Reading a workspace

Load a workspace to inspect its current state: visible runs, active filters
and grouping, sections and panels, coloring, pinned/baseline runs, and search
bars. For a quick summary (views, sections, visible panel counts, open/pinned),
`workspace_ops.py find-view` and `wandb_run_ops.py workspace-views` read this
without hand-querying; the paths below are for deeper spec inspection.

### Saved views (SDK)

For saved views (`?nw=<id>`, not `nwuser`), the SDK can read and write:

```python
workspace = ws.Workspace.from_url("https://wandb.ai/ENTITY/PROJECT?nw=abc123def45")

# Sections and panels
for s in workspace.sections:
    print(f"Section '{s.name}': is_open={s.is_open}, {len(s.panels)} panels")
    # Section-level settings (override workspace defaults)
    ps = s.panel_settings  # SectionPanelSettings: x_axis, x_min, x_max, smoothing_weight, smoothing_type
    ls = s.layout_settings  # SectionLayoutSettings: columns, rows

# Filters, grouping, pinned/baseline
rs = workspace.runset_settings
print(f"filters: {rs.filters}")
print(f"groupby: {rs.groupby}")
print(f"pinned_runs: {rs.pinned_runs}")
print(f"baseline_run: {rs.baseline_run}")
print(f"run_settings: {rs.run_settings}")  # per-run color/disabled

# Run search bar
print(f"query: {rs.query}")
print(f"regex_query: {rs.regex_query}")

# Panel search bar
print(f"panel_search_query: {workspace.settings.panel_search_query}")

# Workspace-level settings
ws_settings = workspace.settings  # WorkspaceSettings
# x_axis, x_min, x_max, max_runs, smoothing_weight, smoothing_type,
# ignore_outliers, sort_panels_alphabetically, remove_legends_from_panels,
# tooltip_color_run_names, tooltip_number_of_runs, point_visualization_method,
# group_by_prefix, panel_search_query, auto_expand_panel_search_results
```

Some features (key-based coloring, panel-level config) are not exposed via the
SDK. Read them straight from the raw spec — load it with `fetch_view` from
`workspace_write`, or your own GraphQL query (reads are unrestricted). Only *writing*
those fields is guarded: persist changes through `mutate_view_with_retry` (see "Edit
the raw spec" below). The same read/write split applies to saved views when the SDK
doesn't cover a field.

### Personal views (raw spec)

The SDK rejects personal views (`?nw=nwuser...`), so you work with their raw spec
directly. Reading is unrestricted — load the spec however is convenient (`fetch_view`
from `workspace_write`, `workspace_ops.py find-view`, or a plain GraphQL query). Only
the *write* must go through the guarded `mutate_view_with_retry` (see "Edit the raw
spec"). The spec contains:

```python
spec = json.loads(node["spec"])
section = spec["section"]
rs = section["runSets"][0]

# Active filters (filterV2 format)
# rs["filters"]["filters"] is a flat list. An item is a filter
# {key: {section, name}, op, value, disabled} or a group {filters: [...],
# disabled}. Every item after the first carries connector: "AND" | "OR",
# its relation to the item before it; a missing connector is AND. AND binds
# tighter than OR, so [A, B(OR), C(AND)] is A OR (B AND C), and a filter
# appended to the end with no connector is ANDed onto the last OR branch
# only. A group counts as one item to its parent and follows the same rules
# inside. The filter editor shows one level of groups: a group inside a
# group renders as a keyless row that cannot be edited.
# key.section is run, config, summary, tags or keys_info. Run attributes are
# {section: "run", name: "displayName" | "state" | "createdAt" | "username" |
# "group" | ...}. Tag filters live under their own section: {section: "tags",
# name: "*"} with IN or NIN over tag names, or {section: "tags", name: "<tag>"}
# with = true / != false for one tag. A key the filter editor cannot map to a
# property (for example {section: "run", name: "tags"}) still filters on the
# backend but renders as a red, uneditable row.
# op is one of =, !=, <, >, <=, >=, REGEX, EXISTS, CONTAINS, CONTAINS_WILDCARD,
# WITHINSECONDS (single value) or IN, NIN (list value).
# There is no negated-substring operator: NIN excludes exact names, and an
# unknown op falls back to equality in the UI. REGEX is RE2 (no lookahead,
# lookbehind, or backreferences), so "name does not contain X" is either NIN
# over the names that match X, or a regex written without negation.
# disabled=true means the filter is toggled off in the UI
filters = rs["filters"]["filters"]
active_filters = [f for f in filters if not f.get("disabled")]

# Grouping
groupby = rs.get("grouping", [])  # e.g. [{"section": "config", "name": "lr"}]

# Run visibility — selections.root controls the default
# root=1: all runs visible (tree entries override to hide)
# root=0: no runs visible (tree entries override to show)
selections = rs.get("selections", {})

# Pinned runs and baseline (base64-encoded run references)
import base64
pinned = [base64.b64decode(p).decode() for p in rs.get("pinnedRunIds", [])]
baseline = rs.get("baselineRunId")
if baseline:
    baseline = base64.b64decode(baseline).decode()

# Per-run colors
custom_colors = section.get("customRunColors", {})  # {run_id: hex_color}

# Key-based coloring (semantic legend)
legend = section.get("semanticLegendSettings", {})
is_key_colored = legend.get("runColorOptions") == "semantic-legend"

# Sections and panels
for s in section["panelBankConfig"]["sections"]:
    print(f"Section '{s['name']}': isOpen={s['isOpen']}, "
          f"{len(s.get('panels', []))} panels")

# Run search bar
run_search = rs.get("search", {})
if run_search.get("query"):
    print(f"Run search: '{run_search['query']}' (regex={run_search.get('isRegex', False)})")

# Panel search bar
panel_search = section.get("panelBankConfig", {}).get("settings", {}).get("searchQuery", "")
if panel_search:
    print(f"Panel search: '{panel_search}'")

# Workspace-level settings
settings = section.get("settings", {})
# xAxis, xAxisMin, xAxisMax, maxRuns, smoothingWeight, smoothingType,
# ignoreOutliers, sortPanelsAlphabetically, removeLegends,
# tooltipColorRunNames, tooltipNumberOfRuns, pointVisualizationMethod,
# groupByPrefix, shouldAutoGeneratePanels

# Section-level settings (override workspace-level per section)
for s in section["panelBankConfig"]["sections"]:
    sec_settings = s.get("localPanelSettings", {})
    # Same keys as workspace settings but scoped to this section:
    # xAxis, xAxisMin, xAxisMax, smoothingWeight, smoothingType

# Panel-level settings (override section-level per panel)
# Individual panels have their own config that can override axes,
# smoothing, grouping, etc. For example, a LinePlot's config:
# {"xAxis": "_step", "smoothingWeight": 0.8, "groupBy": "config:lr", ...}
```

### Line plot x expressions and x ranges

A line plot's `xExpression` (`wr.LinePlot(xaxis_expression=...)`) rewrites the
plotted x of every point. Identifiers are `${metric}` for a history key,
`${_step}` included, plus `${config:key}` and `${summary:key}`; a bare name such
as `x` is read as a metric called `x`, and a panel whose expression names a
metric the runs never logged renders "There's no data for the selected runs".
`xAxisMin`/`xAxisMax` (panel, section or workspace; `range_x` in `wr.LinePlot`)
are in the x-axis key's own units and set both the history query window and the
drawn x domain. On a panel with an `xExpression` the points sit at the expressed
values while the range still counts in raw units, so a range in either unit
leaves that panel empty.

### Key spec paths

| What | Path in spec |
|---|---|
| Filters | `section.runSets[0].filters.filters` (list, check `disabled` flag) |
| Grouping | `section.runSets[0].grouping` |
| Sort order | `section.runSets[0].sort.keys` |
| Run visibility | `section.runSets[0].selections` (`root` + `tree` overrides) |
| Pinned runs | `section.runSets[0].pinnedRunIds` (base64-encoded) |
| Baseline run | `section.runSets[0].baselineRunId` (base64-encoded) |
| Per-run colors | `section.customRunColors` (`{runId: color}`) |
| Key-based coloring | `section.semanticLegendSettings` |
| Sections & panels | `section.panelBankConfig.sections[].panels` |
| Section open/closed | `section.panelBankConfig.sections[].isOpen` |
| Section settings | `section.panelBankConfig.sections[].localPanelSettings` (overrides workspace) |
| Panel-level config | `section.panelBankConfig.sections[].panels[].config` (overrides section) |
| Run search (regex) | `section.runSets[0].search` (`query` + `isRegex`) |
| Panel search (regex) | `section.panelBankConfig.settings.searchQuery` |
| Workspace settings | `section.settings` (xAxis, maxRuns, smoothing, etc.) |
| Column visibility | `section.runSets[0].runFeed.columnVisible` |

## Point aggregation on line plots (the "sawtooth")

Dense line-plot metrics (e.g. per-step loss) are downsampled for display. The
panel's **Point aggregation** setting — `pointVisualizationMethod` (raw spec) /
`point_visualization_method` (SDK), cascading workspace → section → panel — sets
how:

- `bucketing-gorilla` (**Full fidelity**, default): buckets the x-axis; sub-mode
  `fullFidelityMode` is `standard` (shaded min/max **Band**) or `min-max-last`
  (draws each bucket's **min, max, and last** point).
- `sampling` (**Random sampling**): a sampled subset of real points (~1,500), no
  bucketing. `sampling-by-default` is the same, applied as the default.

`min-max-last` connecting each bucket's min↔max is the regular **zigzag /
"sawtooth"** on noisy metrics — faithful (real per-bucket extrema), not a bug.
It's independent of smoothing/range shading; `sampling` avoids it but can miss
true spikes, and zooming into a step range shows the raw values.

## wandb.Tables in workspaces

Tables logged via `wandb.log({"my_table": wandb.Table(...)})` appear in
`run.summary` and can be displayed as workspace panels:

```python
wr.WeavePanelSummaryTable(
    table_name="predictions",       # key used in wandb.log()
    layout=wr.Layout(w=24, h=12),   # tables benefit from extra height
)
```

Also available: `wr.WeavePanelArtifactVersionedFile(artifact="name", version="v0",
file="path")` for files stored in artifacts, and `wr.WeavePanelArtifact(artifact="name",
tab="files")` for browsing an artifact.

`wr.CustomChart.from_table(table_name="my_table", chart_fields={...})` renders a
custom Vega chart backed by a logged table.

**Caveat — table-backed CustomCharts often do not render in a workspace.** The
generated Vega spec references the backing table by key (e.g.
`summaryTable.tableKey`), and the workspace UI frequently cannot resolve that
reference at render time, so it silently drops the panel — the `UpsertView`
mutation succeeds and the panel is present in the spec, but the user sees
nothing. Spec-level verification cannot catch this (see Gotcha #4). Prefer a
native panel (`BarPlot`/`LinePlot`/`ScalarChart`) whenever the chart can be
expressed natively. To display a logged table as-is, use
`wr.WeavePanelSummaryTable` instead of a table-backed Vega chart.

**Scrubbing a table across steps/versions (the "version slider").** How this data
is shaped: a `wandb.Table` logged under one key at each step
(`wandb.log({"val/foo": wandb.Table(...)})`) is written to run **history** as a
table-typed cell at each of those `_step`s — present only at the steps that logged
one, `none` elsewhere — while `run.summary` keeps just the latest. So
`WeavePanelSummaryTable` shows only the latest version and
`WeavePanelArtifactVersionedFile` pins one fixed version; neither has a slider.

The panel built for this shape is the Weave **`run-history-tables-stepper`** ("Run
History Tables Stepper"). Its input is the **runs**, and it does the work this data
requires: pull run history (`opRunHistory`), keep the rows where the table key is
non-null (the steps that actually logged a table), step the slider over those rows'
`_step`, and render the table at the selected step. Its only config is the table
key (`tableHistoryKey`). It has no `wr.WeavePanel*` wrapper.

Why the generic Query-panel **Stepper** (a panel's *Render As → Stepper*) is not a
substitute for a table of this shape: that Stepper builds its slider from the
*numeric properties of the input node's row type* and needs a list of rows each
carrying an integral `_step`. Pointed at `runs.summary["val/foo"]` it sees a single
latest table with no `_step` property and reports `"_step is not a valid step
metric."` A per-step-logged table is media in history, not a per-row numeric column,
so it never presents the row-list-with-`_step` shape the generic Stepper expects —
that is precisely the gap `run-history-tables-stepper` fills (the history fetch +
non-null filter + `_step` stepping). Hand-assembling a generic `PanelStepper` over a
table expression saves to the spec but does not render (per gotcha 4, a spec
read-back cannot observe that), which is the failure mode to avoid.

## Run grouping

Grouping aggregates runs by a shared value — typically a config parameter — so
you can visually compare cohorts. Use grouping when you have multiple runs per
hyperparameter value and want to see aggregated trends (e.g., mean loss curve
per learning rate with variance bands).

### Workspace-level grouping

Groups runs in the sidebar and aggregates them in panels:

```python
ws.RunsetSettings(
    groupby=[ws.Config("learning_rate")],
)
```

Accepts: `ws.Config("key")`, `ws.Summary("key")`, `ws.Metric("key")`, `ws.Tags()`.

### Group by Group

Distinct from config-based grouping. This groups runs by the `group` attribute
set via `wandb.init(group="experiment-1")`. In the SDK, the `group` run attribute
is accessed via `ws.Metric("Group")` (capitalized — internally mapped to the
`group` field via `FE_METRIC_NAME_MAP`).

```python
ws.RunsetSettings(
    groupby=[ws.Metric("Group")],
)
```

This aggregates runs sharing the same `wandb.init(group=...)` value. Useful for
distributed training (all workers in one group) or experiment batches.

### Grouping in the raw spec

A personal view (`?nw=nwuser…`) is only reachable through its raw spec, where
`grouping` is a list of **bare key objects**:

```python
rs["grouping"] = [{"section": "run", "name": "group"}]      # by Group
rs["grouping"] = [{"section": "config", "name": "lr"}]      # by a config key
```

`section` is `run` for run attributes, `config`, `summary`, `tags` or
`keys_info`; the `Group` attribute is `{"section": "run", "name": "group"}`,
which is what `ws.Metric("Group")` serializes to.

Reading a spec back confirms the write landed. It does not confirm the
workspace still renders — a spec the frontend cannot render reads back exactly
as written.

### Panel-level grouping

Override grouping and set aggregation per panel:

```python
wr.LinePlot(
    x="Step", y=["loss"],
    groupby=wr.Config("learning_rate"),
    groupby_aggfunc="mean",       # "mean", "min", "max", "median", "sum", "samples"
    groupby_rangefunc="stddev",   # "minmax", "stddev", "stderr", "none", "samples"
)
```

`groupby_aggfunc` controls the aggregate line. `groupby_rangefunc` controls the
shaded band. `"stddev"` shows ±1σ, `"minmax"` shows the full range, `"samples"`
shows individual runs as faded lines.

BarPlot and ScalarChart also support `groupby_aggfunc` and `groupby_rangefunc`.

## Run filtering

Filters control which runs appear in the workspace. Three syntax forms:

### String syntax

```python
ws.RunsetSettings(
    filters="State = 'finished'",
)
ws.RunsetSettings(
    filters="Config('learning_rate') == 0.001 and State == 'finished'",
)
```

### Structured And/Or combinators

```python
from wandb_workspaces.workspaces import And, Or, Config, Metric, Summary, Tags

# Filter out crashed runs
ws.RunsetSettings(filters=Metric("State") != "crashed")

# AND — all conditions must match
ws.RunsetSettings(
    filters=And(
        Config("model") == "resnet",
        Metric("State") == "finished",
    ),
)

# OR — any group can match
ws.RunsetSettings(
    filters=Or(
        And(Config("model") == "resnet", Summary("accuracy") >= 0.9),
        And(Config("model") == "densenet", Summary("accuracy") >= 0.85),
    ),
)
```

### Filterable fields

Three constructors route to different namespaces:

- `Config("key")` — any hyperparameter key from `wandb.config`
- `Summary("key")` — any metric key in `run.summary`
- `Metric("name")` — built-in run attributes (see table below)

`Metric(...)` maps frontend names to backend fields via `FE_METRIC_NAME_MAP`:

| `Metric(...)` | Backend field | Example |
|---|---|---|
| `"State"` | `state` | `== "finished"`, `!= "crashed"` |
| `"CreatedTimestamp"` | `createdAt` | `.within_last(7, "days")` |
| `"ID"` | `name` (run slug) | `.isin(["abc123", "def456"])` |
| `"Name"` | `displayName` | `== "my-run"` |
| `"User"` | `username` | `== "juliarose"` |
| `"Runtime"` | `duration` | `>= 3600` |
| `"Group"` | `group` | `== "experiment-1"` |
| `"JobType"` | `jobType` | `== "train"` |
| `"Hostname"` | `host` | `== "gpu-node-1"` |
| `"Sweep"` | `sweep` | `== "sweep-id"` |
| `"UsingArtifact"` | `inputArtifacts` | artifact filtering |
| `"OutputtingArtifact"` | `outputArtifacts` | artifact filtering |

`Tags()` filters on run tags: `Tags().isin(["best", "production"])`.

Operators: `==`, `!=`, `<=`, `>=`, `.isin([...])`, `.notin([...])`,
`.within_last(n, "days")` (CreatedTimestamp only). Note: `<` maps to `<=` and
`>` maps to `>=`.

### Discovering available keys

Config and summary keys are project-specific. To discover what's available,
query `historyKeys` on a `RunConnection` (returns all logged metric keys) or
use the `wandb_run_ops.py inspect-schema` helper:

```bash
python wandb_run_ops.py inspect-schema --entity ENTITY --project PROJECT --schema both --sample 50
```

Or via GraphQL on a runs connection:

```python
# historyKeys returns all metric keys across matching runs
# configValues/configValuesCategories enumerate config key values
```

## Run visibility, pinning, and baselines

### Hide/show individual runs (the eye toggle)

Every run in the run selector has an eye icon that toggles its visibility.
Turning the eye on makes a run *eligible* to be drawn in panels; turning it
off excludes it from every chart. Visibility does not guarantee a run is
plotted: each panel draws only up to its own limit (a heuristic such as the
most recent N visible runs), and that number varies by panel type and
settings, mostly to bound rendering cost. So visible runs are the pool that
panels draw from, not necessarily the exact set rendered in any one panel.

Visibility lets the user narrow a workspace down to the runs they care about;
when someone says "my runs" or "these runs" on a workspace they usually mean
the *visible* ones.

Visibility is distinct from filtering: a filter changes which runs are in the
view at all, while the eye toggle flips a run's visibility within the filtered
set.

In the SDK / view spec this same state is `RunSettings(disabled=True)`
(hidden).

```python
ws.RunsetSettings(
    run_settings={
        "abc123": ws.RunSettings(disabled=True),    # hidden
        "def456": ws.RunSettings(color="red"),       # visible, red
        "ghi789": ws.RunSettings(color="#00ff00"),   # visible, green
    },
)
```

Run visibility is controlled at the runset level and applies to all panels
sharing that runset. There is no per-panel run visibility.

`RunSettings(disabled=…)` is the high-level SDK form. When you mutate the **raw
view spec directly** (e.g. via `workspace_write` on a saved or personal `nwuser`
view), visibility lives in `section.runSets[0].selections`, and personal `nwuser`
view specs typically carry **no** `runSettings` map at all, so `selections` is the
*only* place visibility is encoded there.

**Write the complete `selections` object — all three fields — or the frontend
silently ignores it and the runs stay hidden.** The canonical shape (see
`frontends/app/src/util/selectionmanager.ts`, `SelectionState`) is:

```python
rs["selections"] = {"root": 0, "bounds": [], "tree": [run_id, ...]}
```

A partial `{"root": ..., "tree": ...}` (omitting `bounds`) round-trips into
storage but is **not** honored on load — the runs render eyeball-hidden. This is a
real, observed failure. `bounds` is `[]` for a plain run-id set; `tree` holds the
short run ids (the same ids `api.run().id` returns).

`root` is the `Root` enum: `0` = None (nothing shown by default), `1` = All. So
`tree`'s meaning **flips with `root`**: `root=1` → `tree` *hides*; `root=0` →
`tree` *shows*. To make exactly a set of runs visible, write
`{"root": 0, "bounds": [], "tree": [<ids to show>]}` (shows those, hides the
rest). **Never blindly append run ids to `tree`** without setting `root` — appended
to a `root=1` tree it *hides* them. And a run id sitting in `tree` is **not** proof
it's visible — verify against actual state (eyeball-on runs, or by
re-checking what the user sees), never tree membership or the stored spec alone.

### Pinned runs

Pinned runs stay visible regardless of filters and are always fetched for plots.
Max 20 pinned runs.

```python
ws.RunsetSettings(
    pinned_runs=[
        "abc123",                                             # same project
        "other-entity/other-project/def456",                  # cross-project
        ws.RunRef("ghi789", entity="team", project="proj"),   # typed ref
    ],
)
```

### Cross-project pinned runs

Cross-project pins let you compare runs from different projects within your
current workspace. Pinned cross-project runs appear with an open circle icon in
the run selector; hovering shows which project they come from. Artifacts, logs,
and details link back to the original run — no data is imported.

Via SDK:

```python
ws.RunsetSettings(
    pinned_runs=[
        "other-entity/other-project/run_id",
        ws.RunRef("run_id", entity="other-entity", project="other-project"),
    ],
)
```

In the raw spec, cross-project pinned run IDs are base64-encoded references that
include the source project: `Run:v1:<run_id>:<project>:<entity>`.

### Baseline run

A baseline run serves as a reference point for evaluating other runs:

- Appears at the top of the runs table (with a bookmark icon) alongside pinned runs
- In line plots, the baseline run's line is **bolder** and **dashed** on hover — visually distinct for comparison
- Enables **delta columns** in the runs table, showing other runs' metrics relative to the baseline

Setting a baseline automatically pins it.

```python
ws.RunsetSettings(
    baseline_run="abc123",
    # or cross-project:
    baseline_run=ws.RunRef("abc123", entity="team", project="proj"),
)
```

## Run coloring

### Per-run colors

Assign a fixed color to specific runs:

```python
ws.RunsetSettings(
    run_settings={
        "run_id_1": ws.RunSettings(color="red"),
        "run_id_2": ws.RunSettings(color="#00ff00"),
        "run_id_3": ws.RunSettings(color="blue"),
    },
)
```

Colors can be CSS color names or hex values.

### Key-based coloring (semantic legend)

In the UI, key-based coloring assigns run colors by a metric or config value
rather than by run identity. Runs are bucketed by the chosen key's value and
each bucket gets a distinct color from a palette. This is called "semantic
legend" internally.

Key-based coloring is stored in the view spec as `semanticLegendSettings` with
`runColorOptions: "semantic-legend"` (vs `"default"` for per-run colors). It
is a UI feature — not directly exposed via the `wandb_workspaces` Python SDK.
To set it programmatically, edit the view spec via raw GraphQL (see "Edit a
personal workspace view") and set:

```python
spec["section"]["settings"] = {
    **spec["section"].get("settings", {}),
    "semanticLegendSettings": {
        "runColorOptions": "semantic-legend",
        "key": "config:model_type",       # or "summary:accuracy", etc.
        "numberOfBuckets": 5,
    },
}
```

## Create a new saved view

```python
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(
    entity="ENTITY",
    project="PROJECT",
    name="My Analysis View",
    sections=[
        ws.Section(
            name="Training",
            panels=[
                wr.LinePlot(x="Step", y=["train_loss", "val_loss"],
                            title="Loss curves",
                            layout=wr.Layout(w=12, h=8)),
                wr.BarPlot(metrics=["accuracy"], title="Final accuracy",
                           layout=wr.Layout(w=12, h=8)),
            ],
            is_open=True,
        ),
    ],
    settings=ws.WorkspaceSettings(
        x_axis="Step",
        max_runs=20,
    ),
    runset_settings=ws.RunsetSettings(
        filters="State = 'finished'",
        order=[ws.Ordering(ws.Metric("CreatedTimestamp"), ascending=False)],
    ),
)
workspace.save()
print(workspace.url)
```

`save()` creates a new saved view (generates a random `nw` token). The URL is
shareable with teammates who have project access.

- **`nw` tokens are system-generated — never mint your own.** The token is
  lowercase alphanumeric; a hand-chosen slug (e.g. `?nw=my-analysis-view`)
  produces a URL the workspace UI cannot resolve, even though the view exists
  in the spec. Create the view with `save()` and share `workspace.url`; if you
  then need spec-level edits the SDK can't express, load that view by its
  saved token and edit via raw GraphQL.
- **A workspace view holds a single runset.** The SDK (and the workspace UI)
  don't support multiple runsets per view — that's a Report feature
  (`PanelGrid` takes a list of `wr.Runset`). If the comparison needs several
  independently-filtered run groups, build a report; don't hand-write a
  multi-runset workspace spec via `upsertView`.

## Writing safely: every edit is guarded

Every workspace-view **edit** must go through a `workspace_write` helper
(`skills/wandb-primary/scripts/workspace_write.py`) — never a bare `workspace.save()` or
hand-written `upsertView`. The helpers compare-and-swap on `View.updatedAt` and, on
conflict, silently re-read the latest server spec, re-apply your change on top, and
retry (bounded), so an agent edit layers onto a concurrent user/tab change instead
of clobbering it.

Avoid a bare `save()` for two independent reasons:

- **It has no concurrency guard.** `save()` sends no `updatedAt`, so it
  last-write-wins — silently overwriting whatever a user or another tab saved after
  you loaded. The helpers detect that and retry instead.
- **It wipes the description.** wandb-workspaces' `save()` always writes
  `description=""`, blanking any existing view description. The guarded helpers send
  `description` only when you explicitly pass one.

| To edit… | Use |
|---|---|
| a **saved view** with SDK objects (panels, `runset_settings`, …) | `save_workspace_edit_with_retry(api, url=…, edit=…)` |
| **raw spec** fields the SDK can't express, or a **personal `nwuser`** view | `mutate_view_with_retry(api, …, mutate=…)` |
| add a line/bar/scalar panel or a section (saved *or* personal) | `workspace_ops.py add-panel` / `add-section` (wraps the above) |

Creating a brand-new view is the one unguarded case: `ws.Workspace(...).save()` is
an insert with nothing to clobber (see "Create a new saved view").

## Edit an existing saved view

Edit with the SDK object model, persist through the guard. `edit(workspace)` receives
the loaded `ws.Workspace` and mutates it in place with SDK objects; the helper
serializes the result and guarded-upserts it, updating the SAME view:

```python
import sys; sys.path.insert(0, "skills/wandb-primary/scripts")
import wandb
import wandb_workspaces.reports.v2 as wr
from workspace_write import save_workspace_edit_with_retry, read_back_spec

api = wandb.Api(timeout=60)
result = save_workspace_edit_with_retry(
    api,
    url="https://wandb.ai/ENTITY/PROJECT?nw=abc123def45",
    edit=lambda w: w.sections[0].panels.append(
        wr.LinePlot(x="Step", y=["new_metric"], title="New panel")
    ),
    # display_name="Renamed view",  # optional: sent only when you mean to rename
)

# Verify against fresh server state — the helper's return is not proof the spec landed.
verify = read_back_spec(api, "ENTITY", "PROJECT", result["view"]["name"])
titles = [(p.get("config") or {}).get("chartTitle") for s in
          verify["section"]["panelBankConfig"]["sections"] for p in s.get("panels", [])]
assert "New panel" in titles, f"panel missing on read-back: {titles}"
```

- **Saved views only.** Personal `nwuser` views raise here (the SDK's `from_url`
  rejects them too) — edit their raw spec via `mutate_view_with_retry` (below).
- **Never `ws.Workspace(...).save()` to edit** — that mints a new view and orphans
  the original. Use it only when you actually mean to create one.
- **SDK round-trip is lossy.** Like `save()`, editing via the SDK object model drops
  any spec field the SDK doesn't model. When you must preserve such fields, edit the
  raw spec via `mutate_view_with_retry` instead.

## Edit the raw spec (filters, coloring, settings, personal views)

For spec fields the SDK can't express — filters, grouping, run visibility, coloring,
key-based (semantic) legends, workspace/section settings, search bars — and for
**personal `nwuser` views** (which the SDK's `from_url` rejects), edit the raw spec
dict through `mutate_view_with_retry`. It handles the load → mutate → guarded
`UpsertView` → retry plumbing (compare-and-swap on `updatedAt`, re-reading and
re-applying your change on conflict) for both saved and personal views. Personal
views use the same mutation as saved views; they differ only in `locked=true`
(owner-only writes) and the `nw-nwuser<username>-w` name format.

You supply a `mutate(spec)` callback that edits the freshly-read spec dict **in
place**; the helper re-runs it against the latest server spec on every retry, so
never capture or reuse a spec built outside the callback.

### Step 1: Load + mutate through the helper

```python
import sys; sys.path.insert(0, "skills/wandb-primary/scripts")
import wandb
from workspace_write import mutate_view_with_retry, read_back_spec

# 60, not the usual 120: view loads take seconds, and staying under the shell
# budget makes slow calls fail readably (see "Big projects" below).
api = wandb.Api(timeout=60)

VIEW_URL = "https://wandb.ai/ENTITY/PROJECT?nw=nwuserjuliarose"  # saved or personal

def mutate(spec):
    # Edit `spec` in place (see the examples below), then let the helper persist it.
    ...

result, _ = mutate_view_with_retry(
    api, entity=ENTITY, project=PROJECT,
    view_url=VIEW_URL, view_token=None, display_contains=None,
    mutate=mutate,
    # display_name=..., description=...,  # optional: only sent to rename/redescribe
)
```

### Step 2: Modify the spec (the body of `mutate`)

Edit any part of the `spec` dict the callback receives. Common operations:

```python
section = spec["section"]
rs = section["runSets"][0]

# -- Filtering -- (appends `AND state = finished` after the last item; see the
# connector rules under "Personal views (raw spec)")
rs["filters"]["filters"].append({
    "key": {"section": "run", "name": "state"},
    "op": "=", "value": "finished", "disabled": False, "connector": "AND",
})

# -- Grouping --
rs["grouping"] = [{"section": "config", "name": "model_type"}]

# -- Run coloring --
section["customRunColors"]["run_id_1"] = "#ff0000"

# -- Key-based coloring (semantic legend) --
section["settings"] = {
    **section.get("settings", {}),
    "semanticLegendSettings": {
        "runColorOptions": "semantic-legend",
        "key": "config:model_type",
        "numberOfBuckets": 5,
    },
}

# -- Workspace settings --
section["settings"]["xAxis"] = "_step"
section["settings"]["maxRuns"] = 10

# -- Section settings --
sections_list = section["panelBankConfig"]["sections"]
sections_list[0]["localPanelSettings"] = {"xAxis": "wall_time"}
```

### Adding a panel

`workspace_ops.py add-panel` already does this for line/bar/scalar panels
(builds the panel JSON, creates the section, runs `UpsertView`) and works on
personal `nwuser` views — prefer it. Build the panel JSON by hand only for panel
types the helper doesn't support.

Panels live in `spec["section"]["panelBankConfig"]["sections"]`. Each section
has `name`, `isOpen`, `panels` (list), and a unique `__id__`.

```python
import random, string
sections = spec["section"]["panelBankConfig"]["sections"]

# Find the target section
target = None
for s in sections:
    if s["name"].lower() == "training":
        target = s
        break

# Or create a new section
if target is None:
    target = {
        "__id__": "".join(random.choices(string.ascii_lowercase + string.digits, k=9)),
        "name": "Training",
        "isOpen": True,
        "sorted": 0,
        "pinned": False,
        "panels": [],
    }
    sections.insert(0, target)  # add at top

# Panel JSON format (matches what the UI stores)
new_panel = {
    "__id__": "".join(random.choices(string.ascii_lowercase + string.digits, k=9)),
    "viewType": "Run History Line Plot",
    "config": {
        "chartTitle": "Loss Curves",
        "metrics": ["train_loss", "val_loss"],
        "xAxis": "_step",
        "groupBy": "None",
        "legendFields": ["run:displayName"],
    },
    "isAuto": False,
}
target["panels"].append(new_panel)
```

`viewType` maps to the panel classes in the "Panel types" section above (e.g.
`"Run History Line Plot"` → `wr.LinePlot`, `"Bar Chart"` → `wr.BarPlot`, etc.).

### Step 3: Save + verify

`mutate_view_with_retry` (Step 1) performs the guarded `UpsertView` for you — do
**not** hand-write the mutation, choose an `id`/`name`, or set `lastUpdatedAt`; the
helper reads the view's `id` + `updatedAt`, guards on them, and manages the
create-vs-update distinction. It raises a clear error if the view kept changing past
the retry cap.

After it returns, verify against fresh server state — the mutation response alone
does not prove the spec landed:

```python
verify_spec = read_back_spec(api, ENTITY, PROJECT, result["view"]["name"])
# assert the specific field you changed is present in verify_spec
```

To **create** a saved view, use SDK `save()` (system-assigned token), then edit it
through the helpers for anything the SDK can't express. Never hand-invent an `nw`
token or `nw-<token>-v` name: a non-alphanumeric slug yields a `?nw=` URL the UI
cannot open even though the mutation succeeds and the view appears in the spec.

## Custom Vega panels (Vega2): envelope fields and gotchas

Envelope fields the Vega2 workspace renderer requires of a custom-chart panel:

| Field | Fact |
| --- | --- |
| `viewType` | `"Vega2"`. |
| `config.userQuery` | A structured `{"queryFields": [...]}` object — typically one `runSets` field with args `runSets: "${runSets}"` and `limit`, selecting `history` (with `keys`), `id`, and `name`. The renderer does not parse a serialized GraphQL string here. |
| `config.transform` | `{"name": "tableWithLeafColNames"}` — flattens the nested run/history response into the columns the Vega spec reads. Columns are named by **leaf key only** (`loss`, `_step`, `sigma`, `name`), and run-level fields are merged onto every flattened history row. Prefixed names like `runSets_history_loss` / `runSets_config_sigma` belong to the legacy `tableWithFullPathColNames` transform and match nothing here — a spec or `fieldSettings` referencing them renders a valid but **empty** chart. |
| `config.panelDefId` | A builtin preset (e.g. `wandb/line/v0`) or a chart published with `wandb.Api().create_custom_chart(...)`. `${field:...}` slots in the published spec resolve through `fieldSettings`. |
| `config.fieldSettings`, `config.stringSettings` | The panel's settings containers. |

The backend stores panel specs as opaque JSON: an envelope violating this
contract saves successfully and survives read-back, failing only at browser
render — and the failure is not contained to the panel. The whole workspace
shows "Unable to render this workspace" until the malformed panel is removed.

## Big projects: fast queries, split phases, reconcile against the request

On projects with tens of thousands of runs, specs run to hundreds of columns and
GraphQL calls get slow. Structure the work so no single command needs a long
budget and a killed command costs nothing:

- **Load one view, never all views.** `allViews` without `viewName` fetches every
  view's full spec and is the slowest call on a big project — it can burn an
  entire shell budget by itself. The `workspace_write` helpers already load
  narrow-first (single-view by name); avoid `list_views` on a big project. A
  filtered load returns in seconds where an unfiltered one gets killed.
- **Split load / transform / upsert / verify into separate shell commands.**
  Persist state to files between phases (the loaded spec, the intended column
  list, the modified spec). Each phase is fast and independently retryable, and
  a kill leaves files on disk telling you exactly where you stopped. Never fuse
  the phases into one script: if the shell kills a fused script after the
  mutation but before the verify prints, you cannot tell what landed.
- **Print progress unbuffered.** Run scripts with `python -u` or use
  `print(..., flush=True)`. A killed command then still shows which phase it
  reached; a kill after zero output is indistinguishable from a hang.
- **Set the HTTP timeout below the shell budget** (e.g. `wandb.Api(timeout=60)`
  for view work — the usual `timeout=120` is for run-data scans) so a slow
  request fails as a readable Python exception naming the call, instead of the
  shell killing the whole script (exit code 124) with no output. This only
  helps if the shell budget is *above* the HTTP timeout — do not request shell
  timeouts below the default.
- **A killed command is not a failed mutation.** After exit code 124, do a
  narrow read-back *first* — the upsert may have fully landed — then rerun only
  the phases that did not complete.
- **Reconcile against the request, not the read-back.** When a write is derived
  from a user-provided list (columns, panels, metrics), diff the result against
  that list *by name* and report anything that did not land. A plausible-looking
  count is not success. Never silently drop requested items — if a requested
  column is absent from the current spec's keyspace, check whether it is logged
  on the runs and tell the user either way (see `RUNS_TABLE.md`
  `## Column management`).

## URL ↔ view name mapping

| URL param | Internal view name | Type |
|---|---|---|
| `?nw=nwuserjuliarose` | `nw-nwuserjuliarose-w` | Personal workspace |
| `?nw=abc123def45` | `nw-abc123def45-v` | Saved view |

Personal views use `-w` suffix. Saved views use `-v` suffix.

To construct: `f"nw-{nw_param}-w"` for personal, `f"nw-{nw_param}-v"` for saved.

A view's backend **`id`** (`Vmlldzo…`) is a third, separate identifier — it is what
`find-view` and `list_views` report as `id`, and it never appears in a URL. Every
`workspace_ops.py` subcommand takes it as `--view-id`; `fetch_view` /
`mutate_view_with_retry` take it as `view_id=`. Passing an id where a token is
expected (or the reverse) still resolves, so the two are not worth agonizing over.

## Listing all views in a project

`workspace_ops.py find-view` (without a target flag) lists views; `list_views` from
`workspace_write` is the underlying call (returns each view's `id`/`name`/
`displayName`/`spec`/`updatedAt`). Re-target any listed view with
`find-view --view-id <id>`.

```python
import sys; sys.path.insert(0, "skills/wandb-primary/scripts")
import wandb
from workspace_write import list_views

api = wandb.Api(timeout=60)
for node in list_views(api, ENTITY, PROJECT):
    is_personal = "nwuser" in node["name"]
    print(f"{'[personal]' if is_personal else '[saved]   '} {node['displayName']}  ({node['name']})")
```

## Gotchas

1. **Personal views: SDK `from_url()` does not work.** The SDK validator rejects
   `nwuser` URLs on all current versions, and so does
   `save_workspace_edit_with_retry`. Edit their raw spec via `mutate_view_with_retry`
   (which accepts personal URLs/tokens) or use `workspace_ops.py`.

2. **Don't hand-write `UpsertView`.** The `workspace_write` helpers fetch the view's
   `id` + `updatedAt`, guard on them, and manage create-vs-update — a hand-rolled
   mutation that omits `id` silently creates a new view, and one that omits
   `lastUpdatedAt` clobbers concurrent edits. Always go through the helpers.

3. **`inserted=False` is success, not failure.** The `upsertView` payload the helpers
   return has `inserted=False` when updating an existing view (vs `inserted=True` for
   new views). Do not treat this as an error.

4. **Verify edits by re-fetching — but know what a spec read-back does and does
   not prove.** The mutation can return success while the panel does not render.
   Re-fetch the spec and assert the panel exists before reporting success.
   - **Native panels** (`BarPlot`/`LinePlot`/`ScalarChart`/`ScatterPlot`/…):
     spec presence confirms the panel **saved**, not that its data source
     resolves. A native panel whose metric keys don't match real logged keys
     saves fine but renders **empty**, and a title-and-existence read-back (what
     `workspace_ops.py add-panel` does) doesn't distinguish that from a populated
     chart (see the `summary_metrics.` prefix pitfall in gotcha 7).
   - **`CustomChart`/Vega panels**: spec presence does **not** imply render. A
     table/artifact-backed Vega panel can sit in the spec yet be silently
     dropped by the UI when its data source fails to resolve (see the caveat
     under "wandb.Tables in workspaces"). You cannot confirm render from the
     server. Do **not** report success on the basis of a spec read-back alone.
     Prefer recreating the chart as a native panel; if you must keep the Vega
     panel, tell the user it may not appear without a page refresh and ask them
     to confirm it renders, rather than claiming it is done.

5. **`shouldAutoGeneratePanels` setting.** Personal workspaces may have
   `settings.shouldAutoGeneratePanels = true`. When true, the UI auto-generates
   panels from logged metrics. Explicitly added panels coexist with auto-panels.
   If the auto-panel setting is on and sections have no explicit panels, the UI
   renders auto-generated ones that are not stored in the spec.

6. **BarPlot uses `metrics` (list), ScalarChart uses `metric` (singular).**
   LinePlot uses `y` (list). These are not interchangeable.

7. **`BarPlot(metrics=...)` takes bare logged metric-key strings.** Typed
   wrappers (`metrics=[wr.SummaryMetric("success_rate")]`) serialize into the
   saved spec as `config.metrics == ["summary_metrics.success_rate"]`; the
   workspace UI reads that as a literal metric name, finds no such key, and
   renders the bar panel **empty**. The bare form (`metrics=["success_rate",
   "cost_usd"]`) serializes to the raw keys the UI resolves against.

8. **No emoji in workspace names.** `ws.Workspace(name=...)` raises `ValueError`
   for emoji characters.

9. **`save_as_new_view()` vs `save()`.** `save()` updates in place (preserves ID
   and URL). `save_as_new_view()` always creates a new view with a fresh URL.

10. **Runset filters.** Use `ws.RunsetSettings(filters="State = 'finished'")` for
   string filters or structured `ws.Config("lr") == 0.001` expressions. The
   top-level filter list is AND-only.

11. **Section layout.** Use `ws.SectionLayoutSettings(columns=3, rows=2)` to
    control the panel grid within a section. Panels flow left-to-right,
    top-to-bottom.

11. **Panel objects have no `.to_json()`.** `wr.LinePlot`, `wr.BarPlot`, and
    other `wandb_workspaces.reports.v2` panel classes are not JSON-serializable
    via `to_json()`. To build workspace spec JSON by hand:
    `panel._to_model().model_dump(by_alias=True, exclude_none=True)` (then drop
    `layout` if needed — see `workspace_ops.py`). Prefer `workspace_ops.py
    add-panel` or SDK `ws.Workspace(...).save()` instead of hand-serializing.
    Never call name-mangled private methods like `_Base__to_model` — use the
    public `._to_model()` on the panel instance.
