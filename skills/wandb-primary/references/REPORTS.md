<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Reports - wandb-workspaces v2

Use `wandb-workspaces` for programmatic W&B Report authoring. The fast recipe
lives in `SKILL.md` ("Create a W&B Report"); this file is the canonical
reference for report edits, filters, panels, loading, and sharing.

Only save, update, publish, or change sharing after the user asks. Constructing
`wr.Report(...)` is local until `report.save(...)` runs. Default to
`draft=True` when saving.

Use the package only when the task requires report or workspace editing. Install
`wandb-workspaces` or `wandb[workspaces]` if it is missing.

Required imports:

```python
import os

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
```

## Minimal report

```python
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

runset = wr.Runset(entity=entity, project=project, name="All runs")

report = wr.Report(
    entity=entity,
    project=project,
    title="Project analysis",
    description="Auto-generated summary.",
    width="fixed",  # "readable", "fixed", or "fluid"
    blocks=[
        wr.H1("Project analysis"),
        wr.P("Summary of recent W&B runs."),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.LinePlot(title="Loss", x="Step", y=["loss"]),
                wr.BarPlot(title="Accuracy", metrics=["accuracy"], orientation="v"),
                wr.ScalarChart(metric="accuracy", groupby_aggfunc="mean"),
            ],
        ),
    ],
)

report.save(draft=True)
print(f"Report saved: {report.url}")
```

## Runsets and structured filters

A report's plots come from `Runset` objects inside `PanelGrid` blocks. The
runset filters decide which runs appear. Prefer structured filters over raw
filter strings.

```python
runset = wr.Runset(
    entity=entity,
    project=project,
    name="Finished high-accuracy runs",
    filters=[
        ws.Metric("State") == "finished",
        ws.Summary("accuracy") > 0.95,
        ws.Config("learning_rate") == 0.001,
    ],
)
```

Useful filter constructors:

- `ws.Metric("State")`, `ws.Metric("Name")`, and
  `ws.Metric("CreatedTimestamp")` for run-level fields.
- `ws.Metric("name")` for the run ID backend field. Do not use `ID`.
- `ws.Summary("metric_name")` for final metrics from `wandb.log()`.
- `ws.Config("key")` for config values.
- `ws.Tags()` for run tags.

Useful operators include `==`, `!=`, `<`, `>`, `<=`, `>=`, `.isin([...])`,
`.notin([...])`, and `within_last(...)` on timestamp fields.

## Explicit run selection by ID

For specific runs, filter on `ws.Metric("name")`. For "top N by metric", query
run IDs first with `wandb.Api()`, then pass them into the report runset.

```python
import wandb

api = wandb.Api(timeout=120)
runs = api.runs(f"{entity}/{project}", per_page=200)[:200]
top_ids = [
    run.id
    for run in sorted(
        runs,
        key=lambda run: run.summary.get("accuracy", -1),
        reverse=True,
    )[:5]
]

runset = wr.Runset(
    entity=entity,
    project=project,
    name="Top accuracy runs",
    filters=[ws.Metric("name").isin(top_ids)],
)
```

## Tags and recent runs

For tag filters, use structured `ws.Tags()` filters. Do not generate string
filters for tags. In particular, do not write string filters that put one tag in
parentheses, because that can be parsed as a parenthesized string instead of a
collection. For a single dynamic tag, always wrap the value in a list:
`ws.Tags().isin([dataset])`.

```python
variants = ["baseline", "candidate"]
dataset = "eval-set-a"
exclude_tags = ["apa", "manual-review"]

tagged = wr.Runset(
    entity=entity,
    project=project,
    name=f"Runs on {dataset}",
    filters=[
        ws.Tags().isin(variants),
        ws.Tags().isin([dataset]),
        ws.Tags().notin(exclude_tags),
        ws.Metric("State") == "finished",
    ],
)

recent = wr.Runset(
    entity=entity,
    project=project,
    filters=[ws.Metric("CreatedTimestamp").within_last(7, "days")],
)
```

## Multiple runsets

Use multiple runsets when the report compares groups. Name each runset clearly
so the report UI tabs are meaningful.

```python
wr.PanelGrid(
    runsets=[
        wr.Runset(
            name="Adam",
            entity=entity,
            project=project,
            filters=[ws.Config("optimizer") == "adam"],
        ),
        wr.Runset(
            name="SGD",
            entity=entity,
            project=project,
            filters=[ws.Config("optimizer") == "sgd"],
        ),
    ],
    panels=[wr.LinePlot(x="Step", y=["loss", "accuracy"])],
)
```

## OR and grouped filters

Use `Or` and `And` from `wandb_workspaces.expr` for grouped logic.

```python
from wandb_workspaces.expr import And, Or

runset = wr.Runset(
    entity=entity,
    project=project,
    filters=And(
        ws.Metric("State") == "finished",
        Or(
            ws.Config("learning_rate") == 0.01,
            ws.Config("learning_rate") == 0.1,
        ),
    ),
)
```

## Panels

Use concrete metric keys when the user names metrics. Use `metric_regex` when
the user asks for a family of metrics.

Do not invent component classes. Supported top-level blocks include `H1`, `H2`,
`H3`, `P`, `MarkdownBlock`, `PanelGrid`, `CodeBlock`, `Image`,
`HorizontalRule`, `CalloutBlock`, `BlockQuote`, `LatexBlock`,
`TableOfContents`, `OrderedList`, `UnorderedList`, and `CheckedList`. Supported
panels include `LinePlot`, `BarPlot`, `ScalarChart`, `ScatterPlot`,
`ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`,
`MediaBrowser`, `CodeComparer`, `MarkdownPanel`, `WeavePanelSummaryTable`,
`WeavePanelArtifact`, `WeavePanelArtifactVersionedFile`, and `CustomChart`.
Prefer a native panel; use `CustomChart` only when no native type can express the
required visualization.

```python
panels = [
    wr.LinePlot(title="Validation metrics", x="Step", metric_regex="val/.*"),
    wr.BarPlot(title="Accuracy", metrics=["accuracy"], orientation="v"),
    wr.ScalarChart(metric="accuracy", groupby_aggfunc="max"),
]
```

Every panel accepts `layout=wr.Layout(x, y, w, h)` on a 24-unit grid. Use
`w=24` for dense analytical plots and comparison/table panels; reserve smaller
widths for compact scalar or simple bar panels. Pair wide panels with
`Report(width="fluid")` when readability matters.

Group by a run attribute or config value on the `Runset`, not on a panel:

```python
runset = wr.Runset(
    entity=entity,
    project=project,
    groupby=[ws.Config("learning_rate")],
)
panel = wr.LinePlot(
    x="Step",
    y=["loss"],
    groupby_aggfunc="mean",
    groupby_rangefunc="stddev",
)
```

`BarPlot.metrics` must contain bare summary-key strings. Do not wrap them in
`SummaryMetric` objects.

For media, use `wr.Image` for URL-backed static image blocks and
`wr.MediaBrowser` for media logged to runs. Use `gallery_axis` for gallery
views and `grid_x_axis` / `grid_y_axis` for grid views. If you configure both
gallery and grid axes, set `mode` explicitly.

```python
blocks = [
    wr.Image(url="https://example.com/example.png", caption="Example output"),
    wr.PanelGrid(
        panels=[
            wr.MediaBrowser(
                title="Gallery by step",
                media_keys=["train_image"],
                gallery_axis="step",
            ),
        ],
    ),
    wr.PanelGrid(
        panels=[
            wr.MediaBrowser(
                title="Validation samples",
                media_keys=["val_image"],
                grid_x_axis="index",
                grid_y_axis="run",
            ),
        ],
    ),
    wr.PanelGrid(
        panels=[
            wr.MediaBrowser(
                title="Grid mode with prepared gallery axis",
                media_keys=["image"],
                mode="grid",
                gallery_axis="step",
                grid_x_axis="step",
                grid_y_axis="run",
            ),
        ],
    ),
]
```

## Per-run display settings

Use `wr.RunSettings` to color or hide specific runs. Keys are W&B run IDs.

```python
runset = wr.Runset(
    entity=entity,
    project=project,
    run_settings={
        "RUN_ID_A": wr.RunSettings(color="#FF0000"),
        "RUN_ID_B": wr.RunSettings(disabled=True),
    },
)
```

## Runs-table columns

Use column controls when the report should show a curated runs-table view.

Column format:

- `run:state` for run properties.
- `summary:accuracy` for summary metrics.
- `config:learning_rate.value` for config values.
- `tags:__ALL__` for tags.

```python
runset = wr.Runset(
    entity=entity,
    project=project,
    pinned_columns=["summary:accuracy"],
    visible_columns=[
        "summary:loss",
        "config:learning_rate.value",
        "run:state",
    ],
    lock_columns=True,
)
```

## Load and update a report

Use `wr.Report.from_url(...)` to load an existing report. Saving mutates the
report, so ask before calling `save()`.

```python
report = wr.Report.from_url("https://wandb.ai/ENTITY/PROJECT/reports/TITLE--ID")
report.blocks.append(wr.H2("New analysis section"))
report.blocks.append(wr.P("Added programmatically."))
report.save(draft=True)
print(f"Updated report: {report.url}")
```

`from_url()` does not reliably restore a saved report's width. Re-set
`report.width` before saving an edit, and inspect
`wr.Report.from_url(url, as_model=True).spec.width` when width is part of the
requested change. Always load the existing report by URL; constructing a fresh
`Report` mints a new ID instead of editing the old one.

## Read and analyze a report

A Report stores narrative blocks plus runset and panel definitions; it does not
own the metric values. Walk `report.blocks` in order to recover the narrative,
then resolve each runset to W&B runs before validating numeric claims.

- `H1`, `H2`, `H3`, `P`, and `MarkdownBlock` expose text.
- `PanelGrid` exposes runsets and panel definitions.
- `BarPlot` and `ScalarChart` read summary values.
- `LinePlot` needs bounded `run.history(keys=[...], samples=N)` calls.
- System metrics require `stream="system"`.
- Media, Table, and custom panels reference logged keys whose backing data must
  be fetched separately.

Display a `wandb.Table` logged under a summary key with:

```python
wr.WeavePanelSummaryTable(
    table_name="predictions",
    layout=wr.Layout(w=24, h=12),
)
```

The panel shows the last logged table version for that summary key.

## Save and verify

Use the bundled helpers so every save is read back from W&B:

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")

from report_helpers import edit_report_verified, save_report_verified

result = save_report_verified(report, draft=True, expect_text="Project analysis")
assert result["verified"]
```

`draft=True` is the safe non-publishing default, but drafts are not editable in
the report editor. Pass `draft=False` only when the user asked to publish or to
create an editable published report. A successful `save()` call proves only that
the request returned; the read-back must contain the expected blocks/text before
reporting success.

## Additional gotchas

- Report titles containing `/` create slugs that `from_url()` cannot parse.
- `wr.P()` renders literal text; use `wr.MarkdownBlock` for Markdown.
- Use structured filters; dot-path filter strings can produce invalid specs.
- `Runset.query` is a run-name regex, not a tags/config query language.
- `OrderBy` accepts backend run fields or typed summary/config references. A
  bare `summary:key` string can save successfully and still break rendering.
- `wandb.init()` is unnecessary for report authoring and creates an unwanted
  utility run.
- `MediaBrowser` requires history-logged media. Summary-only media should be a
  top-level `wr.Image` block.
- A spec read-back does not prove a Vega/custom panel or persisted styling field
  actually renders; prefer native panels and state this limitation.

## Share links

Share links make a report viewable by anyone with the URL, even if the project
is private. Only enable or disable them when explicitly requested.

```python
report = wr.Report.from_url("https://wandb.ai/ENTITY/PROJECT/reports/TITLE--ID")

share_url = report.enable_share_link()
print(f"Share URL: {share_url}")

print(f"Current share URL: {report.get_share_url()}")

# To revoke:
# report.disable_share_link()
```
