<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Reports with wandb-workspaces v2

This reference is a follow-up to PR #36, which ported `wandb-primary` from the
latest internal core source. It gives the team a separate reference for programmatic report
definitions using the current `wandb-workspaces` v2 API.

Use this file when a user asks to create, update, load, share, or reason about a
W&B Report programmatically. Report creation and sharing mutate W&B state, so
only save or change sharing settings after the user explicitly asks.

## Imports

Use `wandb_workspaces.reports.v2` for report objects, blocks, panels, runsets,
save/load/share behavior, and per-run settings. Use
`wandb_workspaces.workspaces` for structured run filters.

```python
import os

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
```

Install `wandb-workspaces` or `wandb[workspaces]` only when the task requires
programmatic report or workspace editing.

## Minimal report

```python
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

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

Prefer `draft=True` unless the user explicitly asks to publish. Always print the
URL after saving.

## Runsets and structured filters

`wr.Runset` controls which runs appear in a `wr.PanelGrid`. Prefer structured
filters over raw string filters because they are easier to generate safely and
survive field-name mistakes better.

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

panel_grid = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(x="Step", y=["loss", "accuracy"]),
        wr.ScalarChart(metric="accuracy", groupby_aggfunc="max"),
    ],
)
```

Useful filter constructors:

- `ws.Metric("State")`, `ws.Metric("Name")`, and `ws.Metric("CreatedTimestamp")`
  for run-level fields.
- `ws.Summary("metric_name")` for summary metrics from `wandb.log()`.
- `ws.Config("key")` for config values.
- `ws.Tags()` for run tags.

Supported operators include `==`, `!=`, `<`, `>`, `<=`, `>=`, `.isin([...])`,
`.notin([...])`, and `within_last(...)` on timestamp metrics.

## Multiple runsets

Use multiple runsets when the report compares groups. Give each runset a clear
name so the report UI tabs are meaningful.

```python
report = wr.Report(
    entity=entity,
    project=project,
    title="Optimizer comparison",
    blocks=[
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
        ),
    ],
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

## Recent runs

Use `within_last` for time windows.

```python
recent = wr.Runset(
    entity=entity,
    project=project,
    filters=[ws.Metric("CreatedTimestamp").within_last(7, "days")],
)
```

## Metric regex panels

Use `metric_regex` to let the backend select matching y-axis metrics.

```python
wr.LinePlot(
    title="Validation metrics",
    x="Step",
    metric_regex="val/.*",
)
```

## Images and media

Use `wr.Image` for static image blocks from a URL. Use `wr.MediaBrowser` for
media logged to runs.

```python
blocks = [
    wr.H1("Qualitative examples"),
    wr.Image(url="https://example.com/example.png", caption="Example output"),
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
]
```

## Per-run settings

Use `wr.RunSettings` to color or hide specific runs. Keys are W&B run IDs.

```python
highlighted = wr.Runset(
    entity=entity,
    project=project,
    run_settings={
        "RUN_ID_A": wr.RunSettings(color="#FF0000"),
        "RUN_ID_B": wr.RunSettings(disabled=True),
    },
)
```

## Runs-table columns

Use column controls when the report should show only a curated runs-table view.

Column format:

- `run:state` for run properties.
- `summary:accuracy` for summary metrics.
- `config:learning_rate.value` for config values.
- `tags:__ALL__` for tags.

```python
curated = wr.Runset(
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

## Load and update an existing report

Use `wr.Report.from_url(...)` to load an existing report, mutate the object, and
save it again. This changes the report, so ask the user before saving.

```python
report = wr.Report.from_url("https://wandb.ai/ENTITY/PROJECT/reports/TITLE--ID")
report.blocks.append(wr.H2("New analysis section"))
report.blocks.append(wr.P("Added programmatically."))
report.save(draft=True)
print(f"Updated report: {report.url}")
```

## Share links

Share links make a report viewable by anyone with the URL, even if the project
is private. Only enable or disable them when the user explicitly requests it.

```python
report = wr.Report.from_url("https://wandb.ai/ENTITY/PROJECT/reports/TITLE--ID")

share_url = report.enable_share_link()
print(f"Share URL: {share_url}")

print(f"Current share URL: {report.get_share_url()}")

# To revoke:
# report.disable_share_link()
```

## Migration notes from older examples

- Prefer `import wandb_workspaces.reports.v2 as wr` over the older
  `wandb.apis` reports import path.
- Prefer `import wandb_workspaces.workspaces as ws` for structured filters.
- Replace `expr.Config(...)`, `expr.Summary(...)`, `expr.Metric(...)`, and
  `expr.Tags()` examples with `ws.Config(...)`, `ws.Summary(...)`,
  `ws.Metric(...)`, and `ws.Tags()` in new docs and snippets.
- Keep report saves explicit. Constructing `wr.Report(...)` is local until
  `report.save(...)` is called.
