<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Reports — Runset filters and authoring

Full reference for `wandb.apis.reports` (the `wandb_workspaces` library).
The fast recipe lives in `SKILL.md` ("Create a W&B Report"); this file
covers customization — Runset filters, ordering, run selection by ID,
tag-based selection, and the gotchas that crash the report UI.

Use `wandb[workspaces]` or the `wandb-workspaces` package if it is available in the user's environment; install it only when the task requires report or workspace editing.

Required imports:

```python
from wandb.apis import reports as wr
import wandb_workspaces.expr as expr
```

A report's plots come from `Runset` objects inside a `PanelGrid`. The Runset's
filters decide which runs appear.

## Runset filters — structured (preferred)

`filters` accepts a list of `expr.FilterExpr` (ANDed together). Always prefer
this over raw filter strings; it validates, never crashes the UI, and survives
field renames.

```python
runset = wr.Runset(
    entity=entity, project=project, name="Top GPT-4 runs",
    filters=[
        expr.Config("model") == "gpt-4",
        expr.Summary("accuracy") >= 0.9,
        expr.Metric("State") == "finished",
    ],
)
```

Constructors:

- `expr.Config("<key>")` — a hyperparameter from `run.config`.
- `expr.Summary("<key>")` — a final metric from `run.summary`.
- `expr.Metric("<system_field>")` — backend run fields. The most useful are
  `Metric("State")`, `Metric("Name")` (display name), `Metric("name")` (run ID
  — note lowercase, this is the backend field).
- `expr.Tags()` — supports `.isin([...])` for tag-based selection.

Operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `.isin([...])`.

## Explicit run selection by ID

The backend field for run ID is `name` (lowercase). Do NOT use `ID`.

```python
runset = wr.Runset(
    entity=entity, project=project, name="Selected runs",
    filters=[expr.Metric("name").isin(["abc123", "def456"])],
)
```

For "top N by metric": query the IDs first with `wandb.Api()`, then pass them in.

```python
api = wandb.Api(timeout=120)
sample = api.runs(f"{entity}/{project}", per_page=200)[:200]
top_ids = [r.id for r in sorted(
    sample, key=lambda r: r.summary.get("accuracy", -1), reverse=True
)[:5]]
runset = wr.Runset(
    entity=entity, project=project,
    filters=[expr.Metric("name").isin(top_ids)],
)
```

## Tag-based selection

Use `expr.Tags()` instead of the legacy `query="tags:..."` syntax.

```python
runset = wr.Runset(
    entity=entity, project=project,
    filters=[expr.Tags().isin(["healthy_baseline", "exploding_gradients"])],
)
```

## Ordering

```python
runset = wr.Runset(
    entity=entity, project=project,
    order=[wr.OrderBy(name="CreatedTimestamp", ascending=False)],
)
```

## `query` is a regex on run name only

`query` mirrors the W&B UI search box and is **only** a regex over the run
display name. It does not understand `tags:...`, `config:...`, or any field
syntax. Use structured `filters` for everything else; reach for `query` only
when you actually want a name regex (e.g. `query="healthy_baseline|exploding_gradients"`).

If you do use a string filter, preflight it with `expr.expr_to_filters(...)`
and confirm every leaf filter has a `key` before passing it to a Runset.

## Dot-path warning

Never put dot-paths in filter strings: `"config.lr"`, `"summary.loss"`,
`"tags.foo"` all parse to missing keys and can crash the report UI. Always use
`expr.Config("lr")`, `expr.Summary("loss")`, `expr.Tags()`, etc.

## Width and saving

```python
report = wr.Report(
    entity=entity, project=project,
    title="Project analysis",
    description="Summary of recent runs",
    width="fixed",  # "fixed" (medium, default) | "readable" (narrow) | "fluid" (full)
    blocks=[
        wr.H1(text="Project analysis"),
        wr.P(text="Auto-generated summary from W&B API."),
        wr.PanelGrid(runsets=[runset], panels=[
            wr.LinePlot(title="Loss", x="_step", y=["LOSS_KEY"]),
            wr.BarPlot(title="Accuracy", metrics=["ACC_KEY"], orientation="v"),
        ]),
    ],
)
report.save(draft=True)
print(f"Report saved: {report.url}")
```

`report.save()` is mutating — only call it after the user has approved
publishing. Default to `draft=True`. Print `report.url` so the user can open it.
