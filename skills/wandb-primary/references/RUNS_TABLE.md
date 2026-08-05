<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Runs Table state and operations

Use this reference for the project Runs Table's programmable state: identity,
filters, grouping, sorting, visible/hidden runs, pins, baselines, columns,
search, colors, notes, moves, exports, and deletion. The state is stored in the
workspace view spec; use the guarded helpers described in `WORKSPACES.md` for
writes.

## Run identity

| Property | SDK access | Uniqueness | Mutable |
|---|---|---|---|
| Run ID | `run.id` / `api.run("ENTITY/PROJECT/RUN_ID")` | Unique | No |
| Display name | `run.name` | Not unique | Yes, then `run.update()` |

The Python SDK's `run.name` is the display name, not the immutable ID. Resolve
ambiguous display names before a mutation.

```python
run = api.run("ENTITY/PROJECT/RUN_ID")
run.name = "new-display-name"
run.update()
```

## Filters, grouping, sorting, and search

Prefer structured `wandb_workspaces` expressions:

```python
import wandb_workspaces.workspaces as ws

settings = ws.RunsetSettings(
    filters=[
        ws.Metric("State") == "finished",
        ws.Summary("loss") < 0.5,
    ],
    groupby=[ws.Config("learning_rate")],
    query="experiment-v2",
    regex_query=True,
)
```

Raw view-spec paths under `section.runSets[0]`:

| State | Path |
|---|---|
| Filters | `filters.filters` |
| Group keys | `grouping` |
| Sort priority | `sort.keys` |
| Search | `search` (`query` and `isRegex`) |

A filtered count covers matching runs, not necessarily visible runs. Use
`wandb_run_ops.py count-runs` for an exact server-side project count and
`project-snapshot` for a bounded project summary.

## Visibility, pins, and baseline

```python
ws.RunsetSettings(
    run_settings={
        "RUN_ID_A": ws.RunSettings(disabled=True),
        "RUN_ID_B": ws.RunSettings(disabled=False, color="#00ff00"),
    },
    pinned_runs=[
        "RUN_ID_B",
        ws.RunRef("RUN_ID_C", entity="OTHER_ENTITY", project="OTHER_PROJECT"),
    ],
    baseline_run="RUN_ID_B",
)
```

Cross-project pins are references; they do not copy data. In the raw spec,
pinned and baseline references are base64-encoded under
`pinnedRunIds` and `baselineRunId`. Visibility uses `selections.root` as the
default and `selections.tree` for per-run overrides. A baseline enables metric
deltas; `runFeed.showMetricDeltas` controls the global display and
`runFeed.metricValences` stores per-metric directionality.

Pinned/baseline behavior is not supported in grouped views, Reports,
single-run views, or non-line-plot panels.

## Columns

Column state lives in `section.runSets[0].runFeed`:

| State | Path |
|---|---|
| Visibility | `columnVisible` (`{name: bool}`) |
| Pinned columns | `columnPinned` (`{name: bool}`) |
| Order | `columnOrder` |
| Widths | `columnWidths` |

When applying a requested column list, deduplicate it by name while preserving
order. Do not discard names merely because they are absent from the old spec;
a logged key may never have appeared in the view. After a guarded write,
re-fetch and set-diff the requested and applied names. Count visible columns by
truthy values in `columnVisible`, not by dictionary size.

## Notes and mutable metadata

```python
run = api.run("ENTITY/PROJECT/RUN_ID")
if not run.read_only:
    run.notes = "Updated notes"
    run.tags = [*run.tags, "reviewed"]
    run.update()
```

`run.update()` supports fields including display name, notes, tags,
description, group, job type, and config. Mutate only fields the user named.

## Move, export, and delete

Cross-project pins are read-only references. Moving runs transfers ownership
and does not move historical Artifacts; the GraphQL `moveRuns` mutation returns
an asynchronous task that must be polled before reporting completion.

For training curves, export bounded history with explicit keys:

```python
run = api.run("ENTITY/PROJECT/RUN_ID")
df = run.history(samples=500, keys=["loss", "accuracy"])
```

There is no dedicated server-side Runs Table CSV endpoint; a programmatic CSV
export must query the required run/config/summary fields and write the local
file.

Run deletion is destructive:

```python
run.delete(delete_artifacts=False)
```

Confirm exact run IDs and whether Artifacts should be deleted. Deleted
Artifacts are permanently removed, and deleted run IDs cannot be reused.

## Workspace connection

Filters, grouping, sorting, visibility, pins, baseline, colors, and columns are
workspace-view state. Read fresh state before editing and persist with
`workspace_write.mutate_view_with_retry` or
`save_workspace_edit_with_retry`; both use optimistic concurrency and
read-back verification. See `WORKSPACES.md`.
