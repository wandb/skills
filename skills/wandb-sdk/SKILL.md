---
name: wandb-sdk
description: Query and analyze W&B projects, runs, metrics, and Weave traces using the wandb and weave Python SDKs directly. Use wandb.Api() for runs/metrics and weave.init() for traces/evals.
---

# Weights & Biases (SDK)

Use the `wandb` and `weave` Python SDKs to query W&B data. Both are pre-installed. `WANDB_API_KEY` is set in your environment.

**Two SDKs, two purposes:**
- **`wandb`** -- query W&B runs, configs, metrics, artifacts, sweeps
- **`weave`** -- query Weave traces/calls, evaluations, scorers, token usage

## Quick start

```python
import os, wandb, weave

entity = os.environ.get("WANDB_ENTITY") or "<ask-user>"
project = os.environ.get("WANDB_PROJECT") or "<ask-user>"

# W&B runs/metrics
api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# Weave traces/evals
client = weave.init(f"{entity}/{project}")
calls = client.get_calls(limit=10)
```

## Weave SDK

### Initialization

```python
import weave
client = weave.init("entity/project")
```

### Querying calls

```python
from weave.trace_server.trace_server_interface import CallsFilter

# All calls (paginated)
calls = client.get_calls(limit=100)

# Root traces only
calls = client.get_calls(
    filter=CallsFilter(trace_roots_only=True),
    limit=100,
)

# Filter by op name (wildcard for version hash)
op_ref = f"weave:///{client.entity}/{client.project}/op/Evaluation.evaluate:*"
calls = client.get_calls(filter=CallsFilter(op_names=[op_ref]))

# Sort by most recent
calls = client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=10,
)

# Count without fetching all data
count = len(client.get_calls(filter=CallsFilter(trace_roots_only=True)))

# Convert to pandas
df = client.get_calls(limit=500).to_pandas()
```

**Key `get_calls()` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filter` | `CallsFilter` | High-level filter (op_names, parent_ids, trace_ids, trace_roots_only, call_ids) |
| `limit` | `int` | Max calls to return |
| `offset` | `int` | Skip N calls (pagination) |
| `sort_by` | `list[dict]` | `[{"field": "started_at", "direction": "desc"}]` |
| `query` | `Query` | MongoDB-style advanced filter |
| `include_costs` | `bool` | Include token cost data |
| `columns` | `list[str]` | Restrict fields returned |

### Advanced query (MongoDB-style)

```python
from weave.trace_server.interface.query import Query

# Error calls (exception is not null)
error_calls = client.get_calls(
    query=Query(**{
        "$expr": {
            "$not": [{"$eq": [{"$getField": "exception"}, {"$literal": None}]}]
        }
    })
)

# Op name contains substring
calls = client.get_calls(
    query=Query(**{
        "$expr": {
            "$contains": {
                "input": {"$getField": "op_name"},
                "substr": {"$literal": ".score"},
                "case_insensitive": True
            }
        }
    })
)
```

### Reading call data

```python
call = client.get_call("call-id")

call.id              # call UUID
call.op_name         # full ref URI
call.func_name       # function name (e.g. "Evaluation.evaluate")
call.trace_id        # trace UUID
call.parent_id       # None for root traces
call.started_at      # datetime
call.ended_at        # datetime
call.inputs          # dict: function arguments
call.output          # return value
call.exception       # error message if failed

# Status
status = call.summary.get("weave", {}).get("status")

# Token usage (keyed by model name)
usage = call.summary.get("usage", {})
for model, u in usage.items():
    input_tokens = u.get("input_tokens") or u.get("prompt_tokens") or 0
    output_tokens = u.get("output_tokens") or u.get("completion_tokens") or 0
```

### Evaluation call hierarchy

```
Evaluation.evaluate (root)
  ├── Evaluation.predict_and_score (per row × trials)
  │     ├── model.predict
  │     ├── scorer_1.score
  │     └── scorer_2.score
  └── Evaluation.summarize
```

## W&B SDK

### Initialization

```python
import wandb
api = wandb.Api()
```

### Listing and filtering runs

```python
# All runs
runs = api.runs("entity/project")

# Filtered
runs = api.runs("entity/project", filters={"state": "finished"})
runs = api.runs("entity/project", filters={"config.model": "gpt-4"})
runs = api.runs("entity/project", filters={"summary_metrics.accuracy": {"$gt": 0.9}})

# Sorted
runs = api.runs("entity/project", order="-created_at")

# Count
total = runs.length
```

### Reading run data

```python
run = api.run("entity/project/run-id")
run.id, run.name, run.state, run.config, run.summary, run.tags, run.url

# Metrics history (sampled)
df = run.history(samples=500, keys=["loss", "accuracy"])

# Full unsampled history (iterator)
for row in run.scan_history(keys=["loss"]):
    print(row["_step"], row["loss"])
```

### Listing projects

```python
projects = api.projects("entity")
for p in projects:
    print(p.name)
```

### Summarize a project

```python
import pandas as pd

runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=200)
sample = runs[:200]

rows = []
for run in sample:
    summary = getattr(run.summary, "_json_dict", {})
    rows.append({
        "id": run.id, "name": run.name, "state": run.state,
        "created_at": run.created_at,
        **{k: summary.get(k) for k in ["loss", "accuracy"]},
    })

df = pd.DataFrame(rows)
print(df.describe())
```

## Report authoring

```python
from wandb.apis import reports as wr
import wandb_workspaces.expr as expr

runset = wr.Runset(entity=entity, project=project, name="All runs")
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["loss"]),
        wr.BarPlot(title="Accuracy", metrics=["accuracy"], orientation="v"),
    ],
)

report = wr.Report(
    entity=entity, project=project,
    title="Project analysis",
    description="Summary of recent runs",
    width="fixed",
    blocks=[wr.H1(text="Project analysis"), plots],
)
# report.save(draft=True)
```

## Reminders

- Always use `limit` with `get_calls()` -- large projects have thousands of calls.
- Use `columns` to restrict fetched fields for performance.
- `len(calls)` makes a separate count query, not fetching all data.
- Op names need the full ref URI: `weave:///{entity}/{project}/op/{name}:*`
- W&B `api.runs()` returns lazy iterators -- use slicing or break early.
- Prefer small samples first, then expand if needed.
- Use pandas for aggregates; share only compact summaries.
