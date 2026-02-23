---
name: wandb-data-analysis
description: Explore and analyze W&B (Weights & Biases) projects, runs, metrics, and configs via wandb.Api for data analysis, summarization, filtering, or reporting. Use when listing projects, summarizing a project or run set, aggregating metrics, or creating W&B Reports.
---

# W&B Data Analysis

## Quick start

- Determine `entity` and `project` (ask the user or use environment variables like `WANDB_ENTITY`/`WANDB_PROJECT`).
- Initialize `wandb.Api()` and confirm credentials (set `WANDB_API_KEY` or run `wandb login` if needed).

```python
import os
import wandb

entity = os.environ.get("WANDB_ENTITY") or "<ask-user>"
project = os.environ.get("WANDB_PROJECT") or "<ask-user>"
api = wandb.Api()
```

## Zoomed-out to detailed workflow

### 1) List projects (largest zoom)

- Use `api.projects(entity, per_page=...)` and convert to a list.
- Sort locally by last activity when possible; fall back to `created_at`.
- Avoid loading thousands of projects into context; show only the top N.

```python
from datetime import datetime

projects = list(api.projects(entity, per_page=200))

def parse_time(ts):
    if not ts:
        return datetime.min
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.min

# last_run_time is often missing; created_at is usually present
def project_sort_key(p):
    last = parse_time(getattr(p, "last_run_time", None))
    created = parse_time(getattr(p, "created_at", None))
    return last if last != datetime.min else created

projects = sorted(projects, key=project_sort_key, reverse=True)

for p in projects[:20]:
    print(p.name, getattr(p, "created_at", None), getattr(p, "last_run_time", None))
```

### 2) Summarize a project quickly

- Use `api.runs(path, order="-created_at", per_page=...)` to page.
- Avoid `list(runs)` on large projects; slice to sample: `runs[:N]`.
- Summarize with pandas; do not print every metric name.

```python
import pandas as pd

runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=200)
sample = runs[:200]

rows = []
for run in sample:
    summary = getattr(run.summary, "_json_dict", {})
    rows.append({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "user": getattr(run.user, "username", None),
        "tags": list(run.tags or []),
        "loss": summary.get("loss"),
        "accuracy": summary.get("accuracy"),
    })

df = pd.DataFrame(rows)
print(df.head(10))
print(df["state"].value_counts())
```

### 3) Inspect runs and metrics in detail

- Use `run.summary` for final metrics and `run.config` for hyperparameters.
- Use `run.history(samples=..., keys=[...])` for quick, sampled pulls (returns pandas by default).
- Use `run.scan_history(keys=[...], page_size=...)` to stream all history rows.
- Limit keys to avoid massive metric dumps.

```python
run = sample[0]
summary = getattr(run.summary, "_json_dict", {})
config = dict(run.config)

hist = run.history(samples=200, keys=["loss", "accuracy", "_step"], pandas=True)
print(hist.tail())

# Stream without materializing everything
for i, row in enumerate(run.scan_history(keys=["loss", "accuracy"], page_size=1000)):
    if i >= 5:
        break
    print(row)
```

### 4) Aggregate across runs without blowing up context

- Keep the metric list small and explicit.
- Use pandas to compute aggregates and only print summaries.

```python
metric_keys = ["loss", "accuracy"]
rows = []
for run in sample:
    summary = getattr(run.summary, "_json_dict", {})
    rows.append({
        "id": run.id,
        "state": run.state,
        **{k: summary.get(k) for k in metric_keys},
    })

metrics_df = pd.DataFrame(rows)
print(metrics_df.describe(include="all"))
```

### 5) Use filters and paging to avoid expensive scans

- Use `filters` to narrow results (server-side) and `order` to prioritize.
- Use `per_page` and slices to page through results.

```python
filters = {"state": "finished", "config.model": "gpt-4"}
runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at", per_page=100)
recent = runs[:50]
```

## Large history to pandas (many fields)

- `run.history(..., pandas=True)` is fastest for a small, sampled DataFrame.
- `run.scan_history(...)` is the only built-in way to retrieve *all* history rows; convert to pandas in chunks.
- `scan_history(keys=[...])` only returns rows that include **all** keys, so keep the key list small.
- Avoid `keys=None` on very large, wide runs unless you stream to disk.

```python
import itertools
import pandas as pd

history_iter = run.scan_history(keys=["loss", "accuracy"], page_size=2000)
chunks = []
for chunk in iter(lambda: list(itertools.islice(history_iter, 5000)), []):
    chunks.append(pd.DataFrame.from_records(chunk))

df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
print(df.describe(include="all"))
```

## Report authoring (W&B Reports)

- Install the reports extra once: `uv pip install "wandb[workspaces]"`.
- Use `wandb.apis.reports` to create a report and save it.
- `report.save(...)` is mutating; only call it when asked to publish.
- Report widths: prefer `fixed` (medium). Other options: `readable` (narrow), `fluid` (full).

```python
from wandb.apis import reports as wr

runset = wr.Runset(entity=entity, project=project, name="All runs")
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["loss"]),
        wr.BarPlot(title="Accuracy", metrics=["accuracy"], orientation="v"),
    ],
)

report = wr.Report(
    entity=entity,
    project=project,
    title="Project analysis",
    description="Summary of recent runs",
    width="fixed",
    blocks=[
        wr.H1(text="Project analysis"),
        wr.P(text="Auto-generated summary from W&B API."),
        plots,
    ],
)

# report.save(draft=True)
```

### Runset: select specific runs for plots

- A `Runset` defines which runs appear in plots inside a `PanelGrid`.
- Use `filters` for structured selection (preferred) or `query` for a UI-style search string.
- `query` is a **regex search on run name/displayName**, not a filter language. Avoid `tags:...` or `config:...` here.
- `filters` can be:
  - a string expression (supports `=`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `in`, `not in`)
  - a list of `expr.FilterExpr` objects (list is ANDed)
- To target explicit runs, filter by run id (`name`) or display name, or use tags/groups.
- Avoid dot-paths like `config.lr` or `summary.loss` in filter strings; they parse to missing keys and can crash the report UI. Use `Config("lr")`, `Summary("loss")`, `Metric("State")`, `Metric("Name")`, or `Tags().isin([...])` instead.

```python
from wandb.apis import reports as wr
import wandb_workspaces.expr as expr

# Structured filters (ANDed)
filters = [
    expr.Config("model") == "gpt-4",
    expr.Summary("accuracy") >= 0.9,
    expr.Metric("State") == "finished",
]

runset = wr.Runset(
    entity=entity,
    project=project,
    name="Top runs",
    filters=filters,
)

# Explicit run IDs or names (preferred over query)
runset_ids = wr.Runset(
    entity=entity,
    project=project,
    name="Selected runs",
    # Use backend field "name" for run IDs; "ID" produces a key that won't match.
    filters=[expr.Metric("name").isin(["runid1", "runid2"])],
)

# Tags-based selection (do this instead of query="tags:...")
runset_tags = wr.Runset(
    entity=entity,
    project=project,
    name="Tag selection",
    filters=[expr.Tags().isin(["healthy", "exploding_gradients"])],
)
```

```python
# Order runs by a field
runset = wr.Runset(
    entity=entity,
    project=project,
    order=[wr.OrderBy(name=\"CreatedTimestamp\", ascending=False)],
)
```

Notes:
- `query` mirrors the W&B UI search box and is useful for quick ad-hoc filters.
- If you use `query`, write a regex that matches run names (e.g., `\"healthy_baseline|exploding_gradients\"`).
- Use tags/groups to keep run selection stable (e.g., tag the runs you want to plot).
- If you need “top N” runs by a metric, compute IDs with `wandb.Api()` first, then pass `filters=[expr.Metric(\"name\").isin([...])]` to the Runset.
- If you use a string filter, preflight it with `expr.expr_to_filters(...)` and confirm every leaf filter has a `key`.
- For run IDs, use `name in [...]` or `Metric(\"name\") in [...]`. Do not use `ID` in filters.

## Reminders

- Prefer small samples first, then expand if needed.
- Avoid printing full `run.summary.keys()` or full metric lists.
- Use pandas to compute aggregates and share only compact summaries.
- If project-level metadata is sparse, derive activity from run timestamps.
