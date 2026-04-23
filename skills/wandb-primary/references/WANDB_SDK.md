<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B SDK Reference — Runs, Metrics, Artifacts

Reference for querying Weights & Biases training data using the `wandb` Python SDK. Covers runs, metrics history, artifacts, sweeps, and system metrics.

**Key principle**: Always load data into pandas/numpy for analysis. Never dump raw history or run lists into context — it will overwhelm your working memory. Look at structure first, then query specific fields.

## CRITICAL: API initialization for large projects

```python
import wandb
import pandas as pd
import numpy as np

# ALWAYS use timeout=60 (or higher) for large projects.
# The default 19s timeout causes constant failures on projects with 10K+ runs
# or runs with 1K+ summary metrics.
api = wandb.Api(timeout=60)

import os
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# Use per_page to minimize pagination round-trips on large projects
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)

# AVOID len(runs) on large projects — it triggers a slow count query (5s+)
# Instead, just slice directly:
first_50 = runs[:50]
```

---

## Runs

### Listing and filtering

```python
# All runs (lazy iterator)
runs = api.runs(path)

# By state
runs = api.runs(path, filters={"state": "finished"})

# By config value
runs = api.runs(path, filters={"config.model": "gpt-4"})
runs = api.runs(path, filters={"config.learning_rate": {"$lt": 0.01}})

# By summary metric
runs = api.runs(path, filters={"summary_metrics.accuracy": {"$gt": 0.9}})
runs = api.runs(path, filters={"summary_metrics.loss": {"$lt": 0.5}})

# By display name (regex)
runs = api.runs(path, filters={"display_name": {"$regex": ".*v2.*"}})

# By tags
runs = api.runs(path, filters={"tags": {"$in": ["production"]}})
runs = api.runs(path, filters={"tags": {"$nin": ["deprecated"]}})

# Date range
runs = api.runs(path, filters={
    "$and": [
        {"created_at": {"$gt": "2025-01-01T00:00:00"}},
        {"created_at": {"$lt": "2025-06-01T00:00:00"}},
    ]
})

# Config key exists
runs = api.runs(path, filters={"config.special_param": {"$exists": True}})

# Compound filters
runs = api.runs(path, filters={
    "$and": [
        {"config.model": "transformer"},
        {"summary_metrics.loss": {"$lt": 0.5}},
        {"state": "finished"},
    ]
})
```

### Filter operators

`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$regex`, `$and`, `$or`, `$nor`

### Filterable fields

`state`, `display_name`, `config.KEY`, `summary_metrics.KEY`, `tags`, `created_at`, `group`, `job_type`

### Sorting

```python
runs = api.runs(path, order="-created_at")              # newest first
runs = api.runs(path, order="+summary_metrics.loss")     # lowest loss first
runs = api.runs(path, order="-summary_metrics.val_acc")  # highest accuracy first
```

### Pagination

```python
# IMPORTANT: Use per_page to control page size.
# Default is 50 — on projects with 20K metrics per run, each page is huge.
# Use per_page=min(N, 1000) to minimize round-trips.
runs = api.runs(path, per_page=100)
first_50 = runs[:50]    # slice to limit (lazy)

# AVOID on large projects — triggers expensive count query (5s+):
# total = len(runs)
# Instead, just slice and iterate.
```

**IMPORTANT**: Always slice runs — never `list()` all runs on large projects.

### Run properties

```python
run = api.run(f"{path}/run-id")

# Identity
run.id           # str: 8-char hash
run.name         # str: display name
run.entity       # str
run.project      # str
run.url          # str: full URL
run.path         # list: [entity, project, run_id]

# State & metadata
run.state        # "finished", "failed", "crashed", "running"
run.created_at   # str: ISO timestamp
run.tags         # list[str]
run.lastHistoryStep  # int: last step number (-1 if no history)

# Config (hyperparameters)
run.config       # dict — logged at wandb.init()
lr = run.config.get("learning_rate")
# On large projects, DON'T iterate all config keys unless needed:
# clean_config = {k: v for k, v in run.config.items() if not k.startswith("_")}
# Instead, access specific keys:
lr = run.config.get("learning_rate")
model = run.config.get("model")

# Summary (final metric values)
run.summary_metrics  # dict (read-only snapshot — prefer for reads)
final_loss = run.summary_metrics.get("loss")
```

### Runs to DataFrame (large project safe)

```python
# IMPORTANT: Use per_page and only access specific metric keys.
# On runs with 20K metrics, iterating all config/summary keys is slow.
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)
rows = []
for run in runs[:100]:  # ALWAYS slice
    rows.append({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        # Only specific config keys — NOT all config:
        "config.lr": run.config.get("learning_rate"),
        "config.model": run.config.get("model"),
        # Only specific metrics — NOT all summary_metrics:
        "loss": run.summary_metrics.get("loss"),
        "accuracy": run.summary_metrics.get("accuracy"),
    })
df = pd.DataFrame(rows)
print(df.describe())
```

---

## Metrics History

Three methods with different performance characteristics:

### `run.history()` — sampled, fast, DataFrame

```python
# ALWAYS pass keys= on large projects. Without keys, runs with 1K+ metrics will 502.
df = run.history(samples=500, keys=["loss", "val_loss"])

# Specific metrics with more samples
df = run.history(samples=2000, keys=["loss", "val_loss", "learning_rate"])
```

**Behavior**: Server-side downsamples to `samples` points using min/max bucketing. Fast, but misses data points. Good for overviews and plotting.

**WARNING**: `run.history()` without `keys=` fetches ALL metrics. On runs with 1K+ metrics, this will **502 or timeout**.

### `run.scan_history()` — full, unsampled, iterator

```python
# ALWAYS pass keys= — same 502 risk as history() without keys
losses = [row["loss"] for row in run.scan_history(keys=["loss"])]

# Step range
for row in run.scan_history(keys=["loss"], min_step=1000, max_step=2000):
    print(row["_step"], row.get("loss"))

# To DataFrame
rows = list(run.scan_history(keys=["loss", "val_loss"], page_size=2000))
df = pd.DataFrame(rows)
```

**Behavior**: Returns ALL logged rows, unsampled. Uses GraphQL pagination. Good for precision on runs with <10K steps.

### `run.beta_scan_history()` — parquet-backed, fast for large histories

```python
# Downloads history from parquet files instead of GraphQL pagination.
# Significantly faster for runs with 10K+ steps.
# ALWAYS pass keys= to avoid downloading all columns.

# Basic usage
for row in run.beta_scan_history(keys=["loss"]):
    print(row)

# With step range and page size
rows = list(run.beta_scan_history(
    keys=["loss", "val_loss"],
    min_step=0,
    max_step=10000,
    page_size=10_000,
))
df = pd.DataFrame(rows)

# With caching (default True — skips re-download)
rows = list(run.beta_scan_history(keys=["loss"], use_cache=True))
```

**Behavior**: Downloads parquet history files, then reads locally. First call downloads the file; subsequent calls with `use_cache=True` read from local cache. Faster than `scan_history()` for large runs (10K+ steps), but has download overhead for small runs.

**WARNING**: `beta_scan_history()` without `keys=` downloads ALL metric columns in the parquet file. On runs with 20K metrics, this can take **300+ seconds**.

### When to use which

| Use case | Method | Why |
|----------|--------|-----|
| Quick plot / overview | `history(samples=500, keys=[...])` | Fast, sampled |
| Dashboard summary | `history(samples=1000, keys=[...])` | Fast, sampled |
| Exact values, <10K steps | `scan_history(keys=[...])` | Low overhead |
| Exact values, 10K+ steps | `beta_scan_history(keys=[...])` | Parquet is faster |
| Repeated reads of same run | `beta_scan_history(keys=[...], use_cache=True)` | Cached locally |
| System metrics (GPU/CPU) | `history(stream="system")` | Separate stream |
| Step range query | `scan_history(keys=[...], min_step=N, max_step=M)` | Built-in range |

### Bulk history — multiple runs

```python
# Works on small projects, but can timeout on large projects with many metrics.
# ALWAYS pass keys= to avoid fetching all columns.
runs = api.runs(path, filters={"state": "finished"})
df = runs.histories(samples=200, keys=["loss", "val_loss"], format="pandas")
# df has columns: run_id, _step, loss, val_loss
```

---

## Data Analysis Patterns

**Rule**: Always use pandas and numpy. Never print raw data into context.

### Discovering available metrics (do this first on unfamiliar projects)

```python
run = api.run(f"{path}/run-id")

# Check what metrics exist
all_keys = sorted(run.summary_metrics.keys())
metric_keys = [k for k in all_keys if not k.startswith("_")]
print(f"Total metric keys: {len(metric_keys)}")
print(f"First 30: {metric_keys[:30]}")

# Check if run has step history
print(f"Last history step: {run.lastHistoryStep}")
# -1 means no step history (all data is in summary only)
```

### Finding the minimum loss

```python
from wandb_helpers import scan_history

rows = scan_history(run, keys=["loss"])
df = pd.DataFrame(rows)
min_idx = df["loss"].idxmin()
print(f"Min loss: {df.loc[min_idx, 'loss']:.6f} at index {min_idx}")
```

### Rolling statistics for smoothed analysis

```python
from wandb_helpers import scan_history

rows = scan_history(run, keys=["loss"])
df = pd.DataFrame(rows)
loss = df["loss"].dropna()

window = 50
df["rolling_mean"] = loss.rolling(window, center=True).mean()
df["rolling_std"] = loss.rolling(window, center=True).std()

final_window = loss.tail(len(loss) // 10)
print(f"Final 10% — mean: {final_window.mean():.6f}, std: {final_window.std():.6f}")
```

### Spike and anomaly detection

```python
loss = df["loss"].dropna()
window = 50
rolling_mean = loss.rolling(window, center=True).mean()
rolling_std = loss.rolling(window, center=True).std()

# Spikes: points > 3 sigma from rolling mean
threshold = 3.0
spikes = loss[(loss - rolling_mean).abs() > threshold * rolling_std]
print(f"Spikes (>{threshold} sigma): {len(spikes)}")

# Divergence: sustained increase
diff = loss.diff().rolling(window).mean()
diverging = diff[diff > 0]
if len(diverging) > window:
    print(f"Possible divergence starting at step {diverging.index[0]}")

# Plateau: loss barely changing
change_rate = loss.diff().abs().rolling(window).mean()
plateaus = change_rate[change_rate < 1e-5]
if len(plateaus) > window:
    print(f"Plateau at step {plateaus.index[0]}, value ~{loss.iloc[plateaus.index[0]]:.6f}")

# NaN/Inf
nan_steps = df[df["loss"].isna()]["_step"].tolist() if "_step" in df else []
if nan_steps:
    print(f"NaN loss at {len(nan_steps)} steps")
```

### Overfitting detection

```python
from wandb_helpers import scan_history

rows = scan_history(run, keys=["loss", "val_loss"])
df = pd.DataFrame(rows)
train = df["loss"].dropna()
val = df["val_loss"].dropna()

if len(val) > 10:
    tail_pct = len(val) // 5
    train_tail = train.tail(tail_pct).mean()
    val_tail = val.tail(tail_pct).mean()
    gap = val_tail - train_tail
    overfit = val_tail > train_tail * 1.2
    print(f"Train/val gap: {gap:.4f}, overfit: {overfit}")
```

### Sweep analysis — finding best hyperparameters

```python
# Use server-side sort to get best runs first
runs = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss", per_page=100)
rows = []
for run in runs[:100]:
    rows.append({
        "name": run.name,
        # Only specific config keys — don't iterate all config
        "lr": run.config.get("learning_rate"),
        "model": run.config.get("model"),
        "batch_size": run.config.get("batch_size"),
        "loss": run.summary_metrics.get("loss"),
        "val_loss": run.summary_metrics.get("val_loss"),
    })

sweep_df = pd.DataFrame(rows)
print("Best 5 by loss:")
print(sweep_df.nsmallest(5, "loss").to_string(index=False))

# Hyperparameter importance (simple correlation with loss)
numeric = sweep_df.select_dtypes(include=[np.number])
if "loss" in numeric.columns:
    corr = numeric.corr()["loss"].drop("loss", errors="ignore").sort_values()
    print("\nCorrelation with loss:")
    print(corr.to_string())
```

### Side-by-side run comparison

```python
run_a = api.run(f"{path}/run-a")
run_b = api.run(f"{path}/run-b")

# Config diff — use compare_configs helper for selective comparison
from wandb_helpers import compare_configs
diffs = compare_configs(run_a, run_b, keys=["learning_rate", "model", "batch_size"])
print(pd.DataFrame(diffs).to_string(index=False))

# Metric comparison
for key in ["loss", "accuracy", "val_loss"]:
    a = run_a.summary_metrics.get(key, "N/A")
    b = run_b.summary_metrics.get(key, "N/A")
    print(f"  {key}: {a} vs {b}")
```

### Cross-run metric search (server-side filters)

On large projects, **always use server-side filters** instead of iterating all runs:

```python
# Find runs where a metric exceeds a threshold
high_acc = api.runs(path, filters={
    "summary_metrics.accuracy": {"$gt": 0.95}
}, order="-summary_metrics.accuracy", per_page=50)

# Find runs where a metric is negative
negative_reward = api.runs(path, filters={
    "summary_metrics.reward": {"$lt": 0}
}, per_page=50)

# Combine metric + config filters
specific = api.runs(path, filters={
    "$and": [
        {"config.model": "transformer"},
        {"summary_metrics.loss": {"$lt": 0.1}},
        {"state": "finished"},
    ]
}, per_page=50)
```

---

## Artifacts

```python
# Fetch by name + version/alias
artifact = api.artifact(f"{path}/model-weights:latest")
artifact = api.artifact(f"{path}/model-weights:v3")

# Properties
artifact.name         # str
artifact.type         # str: "model", "dataset"
artifact.version      # str: "v0", "v1"
artifact.aliases      # list[str]: ["latest", "best"]
artifact.metadata     # dict
artifact.size         # int: bytes
artifact.created_at   # str

# Download
local_path = artifact.download()
artifact.download(root="/tmp/data")
artifact.download(path_prefix="train/")

# Lineage
producer = artifact.logged_by()
consumers = artifact.used_by()

# From a run
for art in run.logged_artifacts():
    print(art.name, art.type, art.aliases)

# Check existence
exists = api.artifact_exists(f"{path}/my-model:v0")
```

---

## System Metrics

```python
# GPU, CPU, memory, disk
sys_df = run.history(stream="system")
# Columns: system.cpu, system.memory, system.disk,
#           system.gpu.0.gpu, system.gpu.0.memory, system.gpu.0.temp

# GPU utilization analysis
if "system.gpu.0.gpu" in sys_df.columns:
    gpu = sys_df["system.gpu.0.gpu"].dropna()
    print(f"GPU util: mean={gpu.mean():.1f}%, min={gpu.min():.1f}%, max={gpu.max():.1f}%")
    low = gpu[gpu < 50]
    print(f"Low utilization (<50%): {len(low)}/{len(gpu)} samples")

# Memory
if "system.gpu.0.memory" in sys_df.columns:
    mem = sys_df["system.gpu.0.memory"].dropna()
    print(f"GPU memory: mean={mem.mean():.1f}%, max={mem.max():.1f}%")

# Latest snapshot
latest = run.system_metrics  # dict
```

---

## Projects

```python
projects = list(api.projects(entity, per_page=200))
for p in projects[:20]:
    print(p.name, getattr(p, "created_at", None))
```

---

## Common Gotchas

| Gotcha | Wrong | Right |
|--------|-------|-------|
| API timeout | `wandb.Api()` (19s default) | `wandb.Api(timeout=60)` |
| Summary access | `run.summary["loss"]` | `run.summary_metrics.get("loss")` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (always slice) |
| Counting runs | `len(runs)` on large project (5s+) | Just `runs[:N]` |
| Pagination | `api.runs(path)` (per_page=50 default) | `api.runs(path, per_page=min(N, 1000))` |
| History — all fields | `run.history()` → **502** on 1K+ metrics | `run.history(samples=500, keys=["loss"])` |
| scan_history — no keys | `scan_history()` → timeout | `scan_history(keys=["loss"])` (explicit) |
| Large history (10K+ steps) | `scan_history(keys=[...])` (slow GraphQL) | `beta_scan_history(keys=[...])` (parquet) |
| beta_scan — no keys | `beta_scan_history()` (300s+) | `beta_scan_history(keys=["loss"])` |
| Config iteration | `for k,v in run.config.items()` (slow) | `run.config.get("lr")` (specific keys) |
| Raw data in context | `print(run.history())` | Load into DataFrame, compute stats |
| Metric at step N | iterate history | `scan_history(keys=["loss"], min_step=N, max_step=N+1)` |
| Cache staleness | reading live run | `api.flush()` first |
| Cross-run search | iterate all runs client-side | Server-side: `{"summary_metrics.X": {"$gt": Y}}` |

---

## Quick Reference

```python
from wandb_helpers import get_api, scan_history

api = get_api()  # timeout=60
path = f"{entity}/{project}"

# Find best run by loss (server-side sort — no client iteration)
best = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss", per_page=1)[:1]
print(f"Best: {best[0].name}, loss={best[0].summary_metrics.get('loss')}")

# Get a run's full loss curve into numpy (explicit keys)
rows = scan_history(run, keys=["loss"])
losses = np.array([r["loss"] for r in rows])
print(f"Loss: min={losses.min():.6f}, final={losses[-1]:.6f}, steps={len(losses)}")

# Compare N runs on one metric (per_page optimized)
runs = api.runs(path, filters={"state": "finished"}, per_page=20)[:20]
data = [(r.name, r.summary_metrics.get("loss", float("inf"))) for r in runs]
df = pd.DataFrame(data, columns=["run", "loss"]).sort_values("loss")
print(df.to_string(index=False))

# Download best model artifact
best_run = api.runs(path, order="+summary_metrics.loss", per_page=1)[:1][0]
for art in best_run.logged_artifacts():
    if art.type == "model":
        art.download(root="/tmp/best_model")
        print(f"Downloaded {art.name}:{art.version}")
```
