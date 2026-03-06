<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B SDK Reference — Runs, Metrics, Artifacts

Reference for querying Weights & Biases training data using the `wandb` Python SDK. Covers runs, metrics history, artifacts, sweeps, and system metrics.

**Key principle**: Always load data into pandas/numpy for analysis. Never dump raw history or run lists into context — it will overwhelm your working memory. Look at structure first, then query specific fields.

## Quick Start

```python
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()

# Entity and project from environment
import os
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# List runs
runs = api.runs(path, filters={"state": "finished"}, order="-created_at")
print(f"Found {len(runs)} finished runs")
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
runs = api.runs(path, per_page=100)
first_50 = runs[:50]    # slice to limit (lazy)
total = len(runs)        # triggers count query
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

# Config (hyperparameters)
run.config       # dict — logged at wandb.init()
lr = run.config.get("learning_rate")
clean_config = {k: v for k, v in run.config.items() if not k.startswith("_")}

# Summary (final metric values)
run.summary_metrics  # dict (read-only snapshot — prefer for reads)
final_loss = run.summary_metrics.get("loss")
```

### Runs to DataFrame

```python
runs = api.runs(path, filters={"state": "finished"}, order="-created_at")
rows = []
for run in runs[:200]:  # ALWAYS slice
    rows.append({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        **{f"config.{k}": v for k, v in run.config.items() if not k.startswith("_")},
        "loss": run.summary_metrics.get("loss"),
        "accuracy": run.summary_metrics.get("accuracy"),
    })
df = pd.DataFrame(rows)
print(df.describe())
```

---

## Metrics History

Two methods with very different behavior:

### `run.history()` — sampled, fast, DataFrame

```python
# Quick 500-point overview (default)
df = run.history()

# Specific metrics with more samples
df = run.history(samples=2000, keys=["loss", "val_loss", "learning_rate"])
```

**Behavior**: Server-side downsamples to `samples` points using min/max bucketing. Fast, but misses data points. Good for overviews and plotting.

### `run.scan_history()` — full, unsampled, iterator

```python
# ALL loss values (no sampling)
losses = [row["loss"] for row in run.scan_history(keys=["loss"])]

# Step range
for row in run.scan_history(keys=["loss"], min_step=1000, max_step=2000):
    print(row["_step"], row.get("loss"))

# To DataFrame
rows = list(run.scan_history(keys=["loss", "val_loss"], page_size=2000))
df = pd.DataFrame(rows)
```

**Behavior**: Returns ALL logged rows, unsampled. Use when you need precision.

### When to use which

| Use case | Method |
|----------|--------|
| Quick plot / overview | `history(samples=500)` |
| Dashboard summary | `history(samples=1000, keys=[...])` |
| Anomaly detection | `scan_history(keys=[...])` |
| Exact loss values | `scan_history(keys=["loss"])` |
| System metrics (GPU/CPU) | `history(stream="system")` |
| Very long runs | `scan_history()` (iterator) |

### Bulk history — multiple runs

```python
runs = api.runs(path, filters={"state": "finished"})
df = runs.histories(samples=200, keys=["loss", "val_loss"], format="pandas")
# df has columns: run_id, _step, loss, val_loss
```

---

## Data Analysis Patterns

**Rule**: Always use pandas and numpy. Never print raw data into context.

### Finding the minimum loss

```python
df = pd.DataFrame(list(run.scan_history(keys=["loss", "_step"])))
min_idx = df["loss"].idxmin()
print(f"Min loss: {df.loc[min_idx, 'loss']:.6f} at step {df.loc[min_idx, '_step']}")
```

### Rolling statistics for smoothed analysis

```python
df = pd.DataFrame(list(run.scan_history(keys=["loss"])))
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
df = pd.DataFrame(list(run.scan_history(keys=["loss", "val_loss"])))
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
runs = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss")
rows = []
for run in runs[:100]:
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    rows.append({
        "name": run.name,
        **config,
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

# Config diff
config_df = pd.DataFrame([run_a.config, run_b.config]).T
config_df.columns = [run_a.name, run_b.name]
diff = config_df[config_df.iloc[:, 0] != config_df.iloc[:, 1]]
print("Config differences:")
print(diff.to_string())

# Metric comparison
for key in ["loss", "accuracy", "val_loss"]:
    a = run_a.summary_metrics.get(key, "N/A")
    b = run_b.summary_metrics.get(key, "N/A")
    print(f"  {key}: {a} vs {b}")
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
| Summary access | `run.summary["loss"]` | `run.summary_metrics.get("loss")` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (always slice) |
| History — all fields | `run.history()` | `run.history(samples=500, keys=["loss"])` |
| History — no keys | `scan_history()` | `scan_history(keys=["loss"])` (explicit) |
| Raw data in context | `print(run.history())` | Load into DataFrame, compute stats, print summary |
| Comparing configs | manual key-by-key | `pd.DataFrame([a.config, b.config]).T` |
| Metric at step N | iterate history | `scan_history(keys=["loss"], min_step=N, max_step=N+1)` |
| Cache staleness | reading live run | `api.flush()` first |

---

## Quick Reference

```python
# Find best run by loss
best = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss")[:1]
print(f"Best: {best[0].name}, loss={best[0].summary_metrics['loss']}")

# Get a run's full loss curve into numpy
losses = np.array([r["loss"] for r in run.scan_history(keys=["loss"])])
print(f"Loss: min={losses.min():.6f}, final={losses[-1]:.6f}, steps={len(losses)}")

# Compare N runs on one metric
runs = api.runs(path, filters={"state": "finished"})[:20]
data = [(r.name, r.summary_metrics.get("loss", float("inf"))) for r in runs]
df = pd.DataFrame(data, columns=["run", "loss"]).sort_values("loss")
print(df.to_string(index=False))

# Download best model artifact
best_run = api.runs(path, order="+summary_metrics.loss")[:1][0]
for art in best_run.logged_artifacts():
    if art.type == "model":
        art.download(root="/tmp/best_model")
        print(f"Downloaded {art.name}:{art.version}")
```
