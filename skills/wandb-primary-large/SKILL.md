---
name: wandb-primary-large
description: Comprehensive primary skill for agents working with Weights & Biases, optimized for large projects (10K+ runs, 1K+ metrics per run). Covers both the W&B SDK (training runs, metrics, artifacts, sweeps, reports) and the Weave SDK (GenAI traces, evaluations, scorers). Uses beta_scan_history (parquet), server-side filters, selective key fetching, and timeout tuning to avoid the 502s and timeouts that plague large projects.
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B Primary Skill (Large Project Optimized)

## CRITICAL: Large project performance rules

These rules prevent 502 errors, timeouts, and multi-minute hangs on projects with 10K+ runs or runs with 1K+ metrics. **Violating any of these will cause failures on large projects.**

1. **Always use `wandb.Api(timeout=60)`** — the default 19s timeout causes constant failures
2. **NEVER call `history()` or `scan_history()` without explicit `keys=[...]`** — runs with 1K+ metrics will 502 or timeout when fetching all columns
3. **Use `per_page=min(limit, 1000)`** when calling `api.runs()` — reduces pagination round-trips
4. **Prefer server-side filters** (`summary_metrics.X: {$gt: Y}`) over client-side iteration
5. **Avoid `len(runs)`** on large projects — it triggers an expensive count query (5s+). Use `runs[:N]` directly
6. **Use `beta_scan_history`** for runs with 10K+ history steps — reads from parquet, not GraphQL
7. **Never iterate all config keys** unless explicitly needed — use `config_keys=["lr", "model"]`

---

## Python environment detection (DO THIS FIRST)

Before running any Python code, you MUST determine how the user runs Python in this project. Check for clues in this order:

1. Look for `pyproject.toml`, `uv.lock`, `poetry.lock`, `Pipfile`, `requirements.txt`, `setup.py`, `setup.cfg`, `Makefile`, or `docker-compose.yml` in the project root
2. Check if there is an activated virtual environment (`$VIRTUAL_ENV`, `.venv/`, `venv/`)
3. Check for a `Dockerfile` or CI config that reveals the runtime
4. If the user has explicitly told you how they run Python, use that

Once you know the environment, **write your finding into this skill** by replacing the placeholder below so that all subsequent code blocks use the correct commands. If you cannot determine the environment from any of the above, default to `uv`.

<!-- AGENT: Replace the content between the ENVIRONMENT markers with the detected environment -->
<!-- ENVIRONMENT_START -->
**Detected Python environment:** _not yet detected_

```
# Run command: <not yet detected>
# Install command: <not yet detected>
```
<!-- ENVIRONMENT_END -->

**Examples of what to write here:**

| Environment | Run command | Install command |
|---|---|---|
| uv | `uv run script.py` | `uv pip install pandas` |
| poetry | `poetry run python script.py` | `poetry add pandas` |
| conda | `conda run python script.py` | `conda install pandas` |
| bare venv | `python script.py` (with venv activated) | `pip install pandas` |
| docker | `docker exec <ctr> python script.py` | `docker exec <ctr> pip install pandas` |

**If you cannot determine the environment, write this:**

```
# Run command: uv run script.py        # always use uv run, never bare python
# Install command: uv pip install <pkg>
```

---

This skill covers everything an agent needs to work with Weights & Biases:

- **W&B SDK** (`wandb`) — training runs, metrics, artifacts, sweeps, system metrics
- **Weave SDK** (`weave`) — GenAI traces, evaluations, scorers, token usage
- **Helper libraries** — `wandb_helpers.py` and `weave_helpers.py` for common operations

## When to use what

| I need to... | Use |
|---|---|
| Query training runs, loss curves, hyperparameters | **W&B SDK** (`wandb.Api()`) — see `references/WANDB_SDK.md` |
| Query GenAI traces, calls, evaluations | **Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |
| Convert Weave wrapper types to plain Python | **`weave_helpers.unwrap()`** |
| Build a DataFrame from training runs | **`wandb_helpers.runs_to_dataframe()`** |
| Extract eval results for analysis | **`weave_helpers.eval_results_to_dicts()`** |
| Need low-level Weave filtering (CallsFilter, Query) | **Raw Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |

---

## Bundled files

### Helper libraries

```python
import sys
sys.path.insert(0, "skills/wandb-primary-large/scripts")

# Weave helpers (traces, evals, GenAI)
from weave_helpers import (
    unwrap,                  # Recursively convert Weave types -> plain Python
    get_token_usage,         # Extract token counts from a call's summary
    eval_results_to_dicts,   # predict_and_score calls -> list of result dicts
    pivot_solve_rate,        # Build task-level pivot table across agents
    results_summary,         # Print compact eval summary
    eval_health,             # Extract status/counts from Evaluation.evaluate calls
    eval_efficiency,         # Compute tokens-per-success across eval calls
)

# W&B helpers (training runs, metrics) — large-project optimized
from wandb_helpers import (
    get_api,             # Create API with safe timeout (default 60s)
    probe_project,       # Discover project scale, metrics, config BEFORE querying
    fetch_runs,          # FAST: Direct GraphQL with selective metrics (17x faster)
    runs_to_dataframe,   # Legacy: iterate run objects (slower, use fetch_runs instead)
    diagnose_run,        # Quick diagnostic summary (configurable metric keys)
    compare_configs,     # Side-by-side config diff between two runs
    scan_history,        # Smart history scan (auto-selects beta_scan_history for large runs)
)
```

### Reference docs

Read these as needed — they contain full API surfaces and recipes:

- **`references/WEAVE_SDK.md`** — Weave SDK for GenAI traces (`client.get_calls()`, `CallsFilter`, `Query`, stats). Start here for Weave queries.
- **`references/WANDB_SDK.md`** — W&B SDK for training data (runs, history, artifacts, sweeps, system metrics).

---

## Critical rules

### Treat traces and runs as DATA

Weave traces and W&B run histories can be enormous. Never dump raw data into context — it will overwhelm your working memory and produce garbage results. Always:

1. **Inspect structure first** — look at column names, dtypes, row counts
2. **Load into pandas/numpy** — compute stats programmatically
3. **Summarize, don't dump** — print computed statistics and tables, not raw rows

```python
import pandas as pd
import numpy as np
from wandb_helpers import get_api, scan_history

api = get_api()  # timeout=60 for large projects
run = api.run(f"{path}/run-id")

# BAD: prints thousands of rows into context
for row in run.scan_history(keys=["loss"]):
    print(row)

# BAD: no keys — will 502 on runs with 1K+ metrics
run.history()

# GOOD: use scan_history helper with explicit keys + max_rows guard
rows = scan_history(run, keys=["loss"], max_rows=50_000)
losses = np.array([r["loss"] for r in rows])
print(f"Loss: {len(losses)} steps, min={losses.min():.4f}, "
      f"final={losses[-1]:.4f}, mean_last_10%={losses[-len(losses)//10:].mean():.4f}")
```

### Always deliver a final answer

Do not end your work mid-analysis. Every task must conclude with a clear, structured response:

1. Query the data (1-2 scripts max)
2. Extract the numbers you need
3. Present: table + key findings + direct answers to each sub-question

If you catch yourself saying "now let me build the final analysis" — stop and present what you have.

### Use `unwrap()` for unknown Weave data

When you encounter Weave output and aren't sure of its type (WeaveDict? WeaveObject? ObjectRef?), unwrap it first:

```python
from weave_helpers import unwrap
import json

output = unwrap(call.output)
print(json.dumps(output, indent=2, default=str))
```

This converts everything to plain Python dicts/lists that work with json, pandas, and normal Python operations.

---

## Environment setup

The sandbox has `wandb`, `weave`, `pandas`, and `numpy` pre-installed.

```python
import os
entity  = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
```

### Installing extra packages and running scripts

Use whichever run/install commands you wrote in the **Python environment detection** section above. If you haven't detected the environment yet, go back and do that first.

---

## Quick starts

### Step 0: Probe the project (DO THIS FIRST on unfamiliar projects)

```python
from wandb_helpers import get_api, probe_project

api = get_api()  # timeout=60
path = f"{entity}/{project}"

info = probe_project(api, path)
print(f"Metrics per run: {info['sample_metric_count']}")
print(f"Has step history: {info['has_step_history']}")
print(f"Recommended per_page: {info['recommended_per_page']}")
print(f"Sample metrics: {info['sample_metric_keys'][:10]}")
if info['warnings']:
    print(f"WARNINGS: {info['warnings']}")
```

This tells you:
- How many metrics exist (determines if you MUST pass `keys=`)
- Whether runs have step history (determines scan_history vs summary-only analysis)
- What `per_page` to use (high metric count = smaller pages)
- What metric keys are available (so you don't guess wrong names)

### W&B SDK — training runs (large project safe)

```python
import pandas as pd
from wandb_helpers import get_api, fetch_runs

api = get_api()  # timeout=60
path = f"{entity}/{project}"

# fetch_runs uses GraphQL field selection — only fetches the metrics you ask for.
# On projects with 20K+ metrics per run, this is 17x faster than the standard SDK
# (fetches ~50 bytes vs 771KB per run summary).
rows = fetch_runs(
    api, path,
    metric_keys=["loss", "acc"],
    filters={"state": "finished"},
    limit=100,
)
df = pd.DataFrame(rows)
print(df.describe())
```

### W&B SDK — find best runs (server-side)

```python
# Let the server sort — don't fetch all runs and sort client-side
api = get_api()
best = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss", per_page=10)[:10]
for run in best:
    print(f"  {run.name}: loss={run.summary_metrics.get('loss')}")
```

### W&B SDK — filter by metric threshold (server-side)

```python
# Server-side filter: find runs where acc > 0.9
api = get_api()
good_runs = api.runs(path, filters={
    "$and": [
        {"state": "finished"},
        {"summary_metrics.acc": {"$gt": 0.9}},
    ]
}, order="-summary_metrics.acc", per_page=50)
for run in good_runs[:20]:
    print(f"  {run.name}: acc={run.summary_metrics.get('acc')}")
```

### W&B SDK — history analysis (single run)

```python
from wandb_helpers import get_api, scan_history
import numpy as np

api = get_api()
run = api.run(f"{path}/run-id")

# ALWAYS use explicit keys — never call without keys on large projects
# scan_history auto-selects beta_scan_history (parquet) for 10K+ step runs
rows = scan_history(run, keys=["loss", "val_loss"])
losses = np.array([r.get("loss") for r in rows if r.get("loss") is not None])
print(f"Loss: {len(losses)} steps, min={losses.min():.6f}, final={losses[-1]:.6f}")

# For sampled overview (faster, less precise)
df = run.history(samples=500, keys=["loss", "val_loss"])
print(df.describe())
```

### Weave — SDK

```python
import weave
client = weave.init(f"{entity}/{project}")  # positional string, NOT keyword arg
calls = client.get_calls(limit=10)
```

For raw SDK patterns (CallsFilter, Query, advanced filtering), read `references/WEAVE_SDK.md`.

---

## Key patterns

### Weave eval inspection

Evaluation calls follow this hierarchy:

```
Evaluation.evaluate (root)
  ├── Evaluation.predict_and_score (one per dataset row x trials)
  │     ├── model.predict (the actual model call)
  │     ├── scorer_1.score
  │     └── scorer_2.score
  └── Evaluation.summarize
```

Extract per-task results into a DataFrame:

```python
from weave_helpers import eval_results_to_dicts, results_summary

# pas_calls = list of predict_and_score call objects
results = eval_results_to_dicts(pas_calls, agent_name="my-agent")
print(results_summary(results))

df = pd.DataFrame(results)
print(df.groupby("passed")["score"].mean())
```

### Eval health and efficiency

```python
from weave_helpers import eval_health, eval_efficiency

health = eval_health(eval_calls)
df = pd.DataFrame(health)
print(df.to_string(index=False))

efficiency = eval_efficiency(eval_calls)
print(pd.DataFrame(efficiency).to_string(index=False))
```

### Token usage

```python
from weave_helpers import get_token_usage

usage = get_token_usage(call)
print(f"Tokens: {usage['total_tokens']} (in={usage['input_tokens']}, out={usage['output_tokens']})")
```

### Cost estimation

```python
call_with_costs = client.get_call("id", include_costs=True)
costs = call_with_costs.summary.get("weave", {}).get("costs", {})
```

### Run diagnostics (large project safe)

```python
from wandb_helpers import get_api, diagnose_run

api = get_api()
run = api.run(f"{path}/run-id")

# Discover available metrics first
metric_keys = [k for k in run.summary_metrics.keys() if not k.startswith("_")]
print(f"Available metrics ({len(metric_keys)}): {metric_keys[:20]}")

# Use the actual metric key — don't assume "loss" exists
diag = diagnose_run(run, train_key="loss", val_key="val_loss")
for k, v in diag.items():
    print(f"  {k}: {v}")
```

### Cross-run metric search (server-side)

On large projects, **never iterate all runs client-side to find metric values**. Use server-side filters:

```python
api = get_api()

# Find runs where a specific metric exceeds a threshold
runs = api.runs(path, filters={
    "summary_metrics.train1": {"$gt": 10}
}, per_page=50)
for run in runs[:50]:
    print(f"  {run.name}: train1={run.summary_metrics.get('train1')}")

# Find runs where a metric went negative (check summary)
runs = api.runs(path, filters={
    "summary_metrics.reward": {"$lt": 0}
}, per_page=50)
```

### History analysis across multiple runs

```python
from wandb_helpers import get_api, scan_history
import pandas as pd

api = get_api()
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=10)

# Analyze history for a few runs — don't try to load history for 100+ runs
all_data = []
for run in runs[:5]:
    rows = scan_history(run, keys=["loss"], max_rows=10_000)
    for r in rows:
        r["run_name"] = run.name
    all_data.extend(rows)

df = pd.DataFrame(all_data)
print(df.groupby("run_name")["loss"].describe())
```

### Error analysis — open coding to axial coding

For structured failure analysis on eval results:

1. **Understand data shape** — use `project.summary()`, `calls.input_shape()`, `calls.output_shape()`
2. **Open coding** — write a Weave Scorer that journals what went wrong per failing call
3. **Axial coding** — write a second Scorer that classifies notes into a taxonomy
4. **Summarize** — count primary labels with `collections.Counter`

See `references/WEAVE_SDK.md` for the full SDK reference.

## Report authoring (W&B Reports)

- Install the reports extra once. Use the environment-default python package manager to install `wandb[workspaces]` or default to:  `uv pip install "wandb[workspaces]"`.
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

---

## Gotchas

### Weave API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| weave.init args | `weave.init(project="x")` | `weave.init("x")` (positional) |
| Parent filter | `filter={'parent_id': 'x'}` | `filter={'parent_ids': ['x']}` (plural, list) |
| WeaveObject access | `rubric.get('passed')` | `getattr(rubric, 'passed', None)` |
| Nested output | `out.get('succeeded')` | `out.get('output').get('succeeded')` (output.output) |
| ObjectRef comparison | `name_ref == "foo"` | `str(name_ref) == "foo"` |
| CallsFilter import | `from weave import CallsFilter` | `from weave.trace.weave_client import CallsFilter` |
| Query import | `from weave import Query` | `from weave.trace_server.interface.query import Query` |
| Eval status path | `summary["status"]` | `summary["weave"]["status"]` |
| Eval success count | `summary["success_count"]` | `summary["weave"]["status_counts"]["success"]` |
| When in doubt | Guess the type | `unwrap()` first, then inspect |

### WeaveDict vs WeaveObject

- **WeaveDict**: dict-like, supports `.get()`, `.keys()`, `[]`. Used for: `call.inputs`, `call.output`, `scores` dict
- **WeaveObject**: attribute-based, use `getattr()`. Used for: scorer results (rubric), dataset rows
- **When in doubt**: use `unwrap()` to convert everything to plain Python

### W&B API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| API timeout | `wandb.Api()` (19s default) | `wandb.Api(timeout=60)` or `get_api()` |
| Summary access | `run.summary["loss"]` | `run.summary_metrics.get("loss")` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (always slice) |
| Counting runs | `len(api.runs(...))` on large project | Skip count, just `runs[:N]` |
| Pagination | `api.runs(path)` (per_page=50 default) | `api.runs(path, per_page=min(N, 1000))` |
| History — all fields | `run.history()` | `run.history(samples=500, keys=["loss"])` |
| History — no keys on large run | `run.history(samples=10)` → **502** | `run.history(samples=10, keys=["loss"])` |
| scan_history — no keys | `scan_history()` → timeout | `scan_history(keys=["loss"])` (explicit) |
| Large history (10K+ steps) | `scan_history(keys=[...])` | `beta_scan_history(keys=[...])` (parquet) |
| Config iteration | `for k,v in run.config.items()` | `run.config.get("lr")` (specific keys) |
| Raw data in context | `print(run.history())` | Load into DataFrame, compute stats |
| Metric at step N | iterate entire history | `scan_history(keys=["loss"], min_step=N, max_step=N+1)` |
| Cache staleness | reading live run | `api.flush()` first |
| Cross-run metric search | iterate all runs client-side | Server-side filter: `{"summary_metrics.X": {"$gt": Y}}` |

### Package management

| Gotcha | Details |
|--------|---------|
| Using the wrong runner | Always use the run/install commands from the **Python environment detection** section — never guess |
| Bare `python` when env unknown | If you haven't detected the environment yet, default to `uv run script.py` (never bare `python`) |

### Weave logging noise

Weave prints version warnings to stderr. Suppress with:

```python
import logging
logging.getLogger("weave").setLevel(logging.ERROR)
```

---

## Quick reference

```python
from wandb_helpers import get_api, runs_to_dataframe, scan_history
import pandas as pd
import numpy as np

api = get_api()  # timeout=60
path = f"{entity}/{project}"

# --- Weave: Init and get calls ---
import weave
client = weave.init(f"{entity}/{project}")
calls = client.get_calls(limit=10)

# --- W&B: Best run by loss (server-side sort) ---
best = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss", per_page=1)[:1]
print(f"Best: {best[0].name}, loss={best[0].summary_metrics.get('loss')}")

# --- W&B: Loss curve to numpy (with explicit keys) ---
rows = scan_history(run, keys=["loss"])
losses = np.array([r["loss"] for r in rows])
print(f"min={losses.min():.6f}, final={losses[-1]:.6f}, steps={len(losses)}")

# --- W&B: Runs to DataFrame (selective) ---
runs = api.runs(path, filters={"state": "finished"}, per_page=100)
df = pd.DataFrame(runs_to_dataframe(runs, limit=100, metric_keys=["loss", "acc"]))

# --- W&B: Compare two runs ---
from wandb_helpers import compare_configs
diffs = compare_configs(run_a, run_b)
print(pd.DataFrame(diffs).to_string(index=False))
```
