---
name: wandb-primary
description: The definitive skill for working with Weights & Biases (W&B / wandb). Use this skill whenever the user mentions W&B, wandb, Weights and Biases, Weave, training runs, experiment tracking, loss curves, metrics, hyperparameters, sweeps, artifacts, model registry, evaluation, traces, LLM observability, GenAI monitoring, scorers, or any W&B data — even if they don't say "W&B" explicitly. Covers everything across both products. W&B SDK (wandb Python package) for model training — logging runs, querying run history, comparing experiments, analyzing loss/accuracy/metrics, sweep analysis, artifact management, system metrics (GPU/CPU), and report authoring. Weave SDK for GenAI/LLM applications — tracing calls, evaluating models, scoring outputs, token usage, cost estimation, and failure analysis. Includes bundled helper libraries (wandb_helpers.py, weave_helpers.py) with optimized functions like fetch_runs (25x faster than raw SDK on large projects via GraphQL field selection), probe_project (auto-discovers project scale and available metrics), scan_history (smart parquet-backed history reading), diagnose_run (configurable training diagnostics), and eval_results_to_dicts (structured evaluation extraction). Handles projects of any size — from 10 runs to 100K+ runs with thousands of metrics per run. Begins with a brief interactive interview to configure the skill for the user's specific environment, metrics, and analysis goals before any work begins.
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B Primary Skill

## Configuration

> **AGENT**: If `interview_completed` is `false`, you MUST read `references/INTERVIEW.md` and complete the interview before doing any other work. Do not guess — ask or detect.

<!-- CONFIGURATION_START -->
| Key | Value |
|-----|-------|
| interview_completed | true |
| python_run | uv run python |
| python_install | uv add |
| llm_provider | openai |
| llm_model | gpt-5.4-mini |
| llm_reasoning | high |
| llm_endpoint | responses |
<!-- CONFIGURATION_END -->

---

## CRITICAL: Large project performance rules

These rules prevent 502 errors, timeouts, and multi-minute hangs on projects with 10K+ runs or runs with 1K+ metrics. **Violating any of these will cause failures on large projects.**

1. **Always use `wandb.Api(timeout=60)`** — the default 19s timeout causes constant failures
2. **NEVER call `history()` or `scan_history()` without explicit `keys=[...]`** — runs with 1K+ metrics will 502 or timeout when fetching all columns
3. **Use `per_page=min(limit, 1000)`** when calling `api.runs()` — reduces pagination round-trips
4. **Prefer server-side filters** (`summary_metrics.X: {$gt: Y}`) over client-side iteration
5. **Avoid `len(runs)`** on large projects — it triggers an expensive count query (5s+). Use `runs[:N]` directly
6. **Use `beta_scan_history`** for runs with 10K+ history steps — reads from parquet, not GraphQL
7. **Never iterate all config keys** unless explicitly needed — access specific keys by name
8. **Discover metric keys per-project** via `probe_project()` — never hardcode `"loss"` or `"accuracy"`

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
| Build a DataFrame from training runs | **`wandb_helpers.fetch_runs()`** (fast) or **`wandb_helpers.runs_to_dataframe()`** |
| Extract eval results for analysis | **`weave_helpers.eval_results_to_dicts()`** |
| Need low-level Weave filtering (CallsFilter, Query) | **Raw Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |

---

## Bundled files

### Helper libraries

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")

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
- **`references/DOCS_INDEX.md`** — Full index of all W&B documentation pages with descriptions. Use this for docs lookups (see below).

### W&B Documentation lookup

When the user asks a docs question (e.g., "how do I log a metric", "what's the EvalLogger API", "how do sweeps work"), use this workflow:

1. **Read `references/DOCS_INDEX.md`** — find the most relevant page(s) by scanning titles and descriptions
2. **Fetch the page as markdown** — any docs URL can be fetched as markdown by appending `.md`:
   - `https://docs.wandb.ai/models/track/log` → `https://docs.wandb.ai/models/track/log.md`
   - `https://docs.wandb.ai/weave/guides/tracking/ops` → `https://docs.wandb.ai/weave/guides/tracking/ops.md`
3. **Answer from the fetched content** — cite the source URL

```bash
# Example: fetch a docs page as markdown
curl -sL "https://docs.wandb.ai/models/track/log.md"
```

This is faster and more accurate than guessing from memory. Always check the docs index first for docs questions.

---

## Critical rules

### Discover metric keys per-project, use Configuration for everything else

Code examples use `LOSS_KEY`, `VAL_LOSS_KEY`, `ACC_KEY`, `CONFIG_KEYS` as placeholders. These are **not** in the Configuration table — they vary by project. Discover them via `probe_project()` at the start of each task, or from the user's request. For Python env and LLM settings, use the Configuration table.

```python
# WRONG — hardcoded metric name
rows = fetch_runs(api, path, metric_keys=["loss", "accuracy"])

# RIGHT — discovered via probe_project or user's request
rows = fetch_runs(api, path, metric_keys=["train/loss", "train/acc"])
```

### Treat traces and runs as DATA

Weave traces and W&B run histories can be enormous. Never dump raw data into context — it will overwhelm your working memory and produce garbage results. Always:

1. **Inspect structure first** — look at column names, dtypes, row counts
2. **Load into pandas/numpy** — compute stats programmatically
3. **Summarize, don't dump** — print computed statistics and tables, not raw rows

```python
from wandb_helpers import get_api, scan_history

api = get_api()  # timeout=60 for large projects
run = api.run(f"{path}/run-id")

# GOOD: use configured metric key with explicit keys + max_rows guard
rows = scan_history(run, keys=["LOSS_KEY"], max_rows=50_000)
losses = np.array([r["LOSS_KEY"] for r in rows])
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

---

## Environment setup

> **AGENT**: Use the `python_run` and `python_install` commands from Configuration. If they are `_not set_`, complete the interview first.

Entity and project come from environment variables or the user's request — do not hardcode them:

```python
import os
entity  = os.environ.get("WANDB_ENTITY", "<from user's request>")
project = os.environ.get("WANDB_PROJECT", "<from user's request>")
path = f"{entity}/{project}"
```

### Installing extra packages

```bash
# Use the configured install command:
<python_install> pandas numpy
```

### Running scripts

```bash
# Use the configured run command:
<python_run> script.py
```

---

## Quick starts

### Step 0: Probe the project (DO THIS FIRST on unfamiliar projects)

```python
from wandb_helpers import get_api, probe_project

api = get_api()  # timeout=60
path = "{entity}/{project}"  # ← from Configuration

info = probe_project(api, path)
print(f"Metrics per run: {info['sample_metric_count']}")
print(f"Has step history: {info['has_step_history']}")
print(f"Recommended per_page: {info['recommended_per_page']}")
print(f"Sample metrics: {info['sample_metric_keys'][:10]}")
if info['warnings']:
    print(f"WARNINGS: {info['warnings']}")
```

### W&B SDK — training runs

```python
import pandas as pd
from wandb_helpers import get_api, fetch_runs

api = get_api()
path = "{entity}/{project}"

# fetch_runs uses GraphQL field selection — 15-25x faster on large projects
rows = fetch_runs(
    api, path,
    metric_keys=["LOSS_KEY", "ACC_KEY"],  # ← from Configuration
    filters={"state": "finished"},
    limit=100,
)
df = pd.DataFrame(rows)
print(df.describe())
```

### W&B SDK — find best runs (server-side)

```python
api = get_api()
best = api.runs(path, filters={"state": "finished"},
                order="+summary_metrics.LOSS_KEY", per_page=10)[:10]
for run in best:
    print(f"  {run.name}: {run.summary_metrics.get('LOSS_KEY')}")
```

### W&B SDK — history analysis (single run)

```python
from wandb_helpers import get_api, scan_history
import numpy as np

api = get_api()
run = api.run(f"{path}/run-id")

# ALWAYS use explicit keys from Configuration
rows = scan_history(run, keys=["LOSS_KEY", "VAL_LOSS_KEY"])
losses = np.array([r.get("LOSS_KEY") for r in rows if r.get("LOSS_KEY") is not None])
print(f"Loss: {len(losses)} steps, min={losses.min():.6f}, final={losses[-1]:.6f}")
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

### Run diagnostics

```python
from wandb_helpers import get_api, diagnose_run

api = get_api()
run = api.run(f"{path}/run-id")

# Use configured metric keys
diag = diagnose_run(run, train_key="LOSS_KEY", val_key="VAL_LOSS_KEY")
for k, v in diag.items():
    print(f"  {k}: {v}")
```

### Cross-run metric search (server-side)

On large projects, **never iterate all runs client-side**. Use server-side filters with configured metric keys:

```python
api = get_api()

# Find runs where primary loss is below threshold
runs = api.runs(path, filters={
    "summary_metrics.LOSS_KEY": {"$lt": 0.5}
}, per_page=50)
for run in runs[:50]:
    print(f"  {run.name}: {run.summary_metrics.get('LOSS_KEY')}")
```

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

results = eval_results_to_dicts(pas_calls, agent_name="my-agent")
print(results_summary(results))

df = pd.DataFrame(results)
print(df.groupby("passed")["score"].mean())
```

### Token usage

> **AGENT**: The token field names vary by LLM provider. `get_token_usage()` handles both OpenAI and Anthropic conventions automatically.

```python
from weave_helpers import get_token_usage

usage = get_token_usage(call)
print(f"Tokens: {usage['total_tokens']} (in={usage['input_tokens']}, out={usage['output_tokens']})")
```

### Report authoring (W&B Reports)

```python
# Install the reports extra using configured install command:
# <python_install> "wandb[workspaces]"

from wandb.apis import reports as wr

runset = wr.Runset(entity="<wandb_entity>", project="<wandb_project>", name="All runs")
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["LOSS_KEY"]),
        wr.BarPlot(title="Accuracy", metrics=["ACC_KEY"], orientation="v"),
    ],
)

report = wr.Report(
    entity="<wandb_entity>",
    project="<wandb_project>",
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

### W&B API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| API timeout | `wandb.Api()` (19s default) | `wandb.Api(timeout=60)` or `get_api()` |
| Summary access | `run.summary["loss"]` | `run.summary_metrics.get("LOSS_KEY")` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (always slice) |
| Counting runs | `len(api.runs(...))` on large project | Skip count, just `runs[:N]` |
| Pagination | `api.runs(path)` (per_page=50 default) | `api.runs(path, per_page=min(N, 1000))` |
| History — no keys on large run | `run.history(samples=10)` → **502** | `run.history(samples=10, keys=["LOSS_KEY"])` |
| scan_history — no keys | `scan_history()` → timeout | `scan_history(keys=["LOSS_KEY"])` |
| Large history (10K+ steps) | `scan_history(keys=[...])` | `beta_scan_history(keys=[...])` (parquet) |
| Config iteration | `for k,v in run.config.items()` | Use `config_keys` from Configuration |
| Cross-run search | iterate all runs client-side | Server-side filter: `{"summary_metrics.X": {"$gt": Y}}` |

### Weave logging noise

```python
import logging
logging.getLogger("weave").setLevel(logging.ERROR)
```

---

## Quick reference

```python
from wandb_helpers import get_api, fetch_runs, scan_history
import pandas as pd
import numpy as np

api = get_api()
path = "{entity}/{project}"

# --- Weave: Init and get calls ---
import weave
client = weave.init(f"{entity}/{project}")
calls = client.get_calls(limit=10)

# --- W&B: Best run (server-side sort) ---
best = api.runs(path, filters={"state": "finished"},
                order="+summary_metrics.LOSS_KEY", per_page=1)[:1]
print(f"Best: {best[0].name}, loss={best[0].summary_metrics.get('LOSS_KEY')}")

# --- W&B: Loss curve to numpy ---
rows = scan_history(run, keys=["LOSS_KEY"])
losses = np.array([r["LOSS_KEY"] for r in rows])
print(f"min={losses.min():.6f}, final={losses[-1]:.6f}, steps={len(losses)}")

# --- W&B: Runs to DataFrame (selective) ---
df = pd.DataFrame(fetch_runs(api, path,
    metric_keys=["LOSS_KEY", "ACC_KEY"],
    filters={"state": "finished"}, limit=100))

# --- W&B: Compare two runs ---
from wandb_helpers import compare_configs
diffs = compare_configs(run_a, run_b, keys=CONFIG_KEYS)
print(pd.DataFrame(diffs).to_string(index=False))
```
