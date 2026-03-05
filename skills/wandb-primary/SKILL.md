---
description: "Comprehensive primary skill for the W&B Copilot agent. Covers Weave\
  \ queries, W&B run analysis, evaluation inspection, data wrangling, package management,\
  \ and common gotchas \u2014 everything an agent needs to answer W&B questions."
name: wandb-primary
---

# W&B Primary Skill

This skill is the primary reference for W&B agent work. It combines the best patterns into a single reference that covers:

1. **Environment & package management** — installing packages, using uv
2. **Weave SDK** — querying calls, evals, traces, token usage
3. **W&B SDK** — querying runs, configs, metrics, artifacts, history
4. **Data wrangling** — recursive unwrap, pandas patterns, handling Weave types
5. **Evaluation inspection** — drilling into eval results, per-task scores
6. **Training metrics & time-series** — loss curves, anomaly detection, sweep analysis, diagnostics
7. **Error analysis** — open coding → axial coding methodology

## Bundled files

This skill includes scripts and references that are available in the sandbox:

### Helper library — `wandb_helpers.py`

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from wandb_helpers import (
    unwrap,                  # Recursively convert Weave types → plain Python
    get_token_usage,         # Extract token counts from a call's summary
    eval_results_to_dicts,   # predict_and_score calls → list of result dicts
    pivot_solve_rate,        # Build task-level pivot table across agents
    results_summary,         # Print compact eval summary
    eval_health,             # Extract status/counts from Evaluation.evaluate calls
    eval_efficiency,         # Compute tokens-per-success across eval calls
)
```

### Weave API reference — `references/WEAVE_API.md`

Read `skills/wandb-primary/references/WEAVE_API.md` for the full `weave_tools.weave_api` method reference. This covers the high-level `Project`, `Op`, `Eval`, `CallsView` API — a lightweight alternative to using the weave SDK directly.

To use it:
```python
import sys
sys.path.insert(0, "skills/weave/scripts")  # weave skill must also be loaded
from weave_tools.weave_api import init, Project
init("entity/project")
project = Project.current()
print(project.summary())
```

### W&B SDK reference — `references/WANDB_SDK.md`

Read `skills/wandb-primary/references/WANDB_SDK.md` for the full W&B SDK reference covering training runs, metrics history, artifacts, sweeps, and system metrics. This is the traditional W&B Models product — use it for anything related to training, loss curves, hyperparameter analysis, and run comparison.

```python
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
runs = api.runs("entity/project", filters={"state": "finished"})
```

8. **W&B Reports** — programmatic report creation
9. **Common gotchas** — the traps that waste agent time

---

## 1. Environment & Package Management

The sandbox has Python 3.13, `uv`, `wandb`, `weave`, `pandas`, and `numpy` pre-installed. For anything else, install on the fly.

### Installing packages

```bash
# Install into the sandbox (persistent for the session)
uv pip install matplotlib seaborn rich tabulate

# Or use PEP 723 inline script metadata (no global install needed)
cat > analysis.py << 'EOF'
# /// script
# dependencies = ["pandas", "matplotlib", "rich", "tabulate"]
# ///
import pandas as pd
# ... your code ...
EOF
uv run analysis.py

# One-off with --with
uv run --with pandas --with rich python -c "import pandas; print(pandas.__version__)"
```

### Environment variables

```python
import os
entity  = os.environ["WANDB_ENTITY"]    # W&B entity/team
project = os.environ["WANDB_PROJECT"]    # W&B project name
api_key = os.environ["WANDB_API_KEY"]    # already authenticated
base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
```

### Running scripts

- Always write multi-line Python to a file and run it: `uv run script.py`
- Never use `pip install` directly — always `uv pip install` or `uv run --with`
- For quick one-liners: `uv run python -c '...'`

---

## CRITICAL: Always Deliver a Final Answer

**Do not end your work mid-analysis.** Every task must conclude with a clear, structured final response that directly answers the question asked. Common failure modes to avoid:

1. **Research loop trap**: Don't keep writing and iterating on scripts without ever reporting results. If your script runs and produces output, **immediately summarize and present those results** to the user.
2. **"Let me build the final analysis" trap**: If you find yourself saying "now let me build the final analysis" or "I have everything I need, let me compile it" — STOP and produce the answer right now with what you have. Do not start another iteration.
3. **Truncated output**: When presenting tables, use full display names. If a table is large, summarize it with the key findings rather than cutting it off.
4. **Missing the actual question**: Re-read the original question before writing your final answer. Make sure every sub-question is explicitly answered with specific numbers.

**Pattern for analysis tasks:**
1. Query the data (1-2 scripts max)
2. Extract the numbers you need
3. Present a structured answer: table + key findings + direct answers to each sub-question

---

## CRITICAL: Use pandas and numpy for Data Analysis

`pandas` and `numpy` are pre-installed in the sandbox. **Always use them.** Reading raw data (history rows, run lists, trace outputs) directly into your context will overwhelm your working memory and produce garbage results.

**The pattern:**

1. **Inspect structure first** — look at column names, dtypes, and row counts before loading data:
   ```python
   df = run.history(samples=10, keys=["loss", "val_loss"])
   print(df.columns.tolist())
   print(df.dtypes)
   print(f"Shape: {df.shape}")
   ```

2. **Load into pandas/numpy** — then compute stats, never eyeball raw numbers:
   ```python
   import pandas as pd
   import numpy as np

   df = pd.DataFrame(list(run.scan_history(keys=["loss", "val_loss"])))
   print(f"Min loss: {df['loss'].min():.6f} at step {df['loss'].idxmin()}")
   print(f"Final loss: {df['loss'].iloc[-1]:.6f}")
   print(f"Converged: {df['loss'].tail(100).std() < 0.001}")
   ```

3. **Summarize, don't dump** — print computed statistics and tables, not raw data:
   ```python
   # BAD: prints thousands of rows
   for row in run.scan_history(keys=["loss"]):
       print(row)

   # GOOD: prints a summary
   losses = np.array([r["loss"] for r in run.scan_history(keys=["loss"])])
   print(f"Loss curve: {len(losses)} steps, min={losses.min():.4f}, "
         f"final={losses[-1]:.4f}, mean_last_10%={losses[-len(losses)//10:].mean():.4f}")
   ```

See `references/WANDB_SDK.md` for complete pandas/numpy recipes covering loss curves, run comparison, sweep analysis, anomaly detection, and overfitting checks.

---

## 2. Weave SDK — Querying Traces, Calls, and Evals

### Initialization

```python
import weave

# Option A: init with entity/project string
client = weave.init("wandb/my-project")

# Option B: read from environment
import os
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

# The client object is your entry point
# client.entity, client.project, client.get_calls(), client.get_call()
```

**IMPORTANT**: `weave.init()` takes a positional string, NOT keyword args:
```python
weave.init("entity/project")                    # CORRECT
weave.init(project_name="entity/project")       # WRONG — TypeError
```

### Querying calls

```python
from weave.trace.weave_client import CallsFilter

# Basic queries
calls = client.get_calls(limit=100)
calls = client.get_calls(filter=CallsFilter(trace_roots_only=True), limit=50)

# Filter by op name (use wildcard * for version hash)
op_ref = f"weave:///{client.entity}/{client.project}/op/Evaluation.evaluate:*"
calls = client.get_calls(filter=CallsFilter(op_names=[op_ref]))

# Filter by parent (get children of an eval call)
# IMPORTANT: it's parent_ids (PLURAL) and takes a LIST
children = list(client.get_calls(
    filter=CallsFilter(parent_ids=["019ca0aa-745b-73f9-be4c-05e1759d3ca7"])
))

# Sort by most recent
calls = client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=10,
)

# Count without fetching all data (cheap)
count = len(client.get_calls(filter=CallsFilter(trace_roots_only=True)))

# Restrict columns for performance
calls = client.get_calls(columns=["op_name", "started_at"], limit=1000)

# ── Efficient counting for large projects (calls_query_stats) ──
# For projects with 100k+ traces, use the server-side stats endpoint
# instead of fetching all calls. This returns counts without data transfer.
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

stats = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=f"{client.entity}/{client.project}",
    filter={"op_names": [f"weave:///{client.entity}/{client.project}/op/rollout:*"]},
))
print(f"rollout call count: {stats.count}")

# Count by different op names
for op_name in ["rollout", "ruler_score_group", "openai.chat.completions.create"]:
    stats = client.server.calls_query_stats(CallsQueryStatsReq(
        project_id=f"{client.entity}/{client.project}",
        filter={"op_names": [f"weave:///{client.entity}/{client.project}/op/{op_name}:*"]},
    ))
    print(f"  {op_name}: {stats.count}")

# Count root-only traces
stats = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=f"{client.entity}/{client.project}",
    filter={"trace_roots_only": True},
))
print(f"Root traces: {stats.count}")

# Convert to pandas
df = client.get_calls(limit=500).to_pandas()
```

### CallsFilter fields

```python
CallsFilter(
    op_names=["weave:///entity/project/op/MyOp:*"],
    parent_ids=["call-id-123"],       # children of specific call(s)
    trace_ids=["trace-id-abc"],       # all calls in a trace
    call_ids=["id1", "id2"],          # specific calls by ID
    trace_roots_only=True,            # only root-level calls (no parent)
    wb_run_ids=["run-id"],            # calls from a W&B run
)
```

### Advanced query (MongoDB-style)

```python
from weave.trace_server.interface.query import Query

# Find error calls
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
                "case_insensitive": True,
            }
        }
    })
)
```

**Operators:** `$eq`, `$gt`, `$lt`, `$gte`, `$lte`, `$and`, `$or`, `$not`, `$in`, `$contains`
**Operands:** `{"$getField": "field.path"}`, `{"$literal": value}`

### Reading call data

```python
call = client.get_call("call-id-here")

# Identity
call.id              # str: call UUID
call.op_name         # str: full ref URI
call.func_name       # str: just "Evaluation.evaluate"
call.display_name    # str | None
call.trace_id        # str
call.parent_id       # str | None (None = root)

# Timing
call.started_at      # datetime
call.ended_at        # datetime | None
duration = (call.ended_at - call.started_at).total_seconds()

# I/O
call.inputs          # WeaveDict (dict-like)
call.output          # WeaveDict or WeaveObject or None
call.exception       # str | None

# Status
status = call.summary.get("weave", {}).get("status")
# "success", "error", "running", "descendant_error"

# Token usage (keyed by model name)
usage = call.summary.get("usage", {})
for model_name, u in usage.items():
    input_tokens = u.get("input_tokens") or u.get("prompt_tokens") or 0
    output_tokens = u.get("output_tokens") or u.get("completion_tokens") or 0
    total = u.get("total_tokens", 0)

# Children
children = list(client.get_calls(filter=CallsFilter(parent_ids=[call.id])))
```

### Evaluation call hierarchy

```
Evaluation.evaluate (root)
  ├── Evaluation.predict_and_score (one per dataset row × trials)
  │     ├── model.predict (the actual model call)
  │     ├── scorer_1.score
  │     └── scorer_2.score
  └── Evaluation.summarize
```

### Inspecting eval results — the full recipe

```python
import weave
from weave.trace.weave_client import CallsFilter

client = weave.init("entity/project")

# 1. Get the eval call
eval_call = client.get_call("019ca0aa-...")

# 2. Get agent name from display_name
display = eval_call.display_name  # e.g. "all-dataset__improver_all"
agent = display.split("__")[-1] if display and "__" in display else "unknown"

# 3. Get all children
children = list(client.get_calls(
    filter=CallsFilter(parent_ids=[eval_call.id])
))
pas_calls = [c for c in children if "predict_and_score" in str(c.op_name)]

# 4. Extract per-task results
results = []
for c in pas_calls:
    example = c.inputs.get("example")
    task_name = str(example.get("name"))

    out = c.output
    scores = out.get("scores")
    rubric = scores.get("rubric")                              # WeaveObject!
    rubric_passed = getattr(rubric, "passed", None)            # use getattr
    rubric_meta = getattr(rubric, "metadata", None)
    rubric_score = rubric_meta.get("score") if rubric_meta else None

    model_out = out.get("output")                              # output.output!
    succeeded = model_out.get("succeeded") if model_out else None
    error = model_out.get("error") if model_out else None
    tool_calls = model_out.get("tool_calls") if model_out else []
    trajectory = model_out.get("trajectory") if model_out else []

    duration = (c.ended_at - c.started_at).total_seconds() if c.ended_at else None

    results.append({
        "task": task_name,
        "agent": agent,
        "score": rubric_score,
        "passed": rubric_passed,
        "succeeded": succeeded,
        "error": str(error)[:100] if error else None,
        "tool_calls": len(tool_calls) if tool_calls else 0,
        "traj_len": len(trajectory) if trajectory else 0,
        "duration_s": round(duration, 1) if duration else None,
    })

results.sort(key=lambda r: r["task"])
```

---

## 3. Handling Weave Data Types — Recursive Unwrap

Weave returns `WeaveDict`, `WeaveObject`, `ObjectRef`, and other wrapper types. This recursive unwrap function converts everything to plain Python dicts/lists for easy processing:

```python
def unwrap(obj):
    """Recursively convert Weave types to plain Python."""
    # WeaveDict → dict
    if hasattr(obj, "keys") and hasattr(obj, "get") and not isinstance(obj, dict):
        return {k: unwrap(obj[k]) for k in obj.keys()}
    # WeaveObject / ObjectRecord → dict via attributes
    if hasattr(obj, "__dict__") and hasattr(obj, "_val"):
        record = object.__getattribute__(obj, "_val")
        if hasattr(record, "__dict__"):
            return {k: unwrap(v) for k, v in vars(record).items()
                    if not k.startswith("_")}
    # WeaveList → list
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return [unwrap(item) for item in obj]
        except TypeError:
            pass
    # ObjectRef → resolve to string
    if hasattr(obj, "entity") and hasattr(obj, "_digest"):
        return str(obj)
    return obj
```

Usage:
```python
call = client.get_call("some-id")
output = unwrap(call.output)       # now a plain dict, safe for json.dumps
inputs = unwrap(call.inputs)

import json
print(json.dumps(output, indent=2, default=str))
```

### When to use unwrap vs direct access

- **Use `unwrap()`** when you need to serialize to JSON, pass to pandas, or inspect unknown structures
- **Use direct access** (`.get()`, `getattr()`) when you know the exact structure and want to avoid the overhead
- **Key rule**: `WeaveDict` supports `.get()` and `.keys()`. `WeaveObject` requires `getattr()`. When in doubt, unwrap first.

---

## 4. W&B SDK — Querying Runs, Metrics, Artifacts

### Initialization

```python
import wandb
api = wandb.Api()  # uses WANDB_API_KEY from environment

import os
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"
```

### Listing runs

```python
# All runs (lazy iterator — fetches per_page at a time)
runs = api.runs(path)

# With filters (MongoDB-style, server-side)
runs = api.runs(path, filters={"state": "finished"})
runs = api.runs(path, filters={"config.model": "gpt-4"})
runs = api.runs(path, filters={"summary_metrics.accuracy": {"$gt": 0.9}})
runs = api.runs(path, filters={"display_name": {"$regex": ".*v2.*"}})

# Sorted (prefix with + for asc, - for desc)
runs = api.runs(path, order="-created_at")              # newest first
runs = api.runs(path, order="+summary_metrics.loss")     # lowest loss first
runs = api.runs(path, order="-summary_metrics.val_acc")  # highest val_acc

# Pagination
runs = api.runs(path, per_page=100)  # 100 per API page
first_50 = runs[:50]                  # slice to limit (lazy)
total = len(runs)                     # triggers count query
```

### MongoDB-style filter syntax

```python
# By state
runs = api.runs(path, filters={"state": "finished"})

# By config value
runs = api.runs(path, filters={"config.learning_rate": {"$lt": 0.01}})

# By summary metric
runs = api.runs(path, filters={"summary_metrics.loss": {"$lt": 0.5}})

# By display name (regex)
runs = api.runs(path, filters={"display_name": {"$regex": "experiment_v2.*"}})

# By tags
runs = api.runs(path, filters={"tags": {"$in": ["production"]}})
runs = api.runs(path, filters={"tags": {"$nin": ["deprecated"]}})

# Compound filters
runs = api.runs(path, filters={
    "$and": [
        {"config.model": "transformer"},
        {"summary_metrics.loss": {"$lt": 0.5}},
        {"state": "finished"},
    ]
})

# Date range
runs = api.runs(path, filters={
    "$and": [
        {"created_at": {"$gt": "2025-01-01T00:00:00"}},
        {"created_at": {"$lt": "2025-06-01T00:00:00"}},
    ]
})

# Config key exists
runs = api.runs(path, filters={"config.special_param": {"$exists": True}})
```

**Operators:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$regex`, `$and`, `$or`, `$nor`

**Filterable fields:** `state`, `display_name`, `config.KEY`, `summary_metrics.KEY`, `tags`, `created_at`, `group`, `job_type`

### Reading run data

```python
run = api.run(f"{path}/run-id")

# Identity
run.id           # str: 8-char hash like "abc123de"
run.name         # str: human-readable display name
run.entity       # str
run.project      # str
run.url          # str: full URL to W&B UI
run.path         # list: [entity, project, run_id]

# State & metadata
run.state        # "finished", "failed", "crashed", "running"
run.created_at   # str: ISO timestamp
run.tags         # list[str]
run.metadata     # dict: git info, hardware, etc. (from wandb-metadata.json)

# Hyperparameters (input config logged at wandb.init)
run.config       # dict
lr = run.config.get("learning_rate")
clean_config = {k: v for k, v in run.config.items() if not k.startswith("_")}

# Final metrics (last value logged for each key)
run.summary         # HTTPSummary (mutable dict-like)
run.summary_metrics  # dict (read-only snapshot — prefer this for reads)
final_loss = run.summary_metrics.get("loss")
```

### Metrics history — `run.history()` vs `run.scan_history()`

This is critical for loss curve analysis. There are two methods with very different behavior:

#### `run.history()` — sampled, fast, returns DataFrame

```python
# Quick 500-point overview (default)
df = run.history()

# Specific metrics with more samples
df = run.history(samples=2000, keys=["loss", "val_loss", "learning_rate"])

# System metrics (GPU, CPU, memory)
sys_df = run.history(stream="system")

# As list of dicts instead of DataFrame
rows = run.history(pandas=False, keys=["loss"])
```

**Behavior**: Server-side downsamples to `samples` points using min/max bucketing. Fast, but you WILL miss data points if the run logged more than `samples` steps. Good for quick overviews and plotting.

#### `run.scan_history()` — full, unsampled, iterator

```python
# Get ALL loss values (no sampling)
losses = [row["loss"] for row in run.scan_history(keys=["loss"])]

# Specific step range
for row in run.scan_history(keys=["loss", "accuracy"], min_step=1000, max_step=2000):
    print(row["_step"], row.get("loss"))

# Convert to DataFrame
import pandas as pd
rows = list(run.scan_history(keys=["loss", "val_loss"], page_size=2000))
df = pd.DataFrame(rows)
```

**Behavior**: Returns ALL logged history rows, unsampled. Fetches page-by-page (memory efficient). Use when you need precision — anomaly detection, exact loss values, etc.

**NaN note**: Rows where a key was not logged at that step will have NaN. Different metrics can be logged at different frequencies.

#### When to use which

| Use case | Method |
|----------|--------|
| Quick plot / overview | `history(samples=500)` |
| Dashboard summary | `history(samples=1000, keys=[...])` |
| Anomaly detection | `scan_history(keys=[...])` |
| Exact loss values | `scan_history(keys=["loss"])` |
| System metrics (GPU/CPU) | `history(stream="system")` |
| Very long runs | `scan_history()` (iterator, not all in memory) |

### Bulk history — multiple runs at once

```python
# runs.histories() fetches sampled history for ALL runs in one call
runs = api.runs(path, filters={"state": "finished"})
df = runs.histories(samples=200, keys=["loss", "val_loss"], format="pandas")
# df has columns: run_id, _step, loss, val_loss
```

### Runs to pandas — summary table

```python
import pandas as pd

runs = api.runs(path, filters={"state": "finished"}, order="-created_at")
rows = []
for run in runs[:200]:  # ALWAYS slice — never list() all runs
    summary = getattr(run.summary, "_json_dict", {})
    rows.append({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        **{f"config.{k}": v for k, v in run.config.items() if not k.startswith("_")},
        "loss": summary.get("loss"),
        "accuracy": summary.get("accuracy"),
    })
df = pd.DataFrame(rows)
print(df.describe())
```

### Comparing runs

```python
# Side-by-side config comparison
run_a = api.run(f"{path}/run-a")
run_b = api.run(f"{path}/run-b")

config_df = pd.DataFrame([run_a.config, run_b.config]).T
config_df.columns = [run_a.name, run_b.name]
diff = config_df[config_df.iloc[:, 0] != config_df.iloc[:, 1]]
print("Config differences:\n", diff)

# Side-by-side metric comparison
for key in ["loss", "accuracy", "val_loss"]:
    a_val = run_a.summary_metrics.get(key, "N/A")
    b_val = run_b.summary_metrics.get(key, "N/A")
    print(f"  {key}: {a_val} vs {b_val}")
```

### Artifacts

```python
# Fetch by name and version/alias
artifact = api.artifact(f"{path}/model-weights:latest")
artifact = api.artifact(f"{path}/model-weights:v3")
artifact = api.artifact(f"{path}/my-dataset:best")

# Properties
artifact.name         # str
artifact.type         # str: "model", "dataset", etc.
artifact.version      # str: "v0", "v1"
artifact.aliases      # list[str]: ["latest", "best"]
artifact.metadata     # dict: user-defined metadata
artifact.size         # int: total bytes
artifact.created_at   # str

# Download
local_path = artifact.download()
artifact.download(root="/tmp/data")           # custom destination
artifact.download(path_prefix="train/")       # only a subdirectory

# Lineage
producer = artifact.logged_by()    # run that created it
consumers = artifact.used_by()     # runs that consumed it

# From a run
for art in run.logged_artifacts():
    print(art.name, art.type, art.aliases)
for art in run.used_artifacts():
    print(art.name, art.version)

# Check existence
exists = api.artifact_exists(f"{path}/my-model:v0")
```

### Listing projects

```python
projects = list(api.projects(entity, per_page=200))
for p in projects[:20]:
    print(p.name, getattr(p, "created_at", None))
```

### System metrics

```python
# System metrics: GPU, CPU, memory, disk, network
sys_df = run.history(stream="system")
# Columns: system.cpu, system.memory, system.disk,
#           system.gpu.0.gpu, system.gpu.0.memory, system.gpu.0.temp, etc.

# Latest snapshot only
latest = run.system_metrics  # dict
```

### Refreshing cached data

```python
# If you're reading live/running data, flush the cache
api.flush()  # clears cached values, next read fetches fresh data
```

---

## 5. Data Analysis Patterns with Pandas

### Installing pandas in the sandbox

```bash
uv pip install pandas
# or in a script:
# /// script
# dependencies = ["pandas"]
# ///
```

### Weave calls to DataFrame

```python
import pandas as pd

# Method 1: built-in (if available)
df = client.get_calls(limit=500).to_pandas()

# Method 2: manual (more control)
calls = list(client.get_calls(limit=500))
rows = []
for c in calls:
    rows.append({
        "id": c.id,
        "op": c.func_name,
        "started": c.started_at,
        "ended": c.ended_at,
        "duration_s": (c.ended_at - c.started_at).total_seconds() if c.ended_at else None,
        "status": c.summary.get("weave", {}).get("status"),
        "total_tokens": sum(
            u.get("total_tokens", 0)
            for u in c.summary.get("usage", {}).values()
        ),
    })
df = pd.DataFrame(rows)
```

### Eval results to DataFrame

```python
# From the eval inspection recipe above:
df = pd.DataFrame(results)
print(df.describe())
print(df.groupby("passed")["score"].mean())
print(df.sort_values("score", ascending=False).head(10))
```

### Pivot table — solve rate per task across agents

```python
from collections import defaultdict

by_task = defaultdict(list)
for r in all_results:  # collected from multiple eval runs
    by_task[r["task"]].append(r)

pivot_rows = []
for task in sorted(by_task):
    entries = by_task[task]
    n = len(entries)
    passed = sum(1 for e in entries if e["passed"])
    mean_score = sum(e["score"] for e in entries) / n
    pivot_rows.append({
        "task": task,
        "agents_passed": passed,
        "agents_attempted": n,
        "pass_rate": f"{passed/n:.0%}",
        "mean_score": round(mean_score, 3),
    })

pivot_df = pd.DataFrame(pivot_rows)
print(pivot_df.to_string(index=False))
```

---

## 6. Training Metrics & Time-Series Analysis

Patterns for analyzing loss curves, detecting anomalies, comparing training runs, and diagnosing common training issues. **Always load metrics into pandas** — never dump raw history into context.

### Loading a loss curve

```python
import pandas as pd
import wandb

api = wandb.Api()
run = api.run(f"{entity}/{project}/run-id")

# Sampled (fast, good for plotting)
df = run.history(samples=2000, keys=["loss", "val_loss", "learning_rate", "_step"])

# Full precision (use for anomaly detection)
df = pd.DataFrame(list(run.scan_history(keys=["loss", "val_loss"], page_size=2000)))
```

### Loss curve anomaly detection

```python
import numpy as np

df = pd.DataFrame(list(run.scan_history(keys=["loss"])))
loss = df["loss"].dropna()

# Spike detection: points > N standard deviations from rolling mean
window = 50
rolling_mean = loss.rolling(window, center=True).mean()
rolling_std = loss.rolling(window, center=True).std()
threshold = 3.0
spikes = loss[(loss - rolling_mean).abs() > threshold * rolling_std]
print(f"Loss spikes (>{threshold}σ): {len(spikes)} at steps {list(spikes.index)}")

# Divergence detection: sustained increase over a window
diff = loss.diff().rolling(window).mean()
diverging = diff[diff > 0]
if len(diverging) > window:
    first_diverge = diverging.index[0]
    print(f"Loss may be diverging starting at step {first_diverge}")

# Plateau detection: loss barely changing
change_rate = loss.diff().abs().rolling(window).mean()
plateau_threshold = 1e-5
plateaus = change_rate[change_rate < plateau_threshold]
if len(plateaus) > window:
    print(f"Loss plateaued at step {plateaus.index[0]}, value ~{loss.iloc[plateaus.index[0]]:.6f}")

# NaN/Inf detection
nan_steps = df[df["loss"].isna()]["_step"].tolist() if "_step" in df else []
if nan_steps:
    print(f"NaN loss at steps: {nan_steps[:10]}")
```

### Comparing loss curves across runs

```python
import pandas as pd

# Load histories for multiple runs
run_ids = ["abc123", "def456", "ghi789"]
dfs = []
for rid in run_ids:
    run = api.run(f"{path}/{rid}")
    df = run.history(samples=1000, keys=["loss", "_step"])
    df["run_name"] = run.name
    df["run_id"] = rid
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Summary stats per run
summary = combined.groupby("run_name")["loss"].agg(["min", "max", "mean", "std", "last"])
print(summary.sort_values("min"))

# Or use bulk API
runs = api.runs(path, filters={"state": "finished"})
combined = runs.histories(samples=500, keys=["loss", "val_loss"], format="pandas")
# combined has: run_id, _step, loss, val_loss
```

### Gradient / learning rate analysis

```python
# Load LR schedule alongside loss
df = run.history(samples=2000, keys=["loss", "learning_rate", "_step"])

# Detect LR warmup phase
lr = df["learning_rate"].dropna()
if lr.iloc[0] < lr.max() * 0.1:
    warmup_end = lr[lr >= lr.max() * 0.9].index[0]
    print(f"LR warmup phase: steps 0-{warmup_end}")

# Correlate LR changes with loss changes
df["loss_change"] = df["loss"].diff()
df["lr_change"] = df["learning_rate"].diff()
lr_drops = df[df["lr_change"] < -1e-6]
if len(lr_drops):
    print(f"LR drops at steps: {lr_drops['_step'].tolist()[:10]}")
    for _, row in lr_drops.head(5).iterrows():
        step = int(row["_step"])
        # Check if loss improved after LR drop
        after = df[df["_step"] > step].head(50)["loss"].mean()
        before = df[df["_step"] < step].tail(50)["loss"].mean()
        improvement = before - after
        print(f"  Step {step}: LR {row['learning_rate']:.2e}, "
              f"loss change: {improvement:+.4f}")
```

### Training diagnostics summary

```python
def diagnose_run(run):
    """Quick diagnostic summary of a training run."""
    df = pd.DataFrame(list(run.scan_history(keys=["loss", "val_loss"])))
    loss = df["loss"].dropna()

    diagnostics = {
        "total_steps": len(loss),
        "final_loss": loss.iloc[-1] if len(loss) else None,
        "min_loss": loss.min(),
        "min_loss_step": loss.idxmin(),
        "has_nan": loss.isna().any(),
        "final_10pct_mean": loss.tail(len(loss) // 10).mean(),
    }

    # Check for overfitting (val_loss diverging from train loss)
    if "val_loss" in df.columns:
        val = df["val_loss"].dropna()
        if len(val) > 10:
            train_tail = loss.tail(len(loss) // 5).mean()
            val_tail = val.tail(len(val) // 5).mean()
            diagnostics["train_val_gap"] = val_tail - train_tail
            diagnostics["likely_overfit"] = val_tail > train_tail * 1.2

    # Check convergence
    if len(loss) > 100:
        last_pct = loss.tail(len(loss) // 10)
        diagnostics["converged"] = last_pct.std() < last_pct.mean() * 0.01

    return diagnostics

diag = diagnose_run(run)
for k, v in diag.items():
    print(f"  {k}: {v}")
```

### System metrics correlation

```python
# Check if GPU utilization drops correlate with training slowdowns
train_df = run.history(samples=1000, keys=["loss", "_step", "_timestamp"])
sys_df = run.history(stream="system", samples=1000)

# GPU utilization
if "system.gpu.0.gpu" in sys_df.columns:
    gpu_util = sys_df["system.gpu.0.gpu"].dropna()
    low_util = gpu_util[gpu_util < 50]
    print(f"Low GPU utilization (<50%): {len(low_util)}/{len(gpu_util)} samples")
    print(f"Mean GPU util: {gpu_util.mean():.1f}%")

# Memory
if "system.gpu.0.memory" in sys_df.columns:
    gpu_mem = sys_df["system.gpu.0.memory"].dropna()
    print(f"GPU memory: mean={gpu_mem.mean():.1f}%, max={gpu_mem.max():.1f}%")
```

### Sweep analysis — finding best hyperparameters

```python
# Sweeps log many runs with different configs
runs = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss")
rows = []
for run in runs[:100]:
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    summary = run.summary_metrics
    rows.append({
        "run_id": run.id,
        "name": run.name,
        **config,
        "loss": summary.get("loss"),
        "val_loss": summary.get("val_loss"),
        "accuracy": summary.get("accuracy"),
    })

sweep_df = pd.DataFrame(rows)
print("Best runs by loss:")
print(sweep_df.nsmallest(5, "loss")[["name", "loss", "accuracy"]].to_string(index=False))

# Hyperparameter importance (simple correlation)
numeric_cols = sweep_df.select_dtypes(include=[np.number]).columns
correlations = sweep_df[numeric_cols].corr()["loss"].drop("loss").sort_values()
print("\nCorrelation with loss:")
print(correlations.to_string())
```

---

## 7. Error Analysis — Open Coding to Axial Coding

Use this workflow for structured failure analysis on eval results.

### Step 1: Understand the data shape

```python
import sys
sys.path.insert(0, "skills/weave/scripts")
from weave_tools.weave_api import init, Eval

init("entity/project")
ev = Eval.from_call_id("YOUR_EVAL_CALL_ID")
print(ev.summarize())
calls = ev.model_calls()
print(calls.input_shape(depth=4, sample=50))
print(calls.output_shape(depth=4, sample=50))
```

### Step 2: Open coding — write failure notes

Create a Weave scorer that journals what went wrong for each failing call:

```python
import weave
from weave.flow.scorer import Scorer

class OpenCodingNoteV1(Scorer):
    name: str = "open_coding_note_v1"

    @weave.op
    def score(self, output, *, failure=None):
        if not (isinstance(failure, dict)
                and isinstance(failure.get("output"), dict)
                and failure["output"].get("passed") is False):
            return {"text": "(passed or unscored)"}
        # Build record_text from output, call LLM to analyze failure
        # Return {"text": "failure note here"}
        ...
```

### Step 3: Axial coding — classify into taxonomy

```python
from typing import ClassVar

class AxialCodingClassifierV1(Scorer):
    name: str = "axial_coding_classifier_v1"
    TAXONOMY: ClassVar[list[str]] = [
        "wrong_api_usage", "incorrect_filter", "timeout",
        "missing_data", "hallucinated_answer", "partial_answer",
    ]

    @weave.op
    def score(self, output, *, note=None):
        # Use note["text"] + TAXONOMY to classify
        # Return {"labels": [...], "primary": "...", "rationale": "..."}
        ...
```

### Step 4: Summarize

```python
from collections import Counter

primary_counts = Counter()
for call in calls:
    cls = call.feedback("axial_coding_classifier_v1", default=None)
    if isinstance(cls, dict):
        primary = cls.get("primary")
        if primary:
            primary_counts[primary] += 1

for label, count in primary_counts.most_common():
    print(f"  {label}: {count}")
```

---

## 8. W&B Reports — Programmatic Creation

```bash
uv pip install "wandb[workspaces]"
```

```python
from wandb.apis import reports as wr
import wandb_workspaces.expr as expr

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

# Create a runset (defines which runs appear in plots)
runset = wr.Runset(
    entity=entity,
    project=project,
    filters=[
        expr.Metric("State") == "finished",
    ],
)

# Build the report
report = wr.Report(
    entity=entity,
    project=project,
    title="Eval Results",
    width="fixed",
    blocks=[
        wr.H1(text="Evaluation Results"),
        wr.P(text="Auto-generated analysis."),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.LinePlot(title="Loss", x="_step", y=["loss"]),
                wr.BarPlot(title="Accuracy", metrics=["accuracy"], orientation="v"),
            ],
        ),
    ],
)

# Save as draft (only when asked to publish!)
# report.save(draft=True)
```

### Runset filtering tips

```python
# By config values
filters = [expr.Config("model") == "gpt-4"]

# By summary metrics
filters = [expr.Summary("accuracy") >= 0.9]

# By tags
filters = [expr.Tags().isin(["baseline", "v2"])]

# By specific run IDs
filters = [expr.Metric("name").isin(["run_id_1", "run_id_2"])]
```

**Gotcha**: Don't use dot-paths like `config.lr` in filter strings — use `Config("lr")`, `Summary("loss")`, etc.

---

## 9. Common Gotchas — Save Yourself Hours

### Weave API gotchas

| Gotcha | Wrong | Right |
|--------|-------|-------|
| Parent filter | `filter={'parent_id': 'x'}` | `filter={'parent_ids': ['x']}` (plural, list) |
| WeaveObject access | `rubric.get('passed')` | `getattr(rubric, 'passed', None)` |
| Nested output | `out.get('succeeded')` | `out.get('output').get('succeeded')` (output.output) |
| ObjectRef to string | `name_ref == "foo"` | `str(name_ref) == "foo"` |
| weave.init args | `weave.init(project="x")` | `weave.init("x")` (positional) |
| CallsFilter import | `from weave import CallsFilter` | `from weave.trace.weave_client import CallsFilter` |
| Query import | `from weave import Query` | `from weave.trace_server.interface.query import Query` |
| Stats import | `client.count_calls(...)` | `from weave.trace_server.trace_server_interface import CallsQueryStatsReq` |
| Eval status | `call.summary["status"]` | `call.summary["weave"]["status"]` (nested under "weave") |
| Eval success count | `call.summary["success_count"]` | `call.summary["weave"]["status_counts"]["success"]` |

### WeaveDict vs WeaveObject

- **WeaveDict**: dict-like, supports `.get()`, `.keys()`, `[]`. Used for: `call.inputs`, `call.output`, `scores` dict
- **WeaveObject**: attribute-based, use `getattr()`. Used for: scorer results (rubric), dataset rows
- **When in doubt**: use the `unwrap()` function from Section 3

### Weave logging noise

Weave prints version warnings and login messages to stderr. Suppress with:
```python
import logging
logging.getLogger("weave").setLevel(logging.ERROR)
```
Or redirect stderr: `uv run script.py 2>/dev/null`

### W&B API gotchas

| Gotcha | Wrong | Right |
|--------|-------|-------|
| Summary access | `run.summary["loss"]` | `getattr(run.summary, '_json_dict', {}).get('loss')` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (slice to limit) |
| History sampling | `run.history()` (all fields) | `run.history(samples=500, keys=["loss"])` |
| scan_history keys | `scan_history(keys=None)` | `scan_history(keys=["loss"])` (explicit keys) |

### Package management

| Gotcha | Wrong | Right |
|--------|-------|-------|
| Installing packages | `pip install pandas` | `uv pip install pandas` |
| Running scripts | `python script.py` | `uv run script.py` |
| Quick one-off | `pip install rich && python -c ...` | `uv run --with rich python -c ...` |

---

## 10. Token Usage & Cost Analysis

```python
def get_token_usage(call):
    """Extract total token usage from a call's summary."""
    usage = call.summary.get("usage", {})
    total_input = 0
    total_output = 0
    for model, u in usage.items():
        total_input += u.get("input_tokens") or u.get("prompt_tokens") or 0
        total_output += u.get("output_tokens") or u.get("completion_tokens") or 0
    return {"input_tokens": total_input, "output_tokens": total_output,
            "total_tokens": total_input + total_output}

# For cost estimation (include_costs=True required)
call_with_costs = client.get_call("id", include_costs=True)
costs = call_with_costs.summary.get("weave", {}).get("costs", {})
```

---

## 10b. Evaluation Health & Status Analysis

Evaluations track success/error counts in their call summaries. Here's how to extract them for health analysis.

### Extracting eval status and counts

```python
import weave
from weave.trace.weave_client import CallsFilter

client = weave.init("entity/project")

# Get all Evaluation.evaluate calls
op_ref = f"weave:///{client.entity}/{client.project}/op/Evaluation.evaluate:*"
eval_calls = list(client.get_calls(
    filter=CallsFilter(op_names=[op_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=200,
))

rows = []
for ec in eval_calls:
    summary = ec.summary or {}

    # Status: "success", "error", "running", "descendant_error"
    weave_meta = summary.get("weave", {})
    status = weave_meta.get("status", "unknown")

    # Status counts (success/error across ALL descendant calls)
    status_counts = weave_meta.get("status_counts", {})
    success_count = status_counts.get("success", 0)
    error_count = status_counts.get("error", 0)

    # Token usage
    usage = summary.get("usage", {})
    total_tokens = sum(u.get("total_tokens", 0) for u in usage.values())

    # Display name
    display = ec.display_name or "unnamed"

    rows.append({
        "display_name": display,
        "started_at": ec.started_at.strftime("%Y-%m-%d %H:%M") if ec.started_at else "",
        "status": status,
        "success_count": success_count,
        "error_count": error_count,
        "total_tokens": total_tokens,
        "call_id": ec.id,
    })

# ── Summary stats ──
import pandas as pd
df = pd.DataFrame(rows)
total = len(df)
n_success = len(df[df["status"] == "success"])
n_error = len(df[df["status"].isin(["error", "descendant_error"])])
n_running = len(df[df["status"] == "running"])

print(f"Total evals: {total}")
print(f"Successful: {n_success} ({n_success/total:.0%})")
print(f"With errors: {n_error} ({n_error/total:.0%})")
print(f"Running/stale: {n_running}")
print(f"Fraction without errors: {n_success}/{total - n_running} = {n_success/(total - n_running):.0%}")
print()
print(df.to_string(index=False))
```

### Computing cost efficiency across evals

```python
# Build efficiency table: tokens per successful predict_and_score call
efficiency = []
for ec in eval_calls:
    summary = ec.summary or {}
    weave_meta = summary.get("weave", {})
    status = weave_meta.get("status", "unknown")
    if status in ("running", "unknown"):
        continue  # skip incomplete evals

    status_counts = weave_meta.get("status_counts", {})
    success_count = status_counts.get("success", 0)

    usage = summary.get("usage", {})
    total_tokens = sum(u.get("total_tokens", 0) for u in usage.values())

    # Tokens per success (lower is more efficient)
    tokens_per_success = total_tokens / success_count if success_count > 0 else float("inf")

    efficiency.append({
        "display_name": ec.display_name or "unnamed",
        "total_tokens": total_tokens,
        "success_count": success_count,
        "tokens_per_success": round(tokens_per_success),
    })

eff_df = pd.DataFrame(efficiency)
eff_df = eff_df.sort_values("tokens_per_success")
print("Ranked by efficiency (tokens per success):")
print(eff_df.to_string(index=False))
print(f"\nMost efficient: {eff_df.iloc[0]['display_name']}")
print(f"Least efficient: {eff_df.iloc[-1]['display_name']}")
```

### Key locations for eval data in the call summary

```python
summary = eval_call.summary

# Status
summary["weave"]["status"]                    # "success" | "error" | "descendant_error" | "running"

# Descendant call counts
summary["weave"]["status_counts"]["success"]  # int — successful child calls
summary["weave"]["status_counts"]["error"]    # int — errored child calls

# Token usage (keyed by model name)
summary["usage"]["gpt-4o"]["total_tokens"]    # int
summary["usage"]["gpt-4o"]["input_tokens"]    # int
summary["usage"]["gpt-4o"]["output_tokens"]   # int

# Cost (if include_costs=True was used)
summary["weave"]["costs"]                     # dict of cost data
```

---

## 11. Quick Reference — Common Operations

```python
# How many traces in the project?
count = len(client.get_calls(filter=CallsFilter(trace_roots_only=True)))

# What ops are being traced?
from collections import Counter
ops = Counter(c.func_name for c in client.get_calls(columns=["op_name"], limit=5000))
for name, n in ops.most_common(20):
    print(f"  {name}: {n}")

# Find the 5 most recent evaluations
op_ref = f"weave:///{client.entity}/{client.project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(
    filter=CallsFilter(op_names=[op_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=5,
))

# Get a run's config and final metrics
run = api.run(f"{entity}/{project}/run-id")
print(run.config)
print(dict(run.summary))

# Compare two runs
run_a = api.run(f"{entity}/{project}/run-a")
run_b = api.run(f"{entity}/{project}/run-b")
for key in ["loss", "accuracy"]:
    a_val = run_a.summary.get(key, "N/A")
    b_val = run_b.summary.get(key, "N/A")
    print(f"  {key}: {a_val} vs {b_val}")

# Get all calls in an eval's trace tree
trace_calls = list(client.get_calls(filter=CallsFilter(trace_ids=[eval_call.trace_id])))

# Get scorer results for specific predict_and_score calls
for c in pas_calls:
    rubric = c.output.get("scores", {}).get("rubric")
    score = getattr(getattr(rubric, "metadata", None), "score", None) if rubric else None
    print(f"  {c.id}: score={score}")
```
