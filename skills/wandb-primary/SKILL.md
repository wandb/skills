---
name: wandb-primary
description: "Primary skill for querying, analyzing, and launching W&B projects. Covers W&B SDK (training runs, metrics, artifacts, sweeps, reports), Weave SDK (GenAI traces, evaluations, scorers, monitors), and W&B Launch (reproducing runs, submitting training jobs to compute, queue management). Use this for: project overviews, run analysis, trace inspection, eval summaries, report creation, monitor setup, launching/relaunching runs, and any broad 'what's going on' questions."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->


# W&B Primary Skill

## Environment defaults

- **Python**: run scripts with `python`, install packages with `uv add`
- **LLM**: OpenAI `gpt-5.4` (reasoning: high, endpoint: responses)

---

## Fast recipes — use these first

These cover the most common tasks. Each is a single script. Copy, fill in placeholders, run.

### Count runs (exact, fast)

```python
import wandb, os
api = wandb.Api(timeout=60)
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
crashed = len(api.runs(path, filters={"state": "crashed"}, per_page=1, include_sweeps=False, lazy=True))
running = len(api.runs(path, filters={"state": "running"}, per_page=1, include_sweeps=False, lazy=True))
print(f"Total: {total}  |  Finished: {finished}  |  Crashed: {crashed}  |  Running: {running}")
```

### Count traces (fast, server-side)

```python
import weave, os, logging
logging.getLogger("weave").setLevel(logging.ERROR)
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
pid = f"{entity}/{project}"

# Total root traces
stats = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=pid, filter={"trace_roots_only": True}
))
print(f"Root traces: {stats.count}")

# Count by op name
for op in ["Evaluation.evaluate", "my_op.turn"]:
    s = client.server.calls_query_stats(CallsQueryStatsReq(
        project_id=pid,
        filter={"op_names": [f"weave:///{entity}/{project}/op/{op}:*"]},
    ))
    print(f"  {op}: {s.count}")
```

### Summarize project (runs + traces in one script)

```python
import wandb, weave, os, logging
logging.getLogger("weave").setLevel(logging.ERROR)
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# --- Runs ---
api = wandb.Api(timeout=60)
total_runs = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
recent = api.runs(path, order="-created_at", per_page=5)[:5]

print(f"=== Runs ({total_runs} total, {finished} finished) ===")
for r in recent:
    print(f"  {r.name} [{r.state}] {r.created_at[:10]}")

# --- Weave Traces ---
client = weave.init(path)
pid = f"{entity}/{project}"
root_stats = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=pid, filter={"trace_roots_only": True}
))
print(f"\n=== Weave Traces ({root_stats.count} root traces) ===")

recent_calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=5,
    columns=["op_name", "started_at", "display_name"],
))
for c in recent_calls:
    name = c.display_name or c.op_name.split("/")[-1].split(":")[0]
    started = c.started_at.strftime("%Y-%m-%d %H:%M") if c.started_at else "?"
    print(f"  {name} @ {started}")
```

### Inspect a single run

```python
import wandb, os
api = wandb.Api(timeout=60)
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"

run = api.run(f"{path}/RUN_ID")
print(f"Name: {run.name}")
print(f"State: {run.state}")
print(f"Created: {run.created_at}")
print(f"Tags: {run.tags}")
print(f"Last step: {run.lastHistoryStep}")

# Key metrics (replace with actual keys from probe or user request)
for k in ["loss", "val_loss", "accuracy"]:
    v = run.summary_metrics.get(k)
    if v is not None:
        print(f"  {k}: {v}")
```

### Compare two runs

```python
import wandb, os, sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from wandb_helpers import get_api, compare_configs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"

run_a = api.run(f"{path}/RUN_A_ID")
run_b = api.run(f"{path}/RUN_B_ID")

# Config diff
diffs = compare_configs(run_a, run_b)
if diffs:
    print("Config differences:")
    for d in diffs:
        print(f"  {d['key']}: {d[run_a.name]} -> {d[run_b.name]}")
else:
    print("Configs are identical")

# Metric comparison
print("\nMetrics:")
for k in ["loss", "val_loss", "accuracy"]:
    a = run_a.summary_metrics.get(k, "N/A")
    b = run_b.summary_metrics.get(k, "N/A")
    print(f"  {k}: {a} vs {b}")
```

### Summarize latest eval

```python
import weave, os, sys, logging
logging.getLogger("weave").setLevel(logging.ERROR)
from weave.trace.weave_client import CallsFilter
sys.path.insert(0, "skills/wandb-primary/scripts")
from weave_helpers import unwrap, eval_results_to_dicts, results_summary

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

# Get latest eval
op_ref = f"weave:///{entity}/{project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(
    filter=CallsFilter(op_names=[op_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=1,
))

if not evals:
    print("No evaluations found")
else:
    ec = evals[0]
    print(f"Eval: {ec.display_name or 'unnamed'} @ {ec.started_at}")

    # Get predict_and_score children
    pas_ref = f"weave:///{entity}/{project}/op/Evaluation.predict_and_score:*"
    pas = list(client.get_calls(
        filter=CallsFilter(op_names=[pas_ref], parent_ids=[ec.id])
    ))
    results = eval_results_to_dicts(pas, agent_name=ec.display_name or "agent")
    print(results_summary(results))
```

### Inspect recent traces

```python
import weave, os, logging
logging.getLogger("weave").setLevel(logging.ERROR)
sys.path.insert(0, "skills/wandb-primary/scripts")
from weave_helpers import unwrap, get_token_usage

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=10,
))

for c in calls:
    name = c.display_name or c.op_name.split("/")[-1].split(":")[0]
    started = c.started_at.strftime("%Y-%m-%d %H:%M") if c.started_at else "?"
    duration = ""
    if c.started_at and c.ended_at:
        duration = f" ({(c.ended_at - c.started_at).total_seconds():.1f}s)"
    status = c.summary.get("weave", {}).get("status", "?") if c.summary else "?"
    tokens = get_token_usage(c)
    tok_str = f" [{tokens['total_tokens']} tok]" if tokens['total_tokens'] else ""
    print(f"  {name} [{status}] {started}{duration}{tok_str}")
```

### Create a W&B Report

Install once: `uv add wandb-workspaces`. Reports live in the standalone `wandb-workspaces` package; the legacy `wandb.apis.reports` path is deprecated.

```python
import os
import wandb_workspaces.reports.v2 as wr

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

runset = wr.Runset(entity=entity, project=project, name="All runs")
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["LOSS_KEY"]),
        wr.BarPlot(title="Accuracy", metrics=["ACC_KEY"], orientation="v"),
    ],
)

report = wr.Report(
    entity=entity,
    project=project,
    title="Project Analysis",
    description="Auto-generated summary",
    width="fixed",  # "readable" | "fixed" | "fluid"
    blocks=[
        wr.H1(text="Project Analysis"),
        wr.P(text="Auto-generated summary from W&B API."),
        plots,
    ],
)
report.save(draft=True)  # omit draft=True to publish
print(f"Report saved: {report.url}")
```

### Set up a Weave Monitor

```python
import weave, os

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

# Define a scorer
@weave.op()
def my_scorer(output: dict) -> dict:
    """Score based on output quality."""
    # Replace with actual scoring logic
    passed = output.get("succeeded", False)
    return {"passed": passed, "score": 1.0 if passed else 0.0}

# Create monitor
monitor = weave.Monitor(
    entity=entity,
    project=project,
    name="quality-monitor",
    scorers=[my_scorer],
    # Filter which ops to monitor:
    # op_names=["my_agent.run"],
)
print(f"Monitor created: {monitor.name}")
```

### Relaunch a run (1 command, auto-selects queue)

```bash
python skills/wandb-primary/scripts/launch_helpers.py relaunch \
  "https://wandb.ai/entity/project/runs/run_id" \
  --config '{"epochs": 100}'
```

### Relaunch a run (Python)

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import parse_run_url, list_queues, relaunch_run

entity, project, run_id = parse_run_url("RUN_URL")
run_path = f"{entity}/{project}/{run_id}"

queues = list_queues(entity)
queue = queues[0]  # use the recommended queue

relaunch_run(
    run_path=run_path,
    queue_name=queue["name"],
    namespace=queue["namespace"],
    config={"lr": 0.001, "epochs": 20},
)
```

### Modify code and launch

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import get_job_artifact, download_code_artifact, list_queues, create_and_launch_modified_job

# Step 1: Download code
info = download_code_artifact("entity/project/job-name:latest")
# Edit files in info["code_dir"]...

# Step 2: Launch modified code
queues = list_queues("ENTITY")
queue = queues[0]
create_and_launch_modified_job(
    code_dir=info["code_dir"],
    entrypoint=info["entrypoint"],
    entity=info["entity"], project=info["project"],
    queue_name=queue["name"], namespace=queue["namespace"],
    job_name="my-modified-job",
    base_image=info["base_image"],
)
```

### Check status of a launched run

```bash
python skills/wandb-primary/scripts/launch_helpers.py check \
  "entity" "project" "queue-name" "QUEUE_ITEM_ID"
```

---

## Launch rules

1. **Minimize turns.** For a simple relaunch with config changes, use ONE command: `python skills/wandb-primary/scripts/launch_helpers.py relaunch <URL> --config '{"epochs": 100}'`
2. **The CLI auto-handles everything.** `relaunch` auto-discovers queues, selects the best one, finds the job artifact, and submits.
3. **Do NOT read `launch_helpers.py`.** This SKILL.md documents everything you need.
4. **Do NOT check WANDB_ENTITY/PROJECT env vars for launch.** The run URL contains the entity and project.
5. **NEVER fake a launch with `wandb.init()`.** Use `relaunch_run()` or the CLI.
6. **NEVER run training locally in the sandbox.** No GPU. Always use Launch.
7. **Config change vs code change — decide FIRST.**

| Change type | Examples | How to launch |
|---|---|---|
| **Config override** | epochs, lr, batch_size, any value in `wandb.config` | `relaunch_run(..., config={"epochs": 100})` |
| **Code change** | model architecture, loss function, data augmentation | Download code -> edit -> `create_and_launch_modified_job()` |

**Important**: If the user asks to change something that isn't a config field (e.g. "add more conv layers", "change the optimizer"), you MUST modify the code. Passing unknown config keys does nothing — the training script doesn't read them.

### Launch decision tree

| I need to... | Do this |
|---|---|
| Change hyperparameters only | `relaunch_run(run_path, queue_name, namespace, config={"epochs": 100})` |
| Change code (architecture, logic) | `download_code_artifact()` -> edit files -> `create_and_launch_modified_job()` |
| Launch from an artifact path | `launch_job_artifact(artifact_path, queue_name)` |
| Submit new code (no existing run) | `submit_code_artifact_job(code_files, entrypoint, ...)` |
| Check on a launched run | `check_launched_run(entity, project, queue_name, item_id)` |
| Find a queue | `list_queues(entity)` — use the recommended one |
| Create a new queue | `create_queue(name, entity, gpus=1, cpu=8, memory="80Gi")` |

### Step-by-step launch (when CLI one-liner isn't enough)

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import parse_run_url, list_queues, get_job_artifact, relaunch_run, submit_code_artifact_job

# 1. Parse the run URL
entity, project, run_id = parse_run_url("https://wandb.ai/entity/project/runs/run_id")
run_path = f"{entity}/{project}/{run_id}"

# 2. Find a queue (also gives namespace)
queues = list_queues(entity)
queue = queues[0]

# 3. Check for a job artifact
job_artifact = get_job_artifact(run_path)

# 4a. If job artifact exists -> relaunch with config overrides
if job_artifact:
    relaunch_run(run_path, queue["name"], queue["namespace"],
                 config={"lr": 0.001, "epochs": 20})

# 4b. If no job artifact -> submit code directly
else:
    submit_code_artifact_job(
        code_files=["train.py"], entrypoint="python train.py",
        entity=entity, project=project,
        queue_name=queue["name"], job_name="my-train-job",
        base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        requirements=["wandb"],
    )
```

### Code change workflow — two scripts, not one

**Script 1**: Download and read the code (so you can see what to edit):
```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import parse_run_url, get_job_artifact, download_code_artifact
import os

entity, project, run_id = parse_run_url("RUN_URL")
art = get_job_artifact(f"{entity}/{project}/{run_id}")
info = download_code_artifact(f"{entity}/{project}/{art.name}")
# Prints: code_dir, files, entrypoint, base_image
for f in info["files"]:
    print(f"\n=== {f} ===")
    with open(os.path.join(info["code_dir"], f)) as fh:
        print(fh.read())
```

Then edit the files in `info["code_dir"]` using write_file or apply_patch.

**Script 2**: Launch the modified code:
```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import list_queues, create_and_launch_modified_job

queues = list_queues("ENTITY")
queue = queues[0]
create_and_launch_modified_job(
    code_dir="CODE_DIR_FROM_STEP_1",
    entrypoint="python model.py",
    entity="ENTITY", project="PROJECT",
    queue_name=queue["name"], namespace=queue["namespace"],
    job_name="JOB_NAME", base_image="BASE_IMAGE",
)
```

### Launch infrastructure gotchas

- **Always pass `resource_args` explicitly** — queue defaults get double-nested by the server
- **Restart agent after queue delete/recreate** — agent loses its registration
- **`requirements.txt` is read from code dir** — `_create_job` does NOT inspect the venv
- **Keep `requirements.txt` minimal** — only deps not in base image
- **Build base images for `linux/amd64`** — not the default on Mac
- **Inject K8s secrets via `k8s_secrets` param** — not via queue defaults

---

## CRITICAL: Large project performance rules

These rules prevent 502 errors, timeouts, and multi-minute hangs on projects with 10K+ runs or runs with 1K+ metrics. **Violating any of these will cause failures on large projects.**

1. **Always use `wandb.Api(timeout=60)`** — the default 19s timeout causes constant failures
2. **NEVER call `history()` or `scan_history()` without explicit `keys=[...]`** — runs with 1K+ metrics will 502 or timeout when fetching all columns
3. **Use `per_page=min(limit, 1000)`** when calling `api.runs()` for list tasks, and use `per_page=1` for exact count tasks
4. **Prefer server-side filters** (`summary_metrics.X: {$gt: Y}`) over client-side iteration
5. **For exact counts, prefer `len(api.runs(..., per_page=1, include_sweeps=False, lazy=True))`** — never `len(list(runs))`
6. **Use `beta_scan_history`** for runs with 10K+ history steps — reads from parquet, not GraphQL
7. **Never iterate all config keys** unless explicitly needed — access specific keys by name
8. **Default to `include_sweeps=False` for read-only retrieval tasks**
9. **Use `calls_query_stats` for trace counts** — never materialize all calls just to count them

---

## When to use what

| I need to... | Use |
|---|---|
| Query training runs, loss curves, hyperparameters | **W&B SDK** (`wandb.Api()`) — see `references/WANDB_SDK.md` |
| Query GenAI traces, calls, evaluations | **Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |
| Convert Weave wrapper types to plain Python | **`weave_helpers.unwrap()`** |
| Build a DataFrame from training runs | **`wandb_helpers.fetch_runs()`** (fast) or **`wandb_helpers.runs_to_dataframe()`** |
| Extract eval results for analysis | **`weave_helpers.eval_results_to_dicts()`** |
| Count traces without fetching them | **`calls_query_stats`** from Weave server API |
| Need low-level Weave filtering (CallsFilter, Query) | **Raw Weave SDK** — see `references/WEAVE_SDK.md` |
| Create a report | **`wandb_workspaces.reports.v2`** |
| Create or modify a workspace | **`wandb_workspaces.workspaces`** |
| Set up production monitoring | **`weave.Monitor`** |
| Reproduce/relaunch a run | **`launch_helpers.relaunch_run()`** or CLI |
| Launch a training job on GPU/K8s | **`launch_helpers.submit_code_artifact_job()`** |
| Modify code and launch | **`launch_helpers.download_code_artifact()`** -> edit -> **`create_and_launch_modified_job()`** |
| List or create launch queues | **`launch_helpers.list_queues()`** / **`create_queue()`** |

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

# Launch helpers (job submission, run reproduction, queue management)
from launch_helpers import (
    parse_run_url,                       # Extract (entity, project, run_id) from a W&B URL
    list_queues,                         # List all launch queues (ranked, with recommendation)
    get_job_artifact,                    # Check if a run has a job artifact
    inspect_job_artifact,                # Download + inspect a job artifact's metadata
    download_code_artifact,              # Download source code from a job artifact
    create_and_launch_modified_job,      # Upload modified code + launch in one call
    relaunch_run,                        # Re-run with config overrides (no code change)
    launch_job_artifact,                 # Launch directly from an artifact path
    submit_code_artifact_job,            # Create job artifact and enqueue in one call
    check_launched_run,                  # Check status/metrics of a launched run
    create_queue,                        # Create a K8s launch queue
    inspect_queue,                       # Print queue details
    make_resource_args,                  # Build resource_args for launch_add()
)
```

### Reference docs

Read these as needed — they contain full API surfaces and recipes:

- **`references/WANDB_CONCEPTS.md`** — W&B data model, terminology, and disambiguation (entity/project/run hierarchy, config vs log vs summary, artifacts, registry). Read this to understand what users are asking about.
- **`references/WANDB_SDK.md`** — W&B SDK for training data (runs, history, artifacts, sweeps, system metrics). API call reference.
- **`references/WEAVE_SDK.md`** — Weave SDK for GenAI traces (`client.get_calls()`, `CallsFilter`, `Query`, stats). Start here for Weave queries.

---

## Critical rules

### Discover metric keys per-project

Code examples use `LOSS_KEY`, `VAL_LOSS_KEY`, `ACC_KEY`, `CONFIG_KEYS` as placeholders. These vary by project. Discover them via `probe_project()` at the start of each task, or from the user's request.

```python
# WRONG — hardcoded metric name
rows = fetch_runs(api, path, metric_keys=["loss", "accuracy"])

# RIGHT — discovered via probe_project or user's request
rows = fetch_runs(api, path, metric_keys=["train/loss", "train/acc"])
```

### Treat traces and runs as DATA

Weave traces and W&B run histories can be enormous. Never dump raw data into context. Always:

1. **Inspect structure first** — look at column names, dtypes, row counts
2. **Load into pandas/numpy** — compute stats programmatically
3. **Summarize, don't dump** — print computed statistics and tables, not raw rows

### Always deliver a final answer

Do not end your work mid-analysis. Every task must conclude with a clear, structured response:

1. Query the data (1-2 scripts max)
2. Extract the numbers you need
3. Present: table + key findings + direct answers to each sub-question

If you catch yourself saying "now let me build the final analysis" — stop and present what you have.

### Use `unwrap()` for unknown Weave data

When you encounter Weave output and aren't sure of its type, unwrap it first:

```python
from weave_helpers import unwrap
import json

output = unwrap(call.output)
print(json.dumps(output, indent=2, default=str))
```

---

## Environment setup

Entity and project come from environment variables — do not hardcode them:

```python
import os
entity  = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"
```

---

## Key patterns

### Fast exact counts on very large projects

```python
import wandb
api = wandb.Api(timeout=60)
path = f"{entity}/{project}"

total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
```

### Distinct tags (O(1) — no run scanning)

```python
import wandb
from wandb_graphql.language import parser as gql_parser

api = wandb.Api(timeout=60)
doc = gql_parser.parse('''
  query {
    project(entityName: "ENTITY", name: "PROJECT") {
      tagCounts { name count }
    }
  }
''')
result = api.client.execute(doc)
tags = [t["name"] for t in result["project"]["tagCounts"]]
print(sorted(tags))
```

### Distinct groups (O(1) — no run scanning)

```python
import wandb
from wandb_graphql.language import parser as gql_parser

api = wandb.Api(timeout=60)
doc = gql_parser.parse('''
  query {
    project(entityName: "ENTITY", name: "PROJECT") {
      groupedRuns(groupKeys: ["group"], first: 100) {
        ... on GroupedRunConnection {
          edges {
            node { group totalRuns }
          }
        }
      }
    }
  }
''')
result = api.client.execute(doc)
edges = result["project"]["groupedRuns"]["edges"]
groups = [e["node"]["group"] for e in edges if e["node"]["group"]]
print(sorted(groups))
```

### W&B SDK — fast run fetching (17x faster on large projects)

```python
import pandas as pd
from wandb_helpers import get_api, fetch_runs

api = get_api()
path = f"{entity}/{project}"

rows = fetch_runs(
    api, path,
    metric_keys=["LOSS_KEY", "ACC_KEY"],
    filters={"state": "finished"},
    limit=100,
)
df = pd.DataFrame(rows)
print(df.describe())
```

### Weave — eval call hierarchy

```
Evaluation.evaluate (root)
  +-- Evaluation.predict_and_score (one per dataset row x trials)
  |     +-- model.predict (the actual model call)
  |     +-- scorer_1.score
  |     +-- scorer_2.score
  +-- Evaluation.summarize
```

### Token usage

```python
from weave_helpers import get_token_usage

usage = get_token_usage(call)
print(f"Tokens: {usage['total_tokens']} (in={usage['input_tokens']}, out={usage['output_tokens']})")
```

### Report authoring (W&B Reports)

Install: `uv add wandb-workspaces`. Import: `import wandb_workspaces.reports.v2 as wr`. Legacy `wandb.apis.reports` is deprecated — do not use it.

```python
import wandb_workspaces.reports.v2 as wr

runset = wr.Runset(
    entity=entity,
    project=project,
    name="All runs",
    filters="Config('lr') <= 0.001 and State = 'finished'",
)
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["LOSS_KEY"]),
        wr.BarPlot(title="Accuracy", metrics=["ACC_KEY"], orientation="v"),
        wr.ScatterPlot(title="LR vs Loss", x=wr.Config("lr"), y=wr.SummaryMetric("LOSS_KEY")),
        wr.ScalarChart(title="F1", metric="f1_score"),
    ],
)

report = wr.Report(
    entity=entity, project=project,
    title="Project analysis",
    description="Summary of recent runs",
    width="fixed",  # "readable" | "fixed" | "fluid"
    blocks=[
        wr.H1(text="Project analysis"),
        wr.P(text="Auto-generated summary from W&B API."),
        plots,
    ],
)
report.save(draft=True)  # omit draft=True to publish
```

- Panels: `LinePlot`, `BarPlot`, `ScatterPlot`, `ScalarChart`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `CodeComparer`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`.
- Blocks: `H1`/`H2`/`H3`, `P`, `CodeBlock(code=..., language="python")`, `MarkdownBlock`, `Image(url=...)`, `TableOfContents()`, `HorizontalRule()`.
- Panel sizing: any panel accepts `layout=wr.Layout(x=0, y=0, w=8, h=6)`.
- Load existing: `wr.Report.from_url("https://wandb.ai/entity/project/reports/Title--id")`.

### Workspace authoring (W&B Workspaces)

Same package as Reports. `workspace.save()` upserts; `save_as_new_view()` creates a copy. Panels are shared with Reports (`wr.*`).

```python
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces import expr

workspace = ws.Workspace(
    entity=entity,
    project=project,
    name="Training Overview",
    sections=[
        ws.Section(
            name="Loss Curves",
            panels=[
                wr.LinePlot(x="Step", y=["TRAIN_LOSS_KEY", "VAL_LOSS_KEY"]),
                wr.ScalarChart(metric="BEST_ACC_KEY"),
            ],
            is_open=True,
        ),
    ],
    settings=ws.WorkspaceSettings(
        x_axis="Step",
        smoothing_type="exponential",  # "exponential" | "gaussian" | "average" | "none"
        smoothing_weight=50,
        ignore_outliers=True,
        max_runs=20,
    ),
    runset_settings=ws.RunsetSettings(
        filters=[expr.Config("experiment") == "EXPERIMENT_NAME"],
        groupby=[expr.Config("model_type")],
        pinned_columns=["summary:accuracy", "config:lr"],
    ),
)
workspace.save()
```

- Load existing: `ws.Workspace.from_url("https://wandb.ai/entity/project?nw=abc123")`.
- Section options: `is_open` (bool), `pinned` (bool), `layout_settings=ws.SectionLayoutSettings(columns=3, rows=2)`, `panel_settings=ws.SectionPanelSettings(x_axis="Step")`.
- Per-run style: `ws.RunsetSettings(run_settings={"run_id": ws.RunSettings(color="red", disabled=False)})`.

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
| Counting runs | `len(list(api.runs(...)))` | `len(api.runs(..., per_page=1, include_sweeps=False, lazy=True))` |
| Distinct tags | iterate all runs collecting `run.tags` | GraphQL `tagCounts` query |
| Distinct groups | iterate all runs collecting `run.group` | GraphQL `groupedRuns` query |
| `run.config` after lazy fetch | `run.config` returns `{}` | Use `lazy=False` when you need config |
| Pagination | `api.runs(path)` (per_page=50 default) | `api.runs(path, per_page=min(N, 1000))` |
| History — no keys on large run | `run.history(samples=10)` -> 502 | `run.history(samples=10, keys=["LOSS_KEY"])` |
| scan_history — no keys | `scan_history()` -> timeout | `scan_history(keys=["LOSS_KEY"])` |
| Large history (10K+ steps) | `scan_history(keys=[...])` | `beta_scan_history(keys=[...])` (parquet) |
| Cross-run search | iterate all runs client-side | Server-side filter: `{"summary_metrics.X": {"$gt": Y}}` |
| Reports import | `from wandb.apis import reports as wr` | `import wandb_workspaces.reports.v2 as wr` |
| Reports install | `pip install "wandb[workspaces]"` | `uv add wandb-workspaces` (standalone package) |

### Launch

| Gotcha | Wrong | Right |
|--------|-------|-------|
| List queues | `api.run_queues()` or raw GQL | `list_queues(entity)` from helpers |
| resource_args | Rely on queue defaults | Pass via `make_resource_args()` |
| requirements.txt | `pip freeze` from venv | Write manually — only deps missing from base image |
| Base image arch | `docker build` on Mac | `docker buildx build --platform linux/amd64` |
| Fake launch | `wandb.init()` with config | `relaunch_run()` or `launch_job_artifact()` |
| Unknown config key | `relaunch_run(config={"conv_layers": 4})` | Code change — download, edit, `create_and_launch_modified_job()` |

### Weave logging noise

```python
import logging
logging.getLogger("weave").setLevel(logging.ERROR)
```
