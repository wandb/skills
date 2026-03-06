<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Weave SDK Reference — Traces, Calls, Evaluations

Reference for querying Weave GenAI trace data using the `weave` Python SDK. Covers initialization, querying calls, filtering, MongoDB-style queries, server-side stats, and reading call data.

**Key principle**: Weave traces can be large. Always use `unwrap()` to convert Weave wrapper types to plain Python, then load into pandas for analysis. Never dump raw call data into context.

Use the bundled `weave_helpers.py` for common operations (unwrap, token extraction, eval results). This reference covers the full SDK for when you need more control.

## Initialization

```python
import weave

# IMPORTANT: positional string, NOT keyword args
client = weave.init("entity/project")    # CORRECT
# weave.init(project="entity/project")   # WRONG — TypeError

import os
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

# The client object is your entry point
# client.entity, client.project, client.get_calls(), client.get_call()
```

## Querying calls

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

# Convert to pandas
df = client.get_calls(limit=500).to_pandas()
```

## Efficient counting (calls_query_stats)

For projects with 100k+ traces, use server-side stats instead of fetching all calls:

```python
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
```

## CallsFilter fields

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

## Advanced queries (MongoDB-style)

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

## Reading call data

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

# I/O — these are WeaveDict/WeaveObject, use unwrap() if needed
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

## Evaluation call hierarchy

```
Evaluation.evaluate (root)
  +-- Evaluation.predict_and_score (one per dataset row x trials)
  |     +-- model.predict (the actual model call)
  |     +-- scorer_1.score
  |     +-- scorer_2.score
  +-- Evaluation.summarize
```

## Eval status and health data locations

```python
summary = eval_call.summary

# Status
summary["weave"]["status"]                    # "success" | "error" | "descendant_error" | "running"

# Descendant call counts
summary["weave"]["status_counts"]["success"]  # int
summary["weave"]["status_counts"]["error"]    # int

# Token usage (keyed by model name)
summary["usage"]["gpt-4o"]["total_tokens"]    # int
summary["usage"]["gpt-4o"]["input_tokens"]    # int
summary["usage"]["gpt-4o"]["output_tokens"]   # int

# Cost (if include_costs=True was used)
summary["weave"]["costs"]                     # dict
```

## Import paths

These are easy to get wrong:

```python
# CallsFilter
from weave.trace.weave_client import CallsFilter

# Query (MongoDB-style)
from weave.trace_server.interface.query import Query

# Stats endpoint
from weave.trace_server.trace_server_interface import CallsQueryStatsReq
```
