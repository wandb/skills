---
name: weave-analysis
description: "Use for Weave trace, call, evaluation, scorer, token/cost, feedback, prompt-versioning, and model-usage analysis. Do not use for W&B run tables, Reports authoring, Launch, Signal Builder creation, or run-counting tasks."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Weave Trace And Evaluation Analysis

Use this skill for read-only analysis of Weave projects: traces, calls, ops,
evaluations, scorers, feedback, token usage, latency, and cost.

## Rules

- Use `WANDB_ENTITY` and `WANDB_PROJECT` from the environment or the user request.
- Use server-side stats for counts before materializing calls.
- Distinguish root traces, all calls, eval roots, and child calls.
- For trace counts, use `client.server.calls_query_stats(...)`; do not page all calls.
- Use `unwrap()` before inspecting unknown Weave outputs.
- Use helper-backed eval summaries before hand-rolling call hierarchy parsing.
- Cite project id, op filters, trace-root filters, status filters, and caveats.
- Do not use this skill for W&B run tables, Reports authoring, Launch, Signal Builder writes, or run-counting tasks. Use the most specific W&B skill for first-hop routing on ambiguous or mixed W&B prompts.

## Helper Imports

```python
import sys
sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import (
    unwrap,
    count_calls_by_op_substring,
    get_token_usage,
    eval_results_to_dicts,
    results_summary,
    pivot_solve_rate,
    eval_health,
    eval_efficiency,
)
```

## Count Total Calls And Root Traces

Use this for total trace/call counts. Root traces are calls with no parent;
total calls include every child call.

```python
import os
import weave
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
project_id = f"{entity}/{project}"
client = weave.init(project_id)

total = client.server.calls_query_stats(CallsQueryStatsReq(project_id=project_id))
roots = client.server.calls_query_stats(
    CallsQueryStatsReq(project_id=project_id, filter={"trace_roots_only": True})
)
print(f"Total calls: {total.count}")
print(f"Root traces: {roots.count}")
```

## Count Or List Ops

Use stats for counts and a bounded call query for examples. Strip Weave object
refs down to readable op names in the final answer.

If the prompt asks for calls whose op name contains a substring, use
`count_calls_by_op_substring()` for the exact server-side count. Do not sample
recent calls and multiply or infer totals.

```python
import os
import sys
import weave

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import count_calls_by_op_substring

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
project_id = f"{entity}/{project}"
client = weave.init(project_id)
count = count_calls_by_op_substring(client, project_id, ".score")
print({"project": project_id, "op_name_contains": ".score", "count": count})
```

```python
import os
import weave
from collections import Counter

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    columns=["op_name", "display_name", "started_at"],
    limit=1000,
))
counts = Counter(call.op_name.split("/op/")[-1].split(":")[0] for call in calls)
for op_name, count in counts.most_common():
    print(op_name, count)
```

If an exact substring-count task returns a much smaller number than expected,
check whether you accidentally sampled a bounded recent-call list instead of
using the server-side substring count.

## Count Error Or Successful Traces

Use `summary.weave.status` filters. For root-trace status counts, include
`trace_roots_only`.

```python
import os
import weave
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
project_id = f"{entity}/{project}"
client = weave.init(project_id)

for status in ["success", "error", "descendant_error", "running"]:
    stats = client.server.calls_query_stats(CallsQueryStatsReq(
        project_id=project_id,
        filter={"trace_roots_only": True},
        query={"$expr": {"$eq": [{"$getField": "summary.weave.status"}, status]}},
    ))
    print(status, stats.count)
```

## Latest Evaluation Summary

Use this for Evaluation.evaluate tasks and eval-result summaries.

```python
import os
import sys
import weave
from weave.trace.weave_client import CallsFilter

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import eval_results_to_dicts, results_summary

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

eval_ref = f"weave:///{entity}/{project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(
    filter=CallsFilter(op_names=[eval_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=1,
))
if not evals:
    print("No Evaluation.evaluate calls found")
else:
    eval_call = evals[0]
    pas_ref = f"weave:///{entity}/{project}/op/Evaluation.predict_and_score:*"
    children = list(client.get_calls(
        filter=CallsFilter(op_names=[pas_ref], parent_ids=[eval_call.id])
    ))
    rows = eval_results_to_dicts(children, agent_name=eval_call.display_name or "agent")
    print(results_summary(rows))
```

## Evaluation Status Counts

Evaluation roots store aggregate status counts in `summary.weave.status_counts`.
Unwrap before indexing unknown objects.

```python
import os
import sys
import weave
from weave.trace.weave_client import CallsFilter

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import unwrap

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
eval_ref = f"weave:///{entity}/{project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(filter=CallsFilter(op_names=[eval_ref]), limit=100))
for call in evals:
    summary = unwrap(call.summary or {})
    counts = summary.get("weave", {}).get("status_counts", {})
    print(call.display_name or call.id, counts)
```

## Evaluation Success Rates

Use this for tasks that ask for success/error rates per `Evaluation.evaluate`
call. The counts usually live in the evaluation call summary.

First try aggregate summary counts on each `Evaluation.evaluate` root. Only fall
back to child `Evaluation.predict_and_score` calls when the root summary lacks
usable counts, and say that you used the fallback. Always compute the rate from
the same numerator and denominator you report:

```text
success_rate = success_count / (success_count + error_count)
```

```python
import os
import sys
import weave
from weave.trace.weave_client import CallsFilter

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import unwrap

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
eval_ref = f"weave:///{entity}/{project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(
    filter=CallsFilter(op_names=[eval_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    columns=["id", "display_name", "summary", "started_at"],
    limit=100,
))

for call in evals:
    summary = unwrap(call.summary or {})
    weave_summary = summary.get("weave", {}) if isinstance(summary, dict) else {}
    status_counts = weave_summary.get("status_counts", {})
    success = (
        summary.get("success_count")
        or status_counts.get("success")
        or status_counts.get("succeeded")
        or 0
    )
    error = (
        summary.get("error_count")
        or status_counts.get("error")
        or status_counts.get("failed")
        or 0
    )
    total = success + error
    success_rate = (success / total * 100) if total else None
    print(call.display_name or call.id, success, error, success_rate)
```

Final answer requirements:

- Present one row per `Evaluation.evaluate` call.
- Include display name or call id, `success_count`, `error_count`, and success rate.
- State whether counts came from `summary.success_count` / `summary.error_count`
  or from `summary.weave.status_counts`.
- Do not infer success rates from sampled child calls unless summary counts are
  unavailable and you clearly state the fallback.

## Token, Cost, And Model Usage

Inspect bounded calls, extract token usage through helpers, and aggregate by
model name when available.

```python
import os
import sys
import weave
from collections import Counter

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import get_token_usage

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
calls = list(client.get_calls(limit=1000, columns=["op_name", "summary"]))
tokens = Counter()
for call in calls:
    usage = get_token_usage(call)
    tokens["input"] += usage.get("input_tokens", 0)
    tokens["output"] += usage.get("output_tokens", 0)
    tokens["total"] += usage.get("total_tokens", 0)
print(dict(tokens))
```

For model-usage tasks, aggregate by model string, not only total tokens. Inspect
`summary`, unwrapped `usage`, and provider-specific keys before deciding which
field names to count.

```python
import os
import sys
import weave
from collections import Counter, defaultdict

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import unwrap, get_token_usage

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
calls = list(client.get_calls(limit=1000, columns=["op_name", "summary", "output"]))

model_counts = Counter()
tokens_by_model = defaultdict(int)
for call in calls:
    summary = unwrap(call.summary or {})
    output = unwrap(call.output)
    candidates = [
        summary.get("model"),
        summary.get("model_name"),
        summary.get("llm_model"),
        summary.get("usage", {}).get("model") if isinstance(summary.get("usage"), dict) else None,
    ]
    if isinstance(output, dict):
        candidates.extend([output.get("model"), output.get("model_name")])
    model = next((m for m in candidates if m), "<unknown>")
    usage = get_token_usage(call)
    model_counts[model] += 1
    tokens_by_model[model] += usage.get("total_tokens", 0)

for model, count in model_counts.most_common():
    print(model, "calls=", count, "tokens=", tokens_by_model[model])
```

Report both the dominant model by call count and by token usage when they differ.

## Scorer Evolution Over Time

Use this for scorer inventory, scorer version history, or "how did the evaluator
change?" questions.

This is a Weave trace task. Do not search GitHub or the public web for the
project source. Query the named Weave project directly.

```python
import os
import weave
from collections import defaultdict
from weave.trace_server.trace_server_interface import CallsQueryReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
project_id = f"{entity}/{project}"
client = weave.init(project_id)
req = CallsQueryReq(
    project_id=project_id,
    query={
        "$expr": {
            "$contains": {
                "input": {"$getField": "op_name"},
                "substr": {"$literal": "score"},
                "case_insensitive": True,
            }
        }
    },
    columns=["op_name", "display_name", "started_at", "summary"],
    sort_by=[{"field": "started_at", "direction": "asc"}],
    limit=2000,
)
calls = list(client.server.calls_query(req).calls)
families = defaultdict(list)
for call in calls:
    short_name = call.op_name.split("/op/")[-1].split(":")[0]
    family = short_name.split("_v")[0].split("-v")[0]
    families[family].append((short_name, call.started_at, call.display_name))
for family, rows in families.items():
    print(family, "versions=", len({r[0] for r in rows}), "calls=", len(rows))
    for row in rows[:5]:
        print("  ", row)
```

If the project uses different scorer naming conventions, broaden the filter but
keep the final answer explicit about the pattern used.

## Complete Evaluation Workflow

Use this when asked to trace the most recent successful evaluation and describe
the child hierarchy. First find successful `Evaluation.evaluate` roots, then
query children by `parent_ids`; do not stop after only reading the prompt.

```python
import os
import weave
from weave.trace.weave_client import CallsFilter

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
project_id = f"{entity}/{project}"
client = weave.init(project_id)
eval_ref = f"weave:///{entity}/{project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(
    filter=CallsFilter(op_names=[eval_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    columns=["id", "display_name", "started_at", "ended_at", "summary"],
    limit=20,
))
successful = [
    call for call in evals
    if (call.summary or {}).get("weave", {}).get("status") == "success"
]
root = successful[0]
children = list(client.get_calls(
    filter=CallsFilter(parent_ids=[root.id]),
    columns=["id", "op_name", "display_name", "started_at", "summary"],
    limit=500,
))
grandchildren = []
for child in children[:100]:
    grandchildren.extend(client.get_calls(
        filter=CallsFilter(parent_ids=[child.id]),
        columns=["id", "op_name", "display_name", "started_at", "summary"],
        limit=100,
    ))
print(root.id, root.display_name, root.started_at, root.ended_at)
print("children", len(children), "grandchildren", len(grandchildren))
for call in children[:20]:
    print(call.started_at, call.op_name.split("/op/")[-1].split(":")[0], call.display_name)
```

Final answer requirements: parent metadata, child op families in observed order,
maximum depth inspected, data flow, and a structural summary.

## Recent Trace Inspection

Use this when a task asks for recent examples, failures, latency, token usage, or
model behavior. Keep it bounded and tabular.

```python
import os
import sys
import weave

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import unwrap, get_token_usage

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    columns=["op_name", "display_name", "started_at", "summary", "output"],
    limit=20,
))
for call in calls:
    summary = unwrap(call.summary or {})
    usage = get_token_usage(call)
    status = summary.get("weave", {}).get("status")
    print(call.started_at, call.display_name or call.op_name, status, usage)
```

Final answers should distinguish examples from exhaustive counts.

## Exact Eval And Scorer Counts

For exact counts of `Evaluation.evaluate`, `predict_and_score`, scorer calls, or
trace roots, prefer `calls_query_stats` with a specific `op_name` filter. Do not
infer exact counts from a bounded `get_calls(limit=...)` sample.

```python
import os
import weave
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
project_id = f"{entity}/{project}"
client = weave.init(project_id)

def exact_op_count(short_op_name: str) -> int:
    op_ref = f"weave:///{entity}/{project}/op/{short_op_name}:*"
    stats = client.server.calls_query_stats(
        CallsQueryStatsReq(
            project_id=project_id,
            filter={"op_names": [op_ref]},
        )
    )
    return stats.count

counts = {
    "Evaluation.evaluate": exact_op_count("Evaluation.evaluate"),
    "Evaluation.predict_and_score": exact_op_count("Evaluation.predict_and_score"),
    "scorer.score": exact_op_count("score"),
}
print(counts)
```

If the task asks for op names containing a substring, use
`count_calls_by_op_substring()` instead of listing recent calls. Never present a
bounded `get_calls(limit=...)` count as exact.

Final answer template for counts:

```text
For {entity}/{project}, I used server-side `calls_query_stats` with op-name filters:
- Evaluation.evaluate: {eval_count}
- Evaluation.predict_and_score: {pas_count}
- scorer calls: {scorer_count}

These are exact server-side stats for the stated filters, not sampled call lists.
```

## Unknown Outputs

When call inputs/outputs are `WeaveDict`, `WeaveObject`, or refs, unwrap first
and print a small JSON sample.

```python
import json
import sys

sys.path.insert(0, "skills/weave-analysis/scripts")
from weave_helpers import unwrap

print(json.dumps(unwrap(call.output), indent=2, default=str)[:4000])
```

## Final Answer Checklist

- State `entity/project`.
- State whether numbers are root traces, total calls, eval roots, or child calls.
- Include filters: op names, status, trace roots, parent ids, time range.
- For evals, distinguish `Evaluation.evaluate` roots from `predict_and_score` children.
- For counts, say whether you used server-side stats or bounded sampling.
- For exact op counts, name the op filter and say whether the count is exact.

## References

- `references/WEAVE_SDK.md` points to the latest Core Weave SDK reference source.
