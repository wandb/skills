<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Building Weave signals (ClassifierMonitor + LLMAsAJudgeScorer)

Signals are binary classifiers that run automatically on new Weave traces. A signal is an
`LLMAsAJudgeScorer` prompt owned by a `ClassifierMonitor`; the scorer decides what to detect,
and the monitor decides which traces to evaluate.

### Signal rules

1. **Always import and use `signal_helpers.py`.** Do not create signal objects by hand through raw Weave APIs. The helpers handle object serialization, monitor fields, and op ref normalization.
2. **Follow the 4-phase workflow visibly:** call `explore_ops()` before writing any classifier prompt, sample matching traces, build the prompt, test the prompt, then persist after user confirmation.
3. **You can read without asking.** Explore ops, sample traces, inspect schemas, and test prompts without asking for permission. Ask only before creating/updating Weave objects.
4. **Check existing monitors visibly before writing.** Run `list_signal_monitors()` before asking for write approval and before `create_signal()` / `add_scorer_to_monitor()`. Print `monitor_count_before=...`, every existing monitor's ops/query/scorer count, and the candidate monitor filters. If no monitors exist, print that explicitly. Do not only check inside a create script; the pre-create monitor state must be visible before the write.
5. **Hard stop before writing.** A user's initial request to "create" or "set up" a signal is not write approval. After exploring, sampling, prompt testing, and printing `monitor_count_before`, stop and ask for a fresh approval with a summary of the scorer, monitor, op filter, and query filter. Do not call `create_signal()`, `add_scorer_to_monitor()`, `update_scorer_prompt()`, or `update_monitor()` until the user replies with approval after that summary. After approval, execute the write in the next tool call.
6. **Signals are forward-only.** New monitors score future traces; they do not backfill historical traces.
7. **Use full op refs for filters.** Use the `op_ref` returned by `explore_ops()`, not the short `op_name`, when passing `op_names` to `create_signal()` or `sample_traces()`.
8. **"All ops" means no op filter.** Do not enumerate every op ref to simulate all ops. For all ops/root-level/no-parent requests, pass `op_names=None` or omit `op_names`, and use `SUCCESSFUL_ROOT_TRACES_QUERY` when successful root calls are required.
9. **Use schema-specific prompt fields for single-op signals.** If monitoring one op and sampling shows stable dict fields, interpolate the narrowest verified field such as `{output[response]}` or `{output[gen_ai.completion]}`. Use generic `{output}` / `{inputs}` only for all-ops, multiple-op, non-dict, or inconsistent-schema signals.

### Phase 1: match calls

Determine the op filter, status/query filter, and fields available for prompt interpolation.
If the user named an op or said "all ops", honor that. If the request is vague, explore and
choose a sensible default; avoid open-ended questions unless there is real ambiguity.

If the user asks for all ops, all root-level ops, or traces with no parent, still call
`explore_ops(root_only=True)` before writing the prompt so schemas are inspected. Do not select an
op for the monitor itself. Sample with `root_only=True`, and later create the monitor with
`op_names=None`. Use `explore_ops()` only to understand schemas and available data; do not pass
every discovered `op_ref` as the monitor filter.

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from signal_helpers import explore_ops, sample_traces

entity = "wandb"
project = "my-weave-project"

ops = explore_ops(entity, project, limit_per_op=3, max_ops=20)
for op in ops:
    print(f"{op['op_name']} ({op['call_count']} calls)")
    print(f"  op_ref: {op['op_ref']}")
    print(f"  inputs:  {op['input_schema']}")
    print(f"  outputs: {op['output_schema']}")
```

Prefer a single op when possible because one op usually has a stable schema. `explore_ops()`
discovers both root and child ops by default; pass `root_only=True` only when the user explicitly
asks for root-level/no-parent traces. Default to successful calls. Use successful root calls only
when the user explicitly says root/root-level/no parent or asks for all root-level ops.

Sample the selected traces and verify that every interpolated field exists on the sampled calls:

```python
selected_op_ref = ops[0]["op_ref"]  # use the op_ref returned by explore_ops()
traces = sample_traces(
    entity,
    project,
    op_name=selected_op_ref,
    status="success",
    limit=5,
    root_only=False,  # use True only for root-level/no-parent requirements
)
for t in traces:
    print(f"[{t['status']}] {t['op_name']}  {t['op_ref']}")
    print(f"inputs type: {type(t['inputs']).__name__}")
    print(f"output type: {type(t['output']).__name__}")
    print(f"inputs: {str(t['inputs'])[:500]}")
    print(f"output: {str(t['output'])[:500]}")
```

After sampling, choose and print the exact interpolation fields you will use. For one op with a
stable dict schema, prefer a verified leaf field over generic `{output}` / `{inputs}`. Dotted dict
keys are still a single bracket key, for example `{output[gen_ai.completion]}` for
`{"gen_ai.completion": "..."}`. Use generic fields only for all-ops, multiple-op, non-dict, or
inconsistent-schema signals.

### Phase 2: build the prompt

Read `references/SIGNAL_PROMPT_EXAMPLES.md` before writing a new classifier prompt. Write only the
middle scoring prompt; the helper injects the trace header and JSON-output footer.

Prompt structure:

- Lead with a direct yes/no question.
- Add an `IS` section with concrete positive examples.
- Add a `NOT` section with common false positives.
- End with a conservative default.

Interpolation rules:

- `{inputs}`, `{output}`, `{op_name}`, `{status}`, `{exception}`, and `{op_source}` are provided by the header.
- Use `{output[response]}` only after sampling proves `output` is a dict with a `response` key on every targeted trace.
- If the op schema exposes a single relevant response field, use that field. Example: use `{output[gen_ai.completion]}` when `output` contains a `"gen_ai.completion"` key. Do not use generic `{output}` for a single-op signal with stable response fields.
- Escape literal JSON braces by doubling them: `{{"is_match": true}}`.
- If schemas vary, reference `{inputs}` and `{output}` rather than nested fields.

Example:

```python
prompt = """
<classifier name="FrenchResponse">
Does the assistant response in {output[response]} contain a substantive answer written primarily in French?

IS FrenchResponse:
- The assistant's natural-language answer in {output[response]} is mostly French.
- The response includes French sentences, not just a single loanword or a quoted phrase.

NOT FrenchResponse:
- User input is French but the assistant output is not.
- The output only mentions the word "French" or includes a short quoted fragment.
- Empty, missing, structured, or error outputs with no natural-language response.

Default: false. Only match when the assistant's response itself is clearly French.
</classifier>
"""
```

### Phase 3: evaluate

Test on real traces before persisting:

```python
from signal_helpers import test_signal_prompt

results = test_signal_prompt(
    entity=entity,
    project=project,
    scoring_prompt=prompt,
    traces=traces,
    limit=5,
)
for r in results:
    label = "MATCH" if r["is_match"] else "no match"
    conf = r["confidence"] if r["confidence"] is not None else 0
    print(f"{r['trace_id']}: {label} confidence={conf:.2f} reason={r['reason']}")
```

Show the tested prompt and results to the user. If the prompt looks wrong, revise and test again.
If there are no historical matching examples, a well-scoped prompt can still be created "just in
case"; test that it returns false on available negative samples.

### Phase 4: persist

Before writing, list monitors and summarize the intended write:

```python
from signal_helpers import list_signal_monitors

monitors = list_signal_monitors(entity, project)
print(f"monitor_count_before={len(monitors)}")
for m in monitors:
    print(f"{m['object_id']} ops={m['op_names']} query={m['query']} active={m['active']} scorers={len(m['scorers'])}")
print(f"candidate_ops={[selected_op_ref]}")
print(f"candidate_query=SUCCESSFUL_TRACES_QUERY")
```

Show this pre-write monitor check to the user before approval. Then stop. Do not create or update
anything in the same assistant turn as the approval request. After the user approves, re-use the
same ops/query in `create_signal()` or `add_scorer_to_monitor()`; do not silently change scope.
Ask naturally, for example: "I'll create a `FrenchResponse` scorer on successful calls for
`GenerateResponse` and attach it to `Quality-classifiers`. Want me to go ahead?"

After the user approves, create the signal:

```python
from signal_helpers import create_signal, SUCCESSFUL_TRACES_QUERY

result = create_signal(
    entity=entity,
    project=project,
    name="FrenchResponse",
    description="Flags assistant responses written primarily in French.",
    scoring_prompt=prompt,
    query=SUCCESSFUL_TRACES_QUERY,
    op_names=[selected_op_ref],
    sampling_rate=1.0,
    activate=True,
)
print(result["monitor_ref"])
print(result["scorer_ref"])

after = list_signal_monitors(entity, project)
print(f"monitor_count_after={len(after)}")
```

For user requests that explicitly target all root-level ops, omit `op_names` or pass `op_names=None`
and use `SUCCESSFUL_ROOT_TRACES_QUERY`. If you pass `op_names` without an explicit `query`,
`create_signal()` defaults to `SUCCESSFUL_TRACES_QUERY`; if you omit `op_names`, it defaults to
`SUCCESSFUL_ROOT_TRACES_QUERY`. For failed-trace signals, use `FAILED_TRACES_QUERY`. When the write
finishes, report the monitor ref, scorer ref, and that scoring applies to future traces only.

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
