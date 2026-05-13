---
name: signal-builder
description: "Use for Weave signal, LLM-as-judge, classifier monitor, failed or low-quality LLM call detection, trace-field/scorer-output criteria, and monitor-ready binary scorer workflows. Use for designing, testing, creating, or updating signals and classifier monitors over Weave traces."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Signal Builder

Use this skill to build custom Weave signals and classifier monitors.

Signals are binary classifiers that run against Weave traces. A signal is an
LLM-as-judge scorer plus a classifier monitor that decides which traces to
evaluate. This is a write-capable workflow; treat it as safety-critical.

## Rules

- Always use `signal_helpers.py`; do not hand-roll monitor/scorer writes.
- Read signal examples before writing prompts.
- Sample traces and verify fields before interpolating prompt fields.
- Prefer one op name when possible because fields are more consistent.
- Test signal prompts before persisting.
- Check existing monitors before creating new ones.
- Confirm before creating or updating monitors or scorers.
- Remind users that signals are forward-only and do not backfill old traces.
- Escape literal curly braces in prompts by doubling them.
- For read-only design questions, still ground the design in bounded Weave
  inspection: ops, sample traces, scorer outputs, feedback, and field schemas.
- Use `weave-analysis` for read-only trace/eval summaries. Use this skill when
  the user asks to design, author, test, update, or review signal/monitor logic.

## Helper Script

```python
import sys
sys.path.insert(0, "skills/signal-builder/scripts")
from signal_helpers import (
    explore_ops,
    sample_traces,
    test_signal_prompt,
    list_signal_monitors,
    add_scorer_to_monitor,
    create_signal,
    update_scorer_prompt,
    update_monitor,
)
```

## Signal Builder Vs Weave Analysis

Choose this skill when the user asks for any of these:

- A Weave signal.
- A classifier monitor.
- An LLM-as-judge scorer prompt.
- Failed or low-quality LLM call detection criteria.
- How a trace-field or scorer-output rule should later become a monitor.
- Creating, updating, or testing monitor/scorer configuration.

Choose `weave-analysis` instead when the user only asks to count traces, inspect
evals, summarize scorers, list feedback, or report token/cost/model usage. Use
the most specific W&B skill for ambiguous first-hop routing.

## Rubric-Mapped Workflow

For explanation-only tasks, use the same structure even if you do not persist
anything.

| Rubric need | What to include |
| --- | --- |
| Weave grounding | Project, op names, sample trace fields, scorer outputs, feedback, status/error signals |
| Signal structure | Criteria, inputs, prompt fields, output labels/scores, monitor attachment |
| Bounded inspection | Stats first, then 3-5 sample traces; do not scan every call |
| Safety | Explicitly separate read-only design from writes and ask before mutations |

Final explanations should say: "I would inspect these Weave objects, use these
fields, define this binary label/score, test it on sampled traces, then attach it
to this existing or new classifier monitor."

Use this answer skeleton for signal-design tasks:

```text
Project context to inspect first:
- Entity/project: {entity}/{project}
- Candidate ops: {op_names}
- Sample size: 3-5 recent successful and failed traces
- Existing scorer outputs / feedback / status fields checked: {fields}

Trace fields or scorer outputs to use:
- Inputs: {input_fields}
- Outputs: {output_fields}
- Status/error/scorer fields: {status_or_scorer_fields}

Signal criteria:
- IS: {positive_conditions}
- NOT: {false_positive_exclusions}
- Default: false unless the trace visibly satisfies the criteria

Binary output:
- label or score: {label_name}
- reason: short explanation for review

Monitor usage:
- Attach this scorer to {existing_or_new_classifier_monitor}
- Filter to op(s): {op_filters}
- Applies only to new traces going forward

Safety:
- This is a design plan only unless the user confirms a write.
```

For benchmark signal-design tasks, the answer must be self-contained even when
no write is requested. Do not only describe Signal Builder generally. Ground the
plan in the specific Weave project, name the trace fields or scorer outputs you
would inspect, define positive and negative criteria, and state that monitor
attachment is forward-only.

## Phase 1: Match Calls

Discover candidate ops and fields. Recommend a single op if the user is unsure.

```python
ops = explore_ops("entity", "project")
for op in ops:
    print(f"{op['op_name']} ({op['call_count']} calls)")
    print(f"  Inputs: {op['input_schema']}")
    print(f"  Outputs: {op['output_schema']}")
```

Sample traces for the selected op and verify every interpolated field exists.

```python
traces = sample_traces("entity", "project", op_name="predict", status="success", limit=5)
for trace in traces:
    print(trace["op_name"], type(trace["output"]).__name__, str(trace["output"])[:300])
```

For failed or low-quality LLM-call signals, inspect at least:

- `summary.weave.status` or error fields.
- Inputs that contain user prompt/context.
- Outputs that contain model response or score.
- Existing scorer outputs, feedback, or human labels.
- Operation names that isolate the LLM call rather than parent orchestration.

## Phase 2: Build Prompt

Use this structure:

- Lead with a yes/no question over a specific field.
- `IS` section with positive examples.
- `NOT` section with false-positive exclusions.
- `Default: false` calibration note.
- Output contract: a binary label or score with a short reason.
- Monitor use: how the classifier monitor will run this scorer on future traces.

Interpolation rules:

- `{output}` works for any output type.
- `{output[response]}` only works if output is a dict with `response`.
- Escape literal JSON braces as `{{"is_match": true}}`.

## Phase 3: Evaluate Prompt

Test on fresh traces before persisting.

```python
results = test_signal_prompt(
    "entity",
    "project",
    scoring_prompt=prompt,
    op_name="predict",
    limit=5,
)
for row in results:
    print(row["is_match"], row["confidence"], row["reason"])
```

## Phase 4: Persist Safely

Confirm before writing. Summarize scorer name, monitor target, op filters, and
whether you will create a new monitor or add to an existing one.

```python
monitors = list_signal_monitors("entity", "project")
for monitor in monitors:
    print(monitor["object_id"], monitor["op_names"], monitor["active"])
```

If a matching monitor exists, add the scorer to it. Otherwise create a signal:

```python
result = create_signal(
    entity="entity",
    project="project",
    name="My Signal",
    scoring_prompt=prompt,
    op_names=["weave:///entity/project/op/predict:*"],
)
print(result["monitor_ref"])
```

For updates, prefer the helper surface rather than raw object mutation:

```python
updated_prompt = update_scorer_prompt("entity", "project", scorer_ref="SCORER_REF", prompt=prompt)
updated_monitor = update_monitor("entity", "project", monitor_ref="MONITOR_REF", active=True)
print(updated_prompt, updated_monitor)
```

Only run update/create helpers after explicit user confirmation.

## Final Answer Checklist

- State which op and filters the signal monitors.
- State which Weave fields or scorer outputs ground the signal.
- State the binary criteria and the label/score the signal produces.
- State which fields are interpolated.
- State whether this reuses an existing monitor or creates a new one.
- Remind the user signals apply only to new traces going forward.
- Never claim backfill unless a separate backfill workflow exists.
- If no write was requested, say this is a design plan and no monitor/scorer was created.

## References

- `references/SIGNAL_PROMPT_EXAMPLES.md` contains compact signal prompt examples.
