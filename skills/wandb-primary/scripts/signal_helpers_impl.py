# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for building Weave signals — trace sampling, prompt testing, and signal creation.

Usage:
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    from signal_helpers import (
        explore_ops,             # List ops and their input/output field schemas
        sample_traces,           # Query and unwrap traces for inspection
        list_signal_monitors,    # List existing ClassifierMonitors in a project
        add_scorer_to_monitor,   # Add a scorer ref to an existing monitor
        create_signal,           # Full creation flow: model → scorer → monitor → publish → activate
        update_scorer_prompt,    # Update the prompt on an existing scorer
        update_monitor,          # Update fields on an existing ClassifierMonitor
        test_signal_prompt,      # Preview signal behavior on sample traces before committing
        SUCCESSFUL_TRACES_QUERY,
        SUCCESSFUL_ROOT_TRACES_QUERY,
        FAILED_TRACES_QUERY,
    )
"""

from __future__ import annotations

import json
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Typed return types
# ---------------------------------------------------------------------------


class TraceDict(TypedDict):
    """A single unwrapped trace/call."""

    id: str
    op_name: str
    op_ref: str
    status: str
    inputs: Any
    output: Any
    exception: str | None


class OpInfo(TypedDict):
    """Schema summary for a single op, returned by explore_ops()."""

    op_name: str
    op_ref: str
    call_count: int
    input_schema: dict[str, str]
    output_schema: dict[str, str]
    sample_input: Any
    sample_output: Any


class MonitorInfo(TypedDict):
    """Summary of a ClassifierMonitor, returned by list_signal_monitors()."""

    object_id: str
    name: str
    op_names: list[str]
    query: dict[str, Any] | None
    scorers: list[str]
    active: bool
    sampling_rate: float


class CreateSignalResult(TypedDict):
    """Result from create_signal()."""

    monitor_ref: str
    scorer_ref: str
    activated: bool


class AddScorerResult(TypedDict):
    """Result from add_scorer_to_monitor()."""

    monitor_ref: str
    scorer_count: int


class UpdateScorerResult(TypedDict):
    """Result from update_scorer_prompt()."""

    scorer_ref: str


class UpdateMonitorResult(TypedDict):
    """Result from update_monitor()."""

    monitor_ref: str


class TestResult(TypedDict):
    """A single test result from test_signal_prompt()."""

    trace_id: str
    is_match: bool | None
    confidence: float | None
    reason: str

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "coreweave/openai/gpt-oss-20b"

DEFAULT_PROMPT_HEADER = """\
You are a multi-classifier evaluation system. Evaluate a traced function call against multiple binary classifiers.

<trace>
<metadata>
<operation>{op_name}</operation>
<status>{status}</status>
</metadata>

<input>
{inputs}
</input>

<output>
{output}
</output>

<exception>
{exception}
</exception>

<source_code>
{op_source}
</source_code>
</trace>

Evaluate the trace above against each classifier below. Base your judgment strictly on the evidence in the trace.\
"""

DEFAULT_PROMPT_FOOTER = """\
Respond with ONLY a JSON object. No markdown fences, no explanation — just the JSON.

Use the exact classifier name from each <classifier> tag.

{{"classifiers": {{
    "ExactName1": {{"is_match": true, "confidence": 0.95, "reason": "one sentence citing specific evidence from the trace"}},
    "ExactName2": {{"is_match": false, "confidence": 0.80, "reason": "one sentence citing specific evidence from the trace"}},
  }}
}}

Rules:
- "classifiers": include an entry for EVERY classifier (match or not) with is_match, confidence, and reason.
- "is_match": true if this classifier applies (the trace exhibits this issue), false otherwise.
- "confidence": your certainty from 0.0 (uncertain) to 1.0 (certain)
- "reason": cite specific evidence from the trace (quote error messages, describe output content, reference status). Be concise (one sentence). Do NOT give generic reasons like "no evidence found".
- If multiple classifiers could apply, choose the MOST SPECIFIC ones. Only set is_match to true for the most specific matches.\
"""

# MongoDB-style query filter presets for common trace queries.
# These must use the $expr / $getField / $literal syntax and be wrapped
# in {"$expr": ...} to match what the Weave frontend expects.
SUCCESSFUL_TRACES_QUERY = {
    "$expr": {
        "$eq": [{"$getField": "summary.weave.status"}, {"$literal": "success"}]
    }
}

SUCCESSFUL_ROOT_TRACES_QUERY = {
    "$expr": {
        "$and": [
            {"$eq": [{"$getField": "parent_id"}, {"$literal": None}]},
            {"$eq": [{"$getField": "summary.weave.status"}, {"$literal": "success"}]},
        ]
    }
}

FAILED_TRACES_QUERY = {
    "$expr": {
        "$eq": [{"$getField": "summary.weave.status"}, {"$literal": "error"}]
    }
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unwrap(obj: Any) -> Any:
    """Recursively convert Weave wrapper types to plain Python dicts/lists."""
    if hasattr(obj, "keys") and hasattr(obj, "get") and not isinstance(obj, dict):
        return {k: _unwrap(obj[k]) for k in obj.keys()}
    if hasattr(obj, "__dict__") and hasattr(obj, "_val"):
        try:
            record = object.__getattribute__(obj, "_val")
            if hasattr(record, "__dict__"):
                return {
                    k: _unwrap(v)
                    for k, v in vars(record).items()
                    if not k.startswith("_")
                }
        except Exception:
            pass
    if hasattr(obj, "entity") and hasattr(obj, "_digest"):
        return str(obj)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return [_unwrap(item) for item in obj]
        except TypeError:
            pass
    return obj


def _normalize_op_ref(raw: str) -> str:
    """Ensure an op name is a full Weave ref ending in :*.

    Accepts any of these formats and normalizes to the :* wildcard form:
        ``weave:///entity/project/op/Name:hash`` → ``weave:///entity/project/op/Name:*``
        ``weave:///entity/project/op/Name:*``    → ``weave:///entity/project/op/Name:*`` (no-op)
        ``weave:///entity/project/op/Name``      → ``weave:///entity/project/op/Name:*``
        ``Name``                                  → raises ValueError
    """
    raw = str(raw).strip()
    if not raw.startswith("weave:///"):
        raise ValueError(
            f"op_names must be full Weave refs (e.g. 'weave:///entity/project/op/Name:*'), "
            f"got short name: {raw!r}. Use the 'op_ref' field from explore_ops()."
        )
    if raw.endswith(":*"):
        return raw
    # Has a version hash — replace with wildcard
    if ":" in raw.rsplit("/", 1)[-1]:
        return raw.rsplit(":", 1)[0] + ":*"
    # No version at all — append wildcard
    return raw + ":*"


def _normalize_op_names(op_names: list[str] | None) -> list[str]:
    """Normalize a list of op names to full refs with :* suffix."""
    if not op_names:
        return []
    return [_normalize_op_ref(name) for name in op_names]


def _short_op_name(raw: str) -> str:
    """Strip a full op ref to a short display name.

    ``weave:///entity/project/op/Foo.bar:hash`` → ``Foo.bar``
    """
    name = str(raw)
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if ":" in name:
        name = name.rsplit(":", 1)[0]
    return name


def _call_to_dict(call: Any) -> TraceDict:
    """Convert a single Weave call object to a plain dict."""
    raw_op = str(getattr(call, "op_name", None) or "")

    summary = getattr(call, "summary", None) or {}
    weave_meta = summary.get("weave", {}) if hasattr(summary, "get") else {}
    status = weave_meta.get("status", "unknown")

    exception = getattr(call, "exception", None)

    return {
        "id": call.id,
        "op_name": _short_op_name(raw_op),
        "op_ref": raw_op,
        "status": status,
        "inputs": _unwrap(call.inputs),
        "output": _unwrap(call.output),
        "exception": str(exception) if exception else None,
    }


def _infer_schema(obj: Any, max_depth: int = 4, _depth: int = 0) -> str:
    """Infer a human-readable type string from a Python value.

    Examples: "str", "int", "list[dict]", "dict{role: str, content: str}".
    Recurses into dicts and lists up to *max_depth* to show nested structure.
    """
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return "str"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, list):
        if not obj or _depth >= max_depth:
            return "list"
        inner = _infer_schema(obj[0], max_depth, _depth + 1)
        return f"list[{inner}]"
    if isinstance(obj, dict):
        if not obj or _depth >= max_depth:
            return "dict"
        fields = {k: _infer_schema(v, max_depth, _depth + 1) for k, v in list(obj.items())[:10]}
        inner = ", ".join(f"{k}: {v}" for k, v in fields.items())
        return "{" + inner + "}"
    return type(obj).__name__


# ---------------------------------------------------------------------------
# explore_ops — list ops and their input/output field schemas
# ---------------------------------------------------------------------------


def explore_ops(
    entity: str,
    project: str,
    limit_per_op: int = 3,
    max_ops: int = 20,
    root_only: bool = False,
) -> list[OpInfo]:
    """List ops in a project and infer their input/output field schemas.

    Queries recent calls, groups by op name, and for each op samples a few calls
    to infer the field structure. This helps you understand what data is
    available before writing a signal prompt.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        limit_per_op: Number of calls to sample per op for schema inference.
        max_ops: Maximum number of distinct ops to return.
        root_only: If True, discover only root calls. Defaults to False so child
            LLM/provider ops such as openai.responses.create are discoverable.

    Returns:
        List of dicts, each with keys:
            op_name: The operation name (e.g. "predict").
            call_count: Approximate number of calls for this op.
            input_schema: Dict mapping field names to inferred type strings.
            output_schema: Dict mapping field names to inferred type strings.
            sample_input: One unwrapped example input dict.
            sample_output: One unwrapped example output value.
    """
    import weave
    from weave.trace.weave_client import CallsFilter

    client = weave.init(f"{entity}/{project}")

    # Fetch a batch of recent calls to discover ops.
    filter_kwargs: dict[str, Any] = {}
    if root_only:
        filter_kwargs["trace_roots_only"] = True

    calls = list(client.get_calls(
        filter=CallsFilter(**filter_kwargs),
        limit=200,
    ))

    # Group by short op name, keeping one full ref per group.
    from collections import defaultdict
    by_op: dict[str, list] = defaultdict(list)
    op_refs: dict[str, str] = {}  # short_name → full ref (first seen)
    for c in calls:
        raw = str(getattr(c, "op_name", None) or "")
        short = _short_op_name(raw)
        by_op[short].append(c)
        if short not in op_refs:
            # Store the ref with a wildcard version — monitors match across
            # all versions of the op using the :* suffix.
            ref = raw
            if ":" in ref:
                # weave:///entity/project/op/Name:hash → weave:///entity/project/op/Name:*
                ref = ref.rsplit(":", 1)[0] + ":*"
            op_refs[short] = ref

    results = []
    for op_name, op_calls in sorted(by_op.items(), key=lambda x: -len(x[1])):
        if len(results) >= max_ops:
            break

        # Sample a few calls for schema inference.
        sampled = [_call_to_dict(c) for c in op_calls[:limit_per_op]]

        # Infer schemas from the first sample.
        first = sampled[0]
        input_schema = {}
        if isinstance(first["inputs"], dict):
            input_schema = {k: _infer_schema(v) for k, v in first["inputs"].items()}
        output_schema = {}
        if isinstance(first["output"], dict):
            output_schema = {k: _infer_schema(v) for k, v in first["output"].items()}
        elif first["output"] is not None:
            output_schema = {"(value)": _infer_schema(first["output"])}

        results.append({
            "op_name": op_name,
            "op_ref": op_refs.get(op_name, op_name),
            "call_count": len(op_calls),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "sample_input": first["inputs"],
            "sample_output": first["output"],
        })

    return results


# ---------------------------------------------------------------------------
# sample_traces — query and unwrap traces for inspection
# ---------------------------------------------------------------------------


def sample_traces(
    entity: str,
    project: str,
    op_name: str | None = None,
    status: str | None = None,
    limit: int = 10,
    root_only: bool | None = None,
) -> list[TraceDict]:
    """Query Weave traces and return them as plain Python dicts.

    Initializes Weave for the given entity/project, queries calls, and returns a
    list of unwrapped trace dicts.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        op_name: Optional op name substring filter (matched against op_name).
        status: Optional status filter, e.g. "success" or "error".
        limit: Maximum number of traces to return.
        root_only: If True, sample root calls only. If False, sample matching
            calls at any depth. Defaults to True when no op_name is supplied and
            False when op_name is supplied.

    Returns:
        List of dicts, each with keys: id, op_name, status, inputs, output,
        exception.
    """
    import weave
    from weave.trace.weave_client import CallsFilter

    client = weave.init(f"{entity}/{project}")

    if root_only is None:
        root_only = op_name is None

    filter_kwargs: dict[str, Any] = {}
    if root_only:
        filter_kwargs["trace_roots_only"] = True
    if op_name:
        filter_kwargs["op_names"] = [op_name]

    calls_filter = CallsFilter(**filter_kwargs)

    query: dict[str, Any] | None = None
    if status == "success":
        query = SUCCESSFUL_ROOT_TRACES_QUERY if root_only else SUCCESSFUL_TRACES_QUERY
    elif status == "error":
        query = FAILED_TRACES_QUERY

    calls_iter = client.get_calls(
        filter=calls_filter,
        query=query,
        limit=limit,
    )

    return [_call_to_dict(c) for c in calls_iter]


# ---------------------------------------------------------------------------
# list_signal_monitors — list existing ClassifierMonitors in a project
# ---------------------------------------------------------------------------


def list_signal_monitors(
    entity: str,
    project: str,
) -> list[MonitorInfo]:
    """List all ClassifierMonitors in a project.

    Use this to check for existing monitors before creating a new one.
    If a monitor with matching op_names and query already exists, add
    your scorer to it instead of creating a duplicate.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.

    Returns:
        List of dicts, each with keys:
            object_id: The monitor's object ID (e.g. "Quality-classifiers").
            name: The monitor's display name.
            op_names: List of op refs the monitor filters on.
            query: The monitor's query filter dict (or None).
            scorers: List of scorer ref strings.
            active: Whether the monitor is currently active.
            sampling_rate: Fraction of traces to score.
    """
    import weave
    from weave.trace_server.trace_server_interface import (
        ObjQueryReq,
        ObjectVersionFilter,
    )

    client = weave.init(f"{entity}/{project}")
    server = client.server
    project_id = f"{entity}/{project}"

    result = server.objs_query(ObjQueryReq(
        project_id=project_id,
        filter=ObjectVersionFilter(
            base_object_classes=["Monitor"],
            is_op=False,
            latest_only=True,
        ),
    ))

    monitors = []
    for obj in result.objs:
        val = obj.val
        if not isinstance(val, dict):
            continue
        # Only include ClassifierMonitors
        if val.get("_class_name") != "ClassifierMonitor":
            continue
        monitors.append({
            "object_id": obj.object_id,
            "name": val.get("name", obj.object_id),
            "op_names": val.get("op_names", []),
            "query": val.get("query"),
            "scorers": val.get("scorers", []),
            "active": val.get("active", False),
            "sampling_rate": val.get("sampling_rate", 1.0),
        })

    return monitors


# ---------------------------------------------------------------------------
# add_scorer_to_monitor — add a scorer ref to an existing monitor
# ---------------------------------------------------------------------------


def add_scorer_to_monitor(
    entity: str,
    project: str,
    monitor_object_id: str,
    scorer_ref: str,
) -> AddScorerResult:
    """Add a scorer to an existing ClassifierMonitor.

    Reads the monitor's current state, appends the scorer ref to its
    scorers list, and re-publishes the monitor.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        monitor_object_id: The object_id of the monitor to update
            (e.g. "Quality-classifiers"). Get this from list_signal_monitors().
        scorer_ref: Full Weave ref to the scorer to add
            (e.g. "weave:///entity/project/object/My-Signal:digest").

    Returns:
        Dict with keys: monitor_ref, scorer_count.
    """
    import weave
    from weave.trace_server.trace_server_interface import (
        ObjCreateReq,
        ObjQueryReq,
        ObjSchemaForInsert,
        ObjectVersionFilter,
    )

    client = weave.init(f"{entity}/{project}")
    server = client.server
    project_id = f"{entity}/{project}"

    # Read the current monitor state
    result = server.objs_query(ObjQueryReq(
        project_id=project_id,
        filter=ObjectVersionFilter(
            base_object_classes=["Monitor"],
            is_op=False,
            latest_only=True,
            object_ids=[monitor_object_id],
        ),
    ))

    if not result.objs:
        raise ValueError(
            f"Monitor '{monitor_object_id}' not found in {project_id}"
        )

    obj = result.objs[0]
    val = obj.val
    if not isinstance(val, dict):
        raise ValueError(
            f"Monitor '{monitor_object_id}' has unexpected val type: {type(val)}"
        )

    # Append the scorer ref
    updated_scorers = list(val.get("scorers", [])) + [scorer_ref]
    updated_val = {**val, "scorers": updated_scorers}

    # Re-publish the monitor with updated scorers
    monitor_res = server.obj_create(ObjCreateReq(
        obj=ObjSchemaForInsert(
            project_id=project_id,
            object_id=monitor_object_id,
            val=updated_val,
            builtin_object_class="ClassifierMonitor",
        ),
    ))

    monitor_ref = f"weave:///{project_id}/object/{monitor_object_id}:{monitor_res.digest}"
    return {
        "monitor_ref": monitor_ref,
        "scorer_count": len(updated_scorers),
    }


# ---------------------------------------------------------------------------
# update_scorer_prompt — update the scoring prompt on an existing scorer
# ---------------------------------------------------------------------------


def update_scorer_prompt(
    entity: str,
    project: str,
    scorer_object_id: str,
    scoring_prompt: str,
) -> UpdateScorerResult:
    """Update the scoring_prompt on an existing LLMAsAJudgeScorer.

    Reads the scorer's current state, replaces the scoring_prompt, and
    re-publishes it. The new version is automatically picked up by any
    monitor that references this scorer.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        scorer_object_id: The object_id of the scorer to update
            (e.g. "Hallucination"). Get this from list_signal_monitors()
            which includes scorer refs — the object_id is the second-to-last
            segment of the ref.
        scoring_prompt: The new scoring prompt text.

    Returns:
        Dict with keys: scorer_ref (the new version ref).
    """
    import weave
    from weave.trace_server.trace_server_interface import (
        ObjCreateReq,
        ObjQueryReq,
        ObjSchemaForInsert,
        ObjectVersionFilter,
    )

    client = weave.init(f"{entity}/{project}")
    server = client.server
    project_id = f"{entity}/{project}"

    # Read the current scorer state
    result = server.objs_query(ObjQueryReq(
        project_id=project_id,
        filter=ObjectVersionFilter(
            base_object_classes=["Scorer"],
            is_op=False,
            latest_only=True,
            object_ids=[scorer_object_id],
        ),
    ))

    if not result.objs:
        raise ValueError(
            f"Scorer '{scorer_object_id}' not found in {project_id}"
        )

    obj = result.objs[0]
    val = obj.val
    if not isinstance(val, dict):
        raise ValueError(
            f"Scorer '{scorer_object_id}' has unexpected val type: {type(val)}"
        )

    # Update the prompt and re-publish
    updated_val = {**val, "scoring_prompt": scoring_prompt}

    scorer_res = server.obj_create(ObjCreateReq(
        obj=ObjSchemaForInsert(
            project_id=project_id,
            object_id=scorer_object_id,
            val=updated_val,
            builtin_object_class="LLMAsAJudgeScorer",
        ),
    ))

    scorer_ref = f"weave:///{project_id}/object/{scorer_object_id}:{scorer_res.digest}"
    return {
        "scorer_ref": scorer_ref,
    }


# ---------------------------------------------------------------------------
# update_monitor — update fields on an existing ClassifierMonitor
# ---------------------------------------------------------------------------


_SENTINEL = object()


def update_monitor(
    entity: str,
    project: str,
    monitor_object_id: str,
    op_names: list[str] | object = _SENTINEL,
    query: dict[str, Any] | None | object = _SENTINEL,
    sampling_rate: float | object = _SENTINEL,
    active: bool | object = _SENTINEL,
    scorers: list[str] | object = _SENTINEL,
) -> UpdateMonitorResult:
    """Update fields on an existing ClassifierMonitor.

    Only fields that are explicitly passed get updated; everything else is
    preserved from the current version. Use this for changing filters,
    activating/deactivating, removing scorers, etc.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        monitor_object_id: The object_id of the monitor to update
            (e.g. "Quality-classifiers").
        op_names: Replace the monitor's op_names list.
        query: Replace the monitor's query filter (None = no filter).
        sampling_rate: Replace the sampling rate (0.0–1.0).
        active: Activate (True) or deactivate (False) the monitor.
        scorers: Replace the full scorers list (for removals — pass the
            updated list without the scorer you want to remove).

    Returns:
        Dict with keys: monitor_ref.
    """
    import weave
    from weave.trace_server.trace_server_interface import (
        ObjCreateReq,
        ObjQueryReq,
        ObjSchemaForInsert,
        ObjectVersionFilter,
    )

    client = weave.init(f"{entity}/{project}")
    server = client.server
    project_id = f"{entity}/{project}"

    # Read the current monitor state
    result = server.objs_query(ObjQueryReq(
        project_id=project_id,
        filter=ObjectVersionFilter(
            base_object_classes=["Monitor"],
            is_op=False,
            latest_only=True,
            object_ids=[monitor_object_id],
        ),
    ))

    if not result.objs:
        raise ValueError(
            f"Monitor '{monitor_object_id}' not found in {project_id}"
        )

    obj = result.objs[0]
    val = obj.val
    if not isinstance(val, dict):
        raise ValueError(
            f"Monitor '{monitor_object_id}' has unexpected val type: {type(val)}"
        )

    # Apply only the fields that were explicitly passed
    updated_val = dict(val)
    if op_names is not _SENTINEL:
        updated_val["op_names"] = _normalize_op_names(op_names)
    if query is not _SENTINEL:
        updated_val["query"] = query
    if sampling_rate is not _SENTINEL:
        updated_val["sampling_rate"] = sampling_rate
    if active is not _SENTINEL:
        updated_val["active"] = active
    if scorers is not _SENTINEL:
        updated_val["scorers"] = scorers

    # Re-publish the monitor
    monitor_res = server.obj_create(ObjCreateReq(
        obj=ObjSchemaForInsert(
            project_id=project_id,
            object_id=monitor_object_id,
            val=updated_val,
            builtin_object_class="ClassifierMonitor",
        ),
    ))

    monitor_ref = f"weave:///{project_id}/object/{monitor_object_id}:{monitor_res.digest}"
    return {
        "monitor_ref": monitor_ref,
    }


# ---------------------------------------------------------------------------
# create_signal — full creation flow: model → scorer → monitor → publish → activate
# ---------------------------------------------------------------------------


def create_signal(
    entity: str,
    project: str,
    name: str,
    scoring_prompt: str,
    description: str = "",
    query: dict[str, Any] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_header: str | None = None,
    prompt_footer: str | None = None,
    op_names: list[str] | None = None,
    sampling_rate: float = 1.0,
    activate: bool = True,
) -> CreateSignalResult:
    """Create a Weave signal (LLMStructuredCompletionModel → LLMAsAJudgeScorer → ClassifierMonitor).

    Publishes the scorer and monitor to the Weave registry, and optionally
    activates the monitor so it begins running on new traces.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        name: Human-readable name for the signal/monitor.
        scoring_prompt: The classifier-specific prompt body (inserted between
            prompt_header and prompt_footer).
        query: Optional MongoDB-style filter dict constraining which traces the
            monitor evaluates. Defaults to None (all traces).
        model_id: Model identifier for the LLM scorer. Defaults to
            DEFAULT_MODEL_ID.
        prompt_header: Override for the system prompt header. Defaults to
            DEFAULT_PROMPT_HEADER.
        prompt_footer: Override for the system prompt footer. Defaults to
            DEFAULT_PROMPT_FOOTER.
        op_names: Optional list of full op refs to restrict the monitor to.
            Use the ``op_ref`` value from ``explore_ops()`` or ``sample_traces()``,
            NOT the short display name (e.g. use
            ``"weave:///entity/project/op/Foo.bar"`` not ``"Foo.bar"``).
        sampling_rate: Fraction of matching traces to evaluate (0.0–1.0).
        activate: If True, call .activate() on the monitor after publishing.

    Returns:
        Dict with keys: monitor_ref, scorer_ref, activated.
    """
    import weave
    from weave.trace_server.trace_server_interface import ObjCreateReq, ObjSchemaForInsert

    client = weave.init(f"{entity}/{project}")
    server = client.server
    project_id = f"{entity}/{project}"

    header = prompt_header if prompt_header is not None else DEFAULT_PROMPT_HEADER
    footer = prompt_footer if prompt_footer is not None else DEFAULT_PROMPT_FOOTER

    # Sanitize name for use as object_id (replace spaces, special chars).
    scorer_id = name.replace(" ", "-")
    model_id_safe = f"{scorer_id}-model"

    # 1. Create the LLM model as a plain dict (no Python objects, no @op methods).
    #    This matches how the frontend creates models — avoids CustomWeaveType(Op)
    #    serialization that breaks online scoring.
    model_res = server.obj_create(ObjCreateReq(
        obj=ObjSchemaForInsert(
            project_id=project_id,
            object_id=model_id_safe,
            val={
                "_type": "LLMStructuredCompletionModel",
                "_class_name": "LLMStructuredCompletionModel",
                "_bases": ["LLMStructuredCompletionModel", "Model", "Object", "BaseModel"],
                "name": model_id_safe,
                "llm_model_id": model_id,
                "default_params": {
                    "response_format": "json_object",
                    "temperature": 0.0,
                },
            },
            builtin_object_class="LLMStructuredCompletionModel",
        ),
    ))
    model_ref = f"weave:///{project_id}/object/{model_id_safe}:{model_res.digest}"

    # 2. Create the scorer as a plain dict, referencing the model by ref.
    scorer_res = server.obj_create(ObjCreateReq(
        obj=ObjSchemaForInsert(
            project_id=project_id,
            object_id=scorer_id,
            val={
                "_type": "LLMAsAJudgeScorer",
                "_class_name": "LLMAsAJudgeScorer",
                "_bases": ["LLMAsAJudgeScorer", "Scorer", "Object", "BaseModel"],
                "name": scorer_id,
                "description": description or "",
                "model": model_ref,
                "scoring_prompt": scoring_prompt,
            },
            builtin_object_class="LLMAsAJudgeScorer",
        ),
    ))
    scorer_ref = f"weave:///{project_id}/object/{scorer_id}:{scorer_res.digest}"

    # 3. Create (or update) the monitor.
    # Monitor names should be short and readable, following the pattern of
    # default monitors: "Quality-classifiers", "Error-classifiers".
    if query == FAILED_TRACES_QUERY:
        monitor_id = "Error-classifiers"
    else:
        monitor_id = "Quality-classifiers"
    normalized_op_names = _normalize_op_names(op_names)
    default_query = (
        SUCCESSFUL_TRACES_QUERY if normalized_op_names else SUCCESSFUL_ROOT_TRACES_QUERY
    )
    monitor_val = {
        "_type": "ClassifierMonitor",
        "_class_name": "ClassifierMonitor",
        "_bases": ["ClassifierMonitor", "Monitor", "Object", "BaseModel"],
        "name": monitor_id,
        "description": description or "",
        "scorers": [scorer_ref],
        "query": query if query is not None else default_query,
        "prompt_header": header,
        "prompt_footer": footer,
        "op_names": normalized_op_names,
        "sampling_rate": sampling_rate,
        "active": activate,
        "is_traced": True,
    }
    monitor_res = server.obj_create(ObjCreateReq(
        obj=ObjSchemaForInsert(
            project_id=project_id,
            object_id=monitor_id,
            val=monitor_val,
            builtin_object_class="ClassifierMonitor",
        ),
    ))
    monitor_ref = f"weave:///{project_id}/object/{monitor_id}:{monitor_res.digest}"

    return {
        "monitor_ref": monitor_ref,
        "scorer_ref": scorer_ref,
        "activated": activate,
    }


# ---------------------------------------------------------------------------
# test_signal_prompt — preview signal behavior without creating persistent objects
# ---------------------------------------------------------------------------


def test_signal_prompt(
    entity: str,
    project: str,
    scoring_prompt: str,
    traces: list[TraceDict] | None = None,
    op_name: str | None = None,
    limit: int = 5,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[TestResult]:
    """Run a scoring prompt against sample traces without creating persistent objects.

    Fetches sample traces (if not provided), constructs the full prompt for
    each trace, calls the LLM via the OpenAI API directly, and returns parsed
    classifier results.

    Args:
        entity: W&B entity (user or team) name.
        project: W&B project name.
        scoring_prompt: The classifier-specific prompt body.
        traces: Pre-fetched list of trace dicts. If None, sample_traces() is
            called automatically.
        op_name: Optional op name filter for auto-sampling.
        limit: Number of traces to sample if traces is not provided.
        model_id: Model identifier string; passed as the OpenAI model parameter.

    Returns:
        List of dicts, each with keys: trace_id, is_match, confidence, reason.
        On parse error, is_match is None and reason contains the raw LLM output.
    """
    from openai import OpenAI

    if traces is None:
        traces = sample_traces(
            entity=entity,
            project=project,
            op_name=op_name,
            limit=limit,
        )

    client = OpenAI()
    results: list[dict[str, Any]] = []

    for trace in traces:
        filled_header = DEFAULT_PROMPT_HEADER.format(
            op_name=trace.get("op_name", ""),
            status=trace.get("status", ""),
            inputs=json.dumps(trace.get("inputs") or {}, indent=2, default=str),
            output=json.dumps(trace.get("output"), indent=2, default=str),
            exception=trace.get("exception") or "None",
            op_source="",
        )

        full_prompt = "\n\n".join([filled_header, scoring_prompt, DEFAULT_PROMPT_FOOTER])

        try:
            response = client.chat.completions.create(
                model=model_id,
                temperature=0.0,
                messages=[{"role": "user", "content": full_prompt}],
            )
            raw = response.choices[0].message.content or ""

            parsed = json.loads(raw)
            classifiers = parsed.get("classifiers", {})

            # Collapse multi-classifier response: take the first classifier
            # that has is_match=True, else the first entry overall.
            chosen: dict[str, Any] | None = None
            for _clf_name, clf_result in classifiers.items():
                if clf_result.get("is_match"):
                    chosen = clf_result
                    break
            if chosen is None and classifiers:
                chosen = next(iter(classifiers.values()))

            if chosen:
                results.append(
                    {
                        "trace_id": trace["id"],
                        "is_match": chosen.get("is_match"),
                        "confidence": chosen.get("confidence"),
                        "reason": chosen.get("reason"),
                    }
                )
            else:
                results.append(
                    {
                        "trace_id": trace["id"],
                        "is_match": None,
                        "confidence": None,
                        "reason": f"No classifiers in response: {raw}",
                    }
                )

        except json.JSONDecodeError:
            results.append(
                {
                    "trace_id": trace["id"],
                    "is_match": None,
                    "confidence": None,
                    "reason": f"JSON parse error. Raw output: {raw!r}",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "trace_id": trace["id"],
                    "is_match": None,
                    "confidence": None,
                    "reason": f"Error during scoring: {exc}",
                }
            )

    return results
