# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for working with Weave trace data.

Weave is the W&B product for GenAI/LLM application development — tracing,
evaluations, and scorers. These helpers convert Weave's wrapper types to plain
Python and extract structured data from calls and evals for pandas analysis.

Usage:
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    from weave_helpers import (
        unwrap,                  # Recursively convert Weave types -> plain Python
        get_token_usage,         # Extract token counts from a call's summary
        eval_results_to_dicts,   # predict_and_score calls -> list of result dicts
        pivot_solve_rate,        # Build task-level pivot table across agents
        results_summary,         # Print compact eval summary
        eval_health,             # Extract status/counts from Evaluation.evaluate calls
        eval_efficiency,         # Compute tokens-per-success across eval calls
        safe_eval_root_summary,  # Compact aggregate evidence from one Evaluation.evaluate call
        safe_eval_child_summary, # Count predict_and_score rows first; sample full payloads (capped)
        safe_project_eval_summary,  # Project eval landscape without predict_and_score payload scans
    )
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Recursive unwrap — convert Weave types to plain Python
# ---------------------------------------------------------------------------

def unwrap(obj: Any) -> Any:
    """Recursively convert Weave wrapper types to plain Python dicts/lists.

    Weave returns WeaveDict, WeaveObject, ObjectRecord, ObjectRef, and other
    wrapper types that look like Python builtins but aren't. This function
    converts everything to plain dicts/lists so you can:
    - json.dumps() the result
    - Pass it to pandas
    - Inspect unknown structures without guessing the type

    Safe to call on already-plain objects (returns them unchanged).

    Usage:
        call = client.get_call("some-id")
        output = unwrap(call.output)
        print(json.dumps(output, indent=2, default=str))
    """
    # WeaveDict -> dict (has .keys() and .get() but isn't a plain dict)
    if hasattr(obj, "keys") and hasattr(obj, "get") and not isinstance(obj, dict):
        return {k: unwrap(obj[k]) for k in obj.keys()}

    # WeaveObject / ObjectRecord -> dict via internal _val
    if hasattr(obj, "__dict__") and hasattr(obj, "_val"):
        try:
            record = object.__getattribute__(obj, "_val")
            if hasattr(record, "__dict__"):
                return {
                    k: unwrap(v)
                    for k, v in vars(record).items()
                    if not k.startswith("_")
                }
        except Exception:
            pass

    # ObjectRef -> string representation
    if hasattr(obj, "entity") and hasattr(obj, "_digest"):
        return str(obj)

    # Iterable (list, tuple, WeaveList) -> list
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return [unwrap(item) for item in obj]
        except TypeError:
            pass

    # Plain scalar — return as-is
    return obj


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------

def get_token_usage(call: Any) -> dict[str, int]:
    """Extract total token usage from a Weave call's summary.

    Works with both OpenAI-style (prompt_tokens/completion_tokens)
    and Anthropic-style (input_tokens/output_tokens) field names.

    Returns:
        {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    """
    usage = {}
    try:
        usage = call.summary.get("usage", {})
    except Exception:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    total_input = 0
    total_output = 0
    for _model, u in (usage.items() if hasattr(usage, "items") else []):
        total_input += u.get("input_tokens") or u.get("prompt_tokens") or 0
        total_output += u.get("output_tokens") or u.get("completion_tokens") or 0
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


# ---------------------------------------------------------------------------
# Safe eval access — count first, cap payloads (memory-safe on large projects)
# ---------------------------------------------------------------------------
#
# Evaluation.predict_and_score (PAS) payloads are large. Materializing every PAS
# call on a project with thousands of eval rows can exhaust memory. These helpers
# count rows server-side first, fetch a narrow projection by default, and pull
# full payloads only as an explicit, capped sample.

_PAS_NARROW_COLUMNS = [
    "id",
    "parent_id",
    "op_name",
    "display_name",
    "started_at",
    "ended_at",
    "summary",
    "exception",
]
_PAS_DEFAULT_FULL_SAMPLE_CAP = 50
_EVAL_ROOT_COLUMNS = [
    "id",
    "display_name",
    "started_at",
    "ended_at",
    "summary",
    "output",
    "exception",
]
_PAS_SAMPLE_COLUMNS = [*_PAS_NARROW_COLUMNS, "inputs", "output"]
_EVAL_OUTPUT_KEYS = [
    "score",
    "passed",
    "must_pass_ok",
    "scenario_count",
    "scenario_pass_rate",
    "task_count",
    "trial_count",
    "task_trial_count",
    "features",
    "scorer_summary",
]


def _op_ref(entity: str, project: str, op: str) -> str:
    return f"weave:///{entity}/{project}/op/{op}:*"


def _count_calls(client: Any, project_id: str, filter_dict: dict[str, Any]) -> int:
    from weave.trace_server.trace_server_interface import CallsQueryStatsReq

    return client.server.calls_query_stats(
        CallsQueryStatsReq(project_id=project_id, filter=filter_dict)
    ).count


def _limit(limit: int, count: int, cap: int | None = None) -> int:
    safe = min(max(limit, 0), count)
    return min(safe, cap) if cap is not None else safe


def _call_row(call: Any) -> dict[str, Any]:
    summary = unwrap(getattr(call, "summary", None)) or {}
    weave = summary.get("weave", {}) if isinstance(summary, dict) else {}
    started = getattr(call, "started_at", None)
    ended = getattr(call, "ended_at", None)
    duration = None
    if started and ended:
        try:
            duration = round((ended - started).total_seconds(), 3)
        except Exception:
            pass
    op_name = str(getattr(call, "op_name", "") or "")
    return {
        "id": getattr(call, "id", None),
        "parent_id": getattr(call, "parent_id", None),
        "op": op_name.rsplit("/op/", 1)[-1].rsplit(":", 1)[0],
        "display_name": getattr(call, "display_name", None),
        "started_at": str(started or ""),
        "ended_at": str(ended or ""),
        "duration_s": duration,
        "status": weave.get("status") or summary.get("status") or "unknown",
        "status_counts": weave.get("status_counts")
        or summary.get("status_counts")
        or {},
        "exception": str(getattr(call, "exception", "") or "")[:500] or None,
    }


def _call_rows(client: Any, call_filter: Any, limit: int, columns: list[str]) -> list[dict[str, Any]]:
    if not limit:
        return []
    return [
        _call_row(call)
        for call in client.get_calls(filter=call_filter, limit=limit, columns=columns)
    ]


def safe_eval_root_summary(eval_call: Any) -> dict[str, Any]:
    """Return compact aggregate evidence from one Evaluation.evaluate call."""
    output = unwrap(getattr(eval_call, "output", None)) or {}
    output = output if isinstance(output, dict) else {}
    row_summary = output.get("row_summary") or {}
    row = _call_row(eval_call)
    for key in _EVAL_OUTPUT_KEYS:
        row[key] = output.get(key)
    row["row_output"] = row_summary.get("output") or output.get("output")
    row["model_latency"] = output.get("model_latency") or row_summary.get(
        "model_latency"
    )
    return row


def safe_eval_child_summary(
    client: Any,
    entity: str,
    project: str,
    eval_call: Any,
    *,
    narrow_limit: int = 1000,
    full_sample_limit: int = 0,
    full_sample_cap: int = _PAS_DEFAULT_FULL_SAMPLE_CAP,
) -> dict[str, Any]:
    """Count PAS rows first; fetch full payloads only as an explicit capped sample.

    Returns the eval row, the predict_and_score count, a narrow projection of
    child rows (capped at ``narrow_limit``), and — only if ``full_sample_limit``
    is set — a capped sample of full-payload rows (bounded by ``full_sample_cap``).
    """
    from weave.trace.weave_client import CallsFilter

    project_id = f"{entity}/{project}"
    eval_id = getattr(eval_call, "id", None)
    if not eval_id:
        raise ValueError("eval_call must have an id")

    pas_ref = _op_ref(entity, project, "Evaluation.predict_and_score")
    pas_filter = {"op_names": [pas_ref], "parent_ids": [eval_id]}
    pas_count = _count_calls(client, project_id, pas_filter)
    call_filter = CallsFilter(op_names=[pas_ref], parent_ids=[eval_id])
    narrow_rows = _call_rows(
        client, call_filter, _limit(narrow_limit, pas_count), _PAS_NARROW_COLUMNS
    )
    sample_limit = _limit(full_sample_limit, pas_count, max(full_sample_cap, 0))
    sample_rows = _call_rows(client, call_filter, sample_limit, _PAS_SAMPLE_COLUMNS)

    return {
        "eval": _call_row(eval_call),
        "predict_and_score_count": pas_count,
        "narrow_columns": _PAS_NARROW_COLUMNS,
        "narrow_rows_fetched": len(narrow_rows),
        "narrow_rows_truncated": pas_count > len(narrow_rows),
        "narrow_rows": narrow_rows,
        "full_payload_sample_limit": sample_limit,
        "full_payload_sample_rows": sample_rows,
    }


def safe_project_eval_summary(
    client: Any,
    entity: str,
    project: str,
    *,
    eval_limit: int = 1000,
) -> dict[str, Any]:
    """Summarize a project's eval landscape without predict_and_score payload scans."""
    from weave.trace.weave_client import CallsFilter

    project_id = f"{entity}/{project}"
    eval_ref = _op_ref(entity, project, "Evaluation.evaluate")
    pas_ref = _op_ref(entity, project, "Evaluation.predict_and_score")
    eval_count = _count_calls(client, project_id, {"op_names": [eval_ref]})
    pas_count = _count_calls(client, project_id, {"op_names": [pas_ref]})
    root_limit = _limit(eval_limit, eval_count)
    roots = [
        safe_eval_root_summary(call)
        for call in client.get_calls(
            filter=CallsFilter(op_names=[eval_ref]),
            sort_by=[{"field": "started_at", "direction": "desc"}],
            limit=root_limit,
            columns=_EVAL_ROOT_COLUMNS,
        )
    ] if root_limit else []

    return {
        "evaluation_count": eval_count,
        "predict_and_score_count": pas_count,
        "eval_root_columns": _EVAL_ROOT_COLUMNS,
        "eval_roots_fetched": len(roots),
        "eval_roots_truncated": eval_count > len(roots),
        "eval_roots": roots,
    }


# ---------------------------------------------------------------------------
# Eval result extraction
# ---------------------------------------------------------------------------

def eval_results_to_dicts(
    pas_calls: list[Any],
    agent_name: str = "unknown",
) -> list[dict[str, Any]]:
    """Extract per-task results from predict_and_score calls.

    Converts Weave's nested WeaveDict/WeaveObject output into flat dicts
    suitable for pandas DataFrames.

    Args:
        pas_calls: List of Weave predict_and_score call objects.
        agent_name: Name of the agent for labeling.

    Returns:
        List of dicts with keys: task, agent, score, passed, succeeded,
        error, tool_calls, traj_len, duration_s
    """
    results = []
    for c in pas_calls:
        try:
            example = c.inputs.get("example")
            task_name = str(example.get("name")) if example else "unknown"
        except Exception:
            task_name = "unknown"

        out = c.output
        rubric_score = None
        rubric_passed = None
        succeeded = None
        error = None
        tool_calls_count = 0
        traj_len = 0

        if out:
            # Scorer results
            scores = out.get("scores") if hasattr(out, "get") else None
            if scores:
                rubric = scores.get("rubric") if hasattr(scores, "get") else None
                if rubric:
                    rubric_passed = getattr(rubric, "passed", None)
                    meta = getattr(rubric, "metadata", None)
                    if meta:
                        rubric_score = (
                            meta.get("score")
                            if hasattr(meta, "get")
                            else getattr(meta, "score", None)
                        )

            # Model output (nested: output.output)
            model_out = out.get("output") if hasattr(out, "get") else None
            if model_out and hasattr(model_out, "get"):
                succeeded = model_out.get("succeeded")
                error = model_out.get("error")
                tc = model_out.get("tool_calls")
                tool_calls_count = len(tc) if tc else 0
                traj = model_out.get("trajectory")
                traj_len = len(traj) if traj else 0

        # Duration
        duration = None
        if c.started_at and c.ended_at:
            duration = (c.ended_at - c.started_at).total_seconds()

        results.append({
            "task": task_name,
            "agent": agent_name,
            "score": rubric_score,
            "passed": rubric_passed,
            "succeeded": succeeded,
            "error": str(error)[:100] if error else None,
            "tool_calls": tool_calls_count,
            "traj_len": traj_len,
            "duration_s": round(duration, 1) if duration else None,
        })

    results.sort(key=lambda r: r.get("task", ""))
    return results


# ---------------------------------------------------------------------------
# Pivot table — solve rate per task across agents
# ---------------------------------------------------------------------------

def pivot_solve_rate(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a pivot table: one row per task, aggregated across all agents.

    Args:
        all_results: Combined results from multiple eval runs
                     (each dict must have: task, agent, score, passed).

    Returns:
        List of dicts with: task, agents_passed, agents_attempted,
        pass_rate, mean_score, best_agent, worst_agent
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_task[r["task"]].append(r)

    pivot = []
    for task in sorted(by_task):
        entries = by_task[task]
        n = len(entries)
        passed = sum(1 for e in entries if e.get("passed"))
        scores = [e["score"] for e in entries if e.get("score") is not None]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        best = max(entries, key=lambda e: e.get("score") or 0)
        worst = min(entries, key=lambda e: e.get("score") or 0)

        pivot.append({
            "task": task,
            "agents_passed": passed,
            "agents_attempted": n,
            "pass_rate": f"{passed / n:.0%}" if n > 0 else "0%",
            "mean_score": round(mean_score, 3),
            "best_agent": (
                f"{best['agent']} ({best.get('score', 0):.2f})"
                if best.get("score", 0) != worst.get("score", 0)
                else "—"
            ),
            "worst_agent": (
                f"{worst['agent']} ({worst.get('score', 0):.2f})"
                if best.get("score", 0) != worst.get("score", 0)
                else "—"
            ),
        })
    return pivot


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def results_summary(results: list[dict[str, Any]]) -> str:
    """Print a compact summary of eval results."""
    if not results:
        return "No results."

    n = len(results)
    scores = [r["score"] for r in results if r.get("score") is not None]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    passed = sum(1 for r in results if r.get("passed"))
    succeeded = sum(1 for r in results if r.get("succeeded"))
    timed_out = sum(
        1 for r in results
        if r.get("error") and "timeout" in str(r["error"]).lower()
    )

    lines = [
        f"Tasks: {n}",
        f"Mean rubric score: {mean_score:.3f}",
        f"Rubric passed: {passed}/{n} ({passed / n:.1%})",
        f"Succeeded: {succeeded}/{n} ({succeeded / n:.1%})",
    ]
    if timed_out:
        lines.append(f"Timed out: {timed_out}/{n}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval health analysis
# ---------------------------------------------------------------------------

def eval_health(eval_calls: list[Any]) -> list[dict[str, Any]]:
    """Extract health metrics from a list of Evaluation.evaluate calls.

    Args:
        eval_calls: List of Weave call objects for Evaluation.evaluate.

    Returns:
        List of dicts with: display_name, started_at, status, success_count,
        error_count, total_tokens, call_id
    """
    rows = []
    for ec in eval_calls:
        summary = {}
        try:
            summary = ec.summary or {}
        except Exception:
            pass

        weave_meta = summary.get("weave", {}) if hasattr(summary, "get") else {}
        status = weave_meta.get("status", "unknown")
        status_counts = weave_meta.get("status_counts", {})
        success_count = status_counts.get("success", 0)
        error_count = status_counts.get("error", 0)

        usage = summary.get("usage", {}) if hasattr(summary, "get") else {}
        total_tokens = 0
        for _model, u in (usage.items() if hasattr(usage, "items") else []):
            total_tokens += u.get("total_tokens", 0)

        display = getattr(ec, "display_name", None) or "unnamed"
        started = ec.started_at.strftime("%Y-%m-%d %H:%M") if ec.started_at else ""

        rows.append({
            "display_name": display,
            "started_at": started,
            "status": status,
            "success_count": success_count,
            "error_count": error_count,
            "total_tokens": total_tokens,
            "call_id": ec.id,
        })
    return rows


def eval_efficiency(eval_calls: list[Any]) -> list[dict[str, Any]]:
    """Compute cost efficiency (tokens per success) for eval calls.

    Args:
        eval_calls: List of Weave call objects for Evaluation.evaluate.

    Returns:
        List of dicts sorted by tokens_per_success (ascending = most efficient).
    """
    health = eval_health(eval_calls)
    rows = []
    for h in health:
        if h["status"] in ("running", "unknown"):
            continue
        sc = h["success_count"]
        tps = h["total_tokens"] / sc if sc > 0 else float("inf")
        rows.append({
            "display_name": h["display_name"],
            "total_tokens": h["total_tokens"],
            "success_count": sc,
            "error_count": h["error_count"],
            "tokens_per_success": round(tps),
        })
    rows.sort(key=lambda r: r["tokens_per_success"])
    return rows
