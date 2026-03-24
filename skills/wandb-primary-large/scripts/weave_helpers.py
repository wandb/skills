# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for working with Weave trace data.

Weave is the W&B product for GenAI/LLM application development — tracing,
evaluations, and scorers. These helpers convert Weave's wrapper types to plain
Python and extract structured data from calls and evals for pandas analysis.

Usage (in sandbox):
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
    )
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
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
