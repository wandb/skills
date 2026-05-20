#!/usr/bin/env python3
"""Render public contributor-facing skill benchmark reports."""

from __future__ import annotations

from typing import Any


def render_plan(payload: dict[str, Any]) -> str:
    """Render a plan-only provider payload as Markdown."""
    requirements = payload.get("requirements", {})
    secrets = requirements.get("secrets") or []
    scenarios = payload.get("selected_scenarios") or []
    tasks = payload.get("selected_tasks") or []
    lines = [
        f"## Skill benchmark plan: `{payload.get('skill', 'unknown')}`",
        "",
        "No live model eval was run for this check. This plan shows the WBAF "
        "task list and harness that would be used in a trusted live run.",
        "",
        f"- Suite: `{payload.get('suite', 'unknown')}`",
        f"- Harness: `{payload.get('harness', 'unknown')}`",
        f"- Base ref: `{payload.get('base_ref', 'unknown')}`",
        f"- Candidate ref: `{payload.get('candidate_ref', 'unknown')}`",
        f"- Scenarios: {', '.join(scenarios) if scenarios else 'unspecified'}",
        f"- Tasks selected: {len(tasks)}",
        "",
        "### Live-run requirements",
        "",
        f"- Trusted context required: `{requirements.get('trusted_context_required', True)}`",
        f"- Required secrets: {', '.join(f'`{secret}`' for secret in secrets) or 'none'}",
    ]
    if payload.get("estimated_live_command"):
        lines.extend(
            [
                "",
                "### Estimated WBAF provider command",
                "",
                "```bash",
                str(payload["estimated_live_command"]),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def render_live(payload: dict[str, Any]) -> str:
    """Render a live provider result as Markdown."""
    summary = payload.get("summary", {})
    outcomes = payload.get("outcomes") or []
    regressions = [
        outcome for outcome in outcomes if outcome.get("classification") == "regressed"
    ][:10]
    improvements = [
        outcome for outcome in outcomes if outcome.get("classification") == "improved"
    ][:10]
    scenarios = payload.get("selected_scenarios") or []
    lines = [
        f"## Skill benchmark: `{payload.get('skill', 'unknown')}`",
        "",
        f"- Suite: `{payload.get('suite', 'unknown')}`",
        f"- Harness: `{payload.get('harness', 'unknown')}`",
        f"- Mode: `{payload.get('mode', 'live')}`",
        f"- Base: `{payload.get('base_ref', 'unknown')}`",
        f"- Candidate: `{payload.get('candidate_ref', 'unknown')}`",
        f"- Scenarios: {', '.join(scenarios) if scenarios else 'unspecified'}",
        f"- Tasks selected: {len(payload.get('selected_tasks') or [])}",
        "",
        "### Summary",
        "",
        f"- Improved scorer outcomes: {summary.get('improved', 0)}",
        f"- Regressed scorer outcomes: {summary.get('regressed', 0)}",
        f"- Unchanged scorer outcomes: {summary.get('unchanged', 0)}",
        f"- Missing scorer outcomes: {summary.get('missing', 0)}",
        f"- Must-pass regressions: {summary.get('must_pass_regressions', 0)}",
    ]
    if regressions:
        lines.extend(["", "### Notable regressions", ""])
        for outcome in regressions:
            lines.append(_outcome_line(outcome))
    if improvements:
        lines.extend(["", "### Notable improvements", ""])
        for outcome in improvements:
            lines.append(_outcome_line(outcome))
    return "\n".join(lines) + "\n"


def render(payload: dict[str, Any]) -> str:
    """Render either a plan or live provider payload."""
    if payload.get("mode") == "plan":
        return render_plan(payload)
    return render_live(payload)


def _outcome_line(outcome: dict[str, Any]) -> str:
    return (
        f"- `{outcome.get('task_id')}` / `{outcome.get('scorer_id')}`: "
        f"{outcome.get('base_score')} -> {outcome.get('candidate_score')}"
    )
