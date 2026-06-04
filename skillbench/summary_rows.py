"""Convert WBAF `--output` summary JSON into Skill Bench rows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WbafOutput:
    """Generic WBAF output file payload."""

    agent: str
    call_id: str | None
    summary: dict[str, Any] = field(default_factory=dict)


def load_output_json(path: Path) -> WbafOutput:
    """Load a WBAF `factory.run_eval --output` JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    summary = raw.get("summary") or {}
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: summary must be a JSON object")
    return WbafOutput(
        agent=str(raw.get("agent") or ""),
        call_id=raw.get("call_id") if isinstance(raw.get("call_id"), str) else None,
        summary=summary,
    )


def _scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float)) and not isinstance(value, complex)


def _score_and_pass(value: Any) -> tuple[Any | None, bool | None]:
    if _scalar(value):
        return value, bool(value) if isinstance(value, bool) else None
    if not isinstance(value, dict):
        return None, None

    pass_flag = None
    for key in ("passed", "pass"):
        if isinstance(value.get(key), bool):
            pass_flag = value[key]
            break
    for key in ("score", "value"):
        if _scalar(value.get(key)):
            return value[key], pass_flag
    return None, pass_flag


def _task_row(
    *,
    scenario_id: str,
    task_id: str,
    task_summary: dict[str, Any],
) -> dict[str, Any] | None:
    score = task_summary.get("score")
    passed = task_summary.get("passed")
    if not _scalar(score) and not isinstance(passed, bool):
        return None
    row: dict[str, Any] = {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "scorer_id": "__task__",
        "score": score if _scalar(score) else bool(passed),
        "task_weight": float(task_summary.get("weight") or 1.0),
        "must_pass": bool(task_summary.get("must_pass", False)),
    }
    if isinstance(passed, bool):
        row["pass"] = passed
    return row


def _scorer_rows(
    *,
    scenario_id: str,
    task_id: str,
    task_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    trial_data = task_summary.get("trial_data") or []
    if not isinstance(trial_data, list):
        return []

    by_scorer: dict[str, list[tuple[Any, bool | None]]] = {}
    for trial in trial_data:
        if not isinstance(trial, dict):
            continue
        scores = trial.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        for scorer_id, scorer_value in scores.items():
            if not isinstance(scorer_id, str):
                continue
            score, passed = _score_and_pass(scorer_value)
            if score is None:
                continue
            by_scorer.setdefault(scorer_id, []).append((score, passed))

    rows: list[dict[str, Any]] = []
    for scorer_id, values in sorted(by_scorer.items()):
        score_values = [value for value, _passed in values]
        bool_values = [value for value in score_values if isinstance(value, bool)]
        numeric_values = [
            float(value)
            for value in score_values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric_values:
            score: Any = sum(numeric_values) / len(numeric_values)
        elif bool_values:
            score = all(bool_values)
        else:
            score = score_values[-1]
        pass_values = [passed for _value, passed in values if passed is not None]
        row = {
            "task_id": task_id,
            "scenario_id": scenario_id,
            "scorer_id": scorer_id,
            "score": score,
            "task_weight": float(task_summary.get("weight") or 1.0),
            "must_pass": bool(task_summary.get("must_pass", False)),
        }
        if pass_values:
            row["pass"] = all(pass_values)
        rows.append(row)
    return rows


def rows_from_summary(summary: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Extract Skill Bench rows from WBAF scenario/task summary data."""
    scenario_breakdown = summary.get("scenario_breakdown") or {}
    if not isinstance(scenario_breakdown, dict):
        return ()

    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_summary in scenario_breakdown.items():
        if not isinstance(scenario_id, str) or not isinstance(scenario_summary, dict):
            continue
        task_breakdown = scenario_summary.get("task_breakdown") or {}
        if not isinstance(task_breakdown, dict):
            continue
        for task_id, task_summary in task_breakdown.items():
            if not isinstance(task_id, str) or not isinstance(task_summary, dict):
                continue
            task_row = _task_row(
                scenario_id=scenario_id,
                task_id=task_id,
                task_summary=task_summary,
            )
            if task_row is not None:
                rows.append(task_row)
            rows.extend(
                _scorer_rows(
                    scenario_id=scenario_id,
                    task_id=task_id,
                    task_summary=task_summary,
                )
            )
    return tuple(rows)
