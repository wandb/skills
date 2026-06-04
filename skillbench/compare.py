"""Compare WBAF bench rows for a base and candidate skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Classification = Literal["improved", "regressed", "same", "missing-base", "missing-candidate"]


@dataclass(frozen=True)
class Cell:
    """One task/scorer comparison."""

    task_id: str
    scorer_id: str
    base: float | None
    candidate: float | None
    delta: float | None
    classification: Classification
    must_pass: bool = False


@dataclass(frozen=True)
class Summary:
    """Aggregate comparison summary."""

    improved: int = 0
    regressed: int = 0
    same: int = 0
    missing: int = 0
    base_pass_rate: tuple[int, int] | None = None
    candidate_pass_rate: tuple[int, int] | None = None
    must_pass_regressions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonReport:
    """Full comparison payload."""

    skill: str
    base_ref: str
    candidate_ref: str
    suite: str
    agent: str
    cells: tuple[Cell, ...] = ()
    summary: Summary = field(default_factory=Summary)


LOWER_IS_BETTER = {
    "process.mutation_violation",
    "selection.cross_surface_confusion",
    "selection.irrelevant_skill_reads",
    "selection.skill_read_count",
}


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _row_score(row: dict | None) -> float | None:
    if row is None:
        return None
    if "score" in row:
        return _numeric(row["score"])
    if "passed" in row:
        return _numeric(row["passed"])
    if "pass" in row:
        return _numeric(row["pass"])
    return None


def _classify(scorer_id: str, base: float | None, candidate: float | None) -> tuple[float | None, Classification]:
    if base is None and candidate is None:
        return None, "same"
    if base is None:
        return None, "missing-base"
    if candidate is None:
        return None, "missing-candidate"
    delta = candidate - base
    if abs(delta) <= 0.01:
        return delta, "same"
    improved = delta < 0 if scorer_id in LOWER_IS_BETTER else delta > 0
    return delta, "improved" if improved else "regressed"


def _index(rows: tuple[dict, ...]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        task_id = row.get("task_id")
        scorer_id = row.get("scorer_id")
        if isinstance(task_id, str) and isinstance(scorer_id, str):
            out[(task_id, scorer_id)] = row
    return out


def _pass_rate(rows: tuple[dict, ...]) -> tuple[int, int] | None:
    per_task: dict[str, bool] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        score = _row_score(row)
        if score is None:
            continue
        per_task[task_id] = per_task.get(task_id, True) and bool(score)
    if not per_task:
        return None
    return sum(1 for passed in per_task.values() if passed), len(per_task)


def compare_results(
    *,
    skill: str,
    base_ref: str,
    candidate_ref: str,
    suite: str,
    agent: str,
    base_rows: tuple[dict, ...],
    candidate_rows: tuple[dict, ...],
) -> ComparisonReport:
    """Compare base and candidate row sets."""
    base_by_key = _index(base_rows)
    candidate_by_key = _index(candidate_rows)
    cells: list[Cell] = []
    for key in sorted(set(base_by_key) | set(candidate_by_key)):
        base_row = base_by_key.get(key)
        candidate_row = candidate_by_key.get(key)
        base = _row_score(base_row)
        candidate = _row_score(candidate_row)
        delta, classification = _classify(key[1], base, candidate)
        must_pass = bool(
            (base_row and base_row.get("must_pass"))
            or (candidate_row and candidate_row.get("must_pass"))
        )
        cells.append(
            Cell(
                task_id=key[0],
                scorer_id=key[1],
                base=base,
                candidate=candidate,
                delta=delta,
                classification=classification,
                must_pass=must_pass,
            )
        )

    must_pass_regressions = tuple(
        sorted(
            {
                cell.task_id
                for cell in cells
                if cell.must_pass and cell.base == 1.0 and cell.candidate == 0.0
            }
        )
    )
    summary = Summary(
        improved=sum(1 for cell in cells if cell.classification == "improved"),
        regressed=sum(1 for cell in cells if cell.classification == "regressed"),
        same=sum(1 for cell in cells if cell.classification == "same"),
        missing=sum(1 for cell in cells if cell.classification.startswith("missing")),
        base_pass_rate=_pass_rate(base_rows),
        candidate_pass_rate=_pass_rate(candidate_rows),
        must_pass_regressions=must_pass_regressions,
    )
    return ComparisonReport(
        skill=skill,
        base_ref=base_ref,
        candidate_ref=candidate_ref,
        suite=suite,
        agent=agent,
        cells=tuple(cells),
        summary=summary,
    )
