"""Gate decisions for public Skill Bench reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .compare import ComparisonReport

Decision = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class GateSummary:
    """Small gate summary suitable for PR comments and JSON artifacts."""

    decision: Decision
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether this gate is non-blocking."""
        return self.decision in {"pass", "warn"}


def evaluate(report: ComparisonReport) -> GateSummary:
    """Evaluate conservative public-skill gates."""
    reasons: list[str] = []
    if report.summary.must_pass_regressions:
        reasons.append(
            "must-pass regressions: "
            + ", ".join(f"`{task}`" for task in report.summary.must_pass_regressions)
        )
        return GateSummary(decision="fail", reasons=tuple(reasons))
    if report.summary.missing:
        reasons.append(f"missing comparison rows: {report.summary.missing}")
    if report.summary.regressed:
        reasons.append(f"regressed cells: {report.summary.regressed}")
    if reasons:
        return GateSummary(decision="warn", reasons=tuple(reasons))
    return GateSummary(decision="pass", reasons=("no regressions detected",))
