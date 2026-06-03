"""Generate shields.io badge payloads from Skill Bench summaries."""

from __future__ import annotations

from .compare import ComparisonReport


def badge_payload(report: ComparisonReport, *, label: str) -> dict[str, str]:
    """Return a shields.io endpoint payload for a comparison report."""
    rate = report.summary.candidate_pass_rate
    if not rate or rate[1] == 0:
        return {"schemaVersion": 1, "label": label, "message": "no data", "color": "lightgrey"}
    pct = round(100 * rate[0] / rate[1])
    if pct >= 90:
        color = "brightgreen"
    elif pct >= 75:
        color = "yellow"
    else:
        color = "red"
    return {
        "schemaVersion": 1,
        "label": label,
        "message": f"{rate[0]}/{rate[1]} ({pct}%)",
        "color": color,
    }
