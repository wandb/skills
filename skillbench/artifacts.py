"""Write Skill Bench artifact bundles."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .compare import ComparisonReport
from .gates import GateSummary


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_task_deltas(path: Path, report: ComparisonReport) -> None:
    """Write per-cell deltas as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "scorer_id",
                "base",
                "candidate",
                "delta",
                "classification",
                "must_pass",
            ],
        )
        writer.writeheader()
        for cell in report.cells:
            writer.writerow(asdict(cell))


def write_bundle(
    *,
    output_dir: Path,
    report: ComparisonReport,
    gate: GateSummary,
    markdown: str,
    manifest: dict[str, Any],
) -> None:
    """Write the standard artifact set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scorecard.md").write_text(markdown, encoding="utf-8")
    write_json(output_dir / "scorecard.json", {"report": report, "gate": gate})
    write_json(output_dir / "manifest.lock.json", manifest)
    write_task_deltas(output_dir / "task_deltas.csv", report)
