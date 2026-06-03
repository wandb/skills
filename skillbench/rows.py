"""Parse WBAF bench row output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_BENCH_RESULTS_RE = re.compile(
    r"<<BENCH-RESULTS>>\s*(?P<payload>\{.*?\})\s*<</BENCH-RESULTS>>",
    re.DOTALL,
)


@dataclass(frozen=True)
class BenchResults:
    """Structured rows and metadata emitted by WBAF `factory.run_eval`."""

    rows: tuple[dict[str, Any], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    present: bool = False


def parse_bench_results(stdout: str) -> BenchResults:
    """Parse the `<<BENCH-RESULTS>>` JSON block from stdout."""
    match = _BENCH_RESULTS_RE.search(stdout)
    if not match:
        return BenchResults()
    payload = json.loads(match.group("payload"))
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError("bench results rows must be a list")
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    return BenchResults(
        rows=tuple(row for row in rows if isinstance(row, dict)),
        meta=meta,
        present=True,
    )
