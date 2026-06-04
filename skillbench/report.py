"""Render Skill Bench reports for public PRs and step summaries."""

from __future__ import annotations

import re

from .compare import ComparisonReport
from .gates import GateSummary

FORBIDDEN_PATTERNS = (
    re.compile(r"/Users/[^\s`'\"]+"),
    re.compile(r"/home/[^\s`'\"]+"),
    re.compile(r"(ANTHROPIC_API_KEY|OPENAI_API_KEY|WANDB_API_KEY)\s*="),
)


def _fmt_rate(rate: tuple[int, int] | None) -> str:
    if not rate or rate[1] == 0:
        return "n/a"
    return f"{rate[0]}/{rate[1]} ({round(100 * rate[0] / rate[1])}%)"


def scan_for_forbidden(markdown: str) -> None:
    """Abort before publishing if a public report contains local/secrets data."""
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(markdown)
        if match:
            raise RuntimeError(f"forbidden content in report: {match.group(0)!r}")


def _redact_local_paths(text: str) -> str:
    """Remove workstation paths from public markdown."""
    text = re.sub(r"/Users/[^\s`'\"]+", "<local-path>", text)
    text = re.sub(r"/home/[^\s`'\"]+", "<local-path>", text)
    return text


def _display_command(command: list[str]) -> str:
    """Render the command shape without dumping large local file maps."""
    out: list[str] = []
    skip_next = False
    replacements = {
        "--agent.bundled_files": "<bundled-files-json>",
        "--agent.system_prompt_append": "<skillbench-public-skill-prompt>",
        "--output": "<output-json-path>",
    }
    for item in command:
        if skip_next:
            skip_next = False
            continue
        out.append(item)
        if item in replacements:
            out.append(replacements[item])
            skip_next = True
    return _redact_local_paths(" ".join(out))


def render_plan(
    *,
    skill: str,
    base_ref: str,
    candidate_ref: str,
    suite: str,
    agent: str,
    command: list[str],
) -> str:
    """Render a plan-only benchmark report."""
    lines = [
        f"## Skill Bench plan: `{skill}`",
        "",
        "No live model evaluation was run. This verifies the target, refs, "
        "candidate bundle, and WBAF command shape.",
        "",
        f"- Base ref: `{base_ref}`",
        f"- Candidate ref: `{candidate_ref}`",
        f"- Suite: `{suite}`",
        f"- Agent: `{agent}`",
        "",
        "### Planned command",
        "",
        "```bash",
        _display_command(command),
        "```",
    ]
    markdown = "\n".join(lines) + "\n"
    scan_for_forbidden(markdown)
    return markdown


def render_comparison(report: ComparisonReport, gate: GateSummary) -> str:
    """Render a concise public benchmark result."""
    lines = [
        f"## Skill Bench result: `{report.skill}`",
        "",
        f"- Decision: `{gate.decision}`",
        f"- Base ref: `{report.base_ref}`",
        f"- Candidate ref: `{report.candidate_ref}`",
        f"- Suite: `{report.suite}`",
        f"- Agent: `{report.agent}`",
        "",
        "### Summary",
        "",
        f"- Base pass rate: {_fmt_rate(report.summary.base_pass_rate)}",
        f"- Candidate pass rate: {_fmt_rate(report.summary.candidate_pass_rate)}",
        f"- Improved cells: {report.summary.improved}",
        f"- Regressed cells: {report.summary.regressed}",
        f"- Missing rows: {report.summary.missing}",
        "",
        "### Gate notes",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate.reasons)
    markdown = "\n".join(lines) + "\n"
    scan_for_forbidden(markdown)
    return markdown
