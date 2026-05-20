#!/usr/bin/env python3
"""Public skills benchmark CLI.

This CLI owns contributor-facing behavior in the public skills repo. It calls
the WBAF provider for task selection and live scoring instead of parsing WBAF
internals directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.skill_bench import report, wbaf_provider


def _repo_root() -> Path:
    return REPO_ROOT


def _changed_files(base_ref: str, candidate_ref: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", base_ref, candidate_ref],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def detect_changed_skills(base_ref: str, candidate_ref: str) -> list[str]:
    """Return sorted skill names changed between two git refs."""
    skills: set[str] = set()
    for path in _changed_files(base_ref, candidate_ref):
        parts = Path(path).parts
        if len(parts) >= 2 and parts[0] == "skills":
            skills.add(parts[1])
    return sorted(skills)


def cmd_detect(args: argparse.Namespace) -> int:
    skills = detect_changed_skills(args.base_ref, args.candidate_ref)
    payload = {"skills": skills}
    _write_json(payload, args.output)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    payload = wbaf_provider.call_provider(
        wbaf_root=Path(args.wbaf_root).resolve(),
        command="plan",
        skill=args.skill,
        base_ref=args.base_ref,
        candidate_ref=args.candidate_ref,
        pr_repo=args.pr_repo,
        suite=args.suite,
        harness=args.harness,
        scenarios=args.scenarios,
        output_json=output_json,
    )
    output_md.write_text(report.render(payload))
    print(output_md.read_text(), end="")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    payload = wbaf_provider.call_provider(
        wbaf_root=Path(args.wbaf_root).resolve(),
        command="run",
        skill=args.skill,
        base_ref=args.base_ref,
        candidate_ref=args.candidate_ref,
        pr_repo=args.pr_repo,
        suite=args.suite,
        harness=args.harness,
        scenarios=args.scenarios,
        output_json=output_json,
        output_md=output_md,
    )
    # Re-render locally to keep public report formatting owned by this repo.
    output_md.write_text(report.render(payload))
    print(output_md.read_text(), end="")
    return 0


def cmd_render_report(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text())
    markdown = report.render(payload)
    Path(args.output).write_text(markdown)
    print(markdown, end="")
    return 0


def _write_json(payload: dict, output: str | None) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if output:
        Path(output).write_text(text)
    print(text, end="")


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wbaf-root", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--pr-repo", default="wandb/skills")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--harness", default=None)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--output-json", default="skill-bench-result.json")
    parser.add_argument("--output-md", default="skill-bench-report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="public-skill-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect-changed-skills")
    detect.add_argument("--base-ref", default="origin/main")
    detect.add_argument("--candidate-ref", default="HEAD")
    detect.add_argument("--output", default=None)
    detect.set_defaults(func=cmd_detect)

    plan = sub.add_parser("plan")
    _add_provider_args(plan)
    plan.set_defaults(func=cmd_plan)

    live = sub.add_parser("live")
    _add_provider_args(live)
    live.set_defaults(func=cmd_live)

    render = sub.add_parser("render-report")
    render.add_argument("--input", required=True)
    render.add_argument("--output", required=True)
    render.set_defaults(func=cmd_render_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
