#!/usr/bin/env python3
"""Validate public skill package shape and basic sanitization."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

EXPECTED_SKILLS = {
    "signal-builder",
    "wandb-launch",
    "wandb-reports",
    "wandb-models",
    "weave-analysis",
}

SECRET_PATTERNS = (
    re.compile(r"/Users/[^\s`'\"]+"),
    re.compile(r"/home/[^\s`'\"]+"),
    re.compile(r"[A-Za-z]:\\[^\s`'\"]+"),
    re.compile(r"(ANTHROPIC_API_KEY|OPENAI_API_KEY|WANDB_API_KEY)\s*="),
    re.compile(r"(av-team|grpo-cuda|wandb-smle|weave-improver1|bdd100k)"),
    re.compile(r"\b(wbagent|wb_agent|WBAF|wbaf)\b"),
)


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} missing YAML frontmatter")
    raw = text.split("---", 2)[1]
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"')
    return fields


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    skills_root = root / "skills"
    offenders: list[str] = []

    actual_skills = {path.name for path in skills_root.iterdir() if path.is_dir()}
    if actual_skills != EXPECTED_SKILLS:
        offenders.append(
            f"skills set mismatch: expected {sorted(EXPECTED_SKILLS)}, "
            f"got {sorted(actual_skills)}"
        )

    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            offenders.append(f"{skill_dir}: missing SKILL.md")
            continue
        fields = _frontmatter(skill_md)
        if fields.get("name") != skill_dir.name:
            offenders.append(
                f"{skill_md}: frontmatter name {fields.get('name')!r} "
                f"does not match directory {skill_dir.name!r}"
            )
        if not fields.get("description"):
            offenders.append(f"{skill_md}: missing description")

    for path in skills_root.rglob("*"):
        if path.name == "__pycache__" or path.suffix == ".pyc":
            offenders.append(f"{path.relative_to(root)}: generated Python cache")
            continue
        if not path.is_file():
            continue
        if path.suffix in {".md", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    offenders.append(
                        f"{path.relative_to(root)}: matched {pattern.pattern}"
                    )
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                offenders.append(f"{path.relative_to(root)}: {exc}")

    if offenders:
        print("Public skill validation failed:")
        for offender in offenders:
            print(f"- {offender}")
        return 1

    print("Public skill validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
