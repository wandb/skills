#!/usr/bin/env python3
"""Call the WBAF skill-bench provider from the public skills repo."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def call_provider(
    *,
    wbaf_root: Path,
    command: str,
    skill: str,
    base_ref: str,
    candidate_ref: str,
    pr_repo: str,
    suite: str | None = None,
    harness: str | None = None,
    scenarios: str | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict:
    """Call WBAF provider and return parsed JSON."""
    args = [
        "uv",
        "run",
        "python",
        "-m",
        "developer.skill_bench.provider",
        command,
        "--skill",
        skill,
        "--base-ref",
        base_ref,
        "--candidate-ref",
        candidate_ref,
        "--pr-repo",
        pr_repo,
    ]
    if suite:
        args.extend(["--suite", suite])
    if harness:
        args.extend(["--harness", harness])
    if scenarios:
        args.extend(["--scenarios", scenarios])
    if command == "plan":
        if output_json:
            args.extend(["--output", str(output_json)])
    elif command == "run":
        if not output_json or not output_md:
            raise ValueError("run requires output_json and output_md")
        args.extend(["--output-json", str(output_json), "--output-md", str(output_md)])
    else:
        raise ValueError(f"Unsupported provider command: {command}")

    completed = subprocess.run(
        args,
        cwd=wbaf_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "WBAF provider failed\n"
            f"command: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if output_json and output_json.exists():
        return json.loads(output_json.read_text())
    return json.loads(completed.stdout)
