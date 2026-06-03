"""Bridge from public Skill Bench to WBAF `factory.run_eval`."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .candidate import CandidateBundle
from .summary_rows import WbafOutput, load_output_json, rows_from_summary
from .targets import BenchTarget, SelectSpec


@dataclass(frozen=True)
class EvalPlan:
    """A planned WBAF eval invocation."""

    wbaf_root: Path
    target: BenchTarget
    bundle: CandidateBundle
    select: SelectSpec
    output_path: Path
    adapter: str = "public-skill"
    runner_type: str = "ci"
    scorer_parallelism: int | None = None


@dataclass(frozen=True)
class EvalRun:
    """Completed WBAF subprocess run."""

    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    output: WbafOutput | None
    rows: tuple[dict, ...]


def build_command(plan: EvalPlan) -> list[str]:
    """Build a WBAF command without executing it."""
    if plan.adapter == "direct-run-eval":
        return _build_direct_run_eval_command(plan)
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "factory.public_skill_eval",
        "--suite",
        plan.target.suite,
        "--agent",
        plan.target.agent,
        "--skill-name",
        plan.bundle.skill,
        "--skill-dir",
        str(plan.bundle.bundle_dir),
        "--runner.type",
        plan.runner_type,
        "--output",
        str(plan.output_path),
    ]
    for scenario in plan.select.scenarios:
        command.extend(["--scenario", scenario])
    if plan.scorer_parallelism is not None:
        command.extend(["--scorer-parallelism", str(plan.scorer_parallelism)])
    return command


def _build_direct_run_eval_command(plan: EvalPlan) -> list[str]:
    """Build a direct `factory.run_eval` command for debug canaries."""
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "factory.run_eval",
        plan.target.suite,
        "--agent",
        plan.target.agent,
        "--agent.skills",
        "[]",
        "--agent.bundled_files",
        json.dumps(plan.bundle.bundled_files, sort_keys=True),
        "--agent.system_prompt_append",
        plan.bundle.system_prompt_append,
        "--runner.type",
        plan.runner_type,
        "--include-task-breakdown",
        "--output",
        str(plan.output_path),
    ]
    for scenario in plan.select.scenarios:
        command.extend(["--scenario", scenario])
    if plan.scorer_parallelism is not None:
        command.extend(["--scorer-parallelism", str(plan.scorer_parallelism)])
    return command


def _env_with_file(env_file: Path | None) -> dict[str, str]:
    """Load simple KEY=VALUE dotenv lines into a subprocess environment."""
    env = dict(os.environ)
    if env_file is None:
        return env
    if not env_file.is_file():
        raise FileNotFoundError(env_file)
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in env:
            env[key] = value
    return env


def run(plan: EvalPlan, *, env_file: Path | None = None, timeout_seconds: int | None = None) -> EvalRun:
    """Execute WBAF and parse rows from its generic output JSON."""
    command = build_command(plan)
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=plan.wbaf_root,
        env=_env_with_file(env_file),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds or plan.target.timeout_seconds,
    )
    output = load_output_json(plan.output_path) if plan.output_path.exists() else None
    parsed_rows = rows_from_summary(output.summary) if output else ()
    return EvalRun(
        command=tuple(command),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output=output,
        rows=parsed_rows,
    )
