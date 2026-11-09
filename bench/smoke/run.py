#!/usr/bin/env python3
"""Run the smoke tasks through Codex CLI with the skill installed or with the hosted MCP server."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOSTED_MCP_URL = "https://mcp.withwandb.com/mcp"

MCP_REWRITES = (
    ("Using the wandb-primary skill, ", "Using the wandb MCP tools, "),
    (" Use the skill's helper scripts rather than writing your own queries.", ""),
    (" Use the skill's helper scripts.", ""),
    ("show which skill command you used to check", "show which MCP tool you used to check"),
    ("which card in the skill told you how to do it", "which MCP tools you used"),
    (" WANDB_API_KEY is set.", ""),
)


def prompt_for(path: Path, mode: str) -> str:
    text = path.read_text().strip()
    if mode == "mcp":
        for old, new in MCP_REWRITES:
            text = text.replace(old, new)
    return text


def install_skill(project: Path, source: Path, skill: str) -> None:
    subprocess.run(
        ["npx", "-y", "skills", "add", str(source), "--skill", skill, "--agent", "codex", "--copy", "-y"],
        cwd=project,
        check=True,
    )
    installed = project / ".agents" / "skills" / skill / "SKILL.md"
    if not installed.exists():
        raise SystemExit(f"skill install did not produce {installed}")


def setup_mcp_home(codex_home: Path, url: str) -> None:
    default_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    auth = default_home / "auth.json"
    if not auth.exists():
        raise SystemExit("codex is not logged in; run: printf '%s' \"$OPENAI_API_KEY\" | codex login --with-api-key")
    codex_home.mkdir(parents=True, exist_ok=True)
    shutil.copy(auth, codex_home / "auth.json")
    env = {**os.environ, "CODEX_HOME": str(codex_home)}
    subprocess.run(
        ["codex", "mcp", "add", "wandb", "--url", url, "--bearer-token-env-var", "WANDB_API_KEY"],
        check=True,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("skill", "mcp"), default="skill")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--skill", default="wandb-primary")
    parser.add_argument("--source", default=str(REPO), help="skills repo or branch checkout to install from")
    parser.add_argument("--mcp-url", default=HOSTED_MCP_URL)
    parser.add_argument("--tasks", default=str(HERE / "tasks"))
    parser.add_argument("--only", nargs="*", help="task file stems to run (default: all)")
    parser.add_argument("--out", required=True, help="directory for logs and answers")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        raise SystemExit("WANDB_API_KEY is not set")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    project = Path(tempfile.mkdtemp(prefix=f"smoke-{args.mode}-"))
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    env = dict(os.environ)
    if args.mode == "skill":
        install_skill(project, Path(args.source).resolve(), args.skill)
    else:
        codex_home = out / "codex_home"
        setup_mcp_home(codex_home, args.mcp_url)
        env["CODEX_HOME"] = str(codex_home)

    tasks = sorted(Path(args.tasks).glob("*.txt"))
    if args.only:
        tasks = [t for t in tasks if t.stem in set(args.only)]
    if not tasks:
        raise SystemExit("no tasks selected")

    (out / "meta.txt").write_text(f"mode={args.mode}\nmodel={args.model}\nsource={args.source}\nproject={project}\n")
    failures = 0
    for task in tasks:
        prompt = prompt_for(task, args.mode)
        log = out / f"{task.stem}.log"
        answer = out / f"{task.stem}.answer.txt"
        cmd = [
            "codex", "exec", "-m", args.model, "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox", "--ephemeral", "--json",
            "-o", str(answer), prompt,
        ]
        started = time.monotonic()
        with log.open("w") as fh:
            try:
                proc = subprocess.run(cmd, cwd=project, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
        wall = time.monotonic() - started
        (out / f"{task.stem}.time").write_text(f"wall={wall:.1f}\nexit={code}\n")
        status = "ok" if code == 0 and answer.exists() else f"exit={code}"
        failures += status != "ok"
        print(f"{task.stem}: {status} ({wall:.0f}s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
