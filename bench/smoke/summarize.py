#!/usr/bin/env python3
"""Summarize `codex exec --json` logs from run.py into one CSV row per task."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_CALL = re.compile(r"scripts/([a-z_]+\.py)(?:\s+([a-z-]+))?")


def summarize_log(log: Path) -> dict:
    mcp_calls: list[str] = []
    mcp_failed: list[str] = []
    scripts: set[str] = set()
    commands = 0
    usage: dict = {}
    for line in log.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            continue
        item = event.get("item")
        if event.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "mcp_tool_call":
            name = str(item.get("tool"))
            (mcp_calls if item.get("status") == "completed" else mcp_failed).append(name)
        elif kind == "command_execution":
            commands += 1
            command = str(item.get("command", ""))
            for script, sub in SCRIPT_CALL.findall(command):
                scripts.add(f"{script} {sub}".strip())
            if "weave.init" in command or "get_calls" in command:
                scripts.add("<inline weave sdk>")
            if "wandb.Api(" in command or "get_api(" in command:
                scripts.add("<inline wandb api>")
    return {
        "input_tokens": usage.get("input_tokens", ""),
        "cached_tokens": usage.get("cached_input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "commands": commands,
        "scripts": " | ".join(sorted(scripts)),
        "mcp_calls": " | ".join(f"{n}x{mcp_calls.count(n)}" for n in sorted(set(mcp_calls))),
        "mcp_failed": " | ".join(sorted(set(mcp_failed))),
    }


def rows_for(run_dir: Path) -> list[dict]:
    meta = dict(
        line.split("=", 1) for line in (run_dir / "meta.txt").read_text().splitlines() if "=" in line
    ) if (run_dir / "meta.txt").exists() else {}
    rows = []
    for log in sorted(run_dir.glob("*.log")):
        stem = log.stem
        timing = run_dir / f"{stem}.time"
        wall = exit_code = ""
        if timing.exists():
            kv = dict(l.split("=", 1) for l in timing.read_text().splitlines() if "=" in l)
            wall, exit_code = kv.get("wall", ""), kv.get("exit", "")
        answer_path = run_dir / f"{stem}.answer.txt"
        answer = answer_path.read_text().strip() if answer_path.exists() else ""
        short = answer.rsplit("\n", 1)[-1][:300] if answer else "<no answer>"
        rows.append({
            "run": run_dir.name,
            "mode": meta.get("mode", ""),
            "model": meta.get("model", ""),
            "task": stem,
            "wall_s": wall,
            "exit": exit_code,
            **summarize_log(log),
            "final_line": short,
        })
    return rows


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: summarize.py RUN_DIR [RUN_DIR ...]", file=sys.stderr)
        return 2
    rows = [row for arg in argv for row in rows_for(Path(arg))]
    if not rows:
        print("no logs found", file=sys.stderr)
        return 1
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
