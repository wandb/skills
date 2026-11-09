from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / "smoke" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SmokeHarnessTests(unittest.TestCase):
    def test_mcp_prompt_rewrite_drops_skill_wording(self) -> None:
        run = _load("run")
        for task in sorted((ROOT / "bench" / "smoke" / "tasks").glob("*.txt")):
            skill_prompt = run.prompt_for(task, "skill")
            mcp_prompt = run.prompt_for(task, "mcp")
            self.assertIn("wandb-primary skill", skill_prompt)
            self.assertNotIn("skill", mcp_prompt)
            self.assertNotIn("WANDB_API_KEY", mcp_prompt)
            self.assertIn("wandb MCP tools", mcp_prompt)

    def test_summarize_counts_scripts_and_mcp_calls(self) -> None:
        summarize = _load("summarize")
        events = [
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python3 /x/.agents/skills/wandb-primary/scripts/wandb_run_ops.py count-runs --entity e --project p", "exit_code": 0}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python3 - <<'PY'\nimport weave\nc = weave.init('e/p')\nPY", "exit_code": 0}},
            {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "probe_project_tool", "status": "completed"}},
            {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "query_wandb_tool", "status": "failed"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}},
            "not json",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "smoke-skill"
            run_dir.mkdir()
            (run_dir / "meta.txt").write_text("mode=skill\nmodel=gpt-5-mini\n")
            (run_dir / "t1.log").write_text("\n".join(e if isinstance(e, str) else json.dumps(e) for e in events))
            (run_dir / "t1.time").write_text("wall=12.5\nexit=0\n")
            (run_dir / "t1.answer.txt").write_text("long form\nShort answer: 14 runs\n")
            rows = summarize.rows_for(run_dir)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["mode"], "skill")
        self.assertEqual(row["wall_s"], "12.5")
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["commands"], 2)
        self.assertEqual(row["scripts"], "<inline weave sdk> | wandb_run_ops.py count-runs")
        self.assertEqual(row["mcp_calls"], "probe_project_toolx1")
        self.assertEqual(row["mcp_failed"], "query_wandb_tool")
        self.assertEqual(row["final_line"], "Short answer: 14 runs")


if __name__ == "__main__":
    unittest.main()
