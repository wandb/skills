from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.skill_bench import cli, report, wbaf_provider


class SkillBenchTests(unittest.TestCase):
    def test_render_plan_names_suite_harness_and_task_count(self) -> None:
        markdown = report.render(
            {
                "mode": "plan",
                "skill": "wandb-primary",
                "suite": "wba-all",
                "harness": "claude-code-public-skill",
                "base_ref": "main",
                "candidate_ref": "HEAD",
                "selected_scenarios": ["trace-counting"],
                "selected_tasks": ["wba/trace-counting/count-traces"],
                "requirements": {
                    "trusted_context_required": True,
                    "secrets": ["WANDB_API_KEY"],
                },
            }
        )

        self.assertIn("Skill benchmark plan", markdown)
        self.assertIn("wba-all", markdown)
        self.assertIn("claude-code-public-skill", markdown)
        self.assertIn("Tasks selected: 1", markdown)

    def test_render_live_prioritizes_regressions(self) -> None:
        markdown = report.render(
            {
                "mode": "live",
                "skill": "wandb-primary",
                "suite": "wba-all",
                "harness": "claude-code-public-skill",
                "base_ref": "main",
                "candidate_ref": "HEAD",
                "selected_scenarios": ["trace-counting"],
                "selected_tasks": ["task-1"],
                "summary": {
                    "improved": 0,
                    "regressed": 1,
                    "unchanged": 0,
                    "missing": 0,
                    "must_pass_regressions": 1,
                },
                "outcomes": [
                    {
                        "task_id": "task-1",
                        "scorer_id": "must_pass",
                        "base_score": 1.0,
                        "candidate_score": 0.0,
                        "classification": "regressed",
                    }
                ],
            }
        )

        self.assertIn("Notable regressions", markdown)
        self.assertIn("must_pass", markdown)

    def test_detect_changed_skills_from_git_diff(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="skills/wandb-primary/SKILL.md\nREADME.md\n",
            stderr="",
        )
        with patch("scripts.skill_bench.cli.subprocess.run", return_value=fake):
            skills = cli.detect_changed_skills("base", "head")

        self.assertEqual(skills, ["wandb-primary"])

    def test_call_provider_reads_output_json(self) -> None:
        with TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "plan.json"
            output.write_text(json.dumps({"mode": "plan"}))
            fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            with patch(
                "scripts.skill_bench.wbaf_provider.subprocess.run",
                return_value=fake,
            ) as run:
                payload = wbaf_provider.call_provider(
                    wbaf_root=Path(tempdir),
                    command="plan",
                    skill="wandb-primary",
                    base_ref="main",
                    candidate_ref="HEAD",
                    pr_repo="wandb/skills",
                    output_json=output,
                )

        self.assertEqual(payload["mode"], "plan")
        self.assertIn("developer.skill_bench.provider", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
