from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skillbench import candidate, rows, targets, wbaf_eval


class SkillBenchTests(unittest.TestCase):
    def test_load_targets(self) -> None:
        config = targets.load(Path("bench/targets.toml"))

        target = config.for_skill("wandb-primary")
        self.assertEqual(target.suite, "data/evals/wba-all.yaml")
        self.assertEqual(target.agent, "claude-code-public-skill")
        self.assertIn("trace-counting", target.select.scenarios)
        self.assertEqual(config.profile("smoke").trials, 1)
        canary = targets.apply_profile(target, config.profile("canary"))
        self.assertEqual(canary.suite, "data/evals/harness-smoke.yaml")
        self.assertEqual(canary.timeout_seconds, 900)
        self.assertEqual(canary.select.scenarios, ())

    def test_changed_skill_detection_and_policy(self) -> None:
        files = (
            "skills/wandb-primary/SKILL.md",
            "skills/wandb-primary/scripts/helper.py",
            "bench/targets.toml",
        )

        policy = candidate.evaluate_path_policy(files)

        self.assertTrue(policy.ok)
        self.assertEqual(policy.changed_skills, ("wandb-primary",))
        self.assertEqual(
            policy.script_changes,
            ("skills/wandb-primary/scripts/helper.py",),
        )

    def test_path_policy_blocks_unknown_roots(self) -> None:
        policy = candidate.evaluate_path_policy(("unknown/file.txt",))

        self.assertFalse(policy.ok)
        self.assertEqual(policy.blocked_paths, ("unknown/file.txt",))

    def test_build_bundle_maps_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            skill_dir = root / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\n---\n",
                encoding="utf-8",
            )
            (skill_dir / "references").mkdir()
            (skill_dir / "references" / "REF.md").write_text(
                "ref",
                encoding="utf-8",
            )

            bundle = candidate.build_bundle(
                repo_root=root,
                skill="demo",
                output_dir=Path(tmp) / "bundle",
            )

            self.assertIn("skills/demo/SKILL.md", bundle.bundled_files)
            self.assertIn(
                "skills/demo/references/REF.md",
                bundle.bundled_files,
            )
            self.assertIn("skills/demo/SKILL.md", bundle.system_prompt_append)

    def test_parse_bench_results(self) -> None:
        parsed = rows.parse_bench_results(
            "noise\n<<BENCH-RESULTS>>\n"
            '{"rows":[{"task_id":"t","scorer_id":"s","score":1}],'
            '"meta":{"agent":"a"}}\n'
            "<</BENCH-RESULTS>>\n"
        )

        self.assertTrue(parsed.present)
        self.assertEqual(parsed.rows[0]["task_id"], "t")
        self.assertEqual(parsed.meta["agent"], "a")

    def test_wbaf_command_uses_bundle_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "repo" / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\n---\n",
                encoding="utf-8",
            )
            bundle = candidate.build_bundle(
                repo_root=root / "repo",
                skill="demo",
                output_dir=root / "bundle",
            )
            target = targets.BenchTarget(
                publish_name="demo",
                suite="data/evals/demo.yaml",
                agent="claude-code-public-skill",
            )

            command = wbaf_eval.build_command(
                wbaf_eval.EvalPlan(
                    wbaf_root=root / "wbaf",
                    target=target,
                    bundle=bundle,
                    select=targets.SelectSpec(scenarios=("smoke",)),
                )
            )

            self.assertIn("--agent.bundled_files", command)
            self.assertIn("--agent.system_prompt_append", command)
            self.assertIn("--scenario", command)
            self.assertNotIn("data/skills", " ".join(command))


if __name__ == "__main__":
    unittest.main()
