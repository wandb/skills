"""Command line interface for public Skill Bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import artifacts, candidate, checkout, compare, gates, report, targets, wbaf_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = REPO_ROOT / "bench" / "targets.toml"
DEFAULT_WORKDIR = REPO_ROOT / ".tmp" / "skillbench"


def _write_text(path: Path | None, text: str) -> None:
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path | None, payload: object) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text(path, text)


def cmd_detect(args: argparse.Namespace) -> int:
    files = candidate.changed_files(Path(args.repo_root), args.base_ref, args.candidate_ref)
    policy = candidate.evaluate_path_policy(files)
    _write_json(
        Path(args.output) if args.output else None,
        {
            "files": list(policy.files),
            "skills": list(policy.changed_skills),
            "script_changes": list(policy.script_changes),
            "blocked_paths": list(policy.blocked_paths),
            "ok": policy.ok,
        },
    )
    return 0 if policy.ok else 1


def _load_target(args: argparse.Namespace) -> tuple[targets.BenchTarget, targets.SelectSpec]:
    config = targets.load(Path(args.targets))
    profile = config.profile(args.profile)
    target = targets.apply_profile(config.for_skill(args.skill), profile)
    return target, target.select


def _prepare_checkouts(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return base and candidate repo roots, optionally materializing refs."""
    if not args.prepare_checkouts:
        base_root = Path(args.base_repo_root) if args.base_repo_root else Path(args.repo_root)
        return base_root, Path(args.repo_root)
    checkout_root = Path(args.workdir) / "checkouts"
    source_repo = args.source_repo or args.repo_root
    base_root = checkout.materialize_ref(
        source=source_repo,
        ref=args.base_ref,
        output_dir=checkout_root / "base",
    )
    candidate_root = checkout.materialize_ref(
        source=source_repo,
        ref=args.candidate_ref,
        output_dir=checkout_root / "candidate",
    )
    return base_root, candidate_root


def _build_plan(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    bundle_label: str = "candidate",
) -> tuple[wbaf_eval.EvalPlan, list[str]]:
    target, select = _load_target(args)
    bundle = candidate.build_bundle(
        repo_root=repo_root or Path(args.repo_root),
        skill=args.skill,
        output_dir=Path(args.workdir) / bundle_label,
    )
    output_path = (Path(args.output_dir) / f"{bundle_label}.wbaf-output.json").resolve()
    plan = wbaf_eval.EvalPlan(
        wbaf_root=Path(args.wbaf_root),
        target=target,
        bundle=bundle,
        select=select,
        output_path=output_path,
        scorer_parallelism=args.scorer_parallelism,
    )
    return plan, wbaf_eval.build_command(plan)


def cmd_plan(args: argparse.Namespace) -> int:
    plan, command = _build_plan(args)
    markdown = report.render_plan(
        skill=args.skill,
        base_ref=args.base_ref,
        candidate_ref=args.candidate_ref,
        suite=plan.target.suite,
        agent=plan.target.agent,
        command=command,
    )
    manifest = {
        "skill": args.skill,
        "base_ref": args.base_ref,
        "candidate_ref": args.candidate_ref,
        "suite": plan.target.suite,
        "agent": plan.target.agent,
        "command": command,
        "bundle_dir": str(plan.bundle.bundle_dir),
        "bundled_files": plan.bundle.bundled_files,
    }
    output_dir = Path(args.output_dir)
    artifacts.write_json(output_dir / "manifest.lock.json", manifest)
    (output_dir / "scorecard.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_repo_root, candidate_repo_root = _prepare_checkouts(args)
    base_plan, _ = _build_plan(args, repo_root=base_repo_root, bundle_label="base")
    candidate_plan, command = _build_plan(
        args,
        repo_root=candidate_repo_root,
        bundle_label="candidate",
    )
    if base_plan.bundle.bundle_dir == candidate_plan.bundle.bundle_dir:
        raise RuntimeError("base and candidate bundle directories must differ")

    base_run = wbaf_eval.run(base_plan, env_file=Path(args.env_file) if args.env_file else None)
    candidate_run = wbaf_eval.run(
        candidate_plan,
        env_file=Path(args.env_file) if args.env_file else None,
    )
    (output_dir / "base.stdout.txt").write_text(base_run.stdout, encoding="utf-8")
    (output_dir / "base.stderr.txt").write_text(base_run.stderr, encoding="utf-8")
    (output_dir / "candidate.stdout.txt").write_text(
        candidate_run.stdout,
        encoding="utf-8",
    )
    (output_dir / "candidate.stderr.txt").write_text(
        candidate_run.stderr,
        encoding="utf-8",
    )
    missing_outputs: list[str] = []
    if base_run.output is None or not base_run.rows:
        missing_outputs.append("base")
    if candidate_run.output is None or not candidate_run.rows:
        missing_outputs.append("candidate")
    if missing_outputs:
        artifacts.write_json(
            output_dir / "manifest.lock.json",
            {
                "skill": args.skill,
                "base_ref": args.base_ref,
                "candidate_ref": args.candidate_ref,
                "command": command,
                "base_return_code": base_run.return_code,
                "candidate_return_code": candidate_run.return_code,
                "missing_output_rows": missing_outputs,
            },
        )
        print(
            "Missing WBAF output rows from: " + ", ".join(missing_outputs),
            file=sys.stderr,
        )
        return 1
    comparison = compare.compare_results(
        skill=args.skill,
        base_ref=args.base_ref,
        candidate_ref=args.candidate_ref,
        suite=candidate_plan.target.suite,
        agent=candidate_plan.target.agent,
        base_rows=base_run.rows,
        candidate_rows=candidate_run.rows,
    )
    gate = gates.evaluate(comparison)
    markdown = report.render_comparison(comparison, gate)
    artifacts.write_bundle(
        output_dir=output_dir,
        report=comparison,
        gate=gate,
        markdown=markdown,
        manifest={
            "skill": args.skill,
            "base_ref": args.base_ref,
            "candidate_ref": args.candidate_ref,
            "command": command,
            "base_output_json": str(base_plan.output_path),
            "candidate_output_json": str(candidate_plan.output_path),
            "base_return_code": base_run.return_code,
            "candidate_return_code": candidate_run.return_code,
        },
    )
    print(markdown, end="")
    return 0 if base_run.return_code == 0 and candidate_run.return_code == 0 and gate.passed else 1


def _add_common_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--base-repo-root",
        default=None,
        help="Checkout containing the base skill content. Defaults to --repo-root.",
    )
    parser.add_argument(
        "--prepare-checkouts",
        action="store_true",
        help="Materialize --base-ref and --candidate-ref under --workdir/checkouts.",
    )
    parser.add_argument(
        "--source-repo",
        default=None,
        help="Local repo path or Git URL used with --prepare-checkouts.",
    )
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--wbaf-root", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_WORKDIR / "artifacts"))
    parser.add_argument("--scorer-parallelism", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="skillbench")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect-changed-skills")
    detect.add_argument("--repo-root", default=str(REPO_ROOT))
    detect.add_argument("--base-ref", required=True)
    detect.add_argument("--candidate-ref", required=True)
    detect.add_argument("--output", default=None)
    detect.set_defaults(func=cmd_detect)

    plan = sub.add_parser("plan")
    _add_common_bench_args(plan)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run")
    _add_common_bench_args(run)
    run.add_argument("--env-file", default=None)
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
