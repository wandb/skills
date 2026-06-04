"""Build and validate public skill candidates for Skill Bench."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathPolicyResult:
    """Changed-path policy result for a public skills PR."""

    files: tuple[str, ...]
    changed_skills: tuple[str, ...]
    script_changes: tuple[str, ...]
    blocked_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether all changed paths are in known public surfaces."""
        return not self.blocked_paths


@dataclass(frozen=True)
class CandidateBundle:
    """A staged skill bundle that can be passed to WBAF as files."""

    skill: str
    source_dir: Path
    bundle_dir: Path
    bundled_files: dict[str, str]
    system_prompt_append: str


ALLOWED_ROOTS = (
    "skills/",
    "bench/",
    "skillbench/",
    "scripts/validate_skills.py",
    ".github/workflows/validate-skills.yaml",
    ".github/workflows/skillbench-plan.yaml",
    "README.md",
    "CONTRIBUTING.md",
)


def changed_files(
    repo_root: Path,
    base_ref: str,
    candidate_ref: str,
) -> tuple[str, ...]:
    """Return files changed between two git refs."""
    completed = subprocess.run(
        ["git", "diff", "--name-only", base_ref, candidate_ref],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )


def detect_changed_skills(files: tuple[str, ...]) -> tuple[str, ...]:
    """Return public skill names touched by a changed-file list."""
    skills = {
        parts[1]
        for path in files
        if len(parts := Path(path).parts) >= 2 and parts[0] == "skills"
    }
    return tuple(sorted(skills))


def evaluate_path_policy(files: tuple[str, ...]) -> PathPolicyResult:
    """Classify changed paths for safe benchmark orchestration."""
    blocked: list[str] = []
    script_changes: list[str] = []
    for path in files:
        if not any(
            path == root or path.startswith(root)
            for root in ALLOWED_ROOTS
        ):
            blocked.append(path)
        parts = Path(path).parts
        if len(parts) >= 4 and parts[0] == "skills" and parts[2] == "scripts":
            script_changes.append(path)
    return PathPolicyResult(
        files=tuple(files),
        changed_skills=detect_changed_skills(files),
        script_changes=tuple(sorted(script_changes)),
        blocked_paths=tuple(sorted(blocked)),
    )


def _iter_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def build_bundle(
    *,
    repo_root: Path,
    skill: str,
    output_dir: Path,
) -> CandidateBundle:
    """Stage `skills/<skill>` and build a WBAF `bundled_files` mapping."""
    source_dir = repo_root / "skills" / skill
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing skill directory: {source_dir}")
    if not (source_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"missing SKILL.md for skill {skill!r}")

    bundle_dir = output_dir / "skills" / skill
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, bundle_dir)

    bundled_files: dict[str, str] = {}
    for path in _iter_files(bundle_dir):
        rel = path.relative_to(output_dir)
        bundled_files[str(rel)] = str(path)

    prompt = (
        f"\n\nA public W&B skill package named `{skill}` is bundled with this "
        "run under the `skills/` file bundle path. Read `skills/"
        f"{skill}/SKILL.md` first, then use its referenced scripts and "
        "reference files when relevant. Treat these bundled files as the "
        "candidate public skill under evaluation.\n"
    )
    return CandidateBundle(
        skill=skill,
        source_dir=source_dir,
        bundle_dir=bundle_dir,
        bundled_files=bundled_files,
        system_prompt_append=prompt,
    )
