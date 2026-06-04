"""Git checkout helpers for E2E Skill Bench runs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def materialize_ref(*, source: str, ref: str, output_dir: Path) -> Path:
    """Clone `source` and checkout `ref` into `output_dir`.

    `source` may be a local repository path or a remote Git URL.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", source, str(output_dir)],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(output_dir), "checkout", "--quiet", ref],
        check=True,
        text=True,
        capture_output=True,
    )
    return output_dir
