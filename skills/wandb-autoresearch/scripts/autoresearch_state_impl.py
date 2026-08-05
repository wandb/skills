# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Resumable autoresearch state — local markdown synced to a W&B artifact.

W&B tracks the experiments themselves; this tracks the research metadata W&B does
not: the overarching hypothesis, which runs matter and why, the entity/queue in
use, and the next planned trial. Keep it as a single local markdown file the agent
edits as it works, and sync it to a versioned artifact so a future session — or a
fresh environment — can pull it and resume.

The state artifact is logged to one fixed, resumable bookkeeping run per project,
so the metadata lives beside the experiments without cluttering the run list and
every save is a new artifact version.

Run directly:
    python skills/wandb-autoresearch/scripts/autoresearch_state.py save ENTITY PROJECT [--path AUTORESEARCH_STATE.md]
    python skills/wandb-autoresearch/scripts/autoresearch_state.py load ENTITY PROJECT [--dest .]
"""

from __future__ import annotations

import argparse
import os


_DEFAULT_STATE_FILE = "AUTORESEARCH_STATE.md"
_STATE_ARTIFACT = "autoresearch-state"
_STATE_ARTIFACT_TYPE = "autoresearch-state"
_STATE_RUN_ID = "autoresearch-session"
_BOOKKEEPING_JOB_TYPE = "autoresearch-bookkeeping"

_TEMPLATE = """# Autoresearch State — {entity}/{project}

- Queue: TBD
- Metric goal: minimize <metric> over <N> trials
- Hypothesis: <overarching hypothesis>

## Trials

| # | config | metric | run URL | verdict |
|---|--------|--------|---------|---------|

## Decisions / next step

- <what to try next and why>
"""


def ensure_state_file(entity: str, project: str,
                      path: str = _DEFAULT_STATE_FILE) -> str:
    """Create the local state markdown from a template if it does not exist."""
    if not os.path.exists(path):
        with open(path, "w") as handle:
            handle.write(_TEMPLATE.format(entity=entity, project=project))
    return path


def save_state(entity: str, project: str, path: str = _DEFAULT_STATE_FILE,
               name: str = _STATE_ARTIFACT) -> dict:
    """Sync the local state file/dir to a versioned W&B artifact for resume."""
    import wandb
    run = wandb.init(entity=entity, project=project, id=_STATE_RUN_ID,
                     resume="allow", job_type=_BOOKKEEPING_JOB_TYPE,
                     settings=wandb.Settings(silent=True))
    artifact = wandb.Artifact(name, type=_STATE_ARTIFACT_TYPE)
    if os.path.isdir(path):
        artifact.add_dir(path)
    else:
        artifact.add_file(path)
    run.log_artifact(artifact)
    run.finish()
    result = {"artifact": f"{entity}/{project}/{name}:latest", "run_url": run.url}
    print(f"Saved state -> {result['artifact']}")
    return result


def load_state(entity: str, project: str, name: str = _STATE_ARTIFACT,
               dest: str = ".") -> str | None:
    """Download the latest state artifact for resume. None if none exists yet."""
    import wandb
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{entity}/{project}/{name}:latest")
    except Exception:
        print(f"No saved state for {entity}/{project} yet")
        return None
    path = artifact.download(root=dest)
    print(f"Loaded {entity}/{project}/{name}:latest -> {path}")
    return path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Resumable autoresearch state")
    sub = parser.add_subparsers(dest="command")

    p_save = sub.add_parser("save", help="Sync local state to a W&B artifact")
    p_save.add_argument("entity")
    p_save.add_argument("project")
    p_save.add_argument("--path", default=_DEFAULT_STATE_FILE)

    p_load = sub.add_parser("load", help="Download the latest state artifact")
    p_load.add_argument("entity")
    p_load.add_argument("project")
    p_load.add_argument("--dest", default=".")

    args = parser.parse_args()
    if args.command == "save":
        save_state(args.entity, args.project, path=args.path)
    elif args.command == "load":
        load_state(args.entity, args.project, dest=args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
