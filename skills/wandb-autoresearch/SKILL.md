---
name: wandb-autoresearch
description: "Run bounded, evidence-driven training research through W&B Launch: assess project readiness, establish launchable code and queue capacity, smoke-test real jobs, execute serial trials, compare metrics, and persist resumable research state. Use when a coding agent is asked to autonomously test training hypotheses or tune a real W&B-tracked workload."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B autoresearch

Use W&B Launch to execute real training experiments, read their results, and
choose the next bounded trial. Do not create a local `wandb.init()` run as a
substitute for compute that was supposed to run through Launch.

The Launch implementation lives in `skills/wandb-primary`:

- `skills/wandb-primary/scripts/launch_helpers.py`
- the `Launch` section in `skills/wandb-primary/SKILL.md`

Install missing packages explicitly with `uv`; do not assume a fixed working
directory or preinstalled environment.

## Flow

Read only the reference for the current stage:

| Stage | Entry condition | Read |
|---|---|---|
| Understand | Always first; inspect launchable code, queue capacity, data location, and prior state | `references/understand.md` |
| Set up | Launchable code or usable queue capacity is missing | `references/setup.md` |
| Search | Launchable code and queue capacity both exist | `references/search.md` |

The readiness gate is `launchable code + queue with usable capacity`. A dataset
may be a versioned Artifact or may be supplied by the code/image; its storage
form alone does not determine readiness.

## Experiment records

Use each W&B object for one job:

- Runs hold trial config, metrics, system metrics, status, and code/job lineage.
- Job or code Artifacts make code relaunchable.
- Dataset Artifacts provide versioned inputs when the workload needs them.
- Launch queues and agents provide compute.
- A small local markdown file holds the hypothesis, trial tally, decisions,
  queue, and next step.

Keep local state at a user-visible path in the current project or a path the
user supplied. Sync it to a W&B Artifact only when cross-session resume is
useful and the user authorizes the write:

```bash
S=skills/wandb-autoresearch/scripts/autoresearch_state.py
uv run --with wandb python "$S" load ENTITY PROJECT --dest .
uv run --with wandb python "$S" save ENTITY PROJECT --path AUTORESEARCH_STATE.md
```

## Compute safety

- Confirm the queue, metric direction, search axis, and trial/compute budget
  before the first non-smoke launch. Do not exceed the agreed budget.
- Smoke-test the real job at a deliberately small setting before full trials.
- Change only config fields the program actually reads. Use a code edit and a
  new job Artifact for architecture or pipeline changes.
- Run trials serially unless the user explicitly authorizes parallel compute.
  Inspect each completed result before selecting the next trial.
- Report failures, hardware utilization, and uncertainty. Re-run a likely
  winner with another seed when the apparent gap could be noise.
- Never fabricate a run, metric, queue result, or completion state.
