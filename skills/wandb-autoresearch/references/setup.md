<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Set up what's missing

Reach here from `understand.md` when the project can't run experiments yet.
Establish **only** the components it's missing, and re-check after each piece.

## Missing queue / active agent

Read the Launch section in `skills/wandb-primary/SKILL.md`. Default to a
Kubernetes queue; offer Local Docker only if the user wants their own machine.
Two steps use different execution environments:

- **Create the queue with the W&B API** by running `create-queue` from an
  environment with the installed skill and W&B credentials. Confirm the queue
  name and resources first. This does not configure the user's cluster.
- **Bootstrap the launch agent — the user does this** in their own cluster.
  Give them the Helm agent-bootstrap (or `wandb launch-agent`) as a copy/paste
  block; **do not run `kubectl`, `helm`, or `launch-agent` unless the user explicitly authorizes changes to their cluster.**

Stop until they confirm an active agent.

## Missing launchable code

Two doors:

- the user has runs but code-saving was off — have them enable it (`save_code` /
  `WANDB_SAVE_CODE=true`, see
  `skills/wandb-primary/references/WANDB_CONCEPTS.md`) and re-run, or
- the user shares a training script — package and launch it once to create the
  job artifact, **no prior run needed**:

```python
import sys; sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import submit_code_artifact_job

status = submit_code_artifact_job(
    code_files=["train.py"], entrypoint="python train.py",
    entity="ENTITY", project="PROJECT", queue_name="QUEUE", job_name="JOB_NAME",
)
```

The script must read its hyperparameters from `run.config` so later trials can
override them.

## Missing data

If the code needs data the base image doesn't provide, have the user log it as a
`dataset` artifact and consume it via `run.use_artifact(...)` so every trial
shares one versioned dataset.

---

When the gate is met (launchable code + a queue with capacity), continue to
`search.md`.
