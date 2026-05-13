---
name: wandb-launch
description: "Use for W&B Launch workflows: relaunching runs, submitting jobs to queues, inspecting job artifacts, creating or inspecting queues, and running training on remote compute. Use analysis skills first to decide what to run."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Launch

Use this skill for Launch and remote-compute workflows. This skill can submit
jobs to compute and should remain separate from read-only analysis skills.

## Rules

- This is a write-capable remote-compute skill, not a run-analysis skill.
- Do not answer Signal Builder, report authoring, run analysis, run counting,
  trace counting, eval analysis, or scorer-analysis questions from this skill.
  Route those to the relevant read-only skill unless the user explicitly asks to
  launch or relaunch compute.
- Prefer one helper CLI command for simple relaunches.
- Use Launch resource args for GPU, CPU, and memory changes rather than inventing
  local training commands.
- Use Launch helpers; do not fake launches with `wandb.init()`.
- Do not run training locally in the sandbox.
- Do not check `WANDB_ENTITY` or `WANDB_PROJECT` for a run URL task; parse the URL.
- Confirm before creating queues or submitting jobs unless the user explicitly requested that action.
- Use run analysis first when the user has not yet decided what to run.

## Helper Script

```python
import sys
sys.path.insert(0, "skills/wandb-launch/scripts")
from launch_helpers import relaunch_run, list_queues, check_launched_run
```

Use `make_resource_args()` for resource args when changing GPU, CPU, or memory
requirements.

## Fast CLI Path

Use this for a simple config-only relaunch. The helper discovers queues and job
artifacts.

```bash
python skills/wandb-launch/scripts/launch_helpers.py relaunch \
  "https://wandb.ai/entity/project/runs/run_id" \
  --config '{"epochs": 100}'
```

Check status with the queue item id printed by the helper:

```bash
python skills/wandb-launch/scripts/launch_helpers.py check \
  "entity" "project" "queue-name" "QUEUE_ITEM_ID"
```

## Config Change Versus Code Change

| Change type | Examples | Workflow |
| --- | --- | --- |
| Config override | epochs, learning rate, batch size | `relaunch_run(..., config={...})` |
| Code change | architecture, optimizer logic, data augmentation | download job code, edit, then `create_and_launch_modified_job()` |

If the user asks for a code behavior change, do not pass a random config key and
pretend it will work. Download and edit the job code.

## Code Change Workflow

Script 1: inspect/download code.

```python
import sys
sys.path.insert(0, "skills/wandb-launch/scripts")
from launch_helpers import parse_run_url, get_job_artifact, download_code_artifact

entity, project, run_id = parse_run_url("RUN_URL")
job_artifact = get_job_artifact(f"{entity}/{project}/{run_id}")
info = download_code_artifact(f"{entity}/{project}/{job_artifact.name}")
print(info["code_dir"], info["entrypoint"], info["base_image"])
```

Script 2: submit modified code after edits.

```python
import sys
sys.path.insert(0, "skills/wandb-launch/scripts")
from launch_helpers import list_queues, create_and_launch_modified_job

queue = list_queues("ENTITY")[0]
create_and_launch_modified_job(
    code_dir="CODE_DIR_FROM_STEP_1",
    entrypoint="python train.py",
    entity="ENTITY",
    project="PROJECT",
    queue_name=queue["name"],
    namespace=queue["namespace"],
    job_name="modified-job",
    base_image="BASE_IMAGE",
)
```

## Packaging Gate

Do not include this skill in a default read-only split until WBA tasks with
`safety:launches_compute` exist and pass. Until then, keep it as a staged
contribution skill with static helper/import tests.
