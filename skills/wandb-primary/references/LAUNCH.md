<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B Launch Reference

Use this reference for W&B Launch queue setup, run relaunching, code jobs,
queue-item diagnosis, and launch-agent troubleshooting.

## Core Rules

Use `skills/wandb-primary/scripts/launch_helpers.py`. A local `wandb.init()` does not
exercise GPU work; submit through Launch to actually run on the cluster.

Every Launch entrypoint you create must call `wandb.init(...)`, log at least one
metric, and finish the run. The helpers add `wandb` to `requirements.txt` when
it is missing.

For new code jobs the helpers pick a default image by queue type — don't ask
the user about images; only pass `base_image=` when they bring their own
container. On Kubernetes queues the default is `python:3.11-slim`
(`DEFAULT_BASE_IMAGE`); pass `base_image=` for CUDA/GPU workloads. On local-container queues the helpers
switch to `LOCAL_DEFAULT_BASE_IMAGE` (multi-arch) with a pip bootstrap — see
`## Queue Setup`.

## Queue Setup

One question decides the path: where should jobs run — the user's own machine
(local Docker), or a Kubernetes cluster? Keep the user-facing steps dead
simple. Queue creation is a W&B API call (`create-queue`), not cluster work, so run it
yourself after confirming the details.

### Local Docker (the user's machine)

Their machine needs Docker installed **and the engine running** while jobs
execute (`docker info` verifies — installed alone is not enough), and the W&B
CLI logged in as themselves. Their normal `wandb login` is all the auth
required — do not ask for a service account or any key-handling ceremony on a
personal machine; that's for shared, always-on agents. GPU jobs additionally
need NVIDIA container support; CPU-only jobs need nothing more.

Create the queue yourself:

```bash
python skills/wandb-primary/scripts/launch_helpers.py create-queue ENTITY \
  --queue QUEUE --type local-container
```

Then give the user one command to run in a terminal and leave open:

```bash
wandb launch-agent --queue QUEUE --entity ENTITY
```

Code jobs submitted to a local-container queue get working defaults
automatically: the helpers switch the base image to `LOCAL_DEFAULT_BASE_IMAGE`
(multi-arch, so Apple Silicon works) and wrap the entrypoint in a pip
bootstrap that installs `requirements.txt` before the code runs. That base
image is bare, so pass `requirements=[...]` listing **every** package the code
imports. Never use `DEFAULT_BASE_IMAGE` on a local queue — it is amd64-only
and fails on arm64 machines.

### Kubernetes (cluster)

For a missing Kubernetes queue, **you** create `wandb-launch-k8s` (unless the
user names a queue) by running `create-queue` yourself — it's a
W&B API call, not cluster work, so confirm the queue name and resources with the
user first, then run it. Use defaults
`namespace="wandb-launch"`, `gpus=1`, `cpu=8`, `memory="80Gi"` and ask only for
the W&B entity if it is not inferable. Use
`wandb-launch` unless the user explicitly gives a different namespace. Queue
defaults should include namespace and resource requests only; do not put
`WANDB_API_KEY`, service account API keys, or other secrets in queue defaults.

```bash
python skills/wandb-primary/scripts/launch_helpers.py create-queue ENTITY \
  --queue wandb-launch-k8s \
  --namespace wandb-launch \
  --gpus 1 --cpu 8 --memory 80Gi
```

After queue creation, the **user** bootstraps the Kubernetes launch agent with
the existing W&B Launch Helm chart. Give them the command as a copy/paste block
to run in their own cluster environment — it needs cluster credentials, so do not
run it yourself.

Use Helm by default. For a shared cluster agent a team service-account API key
is recommended — it keeps the agent independent of any one person's account —
but a personal key also works; set either in `WANDB_SERVICE_ACCOUNT_API_KEY`
before running the command:

```bash
helm upgrade --install wandb-launch launch-agent \
  --repo https://charts.wandb.ai \
  --namespace wandb-launch \
  --create-namespace \
  --set agent.apiKey="$WANDB_SERVICE_ACCOUNT_API_KEY" \
  --set namespace=wandb-launch \
  --set "additionalTargetNamespaces={wandb-launch}" \
  --set-file launchConfig=<(cat <<YAML
entity: ${WANDB_ENTITY}
queues:
  - wandb-launch-k8s
max_jobs: 10
builder:
  type: noop
verbosity: 1
YAML
)
```

Only provide a kubectl fallback if the user explicitly says they cannot use
Helm.

## Queue Selection

Inspect queues before launching:

```bash
python skills/wandb-primary/scripts/launch_helpers.py list-queues ENTITY
```

`queue_name` is explicit; the helpers do not choose one. `list-queues` reports
queue name, active agent count, recent item states, and resource defaults.

`agents=N` means a Launch agent is polling that queue; `agents=0` means a
submitted item will wait until an agent starts. `items=state:count` shows recent
queue item states; `CLAIMED` is a normal final queue-item state, not proof that
work is still running. `ns=... gpu=... cpu=... mem=...` are the queue's default
resources. Helpers use that queue's defaults unless you pass `gpus=`, `cpu=`, or
`memory=`.

Check liveness before any blocking wait: submitting to (or waiting on) a queue
showing `agents=0` cannot succeed until an agent starts — surface that to the
user immediately instead of waiting. An agent can also die silently while items
sit `PENDING`; if nothing has moved across a couple of bounded checks, re-run
`list-queues` and re-verify the agent rather than extending the wait. The
converse also holds: an agent that claimed an item within the last few minutes
is live — don't re-run `list-queues` before every submission in a working
session.

The default `wait_for="launched"` waits only until Launch assigns a W&B run ID,
then surfaces the run URL and queue item ID. `wait_for="done"` waits for
completion — used when the user asks for it, or for an autoresearch-style serial
loop that must inspect one run before launching the next. For short jobs on a
live queue, prefer `wait_for="done"` with a bounded `timeout_seconds` (a couple
of minutes — the first container pull often dominates) over hand-rolled
sleep-and-check loops, and when the wait expires check once more before handing
back. Keep any in-shell wait well under the shell timeout; for runs that
outlast a turn, report status + run URL instead of sleeping.

Launch helpers return a status dict with `queue_item_id`, `run_url`, `run_id`,
`queue_state`, `run_state`, `launched`, `done`, and `check_command`.

## Common Jobs

Simple relaunch with config overrides:

```bash
python skills/wandb-primary/scripts/launch_helpers.py relaunch RUN_URL \
  --queue QUEUE \
  --config '{"epochs": 100}'
```

New code job:

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import submit_code_artifact_job

status = submit_code_artifact_job(
    code_files=["train.py"],
    entrypoint="python train.py",
    entity="ENTITY",
    project="PROJECT",
    queue_name="QUEUE",
    job_name="JOB_NAME",
)
print(status["run_url"], status["queue_item_id"])
```

Modify an existing job:

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import download_code_artifact, create_and_launch_modified_job

info = download_code_artifact("ENTITY/PROJECT/JOB_NAME:latest")
# edit files in info["code_dir"]; an apply_patch diff contains only its
# prefixed content lines, with no patch envelope, file header, or closing marker.
status = create_and_launch_modified_job(
    code_dir=info["code_dir"],
    entrypoint=info["entrypoint"],
    entity=info["entity"],
    project=info["project"],
    queue_name="QUEUE",
    job_name="JOB_NAME",
    base_image=info["base_image"],
)
print(status["run_url"], status["queue_item_id"])
```

Use queue default GPU/CPU/memory unless the user asks for a specific override.
If you need an override, pass `gpus=`, `cpu=`, and/or `memory=` directly to the
same helper.

If a requested change is not already a config field read by the training script,
edit code instead of passing a new config key.

## Debugging

Debug Launch jobs with `check` first. It prints queue, agent, run, run-log, and
job-version UI links, plus queue-item issues from the same fields used by the UI
sidebar. In Python, use `check_launch(...)`.

```bash
python skills/wandb-primary/scripts/launch_helpers.py check ENTITY PROJECT QUEUE QUEUE_ITEM_ID
```

Useful UI URL shapes:

```text
https://wandb.ai/ENTITY/launch/QUEUE_ID
https://wandb.ai/ENTITY/launch/QUEUE_ID/agents
https://wandb.ai/ENTITY/launch/QUEUE_ID/config
https://wandb.ai/ENTITY/launch/agents/AGENT_ID
https://wandb.ai/ENTITY/launch/agents/AGENT_ID/logs
https://wandb.ai/RUN_ENTITY/RUN_PROJECT/runs/RUN_ID
https://wandb.ai/RUN_ENTITY/RUN_PROJECT/runs/RUN_ID/logs
https://wandb.ai/JOB_ENTITY/JOB_PROJECT/jobs/JOB_COLLECTION_ID/version_details/ALIAS
```

There is no stable queue-item-details URL. Open the queue runs page and click
`Details` for the queue item; its Issues section is backed by queue-item
`error` / `warnings` fields. If an issue has `filePaths`, those files live on the
launch agent run in `ENTITY/model-registry` — `check` / `check_launch()` prints
their tails automatically, so read that output before hand-rolling a fetch.

K8s resources are still needed when the workload reached the cluster:

```bash
kubectl logs -n wandb deploy/launch-agent-wandb-launch --since=1h | rg 'QUEUE_ITEM_ID|RUN_ID|JOB_NAME'
kubectl get jobs,pods -n NAMESPACE --sort-by=.metadata.creationTimestamp
kubectl describe pod -n NAMESPACE POD_NAME
kubectl logs -n NAMESPACE POD_NAME --tail=200
```

There is no supported way to cancel a queue item an agent has already claimed —
don't spend turns hunting SDK internals for one. Tell the user plainly: the
options are letting the run finish, killing it from the run page, or stopping
the launch agent on their side.

Queue item states are not run states. `PENDING` means waiting for an agent,
`LEASED` means an agent popped the item, `CLAIMED` means the agent acknowledged it
and assigned `associatedRunId`, and `FAILED` means Launch marked the item failed.
`CLAIMED` is the normal final queue-item state for a successfully started Launch
run; check the associated W&B run state, run logs, K8s job/pod, and job/source
artifacts to decide whether execution succeeded.

When polling in Python, queue items never reach a `finished` queue state, so do
not poll for one. `check_launch()` returns `queue_state`, `launched`, `run_id`,
`run_url`, `run_state`, and `done`: `launched` is true once a run ID is
assigned, `done` once the run completes.

Use artifacts when a pod cannot find code or uses the wrong entrypoint:

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")
from launch_helpers import inspect_job_artifact, download_code_artifact

inspect_job_artifact("ENTITY/PROJECT/JOB_NAME:latest")
info = download_code_artifact("ENTITY/PROJECT/JOB_NAME:latest")
print(info["code_dir"], info["files"], info["entrypoint"], info["base_image"])
```

Frontend query map: queue pages use `QueueByID` on `ENTITY/model-registry` for
queue, agent list, queue items, issues, and resource config. Agent pages use
`FetchAgentDetails`; agent logs use `FetchRunOutputLog` and `RunLogLines` on the
agent run in `ENTITY/model-registry`. Issue file previews use `SingleFile` on the
agent run. Queue rows use `RunsInformation` to resolve associated run states.

## Gotchas

| Gotcha | Wrong | Right |
|---|---|---|
| List queues | `api.run_queues()` or raw GQL | `list_queues(entity)` from helpers |
| Resources | Build raw Launch resource args | Pick a queue; helpers use its defaults. Use `gpus=`, `cpu=`, or `memory=` only for explicit overrides |
| requirements.txt | `pip freeze` from venv | Let helpers add `wandb`; pass `requirements=[...]` only for extra packages |
| Base image arch | `docker build` on Mac | `docker buildx build --platform linux/amd64` |
| Fake launch | `wandb.init()` with config | `relaunch_run()` or `launch_job_artifact()` |
| Unknown config key | `relaunch_run(config={"conv_layers": 4})` | Code change: download, edit, `create_and_launch_modified_job()` |
