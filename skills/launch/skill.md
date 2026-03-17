---
name: launch
description: Set up and manage W&B Launch agents on Kubernetes. Covers queue creation, helm deployment, agent lifecycle, job submission, and run reproduction. Use this skill when the user wants to deploy a launch agent, create/manage launch queues, submit training jobs, or reproduce existing W&B runs on a K8s cluster.
---

# W&B Launch Skill

This skill covers deploying and managing W&B Launch agents on Kubernetes and submitting training jobs through them.

## Bundled scripts

```python
import sys
sys.path.insert(0, "skills/launch/scripts")
from launch_helpers import (
    create_queue,                        # Create a K8s launch queue with prioritization
    recreate_queue_with_prioritization,  # Delete + recreate queue to enable prioritization
    inspect_queue,                       # Print queue name/type/prioritization
    make_resource_args,                  # Build resource_args for launch_add()
    submit_code_artifact_job,            # Create job artifact and enqueue in one call
    inspect_job_artifact,                # Download + inspect a job artifact's metadata
    launch_job_artifact,                 # Launch directly from an artifact path
    get_job_artifact,                    # Check if a run has a job artifact
    relaunch_run,                        # Re-run an existing run from its job artifact
)
```

### Running launch scripts in projects with GPU-only deps

A `launch.py` only needs `wandb` — it does not need the full project venv (e.g. torch, CUDA). If the project's `pyproject.toml` pins GPU-only deps that can't install on macOS, run outside the project venv:

```bash
# From anywhere — use --no-project to skip the project's venv
uv run --no-project --with wandb python launch.py

# Or from the project directory
cd my-project && uv run --no-project --with wandb python launch.py
```

---

## When to use

| I need to... | Do this |
|---|---|
| Reproduce a W&B run | See **Reproducing a Run** below |
| Deploy a launch agent | See **Agent Setup** below |
| Create a queue | `create_queue(...)` — see **Queue Management** |
| Submit a training job | `submit_code_artifact_job(...)` — see **Submitting Jobs** |
| Troubleshoot | See **Troubleshooting** below |

---

## Job Types

| Type | How it works | When to use |
|------|-------------|-------------|
| **Docker image job** | Agent runs the image directly — no code download, no dep install | Self-contained image with everything baked in |
| **Code artifact + base image** (recommended) | Init container downloads code artifact to emptyDir, installs `requirements.txt`, main container runs with base image | Fast iteration — only code changes per experiment |
| **Code artifact, no base image** | Same as above, agent picks a default Python image | When you don't have a base image |
| **Dockerfile job** | Agent builds an image from a Dockerfile | **Not supported** with `builder.type: noop` |

---

## Reproducing a Run

### Step 1: Check for an existing job artifact

```python
from launch_helpers import get_job_artifact

job_artifact = get_job_artifact("<entity>/<project>/<run_id>")
if job_artifact:
    print(f"Found: {job_artifact.name}")
else:
    print("No job artifact — need to create one")
```

You can also check the **Artifacts** tab in the W&B run UI.

### Step 2a: Re-launch if a job artifact exists

**From a run** (reproduces with original config):
```python
from launch_helpers import relaunch_run

run = relaunch_run(
    run_path="<entity>/<project>/<run_id>",
    queue_name="<QUEUE_NAME>",
    namespace="<NAMESPACE>",
    k8s_secrets={"HF_TOKEN": ("hf-token", "token")},
)
```

**From an artifact path directly** (e.g. from the W&B Artifacts UI):
```python
from launch_helpers import launch_job_artifact, inspect_job_artifact

# Inspect first to see entrypoint, base image, deps
inspect_job_artifact("wandb/my-project/my-job:v6")

# Launch it
run = launch_job_artifact(
    artifact_path="wandb/my-project/my-job:v6",
    queue_name="<QUEUE_NAME>",
    namespace="<NAMESPACE>",
    k8s_secrets={"HF_TOKEN": ("hf-token", "token")},
)
```

### Step 2b: Create a job artifact from scratch

**Prefer code artifact + custom base image** — deps in the image, only code in the artifact.

#### 1. Build a custom base image (one-time)

```dockerfile
# Dockerfile.base
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN pip install --no-cache-dir wandb
# Add project-specific deps here
# Optionally bake in training data
```

```bash
# Must be linux/amd64 for K8s clusters
docker buildx build --platform linux/amd64 \
    -t <registry>/<name>:latest \
    -f Dockerfile.base --push .
```

Standard base images (no custom build):

| Image | Includes |
|-------|---------|
| `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | PyTorch + CUDA (no wandb) |
| `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` | CUDA only |

#### 2. Submit the job

```python
from launch_helpers import submit_code_artifact_job

run = submit_code_artifact_job(
    code_files=["train.py", "model.py"],
    entrypoint="python train.py",
    entity="<ENTITY>",
    project="<PROJECT>",
    queue_name="<QUEUE_NAME>",
    namespace="<NAMESPACE>",
    job_name="my-train-job",
    base_image="<registry>/<name>:latest",
    requirements=["wandb"],  # only deps NOT already in the base image
)
```

---

## Queue Management

### Create a queue

```python
from launch_helpers import create_queue

queue = create_queue(
    name="my-queue",
    entity="<ENTITY>",
    namespace="<NAMESPACE>",
    gpus=1, cpu=8, memory="80Gi",
    prioritization=True,
)
```

### Enable prioritization on an existing queue

Prioritization **cannot be toggled** — you must delete and recreate:

```python
from launch_helpers import recreate_queue_with_prioritization

queue = recreate_queue_with_prioritization(
    name="my-queue",
    entity="<ENTITY>",
    namespace="<NAMESPACE>",
)
# Then restart the agent — it loses registration after queue recreation
```

### Inspect a queue

```python
from launch_helpers import inspect_queue
inspect_queue(name="my-queue", entity="<ENTITY>")
```

---

## Agent Setup

### Helm values (`launch-agent-values.yaml`)

```yaml
agent:
  apiKey: ""  # Pass via --set agent.apiKey=$WANDB_API_KEY
  image: jzhaowandb/launch-agent-dev:emptydir-code-mount-v16  # emptyDir agent — no PVC needed
  imagePullPolicy: Always
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

namespace: <NAMESPACE>
baseUrl: https://api.wandb.ai

additionalTargetNamespaces:
  - <NAMESPACE>

launchConfig: |
  entity: <ENTITY>
  queues:
    - <QUEUE_NAME>
  max_jobs: 10
  builder:
    type: noop
  verbosity: 1
```

### Deploy

```bash
kubectl create namespace <NAMESPACE>
helm repo add wandb https://wandb.github.io/helm-charts && helm repo update

helm upgrade --install wandb-launch wandb/launch-agent \
    -n <NAMESPACE> \
    -f launch-agent-values.yaml \
    --set agent.apiKey=$WANDB_API_KEY
```

### Verify

```bash
kubectl get pods -n <NAMESPACE>
kubectl logs deploy/launch-agent-wandb-launch -n <NAMESPACE> --tail=5
# Should show: "agent XXXX polling on queues <QUEUE_NAME>, running 0 out of a maximum of 10 jobs"
```

---

## Agent Lifecycle

```bash
# Update config or swap image (then force restart)
helm upgrade wandb-launch wandb/launch-agent -n <NAMESPACE> \
    -f launch-agent-values.yaml --set agent.apiKey=$WANDB_API_KEY
kubectl rollout restart deployment/launch-agent-wandb-launch -n <NAMESPACE>

# Stop / start
kubectl scale deployment/launch-agent-wandb-launch -n <NAMESPACE> --replicas=0
kubectl scale deployment/launch-agent-wandb-launch -n <NAMESPACE> --replicas=1

# Restart (required after queue delete/recreate)
kubectl rollout restart deployment/launch-agent-wandb-launch -n <NAMESPACE>

# Teardown
helm uninstall wandb-launch -n <NAMESPACE>
kubectl delete namespace <NAMESPACE>
```

---

## Submitting Jobs

### Full example (low-level)

```python
import os, tempfile, shutil, wandb
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.create_job import _create_job
from launch_helpers import make_resource_args

with tempfile.TemporaryDirectory() as temp_dir:
    shutil.copy2("train.py", temp_dir)

    # _create_job reads requirements.txt directly — does NOT inspect the venv
    # Only list deps NOT already in the base image
    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
        f.write("wandb\n")

    artifact, action, aliases = _create_job(
        wandb.InternalApi(),
        job_type="code",
        path=temp_dir,
        entity=ENTITY, project=PROJECT,
        name="my-job",
        base_image="<registry>/<name>:latest",
        entrypoint="python train.py",
    )

queued_run = launch_add(
    job=f"{ENTITY}/{PROJECT}/my-job:latest",
    entity=ENTITY, project=PROJECT,
    queue_name=QUEUE_NAME,
    entry_point=["python", "train.py"],
    resource_args=make_resource_args(NAMESPACE),
)

run = queued_run.wait_until_running()
print(f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run.id}")
run = queued_run.wait_until_finished()
print(run.summary)
```

### What the agent creates (pod spec)

```
Pod
├── Init container: wandb/launch-agent:latest
│   Downloads code artifact → /mnt/wandb, installs requirements.txt
│
├── Main container: <base_image>
│   Command: python train.py (from /mnt/wandb)
│   Env: WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_ID, ...
│   Resources: GPU, CPU, memory as specified
│
└── Volume: wandb-source-code-volume (emptyDir, shared between containers)
```

---

## Monitoring

```bash
# Agent
kubectl logs deploy/launch-agent-wandb-launch -n <NAMESPACE> -f

# Training pod
kubectl get pods -n <NAMESPACE>
kubectl logs <POD_NAME> -c wandb-source-code-init -n <NAMESPACE>  # init container
kubectl logs <POD_NAME> -c train -n <NAMESPACE>                    # training
kubectl describe pod <POD_NAME> -n <NAMESPACE>                     # events + status
kubectl get pod <POD_NAME> -n <NAMESPACE> -o jsonpath='{.spec.containers[0].resources}'
```

---

## Critical rules

- **Always pass `resource_args` explicitly** in `launch_add()` — queue defaults get double-nested by the server
- **Restart the agent after queue delete/recreate** — agent loses its registration
- **Queue prioritization can't be enabled after creation** — must delete and recreate
- **`requirements.txt` is read directly from the code dir** — `_create_job` does NOT inspect the venv; a temp dir with no `requirements.txt` means no deps installed
- **Keep `requirements.txt` minimal** — only deps not in the base image; bake everything else into the base image for fast startup
- **Build base images for `linux/amd64`** — K8s clusters are amd64; Mac is arm64
- **Must pass `--set agent.apiKey=...` on every `helm upgrade`** — not persisted between upgrades
- **Use `kubectl rollout restart` after image changes** — `helm upgrade` alone may not restart the pod
- **Inject K8s secrets via `k8s_secrets` param** — pass `k8s_secrets={"HF_TOKEN": ("hf-token", "token")}` to `make_resource_args`, `submit_code_artifact_job`, `launch_job_artifact`, or `relaunch_run`. Queue defaults don't reliably inject secrets due to double-nesting.
- **Include `triton` in base images** — `torch.compile` with the Inductor backend requires triton. The cu128 torch wheels don't always bundle it.

## Gotchas

| Gotcha | Wrong | Right |
|--------|-------|-------|
| Queue prioritization | `queue.update(...)` | Delete + recreate with `prioritization_mode="V0"` |
| Agent logs | `-l app.kubernetes.io/name=wandb-launch` | `kubectl logs deploy/launch-agent-wandb-launch` |
| resource_args | Rely on queue defaults | Pass full `resource_args` via `make_resource_args()` |
| requirements.txt | `pip freeze > requirements.txt` from venv | Write it manually with only deps missing from base image |
| Base image arch | `docker build` on Mac | `docker buildx build --platform linux/amd64` |
| Stock agent + code artifact | `wandb/launch-agent:0.25.1` | Fails without PVC — use emptyDir agent image |
| Agent not in W&B UI | Wait for reconnect | `kubectl rollout restart deployment/launch-agent-wandb-launch` |
| Stop/start agent | Delete pod | `kubectl scale ... --replicas=0/1` |
| Agent image not updating | `helm upgrade` only | Also run `kubectl rollout restart` |
| Missing wandb in container | Use stock pytorch base image | Build custom base with `pip install wandb` baked in |

---

## Troubleshooting

**Agent not showing in W&B UI** — Restart it: `kubectl rollout restart deployment/launch-agent-wandb-launch`

**Stock agent: "WANDB_LAUNCH_SOURCE_CODE_PVC_ not set"** — Switch to the emptyDir agent image.

**Init:Error on job pod** — `kubectl logs <POD> -c wandb-source-code-init`. Common causes: bad artifact ID, missing `wandb-api-key-wandb-launch` secret.

**ModuleNotFoundError in container** — Add the missing package to `requirements.txt` in the code dir.

**Agent marks job finished too early** — Bug in older emptyDir agent versions. Use latest version.

**Pod has no GPU** — resource_args double-nesting; pass them explicitly via `make_resource_args()`.

**exec format error** — Image built for wrong arch. Always use `--platform linux/amd64`.

**helm: command not found** — `brew install helm`.
