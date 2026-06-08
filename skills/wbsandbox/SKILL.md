---
name: wbsandbox
description: Reproduce and re-run W&B training jobs in CoreWeave sandboxes. Given a W&B run path, extracts the job artifact, source code, dependencies, and config, then executes the job in a CWSandbox with optional config overrides. Use this skill when the user asks to re-run, reproduce, or relaunch a W&B training run.
---

# W&B Sandbox — Reproduce & Re-run Training Jobs

## Quick start

A reusable script is bundled at `scripts/rerun.py`:

```bash
# Exact replica
uv run python scripts/rerun.py entity/project/run_id

# With config overrides
uv run python scripts/rerun.py entity/project/run_id --override epochs=10 lr=0.001

# With modified source files (iterate on code directly)
uv run python scripts/rerun.py entity/project/run_id --file-override model.py=/tmp/my_model_v2.py

# Combined: new code + config overrides
uv run python scripts/rerun.py entity/project/run_id --override epochs=20 lr=0.01 --file-override model.py=/tmp/experiment.py
```

The script handles the full workflow: fetch job artifact, download code, create sandbox, install deps, apply file overrides, and run training.

---

## Step-by-step workflow

### Step 1: Find the job artifact

Job artifacts are **input artifacts** to a run (the run consumes the job to know what to execute). Find them via `run.used_artifacts()`:

```python
import wandb, json
api = wandb.Api()
run = api.run("entity/project/run_id")

# Job artifacts are USED (input) artifacts, not logged artifacts
job_art = None
for art in run.used_artifacts():
    if art.type == "job":
        job_art = art
        break

if job_art is None:
    raise ValueError("No job artifact found. Was disable_job_creation=False?")
```

If given a job artifact name directly:

```python
job_art = api.artifact("entity/project/job-name:latest", type="job")
```

### Step 2: Parse the job spec

```python
import os

job_dir = job_art.download("/tmp/wandb-job")

with open(os.path.join(job_dir, "wandb-job.json")) as f:
    job_spec = json.load(f)

source_type = job_spec["source_type"]       # "artifact" or "repo"
entrypoint_script = job_spec["source"]["entrypoint"][1]  # e.g. "model.py"
runtime = job_spec.get("runtime", "")       # e.g. "CPython 3.14.2"
```

### Step 3: Get the source code

#### For `artifact` jobs:

Code artifacts are **logged** by the run:

```python
code_art = None
for art in run.logged_artifacts():
    if art.type == "code":
        code_art = art
        break

code_dir = code_art.download("/tmp/wandb-code")

# Collect source files, skipping .venv
source_files = []
for root, dirs, files in os.walk(code_dir):
    dirs[:] = [d for d in dirs if d != ".venv"]
    for fname in files:
        local = os.path.join(root, fname)
        rel = os.path.relpath(local, code_dir)
        source_files.append((local, rel))
```

#### For `repo` jobs:

Git clone happens inside the sandbox (see Step 6).

```python
git_info = job_spec["source"]["git"]
# git_info["remote"]  — repo URL
# git_info["commit"]  — exact commit hash
```

### Step 4: Build config with overrides

```python
config = {k: v for k, v in run.config.items() if not k.startswith("_")}
if overrides:
    config.update(overrides)
```

### Step 5: Create the sandbox

The sandbox needs:
- **Internet egress** (`NetworkOptions(egress_mode="internet")`) for pip install and data downloads
- **`WANDB_API_KEY`** in both the cwsandbox client env (for auth) and sandbox env (for wandb logging)
- **`WANDB_ENTITY_NAME`** in the client env (for cwsandbox auth)
- **`WANDB_CONFIG`** in the sandbox env (for config overrides)

```python
import netrc
from cwsandbox import NetworkOptions, Sandbox, SandboxDefaults

# Resolve API key
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if auth:
        wandb_api_key = auth[2]

# Set client-side auth env vars
os.environ.setdefault("WANDB_API_KEY", wandb_api_key)
os.environ.setdefault("WANDB_ENTITY_NAME", "your-entity")

# Determine container image from runtime
if runtime.startswith("CPython "):
    py_ver = runtime.replace("CPython ", "").rsplit(".", 1)[0]
    container_image = f"python:{py_ver}"
else:
    container_image = "python:3.11"

defaults = SandboxDefaults(
    container_image=container_image,
    max_lifetime_seconds=7200,
    tags=("wandb-rerun",),
    network=NetworkOptions(egress_mode="internet"),
    environment_variables={
        "WANDB_API_KEY": wandb_api_key,
        "WANDB_CONFIG": json.dumps(config),
    },
)
```

### Step 6: Install deps, upload code, and run

```python
with Sandbox.run(defaults=defaults) as sb:
    # Filter and upload frozen requirements
    req_path = os.path.join(job_dir, "requirements.frozen.txt")
    if os.path.exists(req_path):
        with open(req_path) as f:
            lines = f.readlines()
        # Filter out local-only packages not on PyPI
        filtered = [l for l in lines if not l.strip().startswith("cwsandbox")]
        sb.write_file("/app/requirements.txt", "".join(filtered).encode()).result()

        proc = sb.exec(
            ["pip", "install", "-r", "/app/requirements.txt"],
            cwd="/app",
            timeout_seconds=600,
        )
        for line in proc.stdout:
            if "Successfully installed" in line or "ERROR" in line:
                print(f"  {line}", end="")
        result = proc.result()
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed:\n{result.stderr}")

    # Upload source code (artifact jobs) or clone repo (git jobs)
    if source_type == "artifact":
        for local_path, rel_path in source_files:
            with open(local_path, "rb") as f:
                sb.write_file(f"/app/{rel_path}", f.read()).result()
        work_dir = "/app"
    else:
        git_info = job_spec["source"]["git"]
        sb.exec(["git", "clone", git_info["remote"], "/app/repo"],
                timeout_seconds=120).result()
        sb.exec(["git", "checkout", git_info["commit"]],
                cwd="/app/repo").result()
        work_dir = "/app/repo"

    # Apply file overrides — replace source files with modified versions
    if file_overrides:
        for sandbox_path, local_path in file_overrides.items():
            with open(local_path, "rb") as f:
                sb.write_file(f"{work_dir}/{sandbox_path}", f.read()).result()
            print(f"Replaced {sandbox_path}")

    # Run training with real-time output streaming
    process = sb.exec(
        ["python", entrypoint_script],
        cwd=work_dir,
        timeout_seconds=3600,
    )
    for line in process.stdout:
        print(line, end="")

    result = process.result()
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
```

---

## Job artifact structure reference

A W&B job artifact (type=`job`) contains two files:

### `wandb-job.json`

```json
{
  "_version": "v0",
  "source": {
    "artifact": "wandb-artifact://_id/<base64_id>",
    "entrypoint": ["pythonCPython 3.14", "model.py"],
    "notebook": false
  },
  "source_type": "artifact",
  "input_types": {
    "wb_type": "typedDict",
    "params": {
      "type_map": {
        "epochs": { "wb_type": "number" },
        "lr": { "wb_type": "number" }
      }
    }
  },
  "runtime": "CPython 3.14.2"
}
```

For git-based jobs, `source` contains:

```json
{
  "source": {
    "git": {
      "remote": "https://github.com/org/repo.git",
      "commit": "abc123def456"
    },
    "entrypoint": ["pythonCPython 3.14", "train.py"],
    "notebook": false
  },
  "source_type": "repo"
}
```

### `requirements.frozen.txt`

Pip-freeze style list of all Python dependencies with pinned versions.

---

## Iterating on experiments

The `--file-override` flag lets you iterate on model code directly without modifying config. This is the preferred approach for architecture changes (e.g. deeper backbones, BatchNorm, pretrained backbones) since these can't be expressed as simple config overrides.

**Workflow for running experiment sweeps:**

1. Create modified model files locally (e.g. `/tmp/experiments/model_v1.py`, `model_v2.py`, etc.)
2. Launch experiments in parallel using `--file-override`:
   ```bash
   uv run python scripts/rerun.py entity/project/run_id --override epochs=20 --file-override model.py=/tmp/experiments/model_v1.py &
   uv run python scripts/rerun.py entity/project/run_id --override epochs=20 --file-override model.py=/tmp/experiments/model_v2.py &
   ```
3. Compare results in the W&B project dashboard

**Timing considerations for CPU-only sandboxes:**
- Sandboxes run on CPU — no GPU. Lightweight models (custom CNNs, <5M params) train in ~20 min for 20 epochs on small datasets.
- Heavier models (e.g. MobileNetV2, ResNet) can easily exceed 20 min. The default exec timeout is 1 hour.
- If a model is too slow, reduce epochs or simplify the architecture rather than increasing timeouts further.
- pip install of PyTorch + dependencies takes ~80s in the sandbox.

---

## Critical rules

1. **Job artifacts are input artifacts** — find them via `run.used_artifacts()`, NOT `run.logged_artifacts()`
2. **Code artifacts are logged artifacts** — find them via `run.logged_artifacts()`
3. **Always enable internet egress** — use `NetworkOptions(egress_mode="internet")` so pip can install packages and training can download data
4. **Set cwsandbox client auth** — `WANDB_API_KEY` and `WANDB_ENTITY_NAME` must be set as env vars before creating the sandbox
5. **Filter frozen requirements** — remove local-only packages (e.g. `cwsandbox`) that aren't available on PyPI
6. **Config overrides go through `WANDB_CONFIG`** — set it in `environment_variables` in `SandboxDefaults` so it's available before `wandb.init()` runs
7. **Match the Python version** — extract from `runtime` field and use the corresponding `python:X.Y` image
8. **Skip `.venv/` in code artifacts** — filter out `.venv` directories when uploading source files
9. **Stream training output** — iterate over `process.stdout` for real-time logs instead of waiting for completion
10. **`.result()` is required** — all sandbox operations return `OperationRef` or `Process`; you MUST call `.result()` to block and check for errors

---

## CWSandbox API quick reference

```python
from cwsandbox import NetworkOptions, Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=7200,
    tags=("my-tag",),
    network=NetworkOptions(egress_mode="internet"),
    environment_variables={"KEY": "val"},
)

with Sandbox.run(defaults=defaults) as sb:
    sb.write_file("/path", b"content").result()      # Upload file
    data = sb.read_file("/path").result()             # Download file
    result = sb.exec(["cmd"], cwd="/dir").result()    # Run command
    result.stdout / result.stderr / result.returncode # Access output

# Stream exec output in real-time
process = sb.exec(["python", "train.py"])
for line in process.stdout:
    print(line, end="")
result = process.result()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `Sandbox.run(defaults=...)` | `Sandbox` | Create and start a sandbox |
| `sb.exec(command, cwd=..., timeout_seconds=...)` | `Process` | Execute command in sandbox |
| `sb.write_file(path, contents)` | `OperationRef[None]` | Write file to sandbox |
| `sb.read_file(path)` | `OperationRef[bytes]` | Read file from sandbox |
| `sb.stop()` | `OperationRef` | Stop the sandbox |

---

## Gotchas

| Gotcha | Details |
|--------|---------|
| Job artifact lookup | Job artifacts are **used** (input) artifacts, not logged. Use `run.used_artifacts()` |
| No job artifact on run | Run was created without `disable_job_creation=False`, or was invoked indirectly (e.g. `python -c`) so wandb couldn't detect the entrypoint |
| Egress mode string | Use `"internet"`, not `"EGRESS_TYPE_INTERNET"` (the proto enum name does not work) |
| cwsandbox in requirements | `cwsandbox` is a local-only package — filter it from `requirements.frozen.txt` before pip install |
| Code artifact includes `.venv/` | Skip `.venv/` directories when uploading — install from frozen requirements instead |
| `WANDB_CONFIG` not applied | Config must be set in `environment_variables` in `SandboxDefaults`, not via shell `export` after sandbox start |
| cwsandbox client auth | Requires `WANDB_API_KEY` and `WANDB_ENTITY_NAME` as env vars in the calling process |
| Sandbox timeout | Default exec timeout is 1 hour, sandbox lifetime is 2 hours. CPU-only sandboxes are slow — heavy models (MobileNetV2+) may still timeout with 20+ epochs |
| `log_code(".")` includes `.venv` | When logging code artifacts, use `include_fn` or `exclude_fn` to filter `.venv` |
| Architecture changes via config | Don't try to drive architecture changes through `WANDB_CONFIG` — use `--file-override` to replace source files directly |
