# W&B Concepts & Nomenclature

Reference for understanding W&B's data model, terminology, and common points of confusion. Read this when interpreting user requests or debugging unexpected data locations.

---

## W&B Product Lines

W&B has five product lines. Understanding which product a user is asking about determines which APIs and vocabulary to use.

### Models — The ML Platform (Primary)

**What it is:** The umbrella product for the core W&B ML platform — experiment tracking, model registry, artifact management, sweeps, reports, and collaboration. When users say "W&B" without qualification, they usually mean Models.

**Includes:** Runs, Projects, Artifacts, Registry, Sweeps, Reports, Tables, Alerts, Automations. All accessed via `wandb` Python SDK and `wandb.Api()`.

**Relationship to other products:** Models is the foundation. Training jobs (via Launch) produce Runs that live in Models. Weave traces live in the same Projects. The Registry curates artifacts across projects at the org level.

### Weave — LLM Observability & Evaluation

**What it is:** A toolkit for tracing, evaluating, and improving LLM-powered applications (not training — post-deployment/development observability).

**Core terminology:**
- **Op** (`@weave.op()`): A decorated function that automatically logs all calls. The fundamental unit of instrumentation. Ops are versioned — code changes create new versions automatically.
- **Call**: A single invocation of an Op. Records inputs, outputs, latency, token usage.
- **Trace**: The complete execution tree formed when Ops call other Ops. A root Op and all its nested child Ops form one trace. Viewable as a call tree in the UI.
- **Model** (`weave.Model`): A class bundling configuration/weights with a `predict()` method (which is itself an Op). Auto-versioned when code or parameters change.
- **Dataset** (`weave.Dataset`): A versioned collection of rows used as evaluation inputs. Published with `weave.publish()`.
- **Evaluation** (`weave.Evaluation`): Orchestrates running a Model's `predict()` on every Dataset row, then scoring outputs. The call hierarchy is: `Evaluation.evaluate` -> `Evaluation.predict_and_score` (per row) -> `model.predict` + `scorer.score` -> `Evaluation.summarize`.
- **Scorer**: A function or class that evaluates model outputs. Function scorers use `@weave.op()`. Class scorers inherit `weave.Scorer`. Built-in scorers include `HallucinationFreeScorer`, `SummarizationScorer`, `ValidJSONScorer`, `PydanticScorer`, etc.
- **Ref**: A versioned pointer to any Weave object (model, dataset, op). Format: `weave:///entity/project/object_type/name:version`.

**Relationship to other products:** Weave is for GenAI/LLM applications. Training is for model training. A trained model (from Training) might be served and then monitored with Weave. Weave uses `weave.init()` (not `wandb.init()`).

**Gotchas:**
- `weave.init("entity/project")` is positional — not `weave.init(project="x")`.
- Weave autopatches many LLM providers (OpenAI, Anthropic, etc.) — if a user says "I don't see traces," check if their provider is autopatched or if they need manual `@weave.op()`.
- Weave objects use attribute access (`getattr`), not dict access (`.get()`).

### Inference — Hosted LLM API

**What it is:** W&B's hosted API providing access to open-source foundation models through an OpenAI-compatible interface, without self-hosting.

**Core terminology:**
- **OpenAI-compatible API**: Uses the same SDK/format as OpenAI. Switch by changing `base_url` to `https://api.inference.wandb.ai/v1` and using a W&B API key.
- **Inference credits**: Usage-based billing for API calls.
- **Team/project attribution**: Optional `extra_headers` to route usage tracking to a specific W&B team and project.

**Relationship to other products:** Inference provides the LLM. Weave can trace and evaluate calls made through Inference. Training can fine-tune models that Inference then serves.

**Gotchas:**
- Not all open-source models are available — check the model list endpoint.
- Uses W&B API key for auth, not an OpenAI key.
- Usage appears in a W&B project (default: "inference") for monitoring.

### Training — Experiment Tracking & Model Training

**What it is:** The original W&B product for tracking ML training experiments — logging metrics, hyperparameters, and artifacts across runs.

**Core terminology:**
- **Run**: The atomic unit — one execution of training code. Created by `wandb.init()`. Has a unique **run ID** (8-char hash) and a human-readable **run name** (auto-generated words).
- **Config**: Input hyperparameters set at run start. Does not change during training.
- **History**: Time-series metrics logged step-by-step during training (`run.log()`).
- **Summary**: Final/best metric values. By default the last logged value; customizable with `define_metric`.
- **Artifact**: A versioned data object (dataset, model, checkpoint). Has versions (v0, v1...), aliases ("latest", "best"), types ("dataset", "model"), and lineage (which run produced/consumed it).
- **Sweep**: Automated hyperparameter search. Defined by a sweep config (search space + method). Methods: `"bayes"`, `"grid"`, `"random"`. A sweep controller manages multiple sweep agents, each running trials as separate runs.
- **Project**: Groups related runs. Scoped under an entity (user or team).
- **Registry**: Organization-level curated artifact repository above project-level artifacts. Default registries: "Models" and "Datasets".
- **Serverless RL**: Reinforcement learning service for improving model reliability on multi-turn agentic tasks.
- **Serverless SFT**: Supervised fine-tuning service for distillation, style/format, or RL preparation.

**Relationship to other products:** Training produces models/artifacts. Launch executes training jobs on remote compute. Weave monitors the deployed model after training. Inference can serve fine-tuned models.

**Gotchas:**
- `wandb.init()` creates runs (for training scripts). `wandb.Api()` queries existing data (for analysis). Do not confuse them.
- `run.name` (display name, not unique) vs `run.id` (unique hash) is a constant source of confusion.
- Config keys with dots cause flattening collisions.

### Launch — Job Orchestration & Compute Management

**What it is:** A system for packaging, queueing, and executing ML workloads on remote compute infrastructure (Kubernetes, SageMaker, Docker).

**Core terminology:**
- **Job**: A blueprint for a task — encapsulates source code, dependencies, and execution parameters. Created from Docker images, git repos, or local code. Stored as a W&B artifact.
- **Queue**: A FIFO queue targeting a specific compute backend. Jobs are submitted to queues. Each queue has a resource configuration (GPU type, memory, etc.).
- **Agent**: A polling service that watches a queue, pulls jobs, builds/downloads containers, and executes them on its compute backend.
- **Code Artifact**: A snapshot of source code logged to W&B when launching from local code.
- **Image Job**: A pre-built Docker image ready to run without a build step.
- **Resource Args**: Compute resource configuration (GPU count/type, memory, node selectors) passed when submitting a job.

**Supported compute backends:** Kubernetes, Amazon SageMaker, Docker (local).

**Relationship to other products:** Launch executes Training jobs on remote compute. A trained model can then be served via Inference and monitored via Weave. Launch queues and agents are the bridge between "I want to train" and "this GPU cluster runs it."

**Gotchas:**
- Never fake a launch with `wandb.init()` — use Launch APIs.
- Config override vs code change: if the user wants to change a hyperparameter, relaunch with config override. If they want to change model architecture, download code, edit, and launch a modified job.
- `requirements.txt` for launch jobs should only contain deps missing from the base image, not a full `pip freeze`.

---

## Core Hierarchy

A project is a shared container for both training data (runs) and LLM observability data (Weave traces). They coexist under the same entity/project namespace.

```
Entity (username or team)
  +-- Project (groups related work)
  |     |
  |     +-- Runs (training / experiment tracking — wandb SDK)
  |     |     +-- Config         (input hyperparameters)
  |     |     +-- History        (time-series metrics logged during training)
  |     |     +-- Summary        (final/best metric values)
  |     |     +-- Artifacts      (versioned data objects: datasets, models)
  |     |     +-- System Metrics (GPU, CPU, memory usage)
  |     |
  |     +-- Weave Objects (LLM observability — weave SDK)
  |           +-- Ops        (versioned traced functions)
  |           +-- Calls      (individual op invocations, with inputs/outputs/timing)
  |           +-- Traces     (call trees — a root call and all its children)
  |           +-- Models     (versioned config + predict method)
  |           +-- Datasets   (versioned row collections for evals)
  |           +-- Evaluations (orchestrated model + dataset + scorer runs)
  |
  +-- Registry (org-level curated artifacts, above projects)
        +-- Collections (e.g. "production-classifier")
              +-- Linked Artifact Versions
```

- **Entity**: A username or team name. The top-level namespace. Everything is scoped under an entity.
- **Project**: Groups related work. Contains both runs AND Weave objects in the same namespace. If unspecified at `wandb.init()`, defaults to `"uncategorized"` — always specify project. Both `wandb.Api()` and `weave.init()` take the same `"entity/project"` path.
- **Run**: The atomic unit of training computation. Created by `wandb.init()`. Each has:
  - A unique **run ID** (8-char hash, e.g. `"abc12def"`) — used for programmatic access, resuming, and API calls
  - A human-readable **run name** (e.g. `"cosmic-sunset-42"`) — auto-generated if not set, display-only, **not unique**
- **Run path**: `<entity>/<project>/<run_id>` — the canonical way to reference a run in the Public API.
- **Call**: The atomic unit of Weave tracing. Each has a unique **call ID** (UUID), an **op_name**, and optional **parent_id** (linking it into a trace tree).
- **Trace**: A tree of calls sharing the same **trace_id**. The root call has no parent. Child calls are nested function invocations within the root.

### name vs id — a common source of confusion

When a user says "find my run called X", determine whether they mean:
- **`run.name`** (display name): Non-unique, human-friendly. Search with `filters={"display_name": "X"}` or `{"display_name": {"$regex": ".*X.*"}}`.
- **`run.id`** (unique ID): The 8-char hash. Access directly with `api.run("entity/project/run_id")`.

If the value looks like a short hash, it's probably an ID. If it's words or a descriptive string, it's a name.

---

## The Three Data Locations

Understanding where data lives is critical for answering "where's my metric?":

| What | Set by | When | Access via |
|------|--------|------|------------|
| **Config** | `wandb.init(config={...})` or `wandb.config.update()` | Once, at the start | `run.config` (dict) |
| **History** | `run.log({"loss": 0.5})` | Repeatedly, each training step | `run.history()`, `run.scan_history()` |
| **Summary** | Auto-set to last logged value, or `run.summary["key"] = val` | End of run (or overwritten) | `run.summary_metrics` (Public API) |

**Key rules:**
- **Config** = inputs/hyperparameters. Things that don't change during training. Avoid `.` in config keys — W&B uses dots for nested key flattening (e.g. `{"model.lr": 0.01}` and `{"model": {"lr": 0.01}}` collide).
- **History** = time-series metrics. Things logged at each step (loss, accuracy, learning rate schedules).
- **Summary** = final output metrics. By default, the *last* logged value for each key. Users can customize with `run.define_metric("loss", summary="min")` to keep the minimum instead.

**Common mistake**: A user asks "what's the best loss?" — if they used `define_metric(..., summary="min")`, check `summary_metrics`. If not, you need to scan history and compute it yourself.

---

## Two Distinct APIs

W&B has two Python APIs that serve different purposes. Do NOT confuse them:

| | Python SDK | Public API |
|---|---|---|
| **Import** | `import wandb` | `wandb.Api()` |
| **Purpose** | Real-time logging during training | Post-hoc querying, exporting, updating |
| **Creates runs?** | Yes (`wandb.init()`) | No (reads existing data) |
| **Key methods** | `wandb.init()`, `run.log()`, `run.finish()` | `api.run()`, `api.runs()`, `api.artifact()` |
| **When to use** | Writing training scripts | Analyzing results, building dashboards |

**As an agent, you almost always use the Public API** (`wandb.Api()`). You should never call `wandb.init()` to create runs — that's what Launch is for. The only exception is `weave.init()`, which initializes a Weave client for querying traces (different from `wandb.init()`).

---

## Artifacts

Versioned data objects tracked as inputs/outputs of runs. Used for datasets, models, checkpoints, and any file-based data.

### Key concepts

- **Versions** are 0-indexed: `v0`, `v1`, `v2`, etc. The first version logged is `v0`, not `v1`.
- **Aliases** are mutable labels: `"latest"` always points to the newest version. Users can create custom aliases like `"best"`, `"production"`.
- **Type** affects UI grouping: `"dataset"`, `"model"`, `"code"`, etc. Choose the right type.
- **Reference format**: `"entity/project/artifact_name:version_or_alias"` (e.g. `"my-team/my-project/model-weights:latest"`, `"my-team/my-project/model-weights:v3"`).

### Common operations

```python
api = wandb.Api(timeout=60)

# Fetch by name + version or alias
artifact = api.artifact("entity/project/my-dataset:latest")
artifact = api.artifact("entity/project/my-dataset:v3")

# Properties
artifact.name          # "my-dataset:v3"
artifact.type          # "dataset"
artifact.version       # "v3"
artifact.aliases       # ["latest", "best"]
artifact.metadata      # dict (user-set metadata)
artifact.size          # int (bytes)
artifact.created_at    # str (ISO timestamp)

# Download
local_path = artifact.download()
artifact.download(root="/tmp/data")

# Lineage — which run created/used this artifact
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()

# From a run — list what it produced
for art in run.logged_artifacts():
    print(art.name, art.type, art.aliases)

# Check existence without downloading
exists = api.artifact_exists("entity/project/my-model:v0")
```

### Registry

The **Registry** is an organization-level curated repository — a layer above project-level artifacts.

```
Organization
  +-- Registry (e.g. "Models", "Datasets")
        +-- Collection (e.g. "production-classifier")
              +-- Linked Artifact Versions (v0, v1, ...)
```

- Default registries: `"Models"` and `"Datasets"` (created automatically).
- Link artifacts to registry: `run.link_artifact(artifact, target_path="wandb-registry-Models/my-collection")`.
- Download from registry: `api.artifact("wandb-registry-Models/my-collection:latest")`.
- Registry artifacts use the `wandb-registry-` prefix in their path.

---

## Public API Caching

`wandb.Api()` caches network requests for performance. This matters when querying **live/running** data:

- Cached data can be stale if a run is still in progress.
- Call `api.flush()` to clear the cache and get fresh data.
- This is rarely needed for finished runs, but important when monitoring active training.

---

## Sweeps (Hyperparameter Tuning)

Automated hyperparameter search managed by W&B.

- **Define** a sweep config (search space, method, metric to optimize).
- **Initialize** with `sweep_id = wandb.sweep(config, project="my-project")`.
- **Run agents** with `wandb.agent(sweep_id, function=train)` — each agent runs multiple training functions with different hyperparameter combinations.
- **Methods**: `"bayes"` (Bayesian optimization), `"grid"` (exhaustive), `"random"` (random search).
- Sweep runs appear as normal runs in the project, tagged with the sweep ID.
- Query sweep runs: `api.runs(path, filters={"sweep": "sweep_id"})`. Or include sweep runs in general queries by using `include_sweeps=True`.

---

## W&B Tables

Interactive tabular data for visualization and comparison.

```python
# Log a table
table = wandb.Table(columns=["input", "prediction", "ground_truth"])
table.add_data("hello", "world", "world")
run.log({"predictions": table})
```

Tables support rich media (images, audio, HTML) in cells and appear as interactive panels in the W&B UI.

---

## Logging Limits

These hard limits affect code the agent writes. Exceeding them causes errors or silent data loss.

| Resource | Limit |
|----------|-------|
| Distinct metrics per project | 10,000 |
| Single `run.log()` call payload | 25 MB |
| Config size | 10 MB |
| Scalar data points per metric | 100,000 |
| Media data points (images, etc.) | 50,000 |
| Histogram data points | 10,000 |
| Runs per project (SaaS) | 100,000 |
| Table rows per single log call | 10,000 |
| Config key nesting (dots) | max 3 dots |
| Summary key nesting (dots) | max 4 dots |

**Rate limiting**: Calling `run.log()` more than a few times per second adds training latency. HTTP 429 is returned if rate limits are exceeded.

---

## define_metric (Custom X-Axes)

Controls which metric is the x-axis for another metric, and how summary values are computed.

```python
# Make "epoch" the x-axis for validation metrics
run.define_metric("val/loss", step_metric="epoch")
run.define_metric("val/acc", step_metric="epoch")

# Glob pattern — all train/* metrics use train/step as x-axis
run.define_metric("train/*", step_metric="train/step")

# Keep the minimum loss in summary instead of the last value
run.define_metric("val/loss", summary="min")
run.define_metric("val/acc", summary="max")
```

Without `define_metric`, all metrics use `_step` (global step) as x-axis and `summary` stores the last logged value.

---

## Distributed Training

Three approaches for multi-GPU/multi-node training:

| Approach | When to use | How |
|----------|-------------|-----|
| **Rank-0 only** | Simple, most common | Only call `wandb.init()` on rank 0 |
| **Process-per-run** | Need per-GPU metrics | Each process calls `wandb.init(group="experiment-1")` |
| **Shared mode** | Unified view, SDK 0.19.9+ | `wandb.init(mode="shared")` on all processes |

**Gotchas**: Max ~300 concurrent connections tested. Always call `run.finish()` to prevent hanging. Use `wandb.setup()` in main process when spawning subprocesses. Empty connections still push system metrics.

---

## Artifact TTL (Auto-Deletion)

```python
from datetime import timedelta

# Set TTL on an artifact
artifact.ttl = timedelta(days=30)
artifact.save()

# Remove TTL
artifact.ttl = None
artifact.save()
```

- Expiration calculated from `createdAt`, not last modification.
- Cannot set TTL on auto-generated artifacts (`run_table`, `code`, `job`, `wandb-*`).
- Registry-linked artifacts have TTL disabled (protected).
- Teams can set default TTL policies.

## Reference Artifacts (External Files)

Track files in S3/GCS/Azure/HTTP without uploading them to W&B. Only metadata (checksums, ETags) is stored.

```python
artifact = wandb.Artifact("my-dataset", type="dataset")
artifact.add_reference("s3://my-bucket/data/", max_objects=10000)
run.log_artifact(artifact)
```

- Default 10,000 object limit per reference (adjustable via `max_objects`).
- Uses default cloud credential mechanisms (AWS credentials, GCS service account, etc.).
- Rich media rendering unavailable in UI for reference artifacts.

---

## Weave Built-in Scorers

Install with `pip install weave[scorers]`. LLM-based scorers use litellm under the hood.

| Scorer | What it checks | Type |
|--------|---------------|------|
| `HallucinationFreeScorer` | Output grounded in provided context | LLM-as-judge |
| `SummarizationScorer` | Summary quality vs source | LLM-as-judge |
| `OpenAIModerationScorer` | Content policy violations | API call |
| `EmbeddingSimilarityScorer` | Semantic similarity | Embedding |
| `ValidJSONScorer` | Output is valid JSON | Programmatic |
| `ValidXMLScorer` | Output is valid XML | Programmatic |
| `PydanticScorer` | Output matches a Pydantic model | Programmatic |
| `ContextEntityRecallScorer` | Entity recall from context (RAGAS) | Programmatic |
| `ContextRelevancyScorer` | Context relevance to query (RAGAS) | LLM-as-judge |

Use `column_map` to remap column names if your data doesn't match expected field names.

---

## Weave Cost Tracking

Automatic LLM cost calculation from token usage metadata.

```python
# Add custom cost for a model
client.add_cost(
    llm_id="my-fine-tuned-model",
    prompt_token_cost=0.001,
    completion_token_cost=0.002,
)

# Query costs
costs = client.query_costs()

# Include costs in call queries
calls = client.get_calls(include_costs=True)
```

Token usage is automatically captured for supported providers (OpenAI, Anthropic, etc.) via autopatching.

---

## Weave EvaluationLogger (Incremental Evals)

Alternative to standard `Evaluation` when you don't have a predefined dataset — log predictions and scores incrementally.

```python
eval_logger = weave.EvaluationLogger(name="my-eval", scorers=[my_scorer])

with eval_logger.log_prediction(input={"question": "..."}) as prediction:
    output = my_model(prediction.input["question"])
    prediction.log_output(output)
    # Scores can be logged manually too
    prediction.log_score("custom_metric", {"value": 0.95})

eval_logger.finish()
```

**Gotcha**: Initialize EvaluationLogger BEFORE any LLM calls to capture token usage/cost. After `finish()` on a prediction, no more scores can be logged.

---

## Weave Autopatched Integrations

`weave.init()` automatically traces these with zero additional code:

**LLM Providers**: OpenAI, Anthropic, Google Gemini, Cohere, Mistral, Groq, Cerebras, Together AI, OpenRouter, Azure OpenAI, AWS Bedrock, NVIDIA NIM, LiteLLM, Ollama, vLLM

**Agent Frameworks**: LangChain, LlamaIndex, CrewAI, AutoGen, DSPy, Instructor, PydanticAI, OpenAI Agents SDK, Claude Agent SDK, Google ADK, Smolagents

When a user asks "why don't I see traces?" — check if their framework is in this list. If not, they need manual `@weave.op()` decoration.

---

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `WANDB_API_KEY` | Authentication |
| `WANDB_ENTITY` | Default entity |
| `WANDB_PROJECT` | Default project |
| `WANDB_BASE_URL` | Server URL (for self-hosted) |
| `WANDB_MODE` | `online`, `offline`, `disabled` |
| `WANDB_DIR` | Local storage directory |
| `WANDB_SILENT` / `WANDB_QUIET` | Suppress output |
| `WANDB_CONFIG_PATHS` | Comma-separated YAMLs loaded into config |
| `WANDB_IGNORE_GLOBS` | File patterns to exclude from code saving |
| `WANDB_DISABLE_GIT` | Skip git metadata collection |
| `WANDB_DISABLE_CODE` | Don't save code |

---

## Quick Disambiguation

| User says... | They probably mean... | Look here |
|---|---|---|
| "my runs" | Training runs in current project | `api.runs(path)` |
| "my traces" | Weave calls/traces | `client.get_calls()` |
| "my experiments" | Training runs (synonym for runs) | `api.runs(path)` |
| "my evals" | Weave Evaluation.evaluate calls | `client.get_calls(filter=CallsFilter(op_names=[...]))` |
| "my model" | An artifact of type "model" | `api.artifact("entity/project/model-name:latest")` |
| "my dataset" | An artifact of type "dataset" | `api.artifact("entity/project/dataset-name:latest")` |
| "this run's config" | Hyperparameters set at init | `run.config` |
| "this run's metrics" | Time-series logged values | `run.history(keys=[...])` or `run.summary_metrics` |
| "best loss" | Minimum loss across history (or summary if `define_metric` was used) | Check `summary_metrics` first, fall back to `scan_history` |
| "run X" (short hash) | Run by ID | `api.run("entity/project/X")` |
| "run X" (words) | Run by display name | `api.runs(path, filters={"display_name": "X"})` |
