# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""
Helper functions for W&B Launch — queue management, job submission, and run reproduction.

Usage:
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    from launch_helpers import (
        create_queue,
        recreate_queue_with_prioritization,
        submit_code_artifact_job,
        get_job_artifact,
        relaunch_run,
    )
"""

import json
import os
import re
import shlex
import shutil
import tempfile
import time
from copy import deepcopy

import wandb
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.create_job import _create_job

DEFAULT_BASE_IMAGE = "python:3.11-slim"
ACTIVE_AGENT_STATUSES = {"POLLING", "RUNNING"}
TERMINAL_RUN_STATES = {"finished", "failed", "crashed", "killed"}
WAIT_FOR_MODES = {"queued", "launched", "done"}

# ---------------------------------------------------------------------------
# URL parsing and run analysis
# ---------------------------------------------------------------------------

def parse_run_url(url):
    """Extract entity, project, and run_id from a W&B run URL.

    Accepts URLs like:
        https://wandb.ai/entity/project/runs/run_id
        https://wandb.ai/entity/project/runs/run_id?nw=...
        entity/project/run_id  (passthrough)

    Returns: (entity, project, run_id)
    """
    # Already a path like "entity/project/run_id"
    if "/" in url and "://" not in url:
        parts = url.strip("/").split("/")
        if len(parts) == 3:
            return tuple(parts)

    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)", url)
    if m:
        return m.group(1), m.group(2), m.group(3)
    raise ValueError(f"Cannot parse run URL: {url}")


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------



def _gql_query(api, query_str, variables=None):
    """Execute a GQL query using wandb_gql (preferred) or raw HTTP fallback.

    The wandb SDK vendors wandb_gql for query parsing. If that import fails
    (for example, in a minimal environment), fall back to raw HTTP POST to the GraphQL
    endpoint, which accepts query strings directly.
    """
    try:
        from wandb_gql import gql
        parsed = gql(query_str)
        return api.client.execute(parsed, variable_values=variables or {})
    except ImportError:
        pass

    # Fallback: raw HTTP request
    import requests
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    api_key = os.environ.get("WANDB_API_KEY", "")
    resp = requests.post(
        f"{base_url}/graphql",
        json={"query": query_str, "variables": variables or {}},
        auth=("api", api_key),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise Exception(str(data["errors"][0]))
    return data.get("data", {})


def _find_kubernetes_config(value):
    """Find the innermost Kubernetes resource config in a Launch queue config."""
    if not isinstance(value, dict):
        return None

    k8s = value.get("kubernetes")
    if isinstance(k8s, dict):
        nested = _find_kubernetes_config(k8s.get("resource_args"))
        if nested:
            return nested
        if any(key in k8s for key in ("namespace", "spec", "metadata")):
            return k8s

    for key in ("config", "resource_args"):
        nested = _find_kubernetes_config(value.get(key))
        if nested:
            return nested

    return None


def _extract_queue_resources(default_resource_config):
    """Extract namespace and resource defaults from a queue config."""
    config = (default_resource_config or {}).get("config")
    k8s = _find_kubernetes_config(config)
    if not k8s:
        return {"namespace": None, "limits": {}, "requests": {}}

    spec = k8s.get("spec") or {}
    containers = (
        ((spec.get("template") or {}).get("spec") or {}).get("containers") or []
    )
    resources = (containers[0].get("resources") if containers else {}) or {}
    return {
        "namespace": k8s.get("namespace"),
        "limits": resources.get("limits") or {},
        "requests": resources.get("requests") or {},
    }


def _queue_launch_resource_args(default_resource_config):
    """Return queue defaults in the shape launch_add expects."""
    config = (default_resource_config or {}).get("config")
    k8s = _find_kubernetes_config(config)
    if not k8s:
        return None
    return {"kubernetes": deepcopy(k8s)}


def _format_resource_defaults(queue):
    parts = []
    if queue.get("namespace"):
        parts.append(f"ns={queue['namespace']}")
    limits = queue.get("limits") or {}
    requests = queue.get("requests") or {}
    resources = limits or requests
    if resources.get("nvidia.com/gpu"):
        parts.append(f"gpu={resources['nvidia.com/gpu']}")
    if resources.get("cpu"):
        parts.append(f"cpu={resources['cpu']}")
    if resources.get("memory"):
        parts.append(f"mem={resources['memory']}")
    return " ".join(parts)


def _fetch_queues(entity):
    import base64

    api = wandb.Api()
    query_str = """
    query ListQueues($entityName: String!) {
        entity(name: $entityName) {
            launchProject {
                runQueues {
                    id
                    name
                    prioritizationMode
                    access
                    createdBy
                    defaultResourceConfig {
                        config
                    }
                    runQueueItems(first: 20) {
                        edges {
                            node { state }
                        }
                    }
                }
                launchAgents {
                    agentStatus
                    runQueues
                }
            }
        }
    }
    """
    result = _gql_query(api, query_str, {"entityName": entity})
    lp = result.get("entity", {}).get("launchProject", {}) or {}
    queues = lp.get("runQueues", [])
    agents = lp.get("launchAgents", [])

    # Map queue ID -> number of active agents. An agent is POLLING while idle
    # and RUNNING while it has claimed work; both mean the queue has capacity.
    active_agents_by_queue = {}
    for a in agents:
        if a.get("agentStatus") in ACTIVE_AGENT_STATUSES:
            for qid in (a.get("runQueues") or []):
                active_agents_by_queue[qid] = active_agents_by_queue.get(qid, 0) + 1

    # Get current user's numeric ID
    viewer_id = None
    try:
        decoded = base64.b64decode(api.viewer.id).decode()  # "User:248439"
        viewer_id = int(decoded.split(":")[1])
    except Exception:
        pass

    # Enrich and sort
    for q in queues:
        items = [e["node"] for e in q.get("runQueueItems", {}).get("edges", [])]
        state_counts = {}
        for item in items:
            state = str(item.get("state") or "").lower()
            if state:
                state_counts[state] = state_counts.get(state, 0) + 1
        q["item_state_counts"] = state_counts
        q["has_active_agent"] = active_agents_by_queue.get(q["id"], 0) > 0
        q["active_agent_count"] = active_agents_by_queue.get(q["id"], 0)
        q["is_mine"] = (q.get("createdBy") == viewer_id) if viewer_id else False
        # Extract resource defaults for display and optional explicit overrides.
        q.update(_extract_queue_resources(q.get("defaultResourceConfig")))
        q["launch_resource_args"] = _queue_launch_resource_args(
            q.get("defaultResourceConfig")
        )

    queues.sort(key=lambda q: (q["has_active_agent"], q["is_mine"]), reverse=True)
    return queues


def list_queues(entity):
    """List launch queues for an entity with active-agent and resource details.

    Queues with active agents and queues created by the current user are
    shown first, but callers must still choose an explicit queue_name.

    Prints a summary table and returns the full list of queue dicts.
    Each dict has keys: id, name, prioritizationMode, access, createdBy,
    has_active_agent, is_mine, item_state_counts, namespace, limits, requests,
    launch_resource_args.
    """
    queues = _fetch_queues(entity)

    # Print
    if not queues:
        print(f"  No queues found for entity '{entity}'")
        print("  Suggested Kubernetes queue name: wandb-launch-k8s")
        print(
            "  Create one with: python skills/wandb-primary/scripts/launch_helpers.py "
            f"create-queue {entity} --queue wandb-launch-k8s --namespace wandb-launch"
        )
        return queues

    for q in queues:
        flags = []
        if q["has_active_agent"]:
            flags.append(f"agents={q['active_agent_count']}")
        if q["is_mine"]:
            flags.append("yours")
        item_state_counts = q.get("item_state_counts") or {}
        if item_state_counts:
            states = ",".join(
                f"{state}:{count}" for state, count in sorted(item_state_counts.items())
            )
            flags.append(f"items={states}")
        resource_defaults = _format_resource_defaults(q)
        if resource_defaults:
            flags.append(resource_defaults)
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {q['name']:30s}{flag_str}")

    return queues


def get_queue(entity, queue_name):
    """Return one named queue. Raises if the queue does not exist."""
    queues = _fetch_queues(entity)
    queue = next((q for q in queues if q["name"] == queue_name), None)
    if not queue:
        raise ValueError(f"Queue '{queue_name}' not found for entity '{entity}'")
    return queue


def make_k8s_queue_config(namespace="wandb-launch", gpus=1, cpu=8, memory="80Gi"):
    """Build the default K8s resource config for a launch queue."""
    return {
        "kubernetes": {
            "namespace": namespace,
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "train",
                            "resources": {
                                "limits": {"nvidia.com/gpu": str(gpus), "cpu": str(cpu), "memory": memory},
                                "requests": {"nvidia.com/gpu": str(gpus), "cpu": str(cpu), "memory": memory},
                            },
                        }],
                    }
                }
            },
        }
    }


def create_queue(name, entity, namespace="wandb-launch", gpus=1, cpu=8, memory="80Gi", prioritization=True):
    """Create a W&B launch queue with optional prioritization."""
    api = wandb.Api()
    config = make_k8s_queue_config(namespace, gpus=gpus, cpu=cpu, memory=memory)
    queue = api.create_run_queue(
        name=name,
        type="kubernetes",
        entity=entity,
        config=config,
        prioritization_mode="V0" if prioritization else "DISABLED",
    )
    print(f"Created queue: {queue.name}, prioritization: {queue.prioritization_mode}")
    return queue


def recreate_queue_with_prioritization(name, entity, namespace="wandb-launch", **kwargs):
    """Delete and recreate a queue to enable prioritization.

    Prioritization cannot be toggled on an existing queue — it must be recreated.
    Remember to restart the agent after this (it loses its registration).
    """
    api = wandb.Api()
    q = api.run_queue(entity=entity, name=name)
    q.delete()
    print(f"Deleted queue: {name}")
    return create_queue(name, entity, namespace, prioritization=True, **kwargs)



def inspect_queue(name, entity):
    """Print queue details."""
    q_info = get_queue(entity, name)
    api = wandb.Api()
    q = api.run_queue(entity=entity, name=name)
    print(f"Name:            {q.name}")
    print(f"Type:            {q.type}")
    print(f"Prioritization:  {q.prioritization_mode}")
    print(f"Active agents:   {q_info.get('active_agent_count', 0)}")
    defaults = _format_resource_defaults(q_info)
    print(f"Defaults:        {defaults or '(none shown)'}")
    return q


# ---------------------------------------------------------------------------
# Launch argument normalization
# ---------------------------------------------------------------------------


def _entry_point_args(entrypoint):
    if entrypoint is None:
        return None
    if isinstance(entrypoint, str):
        return shlex.split(entrypoint)
    return list(entrypoint)


def _normalize_requirements(requirements, code_dir, ensure_wandb=True):
    req_path = os.path.join(code_dir, "requirements.txt")

    if requirements is None and os.path.exists(req_path):
        with open(req_path) as f:
            lines = [line.strip() for line in f if line.strip()]
    elif requirements is None:
        lines = []
    elif isinstance(requirements, str):
        lines = [requirements]
    else:
        lines = [str(req).strip() for req in requirements if str(req).strip()]

    if ensure_wandb:
        has_wandb = any(re.match(r"(?i)^wandb(?:\[.*\])?(?:[<>=!~ ].*)?$", line) for line in lines)
        if not has_wandb:
            lines.append("wandb")

    if lines:
        with open(req_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def _copy_code_files(code_files, temp_dir):
    for file_path in code_files:
        src = os.path.abspath(file_path)
        if not os.path.isfile(src):
            raise FileNotFoundError(file_path)
        rel = file_path if not os.path.isabs(file_path) else os.path.basename(file_path)
        rel = os.path.normpath(rel)
        if rel.startswith("..") or os.path.isabs(rel):
            raise ValueError(f"Code file must be inside the working directory: {file_path}")
        dst = os.path.join(temp_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _require_queue_name(queue_name):
    if not queue_name:
        raise ValueError(
            "queue_name is required. Run list_queues(entity), show the options to "
            "the user, and ask which queue to use."
        )


def _resolve_base_image(base_image):
    if base_image:
        return base_image
    print(f"Using default base image: {DEFAULT_BASE_IMAGE}")
    return DEFAULT_BASE_IMAGE


def _normalize_launch_resource_args(resource_args):
    if not isinstance(resource_args, dict):
        return resource_args

    if "kubernetes" in resource_args:
        k8s = _find_kubernetes_config(resource_args)
        if k8s:
            return {"kubernetes": deepcopy(k8s)}
        return resource_args

    if any(
        key in resource_args
        for key in ("apiVersion", "kind", "metadata", "namespace", "spec")
    ):
        return {"kubernetes": deepcopy(resource_args)}

    return resource_args


def _resolve_resource_args(
    *,
    entity=None,
    queue_name=None,
    namespace=None,
    gpus=None,
    cpu=None,
    memory=None,
    k8s_secrets=None,
):
    queue = get_queue(entity, queue_name) if entity and queue_name else None
    has_override = any(v is not None for v in (gpus, cpu, memory)) or bool(k8s_secrets)
    if not has_override:
        defaults = (queue or {}).get("launch_resource_args")
        if defaults:
            print(
                f"Using queue defaults from '{queue_name}': "
                f"{_format_resource_defaults(queue)}"
            )
            return _normalize_launch_resource_args(defaults)
        return None

    resolved_namespace = namespace
    if resolved_namespace is None and queue:
        resolved_namespace = queue.get("namespace")
    if not resolved_namespace:
        raise ValueError(
            "namespace is required when overriding Launch resources; pass "
            "namespace from list_queues()/inspect_queue() or leave gpus/cpu/memory unset to use "
            "queue defaults."
        )

    default_resources = ((queue or {}).get("limits") or (queue or {}).get("requests") or {})
    return _make_resource_args(
        resolved_namespace,
        gpus=default_resources.get("nvidia.com/gpu", 1) if gpus is None else gpus,
        cpu=default_resources.get("cpu", 8) if cpu is None else cpu,
        memory=default_resources.get("memory", "80Gi") if memory is None else memory,
        k8s_secrets=k8s_secrets,
    )


def _add_resource_args(launch_kwargs, **kwargs):
    resource_args = _resolve_resource_args(**kwargs)
    if resource_args is not None:
        launch_kwargs["resource_args"] = resource_args


# ---------------------------------------------------------------------------
# Resource args for explicit resource overrides
# ---------------------------------------------------------------------------

def _make_resource_args(namespace, gpus=1, cpu=8, memory="80Gi", k8s_secrets=None):
    """Build resource_args for launch_add().

    Public launch helpers expose gpus/cpu/memory/k8s_secrets directly and keep
    this SDK-specific shape internal.

    Args:
        namespace: K8s namespace. Use list_queues() or inspect_queue() to get it.
        gpus: Number of GPUs.
        cpu: Number of CPUs.
        memory: Memory string (e.g. "80Gi").
        k8s_secrets: Dict mapping env var name to (secret_name, secret_key) tuples.
                     Example: {"HF_TOKEN": ("hf-token", "token")}
    """
    env = [{
        "name": "WANDB_API_KEY",
        "valueFrom": {
            "secretKeyRef": {
                "key": "password",
                "name": "wandb-api-key-wandb-launch",
            }
        },
    }]
    for env_name, (secret_name, secret_key) in (k8s_secrets or {}).items():
        env.append({
            "name": env_name,
            "valueFrom": {
                "secretKeyRef": {
                    "name": secret_name,
                    "key": secret_key,
                }
            },
        })

    return {
        "kubernetes": {
            "namespace": namespace,
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "train",
                            "env": env,
                            "resources": {
                                "limits": {"nvidia.com/gpu": str(gpus), "cpu": str(cpu), "memory": memory},
                                "requests": {"nvidia.com/gpu": str(gpus), "cpu": str(cpu), "memory": memory},
                            },
                        }]
                    }
                }
            },
        }
    }


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_code_artifact_job(
    code_files,
    entrypoint,
    entity,
    project,
    queue_name,
    job_name,
    namespace=None,
    base_image=None,
    requirements=None,
    ensure_wandb=True,
    config=None,
    gpus=None,
    cpu=None,
    memory=None,
    k8s_secrets=None,
    wait_for="launched",
    timeout_seconds=300,
    poll_seconds=10,
):
    """
    Create a code artifact job and submit it to a launch queue.

    Args:
        code_files: List of file paths to include in the artifact.
        entrypoint: Entrypoint string, e.g. "python train.py".
        entity: W&B entity.
        project: W&B project.
        queue_name: Launch queue name. Run list_queues(entity) first.
        namespace: K8s namespace. Only needed when overriding resources.
        job_name: Name for the job artifact.
        base_image: Docker base image. Defaults to python:3.11-slim.
        requirements: List of pip packages to install in the container.
                      Only include packages NOT already in the base image.
                      _create_job reads requirements.txt directly — it does NOT inspect the venv.
        ensure_wandb: Add wandb to requirements.txt when missing.
        config: Dict of hyperparameters to pass to the run.
        gpus/cpu/memory: Optional resource overrides. Omit to use queue defaults.
        wait_for: "queued", "launched", or "done". Defaults to "launched".
        timeout_seconds: Max wait time for wait_for="launched" or "done".
        poll_seconds: Seconds between status checks.

    Returns:
        Status dict with queue item, run URL, run state, launched, and done.
    """
    _require_queue_name(queue_name)
    resolved_base_image = _resolve_base_image(base_image)

    with tempfile.TemporaryDirectory() as temp_dir:
        _copy_code_files(code_files, temp_dir)
        _normalize_requirements(requirements, temp_dir, ensure_wandb=ensure_wandb)

        # Step 1: Create/update the job artifact
        api = wandb.InternalApi()
        kwargs = dict(
            api=api,
            job_type="code",
            path=temp_dir,
            entity=entity,
            project=project,
            name=job_name,
            entrypoint=entrypoint,
        )
        kwargs["base_image"] = resolved_base_image

        artifact, action, aliases = _create_job(**kwargs)
        print(f"Job artifact: {artifact.name} ({action})")

    # Step 2: Enqueue
    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_name}:latest",
        entity=entity,
        project=project,
        queue_name=queue_name,
        entry_point=_entry_point_args(entrypoint),
    )
    _add_resource_args(
        launch_kwargs,
        entity=entity,
        queue_name=queue_name,
        namespace=namespace,
        gpus=gpus,
        cpu=cpu,
        memory=memory,
        k8s_secrets=k8s_secrets,
    )
    if config:
        launch_kwargs["config"] = {"overrides": {"run_config": config}}

    queued_run = launch_add(**launch_kwargs)
    print("Queued! Run ID will be assigned when started.")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status later: check_launch('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")
    return wait_for_launch(
        entity,
        project,
        queue_name,
        queued_run.id,
        wait_for=wait_for,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


# ---------------------------------------------------------------------------
# Run reproduction
# ---------------------------------------------------------------------------

def inspect_job_artifact(artifact_path):
    """
    Download and inspect a job artifact's metadata.

    Args:
        artifact_path: Full artifact path, e.g. "ENTITY/PROJECT/JOB_NAME:latest"

    Returns:
        Dict with keys: entrypoint, base_image, source_artifact, runtime, requirements
    """
    import json
    import tempfile

    api = wandb.Api()
    art = api.artifact(artifact_path)
    d = art.download(tempfile.mkdtemp())

    with open(os.path.join(d, "wandb-job.json")) as f:
        job_meta = json.load(f)

    reqs_path = os.path.join(d, "requirements.frozen.txt")
    requirements = []
    if os.path.exists(reqs_path):
        with open(reqs_path) as f:
            requirements = [line.strip() for line in f if line.strip()]

    source = job_meta.get("source", {})
    info = {
        "entrypoint": source.get("entrypoint"),
        "base_image": source.get("base_image"),
        "source_artifact": source.get("artifact"),
        "runtime": job_meta.get("runtime"),
        "requirements": requirements,
    }

    print(f"Artifact:   {artifact_path}")
    print(f"Entrypoint: {info['entrypoint']}")
    print(f"Base image: {info['base_image']}")
    print(f"Runtime:    {info['runtime']}")
    print(f"Deps:       {len(info['requirements'])} packages")
    return info


# ---------------------------------------------------------------------------
# Code artifact download / edit / re-upload
# ---------------------------------------------------------------------------

def download_code_artifact(job_artifact_path):
    """Download the source code from a job artifact.

    Resolves the job's source code artifact and downloads it to a local directory.
    Returns a dict with the download path, file list, entrypoint, base_image,
    and the source artifact name (needed to create a new job version).

    Args:
        job_artifact_path: Full job artifact path, e.g. "wandb/project/job-name:v19"

    Returns:
        Dict with keys: code_dir, files, entrypoint, base_image, source_artifact_name,
        entity, project, job_name.
    """
    from wandb_gql import gql

    api = wandb.Api()

    # Download job metadata
    job_art = api.artifact(job_artifact_path)
    d = job_art.download(tempfile.mkdtemp())
    with open(os.path.join(d, "wandb-job.json")) as f:
        meta = json.load(f)

    source = meta.get("source", {})
    source_ref = source.get("artifact", "")
    entrypoint = source.get("entrypoint")
    base_image = source.get("base_image")

    # Resolve source artifact ID to a downloadable artifact
    artifact_id = source_ref.replace("wandb-artifact://_id/", "")
    query = gql("""
    query GetArtifact($id: ID!) {
        artifact(id: $id) {
            artifactSequence { name project { name entityName } }
            aliases { alias }
        }
    }
    """)
    result = api.client.execute(query, variable_values={"id": artifact_id})
    art_data = result["artifact"]
    seq = art_data["artifactSequence"]
    entity = seq["project"]["entityName"]
    project = seq["project"]["name"]
    art_name = seq["name"]
    aliases = [a["alias"] for a in art_data["aliases"]]
    version = aliases[0] if aliases else "latest"

    # Download source code
    source_art = api.artifact(f"{entity}/{project}/{art_name}:{version}")
    code_dir = source_art.download(tempfile.mkdtemp())

    files = []
    for root, _, filenames in os.walk(code_dir):
        for fn in filenames:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, code_dir)
            files.append(rel)

    # Parse job name from the artifact path
    parts = job_artifact_path.split("/")
    job_name = parts[2].split(":")[0] if len(parts) >= 3 else "job"

    print(f"Downloaded code from: {entity}/{project}/{art_name}:{version}")
    print(f"Code directory: {code_dir}")
    print(f"Files: {files}")
    print(f"Entrypoint: {entrypoint}")
    print(f"Base image: {base_image}")

    return {
        "code_dir": code_dir,
        "files": files,
        "entrypoint": entrypoint,
        "base_image": base_image,
        "source_artifact_name": f"{entity}/{project}/{art_name}",
        "entity": entity,
        "project": project,
        "job_name": job_name,
    }


def create_and_launch_modified_job(
    code_dir,
    entrypoint,
    entity,
    project,
    queue_name,
    job_name,
    namespace=None,
    base_image=None,
    requirements=None,
    ensure_wandb=True,
    config=None,
    gpus=None,
    cpu=None,
    memory=None,
    k8s_secrets=None,
    wait_for="launched",
    timeout_seconds=300,
    poll_seconds=10,
):
    """Create a new job artifact from a (modified) code directory and launch it.

    This is the code-change equivalent of relaunch_run(). Use it after:
    1. download_code_artifact() to get the code
    2. Editing the files in code_dir
    3. Calling this to upload + launch

    Args:
        code_dir: Path to directory with modified code files.
        entrypoint: Entrypoint string, e.g. "python model.py".
        entity: W&B entity.
        project: W&B project.
        queue_name: Launch queue name. Run list_queues(entity) first.
        namespace: K8s namespace. Only needed when overriding resources.
        job_name: Name for the job artifact.
        base_image: Docker base image. Defaults to python:3.11-slim.
        requirements: List of pip packages (only those NOT in base image).
        ensure_wandb: Add wandb to requirements.txt when missing.
        config: Dict of run config overrides.
        gpus/cpu/memory: Optional resource overrides. Omit to use queue defaults.
        k8s_secrets: Dict mapping env var to (secret_name, secret_key).
        wait_for: "queued", "launched", or "done". Defaults to "launched".
        timeout_seconds: Max wait time for wait_for="launched" or "done".
        poll_seconds: Seconds between status checks.
    """
    _require_queue_name(queue_name)
    resolved_base_image = _resolve_base_image(base_image)
    _normalize_requirements(requirements, code_dir, ensure_wandb=ensure_wandb)

    # Create job artifact
    api_internal = wandb.InternalApi()
    kwargs = dict(
        api=api_internal,
        job_type="code",
        path=code_dir,
        entity=entity,
        project=project,
        name=job_name,
        entrypoint=entrypoint,
    )
    kwargs["base_image"] = resolved_base_image

    artifact, action, aliases = _create_job(**kwargs)
    print(f"Job artifact: {artifact.name} ({action})")

    # Launch
    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_name}:latest",
        entity=entity,
        project=project,
        queue_name=queue_name,
        entry_point=_entry_point_args(entrypoint),
    )
    _add_resource_args(
        launch_kwargs,
        entity=entity,
        queue_name=queue_name,
        namespace=namespace,
        gpus=gpus,
        cpu=cpu,
        memory=memory,
        k8s_secrets=k8s_secrets,
    )
    if config:
        launch_kwargs["config"] = {"overrides": {"run_config": config}}

    queued_run = launch_add(**launch_kwargs)
    print("Queued modified job!")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status: check_launch('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")
    return wait_for_launch(
        entity,
        project,
        queue_name,
        queued_run.id,
        wait_for=wait_for,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def launch_job_artifact(
    artifact_path,
    queue_name,
    namespace=None,
    entrypoint=None,
    config=None,
    gpus=None,
    cpu=None,
    memory=None,
    k8s_secrets=None,
    wait_for="launched",
    timeout_seconds=300,
    poll_seconds=10,
):
    """
    Launch a job directly from an artifact path.

    Args:
        artifact_path: Full artifact path, e.g. "ENTITY/PROJECT/JOB_NAME:latest"
        queue_name: Launch queue to submit to. Run list_queues(entity) first.
        namespace: K8s namespace. Only needed when overriding resources.
        entrypoint: Override entrypoint. If None, uses the artifact's entrypoint.
        config: Dict of hyperparameters to pass to the run.
        gpus/cpu/memory: Optional resource overrides. Omit to use queue defaults.
        k8s_secrets: Dict mapping env var name to (secret_name, secret_key) tuples.
        wait_for: "queued", "launched", or "done". Defaults to "launched".
        timeout_seconds: Max wait time for wait_for="launched" or "done".
        poll_seconds: Seconds between status checks.

    Returns:
        Status dict with queue item, run URL, run state, launched, and done.
    """
    # Parse entity/project from artifact path (e.g. "ENTITY/PROJECT/JOB_NAME:latest")
    parts = artifact_path.split("/")
    entity = parts[0]
    project = parts[1]
    _require_queue_name(queue_name)

    # Resolve entrypoint from artifact metadata if not provided
    resolved_entrypoint = entrypoint
    if resolved_entrypoint is None:
        info = inspect_job_artifact(artifact_path)
        resolved_entrypoint = info["entrypoint"]
        print(f"Using entrypoint from artifact: {resolved_entrypoint}")

    launch_kwargs = dict(
        job=artifact_path,
        entity=entity,
        project=project,
        queue_name=queue_name,
    )
    entry_point_args = _entry_point_args(resolved_entrypoint)
    if entry_point_args is not None:
        launch_kwargs["entry_point"] = entry_point_args
    _add_resource_args(
        launch_kwargs,
        entity=entity,
        queue_name=queue_name,
        namespace=namespace,
        gpus=gpus,
        cpu=cpu,
        memory=memory,
        k8s_secrets=k8s_secrets,
    )
    if config:
        launch_kwargs["config"] = {"overrides": {"run_config": config}}

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued job from artifact: {artifact_path}")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status later: check_launch('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")
    return wait_for_launch(
        entity,
        project,
        queue_name,
        queued_run.id,
        wait_for=wait_for,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


DEFAULT_LAUNCH_PROJECT_NAME = "model-registry"


def _ui_base_url():
    return os.environ.get("WANDB_UI_BASE_URL", "https://wandb.ai").rstrip("/")


def _ui_url(path):
    return f"{_ui_base_url()}/{path.lstrip('/')}"


def _queue_url(entity, queue_id, tab=None):
    path = f"/{entity}/launch/{queue_id}"
    if tab and tab != "runs":
        path += f"/{tab}"
    return _ui_url(path)


def _agent_url(entity, agent_id, tab=None):
    path = f"/{entity}/launch/agents/{agent_id}"
    if tab and tab != "overview":
        path += f"/{tab}"
    return _ui_url(path)


def _run_url(entity, project, run_id, tab=None):
    path = f"/{entity}/{project}/runs/{run_id}"
    if tab:
        path += f"/{tab}"
    return _ui_url(path)


def _job_url_from_run_spec(run_spec):
    job_path = (run_spec or {}).get("job")
    job_collection_id = (run_spec or {}).get("_wandb_job_collection_id")
    if not job_path or not job_collection_id:
        return None

    try:
        job_entity, job_project, versioned_job = job_path.split("/", 2)
        _, alias = versioned_job.rsplit(":", 1)
    except ValueError:
        return None

    return _ui_url(
        f"/{job_entity}/{job_project}/jobs/{job_collection_id}/version_details/{alias}"
    )


def _fetch_launch_queue_item(entity, queue_name, run_queue_item_id):
    """Fetch the queue item fields used by the Launch queue details drawer."""
    api = wandb.Api()
    query_str = """
    query LaunchQueueItemDebug(
        $entityName: String!
        $projectName: String!
        $runQueueName: String!
        $runQueueItemID: ID!
    ) {
        project(name: $projectName, entityName: $entityName) {
            id
            runQueue(name: $runQueueName) {
                id
                name
                runQueueItem(id: $runQueueItemID) {
                    id
                    createdAt
                    updatedAt
                    runSpec
                    state
                    associatedRunId
                    launchAgentId
                    priority
                    templateVariableValues
                    error {
                        message
                        stage
                        filePaths
                    }
                    warnings {
                        message
                        stage
                        filePaths
                    }
                }
            }
        }
    }
    """
    result = _gql_query(
        api,
        query_str,
        {
            "entityName": entity,
            "projectName": DEFAULT_LAUNCH_PROJECT_NAME,
            "runQueueName": queue_name,
            "runQueueItemID": run_queue_item_id,
        },
    )
    run_queue = (result.get("project") or {}).get("runQueue") or {}
    item = run_queue.get("runQueueItem")
    if item:
        item["_queue_id"] = run_queue.get("id")
        item["_queue_name"] = run_queue.get("name")
    return item


def _fetch_launch_agent(agent_id):
    if not agent_id:
        return None
    api = wandb.Api()
    query_str = """
    query LaunchAgentDebug($agentId: ID!) {
        launchAgent(id: $agentId) {
            id
            name
            createdAt
            updatedAt
            heartbeatAt
            runQueues
            hostname
            agentStatus
            stopPolling
            agentConfig
            version
        }
    }
    """
    result = _gql_query(api, query_str, {"agentId": agent_id})
    return result.get("launchAgent")


def _print_launch_ui_links(entity, project, queue_id, item=None, agent=None):
    run_spec = (item or {}).get("runSpec") or {}
    run_entity = run_spec.get("entity") or entity
    run_project = run_spec.get("project") or project
    run_id = (item or {}).get("associatedRunId")
    agent_id = (item or {}).get("launchAgentId")
    agent_name = (agent or {}).get("name")

    print("UI links:")
    if queue_id:
        print(f"  Queue runs:   {_queue_url(entity, queue_id)}")
        print(f"  Queue agents: {_queue_url(entity, queue_id, 'agents')}")
        print(f"  Queue config: {_queue_url(entity, queue_id, 'config')}")
        item_id = (item or {}).get("id")
        details_hint = f" for item {item_id}" if item_id else " for this item"
        print(f"  Queue item:   open Queue runs and click Details{details_hint}")
    if agent_id:
        print(f"  Agent:        {_agent_url(entity, agent_id)}")
        print(f"  Agent logs:   {_agent_url(entity, agent_id, 'logs')}")
    if agent_name:
        print(
            f"  Agent run:    "
            f"{_run_url(entity, DEFAULT_LAUNCH_PROJECT_NAME, agent_name, 'logs')}"
        )
    if run_id:
        print(f"  Run:          {_run_url(run_entity, run_project, run_id)}")
        print(f"  Run logs:     {_run_url(run_entity, run_project, run_id, 'logs')}")

    job_url = _job_url_from_run_spec(run_spec)
    if job_url:
        print(f"  Job version:  {job_url}")


def _print_launch_issues(item):
    if not item:
        return

    entries = []
    error = item.get("error")
    if error and error.get("message"):
        entries.append(("Error", error))
    for warning in item.get("warnings") or []:
        if warning and warning.get("message"):
            entries.append(("Warning", warning))

    if not entries:
        print("Issues: none reported on the queue item")
        return

    print("Issues:")
    for kind, info in entries:
        message = str(info.get("message") or "").strip()
        if len(message) > 500:
            message = message[:500] + "..."
        file_paths = ", ".join(info.get("filePaths") or [])
        suffix = f" files=[{file_paths}]" if file_paths else ""
        print(f"  {kind} stage={info.get('stage')}{suffix}: {message}")


def _is_terminal_run_state(run_state):
    return str(run_state or "").lower() in TERMINAL_RUN_STATES


def check_launch(entity, project, queue_name, run_queue_item_id):
    """Check one Launch queue item and its associated W&B run.

    Queue item state and W&B run state are different. Successful Launch queue
    items usually remain CLAIMED; use launched/done/run_state for execution.

    Returns:
        Dict with keys: queue_item_id, queue_state, launched, done, run_id,
        run_url, run_state, run, check_command.
        Returns None if the queue item can't be found.
    """
    api = wandb.Api()
    try:
        item = _fetch_launch_queue_item(entity, queue_name, run_queue_item_id)
    except Exception as e:
        print(f"Could not fetch Launch queue-item details: {e}")
        item = None

    try:
        agent = _fetch_launch_agent(item.get("launchAgentId")) if item else None
    except Exception as e:
        print(f"Could not fetch Launch agent details: {e}")
        agent = None
    queue_id = item.get("_queue_id") if item else None

    if item:
        state = str(item.get("state") or "").lower()
    else:
        # Fallback for old servers that do not expose runQueueItem by id.
        from wandb.apis.public import QueuedRun

        qr = QueuedRun(api.client, entity, project, queue_name, run_queue_item_id)
        try:
            state = qr.state
            item = qr._get_item()
        except Exception as e:
            print(f"Could not find queue item: {e}")
            return None

    result = {
        "queue_item_id": run_queue_item_id,
        "queue_state": state,
        "launched": False,
        "done": state == "failed",
        "run_state": None,
        "run_id": None,
        "run_url": None,
        "run": None,
        "check_command": _check_command(entity, project, queue_name, run_queue_item_id),
    }
    print(f"Queue item state: {state}")
    _print_launch_ui_links(entity, project, queue_id, item=item, agent=agent)
    _print_launch_issues(item)
    if state == "claimed":
        print(
            "CLAIMED means the agent acknowledged the queue item and assigned a run; "
            "check the associated run state for execution status."
        )

    if state in ("finished", "claimed", "leased"):
        # Try to get the associated run
        try:
            run_id = item.get("associatedRunId") if item else None
            run_spec = (item or {}).get("runSpec") or {}
            run_entity = run_spec.get("entity") or entity
            run_project = run_spec.get("project") or project
            if run_id:
                result["run_id"] = run_id
                result["run_url"] = _run_url(run_entity, run_project, run_id)
                result["launched"] = True
                print(f"Run ID: {run_id}")
                print(f"Run URL: {result['run_url']}")
                try:
                    run = api.run(f"{run_entity}/{run_project}/{run_id}")
                    result["run"] = run
                    result["run_state"] = run.state
                    result["done"] = _is_terminal_run_state(run.state)
                    print(f"Run state: {run.state}")
                    # Print key metrics from summary
                    summary = {k: v for k, v in run.summary.items()
                               if not k.startswith("_") and not isinstance(v, dict)}
                    if summary:
                        print(f"Metrics: {json.dumps(summary, indent=2, default=str)}")
                except Exception:
                    print("Run created but not yet accessible via API")
            else:
                print("Run not yet assigned (still queued)")
        except Exception:
            print("Could not fetch run details")

    elif state == "pending":
        print("Job is still waiting in queue")
    elif state == "failed":
        print("Job failed — check the launch agent logs")

    return result


def _check_command(entity, project, queue_name, run_queue_item_id):
    return (
        "python skills/wandb-primary/scripts/launch_helpers.py check "
        f"{entity} {project} {queue_name} {run_queue_item_id}"
    )


def _queued_launch_status(entity, project, queue_name, run_queue_item_id):
    return {
        "queue_item_id": run_queue_item_id,
        "queue_state": None,
        "launched": False,
        "done": False,
        "run_state": None,
        "run_id": None,
        "run_url": None,
        "run": None,
        "check_command": _check_command(entity, project, queue_name, run_queue_item_id),
    }


def wait_for_launch(
    entity,
    project,
    queue_name,
    run_queue_item_id,
    wait_for="launched",
    timeout_seconds=300,
    poll_seconds=10,
):
    """Wait for a Launch queue item to be queued, launched, or done.

    Modes:
      - queued: return immediately after enqueue.
      - launched: return when Launch assigns an associated W&B run ID.
      - done: return when the associated W&B run reaches a terminal state.
    """
    if wait_for not in WAIT_FOR_MODES:
        raise ValueError(f"wait_for must be one of {sorted(WAIT_FOR_MODES)}")

    if wait_for == "queued":
        return _queued_launch_status(entity, project, queue_name, run_queue_item_id)

    deadline = time.time() + timeout_seconds
    last_status = None
    while True:
        last_status = check_launch(entity, project, queue_name, run_queue_item_id)
        if last_status is None:
            raise RuntimeError(f"Could not find queue item {run_queue_item_id}")

        if wait_for == "launched" and (
            last_status["launched"] or last_status["done"]
        ):
            return last_status
        if wait_for == "done" and last_status["done"]:
            return last_status

        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Launch job to become {wait_for}; "
                f"last status={json.dumps(_compact_launch_status(last_status), default=str)}"
            )

        time.sleep(poll_seconds)


def _compact_launch_status(status):
    return {
        key: status.get(key)
        for key in (
            "queue_item_id",
            "queue_state",
            "launched",
            "done",
            "run_id",
            "run_url",
            "run_state",
        )
    }


def get_job_artifact(run_path):
    """
    Check if a W&B run has a job artifact. Returns the artifact or None.

    Args:
        run_path: "<entity>/<project>/<run_id>"
    """
    api = wandb.Api()
    run = api.run(run_path)
    for art in run.used_artifacts():
        if art.type == "job":
            return art
    return None


def relaunch_run(
    run_path,
    queue_name,
    namespace=None,
    config=None,
    entrypoint=None,
    gpus=None,
    cpu=None,
    memory=None,
    k8s_secrets=None,
    wait_for="launched",
    timeout_seconds=300,
    poll_seconds=10,
):
    """
    Re-launch an existing W&B run using its job artifact.

    Args:
        run_path: "<entity>/<project>/<run_id>"
        queue_name: Launch queue to submit to. Run list_queues(entity) first.
        namespace: K8s namespace. Only needed when overriding resources.
        config: Dict of config overrides. Merged on top of the original run's config.
                e.g. {"epochs": 100, "lr": 0.001}
        entrypoint: Override entrypoint. Defaults to job artifact's entrypoint.
        gpus/cpu/memory: Optional resource overrides. Omit to use queue defaults.
        wait_for: "queued", "launched", or "done". Defaults to "launched".
        timeout_seconds: Max wait time for wait_for="launched" or "done".
        poll_seconds: Seconds between status checks.
    """
    api = wandb.Api()
    run = api.run(run_path)
    entity, project, _ = run_path.split("/")
    _require_queue_name(queue_name)

    job_artifact = get_job_artifact(run_path)
    if job_artifact is None:
        raise ValueError(f"No job artifact found for run {run_path}. Create one first.")

    print(f"Found job artifact: {job_artifact.name}")

    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_artifact.name}",
        entity=entity,
        project=project,
        queue_name=queue_name,
        config={"overrides": {"run_config": {**dict(run.config), **(config or {})}}},
    )
    _add_resource_args(
        launch_kwargs,
        entity=entity,
        queue_name=queue_name,
        namespace=namespace,
        gpus=gpus,
        cpu=cpu,
        memory=memory,
        k8s_secrets=k8s_secrets,
    )
    if entrypoint:
        launch_kwargs["entry_point"] = _entry_point_args(entrypoint)

    run_config = launch_kwargs["config"]["overrides"]["run_config"]
    if config:
        print(f"Config overrides: {json.dumps(config)}")
    print(f"Final run_config: {json.dumps(run_config)}")

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued reproduction of run {run_path}")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status later: check_launch('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")
    return wait_for_launch(
        entity,
        project,
        queue_name,
        queued_run.id,
        wait_for=wait_for,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


# ---------------------------------------------------------------------------
# CLI entry point — run directly instead of writing wrapper scripts
# ---------------------------------------------------------------------------

def _cli():
    """CLI for common launch operations.

    Usage:
        python launch_helpers.py list-queues <entity>
        python launch_helpers.py create-queue <entity> --queue wandb-launch-k8s --namespace <namespace>
        python launch_helpers.py relaunch <run_url> --queue <queue> [--config '{"epochs": 100}']
        python launch_helpers.py inspect <run_url>
    """
    import argparse

    parser = argparse.ArgumentParser(description="W&B Launch helpers CLI")
    sub = parser.add_subparsers(dest="command")

    p_re = sub.add_parser("relaunch", help="Relaunch a run with optional config overrides")
    p_re.add_argument("run_url", help="W&B run URL or entity/project/run_id")
    p_re.add_argument("--config", type=json.loads, default=None,
                       help='JSON config overrides, e.g. \'{"epochs": 100}\'')
    p_re.add_argument("--queue", required=True, help="Queue name. Run list-queues first.")
    p_re.add_argument(
        "--wait-for",
        choices=sorted(WAIT_FOR_MODES),
        default="launched",
        help="queued returns immediately, launched waits for a run URL, done waits for terminal run state",
    )

    p_lq = sub.add_parser("list-queues", help="List launch queues for an entity")
    p_lq.add_argument("entity", help="W&B entity name")

    p_cq = sub.add_parser("create-queue", help="Create a Kubernetes launch queue")
    p_cq.add_argument("entity", help="W&B entity name")
    p_cq.add_argument("--queue", default="wandb-launch-k8s", help="Queue name")
    p_cq.add_argument("--namespace", default="wandb-launch", help="Kubernetes namespace")
    p_cq.add_argument("--gpus", type=int, default=1, help="GPUs per job")
    p_cq.add_argument("--cpu", type=int, default=8, help="CPUs per job")
    p_cq.add_argument("--memory", default="80Gi", help="Memory per job")
    p_cq.add_argument(
        "--no-prioritization",
        action="store_true",
        help="Disable queue prioritization",
    )

    p_in = sub.add_parser("inspect", help="Inspect a run's job artifact")
    p_in.add_argument("run_url", help="W&B run URL or entity/project/run_id")

    p_ck = sub.add_parser("check", help="Check status of a launched run")
    p_ck.add_argument("entity", help="W&B entity")
    p_ck.add_argument("project", help="W&B project")
    p_ck.add_argument("queue_name", help="Queue name")
    p_ck.add_argument("item_id", help="Queue item ID")

    args = parser.parse_args()

    if args.command == "list-queues":
        list_queues(args.entity)

    elif args.command == "create-queue":
        create_queue(
            name=args.queue,
            entity=args.entity,
            namespace=args.namespace,
            gpus=args.gpus,
            cpu=args.cpu,
            memory=args.memory,
            prioritization=not args.no_prioritization,
        )

    elif args.command == "inspect":
        entity, project, run_id = parse_run_url(args.run_url)
        run_path = f"{entity}/{project}/{run_id}"
        art = get_job_artifact(run_path)
        if art:
            print(f"Job artifact: {entity}/{project}/{art.name}")
            inspect_job_artifact(f"{entity}/{project}/{art.name}")
        else:
            print(f"No job artifact found for {run_path}")

    elif args.command == "check":
        check_launch(args.entity, args.project, args.queue_name, args.item_id)

    elif args.command == "relaunch":
        entity, project, run_id = parse_run_url(args.run_url)
        run_path = f"{entity}/{project}/{run_id}"

        relaunch_run(
            run_path=run_path,
            queue_name=args.queue,
            config=args.config,
            wait_for=args.wait_for,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
