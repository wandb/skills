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
        make_resource_args,
    )
"""

import json
import os
import re
import shutil
import tempfile

import wandb
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.create_job import _create_job

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
    (e.g. in a minimal sandbox), fall back to raw HTTP POST to the GraphQL
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
        headers={"Authorization": f"Basic api {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise Exception(str(data["errors"][0]))
    return data.get("data", {})


def _extract_namespace_from_config(default_resource_config):
    """Extract the K8s namespace from a queue's default resource config.

    The namespace can be at different paths depending on how the queue was set up:
      - config.resource_args.kubernetes.kubernetes.namespace  (double-nested)
      - config.resource_args.kubernetes.namespace
    Returns the namespace string or None.
    """
    if not default_resource_config:
        return None
    config = default_resource_config.get("config")
    if not isinstance(config, dict):
        return None
    ra = config.get("resource_args", config)
    k8s = ra.get("kubernetes", {})
    # Check double-nested first (kubernetes.kubernetes.namespace)
    inner = k8s.get("kubernetes", {})
    if isinstance(inner, dict) and inner.get("namespace"):
        return inner["namespace"]
    # Then single level (kubernetes.namespace)
    if k8s.get("namespace"):
        return k8s["namespace"]
    return None


def list_queues(entity):
    """List all launch queues for an entity, ranked by relevance.

    Fetches queues and active agents, then sorts by:
      1. Queues with active polling agents (ready to run jobs)
      2. Queues created by the current user
      3. Everything else

    Prints a summary table and returns the full list of queue dicts.
    Each dict has keys: id, name, prioritizationMode, access, createdBy,
    has_active_agent, is_mine, pending_items.
    """
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
                    runQueueItems(first: 5) {
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

    # Map queue ID -> number of active (POLLING) agents
    active_agents_by_queue = {}
    for a in agents:
        if a.get("agentStatus") == "POLLING":
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
        q["pending_items"] = sum(1 for i in items if i["state"] in ("PENDING", "CLAIMED", "LEASED"))
        q["has_active_agent"] = active_agents_by_queue.get(q["id"], 0) > 0
        q["active_agent_count"] = active_agents_by_queue.get(q["id"], 0)
        q["is_mine"] = (q.get("createdBy") == viewer_id) if viewer_id else False
        # Extract namespace from the queue's default resource config
        q["namespace"] = _extract_namespace_from_config(q.get("defaultResourceConfig"))

    queues.sort(key=lambda q: (q["has_active_agent"], q["is_mine"]), reverse=True)

    # Print
    if not queues:
        print(f"  No queues found for entity '{entity}'")
        return queues

    for q in queues:
        flags = []
        if q["has_active_agent"]:
            flags.append(f"agents={q['active_agent_count']}")
        if q["is_mine"]:
            flags.append("yours")
        if q["pending_items"] > 0:
            flags.append(f"pending={q['pending_items']}")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {q['name']:30s}{flag_str}")

    recommended = [q for q in queues if q["has_active_agent"] and q["is_mine"]]
    if recommended:
        print(f"\n  Recommended: {recommended[0]['name']} (yours, with active agent)")
    elif any(q["has_active_agent"] for q in queues):
        active = [q for q in queues if q["has_active_agent"]][0]
        print(f"\n  Recommended: {active['name']} (has active agent)")

    return queues


def make_k8s_queue_config(namespace="wandb", gpus=1, cpu=8, memory="80Gi"):
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
                            "env": [{
                                "name": "WANDB_API_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "wandb-api-key-wandb-launch",
                                        "key": "password",
                                    }
                                },
                            }],
                        }],
                    }
                }
            },
        }
    }


def create_queue(name, entity, namespace="wandb", gpus=1, cpu=8, memory="80Gi", prioritization=True):
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


def recreate_queue_with_prioritization(name, entity, namespace="wandb", **kwargs):
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
    api = wandb.Api()
    q = api.run_queue(entity=entity, name=name)
    print(f"Name:            {q.name}")
    print(f"Type:            {q.type}")
    print(f"Prioritization:  {q.prioritization_mode}")
    return q


# ---------------------------------------------------------------------------
# Resource args (always pass explicitly to avoid double-nesting)
# ---------------------------------------------------------------------------

def make_resource_args(namespace, gpus=1, cpu=8, memory="80Gi", k8s_secrets=None):
    """Build resource_args for launch_add().

    Must be passed explicitly — queue defaults get double-nested by the server,
    and the agent reads the outer (our) resource_args for namespace. If omitted,
    the agent falls back to "default" which typically lacks permissions.

    Args:
        namespace: K8s namespace. Get this from list_queues() — each queue dict
                   has a 'namespace' key extracted from its default resource config.
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
    namespace="wandb",
    base_image=None,
    requirements=None,
    config=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
    wait=False,
):
    """
    Create a code artifact job and submit it to a launch queue.

    Args:
        code_files: List of file paths to include in the artifact.
        entrypoint: Entrypoint string, e.g. "python train.py".
        entity: W&B entity.
        project: W&B project.
        queue_name: Launch queue name.
        namespace: K8s namespace.
        job_name: Name for the job artifact.
        base_image: Docker base image. If None, agent picks a default.
        requirements: List of pip packages to install in the container.
                      Only include packages NOT already in the base image.
                      _create_job reads requirements.txt directly — it does NOT inspect the venv.
        config: Dict of hyperparameters to pass to the run.
        gpus: Number of GPUs to request.
        cpu: Number of CPUs to request.
        memory: Memory to request (e.g. "80Gi").
        wait: If True, block until the run finishes and return summary.

    Returns:
        queued_run if wait=False, else the finished run object.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy code files
        for f in code_files:
            shutil.copy2(f, temp_dir)

        # Write requirements.txt — _create_job reads this file, not the venv
        if requirements:
            with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                f.write("\n".join(requirements) + "\n")

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
        if base_image:
            kwargs["base_image"] = base_image

        artifact, action, aliases = _create_job(**kwargs)
        print(f"Job artifact: {artifact.name} ({action})")

    # Step 2: Enqueue
    resource_args = make_resource_args(gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets, namespace=namespace)
    entry_point = entrypoint.split()

    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_name}:latest",
        entity=entity,
        project=project,
        queue_name=queue_name,
        entry_point=entry_point,
        resource_args=resource_args,
    )
    if config:
        launch_kwargs["config"] = {"overrides": {"run_config": config}}

    queued_run = launch_add(**launch_kwargs)
    print("Queued! Run ID will be assigned when started.")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status later: check_launched_run('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")

    if not wait:
        return queued_run

    # Step 3: Wait
    run = queued_run.wait_until_running()
    print(f"Run started: {run.id}")
    print(f"View at: https://wandb.ai/{entity}/{project}/runs/{run.id}")
    run = queued_run.wait_until_finished()
    print(f"Run finished. Summary: {dict(run.summary)}")
    return run


# ---------------------------------------------------------------------------
# Run reproduction
# ---------------------------------------------------------------------------

def inspect_job_artifact(artifact_path):
    """
    Download and inspect a job artifact's metadata.

    Args:
        artifact_path: Full artifact path, e.g. "wandb/autoresearch/autoresearch-train:v6"

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
    namespace,
    job_name,
    base_image=None,
    requirements=None,
    config=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
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
        queue_name: Launch queue name.
        namespace: K8s namespace (from list_queues()).
        job_name: Name for the job artifact.
        base_image: Docker base image.
        requirements: List of pip packages (only those NOT in base image).
        config: Dict of run config overrides.
        gpus/cpu/memory: Resource requests.
        k8s_secrets: Dict mapping env var to (secret_name, secret_key).
    """
    # Collect code files
    code_files = []
    for root, _, filenames in os.walk(code_dir):
        for fn in filenames:
            if fn.startswith(".") or fn == "__pycache__":
                continue
            code_files.append(os.path.join(root, fn))

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
    if base_image:
        kwargs["base_image"] = base_image

    artifact, action, aliases = _create_job(**kwargs)
    print(f"Job artifact: {artifact.name} ({action})")

    # Launch
    resource_args = make_resource_args(
        namespace=namespace, gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets,
    )
    entry_point = entrypoint.split()

    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_name}:latest",
        entity=entity,
        project=project,
        queue_name=queue_name,
        entry_point=entry_point,
        resource_args=resource_args,
    )
    if config:
        launch_kwargs["config"] = {"overrides": {"run_config": config}}

    queued_run = launch_add(**launch_kwargs)
    print("Queued modified job!")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status: check_launched_run('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")
    return queued_run


def launch_job_artifact(
    artifact_path,
    queue_name,
    entry_point=None,
    config=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
    wait=False,
):
    """
    Launch a job directly from an artifact path.

    Args:
        artifact_path: Full artifact path, e.g. "wandb/autoresearch/autoresearch-train:v6"
        queue_name: Launch queue to submit to.
        namespace: K8s namespace.
        entry_point: Override entrypoint (list). If None, uses the artifact's entrypoint.
        config: Dict of hyperparameters to pass to the run.
        gpus/cpu/memory: Resource requests.
        k8s_secrets: Dict mapping env var name to (secret_name, secret_key) tuples.
        wait: If True, block until finished.

    Returns:
        queued_run if wait=False, else the finished run object.
    """
    # Parse entity/project from artifact path (e.g. "wandb/autoresearch/job-name:v6")
    parts = artifact_path.split("/")
    entity = parts[0]
    project = parts[1]

    # Resolve entrypoint from artifact metadata if not provided
    if entry_point is None:
        info = inspect_job_artifact(artifact_path)
        entry_point = info["entrypoint"]
        print(f"Using entrypoint from artifact: {entry_point}")

    resource_args = make_resource_args(
        gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets, namespace=namespace,
    )

    launch_kwargs = dict(
        job=artifact_path,
        entity=entity,
        project=project,
        queue_name=queue_name,
        entry_point=entry_point,
        resource_args=resource_args,
    )
    if config:
        launch_kwargs["config"] = {"overrides": {"run_config": config}}

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued job from artifact: {artifact_path}")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status later: check_launched_run('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")

    if not wait:
        return queued_run

    run = queued_run.wait_until_running()
    print(f"Run started: {run.id}")
    print(f"View at: https://wandb.ai/{entity}/{project}/runs/{run.id}")
    run = queued_run.wait_until_finished()
    print(f"Run finished. Summary: {dict(run.summary)}")
    return run


def check_launched_run(entity, project, queue_name, run_queue_item_id):
    """Check the status of a launched run and return its info.

    After relaunch_run() or launch_job_artifact() prints a run_queue_item_id,
    use this to find the actual W&B run that was created.

    Returns:
        Dict with keys: state, run_id, run_url, run (the Run object if started).
        Returns None if the queue item can't be found.
    """
    api = wandb.Api()

    # Use the QueuedRun to check state and find the associated run
    from wandb.apis.public import QueuedRun
    qr = QueuedRun(api.client, entity, project, queue_name, run_queue_item_id)

    try:
        state = qr.state
    except Exception as e:
        print(f"Could not find queue item: {e}")
        return None

    result = {"state": state, "run_id": None, "run_url": None, "run": None}
    print(f"Queue item state: {state}")

    if state in ("finished", "claimed", "leased"):
        # Try to get the associated run
        try:
            item = qr._get_item()
            run_id = item.get("associatedRunId") if item else None
            if run_id:
                result["run_id"] = run_id
                result["run_url"] = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                print(f"Run ID: {run_id}")
                print(f"Run URL: {result['run_url']}")
                try:
                    run = api.run(f"{entity}/{project}/{run_id}")
                    result["run"] = run
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
    namespace,
    config=None,
    entry_point=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
    wait=False,
):
    """
    Re-launch an existing W&B run using its job artifact.

    Args:
        run_path: "<entity>/<project>/<run_id>"
        queue_name: Launch queue to submit to.
        namespace: K8s namespace (from list_queues() queue dict's 'namespace' key).
        config: Dict of config overrides. Merged on top of the original run's config.
                e.g. {"epochs": 100, "lr": 0.001}
        entry_point: Override entrypoint (list). Defaults to job artifact's entrypoint.
        gpus/cpu/memory: Resource requests.
        wait: If True, block until finished.
    """
    api = wandb.Api()
    run = api.run(run_path)
    entity, project, _ = run_path.split("/")

    job_artifact = get_job_artifact(run_path)
    if job_artifact is None:
        raise ValueError(f"No job artifact found for run {run_path}. Create one first.")

    print(f"Found job artifact: {job_artifact.name}")

    resource_args = make_resource_args(namespace=namespace, gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets)

    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_artifact.name}",
        entity=entity,
        project=project,
        queue_name=queue_name,
        resource_args=resource_args,
        config={"overrides": {"run_config": {**dict(run.config), **(config or {})}}},
    )
    if entry_point:
        launch_kwargs["entry_point"] = entry_point

    run_config = launch_kwargs["config"]["overrides"]["run_config"]
    if config:
        print(f"Config overrides: {json.dumps(config)}")
    print(f"Final run_config: {json.dumps(run_config)}")

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued reproduction of run {run_path}")
    print(f"Queue item ID: {queued_run.id}")
    print(f"To check status later: check_launched_run('{entity}', '{project}', '{queue_name}', '{queued_run.id}')")

    if not wait:
        return queued_run

    run = queued_run.wait_until_running()
    print(f"Run started: {run.id}")
    run = queued_run.wait_until_finished()
    print(f"Run finished. Summary: {dict(run.summary)}")
    return run


# ---------------------------------------------------------------------------
# CLI entry point — run directly instead of writing wrapper scripts
# ---------------------------------------------------------------------------

def _cli():
    """CLI for common launch operations.

    Usage:
        python launch_helpers.py relaunch <run_url> [--config '{"epochs": 100}']
        python launch_helpers.py list-queues <entity>
        python launch_helpers.py inspect <run_url>
    """
    import argparse

    parser = argparse.ArgumentParser(description="W&B Launch helpers CLI")
    sub = parser.add_subparsers(dest="command")

    p_re = sub.add_parser("relaunch", help="Relaunch a run with optional config overrides")
    p_re.add_argument("run_url", help="W&B run URL or entity/project/run_id")
    p_re.add_argument("--config", type=json.loads, default=None,
                       help='JSON config overrides, e.g. \'{"epochs": 100}\'')
    p_re.add_argument("--queue", default=None, help="Queue name (auto-selects if omitted)")
    p_re.add_argument("--wait", action="store_true", help="Wait for run to finish")

    p_lq = sub.add_parser("list-queues", help="List launch queues for an entity")
    p_lq.add_argument("entity", help="W&B entity name")

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
        check_launched_run(args.entity, args.project, args.queue_name, args.item_id)

    elif args.command == "relaunch":
        entity, project, run_id = parse_run_url(args.run_url)
        run_path = f"{entity}/{project}/{run_id}"

        queues = list_queues(entity)
        if args.queue:
            queue = next((q for q in queues if q["name"] == args.queue), None)
            if not queue:
                print(f"Error: Queue '{args.queue}' not found")
                return
        else:
            recommended = [q for q in queues if q["has_active_agent"] and q["is_mine"]]
            if not recommended:
                recommended = [q for q in queues if q["has_active_agent"]]
            if not recommended:
                print("Error: No queues with active agents found")
                return
            queue = recommended[0]
            print(f"\nAuto-selected queue: {queue['name']}")

        namespace = queue.get("namespace")
        if not namespace:
            print(f"Warning: No namespace in queue config for '{queue['name']}', using 'default'")
            namespace = "default"

        relaunch_run(
            run_path=run_path,
            queue_name=queue["name"],
            namespace=namespace,
            config=args.config,
            wait=args.wait,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
