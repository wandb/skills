"""
Helper functions for W&B Launch — queue management, job submission, and run reproduction.

Usage:
    import sys
    sys.path.insert(0, "skills/launch/scripts")
    from launch_helpers import (
        create_queue,
        recreate_queue_with_prioritization,
        submit_code_artifact_job,
        get_job_artifact,
        relaunch_run,
        make_resource_args,
    )
"""

import os
import shutil
import tempfile
import wandb
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.create_job import _create_job


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def make_k8s_queue_config(namespace, gpus=1, cpu=8, memory="80Gi"):
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


def create_queue(name, entity, namespace, gpus=1, cpu=8, memory="80Gi", prioritization=True):
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


def recreate_queue_with_prioritization(name, entity, namespace, **kwargs):
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

    Must be passed explicitly — queue defaults get double-nested by the server.

    Args:
        k8s_secrets: Dict mapping env var name to (secret_name, secret_key) tuples.
                     These K8s secrets will be injected as env vars into the container.
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
    namespace,
    job_name,
    base_image=None,
    requirements=None,
    config=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
    wait=True,
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
    resource_args = make_resource_args(namespace, gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets)
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
        launch_kwargs["config"] = config

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued! Run ID will be assigned when started.")

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


def launch_job_artifact(
    artifact_path,
    queue_name,
    namespace,
    entry_point=None,
    config=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
    wait=True,
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
        namespace, gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets,
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
        launch_kwargs["config"] = config

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued job from artifact: {artifact_path}")

    if not wait:
        return queued_run

    run = queued_run.wait_until_running()
    print(f"Run started: {run.id}")
    print(f"View at: https://wandb.ai/{entity}/{project}/runs/{run.id}")
    run = queued_run.wait_until_finished()
    print(f"Run finished. Summary: {dict(run.summary)}")
    return run


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
    entry_point=None,
    gpus=1,
    cpu=8,
    memory="80Gi",
    k8s_secrets=None,
    wait=True,
):
    """
    Re-launch an existing W&B run using its job artifact.

    Args:
        run_path: "<entity>/<project>/<run_id>"
        queue_name: Launch queue to submit to.
        namespace: K8s namespace.
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

    resource_args = make_resource_args(namespace, gpus=gpus, cpu=cpu, memory=memory, k8s_secrets=k8s_secrets)

    launch_kwargs = dict(
        job=f"{entity}/{project}/{job_artifact.name}",
        entity=entity,
        project=project,
        queue_name=queue_name,
        resource_args=resource_args,
        config=dict(run.config),
    )
    if entry_point:
        launch_kwargs["entry_point"] = entry_point

    queued_run = launch_add(**launch_kwargs)
    print(f"Queued reproduction of run {run_path}")

    if not wait:
        return queued_run

    run = queued_run.wait_until_running()
    print(f"Run started: {run.id}")
    run = queued_run.wait_until_finished()
    print(f"Run finished. Summary: {dict(run.summary)}")
    return run
