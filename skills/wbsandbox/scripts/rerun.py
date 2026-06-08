"""Re-run a W&B training job in a CoreWeave sandbox.

Supports uploading modified source files to iterate on code directly.
The sandbox captures each variant as a new job version in W&B.

Usage:
    uv run python rerun.py <run_path> [--override key=value ...] [--file-override path=local_path ...]

Examples:
    uv run python rerun.py entity/project/run_id
    uv run python rerun.py entity/project/run_id --override epochs=20
    uv run python rerun.py entity/project/run_id --file-override model.py=/tmp/model_v2.py
"""

from __future__ import annotations

import argparse
import json
import netrc
import os
import time

import wandb
from cwsandbox import NetworkOptions, Sandbox, SandboxDefaults


def get_wandb_api_key() -> str:
    """Get W&B API key from env var or netrc."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth:
            return auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    raise RuntimeError("WANDB_API_KEY not found in env or ~/.netrc")


def parse_override(s: str) -> tuple[str, int | float | str]:
    """Parse 'key=value' string, auto-casting numeric values."""
    key, _, val = s.partition("=")
    try:
        return key, int(val)
    except ValueError:
        pass
    try:
        return key, float(val)
    except ValueError:
        pass
    return key, val


def rerun_in_sandbox(
    run_path: str,
    overrides: dict | None = None,
    file_overrides: dict[str, str] | None = None,
) -> None:
    """Re-run a W&B job in a CWSandbox.

    Args:
        run_path: W&B run path (entity/project/run_id).
        overrides: Config key=value overrides passed via WANDB_CONFIG.
        file_overrides: Map of {sandbox_relative_path: local_file_path} to replace
            source files before running. Use this to iterate on code directly.
    """
    t0 = time.time()

    def elapsed() -> str:
        return f"[{time.time() - t0:.1f}s]"

    api = wandb.Api()
    run = api.run(run_path)
    print(f"{elapsed()} Run: {run.name} ({run_path})")

    # Step 1: Find job artifact (input artifact to the run)
    print(f"{elapsed()} Finding job artifact...")
    job_art = None
    for art in run.used_artifacts():
        if art.type == "job":
            job_art = art
            break

    if job_art is None:
        raise ValueError(
            f"No job artifact found for run {run_path}. "
            "Was the run created with wandb.Settings(disable_job_creation=False)?"
        )

    job_dir = job_art.download("/tmp/wandb-job")
    with open(os.path.join(job_dir, "wandb-job.json")) as f:
        job_spec = json.load(f)

    entrypoint_script = job_spec["source"]["entrypoint"][1]
    runtime = job_spec.get("runtime", "")
    source_type = job_spec["source_type"]
    print(f"  Source type: {source_type}")
    print(f"  Entrypoint: {entrypoint_script}")
    print(f"  Runtime: {runtime}")

    # Step 2: Build config with overrides
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    if overrides:
        config.update(overrides)
    print(f"  Config: {json.dumps(config)}")

    # Step 3: Download source code
    if source_type == "artifact":
        print(f"{elapsed()} Downloading code artifact...")
        code_art = None
        for art in run.logged_artifacts():
            if art.type == "code":
                code_art = art
                break
        if code_art is None:
            raise ValueError("No code artifact found")

        code_dir = code_art.download("/tmp/wandb-code")
        source_files = []
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d != ".venv"]
            for fname in files:
                local = os.path.join(root, fname)
                rel = os.path.relpath(local, code_dir)
                source_files.append((local, rel))
        print(f"{elapsed()} {len(source_files)} source files downloaded")
    elif source_type == "repo":
        git_info = job_spec["source"]["git"]
        print(f"  Git: {git_info['remote']} @ {git_info['commit']}")
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    # Step 4: Determine container image from runtime
    if runtime.startswith("CPython "):
        py_ver = runtime.replace("CPython ", "").rsplit(".", 1)[0]
        container_image = f"python:{py_ver}"
    else:
        container_image = "python:3.11"
    print(f"  Container image: {container_image}")

    # Step 5: Set up auth for cwsandbox client
    wandb_api_key = get_wandb_api_key()
    os.environ.setdefault("WANDB_API_KEY", wandb_api_key)
    os.environ.setdefault("WANDB_ENTITY_NAME", run_path.split("/")[0])

    # Step 6: Create sandbox and run
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

    print(f"\n{elapsed()} Creating sandbox...")
    with Sandbox.run(defaults=defaults) as sb:
        print(f"  Sandbox ID: {sb.sandbox_id}")

        # Upload and install requirements
        req_path = os.path.join(job_dir, "requirements.frozen.txt")
        if os.path.exists(req_path):
            with open(req_path) as f:
                lines = f.readlines()
            # Filter out local-only packages that aren't on PyPI
            filtered = [l for l in lines if not l.strip().startswith("cwsandbox")]
            sb.write_file("/app/requirements.txt", "".join(filtered).encode()).result()
            print(f"{elapsed()} Installing dependencies...")
            proc = sb.exec(
                ["pip", "install", "-r", "/app/requirements.txt"],
                cwd="/app",
                timeout_seconds=600,
            )
            for line in proc.stdout:
                if "Successfully installed" in line or "ERROR" in line:
                    print(f"    {line}", end="")
            result = proc.result()
            if result.returncode != 0:
                print(f"  pip install failed:\n{result.stderr}")
                return

        # Upload source code or clone repo
        if source_type == "artifact":
            for local_path, rel_path in source_files:
                with open(local_path, "rb") as f:
                    sb.write_file(f"/app/{rel_path}", f.read()).result()
            print(f"{elapsed()} Uploaded {len(source_files)} source files")
            work_dir = "/app"
        else:
            git_info = job_spec["source"]["git"]
            sb.exec(
                ["git", "clone", git_info["remote"], "/app/repo"],
                timeout_seconds=120,
            ).result()
            sb.exec(
                ["git", "checkout", git_info["commit"]],
                cwd="/app/repo",
            ).result()
            print(f"{elapsed()} Cloned repo")
            work_dir = "/app/repo"

        # Apply file overrides — replace source files with modified versions
        if file_overrides:
            for sandbox_path, local_path in file_overrides.items():
                with open(local_path, "rb") as f:
                    sb.write_file(f"{work_dir}/{sandbox_path}", f.read()).result()
                print(f"{elapsed()} Replaced {sandbox_path}")

        # Run training
        print(f"\n{elapsed()} Running: python {entrypoint_script}")
        print("=" * 60)

        process = sb.exec(
            ["python", entrypoint_script],
            cwd=work_dir,
            timeout_seconds=3600,
        )
        for line in process.stdout:
            print(line, end="")

        result = process.result()
        print("=" * 60)
        print(f"{elapsed()} Exit code: {result.returncode}")
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}")

    print(f"\n{elapsed()} Sandbox stopped. Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run a W&B job in a sandbox")
    parser.add_argument("run_path", help="W&B run path (entity/project/run_id)")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs",
    )
    parser.add_argument(
        "--file-override",
        nargs="*",
        default=[],
        help="File overrides as sandbox_path=local_path pairs",
    )
    args = parser.parse_args()

    overrides = dict(parse_override(o) for o in args.override) if args.override else None
    file_overrides = (
        dict(o.split("=", 1) for o in args.file_override) if args.file_override else None
    )
    rerun_in_sandbox(args.run_path, overrides, file_overrides)


if __name__ == "__main__":
    main()
