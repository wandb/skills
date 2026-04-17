"""Probe both projects to understand their scale before benchmarking."""
import time
import wandb
import os

api = wandb.Api()

projects = [
    ("wandb", "large_runs_demo"),
    ("wandb", "demo-project-qwen-email-agent-with-art-weave-models"),
]

for entity, project in projects:
    path = f"{entity}/{project}"
    print(f"\n{'='*60}")
    print(f"Project: {path}")
    print(f"{'='*60}")

    t0 = time.time()
    runs = api.runs(path)
    t_list = time.time() - t0

    t0 = time.time()
    total = len(runs)
    t_count = time.time() - t0
    print(f"  Total runs: {total} (len() took {t_count:.2f}s, api.runs() took {t_list:.2f}s)")

    # Sample first 3 runs to understand data shape
    t0 = time.time()
    sample = runs[:3]
    t_slice = time.time() - t0
    print(f"  Slicing [:3] took {t_slice:.2f}s")

    for run in sample:
        print(f"\n  Run: {run.name} ({run.id}) state={run.state}")
        print(f"    Config keys: {list(run.config.keys())[:10]}")
        print(f"    Summary keys: {list(run.summary_metrics.keys())[:10]}")

        # Check history size
        t0 = time.time()
        df = run.history(samples=10, keys=["loss"])
        t_hist = time.time() - t0
        print(f"    history(samples=10): {len(df)} rows, {t_hist:.2f}s")

        # Check scan_history for first 100
        t0 = time.time()
        rows = []
        for i, row in enumerate(run.scan_history(keys=["loss"])):
            rows.append(row)
            if i >= 99:
                break
        t_scan = time.time() - t0
        print(f"    scan_history first 100: {len(rows)} rows, {t_scan:.2f}s")
