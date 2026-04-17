"""Final validation: test all optimized helpers against both projects."""
import sys
import time

sys.path.insert(0, "skills/wandb-primary-large/scripts")
from wandb_helpers import (
    get_api, probe_project, runs_to_dataframe,
    diagnose_run, compare_configs, scan_history,
)

api = get_api()

for path in ["wandb/large_runs_demo", "wandb/demo-project-qwen-email-agent-with-art-weave-models"]:
    print(f"\n{'='*60}")
    print(f"Project: {path}")
    print(f"{'='*60}")

    # 1. Probe
    t0 = time.time()
    info = probe_project(api, path)
    print(f"\n[probe_project] {time.time()-t0:.2f}s")
    print(f"  Metrics: {info['sample_metric_count']}")
    print(f"  Has history: {info['has_step_history']}")
    print(f"  Recommended per_page: {info.get('recommended_per_page')}")
    print(f"  Warnings: {info['warnings']}")
    print(f"  First 10 metrics: {info['sample_metric_keys'][:10]}")

    # 2. runs_to_dataframe
    pp = info.get("recommended_per_page", 50)
    runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=pp)
    t0 = time.time()
    rows = runs_to_dataframe(runs, limit=10, metric_keys=info["sample_metric_keys"][:3])
    print(f"\n[runs_to_dataframe(10)] {time.time()-t0:.2f}s, {len(rows)} rows")
    if rows:
        print(f"  Sample: {rows[0]}")

    # 3. diagnose_run
    sample_runs = runs[:2]
    if sample_runs:
        run = sample_runs[0]
        key = info["sample_metric_keys"][0] if info["sample_metric_keys"] else "loss"
        t0 = time.time()
        diag = diagnose_run(run, train_key=key)
        print(f"\n[diagnose_run(train_key='{key}')] {time.time()-t0:.2f}s")
        print(f"  Result: {diag}")

    # 4. compare_configs
    if len(sample_runs) >= 2:
        t0 = time.time()
        diffs = compare_configs(sample_runs[0], sample_runs[1])
        print(f"\n[compare_configs] {time.time()-t0:.2f}s, {len(diffs)} diffs")

    # 5. scan_history
    if sample_runs:
        run = sample_runs[0]
        key = info["sample_metric_keys"][0] if info["sample_metric_keys"] else "loss"
        t0 = time.time()
        rows = scan_history(run, keys=[key], max_rows=100)
        print(f"\n[scan_history(keys=['{key}'], max_rows=100)] {time.time()-t0:.2f}s, {len(rows)} rows")

    # 6. Server-side filter
    if info["sample_metric_keys"]:
        key = info["sample_metric_keys"][0]
        t0 = time.time()
        filtered = api.runs(path, filters={
            f"summary_metrics.{key}": {"$exists": True}
        }, per_page=5)
        count = len(filtered[:5])
        print(f"\n[server-side filter on '{key}'] {time.time()-t0:.2f}s, {count} runs")

print("\n\nAll validations passed!")
