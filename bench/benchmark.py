"""Comprehensive benchmark of all runs-related code patterns from wandb-primary skill.

Tests against both large (wandb/large_runs_demo, ~72K runs, 20K metrics/run)
and small (wandb/demo-project-qwen-email-agent-with-art-weave-models, 29 runs) projects.

Outputs: bench/results.json with timing data for each pattern.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import wandb
import pandas as pd
import numpy as np

sys.path.insert(0, "skills/wandb-primary/scripts")
from wandb_helpers import runs_to_dataframe, diagnose_run, compare_configs

# Use a generous timeout for the large project
api = wandb.Api(timeout=120)

PROJECTS = {
    "large": ("wandb", "large_runs_demo"),
    "small": ("wandb", "demo-project-qwen-email-agent-with-art-weave-models"),
}

results = {}


def bench(name, fn, **kwargs):
    """Time a function, catch errors, record results."""
    print(f"  [{name}] ...", end="", flush=True)
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f" {elapsed:.2f}s ✓")
        entry = {"time_s": round(elapsed, 3), "status": "ok", **kwargs}
        if isinstance(result, (int, float, str)):
            entry["value"] = result
        return entry, result
    except Exception as e:
        elapsed = time.time() - t0
        err = f"{type(e).__name__}: {str(e)[:200]}"
        print(f" {elapsed:.2f}s ✗ ({err})")
        return {"time_s": round(elapsed, 3), "status": "error", "error": err, **kwargs}, None


for proj_label, (entity, project) in PROJECTS.items():
    path = f"{entity}/{project}"
    print(f"\n{'='*70}")
    print(f"Project: {path} ({proj_label})")
    print(f"{'='*70}")
    proj_results = {}

    # --- 1. api.runs() creation (lazy) ---
    entry, runs_obj = bench("api.runs()", lambda: api.runs(path, filters={"state": "finished"}, order="-created_at"))
    proj_results["api_runs_create"] = entry

    if runs_obj is None:
        results[proj_label] = proj_results
        continue

    # --- 2. len(runs) — triggers count query ---
    entry, total = bench("len(runs)", lambda: len(runs_obj))
    proj_results["len_runs"] = entry

    # --- 3. Slicing at various limits ---
    for limit in [10, 50, 100, 200]:
        def _slice(lim=limit):
            sliced = runs_obj[:lim]
            # Force materialization by accessing first run
            if sliced:
                _ = sliced[0].id
            return len(sliced)
        entry, _ = bench(f"runs[:{limit}]", _slice, limit=limit)
        proj_results[f"slice_{limit}"] = entry

    # --- 4. runs_to_dataframe (current skill helper) ---
    for limit in [10, 50, 100]:
        def _rtd(lim=limit):
            rows = runs_to_dataframe(runs_obj, limit=lim, metric_keys=["acc"])
            return len(rows)
        entry, _ = bench(f"runs_to_dataframe(limit={limit})", _rtd, limit=limit)
        proj_results[f"runs_to_dataframe_{limit}"] = entry

    # --- 5. Per-run config iteration (bottleneck?) ---
    sample_runs = runs_obj[:5]
    def _config_iter():
        total_keys = 0
        for run in sample_runs:
            for k, v in run.config.items():
                total_keys += 1
        return total_keys
    entry, _ = bench("config iteration (5 runs)", _config_iter)
    proj_results["config_iteration_5"] = entry

    # --- 6. Per-run summary_metrics access ---
    def _summary_access():
        total_keys = 0
        for run in sample_runs:
            total_keys += len(run.summary_metrics)
        return total_keys
    entry, nkeys = bench("summary_metrics access (5 runs)", _summary_access)
    proj_results["summary_metrics_5"] = entry

    # --- 7. Per-run summary_metrics.get() for specific key ---
    def _summary_get():
        vals = []
        for run in sample_runs:
            vals.append(run.summary_metrics.get("acc"))
        return len(vals)
    entry, _ = bench("summary_metrics.get('acc') (5 runs)", _summary_get)
    proj_results["summary_get_5"] = entry

    # --- 8. run.history() with various samples ---
    first_run = sample_runs[0] if sample_runs else None
    if first_run:
        for samples in [10, 100, 500]:
            def _hist(s=samples):
                df = first_run.history(samples=s, keys=["acc"])
                return len(df)
            entry, _ = bench(f"run.history(samples={samples}, keys=['acc'])", _hist, samples=samples)
            proj_results[f"history_samples_{samples}"] = entry

        # --- 9. run.history() WITHOUT keys (fetches all 20K columns) ---
        def _hist_nokeys():
            df = first_run.history(samples=10)
            return f"{len(df)} rows x {len(df.columns)} cols"
        entry, _ = bench("run.history(samples=10) NO KEYS", _hist_nokeys)
        proj_results["history_no_keys"] = entry

        # --- 10. run.scan_history() ---
        def _scan():
            rows = []
            for row in first_run.scan_history(keys=["acc"]):
                rows.append(row)
                if len(rows) >= 100:
                    break
            return len(rows)
        entry, _ = bench("scan_history(keys=['acc']) first 100", _scan)
        proj_results["scan_history_100"] = entry

        # --- 11. diagnose_run (current skill helper) ---
        # This calls scan_history(keys=["loss", "val_loss"]) and loads ALL into pandas
        def _diagnose():
            return diagnose_run(first_run)
        entry, _ = bench("diagnose_run()", _diagnose)
        proj_results["diagnose_run"] = entry

    # --- 12. compare_configs ---
    if len(sample_runs) >= 2:
        def _compare():
            return compare_configs(sample_runs[0], sample_runs[1])
        entry, _ = bench("compare_configs()", _compare)
        proj_results["compare_configs"] = entry

    # --- 13. runs.histories() bulk method ---
    def _bulk_hist():
        df = runs_obj.histories(samples=100, keys=["acc"], format="pandas")
        return f"{len(df)} rows"
    entry, _ = bench("runs.histories(samples=100, keys=['acc'])", _bulk_hist)
    proj_results["bulk_histories"] = entry

    # --- 14. Filtered queries ---
    def _filtered():
        r = api.runs(path, filters={
            "$and": [
                {"state": "finished"},
                {"summary_metrics.acc": {"$gt": 0}},
            ]
        }, order="-created_at", per_page=10)
        _ = r[:10]
        return len(r[:10])
    entry, _ = bench("filtered query (state+metric)", _filtered)
    proj_results["filtered_query"] = entry

    results[proj_label] = proj_results

# --- Write results ---
out = Path("bench/results.json")
out.write_text(json.dumps(results, indent=2))
print(f"\n\nResults written to {out}")

# --- Print summary table ---
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Pattern':<45} {'Large':>10} {'Small':>10}")
print(f"{'-'*45} {'-'*10} {'-'*10}")
all_keys = set()
for v in results.values():
    all_keys |= set(v.keys())
for key in sorted(all_keys):
    large_t = results.get("large", {}).get(key, {}).get("time_s", "—")
    small_t = results.get("small", {}).get(key, {}).get("time_s", "—")
    large_s = results.get("large", {}).get(key, {}).get("status", "—")
    small_s = results.get("small", {}).get(key, {}).get("status", "—")
    l_str = f"{large_t}s" if isinstance(large_t, (int, float)) else large_t
    s_str = f"{small_t}s" if isinstance(small_t, (int, float)) else small_t
    if large_s == "error":
        l_str += " ✗"
    if small_s == "error":
        s_str += " ✗"
    print(f"  {key:<43} {l_str:>10} {s_str:>10}")
