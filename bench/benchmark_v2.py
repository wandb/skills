"""Benchmark: original vs optimized wandb_helpers on large project.

Compares the key operations side-by-side.
"""
import json
import sys
import time
from pathlib import Path

# --- Original helpers ---
sys.path.insert(0, "skills/wandb-primary/scripts")
import wandb_helpers as orig
sys.path.pop(0)

# --- Optimized helpers ---
sys.path.insert(0, "skills/wandb-primary-large/scripts")
import importlib
del sys.modules["wandb_helpers"]
import wandb_helpers as optim
sys.path.pop(0)

import wandb
import pandas as pd

path = "wandb/large_runs_demo"
results = {}


def bench(name, fn):
    print(f"  [{name}] ...", end="", flush=True)
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f" {elapsed:.2f}s")
        return {"time_s": round(elapsed, 3), "status": "ok"}
    except Exception as e:
        elapsed = time.time() - t0
        err = f"{type(e).__name__}: {str(e)[:100]}"
        print(f" {elapsed:.2f}s FAIL: {err}")
        return {"time_s": round(elapsed, 3), "status": "error", "error": err}


# ============================================================
# Test 1: runs_to_dataframe — original vs optimized
# ============================================================
print("\n=== runs_to_dataframe comparison ===")

# Original: uses wandb.Api() with default 19s timeout
# (will likely timeout on pagination)
api_orig = wandb.Api(timeout=120)  # give it a fair chance with increased timeout
runs_orig = api_orig.runs(path, filters={"state": "finished"}, order="-created_at")

for limit in [10, 50]:
    results[f"orig_rtd_{limit}"] = bench(
        f"ORIGINAL runs_to_dataframe(limit={limit})",
        lambda l=limit: orig.runs_to_dataframe(runs_orig, limit=l, metric_keys=["acc"])
    )

# Optimized: uses per_page tuning + selective keys
api_opt = optim.get_api()
runs_opt = api_opt.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)

for limit in [10, 50]:
    results[f"optim_rtd_{limit}"] = bench(
        f"OPTIMIZED runs_to_dataframe(limit={limit})",
        lambda l=limit: optim.runs_to_dataframe(runs_opt, limit=l, metric_keys=["acc"])
    )

# ============================================================
# Test 2: diagnose_run — original vs optimized
# ============================================================
print("\n=== diagnose_run comparison ===")

run = api_opt.run(f"{path}/eval-2025-02-13_09-38-37-432411-35")

results["orig_diagnose"] = bench(
    "ORIGINAL diagnose_run() [hardcoded loss]",
    lambda: orig.diagnose_run(run)
)

results["optim_diagnose"] = bench(
    "OPTIMIZED diagnose_run(train_key='acc')",
    lambda: optim.diagnose_run(run, train_key="acc")
)

# ============================================================
# Test 3: scan_history — original pattern vs optimized helper
# ============================================================
print("\n=== scan_history comparison ===")

results["orig_scan"] = bench(
    "ORIGINAL: list(run.scan_history(keys=['acc']))",
    lambda: list(run.scan_history(keys=["acc"]))
)

results["optim_scan"] = bench(
    "OPTIMIZED: scan_history(run, keys=['acc'])",
    lambda: optim.scan_history(run, keys=["acc"])
)

results["optim_scan_max"] = bench(
    "OPTIMIZED: scan_history(run, keys=['acc'], max_rows=1000)",
    lambda: optim.scan_history(run, keys=["acc"], max_rows=1000)
)

# ============================================================
# Test 4: Server-side filter (new pattern in optimized skill)
# ============================================================
print("\n=== Server-side filter (optimized pattern) ===")

results["server_filter"] = bench(
    "Server-side: acc > 5, per_page=10, first 10 runs",
    lambda: [(r.name, r.summary_metrics.get("acc"))
             for r in api_opt.runs(path, filters={"summary_metrics.acc": {"$gt": 5}}, per_page=10)[:10]]
)

# ============================================================
# Test 5: Best run by metric (server-side sort)
# ============================================================
print("\n=== Best run by metric ===")

results["best_run_server"] = bench(
    "Server-side sort: best by acc, per_page=1, first 1",
    lambda: api_opt.runs(path, filters={"state": "finished"}, order="-summary_metrics.acc", per_page=1)[:1][0].summary_metrics.get("acc")
)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
for name, data in results.items():
    status = "OK" if data["status"] == "ok" else f"FAIL: {data.get('error', '')}"
    print(f"  {name:<50} {data['time_s']:>8.2f}s  {status}")

# Write results
out = Path("bench/results_v2.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nResults written to {out}")
