"""Final benchmark: original SDK vs fetch_runs (GraphQL field selection)."""
import sys
import time
import json

sys.path.insert(0, "skills/wandb-primary-large/scripts")
from wandb_helpers import get_api, fetch_runs, runs_to_dataframe

import wandb

path = "wandb/large_runs_demo"

def bench(name, fn):
    print(f"  [{name}] ...", end="", flush=True)
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f" {elapsed:.2f}s")
        return elapsed, result
    except Exception as e:
        elapsed = time.time() - t0
        print(f" {elapsed:.2f}s FAIL: {type(e).__name__}: {str(e)[:100]}")
        return elapsed, None

print("=" * 60)
print("ORIGINAL SDK (api.runs + iteration)")
print("=" * 60)

for limit in [20, 50, 100]:
    api = wandb.Api(timeout=120)
    runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=50)

    def _orig(lim=limit):
        rows = []
        for run in runs[:lim]:
            rows.append({
                "id": run.id, "name": run.name, "state": run.state,
                "acc": run.summary_metrics.get("acc"),
            })
        return rows

    t, result = bench(f"SDK iteration (limit={limit})", _orig)
    if result:
        print(f"    → {len(result)} rows, {t/len(result)*1000:.0f}ms/run")

print()
print("=" * 60)
print("OPTIMIZED: fetch_runs (GraphQL field selection)")
print("=" * 60)

for limit in [20, 50, 100, 200, 500]:
    api = get_api()

    def _fast(lim=limit):
        return fetch_runs(
            api, path,
            metric_keys=["acc"],
            filters={"state": "finished"},
            order="-created_at",
            limit=lim,
            per_page=50,
        )

    t, result = bench(f"fetch_runs (limit={limit})", _fast)
    if result:
        print(f"    → {len(result)} rows, {t/len(result)*1000:.0f}ms/run")

print()
print("=" * 60)
print("OPTIMIZED: fetch_runs with config_keys")
print("=" * 60)

api = get_api()
t, result = bench(
    "fetch_runs (100 runs, metric+config keys)",
    lambda: fetch_runs(
        api, path,
        metric_keys=["acc"],
        config_keys=["learning_rate", "model"],
        filters={"state": "finished"},
        limit=100,
    ),
)
if result:
    print(f"    → {len(result)} rows, sample: {result[0]}")

print()
print("=" * 60)
print("SPEEDUP SUMMARY")
print("=" * 60)
