"""Clean pagination benchmark — fresh API instances to avoid caching."""
import time
import wandb

path = "wandb/large_runs_demo"

# --- 1. per_page impact (fresh API each time) ---
print("=== per_page impact on fetching 100 runs ===")
for pp in [50, 100, 200, 500, 1000]:
    api = wandb.Api(timeout=120)
    t0 = time.time()
    runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=pp)
    rows = []
    for run in runs[:100]:
        rows.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
    t1 = time.time()
    print(f"  per_page={pp:>4}: {t1-t0:.2f}s for {len(rows)} runs")

# --- 2. What if we skip summary_metrics entirely? ---
print("\n=== Summary metrics access cost ===")
api = wandb.Api(timeout=120)
t0 = time.time()
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)
rows = []
for run in runs[:100]:
    rows.append({"id": run.id, "name": run.name, "state": run.state})
t1 = time.time()
print(f"  100 runs, NO summary access: {t1-t0:.2f}s")

api2 = wandb.Api(timeout=120)
t0 = time.time()
runs2 = api2.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)
rows2 = []
for run in runs2[:100]:
    rows2.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
t1 = time.time()
print(f"  100 runs, WITH summary.get('acc'): {t1-t0:.2f}s")

api3 = wandb.Api(timeout=120)
t0 = time.time()
runs3 = api3.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)
rows3 = []
for run in runs3[:100]:
    rows3.append({"id": run.id, "name": run.name, "n_keys": len(run.summary_metrics)})
t1 = time.time()
print(f"  100 runs, WITH len(summary_metrics): {t1-t0:.2f}s")

# --- 3. Server-side metric filter (pre-filter by metric existence) ---
print("\n=== Server-side filter by metric ===")
api4 = wandb.Api(timeout=120)
t0 = time.time()
runs4 = api4.runs(path, filters={
    "$and": [
        {"state": "finished"},
        {"summary_metrics.acc": {"$exists": True}},
    ]
}, order="-created_at", per_page=100)
rows4 = []
for run in runs4[:100]:
    rows4.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
t1 = time.time()
print(f"  100 runs, filtered by acc exists: {t1-t0:.2f}s ({len(rows4)} rows)")

# --- 4. Order by metric (server-side sort) ---
print("\n=== Server-side sort by metric ===")
api5 = wandb.Api(timeout=120)
t0 = time.time()
runs5 = api5.runs(path, filters={"state": "finished"}, order="+summary_metrics.acc", per_page=10)
rows5 = []
for run in runs5[:10]:
    rows5.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
t1 = time.time()
print(f"  10 runs, sorted by acc asc: {t1-t0:.2f}s")
