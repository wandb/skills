"""Benchmark different approaches to fetching runs metadata at scale.

The current skill iterates run objects one by one, accessing .config and .summary_metrics.
This tests if per_page tuning, GraphQL field selection, or the runs export API helps.
"""
import time
import wandb

api = wandb.Api(timeout=120)
path = "wandb/large_runs_demo"

# --- 1. Default pagination (per_page=50 default) ---
print("=== Default pagination ===")
t0 = time.time()
runs = api.runs(path, filters={"state": "finished"}, order="-created_at")
rows = []
for run in runs[:100]:
    rows.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
t1 = time.time()
print(f"  100 runs with default per_page: {t1-t0:.2f}s")

# --- 2. Larger per_page ---
for pp in [100, 200, 500, 1000]:
    t0 = time.time()
    runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=pp)
    rows = []
    for run in runs[:100]:
        rows.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
    t1 = time.time()
    print(f"  100 runs with per_page={pp}: {t1-t0:.2f}s")

# --- 3. Only accessing .id and .name (no summary_metrics) ---
print("\n=== No summary_metrics access ===")
t0 = time.time()
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)
rows = []
for run in runs[:100]:
    rows.append({"id": run.id, "name": run.name, "state": run.state})
t1 = time.time()
print(f"  100 runs, id/name/state only: {t1-t0:.2f}s")

# --- 4. Using runs API with to_dataframe if available ---
print("\n=== Checking for bulk/export methods ===")
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=100)
methods = [m for m in dir(runs) if not m.startswith("_")]
print(f"  Runs object methods: {methods}")

# --- 5. Avoid len() — check if we can get count differently ---
print("\n=== Avoid len() ===")
t0 = time.time()
runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=1)
first = runs[:1]
t1 = time.time()
print(f"  First run only (per_page=1): {t1-t0:.2f}s")

t0 = time.time()
total = len(runs)
t1 = time.time()
print(f"  len() after: {total} in {t1-t0:.2f}s")
