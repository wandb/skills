"""Benchmark single-run history analysis patterns on large project.

Test: find a metric value, scan specific keys, beta_scan_history comparison.
Also test with a run that actually HAS step history.
"""
import time
import wandb

api = wandb.Api(timeout=120)
path = "wandb/large_runs_demo"

# Find runs with actual step history
print("=== Finding runs with step history ===")
runs = api.runs(path, order="-created_at", per_page=20)
test_run = None
for run in runs[:20]:
    steps = run.lastHistoryStep
    n_summary = len([k for k in run.summary_metrics.keys() if not k.startswith("_")])
    print(f"  {run.name} ({run.id}): lastHistoryStep={steps}, summary_keys={n_summary}, state={run.state}")
    if steps > 0 and test_run is None:
        test_run = run

if not test_run:
    print("\nNo runs with step history in first 20. Trying with history() directly...")
    for run in runs[:20]:
        df = run.history(samples=5, keys=["acc"])
        if len(df) > 0:
            test_run = run
            print(f"  Found: {run.name} has history rows")
            break

if not test_run:
    print("No runs with history found in first 20. Using first run anyway.")
    test_run = runs[0]

print(f"\nUsing: {test_run.name} ({test_run.id}), lastHistoryStep={test_run.lastHistoryStep}")

# Get a sample of metric keys
summary_keys = [k for k in test_run.summary_metrics.keys() if not k.startswith("_")]
print(f"Summary keys ({len(summary_keys)} total): {summary_keys[:5]}...")

# --- Test history() with specific keys ---
test_keys = summary_keys[:3] if summary_keys else ["acc"]
print(f"\nTest keys: {test_keys}")

print(f"\n--- history(samples=100, keys={test_keys}) ---")
t0 = time.time()
df = test_run.history(samples=100, keys=test_keys)
print(f"  {len(df)} rows x {len(df.columns)} cols in {time.time()-t0:.2f}s")

print(f"\n--- history(samples=500, keys={test_keys[:1]}) ---")
t0 = time.time()
df = test_run.history(samples=500, keys=test_keys[:1])
print(f"  {len(df)} rows in {time.time()-t0:.2f}s")

# --- scan_history with keys ---
print(f"\n--- scan_history(keys={test_keys[:1]}) all rows ---")
t0 = time.time()
rows = list(test_run.scan_history(keys=test_keys[:1]))
print(f"  {len(rows)} rows in {time.time()-t0:.2f}s")

# --- beta_scan_history with keys ---
print(f"\n--- beta_scan_history(keys={test_keys[:1]}) all rows ---")
t0 = time.time()
rows_beta = list(test_run.beta_scan_history(keys=test_keys[:1]))
print(f"  {len(rows_beta)} rows in {time.time()-t0:.2f}s")

# --- The real bottleneck: searching across runs for a metric condition ---
print(f"\n--- Cross-run metric search: find runs where {test_keys[0]} > threshold ---")
# Approach 1: Server-side filter
t0 = time.time()
filtered = api.runs(path, filters={
    f"summary_metrics.{test_keys[0]}": {"$gt": 0}
}, per_page=10)
first_10 = filtered[:10]
vals = [(r.name, r.summary_metrics.get(test_keys[0])) for r in first_10]
print(f"  Server-side filter: {len(vals)} runs in {time.time()-t0:.2f}s")
for name, val in vals[:3]:
    print(f"    {name}: {val}")
