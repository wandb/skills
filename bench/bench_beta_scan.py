"""Benchmark beta_scan_history (parquet) vs scan_history on large project."""
import time
import wandb

api = wandb.Api(timeout=120)

# Use a run that has history data with "acc"
path = "wandb/large_runs_demo"
runs = api.runs(path, filters={"state": "finished"}, order="-created_at")

# Find a run with acc data
for run in runs[:10]:
    count = 0
    for row in run.scan_history(keys=["acc"]):
        count += 1
        if count > 0:
            break
    if count > 0:
        print(f"Using run: {run.name} ({run.id})")
        test_run = run
        break
else:
    # Just use first run regardless
    test_run = runs[0]
    print(f"No run with acc history found, using: {test_run.name} ({test_run.id})")

# Get total steps
print(f"  lastHistoryStep: {test_run.lastHistoryStep}")
summary_keys = [k for k in test_run.summary_metrics.keys() if not k.startswith("_")]
print(f"  Summary metric keys: {len(summary_keys)}")

# Pick a metric that exists
test_key = "acc" if "acc" in test_run.summary_metrics else summary_keys[0] if summary_keys else None
if not test_key:
    print("No metrics found!")
    exit(1)
print(f"  Using key: {test_key}")

# --- scan_history ---
print(f"\n--- scan_history(keys=['{test_key}']) ---")
t0 = time.time()
count = 0
for row in test_run.scan_history(keys=[test_key]):
    count += 1
t1 = time.time()
print(f"  {count} rows in {t1-t0:.2f}s")

# --- beta_scan_history ---
print(f"\n--- beta_scan_history(keys=['{test_key}']) ---")
t0 = time.time()
count = 0
for row in test_run.beta_scan_history(keys=[test_key]):
    count += 1
t1 = time.time()
print(f"  {count} rows in {t1-t0:.2f}s")

# --- beta_scan_history without keys (all metrics) ---
print(f"\n--- beta_scan_history() NO KEYS (all {len(summary_keys)} metrics) ---")
t0 = time.time()
count = 0
for row in test_run.beta_scan_history():
    count += 1
t1 = time.time()
print(f"  {count} rows in {t1-t0:.2f}s")

# --- beta_scan_history with multiple keys ---
multi_keys = summary_keys[:10]
print(f"\n--- beta_scan_history(keys={multi_keys[:3]}...) 10 keys ---")
t0 = time.time()
count = 0
for row in test_run.beta_scan_history(keys=multi_keys):
    count += 1
t1 = time.time()
print(f"  {count} rows in {t1-t0:.2f}s")

# --- Pagination: beta_scan_history with large page_size ---
print(f"\n--- beta_scan_history(keys=['{test_key}'], page_size=10000) ---")
t0 = time.time()
count = 0
for row in test_run.beta_scan_history(keys=[test_key], page_size=10000):
    count += 1
t1 = time.time()
print(f"  {count} rows in {t1-t0:.2f}s")
