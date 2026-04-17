"""Deeper probe of the large project — metric shapes, history sizes, key patterns."""
import time
import wandb

api = wandb.Api()
path = "wandb/large_runs_demo"

# 1. Check how many summary_metrics keys exist on typical runs
print("=== Summary metrics key counts ===")
runs = api.runs(path, filters={"state": "finished"}, order="-created_at")
for run in runs[:5]:
    keys = list(run.summary_metrics.keys())
    metric_keys = [k for k in keys if not k.startswith("_")]
    print(f"  {run.name} ({run.id}): {len(keys)} total keys, {len(metric_keys)} metric keys")
    if metric_keys:
        print(f"    First 20: {metric_keys[:20]}")

# 2. Find a run with actual history data
print("\n=== Finding runs with history ===")
for run in runs[:10]:
    t0 = time.time()
    df = run.history(samples=5)
    t = time.time() - t0
    cols = [c for c in df.columns if not c.startswith("_")]
    print(f"  {run.name}: {len(df)} rows, {len(cols)} metric cols, {t:.2f}s")
    if len(df) > 0 and len(cols) > 0:
        print(f"    Cols: {cols[:10]}")
        # Try scan_history with an actual key
        key = cols[0]
        t0 = time.time()
        count = 0
        for row in run.scan_history(keys=[key]):
            count += 1
            if count >= 10:
                break
        t = time.time() - t0
        print(f"    scan_history(keys=[{key!r}]) first 10: {count} rows, {t:.2f}s")

# 3. Benchmark runs_to_dataframe-like iteration at various limits
print("\n=== Iteration cost at different limits ===")
for limit in [10, 50, 100, 200]:
    t0 = time.time()
    rows = []
    for run in runs[:limit]:
        row = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
        }
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[f"config.{k}"] = v
        row["acc"] = run.summary_metrics.get("acc")
        rows.append(row)
    t = time.time() - t0
    print(f"  runs[:{limit}] -> {len(rows)} rows in {t:.2f}s ({t/max(len(rows),1)*1000:.0f}ms/run)")

# 4. Check if server-side filters help
print("\n=== Server-side filter cost ===")
for desc, filters in [
    ("no filter", {}),
    ("state=finished", {"state": "finished"}),
    ("state=finished + order", {"state": "finished"}),
]:
    t0 = time.time()
    r = api.runs(path, filters=filters, order="-created_at" if "order" in desc else None)
    _ = r[:10]  # force fetch
    t = time.time() - t0
    print(f"  {desc}: {t:.2f}s for first page")
