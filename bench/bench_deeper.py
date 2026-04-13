"""Deeper optimization: can we reduce GraphQL payload size?

Tests:
1. per_page impact with fresh API instances (smaller pages = smaller payloads)
2. Whether the Runs object's GraphQL query can be customized
3. Concurrent page fetching
4. Alternative: use runs export or direct GraphQL
"""
import sys
import time
import json
import concurrent.futures

import wandb

path = "wandb/large_runs_demo"

# ============================================================
# Test 1: per_page impact (fresh API each time, timed carefully)
# ============================================================
print("=== per_page impact (fresh API, 20 runs each) ===")
for pp in [5, 10, 20, 50]:
    api = wandb.Api(timeout=120)
    t0 = time.time()
    runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=pp)
    rows = []
    for run in runs[:20]:
        rows.append({"id": run.id, "name": run.name, "acc": run.summary_metrics.get("acc")})
    elapsed = time.time() - t0
    print(f"  per_page={pp:>3}: {elapsed:.2f}s for {len(rows)} runs ({elapsed/len(rows)*1000:.0f}ms/run)")

# ============================================================
# Test 2: Inspect the GraphQL query — can we modify it?
# ============================================================
print("\n=== Runs QUERY inspection ===")
api = wandb.Api(timeout=120)
runs = api.runs(path, per_page=1)
print(f"  Runs type: {type(runs)}")
print(f"  Has QUERY: {hasattr(runs, 'QUERY')}")
if hasattr(runs, 'QUERY'):
    q = str(runs.QUERY)
    # Show if summaryMetrics is in the query
    if 'summaryMetrics' in q:
        idx = q.index('summaryMetrics')
        print(f"  QUERY contains 'summaryMetrics' at pos {idx}")
        print(f"  Context: ...{q[max(0,idx-50):idx+80]}...")
    else:
        print("  QUERY does NOT contain 'summaryMetrics'")
    # Show full query length
    print(f"  Full query length: {len(q)} chars")

# ============================================================
# Test 3: Direct GraphQL with field selection
# ============================================================
print("\n=== Direct GraphQL (selective fields) ===")
api = wandb.Api(timeout=120)

# Minimal query — just id, name, state
MINIMAL_QUERY = """
query Runs($project: String!, $entity: String!, $cursor: String, $perPage: Int = 20, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runCount(filters: $filters)
        runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    state
                    createdAt
                    summaryMetrics(keys: ["acc"])
                }
                cursor
            }
            pageInfo {
                endCursor
                hasNextPage
            }
        }
    }
}
"""

# Try summaryMetrics with keys parameter
SELECTIVE_QUERY = """
query Runs($project: String!, $entity: String!, $perPage: Int = 20, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    state
                    createdAt
                    summaryMetrics(keys: ["acc"])
                }
            }
        }
    }
}
"""

try:
    t0 = time.time()
    result = api.client.execute(
        wandb.vendor.gql_0_2_0.wandb_gql.gql(SELECTIVE_QUERY),
        variable_values={
            "project": "large_runs_demo",
            "entity": "wandb",
            "perPage": 20,
            "order": "-created_at",
            "filters": json.dumps({"state": "finished"}),
        },
    )
    elapsed = time.time() - t0
    edges = result.get("project", {}).get("runs", {}).get("edges", [])
    print(f"  Selective GraphQL (keys=['acc']): {elapsed:.2f}s for {len(edges)} runs")
    if edges:
        node = edges[0]["node"]
        summary = json.loads(node.get("summaryMetrics", "{}"))
        print(f"    First run: {node['name']}, acc={summary.get('acc')}")
        print(f"    Summary size: {len(node.get('summaryMetrics', ''))} chars")
except Exception as e:
    print(f"  Selective GraphQL failed: {type(e).__name__}: {str(e)[:200]}")

# Now try WITHOUT the keys filter for comparison
FULL_QUERY = """
query Runs($project: String!, $entity: String!, $perPage: Int = 20, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    summaryMetrics
                }
            }
        }
    }
}
"""

try:
    t0 = time.time()
    result = api.client.execute(
        wandb.vendor.gql_0_2_0.wandb_gql.gql(FULL_QUERY),
        variable_values={
            "project": "large_runs_demo",
            "entity": "wandb",
            "perPage": 20,
            "order": "-created_at",
            "filters": json.dumps({"state": "finished"}),
        },
    )
    elapsed = time.time() - t0
    edges = result.get("project", {}).get("runs", {}).get("edges", [])
    print(f"\n  Full GraphQL (all metrics): {elapsed:.2f}s for {len(edges)} runs")
    if edges:
        summary_size = len(edges[0]["node"].get("summaryMetrics", ""))
        print(f"    Summary JSON size per run: {summary_size:,} chars")
        total_payload = sum(len(json.dumps(e)) for e in edges)
        print(f"    Total payload: {total_payload:,} chars for {len(edges)} runs")
except Exception as e:
    print(f"  Full GraphQL failed: {type(e).__name__}: {str(e)[:200]}")
