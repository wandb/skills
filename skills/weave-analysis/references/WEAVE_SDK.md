# Weave SDK Reference

This compact reference contains the Weave SDK surfaces used by the parity
`weave-analysis` skill and common W&B tasks.

## Init

```python
import weave

client = weave.init("entity/project")
```

## Exact Call Counts

Use server-side stats for counts.

```python
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

stats = client.server.calls_query_stats(
    CallsQueryStatsReq(project_id="entity/project", filter={"trace_roots_only": True})
)
print(stats.count)
```

For op-specific counts, filter by fully qualified op ref:

```python
op_ref = "weave:///entity/project/op/Evaluation.evaluate:*"
stats = client.server.calls_query_stats(
    CallsQueryStatsReq(project_id="entity/project", filter={"op_names": [op_ref]})
)
```

## Bounded Call Inspection

Use `get_calls` for examples, not exact counts unless paired with stats.

```python
calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    columns=["op_name", "display_name", "started_at", "summary", "output"],
    limit=20,
))
```

## Evaluation Hierarchy

Common eval hierarchy:

- `Evaluation.evaluate` root
- `Evaluation.predict_and_score` children
- scorer calls beneath or alongside `predict_and_score`

State which layer you counted.
