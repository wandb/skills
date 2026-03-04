---
name: wandb-mcp
description: Query and analyze W&B projects, runs, metrics, and Weave traces using the W&B MCP server tools. MCP tools handle auth, pagination, and formatting -- prefer them over writing SDK code.
---

# Weights & Biases (MCP Tools)

You have the W&B MCP server connected. Use these tools for all W&B and Weave queries.

## Available tools

| Tool | Purpose |
|------|---------|
| `query_wandb_entity_projects` | List entities and projects |
| `query_wandb_tool` | GraphQL queries for runs, metrics, configs |
| `query_weave_traces_tool` | Query Weave traces with filters and columns |
| `count_weave_traces_tool` | Count traces efficiently before querying |
| `create_wandb_report_tool` | Create shareable W&B Reports |
| `query_wandb_support_bot` | Ask W&B documentation questions |

## Workflow

1. **Discover** -- call `query_wandb_entity_projects` to find the target entity and project.
2. **Scope** -- call `count_weave_traces_tool` (or `query_wandb_tool` for runs) to understand data volume before pulling rows.
3. **Query** -- call `query_weave_traces_tool` or `query_wandb_tool` with filters to fetch the data you need.
4. **Report** -- if the user wants a shareable artifact, call `create_wandb_report_tool`.

Always call `count_weave_traces_tool` before `query_weave_traces_tool` to scope the query and avoid pulling unnecessary data.

## query_weave_traces_tool

Query Weave traces with filters. Key parameters:

- `project_id` -- `"entity/project"` format
- `filters` -- dict with `op_names`, `trace_roots_only`, `trace_ids`, `parent_ids`, `call_ids`
- `query` -- MongoDB-style advanced filter (`$eq`, `$gt`, `$contains`, `$not`, `$and`, `$or`)
- `sort_by` -- list of `{"field": "started_at", "direction": "desc"}`
- `limit` / `offset` -- pagination
- `columns` -- restrict returned fields for performance
- `include_costs` -- include token cost data

### Common filters

```
# Root traces only
filters: { "trace_roots_only": true }

# By op name (use wildcard for version hash)
filters: { "op_names": ["weave:///entity/project/op/Evaluation.evaluate:*"] }

# Error calls (exception is not null)
query: { "$expr": { "$not": [{ "$eq": [{ "$getField": "exception" }, { "$literal": null }] }] } }

# Op name contains substring
query: { "$expr": { "$contains": { "input": { "$getField": "op_name" }, "substr": { "$literal": ".score" }, "case_insensitive": true } } }
```

## count_weave_traces_tool

Count traces matching filters without fetching data. Same filter/query parameters as `query_weave_traces_tool`. Use this first to understand data volume.

## query_wandb_tool

Run GraphQL queries against the W&B API. Handles pagination automatically.

Common queries:
- List runs in a project with filters on state, config, summary metrics
- Get run details (config, summary, tags, history)
- Compare runs across sweeps or experiments

## query_wandb_entity_projects

List entities the user belongs to and projects within each entity. Call this first if you don't know the entity/project.

## create_wandb_report_tool

Create a W&B Report with markdown content, panels, and run sets. Reports are shareable and embed live data.

## Reminders

- Prefer MCP tools over writing Python/SDK code. The tools handle auth, pagination, and response formatting.
- Always count before querying to understand data volume.
- Use `columns` parameter to restrict fields when you only need specific data.
- Use filters and queries to narrow results server-side rather than fetching everything.
