# Read-back verification

## Check each increment

- Start with a small synthetic request and one model span. Add a tool loop,
  delegation, and concurrency only after the preceding path passes.
- Capture the exact `trace_id`, span/tool-call IDs, input, provider usage,
  execution timings, and delivered output as independent expected results.
- Flush once the test path completes. For local checks, inspect an in-memory
  exporter or local OTLP receiver; do not change the running service's shutdown
  behavior to force delivery.
- Query the same project and trace. Compare parentage, messages, usage, tool
  attempts, and timing using [trace fidelity](trace_fidelity.md).
- Fix, rerun, and requery before the next increment. At the end, run the
  [conformance check](trace_fidelity.md#conformance-check) across completed paths.

## Query stored agent spans

Use the configured trace-service client or authorized connector. The agent-span
endpoint is `POST /agents/spans/query`, not the legacy Calls API. Hosted Weave
uses `https://trace.wandb.ai`; use the deployment's configured endpoint elsewhere.
A browser session must use its authorized same-origin proxy, not copied cookies.
Reuse existing authentication in place; never put keys or cookies into examples,
logs, or the report. Check the installed client/server schema before adapting
this request.

Example body; replace `entity/project` and `TRACE_ID` with the test's identifiers:

```json
{
  "project_id": "entity/project",
  "query": {
    "$expr": {
      "$eq": [{"$getField": "trace_id"}, {"$literal": "TRACE_ID"}]
    }
  },
  "include_details": true,
  "include_costs": true,
  "sort_by": [{"field": "span_id", "direction": "asc"}],
  "limit": 200,
  "offset": 0
}
```

- Inspect `spans` and `total_count`. If results exceed the page, increase
  `offset` by the returned row count until complete; do not assess a truncated tree.
- Allow a bounded ingestion wait with a finite deadline. No rows by the deadline,
  an empty page before completion, or a failed request means **unverified**.
- On `401`/`403`, stop and report authentication/project-access failure. Follow
  the skill's login gate; do not switch identity or elevate permissions.
- Use `include_details` for message/tool payloads and `include_costs` for pricing.
  Omitted fields are not proof of absent instrumentation. Costs may be unavailable
  without matching prices; compare raw usage with provider responses separately.
- For two-turn checks, query each captured trace ID. Keep the query bounded to
  test data; avoid fetching unrelated production conversations.

Record the query scope, returned IDs/counts, checks and failures, and trace link.
Keep private payloads out of reports. Backend receipt does not prove rendering:
inspect the same turns in Agents when authorized. If access is unavailable,
report that check unverified and stop rather than extending unproven changes.
