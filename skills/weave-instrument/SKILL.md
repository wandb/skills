---
name: weave-instrument
description: Add Weave tracing to Python or TypeScript agents and LLM applications. Use when asked to instrument model calls, tools, or agent turns with Weave.
---

# Weave instrumentation

Inspect the application's dependencies, agent boundaries, and existing OTel
provider. Confirm `entity/project` and runtime credentials before sending data;
never read, print, or commit API keys. Initialize once at startup with
`weave.init("entity/project")` in Python or `await weave.init(...)` in Node.

## Choose the mechanism

- **Explicit agent spans:** use the Session SDK for turns, model calls, tools,
  and delegation when auto-instrumentation cannot produce the required tree.
  Read the [Python](references/session_sdk_python.md) or
  [TypeScript](references/session_sdk_typescript.md) reference.
- **Auto-instrumentation:** use only when the installed SDK captures the
  library and desired span shape. Check Python's `INTEGRATION_MODULE_MAPPING`
  or Node's `integrations/hooks.ts`, then verify emitted spans. Read
  [auto-instrumentation caveats](references/otel_auto.md); registry membership
  alone does not prove capture is active. Avoid duplicate manual wrapping.
- **Existing OTel pipeline:** add Weave export to that provider using
  [endpoint configuration](references/otel_endpoint.md).

A turn spans one user exchange, including its tool loop. An LLM span covers
one model call through stream completion; a Tool covers execution; a SubAgent
covers delegation. Record actual output/usage and preserve provider tool-call
IDs. Keep return values, exceptions, retries, and cancellation unchanged.
Close spans on failure and isolate concurrent conversations.

## Verify

Run the application's tests and a minimal traced path. Check parentage and
`gen_ai.operation.name`: `invoke_agent` for turns/subagents, `chat` for LLMs,
`execute_tool` for tools. Agent-shaped traces belong in Agents; flat calls in
Calls. Confirm backend arrival when credentials are available; an init banner
or successful flush is not delivery proof. Otherwise report the unverified
surface and give the application's exact smoke command.

Adapted from [Weave](https://github.com/wandb/weave/tree/c003a2f4e425cde178ebcc4a2a3dbf8534fba1d3/skills/weave-instrument).
Use the installed SDK's exports when they differ from these references.
