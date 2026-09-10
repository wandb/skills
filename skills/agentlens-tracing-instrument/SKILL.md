---
name: agentlens-tracing-instrument
description: Add Forge AgentLens tracing to Python or TypeScript agents. Use when asked to trace agent turns, model calls, tools, or subagents with Forge or AgentLens.
---

# Forge AgentLens instrumentation

Forge exports explicit agent spans through a private OTel provider. It does
not auto-instrument frameworks, replace the global provider, or export that
provider's spans. Use `weave-instrument` when the user requests the Weave SDK.

Read the [Python](references/python.md) or [TypeScript](references/typescript.md)
reference. Check installed exports and pin a verified release/source revision;
these references target Forge `4c222375bcb8f3c0d31b299b76f52e09531d0189`.

## Instrument

- Confirm `entity/project`; a bare project cannot resolve its entity. Initialize
  once per process with runtime `WANDB_API_KEY`. Never print or commit credentials.
- Map Conversation → Turn → LLM/Tool/SubAgent to real boundaries: one turn per
  user exchange, one LLM span through stream consumption, one tool per execution.
  Preserve provider tool-call IDs and use explicit parents for delegated work.
- Preserve outputs, exceptions, retries, and cancellation. Close spans on failure;
  avoid duplicate capture from existing instrumentation.
- Keep project routing stable while work is active. Flush/shutdown at process
  exit, not per request. Python supports content suppression; this TypeScript
  revision requires omitting sensitive fields explicitly. Neither redacts PII.

## Verify

Run application checks and exercise success, failure, and overlapping requests.
Assert parentage, tool-call IDs, and `gen_ai.operation.name` values
`invoke_agent`, `chat`, and `execute_tool` using Forge's provider or a local
OTLP receiver. Confirm backend arrival when authorized credentials are available;
otherwise report it unverified. Init/flush success alone is not delivery proof.
