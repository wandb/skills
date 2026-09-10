---
name: agentlens-tracing-instrument
description: Add Forge AgentLens tracing to Python or TypeScript agents. Use when asked to trace agent turns, model calls, tools, or subagents with Forge or AgentLens.
---

# Forge AgentLens instrumentation

Forge exports explicit agent spans through a private OTel provider. It does
not auto-instrument frameworks, replace the global provider, or export that
provider's spans.

Read the [Python](references/python.md) or [TypeScript](references/typescript.md)
reference. Verify the APIs against the installed SDK version.

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

Run focused application checks for success, failure, and overlapping requests.
If the full suite requires CI, browsers, or unavailable services, run feasible
local checks and report the skipped coverage.
Assert parentage, tool-call IDs, and `gen_ai.operation.name` values
`invoke_agent`, `chat`, and `execute_tool` using Forge's provider or a local
OTLP receiver. Confirm backend arrival when authorized credentials are available;
otherwise report it unverified. Init/flush success alone is not delivery proof.
