---
name: agentlens-tracing-instrument
description: Add Forge AgentLens tracing to Python or TypeScript agents. Use when asked to trace agent turns, model calls, tools, or subagents with Forge or AgentLens.
---

# Forge AgentLens instrumentation

Complete the [login check](#check-login-first) before editing or initializing.

Forge exports explicit agent spans through a private OTel provider. It does
not auto-instrument frameworks, replace the global provider, or export that
provider's spans.

Read the [Python](references/python.md) or [TypeScript](references/typescript.md)
reference and [trace fidelity](references/trace_fidelity.md) before editing.
Verify the APIs against the installed SDK version.

## Check login first

Before editing code or initializing tracing, verify the existing W&B login for
the target deployment with a noninteractive, read-only identity request through
the configured SDK or connector. Use credentials in place; never print, copy,
or inspect secret values. Do not call `login` or tracing `init` as a login probe.

- If no valid login exists, **stop and return**. Ask the user to log in manually
  with `wandb login --verify` for the target deployment; include the
  [W&B login documentation](https://docs.wandb.ai/models/ref/cli/wandb-login).
  Do not launch login, request a key in chat, or create credentials for them.
- If the check is unavailable or fails because of connectivity or permissions,
  report login/access as unverified and stop; do not assume credentials are invalid.
- A browser login does not establish SDK authentication. Confirm the application
  runtime can use its own configured credentials and access `entity/project`.
  Node/Forge runtimes without a `.netrc` fallback need credentials configured
  by the user through their runtime secret mechanism.

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

## Implement incrementally

Add one traced path at a time: a model call, its tool loop, then delegation or
concurrency where applicable. After each change, run the path and follow
[read-back verification](references/read_back.md). Compare emitted and stored
spans with the actual execution before extending the instrumentation.

Fix mismatches and rerun the same check before continuing. Empty results,
failed queries, and unavailable checks are unverified, not passes. Stop at
that boundary and report what the user must do to unblock verification.

## Verify

Run the [conformance check](references/trace_fidelity.md#conformance-check)
for the paths you instrumented, including success, failure, and overlapping requests.
If the full suite requires CI, browsers, or unavailable services, run feasible
local checks and report the skipped coverage.
Assert parentage, tool-call IDs, and `gen_ai.operation.name` values
`invoke_agent`, `chat`, and `execute_tool` using Forge's provider or a local
OTLP receiver. Confirm backend arrival when authorized credentials are available;
otherwise report it unverified. Init/flush success alone is not delivery proof.
