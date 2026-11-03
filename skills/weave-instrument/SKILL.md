---
name: weave-instrument
description: Add Weave tracing to Python or TypeScript agents and LLM applications. Use when asked to instrument model calls, tools, or agent turns with Weave.
---

# Weave instrumentation

Complete the [login check](#check-login-first) before initializing tracing or
sending data from this session. If no valid login exists, send only that
section's reply and stop. The check does not gate code changes that read their
credentials at runtime; see that section.

After a valid login, inspect the application's dependencies, agent boundaries,
and existing OTel provider. Confirm `entity/project` and runtime credentials
before sending data; never read, print, or commit API keys. Read
[trace fidelity](references/trace_fidelity.md) before changing any instrumentation
path. Initialize once at startup with
`weave.init("entity/project")` in Python or `await weave.init(...)` in Node.

## Fail open

Tracing must never stop or change the application. `init` contacts the trace
server and raises when it is unreachable or rejects the credentials: catch that
at startup, log it, and run untraced. If the application keeps its tracing
settings in its own configuration rather than `WANDB_*` variables, build the
exporter from them as in [endpoint configuration](references/otel_endpoint.md),
which does not contact the server at startup; do not rewrite process
environment variables to feed `init`. Test startup and one full request with
the trace URL pointed at a refused port, and with tracing unconfigured.

## Check login first

The login check gates sending trace data from this session. If the request is
to change code that reads its credentials at runtime, and this environment
won't send data (CI, an offline sandbox, or a request that says it can't
reach a trace server), skip the check: make the change, verify it with local
tests and an in-memory exporter, and report backend delivery as unverified.

Otherwise, before editing code or initializing tracing, verify the existing W&B
login for the target deployment with a noninteractive, read-only identity
request through the configured SDK or connector. Use credentials in place; never print, copy,
or inspect secret values. Do not call `login` or tracing `init` as a login probe.

If no valid login exists, stop. Do not edit, initialize, or continue this skill.
Do not run `wandb login` in this session. This session's shell is not a normal
terminal, so `wandb login --verify` exits with `No API key configured` instead
of prompting. Do not ask for a key in this chat or create credentials.

Send only this, and nothing else. No decisions, notes, plans, or
documentation links:

There's no W&B login yet, so this stops here. Open a new terminal tab and run
`wandb login --verify`. Create an API key at
https://wandb.ai/authorize?ref=models. If the browser opens a different page,
open that link again. Paste the key in that terminal, not here, and type
continue.

On a self-hosted or dedicated deployment, use `wandb login --verify --host <base-url>`
and the authorize page that command prints. Still send only that.

- If the check is unavailable or fails because of connectivity or permissions,
  report login/access as unverified and stop; do not assume credentials are invalid.
- After the user continues, confirm the application runtime can use its own
  configured credentials and access `entity/project`. A browser login does not
  establish SDK authentication. Node/Forge runtimes without a `.netrc` fallback
  need credentials configured by the user through their runtime secret mechanism.

## Choose the mechanism

- **Explicit agent spans:** use the Session SDK for turns, model calls, tools,
  and delegation when auto-instrumentation cannot produce the required tree.
  Read the [Python](references/session_sdk_python.md) or
  [TypeScript](references/session_sdk_typescript.md) reference. Create spans
  with the SDK even when the application owns the `TracerProvider`: the SDK
  writes messages, usage, and tool results to the attributes the Agents view
  reads. Do not set `gen_ai.*` content attributes by hand; if the SDK cannot
  express a span, report that gap instead.
- **Auto-instrumentation:** use only when the installed SDK captures the
  library and desired span shape. Check Python's `INTEGRATION_MODULE_MAPPING`
  or Node's `integrations/hooks.ts`, then verify emitted spans. Read
  [auto-instrumentation caveats](references/otel_auto.md); registry membership
  alone does not prove capture is active. Avoid duplicate manual wrapping.
  Patched LLM-client integrations (OpenAI and LiteLLM among them) record each
  call as a classic Weave Call (a `weave.op`), not a `chat` span, so the call
  doesn't appear under a Session SDK turn. If turns come from the Session SDK,
  open LLM spans with it too and disable implicit patching
  (`WEAVE_IMPLICITLY_PATCH_INTEGRATIONS=false`) so calls aren't logged twice.
- **Existing OTel pipeline:** add Weave export to that provider using
  [endpoint configuration](references/otel_endpoint.md).

A turn spans one user exchange, including its tool loop. An LLM span covers
one model call through stream completion; a Tool covers execution; a SubAgent
covers delegation. Record actual output/usage and preserve provider tool-call
IDs. Keep return values, exceptions, retries, and cancellation unchanged.
Close spans on failure and isolate concurrent conversations.

- **LLM spans:** open each at the lowest layer that sees both the exact request
  and the provider's final response, usually the client wrapper rather than the
  agent loop's parsed output. Set input messages before the call. For a stream,
  take output items and usage from the terminal event (completed, incomplete,
  or failed), and record what completed if the stream ends early. Read stream
  events and their payloads by `type` and attribute (`event.type`, then
  `event.response.output` and `event.response.usage`), never by class:
  wrappers and proxies such as LiteLLM return their own event and response
  classes, so an `isinstance` check against the provider SDK's classes, on the
  event or on its `response`, silently skips the recording.
- **Streams:** a streamed call's span lives until the stream is exhausted,
  raises, or is closed. Hold it in a `with` inside the generator that yields
  from the stream, so an exception raised mid-stream becomes the span's error
  instead of a silently empty span. If the application catches a provider
  error and turns it into a value instead of raising it (a retry marker, an
  error event, a fallback reply), record the error on the LLM span where it
  does that: the span's `with` never sees the exception.
- **Long-lived spans:** a span that starts in one function and ends in another,
  such as a turn closed by a callback, still goes through a context manager.
  Enter it with `ExitStack.enter_context` on the object that owns it, and close
  that stack where the work finishes. Never call `__enter__` or `__exit__` by
  hand. The [Python reference](references/session_sdk_python.md) shows both.
- **Prompts:** record a structured prompt (a list of parts or attachments) as
  its text or message parts, never as a language repr such as `str(parts)`.
- **Tool status:** mark a tool span as an error only for what the tool's
  contract calls an error. Record outcomes it reports separately, such as a
  timeout or cancellation, as attributes.

## Implement incrementally

Add one traced path at a time: a model call, its tool loop, then delegation or
concurrency where applicable. Keep the change inside the application being
instrumented: CI configuration and other services are out of scope unless the
request includes them. Still do the bookkeeping the repository's instructions
require of every change, such as ownership entries for new files. After each
change, run the path and
follow [read-back verification](references/read_back.md). Compare emitted and stored
spans with the actual execution before extending the instrumentation.

Keep tracing in one module or package: it holds the SDK imports, initialization,
and span helpers, and the instrumented code calls those helpers. If the
repository's import rules stop an inner layer from importing that module, give
the layer a small tracer interface with a no-op default and inject the
implementation from the entry point. Do not import the SDK in each layer, and
do not loosen the import rules to make room.

Fix mismatches and rerun the same check before continuing. Empty results,
failed queries, and unavailable checks are unverified, not passes. Stop at
that boundary and report what the user must do to unblock verification.

## Verify

Run the application's full test and lint commands, formatting included, on
your branch and on the unmodified tree as a baseline. Tests that error at setup
because infrastructure is missing may be reported as skipped; a test that runs
and fails, or fails only on your branch, is yours to fix. List every failing
test, not a filtered subset. When you add a dependency or module, update every
place that declares dependencies (manifests, lockfiles, inline script
metadata, images) and follow the repository's import-boundary rules.

Then run the [conformance check](references/trace_fidelity.md#conformance-check)
for the paths you instrumented. Check parentage and
`gen_ai.operation.name`: `invoke_agent` for turns/subagents, `chat` for LLMs,
`execute_tool` for tools. Agent-shaped traces belong in Agents; flat calls in
Calls. Confirm backend arrival when credentials are available; an init banner
or successful flush is not delivery proof. Otherwise report the unverified
surface and give the application's exact smoke command.

Adapted from [Weave](https://github.com/wandb/weave/tree/c003a2f4e425cde178ebcc4a2a3dbf8534fba1d3/skills/weave-instrument).
Use the installed SDK's exports when they differ from these references.
