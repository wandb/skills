---
name: agentlens-tracing-instrument
description: Add AgentLens agent tracing using the CoreWeave Forge SDK to Python or TypeScript/Node applications. Use when the user requests Forge or AgentLens tracing for agent turns, model calls, tools, or subagents. Use weave-instrument for requests to use the Weave SDK or its automatic framework instrumentation.
---

# AgentLens tracing with Forge

Instrument the application's real agent boundaries using Forge's explicit
tracing API. Forge exports agent spans over OTLP/HTTP to the Weave Agents
experience without depending on the full Weave SDK. Initializing Forge does
not automatically instrument an LLM client or framework.

## Choose the language and establish the destination

- For Python, read [references/python.md](references/python.md).
- For TypeScript/Node, read [references/typescript.md](references/typescript.md).
- For a mixed service, use both references and instrument each process's entry
  point. Do not assume context crosses a process or network boundary.

Inspect the dependency manifest and installed SDK exports before editing.
These references target Forge revision
`4c222375bcb8f3c0d31b299b76f52e09531d0189`; verify the installed version has the
APIs used. Package names do not establish that a release is available in a
registry. Follow the application's dependency management and pin a verified
release or source revision when necessary.

Confirm the destination as `entity/project`; Forge cannot discover an entity
from a bare project name. Initialize once at process startup, before traced
work. Read credentials from the runtime environment (`WANDB_API_KEY`), never
from source or a value pasted into chat. `WF_TRACE_SERVER_URL` and
`WANDB_BASE_URL` control custom routing; explicit SDK options are documented
in the language references. Check presence without printing secret values.

## Map the application, then instrument

Find the request/conversation boundary, one user-agent exchange, model call
sites, tool dispatch, delegation, streaming, and concurrent work. Explain the
files and boundaries you will change before broad edits.

- A Conversation groups related turns; a Turn represents one user-agent
  exchange. Do not create a new turn for every iteration of a tool loop.
- An LLM span encloses a model call, including stream consumption. Record
  actual messages, output, model, and available token counts; do not invent
  usage or finish a span before its stream completes.
- A Tool span encloses actual tool execution. Carry the provider's tool-call
  ID into `tool_call_id` (Python) or `toolCallId` (TypeScript), including the
  corresponding message records, so parallel calls to the same tool correlate.
- A SubAgent span encloses delegation. Create its children through that
  subagent so they retain the intended parent.

Preserve return values, exceptions, cancellation, retries, and streaming
behavior. Close spans on failure and record the failure without swallowing
the original exception. Use Python context managers and TypeScript error
recording plus `finally`. Do not add a generic wrapper framework just for
instrumentation.

Forge owns a private OpenTelemetry provider. Keep the application's global
provider intact; Forge does not export unrelated global-provider spans.
Inspect existing instrumentation for duplicate capture, and choose an owner
for each operation. Do not copy Weave auto-patching, `weave.init`,
`weave/instrument`, or Weave feature flags into this variant.

Avoid calling init per request or using it to switch projects concurrently.
Finish work before shutdown. Flush at process/job exit, not at every model
call. Do not shut down a shared exporter when one server request finishes.

Choose content recording deliberately. Python supports Conversation content
suppression; the inspected TypeScript SDK does not. Read the language reference
before recording sensitive content. Suppression is not PII redaction. Custom
attributes and error text still need care. Prefer media URIs over large
inline blobs; Forge does not upload media into Weave object storage.

## Verify the result

Run the application's tests and language checks. Exercise a successful turn,
a model/tool failure, and overlapping requests if the application is concurrent.
Check unchanged outputs/exceptions and closed spans. With a local OTLP receiver
or an SDK test exporter, assert `gen_ai.operation.name` values (`invoke_agent`,
`chat`, `execute_tool`), parentage, tool-call IDs, and conversation isolation.
Attach the test exporter to Forge's private provider rather than assuming
that replacing the global provider captures Forge spans.

Use synthetic content for local smoke checks. A flush succeeding or init
returning is not proof that the backend received a trace. When authorized
credentials and a destination are available, run one minimal application path
and confirm its conversation in the destination's Agents view. Otherwise
report local evidence and explicitly leave backend delivery unverified, with
the exact application command the user can run. Do not claim that Forge
prints Weave's startup URL banner.
