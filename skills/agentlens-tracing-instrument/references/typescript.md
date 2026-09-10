# TypeScript / Node

Package: `@coreweave/forge-sdk` (ESM). The inspected package requires Node
`^18.19.0 || >=20.6.0`; check the installed package's engine requirement.
Import `tracing` from `@coreweave/forge-sdk/agentlens`.

Await `tracing.init('entity/project')` once at startup. Explicit options are
`{apiKey, baseUrl}`; environment configuration uses `WANDB_API_KEY`,
`WF_TRACE_SERVER_URL`, and `WANDB_BASE_URL`. Do not assume Python's `.netrc`
fallback exists in Node.

Every independent request/run must execute inside `tracing.runIsolated`.
Starting tracing outside it throws; there is no process-global fallback.
Its async context survives `await`. Initialize outside the request scopes.

This synthetic example calls no LLM provider. Run only against an intended
destination or a local test exporter:

```typescript
import {tracing} from '@coreweave/forge-sdk/agentlens';

await tracing.init('entity/project');
try {
  await tracing.runIsolated(async () => {
    const conversation = tracing.startConversation({agentName: 'example-agent'});
    try {
      const turn = conversation.startTurn({userMessage: 'Say hello'});
      try {
        const llm = turn.startLLM({model: 'synthetic-model'});
        try {
          llm.output('Hello');
        } catch (error) {
          llm.end({error: error instanceof Error ? error : new Error(String(error))});
          throw error;
        } finally {
          llm.end();
        }
        const tool = turn.startTool({
          name: 'echo', args: {text: 'Hello'}, toolCallId: 'call-1',
        });
        try {
          tool.end({result: 'Hello'});
        } catch (error) {
          tool.end({error: error instanceof Error ? error : new Error(String(error))});
          throw error;
        } finally {
          tool.end();
        }
      } catch (error) {
        turn.end({error: error instanceof Error ? error : new Error(String(error))});
        throw error;
      } finally {
        turn.end();
      }
    } finally {
      conversation.end();
    }
  });
} finally {
  await tracing.shutdown();
}
```

Replace synthetic output with existing calls, preserving return values and
rethrowing the original error. `finally` closes a span but does not by itself
mark failure: pass `error` to `end` on the failure path. `end` is idempotent.
Ending an owner also closes remaining descendants; end each child when its
own work completes to keep timings accurate. Ended parents reject new children.

Use camelCase fields: `llm.inputMessages`, `llm.outputMessages`, and
`llm.record({usage: {inputTokens, outputTokens}})`. Tool `args` accepts a JSON
object or serialized string; `tool.end({result})` accepts a JSON value.
`toolCallId` carries the model provider's call ID. In recorded message `parts`,
use that same `id` for `{type: 'tool_call', id, name, arguments}` and
`{type: 'tool_call_response', id, response}` (serialized string payloads).
Subagents use
`turn.startSubagent({name: 'researcher'})`, then that subagent's child factories.
Prefer explicit parent methods when operations overlap rather than relying
on the most recently active LLM. Await child work before closing its owner.

This revision has no `includeContent` option. For content-free telemetry, omit
message/output/tool-argument/result recording and pass only approved metadata;
do not populate `userMessage` either. Custom attributes/error messages still
need privacy review. Do not promise automatic content suppression or redaction.
`llm.attachMedia({uri, modality: 'image'})` records a media URI.

Await `tracing.forceFlush()` or `tracing.shutdown()` at process/job exit;
do not shut down the shared provider at the end of a server request. Forge
uses a private provider and needs no `weave/instrument` preload flag.

Source: [Forge TypeScript tracing](https://github.com/coreweave/forge-sdk/tree/4c222375bcb8f3c0d31b299b76f52e09531d0189/src/agentlens/tracing).
