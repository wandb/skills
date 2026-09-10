# TypeScript / Node

ESM package `@coreweave/forge-sdk`; Node `^18.19.0 || >=20.6.0`.
Await init once outside request scopes. Options `apiKey`/`baseUrl` override
`WANDB_API_KEY`, `WF_TRACE_SERVER_URL`, and `WANDB_BASE_URL`; no `.netrc` fallback.
Every request needs `runIsolated`; tracing outside it throws.
This SDK revision has no `Symbol.dispose`/`Symbol.asyncDispose`, so `using`
cannot replace explicit `end()` calls. Record errors separately from cleanup.

Synthetic example (no model call), for an intended destination or local exporter:

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

`end({error})` records failure; `finally` alone only closes. Rethrow the original
error. End is idempotent; closing an owner closes remaining children, and ended
parents reject new children. Await child work and prefer explicit parent factories.

Use `inputMessages`, `outputMessages`, and `record({usage: {inputTokens, outputTokens}})`.
Tool `args` accepts an object/string; `end({result})` accepts JSON values. Match
`toolCallId` with message parts `tool_call`/`tool_call_response` carrying the same
`id`. Delegate with `turn.startSubagent({name})` and its child factories.

There is no `includeContent` option: omit messages (including `userMessage`),
outputs, and tool data when content must not be recorded. Custom attributes/errors
still need care. Prefer `attachMedia({uri, modality: 'image'})` over inline blobs.
Await `forceFlush()`/`shutdown()` at process exit, never per server request.
