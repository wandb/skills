# TypeScript Session SDK

Package `weave`. Await `weave.init` before tracing. Classes are types only;
use `startSession/startTurn/startLLM/startTool/startSubagent`, not `new Turn()`.
Wrap each concurrent request in `weave.runIsolated(async () => { ... })` because
the default context is process-wide. This text-only, non-streaming example
assumes the application's `openai` client already exists. Apply
[trace fidelity](trace_fidelity.md) when extending it to a model/tool loop
or client continuation:

```ts
import * as weave from 'weave';
await weave.init('entity/project');
const messages = [{role: 'user' as const, content: 'weather in Tokyo?'}];

const session = weave.startSession({agentName: 'research-bot'});
try {
  const turn = weave.startTurn({model: 'gpt-4o-mini'}); // one Turn per user input
  try {
    const llm = weave.startLLM({model: 'gpt-4o-mini', providerName: 'openai'});
    try {
      llm.inputMessages = messages;
      const resp = await openai.chat.completions.create({model: 'gpt-4o-mini', messages});
      const msg = resp.choices[0].message;
      llm.output(msg.content ?? '');
      if (resp.usage != null) {
        llm.record({
          usage: {
            inputTokens: resp.usage.prompt_tokens,
            outputTokens: resp.usage.completion_tokens,
          },
        });
      }
    } finally {
      llm.end();
    }
  } finally {
    turn.end();
  }
} finally {
  session.end();
}
```

`startLLM` requires an active Turn. Tools/subagents attach to the active LLM,
otherwise the Turn; close them before their owner. Use `startSubagent({name})`
for delegation. Preserve exceptions and record failure using APIs supported
by the installed SDK; `finally` alone only guarantees closure.

Wrap actual tool dispatch with `startTool({name, args, toolCallId})`, record
the execution result, and close it in `finally`. Keep the enclosing Turn open
through subsequent model/tool steps until the exchange ends.

Messages use `{role, content}`; usage uses `{inputTokens, outputTokens}`.
Carry each provider tool-call ID into `toolCallId` and matching messages.
Await `weave.flushOTel()` before a short-lived process exits. The
`--import=weave/instrument` preload is for ESM auto-capture, not these explicit spans.
