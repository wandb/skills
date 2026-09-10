# TypeScript Session SDK

Package `weave`. Await `weave.init` before tracing. Classes are types only;
use `startSession/startTurn/startLLM/startTool/startSubagent`, not `new Turn()`.
Wrap each concurrent request in `weave.runIsolated(async () => { ... })` because
the default context is process-wide. This call-site pattern assumes the
application's model client, messages, and tool dispatcher already exist:

```ts
import * as weave from 'weave';
await weave.init('entity/project');

const session = weave.startSession({agentName: 'research-bot'});
try {
  const turn = weave.startTurn({model: 'gpt-4o-mini'}); // one Turn per user input
  try {
    let toolCalls = [];
    const llm = weave.startLLM({model: 'gpt-4o-mini', providerName: 'openai'});
    try {
      llm.inputMessages = [{role: 'user', content: prompt}];
      const resp = await openai.chat.completions.create({model, messages, tools});
      const msg = resp.choices[0].message;
      toolCalls = msg.tool_calls ?? [];
      llm.output(msg.content ?? '');
      llm.record({
        usage: {
          inputTokens: resp.usage?.prompt_tokens,
          outputTokens: resp.usage?.completion_tokens,
        },
      });
    } finally {
      llm.end();
    }

    for (const tc of toolCalls) {
      const tool = weave.startTool({
        name: tc.function.name,
        args: tc.function.arguments,
        toolCallId: tc.id,
      });
      try {
        tool.result = await runTool(JSON.parse(tc.function.arguments));
      } finally {
        tool.end();
      }
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

Messages use `{role, content}`; usage uses `{inputTokens, outputTokens}`.
Carry each provider tool-call ID into `toolCallId` and matching messages.
Await `weave.flushOTel()` before a short-lived process exits. The
`--import=weave/instrument` preload is for ESM auto-capture, not these explicit spans.
