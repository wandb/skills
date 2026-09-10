# Python

Package: `coreweave-forge-sdk`; Python >=3.9. Import
`from coreweave.forge.agentlens import tracing`.

Initialize with `tracing.init("entity/project")`, or explicit keyword arguments
`api_key` and `base_url`. Environment routing uses `WANDB_API_KEY`,
`WF_TRACE_SERVER_URL`, and `WANDB_BASE_URL`; Python also supports a `.netrc`
API-key fallback. Never inspect or print the credential value.

This synthetic example exercises the API without calling an LLM provider.
Run only against an intended destination or a local test exporter:

```python
from coreweave.forge.agentlens import tracing

tracing.init("entity/project")
try:
    with tracing.Conversation(agent_name="example-agent") as conversation:
        with conversation.start_turn(user_message="Say hello") as turn:
            with turn.start_llm(model="synthetic-model") as llm:
                llm.output("Hello")
            with turn.start_tool(
                name="echo", arguments='{"text":"Hello"}', tool_call_id="call-1"
            ) as tool:
                tool.result = "Hello"
finally:
    tracing.shutdown()
```

In application code, wrap the existing model/tool call where the synthetic
output is assigned. Context managers close spans and record exceptions while
allowing the original exception to propagate. `start_*` factories start spans
immediately; entering their returned context manager does not start twice.
Direct `Turn`, `LLM`, `Tool`, and `SubAgent` construction does not emit until
`start()` or context-manager entry. Prefer the factories for live tracing.

Use `tracing.Message` for messages and `tracing.Usage` for usage; inspect their
fields in the installed SDK when mapping a provider response. Python uses
snake_case, including `input_messages`, `output_messages`, `provider_name`,
`input_tokens`, and `output_tokens`. Tool `arguments` and `result` are strings;
serialize structured values as JSON. A Tool's provider call ID is
`tool_call_id`, not its span ID. Use the same ID in `tracing.ToolCallPart`
inside `Message.assistant(tool_calls=[...])` and in
`Message.tool_result(call_id, output)` when recording the next model input.

Use `turn.start_subagent(name="researcher")` as a context manager, then its
`start_llm` and `start_tool` methods for delegated work. Keep children within
their owner's lifetime. Python uses context variables; create a Conversation
inside each independent request/task and explicitly propagate context for
thread/process handoffs as required by the application. Do not share one
ambient conversation across unrelated concurrent requests. When subagents
overlap in one context, pass `set_current=False` to `start_subagent` and use
each subagent's explicit child factories. Otherwise out-of-order completion
can corrupt the ambient context stack.

Set `include_content=False` on Conversation to omit messages, tool arguments,
results, reasoning, and media. The SDK does not redact arbitrary PII.
`llm.attach_media(uri=..., modality="image")` records a media URI.

`tracing.force_flush()` flushes pending spans; `tracing.shutdown()` flushes and
closes the worker and is registered with `atexit`. Reinitializing reroutes new
spans, including new children of older parents. The previous session survives
only until the next init/shutdown, so finish it before either operation.

Source: [Forge tracing implementation](https://github.com/coreweave/forge-sdk/tree/4c222375bcb8f3c0d31b299b76f52e09531d0189/src/coreweave/forge/agentlens/tracing).
