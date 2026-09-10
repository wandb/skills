# Python

`coreweave-forge-sdk` supports Python >=3.9. `tracing.init("entity/project")`
accepts `api_key`/`base_url`; defaults use `WANDB_API_KEY`, `WF_TRACE_SERVER_URL`,
`WANDB_BASE_URL`, and a `.netrc` key fallback. This synthetic example makes no
model call; use an intended destination or local exporter:

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

Replace synthetic results with application calls. Context managers record
exceptions and close spans without swallowing them. Factories start immediately;
direct model construction waits for `start()` or context entry.

Use `tracing.Message`, `tracing.Usage`, snake_case message/usage fields, and
JSON strings for tool arguments/results. Match `tool_call_id` to
`ToolCallPart.id` and `Message.tool_result(call_id, output)`.

Create a Conversation per request/task; propagate context across thread/process
handoffs. For overlapping subagents use `start_subagent(name=..., set_current=False)`
and its child factories to avoid out-of-order context resets.

Conversation `include_content=False` omits messages, tool data, reasoning, and
media; custom attributes/errors still require care. Prefer media URIs with
`llm.attach_media(uri=..., modality="image")` over large inline blobs.
`force_flush()` flushes; `shutdown()` closes the worker. Finish active spans
before reinitializing: new children would use the new project's exporter.
