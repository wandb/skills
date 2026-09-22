# Python Session SDK

Package `weave>=0.52.42`. Check the installed exports before using newer APIs.
Use context managers so spans close on exceptions. This call-site example is
for a text-only, non-streaming request without tools; the application's `client`
already exists. Apply [trace fidelity](trace_fidelity.md) when extending it to
a model/tool loop or client continuation:

```python
import weave
from weave import Message, Usage

weave.init("entity/project")
prompt = "weather in Tokyo?"
messages = [{"role": "user", "content": prompt}]

with weave.start_session(agent_name="weather-bot") as session:
    with session.start_turn(user_message=prompt) as turn:
        with turn.llm(model="gpt-4o-mini", provider_name="openai") as llm:
            llm.input_messages = [Message.user(prompt)]
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            llm.output(resp.choices[0].message.content or "")
            if resp.usage is not None:
                llm.usage = Usage(input_tokens=resp.usage.prompt_tokens,
                                  output_tokens=resp.usage.completion_tokens)
```

Use one Session per conversation and one Turn per user exchange. Delegation:
`with turn.subagent(name="researcher") as sub`, then `sub.llm(...)` or
`sub.tool(...)`. Context follows Python contextvars; propagate it explicitly
across thread/queue boundaries and keep children within their parent's lifetime.

LLM fields use snake_case: `input_messages`, `output_messages`,
`Usage(input_tokens=..., output_tokens=...)`. `.output(text)` appends an assistant
message; `.record(...)` accepts messages, usage, reasoning, response ID, and
finish reasons. Pass `provider_name`; it is not inferred. Record usage only
when the provider returns it.

Wrap the actual dispatcher with `turn.tool(...)`; set its result from execution,
not a constant or a later history entry. Tool arguments/results accept
JSON-compatible values. Carry the provider's `tool_call_id` into
`ToolCallPart(id=..., name=..., arguments=...)` and
`Message.tool_result(call_id=..., output=...)`. Import `Message` and `Usage`
from `weave`, and message-part classes from `weave.session`.

Top-level `start_turn/start_llm/start_tool/start_subagent` use ambient parents;
prefer explicit owner methods when the tree is available. For completed
transcripts, `log_turn(session_id=..., messages=..., spans=[LLM(...), Tool(...)])`
or `log_session(turns=[...])` supports batch logging without restructuring live
code. Preserve original execution timing and identity; history is not execution.
`set_attributes` and `add_event` require newer builds; check availability.
