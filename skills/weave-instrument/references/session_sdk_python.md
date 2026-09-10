# Python Session SDK

Package `weave>=0.52.42`. Check the installed exports before using newer APIs.
Use context managers so spans close on exceptions. The following call-site
example assumes the application's `client` and `messages` already exist:

```python
import weave
from weave import Usage

weave.init("entity/project")

with weave.start_session(agent_name="weather-bot") as session:
    with session.start_turn(user_message="weather in Tokyo?") as turn:   # one Turn per user input
        with turn.llm(model="gpt-4o", provider_name="openai") as llm:
            resp = client.chat.completions.create(model="gpt-4o", messages=messages)
            llm.output(resp.choices[0].message.content or "")
            llm.usage = Usage(input_tokens=resp.usage.prompt_tokens,
                              output_tokens=resp.usage.completion_tokens)
        with turn.tool(name="get_weather", arguments={"city": "Tokyo"}, tool_call_id="tc_1") as tool:
            tool.result = "75F"          # arguments and result: dict, list, or scalar, auto-JSON-encoded
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

Tool arguments/results accept JSON-compatible values. Carry the provider's
`tool_call_id` into `ToolCallPart(id=..., name=..., arguments=...)` and
`Message.tool_result(call_id=..., output=...)`. Import `Message` and `Usage`
from `weave`, and message-part classes from `weave.session`.

Top-level `start_turn/start_llm/start_tool/start_subagent` use ambient parents;
prefer explicit owner methods when the tree is available. For completed
transcripts, `log_turn(session_id=..., messages=..., spans=[LLM(...), Tool(...)])`
or `log_session(turns=[...])` supports batch logging without restructuring live
code. `set_attributes` and `add_event` require newer builds; check availability.
