# Python Session SDK

Package `weave>=0.52.42`. Check the installed exports before using newer APIs:
from 0.53.0 the entry point is `weave.start_conversation` and message parts
come from `weave.conversation`; `weave.start_session` and `weave.session` are
the deprecated 0.52 names. Use context managers so spans close on exceptions.
This call-site example is for a text-only, non-streaming request without tools;
the application's `client` already exists. Apply
[trace fidelity](trace_fidelity.md) when extending it to a model/tool loop or
client continuation:

```python
import weave
from weave import Message, Usage

weave.init("entity/project")
prompt = "weather in Tokyo?"
messages = [{"role": "user", "content": prompt}]

with weave.start_conversation(agent_name="weather-bot") as conversation:
    with conversation.start_turn(user_message=prompt) as turn:
        with turn.llm(model="gpt-4o-mini", provider_name="openai") as llm:
            llm.input_messages = [Message.user(prompt)]
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            llm.output(resp.choices[0].message.content or "")
            if resp.usage is not None:
                llm.usage = Usage(input_tokens=resp.usage.prompt_tokens,
                                  output_tokens=resp.usage.completion_tokens)
```

Use one conversation per user conversation and one Turn per user exchange. Delegation:
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
from `weave`, and message-part classes from `weave.conversation`.

To keep a span-start failure out of the traced call while still using `with`,
enter the span through an `ExitStack` inside a guard and run the call outside
it. Guard recording the same way, so tracing errors are logged, never raised:

```python
import contextlib, logging

logger = logging.getLogger(__name__)

with contextlib.ExitStack() as stack:
    llm = None
    try:
        llm = stack.enter_context(turn.llm(model=model, provider_name="openai"))
        llm.input_messages = [Message.user(prompt)]
    except Exception:
        logger.warning("tracing failed; continuing untraced", exc_info=True)
    resp = client.responses.create(**request)
    if llm is not None:
        record_response(llm, resp)  # wraps its own body in the same guard
```

For a stream, keep the span in a `with` inside the generator that re-yields
the events. A provider error raised mid-stream then reaches the span's
`__exit__` and marks it as an error; a caller that stops reading early ends it
without one:

```python
# Match on `type` and read the payload by attribute: proxies such as LiteLLM
# return their own classes for both the event and its `response`.
TERMINAL = ("response.completed", "response.incomplete", "response.failed")

async def traced_stream(turn, request):
    with turn.llm(model=request["model"], provider_name="openai") as llm:
        llm.input_messages = to_messages(request["input"])
        try:
            async for event in await client.responses.create(**request, stream=True):
                if event.type in TERMINAL:
                    record_response(llm, event.response)
                yield event
        except GeneratorExit:
            return
```

A span that outlives the function that opens it, such as a turn ended by a
completion callback, belongs to an object that owns an `ExitStack`:

```python
class TurnTrace:
    def __init__(self, conversation, prompt: str) -> None:
        self._spans = contextlib.ExitStack()
        self.turn = self._spans.enter_context(conversation.start_turn(user_message=prompt))

    def finish(self, error: BaseException | None = None) -> None:
        if error is not None:
            self.turn.record_error(error)
        self._spans.close()
```

Top-level `start_turn/start_llm/start_tool/start_subagent` use ambient parents;
prefer explicit owner methods when the tree is available. For completed
transcripts, `log_turn(session_id=..., messages=..., spans=[LLM(...), Tool(...)])`
or `log_session(turns=[...])` supports batch logging without restructuring live
code. Preserve original execution timing and identity; history is not execution.
`set_attributes` and `add_event` require newer builds; check availability.
