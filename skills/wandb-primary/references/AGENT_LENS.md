<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Agent Lens

[CoreWeave Agent Lens](https://docs.coreweave.com/products/agent-lens/what-is-agent-lens)
shows agent conversations, turns, model calls, and tools. Open a project at
`https://forge.coreweave.com/agent-lens/{entity}/{project}`. Encode each path
segment separately when constructing a project or conversation link.

## Choose the integration before changing code

Follow the [integration chooser](https://docs.coreweave.com/products/agent-lens/get-started/integrations)
and the matching guide. The package depends on the application:

| Application | Integration |
|---|---|
| OpenAI, Anthropic, or Google Gen AI provider SDK | Python `coreweave` or npm `@coreweave/forge-sdk`, with an explicit conversation |
| OpenAI Agents SDK, Claude Agent SDK, or Google ADK | Weave SDK, with the framework's group or session identifier reused across turns |
| Claude Code, Codex, Google Antigravity, or OpenClaw | The matching Forge plugin |
| Pi | The documented Weave extension |
| Custom agent loop | Forge SDK conversation, turn, model, and tool helpers |

Keep a working Weave integration when its guide requires it. An existing Weave
Calls adapter alone does not establish Agent Lens compatibility. Function
tracing and evaluation workflows continue to use Weave Calls and evaluations.

## Instrument a custom agent

The published Python package is **`coreweave`**; the import namespace is
`coreweave.forge.agentlens.tracing`. The npm package is
**`@coreweave/forge-sdk`**. Neither requires the full Weave package for manual
agent instrumentation.

```bash
pip install --pre coreweave
```

Set `WANDB_API_KEY` using your [Forge API key](https://forge.coreweave.com/settings#apikeys).
This synthetic example records a tool result without calling a model provider:

```python
import json
from coreweave.forge.agentlens import tracing

tracing.init("entity/project")

with tracing.Conversation(agent_name="calculator") as conversation:
    with conversation.start_turn(user_message="What is 18 plus 24?") as turn:
        with turn.start_tool(name="add", arguments=json.dumps({"a": 18, "b": 24})) as tool:
            tool.result = json.dumps({"result": 42})

tracing.shutdown()
```

Reuse the conversation for related turns. For real provider calls, install the
extra specified by the provider guide, such as `coreweave[openai]`, and make the
calls inside a conversation. TypeScript uses `tracing.runIsolated(...)` to
isolate concurrent requests and explicit `end()` calls to close spans.

See the [custom-agent quickstart](https://docs.coreweave.com/products/agent-lens/get-started/custom-agents)
for a complete model-and-tool loop.

## Query the right data

Agent spans and classic Weave Calls are separate data surfaces. An empty
`client.get_calls()` result does not establish that a project has no agent
conversations. Weave 0.53.12 exposes public methods for agent transcripts:

```python
import weave

client = weave.init("entity/project")
conversation = client.get_agent_turns(conversation_id="CONVERSATION_ID", limit=20)
for turn in conversation.turns:
    for message in turn.messages:
        print(message.type)

turn = client.get_agent_turn(trace_id="TRACE_ID")
weave.finish()
```

Check the installed SDK version and [WEAVE_SDK.md](WEAVE_SDK.md) before choosing
a query method. Older releases may not expose these methods. The bundled
`weave_agent_ops.py` helper depends on private SDK internals and is not validated
with Weave 0.53.12; use the public methods above for this release.

`conversation_id`, OTel `trace_id`, OTel `span_id`, Weave Call IDs, and Calls
`thread_id` are different identifiers. Do not substitute one for another.
Use bounded queries, paginate when needed, and report the queried time range
and whether counts describe the full result or a sample.

Insights classifications and conversation tags need their own APIs; raw span
errors are not semantic failure categories. If using the W&B MCP server,
inspect its advertised tools before calling them. Its [Agent Lens tools](https://github.com/wandb/wandb-mcp-server/blob/main/docs/agent-lens-tools.md)
require the explicitly enabled Agent Lens profile and a supported deployment.
They are not available in every installation. Consult the server's
[release evidence](https://github.com/wandb/wandb-mcp-server/blob/main/docs/releases/README.md)
before recommending an upgrade.

For Insights, check classification coverage and clustering status before
reporting category statistics. No classified results does not mean no
conversations or no failures. For tags, distinguish configured criteria,
human-applied tags, and judge results. Preserve coverage and sampling limits
in any answer.

Do not infer a stable public API from an internal service route or database
schema. Use the installed SDK's documented models, advertised MCP tools, and
public product guides for the user's deployment.

## Public guides

- [View agent activity](https://docs.coreweave.com/products/agent-lens/conversations/view-activity)
- [Agent Lens documentation](https://docs.coreweave.com/products/agent-lens)
- [Choose an integration](https://docs.coreweave.com/products/agent-lens/get-started/integrations)
