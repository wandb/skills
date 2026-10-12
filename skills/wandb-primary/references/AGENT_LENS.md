<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Agent Lens

Agent Lens is the Forge product at `/agent-lens/{entity}/{project}`. Its surfaces are Overview, Insights,
Conversations and Tags. This card covers where that data lives, which parts are
queryable, and what the figures on each surface mean.

## Two planes, two ways in

Agent Lens is one UI over two backends. Which one holds the answer decides
how you query it, so settle that before writing code.

| Plane | Holds | Your access |
|---|---|---|
| **Weave Agents plane** | conversations, turns, spans, tools, models, tokens, cost, latency, errors, custom attributes | **the SDK** — `client.server.agent_*` |
| **Agent Lens service** | Insights categories and clusters, the Insights summary, tags and their judges, human-applied conversation tags, saved views | **its HTTP API** at `https://forge.coreweave.com/api/pulse` with the W&B API key (see "The service API" below) |

The Agents plane is the same data Weave's agents surfaces show, so
`skills/wandb-primary/references/WEAVE_SDK.md` and `scripts/weave_agent_ops.py` apply
unchanged. Ingestion is OTel: a project can hold Agents-plane spans and zero
Weave Calls, so `client.get_calls()` returning nothing proves nothing here.

The second plane is a separate service with its own HTTP API. Its clustering
output lives in ClickHouse (`weave_trace_db.signature_cluster_runs`, keyed by
`project_id`, a `signature_type` of `intent` or `failure`, `status`, and
`window_start`). The trace server exposes exactly nine `agent_*` methods and
none reach classifications or tags, and the classifications are **not**
duplicated into any span field — so anything in this plane comes from the
service API, not from a span query.

`is_error` and `status_code` are raw span fields recording whether a span
failed. They are not the semantic failure categories Insights assigns, which
come from classification over the conversation's content — the two describe
different things and will not agree.

## Instrumenting an agent

Agent Lens's SDK is the **Forge SDK**: Python package `coreweave-forge-sdk`,
npm package `@coreweave/forge-sdk`, namespace `coreweave.forge.agentlens.tracing`.
`weave.init` + `weave.start_conversation` is the older path — the same API
under the Weave namespace. Both land in the Agents plane above, and an app
already instrumented either way needs no second integration.

The Forge SDK does not depend on the `weave` or `wandb` packages; it exports
over OTLP/HTTP and stamps every span's resource with `weave.sdk.name = "forge"`.

```python
from coreweave.forge.agentlens import tracing

tracing.init("entity/project")  # reads WANDB_API_KEY; entity is required

with tracing.start_conversation(agent_name="my-agent") as conversation:
    with conversation.start_turn(user_message="…") as turn:
        with turn.start_llm(model="gpt-5") as llm:
            llm.output("…")
        with turn.start_tool(name="search", arguments={"q": "…"}) as tool:
            tool.result = "…"

tracing.shutdown()
```

`start_conversation`, `start_turn`, `start_llm`, `start_tool` and
`start_subagent` are the same names and arguments as the Weave SDK's
"Conversation SDK" (`skills/wandb-primary/references/WEAVE_SDK.md`), so a Weave snippet
translates by swapping the import and `init`. `pip install
"coreweave-forge-sdk[openai]"` or `[anthropic]` autopatches that provider's
client after `tracing.init()`, so a standalone provider call still appears as a
turn. TypeScript wraps each request in `tracing.runIsolated(...)` and ends
spans explicitly (`llm.end()`, `turn.end()`, `conversation.end()`), usually in
a `finally`.

## The methods — nine on the server, eight you can call

On `client.server`, with request models in `weave.shared.agents.types`:

| Need | Method |
|---|---|
| raw spans, or grouped rows | `agent_spans_query` |
| chart-ready series and histograms | `agent_spans_stats` |
| a conversation transcript | `agent_conversation_chat` |
| one turn or trace transcript | `agent_traces_chat` |
| the spans under a conversation | `agent_conversation_spans` — **not bound on the remote client**, see below |
| message-text search | `agent_search` |
| this project's custom attribute keys and types | `agent_custom_attrs_schema` |
| materialized agents and versions | `agent_agents_query`, `agent_versions_query` |

Two separate things about `agent_conversation_spans`, easy to conflate:

`trace_server_interface.py` declares every method with a `...` body — that is
the Protocol convention, `call_start` included — so a `...` there is not
evidence of a stub. The server implements this one:
`clickhouse_trace_server_batched.py` delegates it to
`AgentQueryHandler.conversation_spans`.

But `RemoteHTTPTraceServer`, which is the client you actually hold, binds eight
of the nine and **not this one** — audited by calling the other eight, all of
which return typed responses. The two halves connect: because the client has
no override, the call falls through to the Protocol declaration, and that
`...` body evaluates to `None`. So a bare `None` is the inherited stub
answering, not the server.

That return value is the tell, and it generalizes: these methods answer with a
typed response object or not at all. A typed `AgentConversationSpansRes` whose
collection is empty means no spans; a bare `None` means the client never
dispatched the call, so it carries no information about whether data exists. On
a conversation that returns `None` this way, `agent_spans_query` filtered on
the same `conversation_id` returns 221 spans — that filter is the equivalent
route to a conversation's spans.

For transcripts prefer the helper over hand-rolling pagination:

```bash
python skills/wandb-primary/scripts/weave_agent_ops.py --path ENTITY/PROJECT conversation CONVERSATION_ID
python skills/wandb-primary/scripts/weave_agent_ops.py --path ENTITY/PROJECT trace TRACE_ID
```

## The aggregate vocabulary

`agent_spans_stats` takes `metrics`, each a `value` reference plus
`aggregations`. `agent_spans_query` takes `measures` and returns `spans` when
`group_by` is empty and `groups` when it is not.

**Value sources:** `field` (a semantic span field or span column), `derived`, or
one of `custom_attrs_string` / `custom_attrs_int` / `custom_attrs_float` /
`custom_attrs_bool` for user-defined keys.

**Derived values**, which is where the useful ones live:

| Key | Type |
|---|---|
| `duration_ms`, `total_tokens`, `total_cost_usd`, `input_cost_usd`, `output_cost_usd` | number |
| `is_error`, `is_invocation` | boolean |

**Aggregations allowed per type** — the server rejects the rest:

| Type | Aggregations |
|---|---|
| number | `sum`, `avg`, `min`, `max`, `count`, `count_distinct` |
| boolean | `count`, `count_distinct`, `count_true`, `count_false` |
| string | `count`, `count_distinct` |
| datetime | `min`, `max`, `count`, `count_distinct` |

Three traps in this vocabulary:

**Percentiles are on a 0–100 scale**, validated only for that range, and valid
solely on number metrics. `[50, 95]` is the median and p95. `[0.5, 0.95]` is
accepted and returns the 0.5th and 0.95th percentile — a wrong answer rather
than an error, so a latency chart built that way looks plausible and is not.

**`total_count` is how many matched, not how many you got.** The response
carries `spans` (or `groups`) capped by your `limit` alongside a `total_count`
for the whole match. A response reading 221 matched with 100 returned is one
response describing both numbers, not a partial failure; the rest is reachable
by paging.

**`include_costs` belongs to `agent_spans_query`, not `agent_spans_stats`.**
`AgentSpanStatsReq` has no such field: ask for the derived `total_cost_usd`
metric and stats returns it. On `agent_spans_query` the flag is opt-in and
absent by default, because per-span costs need a join against token prices.
`include_details` is the same shape of flag — it pulls messages and tool
payloads, and is meant for one trace rather than a list.

## Reproducing what the Overview shows

Seven of the Overview's nine cards are Agents-plane aggregates over its
selected time range, so those are reproducible. Two are not:

| Card | How to build it |
|---|---|
| Activity | `count_distinct` of `conversation_id`; turns are `is_invocation` |
| Errors | `count_true` of derived `is_error`, distinct conversations and traces |
| Conversations with errors | the same, grouped by `conversation_id`, most recent first |
| Cost | `sum` of derived `total_cost_usd` |
| Latency | derived `duration_ms` with `percentiles=[50, 95]`, filtered to `invoke_agent` |
| Tools | `group_by` tool name, using span and error counts |
| Tools with errors | the same, as a rate rather than a count |
| Tags, Trending tags | the service API: `POST /conversation-tags/distribution` (see below) |

A count you produce and a count on the page can differ legitimately: the page
has a time range, and it counts distinct conversations where a raw span count
does not.

**`agent_spans_stats` computes these server-side**, including percentiles via
the metric spec — `percentiles=[50, 95]` on a derived `duration_ms` metric.
The same statistic computed in Python over `agent_spans_query` rows differs in
two ways: those rows are capped by `limit`, so the result describes a sample of
the match rather than all of it, and on a busy project the fetch is large
enough to run into the shell's time limit. A week of `wandb-agent-dev` is a few
hundred `invoke_agent` spans; the same week of `wandb-agent-prod` is over
18,000.

## Reading the Insights numbers you cannot compute

These figures come from the Agent Lens service rather than the Agents plane,
so they are visible on the page and not reachable by query. Each breakdown mode
draws **one stacked-area chart** — failure categories over time, or intent
categories over time — above a per-category table. The chart buckets daily,
plots trace counts, and excludes the `no_failure` category, so it shows the
shape of what went wrong rather than overall volume. In that table, **Time/Turn
and Cost/Turn are count-weighted averages across the category's traces**, not
per-conversation means, so one slow expensive cluster can lift a category's
figure well above what a typical turn in it costs. Percent is the category's
share of all classified traces in the selected range; its signed delta compares
against the period of the same length just before it, and is absent rather
than zero when that previous period had none of that category.

Trace, failed and frustrated counts are summed across buckets, but
**`conversation_count` is the largest single bucket's count, not a sum** — the
same conversation can appear in several buckets, so summing would double-count
it. A category's conversation figure is therefore a floor, not a total.
**Intent mode carries a Failure rate column** — failed over total for the
category — that failure mode does not show. Frustration is the share of the category's traces whose user
sentiment was classified as frustrated. A cluster's percentage is its share of **all
classified traces**, not its share within its category — the frontend computes
it as the cluster's within-category fraction multiplied by the category's own
percentage, so cluster figures under one category sum to that category's
percent rather than to 100.

## The service API

Base URL `https://forge.coreweave.com/api/pulse`.
Authentication is `Authorization: Bearer <W&B API key>`; read the key from the
`WANDB_API_KEY` environment variable and never print it. The project goes in `x-wandb-entity` and
`x-wandb-project` headers; the caller's own project permissions apply, so a
project the user cannot read returns `403`, and a missing or invalid key `401`.
Responses are JSON with the payload under `data`. The inventory is what backs
each surface:

| Area | Endpoints |
|---|---|
| Insights | `GET /insights/latest-week` · `GET /insights/intent-category-breakdowns?date=` (serves both views despite the name) · `POST /insights/executive-summary` · `GET /insights/{intent\|failure}/categories/{id}/example-turns` · `GET /insights/matching-turns` · `POST /insights/conversations/search` |
| Tags | `GET/POST /tags` · `GET/PATCH/DELETE /tags/{id}` · `POST /tags/{id}/test` · `GET/POST /tags/{id}/alignment_examples` · `PATCH/DELETE /tags/{tag_id}/alignment_examples/{example_id}` |
| Conversation tags | `POST /conversation-tags/query` · `POST /conversation-tags/conversations/query` · `POST /conversation-tags/distribution` · `GET/POST/DELETE /conversation-tags` |
| Saved views | `GET/POST /views` · `PATCH/DELETE /views/{id}` |
| Other | `GET /models` · `POST /conversations/{id}/summarize` · `GET /users/me` · `GET /users/me/teams` |

The Tags page merges two of these. `GET /tags` returns the **configured** tags
— `id`, `name`, `criteria`, `color`, `llm_judge_enabled`, `agent_name_filter`,
`inserted_at`, `updated_at` — and `GET /conversation-tags` returns the names of
tags people have applied to conversations by hand, which is where a row with no
criteria comes from. A tag is listed once whichever source it came from, so the
page's row count is the union. Conversation counts and last-used times come
from `POST /conversation-tags/distribution`, whose body requires `after_ms`,
`before_ms` and `time_bucket_seconds`; it is also what fills the Overview's tag
cards, which is why those two can be empty while conversations are not.
`/tags/{id}/test` alongside `/tags/{id}/alignment_examples` says a tag is a
judge with an alignment set behind it, not a saved string — so "why did this
tag fire" is a question about criteria and alignment, not about a filter.

The bridge back to your plane: an Insights drill-down resolves to
`conversation_id` and `trace_id`. Once the user gives you an ID from the page,
it is an ordinary Agents-plane lookup and everything above applies.

## Identifiers

`conversation_id`, OTel `trace_id`, OTel `span_id`, a Weave Call ID and a Calls
`thread_id` are distinct identifiers. They coincide only when the instrumented
app deliberately made them match.

## When Insights runs, and four ways it looks empty

Classification is a batch job, not part of ingest. It runs **daily, and only
once the project has logged 1,000 new turns since the last run** — so a project
under 1,000 turns has never had a run, and a quiet project can go days between
runs. The page reports when it last clustered. It also requires the org's **AI
features toggle** (org settings) to be on.

The page looks the same in all four cases, and only the first is visible from
the Agents plane:

1. **No spans at all** — check the Agents plane; this one you can confirm.
2. **Under the threshold** — fewer than 1,000 turns since the last run (or
   ever), so no run has happened yet. Turn counts are on the Agents plane
   (`is_invocation`), so you can tell how far off it is.
3. **No run inside the selected range** — the picker offers the last hour, 6
   hours, 24 hours, 7 days and 30 days; a short range on a project that
   clusters every few days is empty rather than zero.
4. **AI features off for the org** — classification is skipped, so this stays
   empty permanently while conversations keep arriving.

The Agents plane distinguishes the first two from the last two; it holds
nothing that separates the last two from each other.

## Related
- The Agents plane in depth: `skills/wandb-primary/references/WEAVE_SDK.md`
- Transcripts and pagination: `skills/wandb-primary/scripts/weave_agent_ops.py`
- What each page shows: `product-context/pages/agent-lens-overview.md` and siblings
