<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Weave Reference — Traces, Calls, Evals, Signals

## Bounded fetching — calls are heavy (full payloads; eval calls embed dataset rows)

1. Never materialize calls without `limit=N` AND `columns=[...]`. Iterating/`list()` on
   `get_calls()` streams full payloads per call; large evals can take many minutes.
2. Counting never requires fetching: `len(client.get_calls(...))` is a cheap server-side
   count (`calls_query_stats`). Prefer `client.server.calls_query_stats(...).count`.
3. Project only needed columns (`id`/`trace_id`/`op_name`/`started_at` always included;
   `inputs`/`output`/`summary` are the expensive ones). `unwrap()` to plain Python and
   aggregate locally in pandas; never dump raw call payloads into context.
4. For `Evaluation.predict_and_score` rows, use the safe helpers: project/root summaries
   first; counted narrow children only for status/timing; full payloads only for explicit,
   bounded samples after checking counts. The helper defaults to a 50-row full-payload cap;
   raise it only when the task genuinely requires row payloads and the count makes the
   request safe. Avoid raw PAS payload scans via `get_calls`, DataFrame, or parquet
   materialization.

## Init + get_calls

```python
client = weave.init("entity/project")  # positional; weave.init(project=...) -> TypeError
from weave.trace.weave_client import CallsFilter
op_ref = f"weave:///{client.entity}/{client.project}/op/Evaluation.evaluate:*"  # * = version wildcard
calls = client.get_calls(
    filter=CallsFilter(op_names=[op_ref]),          # dict filter also coerces fine
    sort_by=[{"field": "started_at", "direction": "desc"}],
    columns=["display_name", "summary"], limit=10,
    include_costs=True,                              # adds summary["weave"]["costs"]
)
children = client.get_calls(filter=CallsFilter(parent_ids=[call.id]))  # PLURAL, takes a list
df = client.get_calls(limit=500, columns=["op_name"]).to_pandas()
```

CallsFilter fields: `op_names`, `parent_ids`, `trace_ids`, `call_ids`, `trace_roots_only`,
`wb_run_ids` (also `input_refs`, `output_refs`, `thread_ids`, `turn_ids`, `wb_user_ids`).

## Instrumentation — how traces get created

Calls above don't exist until something is traced. Two paths, both gated on `weave.init(...)`:

- **OpenAI auto-tracing**: after `weave.init(...)`, supported OpenAI client calls are traced
  automatically (prompts, responses, token usage, latency, and errors logged); no decorator
  needed. (Equivalently `from weave.integrations.openai import OpenAI`.)
- **`@weave.op()`**: decorate app functions to capture the app-level call tree (inputs/output/
  exception/children) on top of the auto-traced LLM calls. This is what produces custom op names
  like `<app>.turn` or class `*.score` ops seen in queries above.

Both paths above write to the **calls** table (the Traces view). Neither populates the
**Agents → Conversations** view — the newer, recommended surface for agent traces — which reads a
separate GenAI-spans table. An agent traced only with `@weave.op` shows under Traces but stays
absent from Agents ("No agent activity … Check back once traces with agent metadata are
ingested"). That surface is fed by a third path:

- **Conversation SDK** (populates Agents → Conversations): `weave.start_conversation(agent_name=…,
  conversation_id=…)` → `conversation.start_turn(user_message=…, agent_name=…)`; inside a turn,
  `turn.start_tool(name=…, arguments=…, tool_call_id=…)`, `turn.start_llm(model=…)` (which takes
  `record(output_messages=…, usage=weave.Usage(…))`), and `turn.start_subagent(…)`. The turn
  carries its own messages through `turn.record(messages=…, output_messages=…)`: `messages=`
  populates the span's `input_messages`, `output_messages=` populates `output_messages`.
  Passing the final answer in `messages=` leaves `output_messages` empty. Use the
  `start_*` names: the shorter `turn.llm(…)` / `turn.tool(…)` / `turn.subagent(…)` aliases still
  work but are deprecated. Each turn emits an `invoke_agent` GenAI span
  (semantic-convention attrs `gen_ai.agent.name`, `gen_ai.conversation.id`); reusing one
  `conversation_id` groups turns into one conversation. `weave.init(...)` auto-wires the OTel
  exporter, so no extra setup is needed. Any agent can use it, and `@weave.op` / OpenAI
  auto-tracing can run alongside it to keep the Traces view populated — an agent wired for both
  appears in both views.

  **This path writes OTel spans only — it creates no Calls rows.** An agent instrumented with the
  Conversation SDK *alone* is fully readable through the Agents endpoints and returns **zero** from
  `get_calls()`. Calls-plane visibility requires separate, dual Calls instrumentation.

  **Custom attributes: `attributes=` on the starter, `set_attributes(…)` on the span.** The two
  are not interchangeable and the split is exact — on weave 0.53.12, `start_turn`, `start_llm`,
  `start_tool` and `start_subagent` do **not** accept `attributes=` (passing it is a `TypeError`).
  Stamp those from the returned object instead. `Conversation` is the mirror image: it has no
  `set_attributes`, so its starter is the only route.

  | Attach with | Applies to |
  | --- | --- |
  | `attributes={…}` argument | `start_conversation`, `start_session`, `log_turn`, `log_conversation`, `log_session` |
  | `.set_attributes({…})` on the returned object | `Turn`, `LLM`, `Tool`, `SubAgent` |

  ```python
  with weave.start_conversation(agent_name=…, attributes={"user_id": "u_42"}) as conv:
      turn = conv.start_turn(user_message=…)          # no attributes= here
      turn.set_attributes({"tenant": "acme"})         # -> Self, so it chains
  ```

  OTel spans accumulate attributes, so `set_attributes` may be called repeatedly while the span is
  open, including after the work it describes — so attributes whose values are only known once the
  work finishes (results, token counts, error classifications) can be stamped onto a turn that is
  still recording, rather than forcing everything to be computed up front.

## Counting + count semantics

```python
from weave.shared.trace_server_interface import CallsQueryStatsReq
n = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=f"{entity}/{project}",
    filter={"op_names": [op_ref]})).count  # filter/query both optional
```

- Counting convention, for EVERY count question: "traces"/"calls" = ALL calls
  (children included), even phrasings like "traces longer than 60s"; no trace_id
  dedupe unless "unique traces" is asked. Only "root"/"top-level"/"no parent" means
  `filter={"trace_roots_only": True}`.
- "error traces" = status `error` OR `descendant_error` OR non-null `exception`.
- "successful/non-error" = total minus errors; `summary.weave.status=="success"` excludes still-running calls (`running` is a real status), undercounting non-error.
- `Evaluation.evaluate` counts are op counts (`op_names` filter); add `trace_roots_only` only if asked.

## Query grammar (MongoDB subset)

Operators: `$eq` `$gt` `$gte` `$in` `$contains` `$and` `$or` `$not` `$convert`; operands
`{"$getField": "dot.path"}`, `{"$literal": value}`. There is NO `$lt`/`$lte` — use `$not [$gte]`.
`$contains` is a Weave extension absent from MongoDB:
`{"$contains": {"input": <op>, "substr": <op>, "case_insensitive": bool}}`.

```python
from weave.shared.interface.query import Query
error_query = Query(**{"$expr": {"$or": [
    {"$eq": [{"$getField": "summary.weave.status"}, {"$literal": "error"}]},
    {"$eq": [{"$getField": "summary.weave.status"}, {"$literal": "descendant_error"}]},
    {"$not": [{"$eq": [{"$getField": "exception"}, {"$literal": None}]}]},
]}})  # pass as query= to get_calls or CallsQueryStatsReq
```

## Call anatomy + data locations

```python
call.id; call.trace_id; call.parent_id          # parent_id None = root
call.op_name; call.func_name                    # full ref URI; short name "Evaluation.evaluate"
call.display_name; call.started_at; call.ended_at; call.exception
call.inputs; call.output                        # WeaveDict/WeaveObject -> unwrap()
call.summary["weave"]["status"]                 # "success"|"error"|"descendant_error"|"running"
call.summary["status_counts"]                   # descendant counts; keys may be TraceStatus enums OR strings.
#   May live at summary["status_counts"] OR summary["weave"]["status_counts"] depending on eval summary shape — read both.
call.summary["usage"][model]["input_tokens"]    # fallback keys: prompt_tokens/completion_tokens
call.summary["weave"]["costs"]                  # only when include_costs=True
```

WeaveObject access: `getattr(obj, "field", None)`, not `.get()`. ObjectRef compare: `str(ref) == "..."`.

## Eval analysis

```
Evaluation.evaluate (root)
  +-- Evaluation.predict_and_score   (one per dataset row x trials)
  |     +-- model.predict; <scorer>.score
  +-- Evaluation.summarize
```

- Per-row children: `predict_and_score` calls with `parent_ids=[eval_call.id]`.
- Pass/error rates: `summary["status_counts"]` — or `summary["weave"]["status_counts"]` if absent there (`success`/`error`/`descendant_error`).
- Token usage: `summary["usage"]`; sum input/output/total across model keys.
- Dataset ref: `inputs["self"].dataset`; repeated evals may reuse one ref — distinct-ref count != eval-call count.
- Scorer ops are BOTH `*_scorer` wrappers AND class `*.score` ops; versioned class scorers
  like `AxialCodingClassifierV1.score` lack the substring "scorer" — never grep for it.
- `weave.Model` versions by its **fields**. A Model with no fields hashes to one object
  version for every configuration, so swapping the model an evaluation runs against through
  an environment variable or a module constant changes the behaviour but not the Weave
  identity — every run points at the same model ref. Carrying the identity as a field
  (`model_name: str`) gives each checkpoint its own version. A leaderboard ranks model
  object versions, so evaluations that share a ref collapse into one row, and the ref a
  completed evaluation used cannot be reassigned afterwards.
- A leaderboard column names its metric with `summary_metric_path` into the evaluation
  summary. A path the summary does not contain renders as an empty column rather than
  raising, so a board publishes successfully and is still entirely null. Read the summary
  keys first and build the paths from them. That summary is the `Evaluation.evaluate` call's
  **output**; the call's own `summary` field holds only `status_counts` and `weave`.

## Datasets

`weave.Dataset(name=…, rows=[…])` — both keyword-only; `rows` is required. Published
with `weave.publish(ds)`, which returns the `ObjectRef`. In the UI a Dataset is an
**Asset**, listed under the project's **Datasets** section (`.../weave/datasets`).

- Datasets are **content-addressed**. Republishing identical rows under the same name
  returns the same digest and creates no new version; changing a row creates one.
  A version count that did not move means the rows did not change.
- `add_rows(rows)` only works on a **published** dataset. On a local one it raises
  `TypeError: This dataset is not saved to weave. Call weave.publish(dataset) to save
  the dataset before adding rows.` It returns a new Dataset rather than mutating.
- **Object names are unique per project across types.** Publishing an `Evaluation`
  under a name already held by a `Dataset` fails with `400` from `/obj/create`:
  `Cannot publish '<name>' as a Evaluation: that name is already used by a Dataset in
  this project. Object versions cannot share types, publish this object under a
  different name.` The first object publishes and the second does not, so a script
  that names its dataset and its evaluation alike leaves a dataset behind and no
  evaluation.
- Constructors and helpers: `Dataset.from_calls(calls)`, `from_pandas(df)`, `from_hf`,
  and on an instance `to_pandas()`, `select(indices)`, `num_rows`, `columns_names`.

## Playground

The Playground (`/{entity}/{project}/weave/playground`) writes Calls into the project
when its call-tracking toggle is on, so a tracked session is queryable as ordinary
Calls.

## Feedback + misc

Feedback: `client.server.feedback_query(FeedbackQueryReq(project_id=pid, fields=["id"],
limit=1000, offset=o))` (from `weave.shared.trace_server_interface`); rows live in
`res.result` — read it defensively as `getattr(res, "result", None) or getattr(res, "feedback", None)
or getattr(res, "rows", None) or []` so the count survives a response-shape change. Page until a
batch is shorter than `limit`. Quiet logs: `logging.getLogger("weave").setLevel(logging.ERROR)`.

## Signals and Monitors

**Signals** is the UI concept at `/{entity}/{project}/weave/agents/signals`.
Each Signal created there is stored as a versioned Weave object whose
`base_object_class` is `"Monitor"`. A project can also contain older Monitor or
ClassifierMonitor objects that are valid and active but do not appear in the
Signals UI.

Classify an object as a Signal when its leaf class is `Monitor` and its
`op_names` is exactly `["weave.genai.turn_ended"]`. Do not infer Signal identity
from the object ID, display name, or scorer name. ClassifierMonitor objects and
Monitor objects watching other operations are legacy monitors for helper
safety, even if their names look like Signals.

Relevant serialized fields:

- `active`: whether the Monitor runs.
- `sampling_rate`: fraction of matching events to score.
- `op_names`: watched ops. This can be `None` on legacy objects and can contain deserialized `OpRef` values.
- `query`: optional filter over eligible events.
- `scorers`: scorer refs used by the Monitor.
- `description`: user-facing description.

Use the helper for inventory and activation. It prints compact object and
scorer identifiers, not whole Monitor or scorer prompts:

```bash
M=skills/wandb-primary/scripts/weave_monitor_ops.py
python $M list --path ENTITY/PROJECT --include-scorers
python $M set-active --path ENTITY/PROJECT --object-id ID --active false --allow-signals
python $M deactivate-legacy --path ENTITY/PROJECT --object-id ID --dry-run
```

The common path is one helper call for inventory and one helper call for a
bounded, verified state change. Do not replace it with SDK introspection,
`weave.ref(...).get()`, unbounded object reads, or prompt dumps.

For custom bounded inventory code, query raw serialized objects with stable
pagination. Scope every request with `client.project_id`. Never use
`client._project_id`:

```python
from weave.shared.trace_server_interface import (
    ObjectVersionFilter,
    ObjQueryReq,
    SortBy,
)

response = client.server.objs_query(ObjQueryReq(
    project_id=client.project_id,
    filter=ObjectVersionFilter(
        base_object_classes=["Monitor"],
        latest_only=True,
    ),
    limit=100,
    offset=offset,
    sort_by=[SortBy(field="object_id", direction="asc")],
))
```

When creating a Signal Monitor, set its serialized type metadata and watched
operation together:

```json
{
  "_bases": ["Monitor", "Object", "BaseModel"],
  "_class_name": "Monitor",
  "_type": "Monitor",
  "op_names": ["weave.genai.turn_ended"],
  "scorers": ["weave-trace-internal:///PROJECT_INTERNAL_ID/object/SCORER_ID:DIGEST"]
}
```

Scorer and model refs must be fully qualified
`weave-trace-internal:///PROJECT_INTERNAL_ID/object/OBJECT_ID:DIGEST` refs. An
`LLMAsAJudgeScorer` uses `_class_name` and `_type` set to
`"LLMAsAJudgeScorer"`, `_bases` set to
`["LLMAsAJudgeScorer", "Scorer", "Object", "BaseModel"]`, and `model` set to
the model's fully qualified internal ref. A ClassifierMonitor uses
`_class_name` and `_type` set to `"ClassifierMonitor"`, with `_bases` set to
`["Monitor", "Object", "BaseModel"]`.

Store a Monitor filter in the query envelope, for example
`{"$expr": {"$eq": [{"$literal": "error"}]}}`. For an agent-turn LLM judge,
the scoring prompt should include the applicable agent context variables:
`{agent_name}`, `{agent_description}`, `{conversation_name}`,
`{system_instructions}`, `{input_messages}`, `{output_messages}`,
`{status_code}`, and `{status_message}`. These read the `weave.genai.turn_ended` span, so a
turn whose `output_messages` is empty gives the judge no response to score and produces no
Feedback. A tag prompt must require JSON shaped
as `{"scores": [{"value": "<tag_id>", "confidence": 0.0, "reason":
"evidence"}]}`, constrain `value` to a configured tag ID and `confidence` to
`0.0` through `1.0`, require a reason grounded in the trace, use
`{"scores": []}` when no tag applies, and forbid Markdown or extra commentary.

Monitor output is stored as Feedback on the scored call or agent span. The
Feedback `runnable_ref` identifies the scorer, while `trigger_ref` identifies
the Monitor version that scheduled the scoring. Use `trigger_ref` to relate a
result to its Monitor. A successful Monitor publish is not proof of activation;
the helper creates a fresh client, reads the latest object from the server, and
checks the exact `active` value.

## Agents plane — conversations, agent traces, agent spans

The `weave/agents/...` app URLs (where the conversations of any agent instrumented
with the Conversation SDK live, see Instrumentation above) are a separate read
surface from calls. **To read a conversation or one agent trace,
run the helper — it paginates and prints a compact, summary-ready transcript:**

```bash
W=skills/wandb-primary/scripts/weave_agent_ops.py
python $W conversation <conversation_id>   # id from a .../weave/agents/conversations/<id> URL
python $W trace <trace_id>                 # one agent turn
# --path ENTITY/PROJECT (default $WANDB_ENTITY/$WANDB_PROJECT); --feedback; --full disables clipping; --help
```

That answers the common "summarize this agent conversation" turn in one call —
don't hand-roll the fetch for a *summary*.

### These five identifiers are distinct — never substitute one for another

`conversation_id`, OTel `trace_id`, OTel `span_id`, Weave **Call ID**, and Calls
`thread_id` are separate namespaces. Do not assume any one maps to another, or
derive one from another, unless you have *directly verified* it for the specific
agent you are looking at. In particular:

- A `conversation_id` is **not** a `thread_id`, a `trace_id`, or a `call_id`.
- A Conversation SDK span's OTel `trace_id` / `span_id` does **not** map to a
  Weave Call ID.

**`get_calls()` returning 0 for Conversation SDK telemetry is a correct result,
not an ingestion failure.** When the Agents endpoints contain the spans, the data
arrived fine; it simply lives only in the Agents plane. Do not escalate, retry, or
report a logging bug on that basis. Reaching the Calls plane requires the agent to
*also* be instrumented for Calls (see Instrumentation, above).

### Reading Conversation SDK telemetry

Two supported routes, both in the Agents plane:

- **Transcripts** — `weave_agent_ops.py` (above): whole conversation, or one turn.
- **Span-level inspection** — `/agents/spans/query`, for per-span attributes,
  timings, tokens, costs, and errors that the transcript doesn't surface. On weave
  0.53.12 this is a bound method; no `_generic_request` plumbing needed:

```python
import weave
from weave.shared.agents import types as at
client = weave.init("entity/project")
res = client.server.agent_spans_query(at.AgentSpansQueryReq(
    project_id=f"{client.entity}/{client.project}", limit=50,
))
```

Filter by conversation with the `conversation_id` **span field** (see the filter
grammar below) — that is the supported way to go from a conversation to its spans.

> **Dual-instrumented agents are the one exception, and you must verify it.** An
> agent that writes *both* planes may deliberately reuse a single id — passing it
> as `conversation_id` to `start_conversation` and as the Calls `thread_id` — in
> which case `get_calls(filter={"thread_ids": [conversation_id]})` does return that
> conversation's Calls. When it holds, this is the route to per-call inputs,
> outputs and exceptions that the transcript doesn't expose.
>
> That correspondence is a property of *that agent's own instrumentation*, never of
> the Conversation SDK, and it is not the default. Confirm it in the agent's source
> before relying on it; if you cannot, treat the ids as unrelated and read the
> Agents plane. A zero result means the agent isn't dual-instrumented, not that the
> query is malformed.

**Presenting a conversation summary — interpret, don't just recap.** The
transcript is raw material; a flat list of what happened is dull and low-value.
Lead with a short summary paragraph (what the user wanted, what the agent
delivered, the outcome), then add compact sections as the ask warrants:

- **Highlights** — what the agent did well: correct tool/data use, good error
  recovery, a sharp or well-presented final answer.
- **Opportunities** — friction and waste, grounded in the trace: errors and
  retries, redundant or dead-end tool calls, slow steps and token cost (the
  helper prints per-turn `dur`, message counts, and a `tokens=` line), and
  places the agent over- or under-scoped the answer.
- **User experience** — an estimate of how well the user was served: did they
  get what they asked for, did they have to repeat themselves or push back, did
  the agent stall or ask when it could have acted? Feedback rows (`--feedback`)
  and the user's own follow-up turns are the strongest signals; say "no explicit
  signal" rather than inventing sentiment.

Scale to the request — a "quick/brief" ask gets one tight line per section, a
"deep analysis" gets the full treatment. Ground every claim in the transcript;
never infer a problem or a satisfaction level the messages don't show.

### Going beyond the helper (spans, search, custom queries)

On weave 0.53.12 these endpoints **are** bound methods on `client.server` — call
them directly (Req/Res models live in `weave.shared.agents.types`):
`agent_spans_query`, `agent_spans_stats`, `agent_agents_query`,
`agent_versions_query`, `agent_search`, `agent_custom_attrs_schema`,
`agent_conversation_chat`, `agent_traces_chat`. The one exception is
`agent_conversation_spans`, still an unimplemented `...` stub returning `None`.

For a stub, or an endpoint with no method, fall back to the remote binding's
`_generic_request(url, req, ReqType, ResType)`:

```python
import weave
from weave.shared.agents import types as at
client = weave.init("entity/project")
server = getattr(client.server, "_next_trace_server", client.server)  # RemoteHTTPTraceServer under the cache
pid = f"{client.entity}/{client.project}"
res = server._generic_request("/agents/spans/query",
    at.AgentSpansQueryReq(project_id=pid, limit=50), at.AgentSpansQueryReq, at.AgentSpansQueryRes)
```

Endpoints: `/agents/spans/query` (raw rows or grouped aggregates),
`/agents/spans/stats` (chart-ready aggregations), `/agents/query` +
`/agents/agent-versions/query` (materialized agents), `/agents/search` (message
text search), plus `/agents/conversations/chat` + `/agents/traces/chat` (what the
helper wraps; conversation `limit` maxes at 50 — page with `offset=`). Full
list/shapes: `server.get("/openapi.json").json()["paths"]`.

A chat message from the two `chat` endpoints (`AgentChatMessage`) is a flat
envelope: `m.type` is the discriminator and the same-named attr holds the
payload — `getattr(m, m.type)` → `.text` (`user_message`/`assistant_message`) or
`.tool_name`/`.tool_arguments`/`.tool_result` (`tool_call`). The helper already
formats these; you only need this if you post-process the raw response.

### Driving the Conversations list UI (filter, window, sort, columns)

The **Agents → Conversations list** page (`.../weave/agents/conversations`)
encodes its entire view state in **URL query params**. You can't click the UI,
but you *can* build a deep link that reproduces any view and hand it to the user,
and read the same data through the agents API. This grammar isn't in the public
docs (the docs MCP tools can't verify it) — it's documented here from the live app:

- **Filter:** `filters[<source>:<key>]=<value>`, where `<source>` is
  `custom_attrs_string` | `custom_attrs_int` | `custom_attrs_float` |
  `custom_attrs_bool` (or a built-in span field). Example:
  `filters[custom_attrs_string:user_id]=u_42`. The UI's
  filter menu exposes Agent, Agent Version, Provider, Model, Error type, Tool,
  Tool type, Status, Span kind, Operation Name, Conversation ID, and any custom
  attribute. A filter can be **multi-select** (OR over values):
  `filters[<field>][]=chat&filters[<field>][]=invoke_agent`. The default match is
  equality; **override the operator** with a companion `filterOps[<source>:<key>]=<op>`
  param — e.g. not-equal:
  `filters[custom_attrs_string:weave.sys_version]=1&filterOps[custom_attrs_string:weave.sys_version]=ne`.
  The operator vocabulary is the Weave Query `$expr` set (`$eq $gt $gte $lt $lte
  $in $contains` — `$contains` is a case-insensitive substring — plus logical
  `$and $or $not`; the `filterOps` value is the short code, e.g. `ne`). Discover
  the exact set from the OpenAPI schema:
  `server.get("/openapi.json").json()["components"]["schemas"]["Query"]["properties"]["$expr"]`.
- **Numeric-range filter (separate param):** numeric columns — cost, invocation
  / span counts, token counts — are **not** in the categorical filter menu; they
  filter via a range control: `numericRanges[<field>][min]=<n>&numericRanges[<field>][max]=<n>`.
  Both `min` and `max` are required (the UI rejects an empty `max`). Example:
  `numericRanges[invocation_count][min]=2&numericRanges[invocation_count][max]=106`.
  So cost, tokens, and counts filter through `numericRanges[<field>]`, not
  `filters[…]` — hence "cost" isn't in the categorical filter menu.
- **Time window:** `timeWindow[unit]=day|hour|minute&timeWindow[quantity]=N&timeWindow[startMs]=<epoch_ms>`.
  UI presets: `1m 1h 6h 24h 7d 30d` (e.g. 7d → `unit=day&quantity=7`; 6h →
  `unit=hour&quantity=6`).
- **Sort:** `sort[field]=<column>&sort[sort]=asc|desc` (e.g.
  `sort[field]=total_cost_usd&sort[sort]=asc`).
- **Columns:** `cols[show][]=<id>` and `cols[hide][]=<id>` (e.g.
  `cols[show][]=agent_versions`, `cols[hide][]=total_input_tokens`). ~196 columns
  exist; ~11 show by default (Conversation, Spans, Signals, Agent, Invocations,
  In/Out Tokens, Cost, Started, Last Activity).
- **Open one conversation / turn:** `.../conversations/<conversation_id>` opens
  the conversation; add `&convTurn=<n>&chatTrace=<trace_id>` to deep-link one
  turn — `convTurn` is the **1-based turn number** and `chatTrace` is that
  **turn's agent trace id**, as reported by `/agents/spans/query` or the helper's
  per-turn output. It is an Agents-plane id; do not source it from a Call. Any
  `filters`/`timeWindow`/`sort` params carry through, so you can build a link
  straight to a specific turn of a filtered view.

**Query the schema; don't guess the field/column names.** The filterable fields
and their types come from `/agents/spans/query` (via `_generic_request`, above),
not the docs: there are ~79 **built-in** span fields (`agent_name`,
`agent_version`, `conversation_id`, `input_tokens`/`output_tokens`,
`input_cost_usd` and the other `*_cost_usd`, `error_type`, `operation_name`,
`provider_name`, `status`, …) plus the typed **custom-attribute Map columns**
(`custom_attrs_{string,int,float,bool}`). The query also returns a `count`
(+ `has_more`) to answer "how many conversations match this filter"; for keyword
lookups across message text, use `/agents/search` (e.g. the most recent
conversation mentioning "api key").

**Custom attributes vary per project — they are whatever that app traced.** The
built-in fields are always present, but the custom-attribute columns/filters are
not a fixed set: each is an attribute an app attached to its spans, so a
different project exposes a different set. Always query the schema for the
project you're looking at rather than assuming particular keys exist.

**How tracing produces those fields.** Everything filterable/selectable in the
view is derived from the traced spans — nothing is configured in the UI. Custom
attributes reach spans two ways: general Weave wraps traced calls in
`with weave.attributes({"key": value}): …` (attributes attach to every call made
inside); the Conversation SDK (see Instrumentation, above) attaches them per span
as described there. The
**column type is inferred from the value** at trace time — a string lands in
`custom_attrs_string`, an int in `custom_attrs_int`, etc. — which is how the UI
knows a key's type and which operators apply. So a key is filterable **only if it
was traced**, and what an app attaches (and its Python type) determines what the
view can filter, group, and show — including `weave.integration.*`, which the
Agents backend uses to group/filter conversations by integration.

## Analysis facts

- Cross-metric contradictions (M1 vs M2 for the same group) are invisible in run-level summaries — download artifact table rows and compare per group.
- Per-model/per-agent success rates are rarely in W&B run summaries; they live in Weave traces — group by model x task type.

## Prompt objects (weave.StringPrompt / weave.MessagesPrompt)

```python
prompt = weave.StringPrompt("You are ... {style}")        # or MessagesPrompt([{...}])
weave.publish(prompt, name="my_prompt")                    # new version under that name
prompt = weave.ref("weave:///ENTITY/PROJECT/object/my_prompt:latest").get()
prompt.format(style="terse")                               # StringPrompt -> str; MessagesPrompt -> list
```
- `format(**kwargs)` renders the whole prompt on both classes. `MessagesPrompt` also has
  `format_message(message, **kwargs)`, which renders a single message dict. There is no
  `format_messages`.
- "Update a prompt" = publish again under the same name → new version; old versions stay.
- A published Prompt version links straight into Registry, without an Artifact in between:
  `weave.link_prompt_to_registry(prompt, target_path=f"{registry.full_name}/<collection>",
  aliases=["candidate"])`. The project component is `Registry.full_name`, not `Registry.name`.
  Collection layout, alias conventions, and the type allowlist: `$refs/ARTIFACTS_AND_REGISTRY.md`.
- List a project's prompt objects:
```python
from weave.shared.trace_server_interface import ObjectVersionFilter
objs = client._objects(filter=ObjectVersionFilter(base_object_classes=["Prompt"]))
```
