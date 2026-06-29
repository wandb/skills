<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# WB Agent API Reference

Source docs: https://wb-agent.wandb.ai/api/docs

The Swagger UI loads the OpenAPI schema from `https://wb-agent.wandb.ai/openapi.json`.

## Contents

- [Service Model](#service-model)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Create Turn](#create-turn)
- [Turn Response Shape](#turn-response-shape)
- [Query Turns](#query-turns)
- [Waiting and Waking](#waiting-and-waking)

## Service Model

The WB Agent API is an asynchronous agent execution service for Weights & Biases. Submit prompts as turns, poll lifecycle state, and read the answer from the completed turn's messages.

Default base URL:

```text
https://wb-agent.wandb.ai
```

Lifecycle states:

```text
queued
in_progress
completed
errored
cancelled
```

Terminal states are `completed`, `errored`, and `cancelled`.

## Authentication

Most endpoints use HTTP Basic auth. The bundled helper sends `Authorization: Basic ...` for every authenticated endpoint.

Recommended environment variables:

```bash
export WB_AGENT_USERNAME="api"
export WB_AGENT_PASSWORD="$WANDB_API_KEY"
export WANDB_ENTITY="my-team"
export WANDB_PROJECT="my-project"
```

The helper also accepts `WB_AGENT_API_KEY` or `WANDB_API_KEY` as the password. It accepts `WB_AGENT_ENTITY` / `WB_AGENT_PROJECT` or the standard `WANDB_ENTITY` / `WANDB_PROJECT` defaults.

The helper tries `certifi` for TLS verification when it is installed. Use `--insecure` only to diagnose local certificate store problems.

## Endpoints

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| `GET` | `/api/v1/health` | Check service health | No |
| `GET` | `/api/v1/is-available?entity=...` | Check whether WB Agent is enabled for an entity | Yes |
| `GET` | `/api/v1/agent-aliases` | List agent variants requestable via `agent_config_override` | Yes |
| `POST` | `/api/v1/turns` | Create a turn | Yes |
| `GET` | `/api/v1/turns` | Query turns with filters | Yes |
| `GET` | `/api/v1/turns/{turn_id}` | Fetch a turn with messages, tool calls, and state | Yes |
| `PATCH` | `/api/v1/turns/{call_id}` | Update editable turn fields, currently title | Yes |
| `POST` | `/api/v1/turns/{leaf_turn_id}/generate-and-set-thread-title` | Generate a thread title from a leaf turn and store it on the root | Yes |
| `POST` | `/api/v1/turns/{turn_id}/cancel` | Cancel queued or in-progress turn | Yes |
| `POST` | `/api/v1/turns/{turn_id}/archive` | Archive a turn and descendants | Yes |
| `POST` | `/api/v1/turns/{turn_id}/feedback` | Set or clear feedback | Yes |
| `GET` | `/api/v1/turns/{turn_id}/events` | Stream real-time events for a turn (SSE; **beta**, not used by this skill) | Yes |

`GET /api/v1/agent-aliases` returns `{"aliases": [...]}`. Call it to discover the values accepted by
`agent_config_override` on create.

## Create Turn

`POST /api/v1/turns`

Request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `user_prompt` | string or prompt-part array | Required. Non-empty. |
| `entity` | string or null | Optional W&B entity. Root turns can resolve a default if omitted. |
| `project` | string or null | Optional W&B project. Root turns can resolve a default if omitted. |
| `title` | string or null | Optional, max 255 chars. |
| `parent_turn_id` | string or null | Set for conversation continuation. |
| `permissions.allow_all_network_egress` | bool or null | Null inherits. Defaults to false for root turns. |
| `permissions.allowed_network_domains` | string array or null | Domains added to the sandbox outbound allowlist. Null inherits; a list replaces. This is how you grant a `network_access` permission request (see Turn Response Shape) with least privilege. |
| `agent_config_override` | string or null | Request a named agent variant (values from `GET /api/v1/agent-aliases`). Null uses the service default and is **not** inherited from the parent turn — pass it again to stay on a variant. Unknown values are rejected with HTTP 422. |

Plain text example:

```bash
curl -u "$WB_AGENT_USERNAME:$WB_AGENT_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"entity":"my-team","project":"evals","user_prompt":"Find the biggest regression in yesterday runs."}' \
  https://wb-agent.wandb.ai/api/v1/turns
```

Structured prompt part example:

```json
{
  "entity": "my-team",
  "project": "evals",
  "user_prompt": [
    {"type": "text", "text": "Explain this run's failed evals."},
    {"type": "wandb_run", "entity_name": "my-team", "project_name": "evals", "run_name": "abc123"}
  ]
}
```

Prompt part types:

| Type | Required fields |
| --- | --- |
| `text` | `text` |
| `image` | `media_type`, `base64_data` |
| `wandb_entity` | `entity_name` |
| `wandb_project` | `entity_name`, `project_name` |
| `wandb_run` | `entity_name`, `project_name`, `run_name` |
| `wandb_call` | `entity_name`, `project_name`, `call_id` |
| `wandb_web_page_state` | `url` |

`image.media_type` is an enum: `image/jpeg`, `image/png`, `image/gif`, `image/webp`.

`wandb_web_page_state` also accepts optional `entity_name`, `project_name`, and `content`
(a structured workspace/report/sweep page payload) beyond the required `url`.

The schema also accepts unknown prompt part objects with a required `type` string for forward
compatibility. Prefer the documented W&B-specific part types above unless the current OpenAPI
schema says otherwise.

## Turn Response Shape

Important fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique turn ID. |
| `parent_turn_id` | Parent turn ID or null. |
| `root_turn_id` | Root conversation/session ID. |
| `title` | Turn/thread title. |
| `state` | Lifecycle state. |
| `created_at`, `updated_at` | Service timestamps. |
| `expires_at` | When an unstarted turn expires (turns are short-lived; observed ~10 min TTL). |
| `messages` | Agent messages produced during execution (see "Reading the answer"). |
| `tool_calls` | Recorded tool calls and tool responses. |
| `agent_questions` | List of clarifying questions for the caller to answer, or null. See below. |
| `permission_requests` | List of capability requests (e.g. network access) for the caller to grant, or null. See below. |
| `permissions` | Resolved permission state (`allow_all_network_egress`, `allowed_network_domains`). |
| `error_info` | Error string when the turn errors. |
| `feedback_is_positive`, `feedback_reasoning` | User feedback metadata. |
| `user_context`, `thread_last_turn_created_at`, `updated_messages` | Additional metadata. |

> There is **no `final_output` field**. The final answer is the latest assistant message:
> scan `messages` in reverse for the record with `role == "assistant"` and read its `content`
> (a string, or `message.content` as a list of `{type, text}` parts). The helper's
> `--format text` does this for you. Watch the message `role`: a turn also contains `user`,
> `reasoning` (content may be null), and `tool` records.

### Agent questions (`agent_questions`)

Each entry is a `MultipleChoiceQuestion`:

| Field | Meaning |
| --- | --- |
| `type` | `"multiple_choice"` |
| `question` | The question text. |
| `options` | List of suggested answers (the caller may also answer free-form). |

A turn can reach a terminal state while still holding unanswered questions — that means the
agent is waiting on **you**. Answer by creating a continuation turn (`parent_turn_id`) whose
prompt states the chosen option. Questions are identified by position in the list.

### Permission requests (`permission_requests`)

Each entry has a `type` discriminator. The current kind is `NetworkAccessRequest`:

| Field | Meaning |
| --- | --- |
| `type` | `"network_access"` |
| `reason` | Why the agent needs access (show this to the user). |
| `domains` | Hostnames the agent wants reachable. |

Grant by creating a continuation turn with `permissions.allowed_network_domains` set to the
requested domains (least privilege), or decline by continuing without them. New permission
kinds become additional `type` values in this same list.

## Feedback

`POST /api/v1/turns/{turn_id}/feedback`

Request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `feedback_is_positive` | bool or null | `true` = positive, `false` = negative, `null` = clear. |
| `feedback_reasoning` | string or null | Optional explanation. |
| `share_turn_data_with_wandb` | bool | Defaults to `false`; explicit per-turn opt-in to share this turn's data with W&B when sharing is otherwise disabled. |

## Query Turns

`GET /api/v1/turns`

Filters:

```text
entity
project
parent_turn_id
root_turn_id
state
is_root
sort_by=created_at|updated_at|id
sort_desc=true|false
limit=1..100
offset>=0
```

Useful query:

```bash
python3 scripts/aria.py query --entity my-team --project evals --sort-by updated_at --sort-desc --limit 10
```

## Waiting and Waking

Poll `GET /api/v1/turns/{turn_id}` to follow a turn. (An SSE event stream at
`/api/v1/turns/{turn_id}/events` exists but is **beta**; this skill does not use it.)

```bash
python3 scripts/aria.py wake <turn-id> --since-updated-at "<timestamp>" --poll-interval 2
python3 scripts/aria.py wait <turn-id> --until terminal --poll-interval 2
```

`wake` returns on the first observed update (including a new assistant message, question, or
permission request). `wait --until terminal` returns only after `completed`, `errored`, or
`cancelled`. `wait --until response` returns as soon as assistant text is present (or terminal).
