---
name: aria-chat
description: Delegate W&B-specific analysis, debugging, and data questions to the hosted WB Agent over its HTTP API. Use when you need to ask the Weights & Biases agent to inspect W&B entities, projects, runs, Weave calls, page state, or experiment data; continue a WB Agent conversation; poll or wait for async turn completion; or wake when a WB Agent turn updates.
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Aria Chat

## Overview

Use the WB Agent as a specialist for Weights & Biases data analysis. The service is asynchronous: create a turn, keep the turn ID, then poll until the turn completes or wakes with an update.

Prefer the bundled helper script over handwritten `curl` unless the user explicitly asks for raw HTTP.

## Setup

The base URL is `https://wb-agent.wandb.ai`. Override it with `WB_AGENT_BASE_URL` for staging or local testing.

Authenticated endpoints use HTTP Basic auth. Set explicit credentials when available:

```bash
export WB_AGENT_USERNAME="api"
export WB_AGENT_PASSWORD="$WANDB_API_KEY"
```

The helper also accepts `WB_AGENT_API_KEY` or `WANDB_API_KEY` as the password and defaults the username to `api`. If auth fails, ask the user for the correct WB Agent credentials rather than retrying blindly.

Use `WANDB_ENTITY` and `WANDB_PROJECT` as default scope when the user is repeatedly working in one project. The helper also accepts `WB_AGENT_ENTITY` and `WB_AGENT_PROJECT`.
Set entity and project together when you want a default scoped target; a project-only environment default is ignored for unscoped creates so unrelated W&B shell context does not block root turns or continuations.

If Python cannot find a local CA bundle but `curl` works, install/use `certifi` or pass `--insecure` only for local debugging.

The bundled helper adds agent-facing conveniences such as `wait`, `wake`, text extraction, surfacing of `agent_questions`/`permission_requests`, agent-variant selection (`aliases`, `--agent-config-override`), scoped network grants, stdin prompts, and prompt-part JSON files.

### Client attribution

Every turn the helper creates is tagged with a client identifier so the service can attribute traffic that originates from a coding agent. The tag defaults to `coding_agent` and is carried two ways: a `{"type": "client_info", "client": "<tag>"}` prompt-part prepended to `user_prompt` (which the service stores verbatim on the turn) and `X-Wandb-Client` / `User-Agent` request headers. The agent treats the marker as inert and answers normally.

Override the tag with `--client <name>` on `create` or the `WB_AGENT_CLIENT` environment variable; pass an empty value (`--client ""`) to disable tagging entirely. There is no caller-settable `user_context` or top-level `client` field on the create endpoint — the service drops both — so the prompt-part is the only attribution channel that persists on the turn record.

## Core Workflow

1. Check access when an entity is known:

```bash
python3 /path/to/aria-chat/scripts/aria.py available --entity <entity>
```

2. Create a turn with a focused delegation prompt:

```bash
python3 /path/to/aria-chat/scripts/aria.py create \
  --entity <entity> \
  --project <project> \
  --title "Analyze failed eval runs" \
  "Compare the last 20 failed eval runs and identify the dominant failure modes."
```

3. Save the returned `id`, `root_turn_id`, `state`, and `updated_at` in your working notes.

4. Wait for a final answer when the WB Agent result is on the critical path:

```bash
python3 /path/to/aria-chat/scripts/aria.py wait <turn-id> --until terminal --format text
```

5. Continue the conversation by creating a child turn:

```bash
python3 /path/to/aria-chat/scripts/aria.py create \
  --parent-turn-id <turn-id> \
  "Now narrow that down to runs created after 2026-05-01."
```

## Wake Pattern

Use polling to follow a turn. (An SSE event stream at `/api/v1/turns/{turn_id}/events` exists but is
beta — don't rely on it from this skill.)

When you want to delegate and keep working locally, create the turn, then start a wake command that
returns on the next update or terminal state:

```bash
python3 /path/to/aria-chat/scripts/aria.py wake <turn-id> \
  --since-updated-at "<updated_at-from-create>" \
  --poll-interval 2 \
  --format summary
```

Run this as a background command/session if your environment supports it. `wake` polls every two
seconds by default and exits as soon as `updated_at`, state, message count, assistant text, or a new
question/permission request appears. After it returns, fetch the turn or continue with
`wait --until terminal`.

Use `wake` for "tell me when something changed" and `wait --until terminal` for "block until the
answer is done." Use `wait --until response` to return as soon as assistant text exists.

## Selecting an Agent Variant

The service runs named agent variants. List them and pin one for a turn when you need a specific
build (otherwise the service default is used):

```bash
python3 /path/to/aria-chat/scripts/aria.py aliases          # lists available variants
python3 /path/to/aria-chat/scripts/aria.py create --agent-config-override <alias> "..."
```

`agent_config_override` is **not** inherited by continuation turns — pass it again to stay on a
variant. Unknown values are rejected with HTTP 422.

## Responding to the Agent

A turn can reach a terminal state while still waiting on **you**. After a turn finishes, check for:

- **`agent_questions`** — the agent asked a clarifying multiple-choice question. The `--format text`
  output prints these as `AGENT QUESTION [i]`. Answer by creating a continuation turn whose prompt
  states the chosen option:

  ```bash
  python3 .../aria.py create --parent-turn-id <turn-id> "Use option: nightly-evals"
  ```

- **`permission_requests`** — the agent needs network access it does not have. `--format text` prints
  `NETWORK ACCESS REQUESTED: <reason> -> [domains]`. Grant the minimum scope by continuing with the
  requested domains (preferred over allow-all):

  ```bash
  python3 .../aria.py create --parent-turn-id <turn-id> \
    --allowed-network-domain pypi.org --allowed-network-domain files.pythonhosted.org \
    "Granted. Continue."
  ```

If neither is present and assistant text is available, the turn is done.

## Prompting Guidance

Ask the WB Agent for W&B-native work, not generic coding work. Good delegation prompts include:

- "Inspect entity/project X/Y and summarize the metric regressions in runs tagged `nightly`."
- "Analyze this Weave call and identify where latency increased: entity/project/call_id."
- "Compare these run IDs and explain which config values correlate with failed evals."

Include concrete W&B identifiers whenever possible: entity, project, run name, Weave call ID, report/page URL, tags, metric names, and time windows. State the output shape you need, such as a ranked list, short diagnosis, or suggested next experiments.

## Structured Prompt Parts

Use plain text for most requests. Use `--prompt-parts <json-file>` when passing W&B objects explicitly. Supported part types include:

- `text`
- `wandb_entity`
- `wandb_project`
- `wandb_run`
- `wandb_call`
- `wandb_web_page_state`
- `image`

Example:

```json
[
  {"type": "text", "text": "Explain the failure pattern in this run."},
  {"type": "wandb_run", "entity_name": "my-team", "project_name": "evals", "run_name": "abc123"}
]
```

## Safety

Do not send secrets, API keys, private customer data, or broad credentials in prompts. Pass only the minimum W&B identifiers needed for the analysis.

For network access, prefer least privilege: when the agent raises a `network_access` permission request, grant only the requested hosts with `--allowed-network-domain <domain>` (repeatable) on the continuation turn. Use `--allow-all-network-egress` only when the analysis explicitly requires unrestricted network access.

## References

Read `references/api.md` when you need endpoint details, request bodies, lifecycle states, or raw `curl` equivalents. Fetch the live schema from `https://wb-agent.wandb.ai/openapi.json` if the hosted API behavior appears to differ from the bundled reference.
