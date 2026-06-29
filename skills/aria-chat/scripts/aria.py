#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills
"""Small CLI for the hosted WB Agent HTTP API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "https://wb-agent.wandb.ai"
TERMINAL_STATES = {"completed", "errored", "cancelled"}

# Client attribution. The hosted API has no caller-settable client field
# (`user_context` and unknown top-level keys are dropped server-side), so the
# tag is carried two ways: a `client_info` prompt-part that persists on the
# stored turn, and request headers for backends that read request metadata.
DEFAULT_CLIENT = "coding_agent"
CLIENT_PART_TYPE = "client_info"


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def auth_header(required: bool = True) -> str | None:
    username = env("WB_AGENT_USERNAME") or env("WANDB_USERNAME")
    password = env("WB_AGENT_PASSWORD") or env("WB_AGENT_API_KEY") or env("WANDB_API_KEY")
    if not username and password:
        username = "api"
    if not username or not password:
        if required:
            raise ApiError(
                None,
                "Missing WB Agent credentials. Set WB_AGENT_USERNAME/WB_AGENT_PASSWORD "
                "or set WANDB_API_KEY and use the default username 'api'.",
            )
        return None
    raw = f"{username}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def base_url() -> str:
    return (env("WB_AGENT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def default_entity() -> str | None:
    return env("WB_AGENT_ENTITY") or env("WANDB_ENTITY")


def default_project() -> str | None:
    return env("WB_AGENT_PROJECT") or env("WANDB_PROJECT")


def client_name(args: argparse.Namespace | None = None) -> str:
    """Resolve the client attribution tag.

    Precedence: explicit ``--client`` flag, then ``WB_AGENT_CLIENT``, then the
    default. An empty value disables tagging (no marker, no header).
    """
    if args is not None and getattr(args, "client", None) is not None:
        return args.client
    value = os.environ.get("WB_AGENT_CLIENT")
    return value if value is not None else DEFAULT_CLIENT


def ssl_context(insecure: bool) -> ssl.SSLContext | None:
    if not base_url().startswith("https://"):
        return None
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def request_json(
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    body: Any | None = None,
    auth: bool = True,
    timeout: float = 30,
    insecure: bool = False,
) -> Any:
    url = base_url() + path
    if query:
        filtered = {k: v for k, v in query.items() if v is not None}
        if filtered:
            url += "?" + urllib.parse.urlencode(filtered)

    data = None
    headers = {"Accept": "application/json"}
    client = client_name()
    if client:
        headers["User-Agent"] = f"aria-chat-skill ({client})"
        headers["X-Wandb-Client"] = client
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"

    header = auth_header(required=auth)
    if header:
        headers["Authorization"] = header

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context(insecure)) as resp:
            payload = resp.read()
            if resp.status == 204 or not payload:
                return None
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, payload or exc.reason) from exc
    except urllib.error.URLError as exc:
        raise ApiError(None, str(exc.reason)) from exc


def load_prompt(args: argparse.Namespace) -> str | list[dict[str, Any]]:
    if args.prompt_parts:
        if args.prompt_parts == "-":
            raw = sys.stdin.read()
        else:
            with open(args.prompt_parts, encoding="utf-8") as handle:
                raw = handle.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--prompt-parts must contain valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, (str, list)):
            raise SystemExit("--prompt-parts must contain a JSON string or array")
        if isinstance(parsed, str) and not parsed.strip():
            raise SystemExit("--prompt-parts JSON string must be non-empty")
        if isinstance(parsed, list) and not parsed:
            raise SystemExit("--prompt-parts JSON array must be non-empty")
        return parsed

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            return prompt
        raise SystemExit("Prompt text must be non-empty")

    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            return prompt
        raise SystemExit("Prompt text from stdin must be non-empty")

    raise SystemExit("Provide a prompt argument, pipe prompt text on stdin, or pass --prompt-parts")


def scoped_value(args: argparse.Namespace, name: str, default: str | None) -> tuple[str | None, bool]:
    if hasattr(args, name):
        return getattr(args, name) or None, True
    return default, False


def resolve_create_scope(args: argparse.Namespace) -> tuple[str | None, str | None]:
    entity, _entity_explicit = scoped_value(args, "entity", default_entity())
    project, project_explicit = scoped_value(args, "project", default_project())
    if project and not entity:
        if project_explicit:
            raise SystemExit("--entity or WANDB_ENTITY is required when project is provided")
        project = None
    return entity, project


def resolve_query_scope(args: argparse.Namespace) -> tuple[str | None, str | None]:
    entity, _entity_explicit = scoped_value(args, "entity", default_entity())
    project, project_explicit = scoped_value(args, "project", default_project())
    if project and not entity and not project_explicit:
        project = None
    if bool(entity) != bool(project):
        raise SystemExit("Provide both entity and project, or neither")
    return entity, project


def apply_client_marker(prompt: str | list[dict[str, Any]], args: argparse.Namespace) -> str | list[dict[str, Any]]:
    """Prepend a `client_info` prompt-part so the turn is attributable.

    The server stores `user_prompt` verbatim, so the marker survives the round
    trip. The agent treats unknown part types as inert. Returns the prompt
    unchanged when client tagging is disabled.
    """
    client = client_name(args)
    if not client:
        return prompt
    marker = {"type": CLIENT_PART_TYPE, "client": client}
    if isinstance(prompt, str):
        return [marker, {"type": "text", "text": prompt}]
    return [marker, *prompt]


def create_body(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {"user_prompt": apply_client_marker(load_prompt(args), args)}
    entity, project = resolve_create_scope(args)
    for key, value in (("entity", entity), ("project", project), ("title", getattr(args, "title", None))):
        if value is not None:
            body[key] = value
    if args.parent_turn_id:
        body["parent_turn_id"] = args.parent_turn_id
    permissions: dict[str, Any] = {}
    if args.allow_all_network_egress:
        permissions["allow_all_network_egress"] = True
    if getattr(args, "allowed_network_domains", None):
        permissions["allowed_network_domains"] = args.allowed_network_domains
    if permissions:
        body["permissions"] = permissions
    agent_config_override = getattr(args, "agent_config_override", None)
    if agent_config_override:
        body["agent_config_override"] = agent_config_override
    return body


def message_count(turn: dict[str, Any]) -> int:
    return len(turn.get("messages") or [])


def _content_text(content: Any) -> str:
    """Flatten message content that may be a plain string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "".join(chunks)
    return ""


def assistant_text(turn: dict[str, Any]) -> str:
    # The live TurnResponse has no `final_output`; the answer is the latest
    # assistant message. Scan messages in reverse and only trust assistant
    # records so we never echo the user prompt or a reasoning summary back.
    for record in reversed(turn.get("messages") or []):
        if record.get("role") != "assistant":
            continue
        content = record.get("content")
        if isinstance(content, str) and content.strip():
            return content
        message = record.get("message")
        if isinstance(message, dict):
            text = _content_text(message.get("content"))
            if text.strip():
                return text
    return ""


def agent_questions(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return turn.get("agent_questions") or []


def permission_requests(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return turn.get("permission_requests") or []


def turn_summary(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": turn.get("id"),
        "root_turn_id": turn.get("root_turn_id"),
        "parent_turn_id": turn.get("parent_turn_id"),
        "title": turn.get("title"),
        "state": turn.get("state"),
        "created_at": turn.get("created_at"),
        "updated_at": turn.get("updated_at"),
        "expires_at": turn.get("expires_at"),
        "messages": message_count(turn),
        "tool_calls": len(turn.get("tool_calls") or []),
        "has_response": bool(assistant_text(turn)),
        "agent_questions": len(agent_questions(turn)),
        "permission_requests": len(permission_requests(turn)),
        "error_info": turn.get("error_info"),
    }


def interaction_lines(turn: dict[str, Any]) -> list[str]:
    """Human-readable lines for any pending agent questions or permission asks.

    These are surfaced even when assistant text is present, because a turn can
    reach a terminal state while still waiting on the caller to answer a
    question or grant network access on the next (continuation) turn.
    """
    lines: list[str] = []
    for i, q in enumerate(agent_questions(turn)):
        lines.append(f"AGENT QUESTION [{i}]: {q.get('question', '')}")
        for j, opt in enumerate(q.get("options") or []):
            lines.append(f"  {j}. {opt}")
        lines.append("  (answer by creating a continuation turn with --parent-turn-id)")
    for req in permission_requests(turn):
        if req.get("type") == "network_access":
            domains = ", ".join(req.get("domains") or [])
            lines.append(f"NETWORK ACCESS REQUESTED: {req.get('reason', '')} -> [{domains}]")
            lines.append("  (grant by continuing with --allowed-network-domain <domain> ...)")
        else:
            lines.append(f"PERMISSION REQUEST ({req.get('type')}): {json.dumps(req)}")
    return lines


def print_turn(turn: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(turn, indent=2, sort_keys=True))
    elif fmt == "summary":
        print(json.dumps(turn_summary(turn), indent=2, sort_keys=True))
    elif fmt == "text":
        text = assistant_text(turn)
        extras = interaction_lines(turn)
        if text:
            print(text)
        if extras:
            if text:
                print()
            print("\n".join(extras))
        if not text and not extras:
            print(json.dumps(turn_summary(turn), indent=2, sort_keys=True))
    else:
        raise AssertionError(fmt)


def changed(turn: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.since_updated_at and turn.get("updated_at") != args.since_updated_at:
        return True
    if args.since_state and turn.get("state") != args.since_state:
        return True
    if args.since_message_count is not None and message_count(turn) != args.since_message_count:
        return True
    if turn.get("state") in TERMINAL_STATES:
        return True
    if assistant_text(turn):
        return True
    if agent_questions(turn) or permission_requests(turn):
        return True
    return False


def wait_for_turn(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    baseline_set = bool(args.since_updated_at or args.since_state or args.since_message_count is not None)

    while True:
        turn = request_json(
            "GET",
            f"/api/v1/turns/{urllib.parse.quote(args.turn_id)}",
            timeout=args.request_timeout,
            insecure=args.insecure,
        )
        state = turn.get("state")

        if args.until == "terminal" and state in TERMINAL_STATES:
            return turn
        if args.until == "response" and (assistant_text(turn) or state in TERMINAL_STATES):
            return turn
        if args.until == "updated":
            if state in TERMINAL_STATES or assistant_text(turn):
                return turn
            if not baseline_set:
                args.since_updated_at = turn.get("updated_at")
                args.since_state = state
                args.since_message_count = message_count(turn)
                baseline_set = True
            elif changed(turn, args):
                return turn

        if args.timeout and time.monotonic() - start >= args.timeout:
            raise ApiError(None, f"Timed out waiting for turn {args.turn_id} after {args.timeout:g}s")

        if not args.quiet:
            print(
                f"poll turn={args.turn_id} state={state} updated_at={turn.get('updated_at')}",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(args.poll_interval)


def cmd_health(args: argparse.Namespace) -> None:
    result = request_json("GET", "/api/v1/health", auth=False, timeout=args.request_timeout, insecure=args.insecure)
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_aliases(args: argparse.Namespace) -> None:
    result = request_json(
        "GET",
        "/api/v1/agent-aliases",
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_available(args: argparse.Namespace) -> None:
    entity, _entity_explicit = scoped_value(args, "entity", default_entity())
    if not entity:
        raise SystemExit("--entity or WANDB_ENTITY is required")
    result = request_json(
        "GET",
        "/api/v1/is-available",
        query={"entity": entity},
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_create(args: argparse.Namespace) -> None:
    turn = request_json(
        "POST",
        "/api/v1/turns",
        body=create_body(args),
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    if args.wait:
        wait_args = argparse.Namespace(
            turn_id=turn["id"],
            until="terminal",
            since_updated_at=None,
            since_state=None,
            since_message_count=None,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            quiet=args.quiet,
            request_timeout=args.request_timeout,
            insecure=args.insecure,
        )
        turn = wait_for_turn(wait_args)
    print_turn(turn, args.format)


def cmd_get(args: argparse.Namespace) -> None:
    turn = request_json(
        "GET",
        f"/api/v1/turns/{urllib.parse.quote(args.turn_id)}",
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print_turn(turn, args.format)


def cmd_wait(args: argparse.Namespace) -> None:
    turn = wait_for_turn(args)
    print_turn(turn, args.format)


def cmd_query(args: argparse.Namespace) -> None:
    entity, project = resolve_query_scope(args)
    query = {
        "entity": entity,
        "project": project,
        "parent_turn_id": args.parent_turn_id,
        "root_turn_id": args.root_turn_id,
        "state": args.state,
        "is_root": args.is_root,
        "sort_by": args.sort_by,
        "sort_desc": args.sort_desc,
        "limit": args.limit,
        "offset": args.offset,
    }
    turns = request_json("GET", "/api/v1/turns", query=query, timeout=args.request_timeout, insecure=args.insecure)
    if args.format == "json":
        print(json.dumps(turns, indent=2, sort_keys=True))
    else:
        print(json.dumps([turn_summary(turn) for turn in turns], indent=2, sort_keys=True))


def cmd_cancel(args: argparse.Namespace) -> None:
    turn = request_json(
        "POST",
        f"/api/v1/turns/{urllib.parse.quote(args.turn_id)}/cancel",
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print_turn(turn, args.format)


def cmd_archive(args: argparse.Namespace) -> None:
    request_json(
        "POST",
        f"/api/v1/turns/{urllib.parse.quote(args.turn_id)}/archive",
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print(json.dumps({"archived": args.turn_id}, indent=2, sort_keys=True))


def cmd_feedback(args: argparse.Namespace) -> None:
    value: bool | None
    if args.positive:
        value = True
    elif args.negative:
        value = False
    else:
        value = None
    body = {
        "feedback_is_positive": value,
        "feedback_reasoning": args.reason,
        "share_turn_data_with_wandb": bool(getattr(args, "share", False)),
    }
    request_json(
        "POST",
        f"/api/v1/turns/{urllib.parse.quote(args.turn_id)}/feedback",
        body=body,
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print(
        json.dumps(
            {
                "turn_id": args.turn_id,
                "feedback_is_positive": value,
                "feedback_reasoning": args.reason,
                "share_turn_data_with_wandb": body["share_turn_data_with_wandb"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_title(args: argparse.Namespace) -> None:
    turn = request_json(
        "PATCH",
        f"/api/v1/turns/{urllib.parse.quote(args.call_id)}",
        body={"title": args.title},
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print_turn(turn, args.format)


def cmd_generate_title(args: argparse.Namespace) -> None:
    result = request_json(
        "POST",
        f"/api/v1/turns/{urllib.parse.quote(args.leaf_turn_id)}/generate-and-set-thread-title",
        timeout=args.request_timeout,
        insecure=args.insecure,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-timeout", type=float, default=30, help="HTTP request timeout in seconds")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")


def add_turn_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("summary", "json", "text"), default="summary", help="Output format")


def add_wait_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--poll-interval", type=float, default=2, help="Seconds between polls")
    parser.add_argument("--timeout", type=float, default=0, help="Overall wait timeout in seconds; 0 means no timeout")
    parser.add_argument("--quiet", action="store_true", help="Do not print poll progress to stderr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the hosted WB Agent over HTTP")
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="Check service health")
    add_common(health)
    health.set_defaults(func=cmd_health)

    available = sub.add_parser("available", help="Check whether WB Agent is available for an entity")
    add_common(available)
    available.add_argument("--entity", default=argparse.SUPPRESS, help="Defaults to WB_AGENT_ENTITY or WANDB_ENTITY")
    available.set_defaults(func=cmd_available)

    create = sub.add_parser("create", help="Create a WB Agent turn")
    add_common(create)
    add_turn_output(create)
    add_wait_options(create)
    create.add_argument("--entity", default=argparse.SUPPRESS, help="Defaults to WB_AGENT_ENTITY or WANDB_ENTITY")
    create.add_argument("--project", default=argparse.SUPPRESS, help="Defaults to WB_AGENT_PROJECT or WANDB_PROJECT")
    create.add_argument("--title")
    create.add_argument("--parent-turn-id")
    create.add_argument("--prompt-parts", help="Path to JSON string/array prompt parts, or '-' for stdin")
    create.add_argument("--allow-all-network-egress", action="store_true")
    create.add_argument(
        "--allowed-network-domain",
        action="append",
        dest="allowed_network_domains",
        metavar="DOMAIN",
        help="Add a domain to the sandbox outbound allowlist (repeatable). Use to grant a turn's "
        "network_access permission_request scoped access instead of --allow-all-network-egress.",
    )
    create.add_argument(
        "--agent-config-override",
        help="Request a named agent variant for this turn (see the `aliases` command for valid values). "
        "Null uses the service default and is not inherited from the parent turn. Unknown values are rejected with 422.",
    )
    create.add_argument(
        "--client",
        default=None,
        help="Client attribution tag carried as a `client_info` prompt-part and request headers "
        "(default: coding_agent). Overrides WB_AGENT_CLIENT; pass an empty string to disable tagging.",
    )
    create.add_argument("--wait", action="store_true", help="Wait for terminal state after creating the turn")
    create.add_argument("prompt", nargs="*", help="Prompt text; omitted prompt is read from stdin")
    create.set_defaults(func=cmd_create)

    get = sub.add_parser("get", help="Fetch one turn")
    add_common(get)
    add_turn_output(get)
    get.add_argument("turn_id")
    get.set_defaults(func=cmd_get)

    wait = sub.add_parser("wait", help="Poll a turn until a condition is met")
    add_common(wait)
    add_turn_output(wait)
    add_wait_options(wait)
    wait.add_argument("turn_id")
    wait.add_argument("--until", choices=("terminal", "response", "updated"), default="terminal")
    wait.add_argument("--since-updated-at")
    wait.add_argument("--since-state")
    wait.add_argument("--since-message-count", type=int)
    wait.set_defaults(func=cmd_wait)

    wake = sub.add_parser("wake", help="Poll until a turn updates or reaches terminal state")
    add_common(wake)
    add_turn_output(wake)
    add_wait_options(wake)
    wake.add_argument("turn_id")
    wake.add_argument("--since-updated-at")
    wake.add_argument("--since-state")
    wake.add_argument("--since-message-count", type=int)
    wake.set_defaults(func=cmd_wait, until="updated")

    query = sub.add_parser("query", help="List turns with optional filters")
    add_common(query)
    query.add_argument("--format", choices=("summary", "json"), default="summary")
    query.add_argument("--entity", default=argparse.SUPPRESS, help="Defaults to WB_AGENT_ENTITY or WANDB_ENTITY")
    query.add_argument("--project", default=argparse.SUPPRESS, help="Defaults to WB_AGENT_PROJECT or WANDB_PROJECT")
    query.add_argument("--parent-turn-id")
    query.add_argument("--root-turn-id")
    query.add_argument("--state")
    query.add_argument("--is-root", type=lambda v: v.lower() in {"1", "true", "yes"})
    query.add_argument("--sort-by", choices=("created_at", "updated_at", "id"), default="created_at")
    query.add_argument("--sort-desc", action="store_true")
    query.add_argument("--limit", type=int, default=100)
    query.add_argument("--offset", type=int, default=0)
    query.set_defaults(func=cmd_query)

    cancel = sub.add_parser("cancel", help="Cancel a queued or in-progress turn")
    add_common(cancel)
    add_turn_output(cancel)
    cancel.add_argument("turn_id")
    cancel.set_defaults(func=cmd_cancel)

    archive = sub.add_parser("archive", help="Archive a turn and its descendants")
    add_common(archive)
    archive.add_argument("turn_id")
    archive.set_defaults(func=cmd_archive)

    feedback = sub.add_parser("feedback", help="Set or clear feedback on a completed turn")
    add_common(feedback)
    feedback.add_argument("turn_id")
    group = feedback.add_mutually_exclusive_group(required=True)
    group.add_argument("--positive", action="store_true")
    group.add_argument("--negative", action="store_true")
    group.add_argument("--clear", action="store_true")
    feedback.add_argument("--reason")
    feedback.add_argument(
        "--share",
        action="store_true",
        help="Set share_turn_data_with_wandb=true to opt this turn's data in to sharing with W&B.",
    )
    feedback.set_defaults(func=cmd_feedback)

    title = sub.add_parser("title", help="Set, clear, or leave unchanged a turn title")
    add_common(title)
    add_turn_output(title)
    title.add_argument("call_id", help="Turn/call ID")
    title.add_argument("--title", help="New title; omit to send null and leave unchanged", default=None)
    title.set_defaults(func=cmd_title)

    aliases = sub.add_parser("aliases", help="List agent variants accepted by create --agent-config-override")
    add_common(aliases)
    aliases.set_defaults(func=cmd_aliases)

    generate_title = sub.add_parser(
        "generate-title", help="Generate and store a thread title from a leaf turn's conversation"
    )
    add_common(generate_title)
    generate_title.add_argument("leaf_turn_id", help="A turn ID in the thread; its root receives the title")
    generate_title.set_defaults(func=cmd_generate_title)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except ApiError as exc:
        prefix = f"WB Agent API error {exc.status}: " if exc.status else "WB Agent API error: "
        print(prefix + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
