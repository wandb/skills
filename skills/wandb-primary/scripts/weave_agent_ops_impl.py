# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Read the Weave **agents plane** — conversations and agent traces/turns.

The `weave/agents/...` app URLs expose agent conversations on a
separate read surface from calls: a `conversation_id` is NOT a trace/call/thread
id (`get_calls` returns 0), and the SDK's `client.server.agent_*` methods are
unimplemented stubs that silently return None. This CLI goes through the REST
endpoints directly and prints a compact, ready-to-summarize transcript, so a
"summarize this agent conversation" turn is one command, not a hand-written
fetch-and-parse script.

    W=skills/wandb-primary/scripts/weave_agent_ops.py
    python $W conversation <conversation_id> [--path ENTITY/PROJECT]
    python $W trace <trace_id>               [--path ENTITY/PROJECT]
    python $W --help

Defaults `--path` to `WANDB_ENTITY/WANDB_PROJECT`. Output contract mirrors the
other *_ops helpers: machine-readable lines, `answer=` is the headline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

logging.getLogger("weave").setLevel(logging.ERROR)

# Server-enforced ceiling on AgentConversationChatReq.limit (Le(le=50)).
_PAGE_LIMIT = 50
_CONVERSATION_PATH = "/agents/conversations/chat"
_TRACE_PATH = "/agents/traces/chat"
# Default clip widths — tool results/args are often huge; text rarely is.
_CLIP_TEXT = 800
_CLIP_ARGS = 300
_CLIP_RESULT = 600


def _resolve_path(path: str | None) -> str:
    if path:
        return path
    entity, project = os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT")
    if not (entity and project):
        sys.exit("error=no --path given and WANDB_ENTITY/WANDB_PROJECT unset")
    return f"{entity}/{project}"


def _server(pid: str) -> Any:
    """The RemoteHTTPTraceServer underneath weave's caching middleware.

    `_generic_request` lives on the remote binding; the cache wrapper doesn't
    expose the agents methods, so reach through `_next_trace_server`.
    """
    import weave

    client = weave.init(pid)
    server = getattr(client.server, "_next_trace_server", client.server)
    if not hasattr(server, "_generic_request"):
        sys.exit("error=could not reach RemoteHTTPTraceServer._generic_request")
    return server


def _clip(value: Any, width: int, full: bool) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())  # collapse whitespace so one message = one line
    if full or len(text) <= width:
        return text
    return f"{text[:width]}… [+{len(text) - width} chars]"


def _print_turn(turn: Any, index: int, total: int, full: bool) -> tuple[int, int]:
    """Print one turn's header + messages; return (input_tokens, output_tokens)."""
    dur = getattr(turn, "total_duration_ms", None)
    print(
        f"\nTURN {index}/{total}  trace={getattr(turn, 'trace_id', None)}  "
        f"root={getattr(turn, 'root_span_name', None)}  "
        f"dur={dur}ms  msgs={len(getattr(turn, 'messages', []) or [])}"
    )
    in_tok = out_tok = 0
    for m in getattr(turn, "messages", []) or []:
        kind = getattr(m, "type", None)
        payload = getattr(m, kind, None) if kind else None
        if kind == "tool_call":
            print(
                f"  tool[{getattr(payload, 'tool_name', '?')}]> "
                f"args={_clip(getattr(payload, 'tool_arguments', ''), _CLIP_ARGS, full)} "
                f"result={_clip(getattr(payload, 'tool_result', ''), _CLIP_RESULT, full)}"
            )
        elif kind == "assistant_message":
            in_tok += getattr(payload, "input_tokens", 0) or 0
            out_tok += getattr(payload, "output_tokens", 0) or 0
            model = getattr(payload, "model", "") or ""
            print(f"  assistant({model})> {_clip(getattr(payload, 'text', ''), _CLIP_TEXT, full)}")
        elif kind == "user_message":
            print(f"  user> {_clip(getattr(payload, 'text', ''), _CLIP_TEXT, full)}")
        elif kind == "agent_start":
            print(f"  agent_start> model={getattr(payload, 'model', None)} status={getattr(payload, 'status', None)}")
        else:  # agent_handoff, context_compacted, or anything new — fail visible, not silent
            print(f"  {kind}> {_clip(getattr(payload, 'text', '') if payload else '', _CLIP_TEXT, full)}")
    return in_tok, out_tok


def _cmd_conversation(args: argparse.Namespace) -> None:
    from weave.trace_server.agents import types as at

    pid = _resolve_path(args.path)
    server = _server(pid)
    turns: list[Any] = []
    total_reported = 0
    offset = 0
    while True:
        res = server._generic_request(
            _CONVERSATION_PATH,
            at.AgentConversationChatReq(
                project_id=pid, conversation_id=args.conversation_id,
                limit=_PAGE_LIMIT, offset=offset, include_feedback=args.feedback,
            ),
            at.AgentConversationChatReq, at.AgentConversationChatRes,
        )
        if res is None:
            sys.exit(f"error=no conversation {args.conversation_id} in {pid} (check id/project)")
        batch = list(getattr(res, "turns", []) or [])
        total_reported = getattr(res, "total_turns", None) or total_reported
        turns.extend(batch)
        if not batch or not getattr(res, "has_more", False):
            break
        offset += len(batch)

    n = len(turns)
    dur_ms = sum(getattr(t, "total_duration_ms", 0) or 0 for t in turns)
    print(f"answer=conversation {args.conversation_id}: {n} turn(s), ~{round(dur_ms / 1000)}s total")
    if total_reported and total_reported != n:
        print(f"page_caveat=server reports {total_reported} turns but {n} fetched")
    in_tot = out_tot = 0
    for i, turn in enumerate(turns, 1):
        di, do = _print_turn(turn, i, n, args.full)
        in_tot += di
        out_tot += do
    print(f"\ntokens=input {in_tot} / output {out_tot}")


def _cmd_trace(args: argparse.Namespace) -> None:
    from weave.trace_server.agents import types as at

    pid = _resolve_path(args.path)
    server = _server(pid)
    res = server._generic_request(
        _TRACE_PATH,
        at.AgentTraceChatReq(project_id=pid, trace_id=args.trace_id, include_feedback=args.feedback),
        at.AgentTraceChatReq, at.AgentTraceChatRes,
    )
    if res is None:
        sys.exit(f"error=no agent trace {args.trace_id} in {pid}")
    print(f"answer=agent trace {args.trace_id}: root={getattr(res, 'root_span_name', None)}")
    in_tot, out_tot = _print_turn(res, 1, 1, args.full)
    print(f"\ntokens=input {in_tot} / output {out_tot}")


def main() -> None:
    # Shared flags live on a parent so they work whether placed before OR after
    # the subcommand (`conversation <id> --path …` is the natural invocation).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--path", help="ENTITY/PROJECT (default: $WANDB_ENTITY/$WANDB_PROJECT)")
    common.add_argument("--feedback", action="store_true", help="include feedback rows")
    common.add_argument("--full", action="store_true", help="do not clip long tool args/results/text")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("conversation", parents=[common], help="fetch + print a full agent conversation (all turns)")
    c.add_argument("conversation_id", help="from a .../weave/agents/conversations/<id> URL")
    c.set_defaults(func=_cmd_conversation)

    t = sub.add_parser("trace", parents=[common], help="fetch + print one agent trace/turn by trace_id")
    t.add_argument("trace_id")
    t.set_defaults(func=_cmd_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
