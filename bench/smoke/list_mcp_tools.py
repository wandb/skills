#!/usr/bin/env python3
"""Print per-tool description and schema sizes from an MCP server's tools/list.

Usage: list_mcp_tools.py URL OUT_JSON   (reads WANDB_API_KEY for the bearer token)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def _post(url: str, session: str | None, body: dict) -> tuple[str | None, dict | None]:
    headers = {
        "Authorization": f"Bearer {os.environ['WANDB_API_KEY']}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session:
        headers["mcp-session-id"] = session
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode()
        session = resp.headers.get("mcp-session-id", session)
        if "text/event-stream" in resp.headers.get("content-type", ""):
            text = "".join(l[5:] for l in text.splitlines() if l.startswith("data:"))
    return session, (json.loads(text) if text.strip() else None)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    url, out = argv
    session, init = _post(url, None, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "0"}},
    })
    print("server:", (init or {}).get("result", {}).get("serverInfo"))
    _post(url, session, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    _, listed = _post(url, session, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = (listed or {})["result"]["tools"]
    json.dump(tools, open(out, "w"), indent=1)
    total_desc = total_schema = 0
    print(f"{'tool':48} {'desc':>6} {'schema':>7}")
    for tool in sorted(tools, key=lambda t: -len(t.get("description", ""))):
        desc = len(tool.get("description", ""))
        schema = len(json.dumps(tool.get("inputSchema", {})))
        total_desc += desc
        total_schema += schema
        print(f"{tool['name'][:48]:48} {desc:6} {schema:7}")
    total = total_desc + total_schema
    print(f"tools={len(tools)} desc_chars={total_desc} schema_chars={total_schema} total={total} (~{total // 4} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
