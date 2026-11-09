# Codex smoke harness

A small, cheap check that an installed skill actually drives an agent: install
the skill the way the README tells users to (`npx skills add`), run a handful of
read-only prompts through Codex CLI with an inexpensive model, and record what
the agent ran and answered. The same prompts can be run against the hosted W&B
MCP server instead of the skill, which gives a baseline to spot regressions.

This is not Skill Bench. It has no scorer and needs no WBAF checkout; a human
reads `summary.csv` and the answers.

## Requirements

- `node`/`npx` (for `npx skills add`) and `codex` (`npm i -g @openai/codex`)
- Codex authenticated: `printf '%s' "$OPENAI_API_KEY" | codex login --with-api-key`
- `WANDB_API_KEY` in the environment
- A Python with `wandb`, `weave` and `wandb-workspaces` installed first on `PATH`
  (the skill's helpers import them)

Never put keys in task files or in this directory.

## Run

```bash
# skill under test (default: this checkout), 4 tasks, gpt-5-mini
python3 bench/smoke/run.py --mode skill --model gpt-5-mini --out .tmp/smoke-skill

# same prompts through the hosted MCP server, no skill installed
python3 bench/smoke/run.py --mode mcp --model gpt-5-mini --out .tmp/smoke-mcp

python3 bench/smoke/summarize.py .tmp/smoke-skill .tmp/smoke-mcp > summary.csv
```

`run.py` creates a throwaway project directory per run, installs the skill
into it (`--mode skill`) or registers `https://mcp.withwandb.com/mcp` in a
private `CODEX_HOME` (`--mode mcp`), then runs each prompt with
`codex exec --json` so the full event log is kept. The MCP mode runs with
`--dangerously-bypass-approvals-and-sandbox`; Codex's `never` approval policy
otherwise refuses every MCP call. Skill mode needs the same flag because the
helpers make network requests.

`summarize.py` turns the JSONL logs into one CSV row per task: wall time,
tokens, which helper scripts or MCP tools were called, and the final answer.

`list_mcp_tools.py <url> <out.json>` prints the per-tool description and schema
sizes of an MCP server's `tools/list`, which is the per-turn context cost the
server imposes on every client.

## Tasks

Prompts live in `tasks/*.txt` and are read-only against `wandb/moe-pretraining`.
They are phrased for the skill; `run.py --mode mcp` rewrites the skill wording
to "Using the wandb MCP tools" so both sides get the same question.
