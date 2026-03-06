# agent-skills

[![Evaluate Skills](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml/badge.svg)](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml)
[![Codex](https://img.shields.io/badge/codex-25%2F34%20(74%25)-yellow)](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml)
[![Claude Code](https://img.shields.io/badge/claude--code-31%2F34%20(91%25)-brightgreen)](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml)

<!-- Uncomment to make badges live (requires public repo):
[![Codex](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/agent-skills/main/.badges/codex.json)](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml)
[![Claude Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/agent-skills/main/.badges/claude-code.json)](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml)
-->

Agent skills for working with [Weights & Biases](https://wandb.ai). Explore projects, analyze runs, query metrics, and create reports — all from your coding agent.

## Supported Coding Agents

These skills can be installed for any agent supported by [`skills.sh`](https://skills.sh), including Claude Code, Cursor, Windsurf, Goose, and [many more](https://github.com/vercel-labs/skills).

## Prerequisites

- A [W&B API key](https://wandb.ai/authorize)

## Installation

### Quick Install

Using [`npx skills`](https://github.com/vercel-labs/skills):

**Local** (current project):
```bash
npx skills add wandb/agent-skills --skill '*' --yes
```

**Global** (all projects):
```bash
npx skills add wandb/agent-skills --skill '*' --yes --global
```

To link skills to a specific agent (e.g. Claude Code):
```bash
npx skills add wandb/agent-skills --agent claude-code --skill '*' --yes --global
```

### Install Script (Claude Code only)

Alternatively, clone the repo and use the install script:

```bash
git clone https://github.com/wandb/agent-skills.git
cd agent-skills

# Install for Claude Code in current directory (default)
./install.sh

# Install for Claude Code globally
./install.sh --global

# Force reinstall without prompts
./install.sh --force --yes
```

| Flag | Description |
|------|-------------|
| `--global`, `-g` | Install globally instead of current directory |
| `--force`, `-f` | Overwrite skills with same names as this package |
| `--yes`, `-y` | Skip confirmation prompts |

## Usage

After installation, set your API key:

```bash
export WANDB_API_KEY=<your-key>
```

Then run your coding agent from the directory where you installed (for local installs) or from anywhere (for global installs).

## Available Skills

<!-- BEGIN SKILL TABLE -->
| Skill | Description | Status |
|-------|-------------|--------|
| [`wandb-primary`](skills/wandb-primary/) | Comprehensive primary skill for agents working with Weights & Biases. Covers both the W&B SDK (tr... | claude-code: 31/34 (91%) | codex: 25/34 (74%) |
<!-- END SKILL TABLE -->

## Skill Format

Each skill is a directory under `skills/` with a `SKILL.md` file:

```
skills/<name>/
└── SKILL.md          # YAML frontmatter + instructions
```

### SKILL.md structure

```markdown
---
name: my-skill
description: One-line description of when to use this skill.
---

# Skill Title

Instructions, code patterns, and best practices for the agent...
```

The frontmatter requires `name` and `description`. The body contains instructions that get injected into the agent's context.

## Evaluation

Skills are automatically evaluated on merge to `main` using the [WandBAgentFactory](https://github.com/wandb/WandBAgentFactory) eval framework, benchmarked against both Codex and Claude Code agents. Results are tracked in [Weave](https://wandb.ai/site/weave).

## Contributing

1. Create `skills/<your-skill>/SKILL.md` with frontmatter and instructions
2. Open a PR — CI will run the skill through the eval suite
3. Merge to `main` to publish

To update an existing installation:

```bash
./install.sh --force
```
