# agent-skills

[![Evaluate Skills](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml/badge.svg)](https://github.com/wandb/agent-skills/actions/workflows/eval-skills.yml)

Official Agent Skills for Weights & Biases Models and Weave.

## Available Skills

<!-- BEGIN SKILL TABLE -->
| Skill | Description | Status |
|-------|-------------|--------|
| [`wandb`](skills/wandb/) | Explore and analyze W&B projects, runs, metrics, and configs via `wandb.Api` | evaluated |
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

Skills are automatically evaluated on merge to `main` using the [WandBAgentFactory](https://github.com/wandb/WandBAgentFactory) eval framework with a Codex agent and MCP tools. Results are tracked in [Weave](https://wandb.ai/site/weave).

## Contributing

1. Create `skills/<your-skill>/SKILL.md` with frontmatter and instructions
2. Open a PR — CI will run the skill through the eval suite
3. Merge to `main` to publish
