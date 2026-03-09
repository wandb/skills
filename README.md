# skills

[![Evaluate Skills](https://github.com/wandb/skills/actions/workflows/eval-skills.yml/badge.svg?branch=main)](https://github.com/wandb/skills/actions/workflows/eval-skills.yml?query=branch%3Amain)
[![Codex](https://img.shields.io/badge/codex-25%2F34%20(74%25)-yellow)](#benchmarks)
[![Claude Code](https://img.shields.io/badge/claude--code-31%2F34%20(91%25)-brightgreen)](#benchmarks)

<!-- Uncomment to make badges live (requires public repo):
[![Codex](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/skills/main/.badges/codex.json)](#benchmarks)
[![Claude Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/skills/main/.badges/claude-code.json)](#benchmarks)
-->

Skills to guide Claude Code, Codex, and other coding agents on using the [Weights & Biases](https://wandb.ai) AI developer platform to train models and build agents.

### For model training
- Log metrics and rich media during model training and fine-tuning
- Track model training experiments
- Analyze runs and experiment results to understand how the model is learning
- Tune hyperparameters
- Save model checkpoints and other artifacts
- Track artifact lineage
- Use the Registry for model and dataset lifecycle workflows, collaboration, and governance

### For agent building
- Trace agentic AI applications
- Analyze traces and classify them into failure modes
- Evaluate models with labeled datasets
- Run online evaluations for production monitoring
- Turn production traces into test cases and add them to evaluation datasets
- Build guardrails for agentic systems
- Generate dynamic reports to communicate across teams

## Getting Started

```bash
npx skills add wandb/skills
```

Then set your [W&B API key](https://wandb.ai/authorize):

```bash
export WANDB_API_KEY=<your-key>
```

> `npx skills` is a utility for installing skills into major coding agent CLIs. Use `--global` to install for all projects, or `--agent <name>` to target a specific agent. See the [npx skills docs](https://github.com/vercel-labs/skills) for more details.

## Available Skills

<!-- BEGIN SKILL TABLE -->
| Skill | Description | Status |
|-------|-------------|--------|
| [`wandb-primary`](skills/wandb-primary/) | Comprehensive primary skill for agents working with Weights & Biases. Covers both the W&B SDK (tr... | claude-code: 32/35 (91%) | codex: 25/35 (71%) |
<!-- END SKILL TABLE -->

## Benchmarks

We maintain a growing internal benchmark suite that evaluates each skill across coding agents and task categories. Skills are evaluated automatically on every merge to `main`.

| Category | Tasks |
|----------|-------|
| Weave analysis | 26 |
| Weave tooling | 11 |
| Model training | 8 |
| LLM finetuning & RL analysis | 14 |
| Failure & outlier detection | 8 |

*Many tasks span multiple categories.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
