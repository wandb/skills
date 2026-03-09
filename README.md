# skills

[![Evaluate Skills](https://github.com/wandb/skills/actions/workflows/eval-skills.yml/badge.svg)](https://github.com/wandb/skills/actions/workflows/eval-skills.yml)
[![Codex](https://img.shields.io/badge/codex-25%2F34%20(74%25)-yellow)](#benchmarks)
[![Claude Code](https://img.shields.io/badge/claude--code-31%2F34%20(91%25)-brightgreen)](#benchmarks)

<!-- Uncomment to make badges live (requires public repo):
[![Codex](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/skills/main/.badges/codex.json)](#benchmarks)
[![Claude Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/skills/main/.badges/claude-code.json)](#benchmarks)
-->

Give your coding agent the ability to work with [Weights & Biases](https://wandb.ai). Example tasks these skills can help with:

- **Training runs & metrics** — "Plot the loss curves for my last 5 runs" / "Compare hyperparameters across my sweep"
- **GenAI traces & evals** — "Show me the slowest Weave traces from today" / "Summarize my latest evaluation results"
- **Artifacts & datasets** — "Download the best model artifact from this project" / "List all datasets in my team's workspace"
- **Reports & dashboards** — "Create a W&B report comparing these two experiments"

## What can this skill do?

This official distribution of the Weights and Biases skill is purpose-built to give your coding agent full-control over the WandB SDK. 
Using this skill you can query runs, weave-traces, evaluations, triage errors, identify outliers, and much more. 

## Installation

Using [`npx skills`](https://github.com/vercel-labs/skills):

**Local** (current project):
```bash
npx skills add wandb/skills --skill '*' --yes
```

**Global** (all projects):
```bash
npx skills add wandb/skills --skill '*' --yes --global
```

To link skills to a specific agent (e.g. Claude Code):
```bash
npx skills add wandb/skills --agent claude-code --skill '*' --yes --global
```

## Usage

After installation, set your [W&B API key](https://wandb.ai/authorize):

```bash
export WANDB_API_KEY=<your-key>
```

Then run your coding agent from the directory where you installed (for local installs) or from anywhere (for global installs).

## Available Skills

<!-- BEGIN SKILL TABLE -->
| Skill | Description | Status |
|-------|-------------|--------|
| [`wandb-primary`](skills/wandb-primary/) | Comprehensive primary skill for agents working with Weights & Biases. Covers both the W&B SDK (tr... | claude-code: 32/35 (91%) | codex: 25/35 (71%) |
<!-- END SKILL TABLE -->

## Benchmarks

We maintain a growing internal benchmark suite that evaluates each skill across coding agents and task categories. Skills are evaluated automatically on every merge to `main`.

| Category | Claude Code (Sonnet) | Claude Code (Opus) | Codex (gpt-5.3) | Codex (gpt-5.4) |
|----------|----------------------|--------------------|-----------------|-----------------|
| SDK queries | - | - | - | - |
| Weave traces | - | - | - | - |
| Data analysis | - | - | - | - |
| Artifacts | - | - | - | - |
| **Overall** | - | - | - | - |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
