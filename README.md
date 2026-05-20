# skills

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

### For agent building
- Trace agentic AI applications
- Analyze traces and classify them into failure modes
- Evaluate models with labeled datasets
- Run online evaluations for production monitoring

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
| [`wandb-models`](skills/wandb-models/) | W&B Models skill for runs, sweeps, artifacts, configs, metrics, run history, best-run selection, and model analysis. | experimental |
| [`weave-analysis`](skills/weave-analysis/) | W&B Weave skill for traces, calls, evaluations, scorers, token/cost analysis, and model usage. | experimental |
| [`wandb-reports`](skills/wandb-reports/) | W&B Reports skill for report authoring, workspace guidance, report media, and report/project semantics. | experimental |
| [`signal-builder`](skills/signal-builder/) | Signal Builder skill for project-grounded monitor and LLM-as-judge signal design. | experimental |
| [`wandb-launch`](skills/wandb-launch/) | W&B Launch skill for queues, jobs, relaunching runs, and remote compute workflows. | experimental |
<!-- END SKILL TABLE -->

## Benchmarks

We maintain benchmark reports that compare public skill changes against the
appropriate W&B Agent Factory task list. Pull requests get a plan-only report
when WBAF is available to CI. Maintainers can trigger a live benchmark to compare
the skill on `main` against the proposed PR version.

| Category | Tasks | Claude Code (`sonnet4.6`) | Codex (`gpt-5.3-codex`) |
|----------|-------|-------------|-------|
| Weave analysis | 26 | 97%* | 63%* |
| Weave tooling | 11 | 95%* | 83%* |
| Model training | 8 | 90%* | 85%* |
| LLM finetuning & RL analysis | 14 | 72%* | 86%* |
| Failure & outlier detection | 8 | 86%* | 63%* |

*Pass rates are +/- 3%. Many tasks span multiple categories.

See [CONTRIBUTING.md](CONTRIBUTING.md#skill-benchmarks) for how benchmark
reports work and why live evals are maintainer-gated.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
