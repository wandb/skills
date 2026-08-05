# skills

[![Codex](https://img.shields.io/badge/codex-25%2F34%20(74%25)-yellow)](#benchmarks)
[![Claude Code](https://img.shields.io/badge/claude--code-31%2F34%20(91%25)-brightgreen)](#benchmarks)

<!-- Uncomment to make badges live (requires public repo):
[![Codex](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/skills/main/.badges/codex.json)](#benchmarks)
[![Claude Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/skills/main/.badges/claude-code.json)](#benchmarks)
-->

Skills to guide Claude Code, Codex, and other coding agents on using the [Weights & Biases](https://wandb.ai) AI developer platform to train models and build agents.

## For model training

- Log metrics and rich media during model training and fine-tuning
- Track model training experiments
- Analyze runs and experiment results to understand how the model is learning
- Tune hyperparameters

## For agent building

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

## Requirements

The skill helpers are validated against these versions. Older SDKs expose some
of the methods used here as unimplemented stubs, so pin at or above the floor
rather than relying on whatever is already installed.

| Package | Floor | Validated against |
| --- | --- | --- |
| Python | `>=3.13` | 3.13 |
| `wandb` | `>=0.28.1` | 0.28.1 |
| `wandb-workspaces` | `>=0.4.4` | 0.4.4 (pulled in by the `[workspaces]` extra) |
| `weave` | `>=0.52.41` | 0.52.41 |

The `workspaces` extra is required for the Workspaces helpers:

```bash
uv run --with 'wandb[workspaces]>=0.28.1' --with 'weave>=0.52.41' python your_script.py
```

`0.28.1` is a floor rather than a preference: at least one helper branches on
behavior that changed after `0.28.0`, so `>=0.28.0` is not sufficient.

## Available Skills

<!-- BEGIN SKILL TABLE -->
| Skill                                    | Description                                                                                                                               | Status       |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| [`wandb-primary`](skills/wandb-primary/) | Broad W&B project analysis and operations across runs, Artifacts, Registry, Weave, Reports, Workspaces, and Launch. | experimental |
| [`wandb-eval-tables`](skills/wandb-eval-tables/) | Non-destructive conversion of W&B Table artifacts into bounded, verified EvalTable previews. | experimental |
| [`wandb-autoresearch`](skills/wandb-autoresearch/) | Bounded training research through W&B Launch, including readiness checks, serial trials, comparison, and resumable state. | experimental |
<!-- END SKILL TABLE -->

## Benchmarks

We maintain Skill Bench in this repository to evaluate public skill changes
across coding agents and task categories. Skill Bench uses W&B Agent Factory as
the eval runtime for task definitions, agent profiles, sandbox execution, and
structured bench rows.

Pull requests run package validation by default. A maintainer can trigger live
Skill Bench runs for larger changes.

Plan a local benchmark without model calls:

```bash
python3 -m skillbench.cli plan \
  --wbaf-root ../WandBAgentFactory \
  --candidate-ref HEAD \
  --skill wandb-primary
```

| Category                      | Tasks | Claude Code (`sonnet4.6`) | Codex (`gpt-5.3-codex`) |
| ----------------------------- | ----- | ------------------------- | ----------------------- |
| Weave analysis                | 26    | 97%*                      | 63%*                    |
| Weave tooling                 | 11    | 95%*                      | 83%*                    |
| Model training                | 8     | 90%*                      | 85%*                    |
| LLM finetuning & RL analysis  | 14    | 72%*                      | 86%*                    |
| Failure & outlier detection   | 8     | 86%*                      | 63%*                    |

*Pass rates are +/- 3%. Many tasks span multiple categories.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
