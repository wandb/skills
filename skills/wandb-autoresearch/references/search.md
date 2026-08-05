<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Run the search

Reach here once there's **launchable code + a queue with capacity**. This is the
actual autoresearch: a bounded, serial hyperparameter search over the user's real
training job.

## Smoke-test first

Launch the user's *real* code at a deliberately tiny setting (a couple of steps /
one short epoch) to prove queue + agent + image + code + hardware work together
before spending on a full run. If it fails, debug with `check_launch` (see
the Launch section in `skills/wandb-primary/SKILL.md`) before continuing.

For every finished run (smoke test, trial, or full run), include where it ran
(GPU active vs CPU-only) in the reported result.

## Bounded, serial search — not an open-ended sweep

- Tuning can go beyond config field knobs: a code change, an architecture
  tweak, the optimizer, the tokenizer, or the data pipeline can all be a
  trial. Pick the axis that best fits the goal, not only the one that's
  easiest to launch.
- Agree the bounds with the user out loud: the metric and goal (e.g. minimize
  `final_loss`), the trial budget (how many runs), and the axis (which config
  field, what values) — and one line on why that axis.
- If you shrink trials to save compute (shorter runs, less data, fewer GPUs),
  say so and say why — why you shrank them, and why the winner from short runs
  might not be the winner at the real size.
- One trial at a time. Relaunch the base run with a single config override, wait
  for it to finish, read its metric, then choose the next trial from that result:

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")

from launch_helpers import relaunch_run

status = relaunch_run(run_path="ENTITY/PROJECT/RUN_ID", queue_name="QUEUE",
                      config={"lr": 0.03}, wait_for="done")
```

Keep a short running tally (trial → config → metric) so the user sees progress,
and sync research state at meaningful checkpoints (see SKILL
`## Experiment records`).
Stop at the trial budget or when the metric plateaus, then report the best config
and offer to keep going.

Metrics are noisy. When the gap between top configs is smaller than run-to-run
noise could explain, spend a trial re-running the winner with a different seed
before declaring it — one run is one sample, not an answer.
