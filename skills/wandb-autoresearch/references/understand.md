<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Understand the project (state review)

Always the first stage. Find the user's entity and project (the environment may
set `WANDB_ENTITY`/`WANDB_PROJECT`; ask only if you can't infer them). If you've
worked this project before, restore prior research state first (see SKILL
`## Experiment records`) so you don't repeat trials.

Then look at the actual state and reason about it — don't run a classifier, and
don't guess from raw output:

- **Is there launchable code?** Sample the project's recent runs and check
  whether any logged or used a `job`/`code` artifact (`wandb.Api().runs(...)`,
  then `run.logged_artifacts()` / `run.used_artifacts()`). A project that only
  logs metrics has nothing to relaunch.
- **Is there a queue with capacity?** List the entity's queues (`list_queues`)
  and check each for an active agent *and* whether recent items are succeeding
  rather than failing. **Don't silently auto-pick a queue** — running on the
  wrong one is irritating and wastes the user's compute. If more than one queue
  could work, or the right choice isn't obvious, surface what you found and let
  the user pick; a healthy single queue you can propose, but still name it before
  you launch on it. (A failing queue on the same entity shouldn't fool you into
  thinking there's no capacity.)
- **Where does the data come from?** A `dataset` artifact, or baked into the
  code/image. Note which — but it isn't a gate on its own.

This stage is collaborative, not autonomous: tell the user plainly what you
found, then confirm the choices that are ambiguous or spend real compute — which
queue to run on, above all — before moving on.

## Route from here

The readiness gate is **launchable code + a queue with capacity**:

- **Missing either** → `setup.md` to establish only what's missing.
- **Both present** → `search.md` to run the search.
