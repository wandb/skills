<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B Sweeps Hyperparameter Optimization Product

## Concepts
- Hyperparameter optimization: optimize a blackbox F(x) → Y function to find the optimal set of inputs.
A specialized machine learning task where evaluating predictions (the function) is very expensive
- Sweep: Defines a hyperparameter optimization problem including the parameter space, the metric goal,
and the instructions to run training on agents
- Scheduler: A unit that produces the next run suggestions for the sweep, by observing the sweep runs
  and metrics. Is handled by the sweep backend or the SDK CLI. Also handles early termination.
- Agent: A polling unit of execution for a sweep, implemented in wandb/wandb SDK, that resolves the
function output for a point in the parameter space. Periodically communicates state and logs with the
wandb backend and receives new commands.
- Run: A wandb run. The metric goal must be logged through wandb.log() and the wandb agent
must log the input parameters to the run's config.
- Method: The sampling method. wandb natively supports exhaustive search through grid and
random search methods. Bayesian optimization is available through the bayes method
- Early Termination: Commonly known as pruning. Each sweep run may be expensive,
users often do not want to run poorly performing runs to completion. Hyperband stopping
is the only available option, under the early_terminate: config key.

## Sweep Mutations
The `deleteSweep(s)` GQL mutation
is used to delete a sweep. You **MUST** confirm with the user whether they want the associated
runs deleted with the sweep.

- Create a sweep: use `create_sweep` helper. The `program:` config file with the training script
should simply be presented and copied by the user into their environment. Alternatively, you may
consult with the user to select a launch job for sweeps on launch.
- Update a sweep: use `update_sweep` helper to change the sweep config or prior runs.
Note: this is restricted to sweeps in PENDING state and will fail if the user concurrently
starts an agent or logs a run.
- Clone a sweep: Call `copy_sweep` helper to create a copy of the sweep with the same config
and existing runs as prior runs. This is the only option "update" a sweep that is running:
clone the sweep and stop the old one.
- Pause, Stop, Resume, Cancel a sweep: Call `update_sweep_state`. Stop
and Cancel are dangerous as they instruct active agents to exit.

**Avoid destructive actions**: Do not stop, cancel, or delete a sweep without
explicit consent from the user. Obtain consent for each unique operation.
Display the exact operation and impacted sweep IDs, and ensure
the user gets a turn to confirm the operation.

## Agent Mutations
Agents do not have explicit mutations, instead their lifecycle is entirely managed by the
`wandb agent` CLI, and can only be started and stopped by the user.

## Sweep Quirks
A sweep will remain in RUNNING state even if all agents have exited. Similarly, a sweep
may become FINISHED even if there are active runs RUNNING to completion. Do not assume
the progress of runs (running or failed) from the sweep state.

A sweep may attach prior runs: pre-existing project runs attached at creation (e.g. by
`copy_sweep`) that feed the optimizer historical observations but never become child runs
owned by the sweep. **Prior runs are invisible to sweep-scoped run queries**:
`--sweep SWEEP_ID` on `$R` commands and `filters={"sweep": ...}` match only child runs, so
they must be queried separately — read the run names from the sweep's `priorRuns` GraphQL
field, then fetch those runs by name at the project level.

Analyze prior runs with care. Their existing configs and logged metrics may not match the
new sweep parameters and objective metric. Warning: a mismatch can cause sweep UI panels
to look sparse or mutually inconsistent, since some read the workspace's run selection
while others use the sweep relationship. Prior runs may produce inaccurate parameter
importance values.

## Sweeps on launch

`wandb launch-sweep` runs a sweep's agents as W&B Launch jobs instead of a user-managed `wandb agent`
process. Its sweep config needs a top-level `job` (or `image_uri`) naming
the Launch job to run per trial, alongside the usual `method`/`metric`/`parameters`:

```yaml
method: bayes
metric: {name: val_loss, goal: minimize}
job: my-entity/my-project/train-job:latest
```

See `$refs/LAUNCH.md` for queue setup and job creation.

## Sweep Scheduler CLI
`wandb sweep-scheduler` runs a sweep's scheduler loop locally, available on SDK v0.31.
The sweep must already have been created  with `scheduler: engine: wandb/ax/optuna`. Users may provide their own
customizations on top of third party  libraries by using the `scheduler: optimizer:` and `scheduler: search_space:`
config keys. These are function names available in the `scheduler: source:` file, similar to the `program:` file.
Logs will be written to the sweep logs.

Each engine requires extras on the `wandb` SDK (eg. `wandb[optuna]` for Optuna). If the main environment's `wandb` was
installed without them, a separate copy can live in its own venv
(`uv venv`, then `uv pip install --python <venv>/bin/python "wandb[optuna]==<version>"`),
and its `<venv>/bin/wandb` runs the scheduler.

`wandb sweep-scheduler` can be easily stopped and restarted again, and it will rebuild its state from the sweep.

## Debugging
A sweep may have the following failure modes

**training script failure**: A training script may have bugs, and may fail on all or specific run configs.
This results in CRASHED or FAILED runs. The sweeps engine periodically checks for pathological failures
and may transition the sweep to a FLAPPING state to prevent wasted compute on a buggy training script.

Users are expected to retry failed runs by deleting the run, and it will be eventually re-generated.

**transient agent failure**: An agent may be shutdown by the host (eg. Kubernetes). All in-progress runs
are preempted through the `mark_preempting()` API call. These runs are prioritized for execution by
the next available agent.

**backend failure**: The sweeps backend may fail to generate new runs for the sweep, particularly
for large sweeps, either from an exception or a timeout. These errors will not be surfaced to the user,
but manifest as agents receiving no commands for extended periods of time.

### Logs
The sweep has logs that may be viewed through the API by reading the controllerRunName from the
`Api().Sweep` object, and then reading the entire Run's output.log filestream. These logs contain
the run generation by the backend, agent state, and run assignment to agents.

The agent does not have logs, but issues with the agent surface through individual run logs. The list
of runs for a specific agent are available in the `Api().Sweep.agent().runs()` call.

Refer to `$refs/RUN_LOGS.md` for reading these log streams.

## Analysis

### UI
The sweep can be visualized using specific panels in the UI
- Metric goal vs. step number as a line plot, over all runs (intermediate values plot)
- Metric goal vs run number as a scatter plot, to understand per-run improvement
(optimization history plot)
- Individual scatter plots of metric goal vs. individual parameters, colored by run
- Contour plots of pairs of parameters against the metric contour (needs plotly)
- Parallel coordinates plot to manually analyze top clusters in the parameter space
- Parameter importance plot calculates feature importance over a RandomForest, and
the correlation of each parameter to the goal metric.

### API

The sweep config may be read though the `Api().Sweep` class attributes.
The following attributes are available: id, name, displayName, method, state,
description, bestLoss, config, createdAt, updatedAt, runCount, runCountExpected.

`best_run()` is available as a convenience function on the `Api().Sweep` class to return the current
best run.

The parameter importance score and correlation score is available with the `parameter_importance` helper

### Sweep-scoped run_ops commands

These `wandb_run_ops.py` subcommands exist for sweep-related questions. See
`$refs/RUN_OPS.md` for flags and examples.

**`aggregate-named-sweeps`** — Given explicit sweep display names and metric keys,
returns one row per sweep: `metric_stats` (mean/stdev over numeric summary
values in the fetched sample) and `state_counts` from the same run rows.
Resolves sweeps by direct lookup or a filtered `$in` query on `name` /
`config.name` — not a full-project sweep enumeration. One projected
`fetch_runs` per sweep (default `--limit` 500 runs per sweep). Honors common
`--state` (e.g. `finished`). Output includes `state_counts_scope:
"fetched_sample"`; counts and metric stats cover the fetched slice only, not
necessarily the full sweep. Non-numeric summary values are skipped
(`non_numeric_skipped` when applicable).

**`rank-runs --config-eq`** — Ranks runs by a summary metric with a config
filter (`KEY=VALUE`). Operates on runs, not sweep objects.

**`compare-cohorts`** — Compares two cohorts defined by selectors (`non_sweep`,
`sweep:any`, `name_regex:`, etc.) on a metric. Cohort definitions and scan caps
are in `RUN_OPS.md`.

**`count-runs --by-sweep`** — Lazy per-sweep run counts while enumerating sweeps
in the project (default `--sweeps` 50). Does not accept an arbitrary list of
user-provided sweep names. Exact totals for a known sweep id: `count-runs` with
`--sweep SWEEP_ID`.

### Scoping commands to one sweep

**Analyzing patterns across a sweep's runs.** See `$refs/RUN_OPS.md` for the full command reference. Add
`--sweep SWEEP_ID` to scope any `$R` (`wandb_run_ops.py`) command that samples runs to just this
sweep's runs. **Prefer these helpers over direct use of runs queries from the wandb SDK**

For custom Python beyond what these commands cover, `wandb_helpers.fetch_runs(get_api(), path,
metric_keys=[...], filters={"sweep": sweep_id})` fetches the same sweep-scoped run set efficiently.
It should be used instead of plain `.runs()` calls from the Api.
