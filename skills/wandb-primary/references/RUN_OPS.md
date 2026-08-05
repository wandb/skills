<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# `wandb_run_ops.py` — Canonical Invocations

`R=skills/wandb-primary/scripts/wandb_run_ops.py`.

## Canonical grammar

Use these exact shapes first. The helper is intentionally command-scoped:
flags that work on one subcommand are not automatically valid on another.

```bash
python $R COMMAND --entity ENTITY --project PROJECT [command flags]
python $R large-project --entity ENTITY --project PROJECT --recipe RECIPE [recipe flags]
python $R COMMAND --help
```

Project scope is `--entity ENTITY --project PROJECT`. If those are omitted,
the helper reads `WANDB_ENTITY` and `WANDB_PROJECT`; examples pass the flags
explicitly so the project is unambiguous.

For `large-project`, the recipe is a flag, not a positional subcommand:

```bash
# Right
python $R large-project --entity ENTITY --project PROJECT --recipe status
python $R large-project --entity ENTITY --project PROJECT --recipe run-info --run RUN_NAME_OR_ID

# Wrong
python $R large-project status --entity ENTITY --project PROJECT
python $R large-project run-info --entity ENTITY --project PROJECT --run RUN_NAME_OR_ID
```

For order values that start with `-`, use the portable equals form
(`--order=-created_at`, `--order=-summary_metrics.accuracy`); the helper also
normalizes the common `--order -created_at` form. `python $R <subcommand>
--help` is the authoritative flag reference, and parser errors print a
canonical recovery example when a command-specific flag is rejected.

## Recipe table

| Need | Canonical command shape |
|---|---|
| count runs | `python $R count-runs --entity E --project P [--state finished]` |
| count by sweep/group | `python $R count-runs --entity E --project P --by-sweep` or `--by-group --include-ungrouped` |
| project overview | `python $R project-snapshot --entity E --project P --sample 12` |
| schema / metric / config keys | `python $R inspect-schema --entity E --project P --schema both --sample 50` |
| config value vocabulary | `python $R inspect-schema --entity E --project P --distinct-values CONFIG_KEY --sample 50` |
| run-name vocabulary | `python $R name-patterns --entity E --project P --sample 100` |
| rank runs by metric | `python $R rank-runs --entity E --project P --metric METRIC --direction max --limit 10` |
| top-1 run for a config value | `python $R rank-runs --entity E --project P --metric METRIC --direction max --limit 1 --config-eq CONFIG_KEY=VALUE --answer` |
| latest run | `python $R rank-runs --entity E --project P --latest --limit 1` |
| likely hparams for winner | `python $R rank-runs --entity E --project P --metric METRIC --direction min --limit 1 --show-hparams` |
| one run details | `python $R large-project --entity E --project P --recipe run-info --run RUN_NAME_OR_ID` |
| large project status | `python $R large-project --entity E --project P --recipe status` |
| recent/crashed rows | `python $R large-project --entity E --project P --recipe recent --sample 10` or `--recipe latest-crashed` |
| config/summary sample | `python $R large-project --entity E --project P --recipe run-config --config-key KEY --sample 20` |
| workspace views | `python $R workspace-views --entity E --project P` |
| history/stability | `python $R diagnose-history --entity E --project P --metric METRIC --sample 60 --history-samples 80` |
| full project scan in custom Python | `wandb_helpers.fetch_runs(get_api(), "E/P", metric_keys=[...], config_keys=[...], limit=N, jsonl_path="checkpoint.jsonl")` — see `scripts/wandb_helpers.py` |

## CLI flag grammar

Flags are command-scoped: use `python $R <subcommand> --help` before the first
call to a subcommand if the exact invocation is not shown here.

- Project scope is `--entity ENTITY --project PROJECT`, not `--path`.
- Use the documented subcommands below; there is no generic `vocab`,
  `metric-vocab`, or `history` command. Use `inspect-schema` for key
  vocabulary, `rank-runs`/`large-project --recipe run-info` for run identity,
  and `diagnose-history` or `stability-scan` for history questions.
- Most boolean switches work bare. Omit the flag for false. Only
  `--include-sweeps`, `--show-hparams`, and `--answer` tolerate explicit
  `true`/`false` values; `--exclude-sweeps` is a readable false form for
  `--include-sweeps`.
- `--include-sweeps` means include sweep runs in W&B run queries and counts.
- Most helpers print `answer=` automatically. `--answer` is advertised only for
  `rank-runs`; other helpers tolerate it as a no-op.
- `--show-hparams` prints likely hyperparameter keys. For exact config fields,
  repeat `--config-key KEY`; `--hparam KEY` is accepted as an alias.
- `--sample` is the usual requested scan cap, not a promise of exhaustive
  project coverage. Expensive helpers may lower it and will print
  `sample_scope=` plus `sample_caveat=` when that happens. `--limit` is accepted
  only where that subcommand help lists it.

## Output contract

- Helpers print machine-readable lines; `answer=` is the result. Companion keys:
  `interpretation_hint=`, `stop_rule=`, and `headline_outlier=` explain how to
  use the bounded evidence.
- `filter_caveat=`: the SDK result is grounded, but a separate aggregate index may count differently.
- `sample_scope=`: structured JSON showing `requested`, effective cap, actual
  rows/runs observed, strategy, and `project_exhaustive`. Treat sampled outputs
  as evidence only for that scope.
- `effective_scan_limit=`: structured JSON for commands with protective scan
  caps, showing requested vs effective run/sweep limits before results.
- `sample_caveat=`: an automatic cap reduced the requested sample — items absent
  from the sample are omitted, not proven absent project-wide.
- `limit_caveat=`: a row/run cap was reached; aggregates cover the scanned slice only.
- `artifact_scope=`: structured JSON showing whether artifact collections were
  actually checked. If `checked=false`, do not infer anything about artifacts.
- Empty saved workspace view + runs still exist in the project ⇒ the view's regex/run-name filter matches zero runs (check before state/group/runset filters).

## Scale characteristics

`project-snapshot` returns the total run count. Subcommands vary in how they
handle large projects:

- **`large-project` recipes** — most are cheap (single counts, one run by ID,
  aggregated tags). A few (`system-metrics`, `summary-stats`) fetch bounded run
  slices, so a high run-scan cap is slower. `duration-range` uses `--sample` as
  candidate depth for the shortest/longest edge search and hydrates only the
  first runtime-bearing edge runs.
- **`rank-runs`**, **`variant-analysis`**, **`config-diff`** — fetch up to
  their run-scan cap (default varies). Cost scales with the cap, not project size.
  Config-hydrating `rank-runs` and sampled `config-diff` auto-cap expensive
  hydration at 150 runs and print `sample_scope=`.
- **`metric-grid`**, **`diagnose-history`**, **`stability-scan`** — scan bounded
  run/history slices and print `sample_scope=`. High `--history-samples` values
  make history scans much slower, so deep stability scans cap run fanout, print
  `effective_scan_limit=`/`sample_caveat=`, and return timeout rows for runs
  that do not finish inside the command budget.
- **`count-runs`** — server-side count, fast regardless of project size.
- **Custom Python full scans** — when CLI caps are too low or you need every
  run: prefer `wandb_helpers.fetch_runs()` over hydrating `api.runs()`. Pass
  `jsonl_path=...` to append each page as JSONL and flush after every page;
  partial output survives shell timeout (exit 124). Aggregate from the JSONL in a
  second pass.

## Subcommands

### Project overview and counts

`project-snapshot` — samples recent and oldest runs, summarizes run states,
config examples, metric families, and run-name terms. Returns total run count.
On large projects it skips slow artifact collection enumeration and prints
`artifact_scope=` plus `artifact_collection_caveat=`; use `artifact-inventory`
or pass `--include-artifacts` when artifact collections are required.
`--sample N` means N recent plus N oldest runs. Expensive hydrated slices
auto-cap at 75 recent + 75 oldest and print `sample_scope=`/`sample_caveat=`.
`--limit N` is accepted as an alias for `--sample N`. `--include-weave` adds
Weave trace/eval counts.

`count-runs` — server-side lazy count. `--by-sweep` or `--by-group` breaks
down by sweep or group. `--include-ungrouped` includes runs outside any group.

```bash
python $R project-snapshot --entity E --project P --sample 12          # add --include-weave for Weave counts
python $R count-runs --entity E --project P --state finished           # exact lazy count; --by-sweep | --by-group --include-ungrouped
```

### Bounded metadata recipes

`large-project` — a collection of bounded recipes for metadata questions.
It is not the generic wrapper for all run questions: use `rank-runs`,
`inspect-schema`, `name-patterns`, `metric-grid`, table helpers, or history
helpers when those match the task better. Most recipes fetch a single count or
a small slice of runs. Each recipe accepts only the flags it uses; if a flag
belongs to a different recipe, choose that recipe or remove the flag.

| Need | Recipe | Extra flags |
|---|---|---|
| total/finished/crashed/crash-rate | `status` | none |
| newest/oldest/crashed/finished rows | `recent`, `finished`, `most-recent`, `oldest`, `earliest-finished`, `latest-crashed` | `--sample`/`--limit`, `--order` |
| one run identity | `run-info` | `--run RUN_NAME_OR_ID` (`--run-id` also accepted for exact W&B ids) |
| config/summary keys from sampled runs | `run-config`, `summary-stats`, `metric-count` | `--sample`, `--order`, `--state`, `--name-include`, repeat `--config-key` |
| compare two date/config profiles | `config-profiles` | `--left-month`, `--right-month`, repeat `--config-key` |
| duration or system telemetry presence | `duration-range`, `system-metrics` | `--sample`, `--history-samples` for system history |
| name-pattern overview | `name-pattern` | `--sample`, `--name-include`, `--name-prefix`, `--name-regex` |
| tags/groups | `tags`, `groups`, `tag-counts` | `--tag`, `--group`, `--include-ungrouped` where applicable |
| date windows | `date-counts` | `--month` or `--start`/`--end` |

```bash
python $R large-project --entity E --project P --recipe status
python $R large-project --entity E --project P --recipe tag-counts --tag TAG
python $R large-project --entity E --project P --recipe date-counts --month YYYY-MM
python $R large-project --entity E --project P --recipe run-info --run RUN_NAME_OR_ID    # display name, W&B id, or unique visible suffix
python $R large-project --entity E --project P --recipe duration-range --sample 500 # shortest/longest candidate depth, not rows printed
python $R large-project --entity E --project P --recipe run-config --name-include TOKEN --config-key KEY --sample 20
```

### Schema discovery

`inspect-schema` — discovers config keys and summary metric keys from a
bounded, non-exhaustive sample of hydrated runs. Default `--sample` is 50 so
overview scans catch more project-specific keys; requests above 150 auto-cap at
150 hydrated runs and print `sample_scope=`/`sample_caveat=`. Keys absent from
the sample are not proven absent project-wide. `--limit` is accepted as an alias
for `--sample`; use `--order=VALUE` when the value starts with `-`.

`summary_non_numeric_keys` flags keys whose sampled values aren't rankable
scalars, with their type(s). Never `--metric`-sort by these; they yield NaN.

```bash
python $R inspect-schema --entity E --project P --schema both --sample 50 --order=-created_at
python $R inspect-schema --entity E --project P --distinct-values CONFIG_KEY --sample 50
```

### Run names

`name-patterns` — discovers naming conventions from sampled runs. Prints
`recommended_regex=` for use in subsequent filters.

Name filtering is standardized where possible:

| Task | Use |
|---|---|
| discover prefixes/regexes | `name-patterns --name-prefix PREFIX` or `--name-regex REGEX` |
| filter metric/history/stability scans | `--name-include TOKEN` / `--name-exclude TOKEN` |
| summary-table prefix narrowing | `--name-prefix PREFIX` |
| checkpoint path lookup | `--name-include TOKEN` |
| cohort by name regex | `--cohort label=name_regex:REGEX` |

```bash
python $R name-patterns --entity E --project P --name-prefix PREFIX --sample 40
```

### Rankings and lookups

`rank-runs` — returns runs sorted by a metric or recency. Each result includes
summary metrics and optionally config. Metric/latest ranking without config uses
the requested `--limit` via lazy/selective rows. If `--show-hparams`/`--hparams`
or `--config-key` requires config hydration, expensive hydration auto-caps at
150 rows and prints `sample_scope=`/`sample_caveat=`. `--show-hparams` is a bare
switch that prints likely hyperparameter keys; for named fields, repeat
`--config-key KEY` and `--summary-key KEY`. `--metric-hint` resolves a family
name (e.g. "loss") to the exact summary key from the `--sample` hint-resolution
sample, so prefer `--metric` when the exact key is known. Direction can be explicit:
`--direction max|min`, `--maximize`/`--higher-is-better`, or
`--minimize`/`--lower-is-better`. Repeat `--config-eq KEY=VALUE` to apply
server-side W&B config filters before ranking (use this for top-1 per dataset or
sweep axis value instead of scanning the whole project).

```bash
python $R rank-runs --entity E --project P --state finished --metric METRIC --direction max --limit 1 --answer --show-hparams
python $R rank-runs --entity E --project P --state finished --metric METRIC --direction max --limit 1 --answer --config-key CONFIG_KEY --summary-key SUMMARY_KEY
python $R rank-runs --entity E --project P --state finished --metric METRIC --direction max --limit 1 --config-eq CONFIG_KEY=VALUE --answer
python $R rank-runs --entity E --project P --latest --limit 1 --answer
```

### Comparisons

`variant-analysis` — groups runs by explicit variant specs or a regex pattern,
then compares metric performance across groups. Pass at least one selector:
`--variant NAME`, `--variant NAME=REGEX`, or `--pattern REGEX`. Use
`--config-key KEY` when the variant lives in config text. Returns per-group best,
mean, and count from `run.summary_metrics`. `--metric-hint` is resolved from the
`--sample` hint-resolution sample; pass `--metric` for a known exact key.

`compare-cohorts` — compares two named cohorts (e.g. sweep vs non-sweep) on a
metric. Cohorts use `NAME=SELECTOR`; selectors are `non_sweep`, `sweep:any`,
`sweep:<id>`, `tag:<tag>`, `group:<group>`, `name_regex:<regex>`,
`run_name:<regex>`, or `name:<regex>`. Picks the single best run per cohort.
Also reports step counts (steps_min/median/max) per cohort. `--metric-hint` is
resolved from the `--sample` hint-resolution sample; pass `--metric` for a known
exact key. `--scan` is the requested per-cohort run scan and auto-caps at 40
unless `--large-scan` is passed. `sweep:any` also caps `--sweeps` at 50 unless
`--large-scan` is passed. Output includes `effective_scan_limit=`,
`sample_scope=`, and `limit_caveat=` when a cap changes the request. Python-side
selectors such as `non_sweep` and `name_regex:` only see the effective scan
slice, not the entire project.

`config-diff` — shows config keys that vary across a set of runs and which are
constant. Without explicit run refs, it samples hydrated runs, auto-caps at 150,
and prints `sample_scope=`/`sample_caveat=`; differences are only across that
sample. For a specific pair or small set of runs, pass `--run-a A --run-b B`,
repeat `--run RUN`, or pass `--runs A B ...`; explicit refs are not auto-capped.

```bash
python $R variant-analysis --entity E --project P --metric METRIC --variant A --variant B   # --config-key KEY if axis is in config;
                                                                                # --rank-by balanced_mean --stratify-by KEY
python $R compare-cohorts --entity E --project P --metric-hint FAMILY --cohort baseline=non_sweep --cohort sweep=sweep:any --scan 20
python $R config-diff --entity E --project P --state finished --sample 100
python $R config-diff --entity E --project P --run-a RUN_A --run-b RUN_B
```

### Hyperparameter grids

`metric-grid` — cross-tabulates config axis values against summary metrics.
For each unique value of a config axis (e.g. `model_dim=128`), reports the
best and mean metric across runs sharing that value, plus the winning run ID.
Also produces axis-pair summaries (two axes x metric). Reads final summary
metrics, not training history. When both `--axis` and `--metric` are explicit,
it uses selective run-table fields and honors the requested `--sample` row cap,
printing `limit_caveat=` if more rows may exist. Auto-discovery mode hydrates
runs to infer axes/metrics, auto-caps at 150, and prints
`sample_scope=`/`sample_caveat=`.

```bash
python $R metric-grid --entity E --project P --metric-hint FAMILY --axis CONFIG_AXIS --sample 200   # repeat --axis; --config-eq K=V
```

### Tables

`summary-table` — reads W&B table files stored in `run.summary`. `--mode
success` computes success rates from `--success-column` and optional
`--group-column`; `--mode snapshot` prints matching rows from capacity/benchmark
tables using `--table-hint`, `--label-hint`, `--metric-hint`, `--sort-hint`, and
`--row-limit`. In either mode, `--name-prefix PREFIX` first narrows the sampled
runs before table detection. Wrong-mode flags are rejected before any W&B calls.

```bash
python $R summary-table --entity E --project P --summary-key KEY --sample 5
python $R summary-table --entity E --project P --mode snapshot --table-hint HINT --metric-hint FAMILY --sort-hint HINT
```

### Artifacts and checkpoints

Canonical Artifact ops guidance (list/filter/inspect): `$refs/ARTIFACT_OPS.md`
and `scripts/artifact_helpers.py` (`versions`).

`artifact-inventory` — lists artifact types, collection counts, and bounded
version totals. Version counts are explicitly scoped as first 50 versions per
collection when counted; use `--version-mode none` to skip them.

`artifact-file-summary` — shows artifact metadata, manifest entries, file
summaries, row counts, and sampled rows.

`checkpoint-locations` — finds path-like config keys and logged checkpoint
artifacts for runs matching a name filter.

```bash
python $R artifact-inventory --entity E --project P
python $R artifact-file-summary --entity E --project P --artifact NAME:VERSION    # or --collection-hint HINT --type TYPE;
                                                                      # --file-regex RE --max-files N --sample-rows N
python $R checkpoint-locations --entity E --project P --run-name-include TOKEN
```

### Training history and stability

`diagnose-history` — analyzes training curves for outliers, anomalies, and
trends. Outlier mode uses `--metric`, `--metric-hint`, `--target-run`, and
`--layer-metrics`. `--mode gpu` checks GPU/system utilization from the system
stream. In outlier mode, `--metric-hint` is resolved from sampled run summaries;
pass `--metric` for a known exact history key. Use `--sample` for the number of
runs to inspect and `--history-samples` for points pulled from each run's history. In GPU mode,
`--training-like` drops obvious eval/toy/data-prep runs,
`--min-points-for-verdict` requires enough system samples before classifying a
run, and `--good-threshold` is the mean GPU-utilization percent considered
healthy. Wrong-mode flags are rejected before any W&B calls.

`stability-scan` — checks for spikes, NaNs, divergence, and gradient-norm
extrema across runs. Inspects objective losses, component losses, regularizer
losses, and gradient norms together. It requests history for selected sampled
runs only, prints `effective_scan_limit=`/`sample_scope=`, and caps run fanout
at 40 normally or 18 when `--history-samples >= 1000`; pass `--large-scan` only
when the full requested run count is worth the API time. Spike counts and
stability rows are evidence only for `selected_runs`. The command uses up to
`--history-budget-seconds` seconds for per-run history probes and emits
`history_timeout` rows for unfinished runs. The slower per-key
`run.scan_history` fallback is off by default; pass `--scan-history-fallback`
only when sampled `run.history` misses known keys and the extra cost is useful.

`gradient-noise-summary` — computes noise-to-signal ratios and signal/noise
power across sampled runs. It hydrates run summaries to find grad-noise metrics,
auto-caps at 150 runs, and prints `sample_scope=`/`sample_caveat=`. The
`gradient_noise_runs=` count is the subset of sampled runs that actually had
grad-noise metrics.

```bash
python $R diagnose-history --entity E --project P --state finished --metric METRIC --sample 60 --history-samples 80 --workers 12
python $R diagnose-history --entity E --project P --mode gpu --training-like --min-points-for-verdict 50 --history-samples 300
python $R diagnose-history --entity E --project P --target-run RUN --layer-metrics
python $R stability-scan --entity E --project P --state finished --sample 40 --history-samples 120 --history-budget-seconds 55
python $R gradient-noise-summary --entity E --project P --state finished --sample 120
```

### Workspace views

`workspace-views` — lists saved views with their sections, visible panel
counts, and open/pinned state.

```bash
python $R workspace-views --entity E --project P
python $R workspace-views --view-url 'https://wandb.ai/E/P?nw=TOKEN'
```
