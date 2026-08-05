<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# EvalTables

Reference for reasoning about W&B EvalTables: what they are, how to log them, how
they're viewed/compared, their limits, and the bounded Table-to-EvalTable
preview workflow.

## What an EvalTable is

An EvalTable is a typed successor to `wandb.Table`. A `wandb.Table` is a flat,
untyped grid of columns; an EvalTable groups its columns into three semantic
roles:

- **input columns** — example inputs / prompts / dataset fields
- **output columns** — the model's predictions / generated outputs
- **score columns** — evaluation results / metrics for each example

That grouping is what powers the dedicated **Evaluation Tables** panel: because
the product knows which columns are inputs vs outputs vs scores, it can line the
same examples up across runs and steps, aggregate the score columns, and surface
regressions — none of which a Table panel can do. A value logged under a
key whose type is `eval-table` routes to that panel automatically.

Rule of thumb: use an EvalTable when the data is "per-example evaluation results
you want to compare".

## Logging an EvalTable (Python)

`wandb.EvalTable` requires a W&B SDK release that exposes the API. Use an
explicit environment when needed:

```bash
uv run --with 'wandb[workspaces]>=0.28.1' python your_script.py
```

Older installs raise `AttributeError`.

Untyped — mirror a table's columns as-is:

```python
eval_table = wandb.EvalTable(columns=columns, data=rows)
run.log({"predictions_eval": eval_table})
```

Typed (preferred) — assign semantic column groups. The `data` rows must be
ordered input → output → score to match the column lists:

```python
eval_table = wandb.EvalTable(
    input_columns=["prompt"],
    output_columns=["completion"],
    score_columns=["exact_match", "score"],
    data=rows,  # each row ordered: prompt, completion, exact_match, score
)
run.log({"predictions_eval": eval_table}, step=step)  # step optional
```

When logging several tables at the same step, pass `commit=False` and commit on
the last one.

## Viewing & comparing

EvalTables render in the **Evaluation Tables** workspace panel (a compare view):

- **Compare across runs/steps** — the same examples align side by side; the panel
  seeds with 2 runs and compares up to 10.
- **Aggregate metrics** — score columns roll up to summary metrics (e.g. mean /
  fraction-true) per scorer.
- **Filter** — on input/output/score columns; filters persist in the panel
  config.
- **Drill into examples** — click a row to see the full input/output/scores for a
  single example.
- **Find regressions** — compare a run against a reference to surface where
  scores dropped (top-k regressions); sort score columns by the delta against the
  reference to rank examples by how much they moved.

Panel scale limits: up to 100 runs and 1,000 eval calls per panel, 50 rows per
page.

## Limits & gotchas

- **Size caps → truncation**: 10,000 rows and 100 columns per table. Larger
  tables are **truncated** to the caps (the preview keeps the leading 10,000 rows
  and leading 100 columns, dropping the trailing ones), not skipped. The scan
  reports the truncation up front — see **Truncating oversized tables** below.
- **Only the first 20 score columns are shown**: you can log more (they count
  toward the 100-column table cap alongside input/output columns), but the compare
  view only displays the first 20, so put the scores that matter most up front.
- **Row warning**: over 1,000 rows still works but may be slow or fail.
- **Media support is currently plain-only.** `wandb.Image`, `wandb.Audio`, and
  `wandb.Video` render in EvalTables, but only in their plain form — richer media
  (point clouds / `Object3D`, image annotations like masks, bounding boxes, and
  class labels, molecules, etc.) is not converted yet. Support for these is
  planned soon; until then those cells won't render with the rich extras.
- **Video support is optional.** Run the helper in an environment with
  `wandb[eval-table-video-support,workspaces]` when converting MoviePy-backed
  `wandb.Video` values. If video conversion fails, report the helper error.
- **A cell holding a list of media isn't handled by EvalTable yet** (planned
  soon). EvalTable converts one media value per cell with
  `wandb.integration.weave.media_adapters.unwrap_value(value, column, unsupported_media_mode="stub")`
  (Image→PIL, Audio→weave.Audio, Video→MoviePy, and a placeholder string for
  unsupported types), but it does not iterate a list. The preview helper works
  around this for you: it unwraps each media element of a list cell before
  logging (non-media elements are left untouched), so you don't hand-normalize
  cells. This is a temporary shim — it no-ops on SDKs without `media_adapters`
  and will be removed once the SDK converts list-of-media natively.
- **Preview keys are separate**: previews log under `<table_key>_preview`, so
  they never overwrite the original table's key (which stays in the same
  project).

## Preview helper

Use `skills/wandb-eval-tables/scripts/table_artifact_to_eval_table.py` to preview W&B Table
artifacts as EvalTables — it logs new EvalTables alongside the originals so users
can try the feature. Always run `scan` first and present a concise plan before any
write. Summarize the plan rather than pasting raw scan JSON.

`scan` plans across all keys in one call. `preview` converts **exactly one
`--table-key` per call**, so run it once per key: failures stay isolated (one
key failing doesn't block the others) and each call does less work. After
scanning, preview each key you want in its own call and report per-key results.

```bash
# Plan across all keys in one scan.
uv run python skills/wandb-eval-tables/scripts/table_artifact_to_eval_table.py scan \
  --source-workspace entity/project \
  --max-runs 4

# Convert one key per call — loop over the keys the user wants.
uv run python skills/wandb-eval-tables/scripts/table_artifact_to_eval_table.py preview \
  --source-workspace entity/project \
  --target-project entity/project \
  --max-runs 4 \
  --table-key predictions
```

- The helper defaults project/sweep scans to 4 recent runs and the latest 5
  table artifact versions per key/run, so the default plan can demonstrate both
  multiple runs and multiple recent steps. Pass `--run-id` to choose exact runs,
  `--max-steps-per-run-key 1` for latest-only behavior, another
  `--max-steps-per-run-key N` value for a bounded recent step history, or
  `--all-steps` only when the user explicitly wants every table artifact
  version.
- When using the default recent-step scope, tell the user that the plan previews
  the same table logged at up to 5 recent steps within a run, so they are not
  surprised by multiple previewed steps.
- The helper truncates tables over 10,000 rows or 100 columns to the caps (keeps
  the leading rows/columns) and reports the truncation in the scan; it still
  refuses to preview a single table key with more than 50 eligible table
  artifacts (the cap is per key — see **Reacting to the scan**).
- Tables over 1,000 rows stay eligible but include a warning that the preview may
  fail due to number of rows.
- `preview --dry-run` reuses the scan path and creates no runs.
- Destination defaults to the source project (the preview is meant to live
  alongside the original tables). Each EvalTable is logged under a
  `<table_key>_preview` key so it gets its own workspace panel and never clashes
  with the original table's key.
- Preview runs are tagged `eval-table-preview` and include source artifacts,
  table keys, source run, and source steps in config/summary metadata. Run names
  default to `<source run display name> preview`. Multiple selected table
  artifact versions for one source run are logged to that same preview run in
  source-step order, letting `run.log` assign consecutive destination steps.
- Existing previews are skipped unless `--force` is passed.

### Choosing the source

The helper accepts four source kinds. Pick the **narrowest that matches what the
user is asking to preview**; otherwise default to the workspace/project. Infer the
source from the request and locally available W&B context when the scope is clear.

- **A project / workspace (default):** `--source-workspace <entity/project>`. Scans
  the recent runs (or specific ones via `--run-id`) and previews their tables. Use
  this for "preview the tables in this project" and for previewing specific runs
  within a project — pass the baseline / pinned / chosen runs as `--run-id`
  (repeatable). This is the common case.
- **A single run:** `--source-run <run_id_or_name> --source-project <entity/project>`
  — when the user means just one run's table. (Equivalent to a workspace scan
  narrowed to one `--run-id`; prefer it only when the scope is explicitly one run.)
- **A sweep:** `--source-sweep <entity/project/sweep_id>` — to preview across a
  sweep's runs.
- **A specific table artifact/version:** pass the `run_table` artifact URL or
  `entity/project/name:version` as the positional source — when the user points at
  an exact artifact version.

Rule of thumb: `--source-workspace … --run-id …` covers "these runs in this
project"; reserve `--source-run`,
`--source-sweep`, and the artifact source for when the user's scope is explicitly a
lone run, a sweep, or a specific artifact.

### Choosing column groups

By default (no `--input-columns` / `--output-columns` / `--score-columns`) the
helper logs an untyped EvalTable that mirrors the table's columns. Prefer to
classify each column into one of three roles — **input**, **output**, or
**score** — inferring from the column naming; use the rules below to identify these
categories and does not need per-role examples. A score must be numeric or boolean
(it aggregates to a mean / fraction-true); a per-example free-text rationale or
label string is an output, not a score.

Pass the groups you're confident about, e.g. `--input-columns prompt
--output-columns completion --score-columns exact_match,score`. If you specify
any group, unmentioned columns default to outputs. Use
`--no-unspecified-columns-as-output` only when every column must be named
explicitly. When column roles are ambiguous, fall back to the untyped default.
You can type more than 20 score columns, but the compare view only shows the
first 20 — put the most important scores first so they aren't the ones dropped
from the display.

For a separately named row-order retry after a successful preview, pass every
successful original preview run from `created_runs` as a repeatable
`--from-preview-run` argument, plus `--force --row-index-match` and the same one
`--table-key`. This reuses the exact source artifact manifest saved on the
original preview runs instead of scanning the workspace again. Omit
`--input-columns`, but keep any safe numeric/boolean `--score-columns`; the
helper treats other columns as outputs. It logs the EvalTable and destination run with the
`_row_index_match` and ` row_index_match` suffixes respectively,
preserving the regular preview alongside it.

### Helper failures and verification

The helper is the only supported migration path for this flow. If
`table_artifact_to_eval_table.py preview` fails, you can change the helper inputs
and retry when the error points to a safe adjustment: narrow the source with
`--run-id`, change step selection, correct column groups, pass `--force`, or use
a writable target project. Do **not** bypass the helper by manually logging
EvalTables, creating runs/artifacts, or calling weave with other scripts. If the
helper still fails, report the helper error to the user.

`scan` is preflight only. A successful scan means the source was inspectable; it
does not mean an EvalTable was created or saved to run history.

After `preview` reports created runs, verify the last created run with the
helper before saying the conversion succeeded:

```bash
uv run python skills/wandb-eval-tables/scripts/table_artifact_to_eval_table.py verify-preview \
  --run entity/project/preview_run_id \
  --table-key val_predictions_preview
```

Pass only the final run path from `created_runs` with `--run`, and pass the exact
logged EvalTable key (`<table_key>_preview`) with `--table-key`.

`verify-preview` performs both backing checks:

1. Read the preview run history for the final run path returned by the helper in
   `created_runs`.
2. Find entries under the logged `<table_key>_preview` key where the value has
   `_type: "eval-table"` and an `evaluate_call_id`.
3. Treat those `evaluate_call_id` values as the root weave call IDs for the
   logged EvalTables, and confirm those calls exist in Weave.

Only report a successful conversion when `verify-preview` returns
`verified: true` for the final run path in `created_runs`. If it reports
`missing_history`, `missing_weave_calls`, or any other non-verified status, say
that the preview command completed but verification failed, include the verifier
status, and do not present the conversion as successful. Do not claim the preview
is verified from a created run path, a successful scan, or the absence of a
helper exception alone.

### Removing a preview (undo)

Previews are throwaway, so the helper can undo them. An EvalTable is weave-backed:
logging one creates a **weave evaluation** and records its call id in the preview
run's history, under the `<table_key>_preview` key, as
`{"_type": "eval-table", "evaluate_call_id": "<id>", ...}`. Fully removing a
preview therefore means two deletes: the weave evaluation (by that call id) and
the preview run itself.

Use the `delete-preview` subcommand — do not hand-delete. It scans each run's
history for those `evaluate_call_id`s, deletes the weave evaluations by id
(reliable — not by name), then deletes the run. It only touches runs this helper
created (tagged `eval-table-preview` or carrying the `eval_table_preview`
config); any other run is refused, never deleted. Pass the preview run paths the
preview step reported in `created_runs`:

```bash
# Preview by preview: delete the weave evaluation(s) + the run. Repeatable.
uv run python skills/wandb-eval-tables/scripts/table_artifact_to_eval_table.py delete-preview \
  --run entity/project/abc123 \
  --run entity/project/def456
```

- `--dry-run` reports the runs and `evaluate_call_id`s it would delete without
  deleting anything — confirm with the user before the real delete.
- `--delete-artifacts` also removes artifacts logged by the preview run (off by
  default; the EvalTable data lives in weave, not run artifacts).
- Deleting a weave call cascades to its children, so one call id per EvalTable is
  enough. A run with no eval-table history is still deleted (nothing to unlink).
- Deletion is irreversible; only run it after the user confirms, and only on the
  preview runs the preview step created — never the user's source runs.

### Reacting to the scan

- **Steps vs runs vs versions.** The scan reports `eligible_summary_by_table_key`
  with, per key, `runs` (distinct source runs), `steps_per_run` (distinct steps the
  key was logged at _within a run_ — one run can log a table at several steps), and
  `artifacts` (= `runs` × `steps_per_run`). In the plan table, **Steps = `steps_per_run`,
  Runs = `runs`**. Never report the artifact-version count as Steps: a key logged
  once in each of 3 runs is Steps 1, Runs 3 — not Steps 3.
- The 50-artifact batch cap is **per key** (preview converts one key per call).
  The scan reports `eligible_by_table_key` (per-key counts) and lists only the
  offenders in `keys_over_batch_limit`; `batch_limit_exceeded` is true only when
  some single key is over. For an over-cap key, narrow with `--run-id`,
  `--max-steps-per-run-key`, or a smaller `--max-runs`.
- **Strongly prefer previewing a single table key.** The preview exists to let
  the user see the feature and decide whether to convert the rest, so convert just
  one table by default — one is enough, and it keeps the preview fast. Do not
  convert every key in the section up front; preview additional keys (or widen the
  runs) only if the user explicitly asks.
- Surface warning tables (over 1,000 rows) from the scan only when they exist.

### Truncating oversized tables

Tables over 10,000 rows or 100 columns are **truncated** to the caps, not
skipped: the preview keeps the leading 10,000 rows and the leading 100 columns
and drops the trailing ones. The scan reports each cut in `truncated_count` and a
`truncations` list — one entry per table with its `table_key`, `source_run_name`,
`source_step`, and a `truncation` object holding `rows` (`kept`/`dropped`) and/or
`columns` (`kept`/`dropped` plus `dropped_columns`, the exact names being left
off).

When the scan reports any truncation, **tell the user as you proceed** (don't stop
to confirm): say the table exceeds the cap and will be truncated, give the
kept-vs-dropped counts, and — when columns are truncated — name the
`dropped_columns` that will be left off, so they can re-scope afterward if they
want (e.g. narrow to a specific `--table-key` or ask to keep different columns)
rather than silently losing data. Truncation preserves the original table, so it
is safe and reversible; it just bounds what the preview shows.

Column truncation **always keeps columns you explicitly assign a role**
(`--input-columns` / `--output-columns` / `--score-columns`), wherever they sit —
so if the user cares about a column the scan lists in `dropped_columns`, classify
it (e.g. as an output) and it is preserved; the remaining budget then fills with
the other columns in order. The scan's `dropped_columns` is a leading-columns-kept
estimate (column roles aren't known yet at scan time); the actual drop only
shrinks once roles pin columns, never grows.
