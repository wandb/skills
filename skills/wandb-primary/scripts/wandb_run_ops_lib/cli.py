# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Argparse surface for the W&B run operation helpers."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Callable

from . import core
from .common import COHORT_NAME_SELECTOR_PREFIXES, COHORT_SELECTOR_HELP

DESCRIPTION = "Low-latency W&B run-table, artifact, history, and project-inspection helpers."
TOP_LEVEL_EPILOG = """\
Canonical grammar:
  python $R COMMAND --entity ENTITY --project PROJECT [command flags]
  python $R large-project --entity ENTITY --project PROJECT --recipe RECIPE [recipe flags]
  python $R COMMAND --help

Common recipes:
  count-runs      --state finished
  inspect-schema --schema both --sample 50
  name-patterns  --sample 100
  rank-runs      --metric METRIC --direction max --limit 10
  large-project  --recipe status

Use the exact subcommand help before first use; flags are command-scoped.
"""
ArgSpec = tuple[tuple[str, ...], dict[str, Any]]


def _arg(*flags: str, **kwargs: Any) -> ArgSpec:
    return flags, kwargs


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def percent_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def bool_value(value: str | bool) -> bool:
    """Parse explicit boolean values while preserving bare switch behavior."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def _split_values(values: list[str] | None) -> list[str]:
    """Split repeated CLI values and comma lists into non-empty tokens."""
    if not values:
        return []
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    return tokens


def _command_help(command: str) -> str:
    return f"Run: python $R {command} --help"


COMMAND_EPILOGS = {
    "count-runs": """\
Canonical examples:
  python $R count-runs --entity ENTITY --project PROJECT
  python $R count-runs --entity ENTITY --project PROJECT --state finished
  python $R count-runs --entity ENTITY --project PROJECT --by-sweep
""",
    "inspect-schema": """\
Canonical examples:
  python $R inspect-schema --entity ENTITY --project PROJECT --schema both --sample 50
  python $R inspect-schema --entity ENTITY --project PROJECT --schema summary --order=-created_at
  python $R inspect-schema --entity ENTITY --project PROJECT --distinct-values CONFIG_KEY --sample 50
Use rank-runs --show-hparams for likely hyperparameter keys.
""",
    "name-patterns": """\
Canonical examples:
  python $R name-patterns --entity ENTITY --project PROJECT --sample 100
  python $R name-patterns --entity ENTITY --project PROJECT --name-prefix PREFIX
  python $R name-patterns --entity ENTITY --project PROJECT --name-regex REGEX
""",
    "rank-runs": """\
Canonical examples:
  python $R rank-runs --entity ENTITY --project PROJECT --metric METRIC --direction max --limit 10
  python $R rank-runs --entity ENTITY --project PROJECT --latest --limit 5
  python $R rank-runs --entity ENTITY --project PROJECT --metric METRIC --direction min --limit 1 --show-hparams
  python $R rank-runs --entity ENTITY --project PROJECT --metric METRIC --direction max --limit 1 --config-eq CONFIG_KEY=VALUE --answer
Use --direction max|min, --limit for rows, and --sample for metric-hint discovery.
Repeat --config-eq for multiple scalar config filters (server-side).
""",
    "workspace-views": """\
Canonical examples:
  python $R workspace-views --entity ENTITY --project PROJECT
  python $R workspace-views --view-url 'https://wandb.ai/ENTITY/PROJECT?nw=TOKEN'
""",
    "large-project": """\
Canonical examples:
  python $R large-project --entity ENTITY --project PROJECT --recipe status
  python $R large-project --entity ENTITY --project PROJECT --recipe run-info --run RUN_NAME_OR_ID
  python $R large-project --entity ENTITY --project PROJECT --recipe recent --sample 10
Wrong: python $R large-project status --entity ENTITY --project PROJECT
Right: python $R large-project --entity ENTITY --project PROJECT --recipe status
""",
}


def _command_error_hint(command: str | None) -> str:
    if not command:
        return (
            "Use: python $R COMMAND --entity ENTITY --project PROJECT [command flags]. "
            "Run: python $R --help"
        )
    examples = COMMAND_EPILOGS.get(command)
    if examples:
        return examples.strip()
    return _command_help(command)


COMMON = [
    _arg("--entity", help="W&B entity. If omitted, read WANDB_ENTITY from the environment."),
    _arg("--project", help="W&B project. If omitted, read WANDB_PROJECT from the environment."),
    _arg("--state", help="Optional run state filter, such as finished, crashed, or running."),
    _arg("--include-sweeps", nargs="?", const=True, default=False, type=bool_value, help="Include sweep runs in W&B run queries and counts."),
    _arg("--exclude-sweeps", action="store_false", dest="include_sweeps", default=argparse.SUPPRESS, help="Exclude sweep runs from W&B run queries and counts."),
]

OPS = "status recent finished most-recent oldest earliest-finished latest-crashed run-info creator name-pattern project-description run-config config-profiles summary-stats metric-count duration-range system-metrics tags groups tag-counts date-counts".split()

COMMANDS: list[tuple[str, str, Callable[[argparse.Namespace], None], list[ArgSpec]]] = [
    ("count-runs", "Count runs quickly, optionally by sweep or group.", core.cmd_count_runs, [
        _arg("--by-sweep", action="store_true", help="Print counts for each sampled sweep."),
        _arg("--sweeps", type=positive_int, default=50, help="Maximum sweeps to inspect."),
        _arg("--by-group", action="store_true", help="Print counts for groups found in a bounded sample."),
        _arg("--group", action="append", default=[], help="Explicit group to count; repeat as needed."),
        _arg("--group-sample", type=positive_int, default=500, help="Runs per discovery slice for --by-group; default samples newest 500 and oldest 500."),
        _arg("--include-ungrouped", action="store_true", help="Also count runs without a group."),
        _arg("--order", help="Optional single W&B order for group discovery; default samples recent and oldest slices."),
    ]),
    ("artifact-inventory", "List artifact types, collection counts, and bounded version totals.", core.cmd_artifact_inventory, [
        _arg("--type", action="append", default=[], help="Artifact type to include; repeat as needed."),
        _arg("--per-page", type=positive_int, default=1000, help="Artifact collections to request per page."),
        _arg("--version-mode", choices=["auto", "all", "none"], default="auto", help="Whether to count artifact versions."),
        _arg("--max-version-collections", type=positive_int, default=1000, help="Auto mode version-count threshold."),
        _arg("--workers", type=positive_int, default=16, help="Parallel version-count workers."),
    ]),
    ("artifact-file-summary", "Summarize artifact metadata, provenance, bounded files, and compact field counts for small structured sidecar files.", core.cmd_artifact_file_summary, [
        _arg("--artifact", help="Artifact ref or collection name."),
        _arg("--type", help="Artifact type filter."),
        _arg("--collection-hint", help="Substring used to find a collection when --artifact is omitted."),
        _arg("--file-regex", help="Only summarize downloaded files whose relative path matches this regex."),
        _arg("--max-files", type=positive_int, default=30, help="Maximum manifest entries and downloaded files to summarize."),
        _arg("--sample-rows", type=non_negative_int, default=2, help="Rows to sample from JSONL/CSV files."),
    ]),
    ("checkpoint-locations", "Inspect path-like config keys and logged checkpoint artifacts.", core.cmd_checkpoint_locations, [
        _arg("--run-id", help="Inspect one run id."),
        _arg("--name-include", "--run-name-include", action="append", default=[], help="Keep runs whose names contain this token; repeat as needed."),
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=50, help="Maximum runs to scan."),
        _arg("--run-limit", type=positive_int, default=3, help="Runs to print."),
        _arg("--artifact-limit", type=positive_int, default=20, help="Logged artifacts to inspect per run."),
        _arg("--order", default="-created_at", help="W&B run order for scan."),
    ]),
    ("project-snapshot", "Summarize recent/oldest sampled runs, optional artifacts, run states, configs, metrics, and optional Weave stats.", core.cmd_project_snapshot, [
        _arg("--sample", "--limit", dest="sample", type=positive_int, default=12, help="Requested recent and oldest runs per side; expensive hydrated slices auto-cap at 75 per side and print sample_scope/sample_caveat."),
        _arg("--sample-rows", type=non_negative_int, default=0, help="Example run rows to print."),
        _arg("--artifact-per-page", type=positive_int, default=1000, help="Artifact collections to request only when artifact inspection is enabled."),
        _arg("--include-artifacts", action="store_true", help="Inspect artifact collections even for large projects; otherwise snapshots over 10000 runs skip artifacts and print artifact_scope."),
        _arg("--include-weave", action="store_true", help="Include optional Weave call/object counts when available."),
    ]),
    ("workspace-views", "Print a compact Markdown inventory of W&B workspace views, sections, visible panel counts, open/pinned state, and panel/chart names.", core.cmd_workspace_views, [
        _arg("--view-url", help="Specific workspace view URL containing an nw query token, such as ?nw=abc123. If omitted, inventory all workspace views for --entity/--project."),
    ]),
    ("metric-grid", "Compare metric values across config axes for hparam and tradeoff questions.", core.cmd_metric_grid, [
        _arg("--metric", action="append", default=[], help="Exact summary metric; repeat as needed."),
        _arg("--metric-hint", action="append", default=[], help="Metric family hint resolved from sampled summary keys, such as accuracy, loss, reward, or latency."),
        _arg("--axis", action="append", default=[], help="Config key to group by; repeat for multiple axes."),
        _arg("--max-axes", type=positive_int, default=6, help="Auto-selected config axes to consider."),
        _arg("--metric-limit", type=positive_int, default=12, help="Auto-selected metrics to consider."),
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=200, help="Requested runs to scan; auto-discovery hydration caps at 150 and prints sample_scope, while explicit --axis+--metric uses the requested selective row limit."),
        _arg("--row-limit", type=positive_int, default=80, help="Maximum summary rows to print."),
        _arg("--order", default="+created_at", help="W&B run order."),
        _arg("--direction", choices=["auto", "min", "max"], default="auto", help="Whether lower or higher metric values are better."),
        _arg("--name-include", action="append", default=[], help="Keep only runs whose names contain this token."),
        _arg("--name-exclude", action="append", default=[], help="Drop runs whose names contain this token."),
        _arg("--config-eq", action="append", default=[], help="Keep runs with scalar config KEY=VALUE; repeat as needed."),
    ]),
    ("inspect-schema", "Discover config and summary keys from a bounded, non-exhaustive run sample.", core.cmd_inspect_schema, [
        _arg("--schema", choices=["config", "summary", "both"], default="config", help="Key namespace to inspect."),
        _arg("--distinct-values", action="append", default=[], metavar="KEY", help="Enumerate distinct scalar config values for KEY from the sampled runs; repeat for multiple keys."),
        _arg("--sample", "--limit", dest="sample", type=positive_int, default=50, help="Requested hydrated runs to sample for key discovery; auto-caps at 150 and prints sample_scope/sample_caveat."),
        _arg("--order", default="-created_at", help="W&B run order."),
    ]),
    ("name-patterns", "Inspect run-name prefixes, regex matches, and likely naming schemes.", core.cmd_name_patterns, [
        _arg("--name-prefix", "--prefix", dest="prefix", help="Run-name prefix to count."),
        _arg("--name-regex", "--pattern", dest="pattern", help="Regex to match against display names."),
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=100, help="Runs to sample."),
        _arg("--limit-matches", type=positive_int, default=20, help="Matching names to print."),
        _arg("--order", default="+created_at", help="W&B run order."),
        _arg("--show-sample", action="store_true", help="Print sampled names."),
    ]),
    ("variant-analysis", "Group runs by explicit variant tokens or regex and rank variants by metric.", core.cmd_variant_analysis, [
        _arg("--variant", action="append", default=[], help="Variant token or NAME=REGEX; repeat for each variant."),
        _arg("--pattern", help="Regex used to extract variants from run names/config; first group wins."),
        _arg("--metric", help="Exact summary metric to compare."),
        _arg("--metric-hint", help="Metric family hint to resolve from sampled runs."),
        _arg("--direction", choices=["auto", "min", "max"], default="auto", help="Whether lower or higher is better."),
        _arg("--rank-by", choices=["mean", "median", "balanced_mean", "best", "count"], default="mean", help="Statistic used to rank variants when a metric is provided."),
        _arg("--config-key", action="append", default=[], help="Config key text used for variant matching."),
        _arg("--stratify-by", action="append", default=[], help="Config key used to report balanced per-stratum aggregates, e.g. env.name."),
        _arg("--limit", type=positive_int, default=1000, help="Runs to scan for exact aggregate comparisons."),
        _arg("--sample", type=positive_int, default=80, help="Runs sampled to resolve --metric-hint; hint resolution is not exhaustive."),
        _arg("--order", help="W&B run order; metric sorts are inferred when possible."),
        _arg("--top", type=positive_int, default=20, help="Variant rows to print."),
        _arg("--fast-best", action="store_true", help="Stop after every explicit variant has a metric-sorted match."),
    ]),
    ("rank-runs", "Return latest runs, best metric runs, or likely hparams for winners.", core.cmd_rank_runs, [
        _arg("--latest", action="store_true", help="Order by newest created_at."),
        _arg("--order", help="Explicit W&B order, e.g. -created_at or -summary_metrics.accuracy."),
        _arg("--metric", help="Exact summary metric to rank."),
        _arg("--metric-hint", help="Metric family hint to resolve from sampled runs."),
        _arg("--direction", choices=["auto", "min", "max"], default="auto", help="Whether lower or higher metric values are better when ranking by metric."),
        _arg("--maximize", "--higher-is-better", action="store_true", dest="maximize", help="Alias for --direction max."),
        _arg("--minimize", "--lower-is-better", action="store_true", dest="minimize", help="Alias for --direction min."),
        _arg("--sample", type=positive_int, default=100, help="Runs sampled to resolve --metric-hint; hint resolution is not exhaustive."),
        _arg("--limit", type=positive_int, default=3, help="Requested ranked rows to print; config/hparam hydration auto-caps at 150 and prints sample_scope/sample_caveat."),
        _arg("--answer", nargs="?", const=True, default=False, type=bool_value, help="Print a compact answer line for the selected winner."),
        _arg("--config-key", "--hparam", "--key", "--keys", action="append", default=[], help="Config key to include with --answer or --show-hparams; repeat for multiple keys."),
        _arg("--show-hparams", nargs="?", const=True, default=False, type=bool_value, dest="hparams", help="Print likely hyperparameter config keys for the selected run."),
        _arg("--hparams", nargs="*", default=None, dest="hparam_values", help=argparse.SUPPRESS),
        _arg("--summary-key", action="append", default=[], help="Extra summary metric keys to print."),
        _arg("--config-eq", action="append", default=[], help="Keep runs with scalar config KEY=VALUE; repeat as needed."),
    ]),
    ("summary-table", "Download W&B summary table files and compute compact success or snapshot rows.", core.cmd_summary_table, [
        _arg("--summary-key", help="Summary key containing a W&B table file; omit to auto-detect."),
        _arg("--mode", choices=["success", "snapshot"], default="success", help="success computes rates; snapshot prints matching rows."),
        _arg("--success-column", default="success", help="Boolean-ish success column."),
        _arg("--group-column", help="Column used for grouped rates."),
        _arg("--name-prefix", help="Only include runs whose names start with this prefix."),
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=20, help="Runs to inspect."),
        _arg("--order", default="-created_at", help="W&B run order."),
        _arg("--top", type=positive_int, default=10, help="Groups or rows to print."),
        _arg("--row-limit", type=positive_int, default=12, help="Rows to print in snapshot mode."),
        _arg("--table-hint", action="append", default=[], help="Table summary-key hint for auto-detection."),
        _arg("--label-column", help="Column to print as a row label."),
        _arg("--label-hint", action="append", default=[], help="Column-name hint for labels."),
        _arg("--metric-column", action="append", default=[], help="Metric column to print."),
        _arg("--metric-hint", action="append", default=[], help="Column-name hint for metrics."),
        _arg("--sort-column", help="Column used to sort snapshot rows."),
        _arg("--sort-hint", action="append", default=[], help="Column-name hint for sorting."),
        _arg("--verbose-errors", action="store_true", help="Print table download/parse errors."),
    ]),
    ("compare-cohorts", "Compare best metric values across named cohorts.", core.cmd_compare_cohorts, [
        _arg("--metric", help="Exact summary metric to compare."),
        _arg("--metric-hint", help="Metric family hint to resolve from sampled runs."),
        _arg("--sample", type=positive_int, default=20, help="Runs sampled to resolve --metric-hint; hint resolution is not exhaustive."),
        _arg("--scan", type=positive_int, default=10, help="Requested runs to scan per cohort; caps at 40 unless --large-scan is passed and prints effective_scan_limit/limit_caveat."),
        _arg("--sweeps", type=positive_int, default=20, help="Requested sweeps to inspect for sweep:any cohorts; caps at 50 unless --large-scan is passed."),
        _arg("--cohort", action="append", required=True, help=f"Cohort spec NAME=SELECTOR. {COHORT_SELECTOR_HELP}"),
        _arg("--limit", type=positive_int, default=3, help="Compatibility scan-size floor; compare-cohorts prints one best run per cohort."),
        _arg("--direction", choices=["auto", "min", "max"], default="auto", help="Whether lower or higher is better."),
        _arg("--large-scan", action="store_true", help="Allow the full requested --scan/--sweeps instead of the protective compare-cohorts caps."),
    ]),
    ("config-diff", "Compare run configs and print shared and differing keys.", core.cmd_config_diff, [
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=100, help="Requested hydrated runs to compare when explicit run refs are not provided; auto-caps at 150 and prints sample_scope/sample_caveat."),
        _arg("--order", default="+created_at", help="W&B run order."),
        _arg("--shared-limit", type=positive_int, default=80, help="Shared keys to print."),
        _arg("--skip-key-contains", action="append", default=[], help="Skip config keys containing this substring."),
        _arg("--name-include", "--run-name-include", action="append", default=[], help="Keep runs whose names contain this token; repeat as needed."),
        _arg("--name-exclude", action="append", default=[], help="Drop runs whose names contain this token; repeat as needed."),
        _arg("--run", action="append", default=[], help="Run id, display name, or unique visible suffix to compare; repeat for two or more runs."),
        _arg("--runs", nargs="+", default=[], help="Run ids/display names/unique visible suffixes to compare."),
        _arg("--run-a", help="First run id/display name for a two-run config diff."),
        _arg("--run-b", help="Second run id/display name for a two-run config diff."),
    ]),
    ("stability-scan", "Scan bounded, sampled histories for loss and gradient spikes.", core.cmd_stability_scan, [
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=40, help="Requested runs for history requests; caps at 40, or 18 when --history-samples >= 1000, and prints sample_scope/sample_caveat."),
        _arg("--order", default="-created_at", help="W&B run order."),
        _arg("--history-samples", "--samples", dest="samples", type=positive_int, default=120, help="Requested history points per selected run; high values reduce stability-scan run fanout."),
        _arg("--history-budget-seconds", type=positive_float, default=55.0, help="Command-level budget for per-run history probes; unfinished runs emit history_timeout rows."),
        _arg("--scan-history-fallback", action="store_true", help="Allow slower per-key run.scan_history fallback when sampled run.history has no points."),
        _arg("--large-scan", action="store_true", help="Allow the full requested run count instead of the protective stability-scan caps."),
        _arg("--top", type=positive_int, default=8, help="Rows to print."),
        _arg("--workers", type=positive_int, default=12, help="Parallel history workers."),
        _arg("--metric", action="append", default=[], help="Exact history metric to inspect."),
        _arg("--metric-limit", type=positive_int, default=5, help="Auto-selected history metrics."),
        _arg("--name-include", action="append", default=[], help="Keep runs whose names contain this token."),
        _arg("--name-exclude", action="append", default=[], help="Drop runs whose names contain this token."),
        _arg("--spike-ratio", type=positive_float, default=3.0, help="Max-to-median ratio considered a spike."),
    ]),
    ("gradient-noise-summary", "Summarize grad_noise metrics by optimizer, materialization, and regime from sampled runs.", core.cmd_gradient_noise_summary, [
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=120, help="Requested hydrated runs to scan for grad_noise metrics; auto-caps at 150 and prints sample_scope/sample_caveat."),
        _arg("--order", default="-created_at", help="W&B run order."),
        _arg("--row-limit", type=positive_int, default=12, help="Rows to print."),
        _arg("--materialization", action="append", default=[], help="Materialization filter."),
        _arg("--optimizer", action="append", default=[], help="Optimizer names to compare."),
        _arg("--contrast-metric", action="append", default=["grad_noise/noise_to_signal", "grad_noise/signal_power", "grad_noise/noise_power"], help="Grad-noise summary metric to compare."),
    ]),
    ("diagnose-history", "Diagnose history outliers or GPU/system metric utilization.", core.cmd_diagnose_history, [
        _arg("--mode", choices=["outlier", "gpu"], default="outlier", help="Diagnosis mode."),
        _arg("--metric", help="Exact history metric for outlier mode."),
        _arg("--metric-hint", default="loss", help="Metric family hint for outlier mode, resolved from sampled run summaries."),
        _arg("--metric-fallback", default="train/loss", help="Fallback metric if hint discovery finds nothing."),
        _arg("--target-run", help="Run id or display name to explain in outlier mode."),
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=60, help="Runs to inspect."),
        _arg("--order", default="+created_at", help="W&B run order."),
        _arg("--history-samples", "--samples", dest="samples", type=positive_int, default=80, help="History points per run."),
        _arg("--top", type=positive_int, default=8, help="Rows to print."),
        _arg("--workers", type=positive_int, default=12, help="Parallel history workers."),
        _arg("--layer-metrics", action="store_true", help="Include layer metric z-score diagnostics."),
        _arg("--name-include", action="append", default=[], help="Keep runs whose names contain this token."),
        _arg("--name-exclude", action="append", default=[], help="Drop runs whose names contain this token."),
        _arg("--training-like", action="store_true", help="For GPU mode, exclude obvious eval/toy/data-prep runs."),
        _arg("--min-points-for-verdict", type=positive_int, default=1, help="Minimum system samples used for verdict stats."),
        _arg("--good-threshold", type=percent_float, default=70.0, help="Mean GPU utilization percent considered healthy."),
    ]),
    ("large-project", "Fast bounded recipes for very large W&B run tables.", core.cmd_large_project, [
        _arg("--recipe", choices=OPS, help="Large-project recipe to run."),
        _arg("--op", choices=OPS, help="Alias for --recipe."),
        _arg("--run-id", help="Exact W&B run id. If a visible run-name fragment is passed, the helper will try to resolve it."),
        _arg("--run", "--run-name", dest="run_name", help="Run display name, exact W&B id, or unique visible run-name suffix for run-info/run-config lookups."),
        _arg("--sample", "--limit", dest="limit", type=positive_int, default=5, help="Runs or rows to inspect/print."),
        _arg("--history-samples", "--samples", dest="samples", type=positive_int, default=500, help="Runs per discovery slice for group/tag/name recipes; history points for system-metrics."),
        _arg("--order", help="Explicit W&B order for run-selection ops."),
        _arg("--tag", action="append", default=[], help="Tag to list or count; repeat as needed."),
        _arg("--group", action="append", default=[], help="Group to list or count; repeat as needed."),
        _arg("--name-include", "--run-name-include", action="append", default=[], help="Keep sampled runs whose names contain this token."),
        _arg("--name-exclude", action="append", default=[], help="Drop sampled runs whose names contain this token."),
        _arg("--name-prefix", "--prefix", action="append", default=[], help="Keep sampled runs whose names start with this prefix."),
        _arg("--name-regex", "--pattern", action="append", default=[], help="Keep sampled runs whose names match this regex."),
        _arg("--include-ungrouped", action="store_true", help="Include a direct SDK count for ungrouped runs."),
        _arg("--month", help="Date-count month in YYYY-MM form."),
        _arg("--start", help="Inclusive created_at lower bound."),
        _arg("--end", help="Exclusive created_at upper bound."),
        _arg("--label", help="Label for date-count output."),
        _arg("--left-month", help="Left config-profile month in YYYY-MM form."),
        _arg("--right-month", help="Right config-profile month in YYYY-MM form."),
        _arg("--left-start", help="Left profile inclusive lower bound."),
        _arg("--left-end", help="Left profile exclusive upper bound."),
        _arg("--right-start", help="Right profile inclusive lower bound."),
        _arg("--right-end", help="Right profile exclusive upper bound."),
        _arg("--left-label", help="Display label for left config profile."),
        _arg("--right-label", help="Display label for right config profile."),
        _arg("--left-order", default="+created_at", help="W&B order for left config profile."),
        _arg("--right-order", default="-created_at", help="W&B order for right config profile."),
        _arg("--config-key", "--hparam", "--key", "--keys", action="append", default=[], help="Config key to include in config profile comparison; repeat for multiple keys."),
    ]),
]

TOP_LEVEL_COMMANDS = {name for name, _, _, _ in COMMANDS}
LARGE_PROJECT_RECIPE_ALIASES = set(OPS) - TOP_LEVEL_COMMANDS


# Parser contract sets: these reject mode- or recipe-specific flags before any
# W&B API call can silently ignore a model's mistaken argument.
ORDER_VALUE_FLAGS = {"--order", "--left-order", "--right-order"}

# Flags accepted only by one summary-table mode.
SUMMARY_SUCCESS_ONLY = {"--success-column", "--group-column"}
SUMMARY_SNAPSHOT_ONLY = {"--row-limit", "--table-hint", "--label-column", "--label-hint", "--metric-column", "--metric-hint", "--sort-column", "--sort-hint"}

# Flags accepted only by one diagnose-history mode.
DIAGNOSE_OUTLIER_ONLY = {"--metric", "--metric-hint", "--metric-fallback", "--target-run", "--layer-metrics"}
DIAGNOSE_GPU_ONLY = {"--training-like", "--min-points-for-verdict", "--good-threshold"}

# Shared and recipe-specific large-project flags consumed by core.cmd_large_project.
LARGE_PROJECT_COMMON_FLAGS = {"--entity", "--project", "--include-sweeps", "--exclude-sweeps", "--recipe", "--op", "--answer"}
LARGE_PROJECT_FLAGS = {
    "status": set(),
    "project-description": set(),
    "run-info": {"--run-id", "--run", "--run-name", "--sample", "--limit", "--order"},
    "recent": {"--sample", "--limit", "--order"},
    "finished": {"--sample", "--limit", "--order"},
    "most-recent": {"--sample", "--limit", "--order"},
    "oldest": {"--sample", "--limit", "--order"},
    "earliest-finished": {"--sample", "--limit", "--order"},
    "latest-crashed": {"--sample", "--limit", "--order"},
    "creator": {"--sample", "--limit"},
    "name-pattern": {"--sample", "--limit", "--samples", "--name-include", "--run-name-include", "--name-exclude", "--name-prefix", "--prefix", "--name-regex", "--pattern"},
    "run-config": {"--sample", "--limit", "--order", "--state", "--run-id", "--run", "--run-name", "--name-include", "--run-name-include", "--name-exclude", "--name-prefix", "--prefix", "--name-regex", "--pattern", "--config-key", "--hparam", "--key", "--keys"},
    "summary-stats": {"--sample", "--limit", "--order", "--state", "--run-id", "--run", "--run-name", "--name-include", "--run-name-include", "--name-exclude", "--name-prefix", "--prefix", "--name-regex", "--pattern", "--config-key", "--hparam", "--key", "--keys"},
    "metric-count": {"--sample", "--limit", "--order", "--state", "--run-id", "--run", "--run-name", "--name-include", "--run-name-include", "--name-exclude", "--name-prefix", "--prefix", "--name-regex", "--pattern", "--config-key", "--hparam", "--key", "--keys"},
    "config-profiles": {"--left-month", "--right-month", "--left-start", "--left-end", "--right-start", "--right-end", "--left-label", "--right-label", "--left-order", "--right-order", "--config-key", "--hparam", "--key", "--keys"},
    "duration-range": {"--sample", "--limit"},
    "system-metrics": {"--sample", "--limit", "--state", "--history-samples", "--samples"},
    "tags": {"--sample", "--limit", "--samples", "--tag"},
    "groups": {"--sample", "--limit", "--samples", "--group", "--include-ungrouped", "--order"},
    "tag-counts": {"--sample", "--limit", "--samples", "--tag"},
    "date-counts": {"--month", "--start", "--end", "--label"},
}


def _provided_options(argv: list[str]) -> set[str]:
    return {token.split("=", 1)[0] for token in argv if token.startswith("--")}


def _normalize_dash_option_values(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ORDER_VALUE_FLAGS and index + 1 < len(argv) and argv[index + 1].startswith("-") and not argv[index + 1].startswith("--"):
            normalized.append(f"{token}={argv[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def _normalize_large_project_recipe_alias(argv: list[str]) -> list[str]:
    if argv and argv[0] in LARGE_PROJECT_RECIPE_ALIASES:
        return ["large-project", "--recipe", argv[0], *argv[1:]]
    return argv


def _large_project_positional_recipe(argv: list[str]) -> str | None:
    """Return a recipe passed positionally after large-project, if present."""
    if not argv or argv[0] != "large-project":
        return None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return None
        if token in {"--recipe", "--op"}:
            return None
        if token.startswith("--"):
            has_separate_value = "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("--")
            index += 2 if has_separate_value else 1
            continue
        return token if token in OPS else None
    return None


def _reject_provided(parser: argparse.ArgumentParser, args: argparse.Namespace, flags: set[str], message: str) -> None:
    invalid = sorted(getattr(args, "_provided_options", set()) & flags)
    if invalid:
        parser.error(f"{', '.join(invalid)} {message}")


def _canonical_example(command: str, args: argparse.Namespace | None = None) -> str:
    scope = "--entity ENTITY --project PROJECT"
    if command != "large-project":
        return f"python $R {command} {scope} --help"

    recipe = getattr(args, "recipe", None) or getattr(args, "op", None) or "status"
    extras = {
        "run-info": "--run RUN_NAME_OR_ID",
        "run-config": "--name-include TOKEN --config-key KEY --sample 20",
        "summary-stats": "--name-include TOKEN --sample 20",
        "metric-count": "--name-include TOKEN --sample 20",
        "config-profiles": "--left-month YYYY-MM --right-month YYYY-MM --config-key KEY",
        "system-metrics": "--sample 20 --history-samples 500",
        "name-pattern": "--name-regex REGEX --sample 100",
        "tag-counts": "--tag TAG --sample 500",
        "groups": "--sample 500",
        "date-counts": "--month YYYY-MM",
    }.get(recipe, "")
    return f"python $R large-project {scope} --recipe {recipe}" + (f" {extras}" if extras else "")


def _valid_flags_text(flags: set[str]) -> str:
    return ", ".join(sorted(flags))


def _validate_regex(parser: argparse.ArgumentParser, pattern: str, label: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        parser.error(f"Invalid {label} {pattern!r}: {exc}")


def _validate_rank_runs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    requested: list[tuple[str, str]] = []
    if args.direction != "auto":
        requested.append(("--direction", args.direction))
    if args.maximize:
        requested.append(("--maximize/--higher-is-better", "max"))
    if args.minimize:
        requested.append(("--minimize/--lower-is-better", "min"))
    directions = {direction for _, direction in requested}
    if len(directions) > 1:
        parser.error("rank-runs direction flags disagree: " + ", ".join(f"{flag}={direction}" for flag, direction in requested))
    if requested:
        args.direction = requested[0][1]
    if args.hparam_values is not None:
        args.hparams = True
        args.config_key.extend(_split_values(args.hparam_values))


def _validate_variant_analysis(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.variant and not args.pattern:
        parser.error("variant-analysis needs a selector. Pass --variant NAME[=REGEX] or --pattern REGEX.")
    if args.pattern:
        _validate_regex(parser, args.pattern, "--pattern regex")
    for spec in args.variant:
        if "=" in spec:
            _validate_regex(parser, spec.split("=", 1)[1], "--variant NAME=REGEX pattern")


def _validate_name_patterns(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.pattern:
        _validate_regex(parser, args.pattern, "--name-regex pattern")


def _valid_cohort_selector(selector: str) -> bool:
    return selector == "non_sweep" or selector == "sweep:any" or selector.startswith(("sweep:", "tag:", "group:", *COHORT_NAME_SELECTOR_PREFIXES))


def _validate_compare_cohorts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for spec in args.cohort:
        if "=" not in spec or not all(part.strip() for part in spec.split("=", 1)):
            parser.error(f"--cohort must be NAME=SELECTOR. {COHORT_SELECTOR_HELP}")
        _, selector = spec.split("=", 1)
        if not _valid_cohort_selector(selector):
            parser.error(f"Unknown cohort selector {selector!r}. {COHORT_SELECTOR_HELP}")
        if selector.startswith(COHORT_NAME_SELECTOR_PREFIXES):
            try:
                re.compile(selector.split(":", 1)[1])
            except re.error as exc:
                parser.error(f"Invalid run-name cohort selector {selector!r}: {exc}")


def _validate_config_diff(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    refs: list[str] = []
    if args.run_a or args.run_b:
        if not (args.run_a and args.run_b):
            parser.error("config-diff requires both --run-a RUN and --run-b RUN")
        refs.extend([args.run_a, args.run_b])
    refs.extend(getattr(args, "run", []) or [])
    refs.extend(getattr(args, "runs", []) or [])
    if refs and len(set(refs)) < 2:
        parser.error("config-diff explicit run comparison requires at least two distinct run refs")


def _validate_summary_table(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "snapshot":
        _reject_provided(parser, args, SUMMARY_SUCCESS_ONLY, "is only valid with summary-table --mode success")
    else:
        _reject_provided(parser, args, SUMMARY_SNAPSHOT_ONLY, "is only valid with summary-table --mode snapshot")


def _validate_diagnose_history(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "gpu":
        _reject_provided(parser, args, DIAGNOSE_OUTLIER_ONLY, "is only valid with diagnose-history --mode outlier")
    else:
        _reject_provided(parser, args, DIAGNOSE_GPU_ONLY, "is only valid with diagnose-history --mode gpu")


def _validate_large_project(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    recipe, op = getattr(args, "recipe", None), getattr(args, "op", None)
    if recipe and op and recipe != op:
        parser.error(
            "--op and --recipe disagree; pass only one recipe. "
            f"Example: {_canonical_example('large-project', argparse.Namespace(recipe=recipe, op=None))}"
        )
    selected = recipe or op
    if not selected:
        parser.error(
            "large-project requires --recipe RECIPE. "
            f"Example: {_canonical_example('large-project', argparse.Namespace(recipe='status', op=None))}"
        )
    if selected in {"name-pattern", "tags", "groups", "tag-counts"} and getattr(args, "_provided_options", set()) & {"--sample", "--limit"}:
        args.samples = args.limit
    allowed = LARGE_PROJECT_COMMON_FLAGS | LARGE_PROJECT_FLAGS[selected]
    invalid = sorted(getattr(args, "_provided_options", set()) - allowed)
    if invalid:
        parser.error(
            f"large-project --recipe {selected} does not use {', '.join(invalid)}. "
            f"Valid flags: {_valid_flags_text(allowed)}. "
            f"Try: {_canonical_example('large-project', args)}. "
            "Run: python $R large-project --help"
        )
    for pattern in getattr(args, "name_regex", []) or []:
        _validate_regex(parser, pattern, "--name-regex pattern")
    if selected == "run-info" and not (args.run_id or args.run_name):
        parser.error(
            "large-project --recipe run-info requires --run RUN_NAME_OR_ID or --run-id RUN_ID. "
            f"Try: {_canonical_example('large-project', args)}"
        )
    if selected == "date-counts" and not (args.month or args.start or args.end):
        parser.error(
            "large-project --recipe date-counts requires --month or --start/--end. "
            f"Try: {_canonical_example('large-project', args)}"
        )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if hasattr(args, "config_key"):
        args.config_key = _split_values(args.config_key)
    validators = {
        "name-patterns": _validate_name_patterns,
        "rank-runs": _validate_rank_runs,
        "variant-analysis": _validate_variant_analysis,
        "compare-cohorts": _validate_compare_cohorts,
        "config-diff": _validate_config_diff,
        "summary-table": _validate_summary_table,
        "diagnose-history": _validate_diagnose_history,
        "large-project": _validate_large_project,
    }
    if validator := validators.get(args.command):
        validator(parser, args)


class RunOpsArgumentParser(argparse.ArgumentParser):
    """Top-level parser with model-friendly normalization and validation."""

    def error(self, message: str) -> None:
        command = getattr(self, "_run_ops_command", None)
        hint = getattr(self, "_error_hint", None) or _command_error_hint(command)
        if hint and "Canonical examples:" not in message:
            message = f"{message}\n\n{hint}"
        super().error(message)

    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        raw_args = list(sys.argv[1:] if args is None else args)
        self._run_ops_command = raw_args[0] if raw_args and raw_args[0] in TOP_LEVEL_COMMANDS else None
        if recipe := _large_project_positional_recipe(raw_args):
            self.error(
                "large-project recipes must be passed with --recipe. "
                f"Wrong: python $R {' '.join(raw_args)}. "
                f"Right: {_canonical_example('large-project', argparse.Namespace(recipe=recipe, op=None))}"
            )
        normalized = _normalize_large_project_recipe_alias(raw_args)
        parsed = super().parse_args(_normalize_dash_option_values(normalized), namespace)
        parsed._provided_options = _provided_options(raw_args)
        _validate_args(self, parsed)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the public wandb_run_ops.py argument parser."""
    parser = RunOpsArgumentParser(
        description=DESCRIPTION,
        epilog=TOP_LEVEL_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser._run_ops_command = None
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text, func, specs in COMMANDS:
        child = sub.add_parser(
            name,
            help=help_text,
            description=help_text,
            epilog=COMMAND_EPILOGS.get(name),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        child._run_ops_command = name
        child._error_hint = _command_error_hint(name)
        for flags, kwargs in COMMON + specs:
            child.add_argument(*flags, **kwargs)
        if name != "rank-runs":
            child.add_argument("--answer", nargs="?", const=True, default=False, type=bool_value, help=argparse.SUPPRESS)
        child.set_defaults(func=func)
    return parser


def main() -> None:
    """Parse CLI args and dispatch to the selected command."""
    args = build_parser().parse_args()
    args.func(args)
