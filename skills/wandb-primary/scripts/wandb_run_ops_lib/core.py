# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from .common import *  # noqa: F403
from wandb_run_vocab import CONFIG_AXIS_TOKENS, CONFIG_OVERVIEW_TOKENS, GPU_TRAINING_EXCLUDE_TERMS, LAYER_METRIC_TOKENS, LAYER_STAT_TOKENS, STABILITY_GRADIENT_TOKENS, STABILITY_PRIMARY_LOSS_TOKENS, STABILITY_REGULARIZER_LOSS_TOKENS
DATA_SUFFIXES = {".jsonl", ".ndjson", ".json", ".csv", ".tsv"}; MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov"}; CHECKPOINT_TOKENS = "checkpoint ckpt model weights adapter lora state_dict train-state best last".split()
STRUCTURED_FIELD_COUNT_FILENAME_TOKENS = "manifest metadata meta stats statistic statistics summary summaries index indices labels annotations records rows results evaluations metrics scores monitoring monitors traces trace calls events telemetry".split()
# Exclude noisy/high-cardinality fields so compact counts surface useful categorical breakdowns.
STRUCTURED_COUNT_EXCLUDE_TOKENS = set("id uuid guid digest hash sha checksum path url uri time timestamp created updated date file filename filepath image media screenshot video audio text body prompt message content html markdown payload raw bytes blob embedding vector array tensor matrix data step steps index idx line lineno".split())
STRUCTURED_COUNT_MAX_BYTES = 25_000_000
ARTIFACT_SUMMARY_MAX_BYTES = 500_000_000
STRUCTURED_COUNT_MAX_FIELDS = 20
STRUCTURED_COUNT_MAX_VALUES = 100
STRUCTURED_COUNT_MAX_TRACKED_VALUES = 1_000
STRUCTURED_COUNT_MIN_COVERAGE = 0.5
STRUCTURED_DATETIME_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-]+Z?)?$")
STRUCTURED_KEY_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
PROJECT_NAME_STOPWORDS = set("run runs wandb trace traces latency report html analysis visualization dashboard finished failed crashed resumed latest test eval evaluation".split())
WEAVE_COMPONENT_TERMS = "agent answer chat correctness dataset embedding eval faithfulness hallucination mrr retrieval rerank reward score scorer search tool trace trajectory".split()
HYDRATED_RUN_SAMPLE_CAP = 150
INSPECT_DISTINCT_VALUES_MAX = 100
HISTORY_RUN_SAMPLE_CAP = 40
DEEP_HISTORY_RUN_SAMPLE_CAP = 18
COHORT_SCAN_CAP = 40
COHORT_SWEEP_CAP = 50
PROJECT_SNAPSHOT_ARTIFACT_RUN_COUNT_LIMIT = 10_000
PROJECT_SNAPSHOT_HYDRATED_SIDE_CAP = 75
STEP_SUMMARY_KEYS = ["_step", "train/global_step", "global_step", "step"]


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def emit(key: str, value: Any) -> None:
    """Print a stable key=json payload line."""
    print(f"{key}=" + json.dumps(value, sort_keys=True, default=str))


def ctx(args: argparse.Namespace) -> tuple[Any, str]:
    return api(), resolve_path(getattr(args, "entity", None), getattr(args, "project", None))


def runs(wb: Any, path: str, args: argparse.Namespace, *, filters: dict[str, Any] | None = None, order: str | None = None, limit: int | None = None, lazy: bool = True) -> list[Any]:
    """Fetch a bounded ordered run list through the W&B SDK."""
    limit = limit or getattr(args, "limit", 5)
    return get_runs(wb, path, filters=filters, order=order or getattr(args, "order", None) or "-created_at", limit=limit, include_sweeps=getattr(args, "include_sweeps", False), lazy=lazy)


def count(wb: Any, path: str, filters: dict[str, Any] | None = None, *, include_sweeps: bool = False) -> int:
    try:
        return len(wb.runs(path, filters=filters or None, per_page=1, include_sweeps=include_sweeps, lazy=True))
    except Exception as exc:
        raise SystemExit(
            f"W&B run count failed for {path}: {type(exc).__name__}: {exc}. "
            f"filters={filters or {}} include_sweeps={include_sweeps}"
        ) from exc


def capped_hydrated_limit(requested: int, cap: int = HYDRATED_RUN_SAMPLE_CAP) -> int:
    return min(max(1, requested), cap)


def print_cap_caveat(label: str, requested: int, used: int) -> None:
    if used < requested:
        print(f"sample_caveat={label} capped run scan at {used} of requested {requested} to avoid expensive per-run data loads")


def print_limit_caveat(label: str, requested: int, used: int, *, override_flag: str = "--large-scan") -> None:
    if used < requested:
        print(f"limit_caveat={label} capped scan at {used} of requested {requested}; pass {override_flag} only when the full scan is worth the extra W&B API time")


def print_effective_scan_limit(label: str, requested: int, used: int, *, unit: str, large_scan: bool) -> None:
    emit(
        "effective_scan_limit",
        {
            "command": label,
            "unit": unit,
            "requested": requested,
            "effective": used,
            "cap_applied": used < requested,
            "large_scan": large_scan,
        },
    )


def print_sample_scope(
    label: str,
    *,
    requested: Any,
    effective: Any,
    actual: int,
    unit: str = "runs",
    strategy: str,
    cap_applied: bool | None = None,
    project_exhaustive: bool = False,
    note: str | None = None,
) -> None:
    if cap_applied is None:
        cap_applied = isinstance(requested, int) and isinstance(effective, int) and effective < requested
    payload = {
        "command": label,
        "unit": unit,
        "requested": requested,
        "effective": effective,
        "actual": actual,
        "cap_applied": cap_applied,
        "project_exhaustive": project_exhaustive,
        "strategy": strategy,
        "meaning": "results describe only the sampled/returned data, not every run in the project",
    }
    if note:
        payload["note"] = note
    emit("sample_scope", payload)


def stability_run_sample_cap(history_samples: int) -> int:
    return DEEP_HISTORY_RUN_SAMPLE_CAP if history_samples >= 1000 else HISTORY_RUN_SAMPLE_CAP


def metric_from_args(wb: Any, path: str, args: argparse.Namespace, filters: dict[str, Any] | None = None) -> str | None:
    metric = first(getattr(args, "metric", None))
    if metric:
        return metric
    hint = first(getattr(args, "metric_hint", None))
    if not hint:
        return None
    sample = runs(wb, path, args, filters=filters, order="-created_at", limit=getattr(args, "sample", 80))
    return choose_metric_from_runs(sample, hint)


def first(value: Any, default: Any = None) -> Any: return value[0] if isinstance(value, list) and value else value if value not in ([], None) else default


def metric_order(metric: str | None, direction: str | None, fallback: str = "-created_at") -> str: return fallback if not metric else f"{'+' if direction == 'min' else '-'}summary_metrics.{metric}"


def step_budget_from_summary(summary: dict[str, Any]) -> int | None:
    for key in STEP_SUMMARY_KEYS:
        value = coerce_float(summary.get(key))
        if value is not None:
            return int(value)
    return None


def first_run_with_runtime(candidates: list[Any]) -> tuple[Any | None, float | None, int]:
    """Return the first candidate with a summary runtime and the number checked."""
    for checked, run in enumerate(candidates, start=1):
        runtime = runtime_seconds(run)
        if runtime is not None:
            return run, runtime, checked
    return None, None, len(candidates)


def filter_names(
    items: list[Any],
    include: list[str],
    exclude: list[str],
    prefixes: list[str] | None = None,
    regexes: list[str] | None = None,
) -> list[Any]:
    inc = [x.lower() for x in include]
    exc = [x.lower() for x in exclude]
    prefix_values = [x.lower() for x in prefixes or []]
    patterns = [re.compile(x, re.I) for x in regexes or []]
    rows = []
    for run in items:
        label = run_label(run)
        lower = label.lower()
        if inc and not all(t in lower for t in inc):
            continue
        if any(t in lower for t in exc):
            continue
        if prefix_values and not any(lower.startswith(prefix) for prefix in prefix_values):
            continue
        if patterns and not any(pattern.search(label) for pattern in patterns):
            continue
        rows.append(run)
    return rows


def find_run_by_name(
    wb: Any,
    path: str,
    query: str,
    *,
    limit: int = 200,
    include_sweeps: bool = False,
) -> Any | None:
    """Find one run by visible name/display-name/id fragment."""
    return resolve_run_ref(
        wb,
        path,
        query,
        limit=limit,
        include_sweeps=include_sweeps,
        prefer_direct_id=False,
    )


def is_missing_run_error(exc: Exception) -> bool:
    """Return whether a direct W&B run lookup failed because the run id is absent."""
    return type(exc).__name__ == "CommError" and "could not find run" in str(exc).lower()


def verify_run_ref(wb: Any, path: str, run_id: str) -> Any | None:
    """Return a direct W&B run lookup only if it can be loaded."""
    try:
        run = wb.run(f"{path}/{run_id}")
        run_label(run)
        getattr(run, "state", None)
        return run
    except Exception as exc:
        if is_missing_run_error(exc):
            return None
        raise SystemExit(f"Run id lookup failed for {path}/{run_id}: {type(exc).__name__}: {exc}") from exc


def unique_run_match(matches: list[Any], query: str, source: str) -> Any | None:
    """Return the only run in a candidate list, or fail with useful candidates."""
    unique: list[Any] = []
    seen: set[Any] = set()
    for run in matches:
        key = getattr(run, "id", None) or id(run)
        if key in seen:
            continue
        seen.add(key)
        unique.append(run)

    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]

    candidates = [
        f"{run_label(run)} ({getattr(run, 'id', '<unknown>')})"
        for run in unique[:8]
    ]
    raise SystemExit(
        f"Run reference {query!r} matched multiple {source} runs. "
        "Use --run with the exact display name or --run-id with the exact W&B run id. "
        "candidates=" + ", ".join(candidates)
    )


def display_name_matches(
    wb: Any,
    path: str,
    query: str,
    *,
    exact: bool,
    limit: int,
    include_sweeps: bool,
) -> list[Any]:
    """Query W&B's display-name filter and verify matches locally."""
    lowered = query.lower()
    escaped = re.escape(query)
    pattern = re.compile(rf"(^|/|\.){escaped}$", re.I)
    run_filter = {"display_name": query} if exact else {"display_name": {"$regex": pattern.pattern}}
    matches: list[Any] = []
    mode = "exact display-name" if exact else "display-name suffix"
    try:
        candidates = get_runs(
            wb,
            path,
            filters=run_filter,
            order="-created_at",
            limit=limit,
            include_sweeps=include_sweeps,
        )
    except SystemExit as exc:
        raise SystemExit(f"Run {mode} lookup failed for {path}: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Run {mode} lookup failed for {path}: {type(exc).__name__}: {exc}") from exc
    for run in candidates:
        labels = [run_label(run), str(getattr(run, "name", ""))]
        if (exact and any(label and label.lower() == lowered for label in labels)) or (
            not exact and any(label and pattern.search(label) for label in labels)
        ):
            matches.append(run)
    return matches


def bounded_run_name_matches(
    wb: Any,
    path: str,
    query: str,
    *,
    limit: int,
    include_sweeps: bool,
) -> list[Any]:
    """Fallback visible-name/id search over recent and oldest slices."""
    lowered = query.lower()
    exact_matches = []
    partial_matches = []
    for run in sample_run_slices(wb, path, limit=limit, include_sweeps=include_sweeps):
        labels = {str(getattr(run, "id", "")), run_label(run), str(getattr(run, "name", ""))}
        if any(label and label.lower() == lowered for label in labels):
            exact_matches.append(run)
        elif any(label and lowered in label.lower() for label in labels):
            partial_matches.append(run)
    return exact_matches or partial_matches


def resolve_run_ref(
    wb: Any,
    path: str,
    query: str,
    *,
    limit: int = 200,
    include_sweeps: bool = False,
    prefer_direct_id: bool = True,
) -> Any | None:
    """Resolve a W&B run id, display name, or unique display-name suffix."""
    if prefer_direct_id:
        if run := verify_run_ref(wb, path, query):
            return run

    if run := unique_run_match(
        display_name_matches(
            wb,
            path,
            query,
            exact=True,
            limit=limit,
            include_sweeps=include_sweeps,
        ),
        query,
        "exact display-name",
    ):
        return run

    if run := unique_run_match(
        display_name_matches(
            wb,
            path,
            query,
            exact=False,
            limit=limit,
            include_sweeps=include_sweeps,
        ),
        query,
        "display-name suffix",
    ):
        return run

    return unique_run_match(
        bounded_run_name_matches(
            wb,
            path,
            query,
            limit=limit,
            include_sweeps=include_sweeps,
        ),
        query,
        "bounded-sample",
    )


def should_probe_direct_run_id(run_id_arg: Any, target: str) -> bool:
    return bool(run_id_arg) or "/" not in target


def discover_group_counts(wb: Any, path: str, args: argparse.Namespace, filters: dict[str, Any] | None = None, sample_size: int | None = None) -> tuple[list[Any], list[str], dict[str, int], list[str], int]:
    """Discover groups from bounded slices and exact-count discovered or explicit groups."""
    orders = [args.order] if getattr(args, "order", None) else list(RECENT_AND_OLDEST_ORDERS)
    per_slice = max(1, sample_size or args.group_sample)
    sample = sample_run_slices(wb, path, filters=filters or None, orders=orders, limit=per_slice, include_sweeps=getattr(args, "include_sweeps", False))
    groups = sorted({str(getattr(run, "group", "") or "") for run in sample if getattr(run, "group", None)} | set(getattr(args, "group", []) or []))
    counts = {g: count(wb, path, {**(filters or {}), "group": g}, include_sweeps=getattr(args, "include_sweeps", False)) for g in groups}
    if getattr(args, "include_ungrouped", False):
        counts["<none>"] = count(wb, path, {**(filters or {}), "group": None}, include_sweeps=getattr(args, "include_sweeps", False))
    return sample, groups, counts, orders, per_slice


def print_group_counts(sample: list[Any], groups: list[str], counts: dict[str, int], orders: list[str], per_slice: int) -> None:
    """Print exact group counts plus bounded-discovery caveats."""
    print(f"sampled_runs={len(sample)}")
    print(f"group_discovery_sampled_runs={len(sample)}")
    print(f"group_discovery_per_slice_limit={per_slice}")
    print("group_discovery_orders=" + ",".join(orders))
    print("sample_groups=" + ", ".join(groups))
    print("group_count_scope=exact_for_discovered_or_explicit_groups")
    print("sample_caveat=groups absent from the bounded discovery sample are omitted, not undercounted")
    emit("group_counts", counts)
    for group, value in counts.items():
        print(f"group_count.{group}={value}")
    print("answer=" + ("; ".join(f"{group}={value}" for group, value in counts.items()) if counts else "No groups found in bounded discovery sample."))


def stability_metric_role(key: str) -> str | None:
    lower = key.lower()
    if any(t in lower for t in STABILITY_GRADIENT_TOKENS): return "gradient_norm"
    if "loss" not in lower and "kl" not in lower: return None
    if any(t in lower for t in STABILITY_REGULARIZER_LOSS_TOKENS): return "regularizer_loss"
    if any(t in lower for t in STABILITY_PRIMARY_LOSS_TOKENS): return "primary_loss"
    return "aggregate_loss"


def stability_metric_score(key: str, role: str) -> tuple[int, int, int, str]:
    lower = key.lower(); priority = {"primary_loss": 0, "regularizer_loss": 1, "gradient_norm": 2, "aggregate_loss": 3}.get(role, 9)
    return (priority, 0 if any(t in lower for t in ("train", "training")) else 1, 0 if re.search(r"loss[_/-][a-z0-9]+|[a-z0-9]+[_/-]loss", lower) else 1, key)


def select_stability_metric_keys(sample: list[Any], limit: int) -> list[str]:
    by_role: dict[str, set[str]] = defaultdict(set)
    for run in sample:
        for key, value in (run.summary_metrics or {}).items():
            role = stability_metric_role(str(key))
            if role and coerce_float(value) is not None:
                by_role[role].add(str(key))
    selected: list[str] = []
    for role in ["primary_loss", "regularizer_loss", "gradient_norm", "aggregate_loss"]:
        keys = sorted(by_role.get(role, ()), key=lambda k: stability_metric_score(k, role))
        if keys and keys[0] not in selected:
            selected.append(keys[0])
    for role, keys in sorted(by_role.items()):
        for key in sorted(keys, key=lambda k: stability_metric_score(k, role)):
            if len(selected) >= limit: return selected
            if key not in selected: selected.append(key)
    return selected[:limit]


# -----------------------------------------------------------------------------
# Basic run inspection
# -----------------------------------------------------------------------------


def cmd_count_runs(args: argparse.Namespace) -> None:
    """Print fast SDK-backed run, sweep, or group counts."""
    wb, path = ctx(args)
    print(f"path={path}")
    filters = filters_from_args(args)
    if args.by_sweep:
        entity, project = split_path(path)
        rows = [{"sweep_id": s.id, "name": getattr(s, "name", None), "state": getattr(s, "state", None), "count": count(wb, path, merge_filters(filters, {"sweep": s.id}))} for s in wb.project(project, entity=entity).sweeps(per_page=args.sweeps)]
        print(f"sweep_count={len(rows)}")
        print(f"sweep_count_state_scope={args.state or 'all_states'}")
        print(f"total_sweep_runs={sum(r['count'] for r in rows)}")
        for row in sorted(rows, key=lambda r: r["count"], reverse=True):
            print(json.dumps(row, sort_keys=True, default=str))
        return
    if args.by_group:
        print_group_counts(*discover_group_counts(wb, path, args, filters))
        return
    label = f"{args.state}_run_count" if args.state else "run_count"
    print(f"{label}={count(wb, path, filters, include_sweeps=args.include_sweeps)}")


def distinct_config_value_rows(sample: list[Any], keys: list[str]) -> list[dict[str, Any]]:
    """Summarize distinct scalar config values for requested keys from a run sample."""
    counters = {key: Counter() for key in keys}
    runs_with_key = {key: 0 for key in keys}
    for run in sample:
        config = clean_config(run.config)
        for key in keys:
            raw = nested_config_value(config, key)
            if raw is None:
                continue
            scalar = scalar_config_value(raw)
            if scalar is None:
                continue
            runs_with_key[key] += 1
            counters[key][jsonable(scalar)] += 1

    rows: list[dict[str, Any]] = []
    for key in keys:
        counter = counters[key]
        counts = dict(counter.most_common(INSPECT_DISTINCT_VALUES_MAX))
        row: dict[str, Any] = {
            "key": key,
            # Values come only from the bounded run sample, not the whole project. A value
            # missing here may still exist in unsampled runs; don't treat absence as proof.
            "sample_only": True,
            "runs_with_key": runs_with_key[key],
            "distinct_count": len(counter),
            "counts": counts,
        }
        omitted = len(counter) - len(counts)
        if omitted:
            row["truncated"] = True
            row["omitted_distinct_count"] = omitted
        rows.append(row)
    return rows


def cmd_inspect_schema(args: argparse.Namespace) -> None:
    """List sampled config and/or summary keys for a project."""
    wb, path = ctx(args)
    sample_limit = capped_hydrated_limit(args.sample)
    sample = runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=sample_limit, lazy=False)
    # Strip and drop blank keys (e.g. --distinct-values "") while de-duplicating.
    distinct_keys = list(dict.fromkeys(k.strip() for k in (getattr(args, "distinct_values", None) or []) if k.strip()))
    keys = {"config": set(), "summary": set()}
    # Track the set of value types observed for each summary key across the sample so
    # callers can skip non-rankable keys (images/tables/etc.)
    summary_types: dict[str, set[str]] = defaultdict(set)
    for run in sample:
        keys["config"].update(clean_config(run.config))
        for k, v in (run.summary_metrics or {}).items():
            keys["summary"].add(k)
            if v is None:  # null carries no type signal; don't false-flag an otherwise-numeric key
                continue
            if isinstance(v, bool):
                summary_types[k].add("bool")
            elif isinstance(v, (int, float)):
                summary_types[k].add("numeric")
            elif isinstance(v, dict) and "_type" in v:
                summary_types[k].add(str(v["_type"]))
            else:
                summary_types[k].add(type(v).__name__)
    print(f"path={path}")
    print_sample_scope(
        "inspect-schema",
        requested=args.sample,
        effective=sample_limit,
        actual=len(sample),
        strategy=f"hydrated SDK runs ordered by {args.order}",
        note="config_keys and summary_keys are discovered from sampled runs only",
    )
    print_cap_caveat("inspect-schema", args.sample, sample_limit)
    print(f"sampled_runs={len(sample)}")
    if distinct_keys:
        for row in distinct_config_value_rows(sample, distinct_keys):
            emit("distinct_config_values", row)
    # Keys whose observed values are not purely rankable scalars — ranking runs by these
    # yields NaN, so surface them explicitly with their observed type(s).
    non_numeric_keys = sorted(
        f"{k} [{', '.join(sorted(types))}]"
        for k, types in summary_types.items()
        if types and not types.issubset({"numeric", "bool"})
    )
    for name in ("config", "summary"):
        if args.schema in {name, "both"}:
            # Bounded output: full key list runs to MB on ultra-wide projects and
            # bloats the turn. Emit count + top families + a sample instead.
            ks = sorted(keys[name])
            print(f"{name}_key_count={len(ks)}")
            emit(f"{name}_key_families", dict(Counter(re.split(r"[/.]", k, 1)[0] for k in ks).most_common(50)))
            if name == "summary" and non_numeric_keys:
                emit("summary_non_numeric_keys", non_numeric_keys[:100])
                if len(non_numeric_keys) > 100:
                    print("summary_non_numeric_keys_truncated=true")
            emit(f"{name}_keys_sample", ks[:200])
            if len(ks) > 200:
                print(f"{name}_keys_truncated=true")


def cmd_name_patterns(args: argparse.Namespace) -> None:
    """Inspect run names with prefix, regex, and sampled pattern summaries."""
    wb, path = ctx(args)
    pattern = re.compile(args.pattern, re.I) if args.pattern else None
    sample = (
        runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=args.limit)
        if (args.prefix or pattern)
        else sample_metadata_runs(wb, path, max(args.limit, 20), include_sweeps=args.include_sweeps)
    )
    matched = []
    for run in sample:
        label = run_label(run)
        ok = (not args.prefix or label.lower().startswith(args.prefix.lower())) and (not pattern or pattern.search(label))
        if ok:
            matched.append(run_row(run))
        if args.show_sample:
            emit("sample", {**run_row(run), "matched": bool(ok)})
    print(f"path={path}")
    print(f"sample_mode={args.order if (args.prefix or pattern) else 'recent_and_oldest'}")
    print(f"scanned_runs={len(sample)}")
    print(f"matched_runs={len(matched)}")
    for row in matched[: args.limit_matches]:
        emit("match", row)
    if args.prefix:
        if matched:
            print(f"answer={len(matched)} of {len(sample)} scanned runs matched name prefix {args.prefix!r}; recommended_regex=^{re.escape(args.prefix)}")
        else:
            fallback = sample_metadata_runs(
                wb,
                path,
                max(args.limit, 20),
                include_sweeps=args.include_sweeps,
            )
            candidates = nearby_prefix_candidates([run_label(run) for run in fallback], args.prefix)
            print(f"fallback_sampled_runs={len(fallback)}")
            for candidate in candidates[:5]:
                emit("candidate_prefix", candidate)
            if candidates:
                best = candidates[0]
                print(f"answer=0 direct matches for {args.prefix!r}; nearest observed run-name prefix is {best['prefix']!r} with {best['count']} sampled examples; recommended_regex=^{re.escape(best['prefix'])}")
            else:
                print(f"answer=0 of {len(sample)} scanned runs matched name prefix {args.prefix!r}; inspect sampled run names before recommending a regex")
    elif args.pattern:
        print(f"answer={len(matched)} of {len(sample)} scanned runs matched regex {args.pattern!r}")
    else:
        names = [run_label(run) for run in sample]
        label, buckets = name_pattern_summary(names)
        for key, values in buckets.items():
            print(f"pattern_count.{key}={len(values)}")
            if values:
                print(f"pattern_examples.{key}=" + ", ".join(values[:5]))
        print(f"pattern={label}")
        print("answer=" + label + "; sample_slices=recent_and_oldest; examples=" + ", ".join(names[: args.limit_matches]))


# -----------------------------------------------------------------------------
# Variants and run ranking
# -----------------------------------------------------------------------------


def cmd_variant_analysis(args: argparse.Namespace) -> None:
    """Group runs into variants and summarize counts or best metric values."""
    if not getattr(args, "variant", None) and not getattr(args, "pattern", None):
        raise SystemExit("variant-analysis needs a selector. Pass --variant NAME[=REGEX] or --pattern REGEX.")
    wb, path = ctx(args)
    base_filters = filters_from_args(args) or None
    metric = metric_from_args(wb, path, args, base_filters)
    direction = numeric_direction(metric, first(args.metric_hint)) if args.direction == "auto" else args.direction
    metric_filter = {f"summary_metrics.{metric}": {"$exists": True}} if metric else None
    filters = merge_filters(base_filters, metric_filter) or None
    fast_best = bool(args.fast_best and metric and args.rank_by == "best")
    order = args.order or (metric_order(metric, direction) if fast_best else "-created_at")
    stratify_by = list(dict.fromkeys(getattr(args, "stratify_by", []) or []))
    fetch_config_keys = list(dict.fromkeys([*(args.config_key or []), *stratify_by]))
    specs, extract = [parse_variant_spec(s) for s in args.variant], re.compile(args.pattern, re.I) if args.pattern else None
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filter_caveat = None
    try:
        fetched_rows, has_more = fetch_run_table_rows(wb, path, summary_keys=[metric] if metric else [], config_keys=fetch_config_keys, filters=filters, order=order, limit=args.limit)
    except Exception as exc:
        if not metric_filter:
            raise
        filter_caveat = f"metric_exists_filter_failed={type(exc).__name__}; retried_without_metric_exists_filter"
        filters = base_filters
        fetched_rows, has_more = fetch_run_table_rows(wb, path, summary_keys=[metric] if metric else [], config_keys=fetch_config_keys, filters=filters, order=order, limit=args.limit)
    unmatched = scanned = 0
    for row in fetched_rows:
        scanned += 1
        raw_text = run_table_raw_search_text(row)
        text = run_table_search_text(row)
        matched_specs = []
        for name, rx in specs:
            if rx.search(text):
                observed = None
                token_match = re.search(re.escape(name), raw_text, re.I) if name else None
                if token_match:
                    observed = token_match.group(0)
                matched_specs.append((name, observed))
        names = [name for name, _ in matched_specs]
        if extract:
            names += [m.group(1) if m.groups() else m.group(0) for m in extract.finditer(text)]
        if not names:
            unmatched += 1
        for name in set(filter(None, names)):
            stratum = "|".join(f"{key}={(row.get('config') or {}).get(key)}" for key in stratify_by) if stratify_by else None
            observed = next((token for matched_name, token in matched_specs if matched_name == name and token), None)
            buckets[name].append({"run": row["name"], "id": row["id"], "state": row.get("state"), "metric_value": coerce_float((row.get("summary") or {}).get(metric)) if metric else None, "stratum": stratum, "matched_token": observed, "case_mismatch": bool(observed and observed != name)})
        if fast_best and specs and len(buckets) >= len(specs):
            break
    rows = []
    for name, items in buckets.items():
        numeric = [i for i in items if i["metric_value"] is not None]
        row = {"variant": name, "count": len(items), "metric_count": len(numeric)}
        matched_tokens = sorted({str(i["matched_token"]) for i in items if i.get("matched_token")})
        case_mismatch_count = sum(1 for i in items if i.get("case_mismatch"))
        if matched_tokens:
            row["matched_token_examples"] = matched_tokens[:5]
        if case_mismatch_count:
            row["case_mismatch_count"] = case_mismatch_count
        if numeric:
            vals = [i["metric_value"] for i in numeric]
            best = (min if direction == "min" else max)(numeric, key=lambda i: i["metric_value"])
            row |= {"best_run": best["run"], "best_run_id": best["id"], "best_metric_value": best["metric_value"], "mean_metric_value": statistics.mean(vals), "median_metric_value": statistics.median(vals)}
            if stratify_by:
                by_stratum: dict[str, list[float]] = defaultdict(list)
                for item in numeric:
                    if item.get("stratum"):
                        by_stratum[item["stratum"]].append(item["metric_value"])
                stratum_means = {key: statistics.mean(values) for key, values in by_stratum.items()}
                row |= {"stratify_by": stratify_by, "strata_count": len(stratum_means), "balanced_mean_metric_value": statistics.mean(stratum_means.values()) if stratum_means else None, "_stratum_mean_values": stratum_means}
        rows.append(row)
    common_strata: set[str] = set()
    if stratify_by and metric:
        stratum_sets = [set(r.get("_stratum_mean_values", {})) for r in rows if r.get("_stratum_mean_values")]
        common_strata = set.intersection(*stratum_sets) if stratum_sets else set()
        for row in rows:
            means = row.pop("_stratum_mean_values", {})
            row["common_strata_count"] = len(common_strata)
            if means and common_strata and common_strata.issubset(means):
                row["common_balanced_mean_metric_value"] = statistics.mean(means[key] for key in common_strata)
            if means:
                row["stratum_mean_values"] = dict(sorted(means.items()))
    if metric:
        rank_field = {"mean": "mean_metric_value", "median": "median_metric_value", "balanced_mean": "common_balanced_mean_metric_value" if stratify_by else "mean_metric_value", "best": "best_metric_value", "count": "count"}[args.rank_by]
        if args.rank_by == "count":
            rows.sort(key=lambda r: r.get("count", 0), reverse=True)
        else:
            rows.sort(key=lambda r: (r.get(rank_field) is None, r.get(rank_field) if direction == "min" else -(r.get(rank_field) or 0)))
    else:
        rows.sort(key=lambda r: r["count"], reverse=True)
    print(f"path={path}")
    print(f"scanned_runs={scanned}")
    print(f"selective_run_table_fetch=summaryMetrics(keys={[metric] if metric else []})")
    print(f"order={order}")
    print(f"status_scope={args.state or 'all_states'}")
    print(f"unmatched_runs={unmatched}")
    if stratify_by:
        print(f"stratification=keys={stratify_by}; common_strata={len(common_strata)}")
    elif metric and args.rank_by in {"mean", "median"}:
        print("stratification=pooled_unstratified; stratify_by=[]; report this as a pooled aggregate or rerun with --stratify-by for balanced strata")
    for row in rows:
        if row.get("case_mismatch_count"):
            print(f"case_caveat=variant {row['variant']} matched observed token(s) {row.get('matched_token_examples')} case-insensitively; mention this normalization in the answer")
    if filter_caveat:
        print(f"filter_caveat={filter_caveat}")
    if has_more:
        print(f"limit_caveat=reached --limit {args.limit}; increase --limit for exact aggregates over larger filtered cohorts")
    if metric:
        print(f"metric={metric}")
        print(f"direction={direction}")
        print(f"rank_by={args.rank_by}")
    for row in rows[: args.top]:
        print(json.dumps(row, sort_keys=True, default=str))
    if rows:
        best = rows[0]
        if metric and best.get("best_metric_value") is not None:
            stratification_note = f"keys={stratify_by}" if stratify_by else "pooled_unstratified"
            print(f"answer=best_variant={best['variant']}; metric={metric}; direction={direction}; rank_by={args.rank_by}; stratification={stratification_note}; mean={best.get('mean_metric_value')}; median={best.get('median_metric_value')}; balanced_mean={best.get('balanced_mean_metric_value')}; common_balanced_mean={best.get('common_balanced_mean_metric_value')}; best_value={best.get('best_metric_value')}; best_run={best.get('best_run')}; count={best['count']}")
        else:
            print(f"answer=status_scope={args.state or 'all_states'}; most_common_variant={best['variant']}; most_common_count={best['count']}; counts=" + ", ".join(f"{r['variant']}={r['count']}" for r in rows[: args.top]))


def cmd_rank_runs(args: argparse.Namespace) -> None:
    """Print latest or metric-ranked runs with selected config/summary fields."""
    wb, path = ctx(args)
    filters = filters_from_args(args) or None
    metric = metric_from_args(wb, path, args, filters)
    direction = (numeric_direction(metric, first(args.metric_hint)) if getattr(args, "direction", "auto") == "auto" else args.direction) if metric else None
    order = args.order or ("-created_at" if args.latest or not metric else metric_order(metric, direction))
    print(f"path={path}")
    print(f"order={order}")
    if metric:
        print(f"metric={metric}")
        print(f"direction={direction}")
    needs_config = bool(args.hparams or args.config_key)
    summary_keys = list(dict.fromkeys([key for key in [metric, *args.summary_key] if key]))
    if summary_keys and not needs_config:
        fetched, has_more = fetch_run_table_rows(wb, path, summary_keys=summary_keys, filters=filters, order=order, limit=args.limit)
        print_sample_scope(
            "rank-runs",
            requested=args.limit,
            effective=args.limit,
            actual=len(fetched),
            unit="ranked_rows",
            strategy=f"selective run-table rows ordered by {order}",
            cap_applied=False,
            project_exhaustive=not has_more,
            note="ranked output covers returned rows only; increase --limit for a larger ranked slice",
        )
        if has_more:
            print(f"limit_caveat=reached --limit {args.limit}; increase --limit for more ranked rows")
        if fetched and args.answer:
            best = fetched[0]
            parts = [f"best_run={best['name']}", f"id={best['id']}"]
            if args.latest:
                parts.append(f"created_at={best.get('created_at')}")
            if metric:
                parts.append(f"{metric}={(best.get('summary') or {}).get(metric)}")
            prefix = f"ranked_by={metric}; " if metric else ""
            print(f"answer={prefix}" + "; ".join(parts))
        for row in fetched:
            out = {"name": row["name"], "id": row["id"], "state": row.get("state"), "created_at": row.get("created_at")}
            out |= {key: (row.get("summary") or {}).get(key) for key in summary_keys if (row.get("summary") or {}).get(key) is not None}
            print(json.dumps(out, sort_keys=True, default=str))
        return

    row_limit = capped_hydrated_limit(args.limit) if needs_config else args.limit
    rows = runs(wb, path, args, filters=filters, order=order, limit=row_limit, lazy=not needs_config)
    print_sample_scope(
        "rank-runs",
        requested=args.limit,
        effective=row_limit,
        actual=len(rows),
        unit="ranked_rows",
        strategy=f"{'hydrated SDK runs with config' if needs_config else 'lazy SDK runs'} ordered by {order}",
        note="ranked output covers returned rows only; increase --limit for a larger ranked slice",
    )
    print_cap_caveat("rank-runs", args.limit, row_limit)
    if rows and args.answer:
        best, config = rows[0], clean_config(rows[0].config) if needs_config else {}
        keys = likely_hparam_keys(config, args.config_key) if args.hparams else list(args.config_key)
        parts = [f"best_run={run_label(best)}", f"id={best.id}"]
        if args.latest:
            parts.append(f"created_at={best.created_at}")
        if metric:
            parts.append(f"{metric}={best.summary_metrics.get(metric)}")
        parts += [f"{k}={scalar_config_value(config.get(k))}" for k in keys]
        # Lead with `ranked_by=<metric>` so the agent labels the ranking
        # metric explicitly when composing the user-facing response.
        prefix = f"ranked_by={metric}; " if metric else ""
        print(f"answer={prefix}" + "; ".join(parts))
    for run in rows:
        config = clean_config(run.config) if needs_config else {}
        keys = likely_hparam_keys(config, args.config_key) if args.hparams else list(args.config_key)
        row = {"name": run_label(run), "id": run.id, "state": getattr(run, "state", None), "created_at": getattr(run, "created_at", None)}
        if metric:
            row[metric] = run.summary_metrics.get(metric)
        row |= {k: scalar_config_value(config.get(k)) for k in keys} | {k: run.summary_metrics.get(k) for k in args.summary_key}
        print(json.dumps(row, sort_keys=True, default=str))


# -----------------------------------------------------------------------------
# Summary tables
# -----------------------------------------------------------------------------


def is_table_summary_value(value: Any) -> bool:
    """Return whether a W&B summary value points at a table JSON file."""
    if not isinstance(value, dict):
        return False
    path = str(value.get("path") or "")
    value_type = str(value.get("_type") or "").lower()
    return value_type == "table-file" or path.lower().endswith(".table.json")


def table_payload(run: Any, key: str, *, report_errors: bool = False) -> tuple[list[str], list[dict[str, Any]]] | None:
    value = (run.summary_metrics or {}).get(key)
    if not isinstance(value, dict) or not value.get("path"):
        return None
    if not is_table_summary_value(value):
        if report_errors:
            print(f"summary_key_not_table run={run_label(run)} summary_key={key} type={value.get('_type')} path={value.get('path')}")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run.file(value["path"]).download(root=tmp, replace=True)
            with open(os.path.join(tmp, value["path"])) as handle:
                payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if report_errors:
                print(f"table_parse_error run={run_label(run)} summary_key={key} error={type(exc).__name__}: {exc}")
            return None
    if not isinstance(payload, dict):
        if report_errors:
            print(f"table_parse_error run={run_label(run)} summary_key={key} error=payload is not a JSON object")
        return None
    cols = [str(c) for c in payload.get("columns") or []]
    return cols, [{c: row[i] if i < len(row) else None for i, c in enumerate(cols)} for row in payload.get("data") or []]


def table_keys(run: Any, explicit: str | None) -> list[str]:
    return [explicit] if explicit else [k for k, v in sorted((run.summary_metrics or {}).items()) if is_table_summary_value(v)]


def truthy(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else ((num > 0) if (num := coerce_float(value)) is not None else str(value).lower() in {"true", "yes", "success", "succeeded"})


def normalize_text(text: str) -> str:
    """Normalize free text for hint matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def hint_matches_text(text: str, hints: list[str]) -> bool:
    """Return true when every hint appears in normalized text."""
    haystack = normalize_text(text)
    return not hints or all(normalize_text(hint) in haystack for hint in hints)


def column_from_hints(columns: list[str], explicit: str | None, hints: list[str], defaults: list[str]) -> str | None:
    """Choose a table column using explicit name, hints, then defaults."""
    if explicit and explicit in columns:
        return explicit
    for hint in hints:
        match = next((c for c in columns if normalize_text(hint) in normalize_text(c)), None)
        if match:
            return match
    return next((c for c in columns if normalize_text(c) in defaults or any(d in normalize_text(c) for d in defaults)), None)


def numeric_columns(columns: list[str], rows_: list[dict[str, Any]]) -> list[str]:
    """Return columns with at least one numeric value."""
    return [c for c in columns if any(coerce_float(row.get(c)) is not None for row in rows_)]


def print_table_snapshot(tables: list[dict[str, Any]], args: argparse.Namespace, path: str, scanned: int) -> None:
    """Print table metadata, selected rows, and progression evidence."""
    print(f"path={path}")
    print(f"summary_key={first(args.summary_key, 'auto')}")
    print(f"scanned_runs={scanned}")
    filtered = [t for t in tables if hint_matches_text(" ".join([t["summary_key"], *t["columns"]]), args.table_hint)]
    print(f"tables_found={len(filtered)}")
    if not filtered:
        print("answer=no matching summary table found"); return
    table, rows_ = filtered[0], filtered[0]["rows"]
    columns = table["columns"]
    label_col = column_from_hints(columns, args.label_column, args.label_hint, ["model", "name", "variant", "recipe", "group", "size"])
    sort_col = column_from_hints(columns, args.sort_column, args.sort_hint, ["parameter", "params", "million", "flop", "size", "cost"])
    metric_cols = [c for c in args.metric_column if c in columns] or [c for c in columns if hint_matches_text(c, args.metric_hint) and c in numeric_columns(columns, rows_)]
    if not metric_cols:
        metric_cols = [c for c in numeric_columns(columns, rows_) if any(t in normalize_text(c) for t in PERFORMANCE_METRIC_TOKENS)][:3]
    if sort_col:
        rows_ = sorted(rows_, key=lambda r: (coerce_float(r.get(sort_col)) is None, coerce_float(r.get(sort_col)) or 0))
    emit("table", {"run": table["run"], "id": table["id"], "summary_key": table["summary_key"], "columns": columns, "label_column": label_col, "sort_column": sort_col, "metric_columns": metric_cols, "row_count": len(rows_)})
    limited = rows_[: args.row_limit]
    for row in limited:
        emit("row", row)
    progression = []
    if label_col and metric_cols:
        for row in limited:
            bits = [str(row.get(label_col))]
            if sort_col: bits.append(f"{sort_col}={row.get(sort_col)}")
            bits += [f"{m}={row.get(m)}" for m in metric_cols]
            progression.append(" ".join(bits))
    if progression:
        first_row, last_row = limited[0], limited[-1]
        deltas = []
        for metric in metric_cols:
            a, b = coerce_float(first_row.get(metric)), coerce_float(last_row.get(metric))
            if a is not None and b is not None:
                deltas.append(f"{metric}: {a:g}->{b:g} (delta {b-a:g})")
        print("progression=" + " | ".join(progression))
        print(f"answer=table_key={table['summary_key']}; label_column={label_col}; sort_column={sort_col}; metrics={', '.join(metric_cols)}; first={first_row.get(label_col)}; last={last_row.get(label_col)}; {'; '.join(deltas)}")
        print("interpretation_hint=Use the table rows as the progression, ordered by the sort column when present. Report first-to-last metric deltas and cost/tradeoff columns from table evidence instead of inferring trends from run names.")
    else:
        print("answer=printed matching table rows")


def cmd_summary_table(args: argparse.Namespace) -> None:
    """Compute success rates or print snapshot rows from W&B summary tables."""
    wb, path = ctx(args)
    sample = runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=args.limit)
    if args.name_prefix:
        sample = [r for r in sample if run_label(r).lower().startswith(args.name_prefix.lower())]
    tables = []
    report_table_errors = bool(args.verbose_errors or first(args.summary_key))
    for run in sample:
        for key in table_keys(run, first(args.summary_key)):
            if loaded := table_payload(run, key, report_errors=report_table_errors):
                cols, rows_ = loaded
                tables.append({"run": run_label(run), "id": run.id, "summary_key": key, "columns": cols, "rows": rows_})
    if args.mode == "snapshot":
        print_table_snapshot(tables, args, path, len(sample))
        return
    print(f"path={path}")
    print(f"summary_key={first(args.summary_key, 'auto')}")
    print(f"scanned_runs={len(sample)}")
    print(f"tables_found={len(tables)}")
    total = ok = 0
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for table in tables:
        cols = table["columns"]
        group_col = args.group_column if args.group_column in cols else next((c for c in cols if c.lower() in {"group", "category", "type", "task", "label", "name"}), None)
        run_total = run_ok = 0
        for row in table["rows"]:
            passed = truthy(row.get(args.success_column))
            total += 1; ok += int(passed); run_total += 1; run_ok += int(passed)
            if group_col and row.get(group_col) is not None:
                by_group[str(row[group_col])][0] += int(passed); by_group[str(row[group_col])][1] += 1
        print(json.dumps({"run": table["run"], "id": table["id"], "summary_key": table["summary_key"], "rows": run_total, "successes": run_ok, "success_rate": run_ok / run_total if run_total else None}, sort_keys=True, default=str))
    lowest = min(((g, s, n, s / n if n else 0.0) for g, (s, n) in by_group.items()), key=lambda x: x[3], default=None)
    if by_group:
        print("group_success_rates=")
        for g, (s, n) in sorted(by_group.items()):
            emit("group_success_rate", {"group": g, "rows": n, "successes": s, "success_rate": s / n if n else 0.0})
    if lowest:
        emit("lowest_group", {"group": lowest[0], "rows": lowest[2], "successes": lowest[1], "success_rate": lowest[3]})
    print(f"answer=success_rate={(ok / total if total else 0.0):.4f}; successes={ok}; total={total}; tables={len(tables)}" + (f"; lowest_group={lowest[0]}; lowest_group_success_rate={lowest[3]:.4f}; lowest_group_successes={lowest[1]}; lowest_group_total={lowest[2]}" if lowest else ""))


# -----------------------------------------------------------------------------
# Cohort and config comparison
# -----------------------------------------------------------------------------

def run_step_budget(run: Any) -> int | None:
    """last-logged-step proxy for history completeness; from metadata/summary, no history fetch."""
    step = getattr(run, "lastHistoryStep", None)
    if isinstance(step, (int, float)) and step >= 0:
        return int(step)
    summ = run.summary_metrics or {}
    for key in ("_step", "train/global_step", "global_step", "step"):
        value = coerce_float(summ.get(key))
        if value is not None:
            return int(value)
    return None

def budget_stats(steps: list[Any]) -> dict[str, Any]:
    """Report the spread of logged step counts across a cohort; A completeness fact, not a verdict."""
    vals = sorted(s for s in steps if isinstance(s, int) and s >= 0)
    if not vals:
        return {}
    return {"steps_min": vals[0], "steps_median": int(statistics.median(vals)), "steps_max": vals[-1]}


def cohort_name_pattern(selector: str) -> str | None:
    return selector.split(":", 1)[1] if selector.startswith(COHORT_NAME_SELECTOR_PREFIXES) else None


def valid_cohort_selector(selector: str) -> bool:
    """Return whether a cohort selector matches the documented grammar."""
    return selector == "non_sweep" or selector == "sweep:any" or selector.startswith(("sweep:", "tag:", "group:", *COHORT_NAME_SELECTOR_PREFIXES))


def parse_cohort_specs(values: list[str] | str) -> list[tuple[str, str]]:
    """Parse and validate NAME=SELECTOR cohort specs before any API calls."""
    cohort_args = values if isinstance(values, list) else [values]
    specs = []
    for spec in cohort_args:
        if "=" not in spec:
            raise SystemExit(f"--cohort must be NAME=SELECTOR. {COHORT_SELECTOR_HELP}")
        name, selector = (part.strip() for part in spec.split("=", 1))
        if not name or not selector:
            raise SystemExit(f"--cohort must be NAME=SELECTOR. {COHORT_SELECTOR_HELP}")
        if not valid_cohort_selector(selector):
            raise SystemExit(f"Unknown cohort selector {selector!r}. {COHORT_SELECTOR_HELP}")
        if cohort_name_pattern(selector) is not None:
            try:
                re.compile(cohort_name_pattern(selector) or "")
            except re.error as exc:
                raise SystemExit(f"Invalid run-name cohort selector {selector!r}: {exc}") from exc
        specs.append((name, selector))
    return specs


def cohort_row(wb: Any, path: str, name: str, selector: str, metric: str, direction: str, limit: int, max_sweeps: int = 50) -> dict[str, Any]:
    """Return the best metric row for one cohort selector."""
    if selector == "sweep:any":
        entity, project = split_path(path)
        candidates, sweep_rows = [], []
        sweeps = list(wb.project(project, entity=entity).sweeps(per_page=max_sweeps))[:max_sweeps]
        for sweep in sweeps:
            run_count = count(wb, path, {"sweep": sweep.id})
            sweep_rows.append(
                {
                    "sweep_id": sweep.id,
                    "name": getattr(sweep, "name", None),
                    "state": getattr(sweep, "state", None),
                    "run_count": run_count,
                }
            )
            best = wb.runs(path, filters={"state": "finished", "sweep": sweep.id}, order=metric_order(metric, direction), per_page=1)[:1]
            if best and (value := coerce_float(best[0].summary_metrics.get(metric))) is not None:
                candidates.append((value, best[0], sweep.id))
        total = sum(row["run_count"] for row in sweep_rows)
        sweep_summary = {
            "count": total,
            "scanned_sweeps": len(sweeps),
            "sweep_count": len(sweep_rows),
            "sweep_counts_scope": f"first_{max_sweeps}",
            "total_sweep_runs": total,
            "sweeps": sweep_rows,
        }
        if not candidates:
            return {"cohort": name, "selector": selector, **sweep_summary, "best": None}
        value, run, sweep_id = (min if direction == "min" else max)(candidates, key=lambda x: x[0])
        return {"cohort": name, "selector": selector, **sweep_summary, "best_run": run_label(run), "best_run_id": run.id, "sweep_id": sweep_id, "best_metric_value": value}
    filters, keep = {"state": "finished"}, lambda run: True
    if selector.startswith("sweep:"):
        filters["sweep"] = selector.split(":", 1)[1]
    elif selector == "non_sweep":
        keep = lambda run: getattr(run, "sweep", None) is None
    elif selector.startswith("tag:"):
        filters["tags"] = {"$in": [selector.split(":", 1)[1]]}
    elif selector.startswith("group:"):
        filters["group"] = selector.split(":", 1)[1]
    elif (pattern := cohort_name_pattern(selector)) is not None:
        regex = re.compile(pattern); keep = lambda run: bool(regex.search(run_label(run)))
    else:
        raise SystemExit(f"Unknown cohort selector {selector!r}. {COHORT_SELECTOR_HELP}")
    scanned = list(wb.runs(path, filters=filters, order=metric_order(metric, direction), per_page=max(limit, 50), include_sweeps=True)[:limit])
    candidates = [(value, run) for run in scanned if keep(run) for value in [coerce_float(run.summary_metrics.get(metric))] if value is not None]
    if not candidates:
        return {"cohort": name, "selector": selector, "scanned_runs": len(scanned), "count": 0, "best": None}
    value, run = (min if direction == "min" else max)(candidates, key=lambda x: x[0])
    budget = budget_stats([run_step_budget(r) for _, r in candidates])
    return {"cohort": name, "selector": selector, "scanned_runs": len(scanned), "count": len(candidates), "best_run": run_label(run), "best_run_id": run.id, "best_metric_value": value, "best_run_steps": run_step_budget(run), **budget}

def cmd_compare_cohorts(args: argparse.Namespace) -> None:
    """Compare the best metric value across requested cohorts."""
    specs = parse_cohort_specs(args.cohort)
    wb, path = ctx(args)
    metric, hint = first(args.metric), first(args.metric_hint)
    if not metric:
        if not hint:
            raise SystemExit("Pass --metric METRIC or --metric-hint HINT")
        metric = sample_metric_from_hint(wb, path, hint, args.sample)
    direction = numeric_direction(metric, hint) if args.direction == "auto" else args.direction
    selectors = {selector for _, selector in specs}
    requested_scan_limit = max(getattr(args, "scan", args.sample), args.limit)
    scan_limit = requested_scan_limit if args.large_scan else capped_hydrated_limit(requested_scan_limit, COHORT_SCAN_CAP)
    requested_sweep_limit = args.sweeps
    sweep_limit = requested_sweep_limit if args.large_scan else capped_hydrated_limit(requested_sweep_limit, COHORT_SWEEP_CAP)
    rows_ = [cohort_row(wb, path, name, selector, metric, direction, scan_limit, sweep_limit) for name, selector in specs]
    print(f"path={path}")
    print(f"metric={metric}")
    print(f"direction={direction}")
    print_effective_scan_limit("compare-cohorts", requested_scan_limit, scan_limit, unit="runs_per_cohort", large_scan=args.large_scan)
    print_limit_caveat("compare-cohorts", requested_scan_limit, scan_limit)
    if "sweep:any" in selectors:
        print_effective_scan_limit("compare-cohorts", requested_sweep_limit, sweep_limit, unit="sweeps", large_scan=args.large_scan)
        print_limit_caveat("compare-cohorts sweeps", requested_sweep_limit, sweep_limit)
    print_sample_scope(
        "compare-cohorts",
        requested=requested_scan_limit,
        effective=scan_limit,
        actual=sum(row.get("scanned_runs", row.get("scanned_sweeps", 0)) for row in rows_),
        unit="candidate_runs_or_sweeps",
        strategy=f"bounded SDK runs ordered by {metric_order(metric, direction)}; local selectors only see the bounded scan",
        note="best-run comparisons are only across the effective scan for each cohort; pass --large-scan for an uncapped expensive scan",
    )
    for row in rows_:
        print(json.dumps(row, sort_keys=True, default=str))
    if len(rows_) == 2 and all(r.get("best_metric_value") is not None for r in rows_):
        left, right = rows_
        better = left["best_metric_value"] < right["best_metric_value"] if direction == "min" else left["best_metric_value"] > right["best_metric_value"]
        op = ("<" if direction == "min" else ">") if better else (">=" if direction == "min" else "<=")
        print(f"answer={left['cohort']} {'was better than' if better else 'was not better than'} {right['cohort']} on {metric} ({direction} is better): {left['best_run']}={left['best_metric_value']} {op} {right['best_run']}={right['best_metric_value']}")


def config_diff_run_refs(args: argparse.Namespace) -> list[str]:
    refs: list[str] = []
    if args.run_a or args.run_b:
        if not (args.run_a and args.run_b):
            raise SystemExit("config-diff requires both --run-a RUN and --run-b RUN")
        refs.extend([args.run_a, args.run_b])
    refs.extend(getattr(args, "run", []) or [])
    refs.extend(getattr(args, "runs", []) or [])
    unique = list(dict.fromkeys(ref for ref in refs if ref))
    if refs and len(unique) < 2:
        raise SystemExit("config-diff explicit run comparison requires at least two distinct run refs")
    return unique


def resolve_config_diff_runs(wb: Any, path: str, refs: list[str], *, include_sweeps: bool) -> list[Any]:
    resolved = []
    for ref in refs:
        run = resolve_run_ref(
            wb,
            path,
            ref,
            limit=max(200, len(refs)),
            include_sweeps=include_sweeps,
            prefer_direct_id=should_probe_direct_run_id(False, ref),
        )
        if run is None:
            raise SystemExit(
                f"Could not find config-diff run {ref!r}. "
                "Use a W&B run id, exact display name, or unique visible run-name suffix."
            )
        resolved.append(run)
    return resolved


def cmd_config_diff(args: argparse.Namespace) -> None:
    """Print varying and shared config keys across bounded runs."""
    wb, path = ctx(args)
    explicit_refs = config_diff_run_refs(args)
    if explicit_refs:
        sample_limit = len(explicit_refs)
        sample = resolve_config_diff_runs(wb, path, explicit_refs, include_sweeps=args.include_sweeps)
    else:
        sample_limit = capped_hydrated_limit(args.limit)
        sample = runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=sample_limit, lazy=False)
    sample = filter_names(sample, args.name_include, args.name_exclude)
    values: dict[str, dict[str, str]] = defaultdict(dict)
    for run in sample:
        for key, value in clean_config(run.config).items():
            if not any(skip in key.lower() for skip in args.skip_key_contains):
                values[key][run_label(run)] = jsonable(value)
    varying = {k: v for k, v in values.items() if len(set(v.values())) > 1}
    shared = [k for k, v in values.items() if len(set(v.values())) == 1]
    print(f"path={path}")
    print("run_selection=" + ("explicit_refs" if explicit_refs else "sample"))
    if not explicit_refs:
        print_sample_scope(
            "config-diff",
            requested=args.limit,
            effective=sample_limit,
            actual=len(sample),
            strategy=f"hydrated SDK runs ordered by {args.order}",
            note="config differences are computed only across these sampled runs",
        )
        print_cap_caveat("config-diff", args.limit, sample_limit)
    print(f"runs={len(sample)}")
    print("run_names=" + ", ".join(run_label(r) for r in sample))
    print(f"varying_config_keys={len(varying)}")
    for key in sorted(varying):
        print(f"\n[{key}]")
        for run_name, value in sorted(varying[key].items()):
            print(f"{run_name}: {value}")
    print(f"\nshared_config_keys={len(shared)}")
    print(", ".join(sorted(shared)[: args.shared_limit]))


# -----------------------------------------------------------------------------
# Metric grid comparison
# -----------------------------------------------------------------------------


def grid_rows(sample: list[Any], axes: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    rows_ = []
    for run in sample:
        config, row = clean_config(run.config), {"run": run_label(run), "id": run.id, "state": getattr(run, "state", None), "steps": run_step_budget(run)}
        if scale := run_scale_terms(row["run"]):
            row["run_scale_terms"] = scale
        row |= {f"config.{axis}": scalar_config_value(config.get(axis)) for axis in axes}
        row |= {metric: value for metric in metrics if (value := coerce_float((run.summary_metrics or {}).get(metric))) is not None}
        if any(metric in row for metric in metrics):
            rows_.append(row)
    return rows_


def grid_rows_from_run_table(rows: list[dict[str, Any]], axes: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    rows_ = []
    for source in rows:
        config = source.get("config") or {}
        summary = source.get("summary") or {}
        row = {"run": source["name"], "id": source["id"], "state": source.get("state"), "steps": step_budget_from_summary(summary)}
        if scale := run_scale_terms(row["run"]):
            row["run_scale_terms"] = scale
        row |= {f"config.{axis}": scalar_config_value(config.get(axis)) for axis in axes}
        row |= {metric: value for metric in metrics if (value := coerce_float(summary.get(metric))) is not None}
        if any(metric in row for metric in metrics):
            rows_.append(row)
    return rows_


def auto_axes(sample: list[Any], explicit: list[str], limit: int) -> list[str]:
    values: dict[str, set[str]] = defaultdict(set)
    for run in sample:
        for key, value in clean_config(run.config).items():
            if (scalar := scalar_config_value(value)) is not None:
                values[str(key)].add(str(scalar))
    priority = {key: index for index, key in enumerate(HPARAM_PRIORITY_KEYS)}
    scored = []
    for key, seen in values.items():
        if key in explicit or not 1 < len(seen) <= 16:
            continue
        lower = key.lower()
        score = 40 * (key in priority) + 10 * (len(seen) >= 2) + min(len(seen), 6)
        score += 9 * any(t in lower for t in CONFIG_AXIS_TOKENS)
        scored.append((score, key))
    return [*explicit, *[key for _, key in sorted(scored, reverse=True)[: max(0, limit - len(explicit))]]]


def parse_key_value_specs(specs: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE config filters."""
    return parse_config_eq_specs(specs)


def config_matches(config: dict[str, Any], required: dict[str, str]) -> bool:
    """Return whether scalar config values satisfy all required strings."""
    return all((actual := scalar_config_value(config.get(k))) is not None and str(actual).lower() == v.lower() for k, v in required.items())


def metric_nfe_value(key: str) -> int | None:
    """Extract an NFE-like integer from a metric key."""
    values = [int(m) for m in re.findall(r"(?:^|[/_-])n(\d+)", key.lower())]
    return max(values) if values else None


def selected_regime_metrics(metrics: list[str]) -> list[str]:
    """Pick headline and low-NFE metrics for compact regime summaries."""
    selected = metrics[:1]
    nfe = [(metric_nfe_value(m), m) for m in metrics]
    low = min((item for item in nfe if item[0] is not None), default=None)
    if low and low[1] not in selected:
        selected.append(low[1])
    return selected[:2]


def metric_grid_sort_key(key: str) -> tuple[Any, ...]:
    """Prefer canonical/best validation metrics and larger NFE metrics."""
    lower = key.lower(); nfe = metric_nfe_value(lower) or -1
    return (0 if lower in {"best_val_loss", "best/val_loss", "best-validation-loss"} else 1, 0 if "best" in lower else 1, -nfe, metric_sort_key(key, prefer_comprehensive=True), key)


def metric_grid_keys(sample: list[Any], hints: list[str], explicit: list[str], limit: int) -> list[str]:
    """Select numeric summary metrics for a grid."""
    keys: set[str] = set(explicit)
    for run in sample:
        for key, value in (run.summary_metrics or {}).items():
            if coerce_float(value) is None:
                continue
            key = str(key); lower = key.lower()
            if key in explicit or (hints and any(metric_hint_matches(key, h) for h in hints)) or (not explicit and not hints and any(t in lower for t in PERFORMANCE_METRIC_TOKENS)):
                keys.add(key)
    if not keys and hints:
        raise SystemExit(f"No numeric summary metrics matched hints {hints}")
    return sorted(keys, key=metric_grid_sort_key)[:limit]


def axis_summary(rows_: list[dict[str, Any]], axis: str, metric: str, direction: str) -> dict[str, Any] | None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        value = row.get(f"config.{axis}")
        if value is not None and metric in row:
            groups[str(value)].append(row)
    if not groups:
        return None
    values = []
    for value, group in groups.items():
        best = (min if direction == "min" else max)(group, key=lambda r: r[metric])
        values.append({"value": value, "count": len(group), "best": best[metric], "mean": statistics.fmean(r[metric] for r in group), "best_run": best["run"]})
    values.sort(key=lambda r: r["best"], reverse=direction == "max")
    return {"axis": axis, "metric": metric, "direction": direction, "unique_values": len(values), "values": values}


def axis_pair_summary(rows_: list[dict[str, Any]], axes: list[str], metric: str, direction: str) -> dict[str, Any] | None:
    """Summarize best metric values for the first two config axes."""
    if len(axes) < 2:
        return None
    pair, groups = axes[:2], defaultdict(list)
    for row in rows_:
        first, second = row.get(f"config.{pair[0]}"), row.get(f"config.{pair[1]}")
        if first is not None and second is not None and metric in row:
            groups[(str(first), str(second))].append(row)
    rows_out = []
    for (first, second), group in groups.items():
        best = (min if direction == "min" else max)(group, key=lambda r: r[metric])
        rows_out.append({pair[0]: first, pair[1]: second, "count": len(group), "best": best[metric], "mean": statistics.fmean(r[metric] for r in group), "best_run": best["run"], "id": best["id"], **budget_stats([r.get("steps") for r in group])})
    if not rows_out:
        return None
    return {"axes": pair, "metric": metric, "direction": direction, "groups": sorted(rows_out, key=lambda r: (r[pair[1]], r[pair[0]], r["best"] if direction == "min" else -r["best"]))}


def cmd_metric_grid(args: argparse.Namespace) -> None:
    """Print metric winners and config-axis summaries for a W&B run grid."""
    wb, path = ctx(args)
    req = parse_key_value_specs(args.config_eq)
    filters = filters_from_args(args) or None
    has_more = False
    if args.axis and args.metric:
        axes = args.axis
        metrics = args.metric
        config_keys = list(dict.fromkeys([*axes, *req]))
        fetched_rows, has_more = fetch_run_table_rows(
            wb,
            path,
            summary_keys=list(dict.fromkeys([*metrics, *STEP_SUMMARY_KEYS])),
            config_keys=config_keys,
            filters=filters,
            order=args.order,
            limit=args.limit,
        )
        if req:
            fetched_rows = [row for row in fetched_rows if config_matches(row.get("config") or {}, req)]
        sample_count = len(fetched_rows)
        rows_ = grid_rows_from_run_table(fetched_rows, axes, metrics)
        method = "selective GraphQL summary/config keys"
    else:
        sample_limit = capped_hydrated_limit(args.limit)
        sample = filter_names(runs(wb, path, args, filters=filters, order=args.order, limit=sample_limit, lazy=False), args.name_include, args.name_exclude)
        if req:
            sample = [r for r in sample if config_matches(clean_config(r.config), req)]
        axes = auto_axes(sample, args.axis, args.max_axes)
        metrics = metric_grid_keys(sample, args.metric_hint, args.metric, args.metric_limit)
        rows_ = grid_rows(sample, axes, metrics)
        sample_count = len(sample)
        method = "hydrated SDK sample"
    print(f"path={path}")
    print(f"method={method}")
    print_sample_scope(
        "metric-grid",
        requested=args.limit,
        effective=args.limit if args.axis and args.metric else sample_limit,
        actual=sample_count,
        unit="runs_considered",
        strategy=method,
        cap_applied=False if args.axis and args.metric else sample_limit < args.limit,
        project_exhaustive=not has_more and bool(args.axis and args.metric),
        note="metric-grid summaries and winners are computed only from runs_considered",
    )
    if not (args.axis and args.metric):
        print_cap_caveat("metric-grid", args.limit, sample_limit)
    if has_more:
        print(f"limit_caveat=reached --sample {args.limit}; increase --sample for more rows")
    print(f"runs_considered={sample_count}")
    if req:
        emit("config_filters", req)
    print("axes=" + ", ".join(axes))
    print("metrics=" + ", ".join(metrics))
    for row in rows_[: args.row_limit]:
        emit("metric_grid_row", row)
    constants = {}
    for axis in axes:
        vals = sorted({str(row.get(f"config.{axis}")) for row in rows_ if row.get(f"config.{axis}") is not None})
        if len(vals) == 1:
            constants[axis] = vals[0]
            print(f"constant_axis={axis}:{vals[0]}")
    summaries = []
    for metric in selected_regime_metrics(metrics):
        direction = numeric_direction(metric) if args.direction == "auto" else args.direction
        metric_rows = [r for r in rows_ if metric in r]
        if metric_rows:
            best = (min if direction == "min" else max)(metric_rows, key=lambda r: r[metric])
            emit("best_by_metric", {"metric": metric, "direction": direction, "run": best["run"], "id": best["id"], "value": best[metric], "run_scale_terms": best.get("run_scale_terms", []), "axes": {a: best.get(f"config.{a}") for a in axes}})
            if pair := axis_pair_summary(rows_, axes, metric, direction):
                emit("axis_pair_metric_summary", pair)
            for axis in axes:
                if summary := axis_summary(rows_, axis, metric, direction):
                    summaries.append(summary)
                    emit("axis_metric_summary", summary)
    if metrics and rows_:
        metric, direction = metrics[0], (numeric_direction(metrics[0]) if args.direction == "auto" else args.direction)
        ordered = sorted([r for r in rows_ if metric in r], key=lambda r: r[metric], reverse=direction == "max")
        best, runner = ordered[0], ordered[1] if len(ordered) > 1 else None
        gap = abs(float(best[metric]) - float(runner[metric])) if runner else None
        print(f"answer=rows={len(rows_)}; axes={', '.join(axes)}; headline_metric={metric}; direction={direction}; best_run={best['run']}; value={best[metric]}; run_scale_terms={json.dumps(best.get('run_scale_terms', []), sort_keys=True)}; " + (f"runner_up={runner['run']}; runner_up_value={runner[metric]}; gap={gap}; " if runner else "") + f"best_axes={json.dumps({a: best.get(f'config.{a}') for a in axes}, sort_keys=True)}; constant_axes={json.dumps(constants, sort_keys=True)}")
        print("interpretation_hint=Use best_by_metric plus axis_metric_summary and axis_pair_metric_summary to compare config choices. Report runner-up gap and constant axes when relevant; avoid extra SDK scans unless the requested axis or metric is absent.")


# -----------------------------------------------------------------------------
# Stability and gradient diagnostics
# -----------------------------------------------------------------------------


def cmd_stability_scan(args: argparse.Namespace) -> None:
    """Scan bounded run histories for loss or gradient spikes."""
    wb, path = ctx(args)
    requested_limit = args.limit
    sample_cap = stability_run_sample_cap(args.samples)
    sample_limit = requested_limit if args.large_scan else capped_hydrated_limit(requested_limit, sample_cap)
    sample = filter_names(runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=sample_limit), args.name_include, args.name_exclude)
    explicit_metrics = [args.metric] if isinstance(args.metric, str) else args.metric
    keys = explicit_metrics or select_stability_metric_keys(sample, args.metric_limit) or choose_history_keys(sample, args.metric_limit)
    print(f"path={path}\nrun_query_count={len(sample)}\nselected_runs={len(sample)}")
    print_effective_scan_limit("stability-scan", requested_limit, sample_limit, unit="runs_with_history_requests", large_scan=args.large_scan)
    print_sample_scope(
        "stability-scan",
        requested=requested_limit,
        effective=sample_limit,
        actual=len(sample),
        unit="runs_with_history_requests",
        strategy=f"lazy SDK runs ordered by {args.order}; each selected run reads up to {args.samples} history rows",
        note="stability rows and spike counts are computed only for selected_runs; scan_history fallback is disabled unless --scan-history-fallback is passed",
    )
    print_cap_caveat("stability-scan", requested_limit, sample_limit)
    print_limit_caveat("stability-scan", requested_limit, sample_limit)
    print(f"history_budget_seconds={args.history_budget_seconds}")
    print("scan_history_fallback=" + ("enabled" if args.scan_history_fallback else "disabled"))
    print("selected_run_names=" + ", ".join(run_label(r) for r in sample[: args.top]) + "\nstability_metric_keys=" + ", ".join(keys))
    if not sample:
        print("answer=no runs matched the requested stability scan filters"); return
    if not keys:
        print("answer=no numeric training-loss or gradient-norm keys found in sampled runs"); return
    rows_ = collect_stability_rows(sample, keys, args.samples, args.workers, args.history_budget_seconds, args.scan_history_fallback)
    rows_.sort(key=lambda r: r.get("run", ""))
    for row in rows_:
        emit("stability_row", row)
    presence = {key: sum(1 for row in rows_ if key in row.get("metrics", {})) for key in keys}
    spikes = [(row["run"], key, stats.get("max_to_median"), stats.get("max")) for row in rows_ for key, stats in row.get("metrics", {}).items() if (stats.get("max_to_median") or 0) >= args.spike_ratio and (stats.get("max_index") or 0) >= 5]
    spikes.sort(key=lambda s: (s[2] or 0), reverse=True)
    emit("metric_presence", presence)
    print("possible_spikes=" + ("; ".join(f"{r}:{k} max_to_median={ratio} max={mx}" for r, k, ratio, mx in spikes[: args.top]) if spikes else f"none above max_to_median {args.spike_ratio}"))
    if spikes:
        top_ratio, top_abs = max(spikes, key=lambda s: (s[2] or 0)), max(spikes, key=lambda s: (s[3] or 0))
        emit("spike_extrema", {"largest_spike_ratio": {"run": top_ratio[0], "metric": top_ratio[1], "max_to_median": top_ratio[2], "max": top_ratio[3]}, "largest_max_value": {"run": top_abs[0], "metric": top_abs[1], "max": top_abs[3], "max_to_median": top_abs[2]}, "runs_flagged": len({s[0] for s in spikes})})
        print("spike_caveat=report the largest spike-ratio run and the largest absolute-max run separately (they are often different runs), state how many runs spiked, and remember a spiky run can still converge — do not attribute instability to one optimizer/cohort without checking the spread across runs.")
    print(f"answer=checked {len(rows_)} runs with metric keys {', '.join(keys)}; spike_flags={len(spikes)}; use stability_row and spike_extrema evidence for primary/component losses and gradient norms")
    print("interpretation_hint=State whether training looks clean from sampled histories; cite stability_metric_keys checked, including component losses and gradient norms when present; mention possible_spikes only when supported by a stability_row.")


def choose_history_keys(sample: list[Any], limit: int) -> list[str]:
    keys = []
    for run in sample:
        for key, value in (run.summary_metrics or {}).items():
            lower = str(key).lower()
            if coerce_float(value) is not None and ("loss" in lower or "grad" in lower or "kl" in lower) and key not in keys:
                keys.append(str(key))
    return keys[:limit]


def series_values_with_skips(df: Any, key: str) -> tuple[list[float], int]:
    """Return numeric history values plus dropped nonnumeric/NaN point count."""
    values, skipped = [], 0
    for value in df.get(key, []):
        number = coerce_float(value)
        if number is None:
            if value not in (None, ""):
                skipped += 1
            continue
        values.append(number)
    return values, skipped


def series_values(df: Any, key: str) -> list[float]:
    return series_values_with_skips(df, key)[0]


def series_stats(values: list[float], skipped: int = 0) -> dict[str, Any]:
    if not values:
        row = {"points": 0}
        if skipped:
            row["non_numeric_or_nan_points"] = skipped
        return row
    head, tail, median = values[: max(1, min(5, len(values)))], values[-max(1, min(5, len(values))):], statistics.median(values)
    row = {"points": len(values), "first": round(values[0], 6), "final": round(values[-1], 6), "head_mean": round(statistics.mean(head), 6), "tail_mean": round(statistics.mean(tail), 6), "min": round(min(values), 6), "max": round(max(values), 6), "max_index": values.index(max(values)), "final_to_first": round(values[-1] / values[0], 3) if abs(values[0]) > 1e-12 else None, "max_to_median": round(max(values) / median, 3) if abs(median) > 1e-12 else None}
    if skipped:
        row["non_numeric_or_nan_points"] = skipped
    return row


def stability_timeout_row(run: Any, budget_seconds: float) -> dict[str, Any]:
    return {
        "run": run_label(run),
        "id": run.id,
        "state": getattr(run, "state", None),
        "history_timeout": True,
        "history_timeout_seconds": budget_seconds,
        "history_error": f"history probe did not finish within the {budget_seconds:g}s command budget",
        "metrics": {},
    }


def collect_stability_rows(sample: list[Any], keys: list[str], samples: int, workers: int, budget_seconds: float, allow_scan_history: bool) -> list[dict[str, Any]]:
    """Collect per-run stability rows within a command-level budget."""
    tasks: queue.Queue[tuple[int, Any]] = queue.Queue()
    results: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
    for index, run in enumerate(sample):
        tasks.put((index, run))

    def worker() -> None:
        while True:
            try:
                index, run = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                row = stability_run_stats(run, keys, samples, allow_scan_history=allow_scan_history)
            except Exception as exc:
                row = {
                    "run": run_label(run),
                    "id": run.id,
                    "state": getattr(run, "state", None),
                    "history_error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                }
            results.put((index, row))

    for _ in range(min(max(1, workers), len(sample))):
        threading.Thread(target=worker, daemon=True).start()

    rows, completed = [], set()
    deadline = time.monotonic() + budget_seconds
    while len(completed) < len(sample):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            index, row = results.get(timeout=min(0.2, remaining))
        except queue.Empty:
            continue
        if index in completed:
            continue
        completed.add(index)
        rows.append(row)

    rows.extend(stability_timeout_row(run, budget_seconds) for index, run in enumerate(sample) if index not in completed)
    return rows


def stability_run_stats(run: Any, keys: list[str], samples: int, *, allow_scan_history: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"run": run_label(run), "id": run.id, "state": getattr(run, "state", None)}
    try:
        df = run.history(samples=samples, keys=keys)
    except Exception as exc:
        row["history_error"] = f"{type(exc).__name__}: {exc}"
        row["metrics"] = {}
        return row
    metrics = {}
    for key in keys:
        if key not in df:
            continue
        values, skipped = series_values_with_skips(df, key)
        if values or skipped:
            metrics[key] = series_stats(values, skipped)
    if not metrics:
        if not allow_scan_history:
            row["history_fallback"] = "skipped"
            row["history_fallback_reason"] = "sampled run.history returned no numeric or nonnumeric points for requested keys; pass --scan-history-fallback for the slower per-key scan_history fallback"
            row["metrics"] = {}
            return row
        scanned: dict[str, list[float]] = defaultdict(list); skipped_points: Counter[str] = Counter()
        row["history_fallback"] = "scan_history"
        for key in keys:
            try:
                for index, item in enumerate(run.scan_history(keys=[key], page_size=500)):
                    if index >= samples: break
                    raw = item.get(key); value = coerce_float(raw)
                    if value is not None: scanned[key].append(value)
                    elif raw not in (None, ""): skipped_points[key] += 1
            except Exception as exc:
                row.setdefault("scan_history_errors", {})[key] = f"{type(exc).__name__}: {exc}"
        metrics = {key: series_stats(values, skipped_points[key]) for key, values in scanned.items() if values or skipped_points[key]}
    row["metrics"] = metrics
    return row


def gradient_noise_metrics(run: Any) -> dict[str, float]:
    """Return numeric gradient-noise metrics from a run summary."""
    return {str(k): n for k, v in (run.summary_metrics or {}).items() if ("grad_noise" in str(k).lower() or "gradient_noise" in str(k).lower()) and (n := coerce_float(v)) is not None}


def gradient_noise_row(run: Any) -> dict[str, Any]:
    """Build one comparable gradient-noise row."""
    config = clean_config(run.config)
    return {"run": run_label(run), "id": run.id, "optimizer": config.get("optimizer") or config.get("optim"), "model_size": config.get("paper_model_id") or config.get("model_size") or config.get("parameter_count") or config.get("model"), "materialization": config.get("model_materialization") or config.get("materialization"), "dataset": config.get("dataset") or config.get("dataset_name"), "metrics": gradient_noise_metrics(run)}


def cmd_gradient_noise_summary(args: argparse.Namespace) -> None:
    """Group gradient-noise metrics by comparable optimizer regimes."""
    wb, path = ctx(args)
    sample_limit = capped_hydrated_limit(args.limit)
    sampled_runs = runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=sample_limit, lazy=False)
    rows_ = [
        gradient_noise_row(run)
        for run in sampled_runs
    ]
    rows_ = [row for row in rows_ if row["metrics"]]
    if args.materialization:
        wanted = {m.lower() for m in args.materialization}
        rows_ = [row for row in rows_ if str(row.get("materialization") or "").lower() in wanted]
    by_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        key = (str(row.get("materialization") or "<none>"), str(row.get("model_size") or "<unknown>"), str(row.get("dataset") or "<unknown>"), str(row.get("optimizer") or "<unknown>").lower())
        by_pair[key].append(row)
    summaries = []
    for (mat, size, dataset, opt), items in by_pair.items():
        metrics: dict[str, list[float]] = defaultdict(list)
        for row in items:
            for key, value in row["metrics"].items():
                metrics[key].append(value)
        summaries.append({"materialization": mat, "model_size": size, "dataset": dataset, "optimizer": opt, "runs": [r["run"] for r in items[:4]], "run_ids": [r["id"] for r in items[:4]], "count": len(items), "metrics": {key: round(statistics.median(vals), 6) for key, vals in metrics.items()}})
    summaries.sort(key=lambda row: (row["materialization"], row["model_size"], row["dataset"], row["optimizer"]))
    print(f"path={path}\nmethod=wandb.Api regular W&B Python SDK\ngradient_noise_runs={len(rows_)}")
    print_sample_scope(
        "gradient-noise-summary",
        requested=args.limit,
        effective=sample_limit,
        actual=len(sampled_runs),
        strategy=f"hydrated SDK runs ordered by {args.order}",
        note="gradient_noise_runs counts only sampled runs that contained gradient-noise metrics",
    )
    print_cap_caveat("gradient-noise-summary", args.limit, sample_limit)
    for row in summaries:
        emit("gradient_noise_pair_summary", row)
    contrast_count = 0
    by_regime: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_regime[(row["materialization"], row["model_size"], row["dataset"])][row["optimizer"]] = row
    requested_opts = {name.lower() for name in args.optimizer}
    for (mat, size, dataset), opts in sorted(by_regime.items()):
        names = sorted(name for name in opts if not requested_opts or name in requested_opts)
        for left_name, right_name in combinations(names, 2):
            left, right = opts[left_name], opts[right_name]
            common = [key for key in args.contrast_metric if key in left["metrics"] and key in right["metrics"]] or sorted(set(left["metrics"]) & set(right["metrics"]))[:5]
            if not common:
                continue
            metrics = {key: {"left": left["metrics"].get(key), "right": right["metrics"].get(key), "left_over_right": round(left["metrics"][key] / right["metrics"][key], 6) if right["metrics"].get(key) not in (None, 0) else None} for key in common}
            emit("gradient_noise_contrast", {"materialization": mat, "model_size": size, "dataset": dataset, "left_optimizer": left_name, "right_optimizer": right_name, "left_run_ids": left["run_ids"], "right_run_ids": right["run_ids"], "metrics": metrics})
            contrast_count += 1
    print(f"gradient_noise_contrast_count={contrast_count}")
    for row in rows_[: args.row_limit]:
        emit("gradient_noise_row_sample", row)
    print("answer=compare noise_to_signal separately from signal_power/noise_power; use gradient_noise_pair_summary grouped by materialization, model_size, dataset, and optimizer")
    print("interpretation_hint=For each comparable regime, report requested gradient-noise metrics for each optimizer separately. If ratios are similar but absolute signal/noise powers differ, say so and cite run ids; do not call one optimizer cleaner unless same-regime evidence supports it.")


# -----------------------------------------------------------------------------
# History, layer, and GPU diagnostics
# -----------------------------------------------------------------------------


def is_layer_metric_key(key: str) -> bool:
    lower = key.lower()
    return any(t in lower for t in LAYER_METRIC_TOKENS) and any(t in lower for t in LAYER_STAT_TOKENS)


def layer_metric_index(key: str) -> int | None:
    match = re.search(r"(?:layer|layers|block|blocks)[/_\-.]?(\d+)", key.lower())
    return int(match.group(1)) if match else None


def layer_metric_family(key: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"(layer|layers|block|blocks)[/_\-.]?\d+", r"\1/#", key.lower()))


def select_layer_trend_keys(summary: dict[str, Any], limit: int = 8) -> list[str]:
    candidates = [str(k) for k, v in (summary or {}).items() if is_layer_metric_key(str(k)) and coerce_float(v) is not None]
    by_family: dict[str, list[str]] = defaultdict(list)
    for key in candidates: by_family[layer_metric_family(key)].append(key)
    selected: list[str] = []
    for keys in sorted(by_family.values(), key=lambda ks: (len(ks), ks[0])):
        ordered = sorted(keys, key=lambda k: (layer_metric_index(k) is None, layer_metric_index(k) or 10**9, k))
        for key in (ordered[0], ordered[-1]):
            if key not in selected and len(selected) < limit: selected.append(key)
    return selected


def compact_history_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = dict(row); compact.pop("layer_metrics", None); return compact


def layer_metric_zscores(rows_: list[dict[str, Any]], target: dict[str, Any]) -> list[tuple[str, float, float]]:
    zscores = []
    for key in sorted({k for row in rows_ for k in row.get("layer_metrics", {}) if k in target.get("layer_metrics", {})}):
        values = [row["layer_metrics"][key] for row in rows_ if key in row.get("layer_metrics", {})]
        if len(values) >= 4:
            zscores.append((key, target["layer_metrics"][key], (target["layer_metrics"][key] - statistics.mean(values)) / (statistics.pstdev(values) or 1.0)))
    return sorted(zscores, key=lambda item: abs(item[2]), reverse=True)


def boundary_hparam_name_score(row: dict[str, Any]) -> float:
    """Score obvious boundary-condition hparams in run names/configs."""
    label, score = str(row.get("run", "")).lower(), 0.0
    if any(t in label for t in ("init_lr", "initial_lr", "max_lr", "min_lr")):
        score += 3.0
    if re.search(r"(100pct|100%|1p0|full|max|min|zero)", label):
        score += 2.0
    for key in ["init_lr_frac", "learning_rate", "lr"]:
        value = coerce_float(row.get(key))
        if value is not None and (value <= 0 or value >= 1):
            score += 1.0
    return score


def add_layer_trend_scores(rows_: list[dict[str, Any]]) -> None:
    """Add normalized layer/boundary outlier scores to history rows."""
    trend_keys = sorted({key for row in rows_ for key in row.get("layer_trends", {})})
    stats = {}
    for key in trend_keys:
        values = [row["layer_trends"][key] for row in rows_ if key in row.get("layer_trends", {})]
        if len(values) >= 4:
            stats[key] = (statistics.mean(values), statistics.pstdev(values) or 1.0)
    for row in rows_:
        zscores = [abs((value - stats[key][0]) / stats[key][1]) for key, value in row.get("layer_trends", {}).items() if key in stats]
        row["layer_trend_score"] = round(sum(sorted(zscores, reverse=True)[:5]), 3)
        row["boundary_hparam_score"] = boundary_hparam_name_score(row)
        row["combined_outlier_score"] = round(max(abs(row.get("tail_mean_z", 0.0)), row["layer_trend_score"]) + 2.0 * row["boundary_hparam_score"], 3)


def history_row(run: Any, metric: str, samples: int, include_layer: bool = False) -> dict[str, Any]:
    try:
        vals, skipped = series_values_with_skips(run.history(samples=samples, keys=[metric]), metric)
    except Exception as exc:
        return {"run": run_label(run), "id": run.id, "error": str(exc)}
    if not vals:
        row = {"run": run_label(run), "id": run.id, "error": f"no numeric {metric} history"}
        if skipped:
            row["non_numeric_or_nan_points"] = skipped
        return row
    tail = vals[-min(10, len(vals)):]
    config = clean_config(run.config)
    row: dict[str, Any] = {"run": run_label(run), "id": run.id, "points": len(vals), "final": vals[-1], "tail_mean": statistics.mean(tail), "min": min(vals), "max": max(vals)}
    if skipped:
        row["non_numeric_or_nan_points"] = skipped
    for key in ["init_lr_frac", "warmup_steps", "learning_rate", "lr"]:
        if key in config: row[key] = config.get(key)
    if include_layer:
        row["layer_metrics"] = {str(k): n for k, v in (run.summary_metrics or {}).items() if is_layer_metric_key(str(k)) and (n := coerce_float(v)) is not None}
        trend_keys = select_layer_trend_keys(run.summary_metrics or {})
        row["layer_trend_keys"] = trend_keys
        try:
            layer_df = run.history(samples=samples, keys=trend_keys) if trend_keys else {}
            row["layer_trends"] = {f"{key}_slope": values[-1] - values[0] for key in trend_keys if (values := series_values(layer_df, key)) and len(values) >= 2}
        except Exception as exc:
            row["layer_trend_error"] = str(exc)
    return row


def longest_streak_minutes(df: Any, mask: list[bool]) -> float | None:
    """Return the longest true-mask streak in minutes using runtime/timestamp columns."""
    key = next((k for k in ["_runtime", "_timestamp", "timestamp"] if k in df), None)
    if key is None or len(mask) < 2:
        return None
    times, best, start, previous = [coerce_float(v) for v in df[key].tolist()], 0.0, None, None
    for is_low, current in zip(mask, times, strict=False):
        if current is None:
            continue
        if is_low:
            start = current if start is None else start; previous = current
        elif start is not None and previous is not None:
            best = max(best, previous - start); start = previous = None
    if start is not None and previous is not None:
        best = max(best, previous - start)
    return round(best / 60.0, 1) if best > 0 else 0.0


def gpu_row(run: Any, samples: int) -> dict[str, Any]:
    """Summarize system-stream GPU utilization for one run."""
    df = run.history(stream="system", samples=samples)
    key = next((k for k in ["system.gpu.0.gpu", "system/gpu.0.gpu"] if k in df), None)
    if not key:
        return {"run": run_label(run), "error": "no gpu utilization key in system stream"}
    series = df[key].dropna()
    vals = [float(v) for v in series.tolist()]
    if not vals:
        return {"run": run_label(run), "error": "empty gpu utilization series"}
    aligned = df.loc[series.index]
    under_10, under_30 = [v < 10 for v in vals], [v < 30 for v in vals]
    return {"run": run_label(run), "points": len(vals), "gpu_key": key, "gpu_mean": round(statistics.mean(vals), 1), "gpu_min": round(min(vals), 1), "gpu_max": round(max(vals), 1), "zero_pct": round(100 * sum(v == 0 for v in vals) / len(vals), 1), "under_30_pct": round(100 * sum(under_30) / len(vals), 1), "longest_under_10_min": longest_streak_minutes(aligned, under_10), "longest_under_30_min": longest_streak_minutes(aligned, under_30)}


def cmd_diagnose_history(args: argparse.Namespace) -> None:
    """Diagnose history outliers or GPU utilization."""
    wb, path = ctx(args)
    sample = filter_names(runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=args.limit), args.name_include, args.name_exclude)
    print(f"path={path}")
    if args.mode == "gpu":
        if args.training_like:
            sample = [run for run in sample if not any(term in run_label(run).lower() for term in GPU_TRAINING_EXCLUDE_TERMS)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows_ = sorted([row for row in [f.result() for f in as_completed([pool.submit(gpu_row, run, args.samples) for run in sample])] if "gpu_mean" in row], key=lambda r: r["run"])
        verdict_rows = [row for row in rows_ if row["points"] >= args.min_points_for_verdict]
        print(f"run_query_count={len(sample)}\ntraining_runs_considered={len(rows_)}\nverdict_runs_considered={len(verdict_rows)}\ntraining_like_filter={args.training_like}\nsource=run.history(stream='system', samples={args.samples})")
        for row in rows_:
            print(json.dumps(row, sort_keys=True))
        if verdict_rows:
            means = sorted(row["gpu_mean"] for row in verdict_rows)
            median = round(statistics.median(means), 1)
            verdict = "well_saturated_gpu_utilization" if median >= args.good_threshold else "low_to_moderate_gpu_utilization"
            max_under_30 = max((r.get("longest_under_30_min") or 0.0 for r in verdict_rows), default=0.0)
            max_under_10 = max((r.get("longest_under_10_min") or 0.0 for r in verdict_rows), default=0.0)
            print(f"answer=verdict={verdict}; overall_mean_gpu={round(statistics.mean(means), 1)}; median_run_gpu_mean={median}; typical_run_gpu_mean_range={means[0]}-{means[-1]}; mean_under_30_pct={round(statistics.mean(r['under_30_pct'] for r in verdict_rows), 1)}; mean_zero_pct={round(statistics.mean(r['zero_pct'] for r in verdict_rows), 1)}; max_under_30_streak_min={max_under_30}; max_under_10_streak_min={max_under_10}; runs={', '.join(r['run'] for r in verdict_rows)}")
        return
    metric = first(args.metric) or choose_metric_from_runs(sample, first(args.metric_hint, "loss"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows_ = [row for row in [f.result() for f in as_completed([pool.submit(history_row, run, metric, args.samples, args.layer_metrics) for run in sample])] if "tail_mean" in row]
    if not rows_:
        print(f"metric={metric}\nhistory_runs=0\nanswer=no history rows found for {metric}"); return
    mean, stdev = statistics.mean(r["tail_mean"] for r in rows_), statistics.pstdev(r["tail_mean"] for r in rows_) or 1.0
    for row in rows_:
        row["tail_mean_z"] = (row["tail_mean"] - mean) / stdev
    ranked = sorted(rows_, key=lambda r: abs(r["tail_mean_z"]), reverse=True)
    add_layer_trend_scores(rows_)
    layer_ranked = sorted(rows_, key=lambda r: r.get("combined_outlier_score", 0.0), reverse=True)
    target = next((r for r in rows_ if r["run"] == args.target_run or r["id"] == args.target_run), None) if args.target_run else (layer_ranked[0] if args.layer_metrics and layer_ranked else ranked[0])
    if target: target["tail_mean_abs_z_rank"] = ranked.index(target) + 1
    layer_z = layer_metric_zscores(rows_, target) if args.layer_metrics and target else []
    print(f"metric={metric}")
    print("metric_source=" + ("explicit" if first(args.metric) else f"metric_hint:{first(args.metric_hint, 'loss')}"))
    print(f"history_runs={len(rows_)}")
    print(f"method=tail_mean_zscore over run.history(samples={args.samples}, keys={[metric]})")
    if args.target_run:
        print("primary_outlier_source=target_run_argument")
        print("interpretation_note=When target-run is provided, treat that run as the headline outlier and use z-score/layer evidence to explain it; other high-z runs are secondary comparisons.")
    if target:
        print(f"headline_outlier={target['run']}")
        emit("target", compact_history_row(target))
        print(f"target_evidence=tail_mean={target['tail_mean']}; tail_mean_z={target['tail_mean_z']}; abs_z_rank={target['tail_mean_abs_z_rank']}/{len(rows_)}")
        if layer_z:
            largest = max(layer_z, key=lambda item: abs(item[2]))
            print(f"target_layer_summary={len(layer_z)} layer deviations shown; largest={largest[0]} z={largest[2]}")
            for key, value, z in layer_z[: args.top]:
                print(f"target_layer_metric_z_score={key}\tvalue={value}\tz={z}")
        boundary = f"; boundary_hparam_score={target.get('boundary_hparam_score')}" if target.get("boundary_hparam_score") else ""
        combined = f"; combined_outlier_score={target.get('combined_outlier_score')}" if target.get("combined_outlier_score") is not None else ""
        print(f"answer=headline_outlier={target['run']}; {metric}_tail_mean={target['tail_mean']}; tail_mean_z={target['tail_mean_z']}; abs_z_rank={target['tail_mean_abs_z_rank']}/{len(rows_)}{boundary}{combined}" + (f"; largest_layer_deviation={max(layer_z, key=lambda item: abs(item[2]))[0]}" if layer_z else ""))
        print(f"interpretation_hint={target['run']} is the headline outlier. Cite {metric} tail_mean/z/rank plus layer trend and boundary-hyperparameter evidence when present; mention secondary high-loss runs only as comparisons.")
        print("stop_rule=headline_outlier is the final-answer headline; do not override it with a tail-loss-only secondary run unless the user explicitly asks for loss-only ranking")
    if args.target_run:
        print("secondary_loss_z_comparisons=" + "; ".join(f"{row['run']}:tail_mean_z={row['tail_mean_z']}" for row in ranked[: min(args.top, 4)] if row is not target) + " (do not make these the headline when target-run is provided)")
        print("analysis_complete=true")
    else:
        for row in ranked[: args.top]:
            print(json.dumps(compact_history_row(row), sort_keys=True, default=str))


# -----------------------------------------------------------------------------
# Artifacts and logged files
# -----------------------------------------------------------------------------


def artifact_collection_rows(wb: Any, path: str, type_filter: list[str] | None = None, per_page: int = 1000) -> list[dict[str, Any]]:
    wanted = {type_filter} if isinstance(type_filter, str) else set(type_filter or [])
    rows_ = []
    for typ in wb.artifact_types(project=path):
        name = str(getattr(typ, "name", ""))
        if wanted and name not in wanted:
            continue
        try:
            collections = list(wb.artifact_collections(path, name, per_page=per_page))
        except Exception as exc:
            rows_.append({"type": name, "collections": [], "collection_count": 0, "samples": [], "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows_.append({"type": name, "collections": collections, "collection_count": len(collections), "samples": [c.name for c in collections[:5]]})
    return rows_


def artifact_version_count(collection: Any) -> tuple[int | None, str | None]:
    try:
        return len(collection.artifacts(per_page=50)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def cmd_artifact_inventory(args: argparse.Namespace) -> None:
    """Print artifact type, collection, and optional version counts."""
    wb, path = ctx(args)
    rows_ = artifact_collection_rows(wb, path, args.type, args.per_page)
    for row in rows_:
        if row.get("error"):
            emit("artifact_collection_error", {"type": row["type"], "error": row["error"]})
    collections = [c for r in rows_ for c in r["collections"]]
    count_versions = args.version_mode == "all" or (args.version_mode == "auto" and len(collections) <= args.max_version_collections)
    versions: dict[str, tuple[int | None, str | None]] = {}
    if count_versions and collections:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(artifact_version_count, c): c.id for c in collections}
            versions = {futures[future]: future.result() for future in as_completed(futures)}
    failed_versions = {cid: error for cid, (_, error) in versions.items() if error}
    output = [{"type": r["type"], "collection_count": r["collection_count"], "version_count": sum((versions.get(c.id) or (0, None))[0] or 0 for c in r["collections"]) if count_versions else None, "sample_collections": r["samples"], **({"error": r["error"]} if r.get("error") else {})} for r in rows_]
    print(f"path={path}\nartifact_type_count={len(output)}\nartifact_collection_count={sum(r['collection_count'] for r in output)}")
    print(f"artifact_version_count_mode={args.version_mode}")
    if count_versions:
        print("artifact_version_count_scope=bounded_first_50_versions_per_collection")
        print(f"artifact_version_count={sum(r['version_count'] or 0 for r in output)}")
        if failed_versions:
            print(f"artifact_version_count_failures={len(failed_versions)}")
            emit("artifact_version_count_errors", dict(list(failed_versions.items())[:10]))
    else:
        print(f"artifact_version_count_scope=not_counted; collections={len(collections)}; auto_threshold={args.max_version_collections}")
    for row in output:
        emit("artifact_type_row", row)
    print(f"answer=artifact_types={len(output)}; total_collections={sum(r['collection_count'] for r in output)}")


def short_file_row(file_path: Path, root: Path) -> dict[str, Any]:
    """Return compact file metadata relative to an artifact root."""
    rel = file_path.relative_to(root)
    row = {"path": str(rel), "suffix": file_path.suffix.lower(), "size_bytes": file_path.stat().st_size}
    if row["suffix"] in MEDIA_SUFFIXES:
        try:
            from PIL import Image
            with Image.open(file_path) as image:
                row.update({"width": image.width, "height": image.height})
        except Exception as exc:
            row["media_metadata_error"] = f"{type(exc).__name__}: {exc}"
    return row


def is_structured_field_count_candidate(rel: Path) -> bool:
    """Return whether a data file name suggests useful low-cardinality counts."""
    lower = str(rel).lower()
    return rel.suffix.lower() in DATA_SUFFIXES and any(token in lower for token in STRUCTURED_FIELD_COUNT_FILENAME_TOKENS)


def structured_count_key_tokens(key: str) -> set[str]:
    """Split field names into word tokens without treating substrings as tokens."""
    separated = STRUCTURED_KEY_CAMEL_RE.sub("_", key)
    return {token for token in re.split(r"[^A-Za-z0-9]+", separated.lower()) if token}


def structured_count_key(key: str) -> bool:
    """Return whether a field is likely useful for low-cardinality breakdowns."""
    lower = key.lower()
    if structured_count_key_tokens(key) & STRUCTURED_COUNT_EXCLUDE_TOKENS:
        return False
    if lower.endswith("_at") or lower.endswith(".at") or lower in {"at"}:
        return False
    return bool(lower.strip()) and not lower.startswith("_")


def structured_count_value(value: Any) -> str | None:
    """Normalize scalar values for compact field-count summaries."""
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and len(value) <= 160:
        text = value.strip()
        return None if STRUCTURED_DATETIME_VALUE_RE.match(text) else value
    return None


def update_field_counts(obj: dict[str, Any], counters: dict[str, Counter[str]], skipped: dict[str, str], omitted_fields: set[str]) -> None:
    """Update bounded low-cardinality counters from one structured row."""
    for key, value in obj.items():
        key = str(key)
        if key in skipped or not structured_count_key(key):
            continue
        scalar = structured_count_value(value)
        if scalar is None:
            continue
        if key not in counters and len(counters) >= STRUCTURED_COUNT_MAX_FIELDS:
            omitted_fields.add(key)
            continue
        counter = counters.setdefault(key, Counter())
        counter[scalar] += 1
        if len(counter) > STRUCTURED_COUNT_MAX_TRACKED_VALUES:
            skipped[key] = f"more than {STRUCTURED_COUNT_MAX_TRACKED_VALUES} distinct scalar values"
            counters.pop(key, None)


def field_count_rows(path: str, rows: int, counters: dict[str, Counter[str]], *, complete: bool) -> list[dict[str, Any]]:
    """Return stable field-count rows sorted by coverage and field name."""
    ranked = sorted(counters.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    output = []
    for key, counter in ranked:
        if not counter or (rows and sum(counter.values()) / rows < STRUCTURED_COUNT_MIN_COVERAGE):
            continue
        counts = dict(counter.most_common(STRUCTURED_COUNT_MAX_VALUES))
        row = {
            "path": path,
            "field": key,
            "rows": rows,
            "complete": complete,
            "counts": counts,
        }
        omitted_value_count = max(0, len(counter) - len(counts))
        if omitted_value_count:
            omitted_observation_count = sum(counter.values()) - sum(counts.values())
            row.update(
                {
                    "truncated": True,
                    "omitted_value_count": omitted_value_count,
                    "omitted_observation_count": omitted_observation_count,
                    "note": f"showing top {STRUCTURED_COUNT_MAX_VALUES} values; {omitted_value_count} more distinct values omitted",
                }
            )
        output.append(row)
    return output


def field_count_skip_rows(path: str, skipped: dict[str, str], omitted_fields: set[str]) -> list[dict[str, Any]]:
    """Return compact notes for structured fields not summarized because of caps."""
    rows = [{"path": path, "field": key, "reason": reason} for key, reason in sorted(skipped.items())]
    if omitted_fields:
        sample = sorted(omitted_fields)[:20]
        rows.append(
            {
                "path": path,
                "reason": f"field cap reached; summarizing first {STRUCTURED_COUNT_MAX_FIELDS} eligible fields",
                "omitted_field_count": len(omitted_fields),
                "omitted_fields_sample": sample,
            }
        )
    return rows


def summarize_jsonl(path: Path, sample_rows: int = 2, *, field_counts: bool = False) -> dict[str, Any]:
    """Count JSONL rows and capture compact sample keys/rows."""
    rows, invalid_json_lines, sample, keys, counters, skipped, omitted_fields = 0, 0, [], set(), {}, {}, set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle
        for line in lines:
            if not line.strip(): continue
            rows += 1
            try: obj = json.loads(line)
            except Exception:
                invalid_json_lines += 1
                continue
            if isinstance(obj, dict):
                keys.update(map(str, obj))
                if field_counts:
                    update_field_counts(obj, counters, skipped, omitted_fields)
                if len(sample) < sample_rows:
                    wanted = ["agent_name", "model", "model_name", "task_name", "success", "cum_reward", "n_steps", "run_id"]
                    sample.append({key: obj.get(key) for key in wanted if key in obj})
    result = {"rows": rows, "sample_keys": sorted(keys), "sample_rows": sample}
    if invalid_json_lines:
        result["invalid_json_lines"] = invalid_json_lines
    if field_counts:
        result["field_counts"] = field_count_rows(str(path.name), rows, counters, complete=True)
        result["field_count_skips"] = field_count_skip_rows(str(path.name), skipped, omitted_fields)
    return result


def summarize_csv(path: Path, sample_rows: int = 2, *, field_counts: bool = False) -> dict[str, Any]:
    """Count CSV rows and capture compact sample keys/rows."""
    rows, sample, fieldnames, counters, skipped, omitted_fields = 0, [], [], {}, {}, set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ","); fieldnames = reader.fieldnames or []
        for row in reader:
            rows += 1
            if field_counts:
                update_field_counts(row, counters, skipped, omitted_fields)
            if len(sample) < sample_rows:
                sample.append({key: row.get(key) for key in fieldnames[:8]})
    result = {"rows": rows, "sample_keys": fieldnames, "sample_rows": sample}
    if field_counts:
        result["field_counts"] = field_count_rows(str(path.name), rows, counters, complete=True)
        result["field_count_skips"] = field_count_skip_rows(str(path.name), skipped, omitted_fields)
    return result


def summarize_json(path: Path, *, field_counts: bool = False) -> dict[str, Any]:
    """Summarize small JSON files without expanding large nested payloads."""
    result: dict[str, Any] = {"rows": None, "sample_keys": [], "sample_rows": []}
    if not field_counts:
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        result["structured_stats_error"] = f"{type(exc).__name__}: {exc}"
        return result
    records = data if isinstance(data, list) else next((value for value in data.values() if isinstance(value, list)), []) if isinstance(data, dict) else []
    counters: dict[str, Counter[str]] = {}
    skipped: dict[str, str] = {}
    omitted_fields: set[str] = set()
    rows = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        rows += 1
        update_field_counts(item, counters, skipped, omitted_fields)
    result["rows"] = rows if rows else None
    if isinstance(data, dict):
        result["sample_keys"] = sorted(map(str, data))[:40]
    if rows:
        result["field_counts"] = field_count_rows(str(path.name), rows, counters, complete=True)
        result["field_count_skips"] = field_count_skip_rows(str(path.name), skipped, omitted_fields)
    return result


def download_artifact_entry(artifact: Any, rel_path: str, root: Path) -> Path:
    """Download one artifact entry and return its local path."""
    if hasattr(artifact, "get_entry"):
        return Path(artifact.get_entry(rel_path).download(root=str(root)))
    return Path(artifact.get_path(rel_path).download(root=str(root)))


def compact_artifact_value(value: Any, *, depth: int = 0) -> Any:
    """Return artifact metadata/context values in a bounded JSON-friendly shape."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) or len(value) <= 300 else value[:300] + "..."
    if isinstance(value, list):
        items = [compact_artifact_value(item, depth=depth + 1) for item in value[:10]]
        return items + ([f"... {len(value) - 10} more"] if len(value) > 10 else [])
    if isinstance(value, dict):
        if depth >= 2:
            return f"<{len(value)} keys>"
        items = list(value.items())
        compact = {str(key): compact_artifact_value(child, depth=depth + 1) for key, child in items[:30]}
        if len(items) > 30:
            compact["..."] = f"{len(items) - 30} more keys"
        return compact
    return str(value)[:300]


def artifact_context_row(artifact: Any) -> dict[str, Any]:
    """Return compact artifact-level provenance without downloading files."""
    row = {
        "description": getattr(artifact, "description", None),
        "created_at": getattr(artifact, "created_at", None),
        "updated_at": getattr(artifact, "updated_at", None),
        "source_name": getattr(artifact, "source_name", None),
        "source_version": getattr(artifact, "source_version", None),
        "digest": getattr(artifact, "digest", None),
        "commit_hash": getattr(artifact, "commit_hash", None),
    }
    try:
        logged_by = artifact.logged_by()
    except Exception as exc:
        row["logged_by_error"] = f"{type(exc).__name__}: {exc}"
    else:
        if logged_by is not None:
            row["logged_by_run"] = {
                "path": getattr(logged_by, "path", None),
                "id": getattr(logged_by, "id", None),
                "name": run_label(logged_by),
                "state": getattr(logged_by, "state", None),
            }
    return {key: compact_artifact_value(value) for key, value in row.items() if value not in (None, {}, [])}


def artifact_entries_to_inspect(entries: list[Any], args: argparse.Namespace) -> list[tuple[Path, int]]:
    """Return bounded artifact paths this command can inspect."""
    selected: list[tuple[Path, int]] = []
    file_rx = re.compile(args.file_regex) if getattr(args, "file_regex", None) else None
    for entry in entries:
        entry_path = getattr(entry, "path", None)
        if not entry_path:
            continue
        rel = Path(entry_path)
        if rel.suffix.lower() not in (DATA_SUFFIXES | MEDIA_SUFFIXES):
            continue
        if file_rx and not file_rx.search(str(rel)):
            continue
        selected.append((rel, getattr(entry, "size", None) or 0))
        if len(selected) >= args.max_files:
            break
    return selected


def summarize_artifact_file(artifact: Any, rel: Path, size: int, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Download and summarize one selected artifact file."""
    suffix = rel.suffix.lower()
    if size > ARTIFACT_SUMMARY_MAX_BYTES:
        return {"path": str(rel), "suffix": suffix, "size_bytes": size, "summary_skipped": f"file exceeds {ARTIFACT_SUMMARY_MAX_BYTES} byte safety cap"}
    file_path = download_artifact_entry(artifact, str(rel), root)
    try:
        row = short_file_row(file_path, root)
    except ValueError:
        row = {"path": str(rel), "suffix": suffix, "size_bytes": file_path.stat().st_size}
    include_field_counts = is_structured_field_count_candidate(rel) and size <= STRUCTURED_COUNT_MAX_BYTES
    if is_structured_field_count_candidate(rel) and not include_field_counts:
        row.setdefault("field_count_skips", []).append({"size_bytes": size, "reason": f"file exceeds {STRUCTURED_COUNT_MAX_BYTES} byte sidecar stats cap"})
    if suffix in {".jsonl", ".ndjson"}:
        row.update(summarize_jsonl(file_path, args.sample_rows, field_counts=include_field_counts))
    elif suffix in {".csv", ".tsv"}:
        row.update(summarize_csv(file_path, args.sample_rows, field_counts=include_field_counts))
    elif suffix == ".json":
        row.update(summarize_json(file_path, field_counts=include_field_counts))
    return row


def cmd_artifact_file_summary(args: argparse.Namespace) -> None:
    """Download an artifact and summarize bounded data files."""
    wb, path = ctx(args)
    ref, typ = resolve_artifact_ref(wb, path, args)
    try:
        artifact = wb.artifact(ref, type=typ)
    except Exception as exc:
        raise SystemExit(f"Artifact lookup failed for {ref} type={typ or '<default>'}: {type(exc).__name__}: {exc}") from exc
    print(f"path={path}\nartifact={artifact.name}\nartifact_type={artifact.type}")
    print(f"artifact_version={getattr(artifact, 'version', None)}")
    print(f"artifact_size_bytes={getattr(artifact, 'size', None)}")
    metadata = getattr(artifact, "metadata", {}) or {}
    print(f"artifact_metadata_key_count={len(metadata) if isinstance(metadata, dict) else 0}")
    if metadata:
        emit("artifact_metadata", compact_artifact_value(metadata))
    if context := artifact_context_row(artifact):
        emit("artifact_context", context)
    entries = sorted(getattr(getattr(artifact, "manifest", None), "entries", {}).values(), key=lambda entry: getattr(entry, "path", ""))
    print(f"manifest_file_count={len(entries)}")
    for entry in entries[: args.max_files]:
        emit("manifest_entry", {"path": getattr(entry, "path", None), "size_bytes": getattr(entry, "size", None)})
    selected_entries = artifact_entries_to_inspect(entries, args)
    with tempfile.TemporaryDirectory(prefix="wandb-artifact-") as tmp:
        root = Path(tmp)
        data_rows = metadata_rows = 0
        by_group: Counter[str] = Counter()
        for rel, size in selected_entries:
            row = summarize_artifact_file(artifact, rel, size, root, args)
            for field_count in row.pop("field_counts", []):
                field_count["path"] = str(rel)
                emit("structured_field_counts", field_count)
            for field_count_skip in row.pop("field_count_skips", []):
                field_count_skip["path"] = str(rel)
                emit("structured_field_counts_skipped", field_count_skip)
            rows_count = row.get("rows") or 0
            group = rel.parts[0] if rel.parts else "<root>"
            if "metadata" in str(rel).lower():
                metadata_rows += rows_count
            else:
                data_rows += rows_count; by_group[group] += rows_count
            emit("file_summary", row)
        print(f"inspected_files={len(selected_entries)}")
        print(f"data_record_count={data_rows}\nmetadata_record_count={metadata_rows}")
        emit("data_records_by_path_prefix", dict(sorted(by_group.items())))
        print(f"answer=artifact={artifact.name}; data_records={data_rows}; metadata_records={metadata_rows}")


# -----------------------------------------------------------------------------
# Checkpoint locations
# -----------------------------------------------------------------------------


def resolve_artifact_ref(wb: Any, path: str, args: argparse.Namespace) -> tuple[str, str | None]:
    if args.artifact:
        return (args.artifact if "/" in args.artifact else f"{path}/{args.artifact}", args.type)
    if not args.collection_hint:
        raise SystemExit("Pass --artifact or --collection-hint")
    tokens = [t for t in re.split(r"[^a-z0-9]+", args.collection_hint.lower()) if t]
    candidates = [
        (sum(t in collection.name.lower() for t in tokens) + (2 if args.type == row["type"] else 0), row["type"], collection.name)
        for row in artifact_collection_rows(wb, path, [args.type] if args.type else None)
        for collection in row["collections"]
    ]
    candidates = [c for c in candidates if c[0]]
    if not candidates:
        raise SystemExit(f"No artifact collection matching hint {args.collection_hint!r}")
    _, typ, name = max(candidates, key=lambda c: (c[0], c[2]))
    return f"{path}/{name}:latest", typ


def infer_checkpoint_file_pattern(files: list[str]) -> dict[str, Any]:
    """Infer common checkpoint filename structure from artifact files."""
    matches = []
    for name in files:
        match = re.search(r"(?P<prefix>.*?)(?P<step>\d{2,})(?P<suffix>\.[^.]+(?:\.[^.]+)?)?$", name)
        if match:
            matches.append({"path": name, "prefix": match.group("prefix"), "step": int(match.group("step")), "suffix": match.group("suffix") or ""})
    if not matches:
        return {}
    prefixes, suffixes = Counter(m["prefix"] for m in matches), Counter(m["suffix"] for m in matches)
    return {"pattern": f"{prefixes.most_common(1)[0][0]}<step>{suffixes.most_common(1)[0][0]}", "observed_steps": sorted({m["step"] for m in matches})[:12], "examples": [m["path"] for m in matches[:5]]}


def checkpoint_artifact_rows(run: Any, limit: int) -> list[dict[str, Any]]:
    """Summarize checkpoint-like artifacts logged by a run."""
    rows_ = []
    for artifact in list(run.logged_artifacts())[:limit]:
        text = f"{getattr(artifact, 'type', '')} {getattr(artifact, 'name', '')}".lower()
        if not any(token in text for token in CHECKPOINT_TOKENS):
            continue
        file_source = getattr(artifact, "files", [])
        file_objs = file_source() if callable(file_source) else file_source
        files = [getattr(f, "name", getattr(f, "path", "")) for f in list(file_objs)[:20]]
        metadata = getattr(artifact, "metadata", {}) or {}
        rows_.append({"name": artifact.name, "type": getattr(artifact, "type", None), "aliases": list(getattr(artifact, "aliases", []) or []), "metadata": metadata, "files": files, "checkpoint_file_pattern": infer_checkpoint_file_pattern(files)})
    return rows_


PATH_CONFIG_KEY_RE = re.compile(r"(^|[._/-])(path|dir|directory|folder|uri|url)([._/-]|$)")
PATH_CONFIG_VALUE_RE = re.compile(r"^(?:[./~]|[A-Za-z]:\\|[a-z][a-z0-9+.-]*://)|[/\\]")


def path_like_config_entry(key: str, value: Any) -> bool:
    """Return whether a config key/value pair looks like a logged file location."""
    scalar = scalar_config_value(value)
    if not isinstance(scalar, str) or not scalar.strip():
        return False
    if not PATH_CONFIG_KEY_RE.search(str(key).lower()):
        return False
    text = scalar.strip()
    return bool(PATH_CONFIG_VALUE_RE.search(text) or "/" in text or "\\" in text or Path(text).suffix)


def cmd_checkpoint_locations(args: argparse.Namespace) -> None:
    """Print config paths and checkpoint artifacts for selected runs."""
    wb, path = ctx(args)
    sample = [wb.run(f"{path}/{args.run_id}")] if args.run_id else runs(wb, path, args, filters=filters_from_args(args) or None, order=args.order, limit=args.limit)
    sample = filter_names(sample, args.name_include, [])
    print(f"path={path}")
    print(f"matched_runs={len(sample)}")
    for run in sample[: args.run_limit]:
        config_paths = {k: v for k, v in clean_config(run.config).items() if path_like_config_entry(k, v)}
        artifacts = checkpoint_artifact_rows(run, args.artifact_limit)
        emit("checkpoint_location_row", {"run": run_label(run), "id": run.id, "state": getattr(run, "state", None), "config_path_keys": config_paths, "checkpoint_artifacts": artifacts})
        print(f"answer=run={run_label(run)}; config_paths={json.dumps(config_paths, sort_keys=True, default=str) if config_paths else 'no path-like config key found in run.config'}; checkpoint_artifacts={len(artifacts)}")
        print("interpretation_hint=Report checkpoint locations only from printed config_path_keys or checkpoint_artifacts. If neither contains a local path, say the run logged artifact names/files but no trustworthy local absolute path; do not invent one.")


# -----------------------------------------------------------------------------
# Project overview signals
# -----------------------------------------------------------------------------


def interesting_config_items(config: dict[str, Any], limit: int = 16) -> dict[str, Any]:
    return dict(list(sorted((k, v) for k, v in config.items() if any(t in k.lower() for t in CONFIG_OVERVIEW_TOKENS) and scalar_config_value(v) is not None))[:limit])


def config_value_examples(sample: list[Any], limit: int = 18) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = defaultdict(list)
    for run in sample:
        for key, value in clean_config(run.config).items():
            scalar = scalar_config_value(value)
            if scalar is not None and any(t in key.lower() for t in ("model", "dataset", "task", "optimizer", "loss", "method", "flow", "job")) and scalar not in values[key]:
                values[key].append(scalar)
    return dict(list(sorted(values.items(), key=lambda item: (-len(item[1]), item[0])))[:limit])


def run_name_terms(sample: list[Any], limit: int = 18) -> dict[str, int]:
    counts = Counter(t.strip("_.-") for run in sample for t in re.findall(r"[a-z][a-z0-9_.-]{2,}", run_label(run).lower()))
    return dict((k, v) for k, v in counts.most_common(limit) if k not in PROJECT_NAME_STOPWORDS)


def terms_from_text(text: str, terms: list[str], limit: int = 18) -> dict[str, int]:
    lower = text.lower()
    return dict(Counter(t for t in terms if re.search(rf"(^|[^a-z0-9]){re.escape(t)}", lower)).most_common(limit))


def artifact_name_terms(compact: list[dict[str, Any]], limit: int = 18) -> dict[str, int]:
    counts = Counter(t.strip("_.-") for row in compact for name in row.get("sample_collections", []) for t in re.findall(r"[a-z][a-z0-9_.-]{2,}", str(name).lower()))
    return dict((k, v) for k, v in counts.most_common(limit) if k not in PROJECT_NAME_STOPWORDS)


def metric_families(keys: list[str]) -> list[str]:
    return sorted({part for key in keys for part in str(key).split("/") if part and not part.startswith("_") and not re.search(r"\d+$", part)})


def project_workflow_hints(name_terms: dict[str, int], artifact_terms: dict[str, int], metric_counts: dict[str, int], artifacts: list[dict[str, Any]], weave_counts: dict[str, Any]) -> list[str]:
    weave_terms = []
    for key in ("sample_component_terms", "sample_root_ops"):
        if isinstance(weave_counts.get(key), dict): weave_terms.extend(weave_counts[key])
    text = " ".join([*name_terms, *artifact_terms, *metric_counts, *(a.get("type", "") for a in artifacts), *weave_terms]).lower()
    hints = []
    if any(t in text for t in ("collect", "trajectory", "episode")) and any(t in text for t in ("sft", "fine-tun", "finetun", "checkpoint", "lora", "model")):
        hints.append("supports both trajectory/data collection and fine-tuning/model-training jobs or outputs")
    if any(t in text for t in ("embedding", "rerank", "retrieval", "search", "vector", "query")) and any(t in text for t in ("eval", "score", "scorer", "correctness", "recall", "mrr", "chat", "answer", "agent")):
        hints.append("retrieval/RAG-style chatbot or agent evaluation pipeline")
    return hints


def run_snapshot_row(run: Any, path: str) -> dict[str, Any]:
    config, summary = clean_config(run.config), dict(run.summary_metrics or {})
    numeric = [k for k, v in summary.items() if not str(k).startswith("_") and coerce_float(v) is not None]
    return {**run_row(run, path=path), "job_type": getattr(run, "job_type", None), "config_keys": sorted(config)[:40], "interesting_config": interesting_config_items(config), "metric_families": metric_families(numeric)[:30], "summary_metric_examples": {k: summary.get(k) for k in numeric[:12]}}


def optional_weave_counts(path: str) -> dict[str, Any]:
    try:
        import logging, weave
        from weave.trace_server.trace_server_interface import CallsQueryStatsReq
        logging.getLogger("weave").setLevel(logging.ERROR)
        client = weave.init(path)
        total = client.server.calls_query_stats(CallsQueryStatsReq(project_id=path)).count
        root = client.server.calls_query_stats(CallsQueryStatsReq(project_id=path, filter={"trace_roots_only": True})).count
        ops, terms = Counter(), Counter()
        for call in client.get_calls(limit=40, filter={"trace_roots_only": True}, columns=["op_name", "display_name", "inputs", "output"]):
            short = (getattr(call, "op_name", None) or "").split("/op/")[-1].rsplit(":", 1)[0]
            if short: ops[short] += 1
            terms.update(terms_from_text(" ".join(str(getattr(call, a, "") or "")[:1500] for a in ["op_name", "display_name", "inputs", "output"]), WEAVE_COMPONENT_TERMS))
        return {"checked": True, "total_calls": total, "root_calls": root, "sample_root_ops": dict(ops.most_common(12)), "sample_component_terms": dict(terms.most_common(18))}
    except Exception as exc:
        return {"checked": False, "error": str(exc)[:240]}


# -----------------------------------------------------------------------------
# Workspaces
# -----------------------------------------------------------------------------


# Sample recent, middle, and oldest runs so auto-panel inference sees older metric families.
WORKSPACE_HISTORY_KEY_SAMPLE_SIZE = 1000
WORKSPACE_HISTORY_WINDOWS = ("recentRuns", "middleRuns", "oldestRuns")
WORKSPACE_VIEW_PAGE_SIZE = 100

WORKSPACE_RUN_COUNT_GQL_QUERY = """query PublicWorkspaceRunCount($entityName:String!, $projectName:String!){
  project(entityName:$entityName, name:$projectName){
    runCount
  }
}"""

WORKSPACE_VIEWS_GQL_QUERY = """query PublicWorkspaceViews($entityName:String!, $projectName:String!, $first:Int!, $after:String, $viewName:String){
  project(entityName:$entityName, name:$projectName){
    allViews(viewType:"project-view", first:$first, after:$after, viewName:$viewName){
      edges{ node{ id name displayName spec } }
      pageInfo{ hasNextPage endCursor }
    }
  }
}"""

WORKSPACE_HISTORY_KEYS_GQL_QUERY = """query PublicWorkspaceHistoryKeys($entityName:String!, $projectName:String!, $sampleSize:Int!, $after:String, $order:String!){
  project(entityName:$entityName, name:$projectName){
    runs(first:$sampleSize, after:$after, order:$order){
      historyKeys(format:PLAINTEXT)
    }
  }
}"""


def workspace_view_name_from_url_token(token: str) -> str:
    if token.startswith("nw-"):
        return token
    suffix = "w" if token.startswith("nwuser") else "v"
    return f"nw-{token}-{suffix}"


def workspace_url_target(url: str) -> tuple[str, str]:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise SystemExit("Workspace URL must include entity and project.")

    view_token = (parse_qs(parsed.query).get("nw") or [""])[0]
    if not view_token:
        raise SystemExit("Workspace view URL must include an `nw` query token; omit `--view-url` to inventory all project views.")
    return f"{parts[0]}/{parts[1]}", workspace_view_name_from_url_token(view_token)


def compact_workspace_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        items = [compact_workspace_value(item, depth=depth + 1) for item in value[:8]]
        return items + ([f"... {len(value) - 8} more"] if len(value) > 8 else [])
    if isinstance(value, dict):
        if depth >= 2:
            return f"<{len(value)} keys>"
        items = list(value.items())
        compact = {str(key): compact_workspace_value(child, depth=depth + 1) for key, child in items[:8]}
        if len(items) > 8:
            compact["..."] = f"{len(items) - 8} more keys"
        return compact
    return str(value)[:200]


def panel_summary(panel: dict[str, Any]) -> dict[str, Any]:
    config = panel.get("config") or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Workspace panel `{panel.get('__id__') or panel.get('id')}` has non-object config.")

    row = {
        "id": panel.get("__id__") or panel.get("id"),
        "type": panel.get("viewType") or panel.get("type") or panel.get("__typename"),
        "title": panel.get("title") or config.get("title"),
        "is_auto": panel.get("isAuto"),
        "config": compact_workspace_value(config),
    }
    return {key: value for key, value in row.items() if value not in (None, {}, [])}


def history_key_types(spec: Any) -> list[str]:
    if not isinstance(spec, dict):
        return []
    type_counts = spec.get("typeCounts") or []
    if not isinstance(type_counts, list):
        return []
    return [str(row.get("type")) for row in type_counts if isinstance(row, dict) and row.get("type")]


def history_key_section_name(key: str) -> str:
    if key.startswith("_"):
        return ""
    if "/" in key:
        return key.split("/", 1)[0].lower()
    return "charts"


def auto_panel_summary(key: str, spec: Any) -> dict[str, Any]:
    types = history_key_types(spec)
    return {
        "title": key.split("/", 1)[1] if "/" in key else key,
        "key": key,
        "type": first(types, "metric"),
        "types": types,
        "source": "project_historyKeys",
        "spec": compact_workspace_value(spec),
    }


def workspace_history_window_keys(window: Any) -> dict[str, Any]:
    keys = ((window or {}).get("historyKeys") or {}).get("keys") or {}
    return keys if isinstance(keys, dict) else {}


def workspace_history_windows(project_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in WORKSPACE_HISTORY_WINDOWS:
        rows.append({"name": name.removesuffix("Runs"), "key_count": len(workspace_history_window_keys(project_data.get(name)))})
    return rows


def workspace_history_window(name: str, response: dict[str, Any]) -> dict[str, Any]:
    project_data = response.get("project")
    if project_data is None:
        return {"name": name, "historyKeys": {"keys": {}}}
    return {"name": name, "historyKeys": ((project_data.get("runs") or {}).get("historyKeys") or {})}


def auto_panels_by_section(project_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    history_keys: dict[str, Any] = {}
    for name in WORKSPACE_HISTORY_WINDOWS:
        for key, spec in workspace_history_window_keys(project_data.get(name)).items():
            history_keys.setdefault(str(key), spec)

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, spec in sorted(history_keys.items()):
        section_name = history_key_section_name(str(key))
        if section_name:
            rows[section_name].append(auto_panel_summary(str(key), spec))
    return dict(rows)


def workspace_middle_after_cursor(run_count: int, sample_size: int) -> str | None:
    import base64

    start = max((run_count - sample_size) // 2, 0)
    if start <= 0:
        return None
    return base64.b64encode(f"arrayconnection:{start - 1}".encode()).decode()


def workspace_execute_query(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    return execute_graphql(api(), query, variables)


def workspace_history_keys_query(entity: str, project: str, *, order: str, after: str | None = None) -> dict[str, Any]:
    return workspace_execute_query(
        WORKSPACE_HISTORY_KEYS_GQL_QUERY,
        {
            "entityName": entity,
            "projectName": project,
            "sampleSize": WORKSPACE_HISTORY_KEY_SAMPLE_SIZE,
            "after": after,
            "order": order,
        },
    )


def workspace_views_query(entity: str, project: str, *, view_name: str | None = None) -> dict[str, Any]:
    edges = []
    after = None
    project_data: dict[str, Any] | None = None
    while True:
        response = workspace_execute_query(
            WORKSPACE_VIEWS_GQL_QUERY,
            {
                "entityName": entity,
                "projectName": project,
                "first": WORKSPACE_VIEW_PAGE_SIZE,
                "after": after,
                "viewName": view_name,
            },
        )
        project_data = response.get("project")
        if project_data is None:
            raise SystemExit(f"Project `{entity}/{project}` not found or not accessible.")
        connection = project_data.get("allViews") or {}
        edges.extend(connection.get("edges") or [])
        page_info = connection.get("pageInfo") or {}
        if view_name or not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")
        if not after:
            raise SystemExit("Workspace view response was missing the next page cursor.")
    return {"allViews": {"edges": edges}}


def workspace_project_data(entity: str, project: str, *, view_name: str | None = None) -> dict[str, Any]:
    base = {"entityName": entity, "projectName": project}
    with ThreadPoolExecutor(max_workers=4) as pool:
        count_future = pool.submit(workspace_execute_query, WORKSPACE_RUN_COUNT_GQL_QUERY, base)
        views_future = pool.submit(workspace_views_query, entity, project, view_name=view_name)
        recent_future = pool.submit(workspace_history_keys_query, entity, project, order="-created_at")
        oldest_future = pool.submit(workspace_history_keys_query, entity, project, order="+created_at")

        count_data = count_future.result().get("project")
        if count_data is None:
            raise SystemExit(f"Project `{entity}/{project}` not found or not accessible.")
        run_count = int(count_data.get("runCount") or 0)
        middle_after = workspace_middle_after_cursor(run_count, WORKSPACE_HISTORY_KEY_SAMPLE_SIZE)
        middle_future = (
            pool.submit(workspace_history_keys_query, entity, project, order="-created_at", after=middle_after)
            if middle_after
            else None
        )

        project_data = views_future.result()
        if not project_data:
            raise SystemExit(f"Project `{entity}/{project}` not found or not accessible.")
        project_data["runCount"] = run_count
        project_data["recentRuns"] = workspace_history_window("recent", recent_future.result())
        project_data["middleRuns"] = workspace_history_window("middle", middle_future.result()) if middle_future else {"historyKeys": {"keys": {}}}
        project_data["oldestRuns"] = workspace_history_window("oldest", oldest_future.result())
        return project_data


def section_summary(section: dict[str, Any], auto_panels: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    panels = section.get("panels") or []
    if not isinstance(panels, list):
        raise SystemExit(f"Workspace section `{section.get('name')}` has non-list panels.")

    panel_rows = []
    for panel in panels:
        if not isinstance(panel, dict):
            raise SystemExit(f"Workspace section `{section.get('name')}` has a non-object panel.")
        panel_rows.append(panel_summary(panel))

    auto_section_metric_key_hints = []
    if section.get("isPanelsAuto"):
        auto_section_metric_key_hints = auto_panels.get(str(section.get("name", "")).lower(), [])

    row = {
        "name": section.get("name", ""),
        "panel_count": len(panel_rows),
        "confirmed_saved_panel_count": len(panel_rows),
        "auto_section_metric_key_hint_count": len(auto_section_metric_key_hints),
        "saved_plus_metric_hint_count": len(panel_rows) + len(auto_section_metric_key_hints),
        "is_open": section.get("isOpen"),
        "is_panels_auto": section.get("isPanelsAuto"),
        "pinned": section.get("pinned"),
        "panels": panel_rows,
    }
    if auto_section_metric_key_hints:
        row["auto_section_metric_key_hints"] = auto_section_metric_key_hints
    return row


def workspace_view_summary(node: dict[str, Any], auto_panels: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    display_name = node.get("displayName") or node.get("name")
    raw_spec = node.get("spec")
    if not raw_spec:
        raise SystemExit(f"Workspace view `{display_name}` has no JSON spec.")

    try:
        spec = json.loads(raw_spec)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Workspace view `{display_name}` has invalid JSON spec: {exc}") from exc

    section = spec.get("section")
    if not isinstance(section, dict):
        raise SystemExit(f"Workspace view `{display_name}` has no section spec.")

    panel_bank = section.get("panelBankConfig")
    if not isinstance(panel_bank, dict):
        raise SystemExit(f"Workspace view `{display_name}` has no panel bank config.")

    sections = panel_bank.get("sections")
    if not isinstance(sections, list):
        raise SystemExit(f"Workspace view `{display_name}` has non-list sections.")

    section_rows = []
    for section in sections:
        if not isinstance(section, dict):
            raise SystemExit(f"Workspace view `{display_name}` has a non-object section.")
        section_rows.append(section_summary(section, auto_panels))

    return {
        "id": node.get("id"),
        "name": node.get("name"),
        "display_name": display_name,
        "section_count": len(section_rows),
        "panel_count": sum(section["panel_count"] for section in section_rows),
        "confirmed_saved_panel_count": sum(section["confirmed_saved_panel_count"] for section in section_rows),
        "auto_section_metric_key_hint_count": sum(section["auto_section_metric_key_hint_count"] for section in section_rows),
        "saved_plus_metric_hint_count": sum(section["saved_plus_metric_hint_count"] for section in section_rows),
        "sections": section_rows,
    }


def project_workspace_views(path: str, *, view_name: str | None = None) -> dict[str, Any]:
    entity, project = split_path(path)
    project_data = workspace_project_data(entity, project, view_name=view_name)

    auto_panels = auto_panels_by_section(project_data)
    views = []
    for edge in ((project_data.get("allViews") or {}).get("edges") or []):
        if not isinstance(edge, dict):
            raise SystemExit("Workspace view response contained a non-object edge.")
        node = edge.get("node") or {}
        if not isinstance(node, dict):
            raise SystemExit("Workspace view response contained a non-object node.")

        if view_name and node.get("name") != view_name:
            continue
        views.append(workspace_view_summary(node, auto_panels))

    if view_name and not views:
        raise SystemExit(f"No saved workspace view named `{view_name}` found for `{path}`.")

    return {
        "source": "graphql_view_url" if view_name else "graphql_allViews",
        "path": path,
        "run_count": project_data.get("runCount"),
        "auto_section_metric_key_hint_source": f"project runs.historyKeys(recent/middle/oldest first={WORKSPACE_HISTORY_KEY_SAMPLE_SIZE})",
        "panel_key_windows": workspace_history_windows(project_data),
        "view_count": len(views),
        "section_count": sum(view["section_count"] for view in views),
        "panel_count": sum(view["panel_count"] for view in views),
        "confirmed_saved_panel_count": sum(view["confirmed_saved_panel_count"] for view in views),
        "auto_section_metric_key_hint_count": sum(view["auto_section_metric_key_hint_count"] for view in views),
        "saved_plus_metric_hint_count": sum(view["saved_plus_metric_hint_count"] for view in views),
        "views": views,
    }


def workspace_inventory_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "views": int(payload.get("view_count") or 0),
        "sections": int(payload.get("section_count") or 0),
        "panels": workspace_visible_panel_count(payload),
    }


def workspace_visible_panel_count(row: dict[str, Any]) -> int:
    return int(row.get("saved_plus_metric_hint_count", row.get("panel_count", 0)) or 0)


WORKSPACE_PANEL_LABEL_LIMIT = 100
WORKSPACE_SHARED_SECTION_MIN_VIEWS = 2


def markdown_cell(value: Any, default: str = "<unnamed>") -> str:
    text = default if value in (None, "") else str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def panel_label(panel: dict[str, Any]) -> str:
    config = panel.get("config") if isinstance(panel.get("config"), dict) else {}
    metrics = config.get("metrics") or config.get("metric") or panel.get("key")
    if isinstance(metrics, list):
        metrics = ", ".join(str(metric) for metric in metrics[:3])
    return str(panel.get("title") or metrics or panel.get("type") or panel.get("id") or "<unnamed>")


def section_panel_label_list(section: dict[str, Any]) -> list[str]:
    panels = [*(section.get("panels") or []), *(section.get("auto_section_metric_key_hints") or [])]
    return [panel_label(panel) for panel in panels if isinstance(panel, dict)]


def section_panel_labels(section: dict[str, Any], *, limit: int = WORKSPACE_PANEL_LABEL_LIMIT) -> str:
    labels = section_panel_label_list(section)
    if len(labels) > limit:
        labels = [*labels[:limit], f"... +{len(labels) - limit} more"]
    return "; ".join(labels) if labels else "-"


def section_inventory_signature(section: dict[str, Any]) -> tuple[Any, ...]:
    return (section.get("name") or "", workspace_visible_panel_count(section), tuple(section_panel_label_list(section)))


def view_display_name(view: dict[str, Any]) -> str:
    return str(view.get("display_name") or view.get("name") or "<unnamed>")


def compact_name_list(names: list[str], *, limit: int = 8) -> str:
    if len(names) > limit:
        names = [*names[:limit], f"... +{len(names) - limit} more"]
    return ", ".join(names)


def shared_section_inventory(payload: dict[str, Any]) -> dict[tuple[Any, ...], str]:
    grouped: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for view in payload.get("views", []):
        for section in view.get("sections", []):
            grouped[section_inventory_signature(section)].append((view, section))

    shared: dict[tuple[Any, ...], str] = {}
    rows = []
    for refs in grouped.values():
        view_names = list(dict.fromkeys(view_display_name(view) for view, _section in refs))
        if len(view_names) < WORKSPACE_SHARED_SECTION_MIN_VIEWS:
            continue

        section = refs[0][1]
        shared_id = f"S{len(rows) + 1}"
        shared[section_inventory_signature(section)] = shared_id
        rows.append((shared_id, view_names, section))

    if rows:
        print("## Shared Section Inventories")
        print()
        print("These section contents appear in multiple views; individual view rows below reference them by ID.")
        print()
        print("| ID | Views | Section | Panels | Panel/chart names |")
        print("|---|---|---|---:|---|")
        for shared_id, view_names, section in rows:
            print(
                f"| {shared_id} | "
                f"{markdown_cell(compact_name_list(view_names))} | "
                f"{markdown_cell(section.get('name'))} | "
                f"{workspace_visible_panel_count(section)} | "
                f"{markdown_cell(section_panel_labels(section))} |"
            )
        print()

    return shared


def print_workspace_view(view: dict[str, Any], shared_sections: dict[tuple[Any, ...], str]) -> None:
    view_name = view.get("display_name") or view.get("name") or "<unnamed>"

    print(f"## View: {view_name}")
    print()
    print(f"- Internal name: `{view.get('name')}`")
    if view.get("id"):
        print(f"- ID: `{view.get('id')}`")
    print(f"- Sections: {view.get('section_count', 0)}")
    print(f"- Panels: {workspace_visible_panel_count(view)}")
    print()
    print("| Section | Panels | Panel/chart names | Open | Pinned |")
    print("|---|---:|---|---|---|")
    for section in view.get("sections", []):
        shared_id = shared_sections.get(section_inventory_signature(section))
        panel_names = f"Same as {shared_id}" if shared_id else section_panel_labels(section)
        print(
            f"| {markdown_cell(section.get('name'))} | "
            f"{workspace_visible_panel_count(section)} | "
            f"{markdown_cell(panel_names)} | "
            f"{'yes' if section.get('is_open') else 'no'} | "
            f"{'yes' if section.get('pinned') else 'no'} |"
        )
    print()


def print_workspace_views(payload: dict[str, Any]) -> None:
    counts = workspace_inventory_counts(payload)

    print("# Workspace View Inventory")
    print()
    print(f"- Path: `{payload.get('path')}`")
    print(f"- Source: `{payload.get('source')}`")
    print(f"- Views: {counts['views']}")
    print(f"- Sections: {counts['sections']}")
    print(f"- Visible workspace panels: {counts['panels']}")
    print(f"- Panel/chart names: shown up to {WORKSPACE_PANEL_LABEL_LIMIT} per section")
    print()

    shared_sections = shared_section_inventory(payload)

    for view in payload.get("views", []):
        print_workspace_view(view, shared_sections)


def cmd_workspace_views(args: argparse.Namespace) -> None:
    """Print compact workspace evidence: views, sections, panel counts, chart names, and open/pinned state."""
    if getattr(args, "view_url", None):
        path, view_name = workspace_url_target(args.view_url)
        payload = project_workspace_views(path, view_name=view_name)
        payload["view_url"] = args.view_url
    else:
        payload = project_workspace_views(resolve_path(getattr(args, "entity", None), getattr(args, "project", None)))

    print_workspace_views(payload)


# -----------------------------------------------------------------------------
# Project snapshots and scale checks
# -----------------------------------------------------------------------------


def cmd_project_snapshot(args: argparse.Namespace) -> None:
    """Print a compact project overview from bounded samples."""
    wb, path = ctx(args)
    include_sweeps = args.include_sweeps
    sample_limit = capped_hydrated_limit(args.sample, PROJECT_SNAPSHOT_HYDRATED_SIDE_CAP)
    sample = sample_metadata_runs(wb, path, sample_limit, include_sweeps=include_sweeps, lazy=False)
    states = {
        state: count(wb, path, {"state": state}, include_sweeps=include_sweeps)
        for state in ["finished", "failed", "crashed", "running"]
    }
    configs = Counter(k for r in sample for k in clean_config(r.config))
    metric_counts = Counter(f for r in sample for f in metric_families([k for k, v in (r.summary_metrics or {}).items() if coerce_float(v) is not None]))
    job_types = Counter(getattr(r, "job_type", None) or "<none>" for r in sample)
    job_type_counts = {
        jt: count(wb, path, {"jobType": jt}, include_sweeps=include_sweeps)
        for jt in sorted(j for j in job_types if j != "<none>")[:8]
    }
    users = sorted({u for r in sample for u in [getattr(getattr(r, "user", None), "username", None) or getattr(getattr(r, "user", None), "name", None)] if u})
    run_count = count(wb, path, include_sweeps=include_sweeps)
    print(f"path={path}\nrun_count={run_count}")
    print_sample_scope(
        "project-snapshot",
        requested={"recent": args.sample, "oldest": args.sample, "total": args.sample * 2},
        effective={"recent": sample_limit, "oldest": sample_limit, "total": sample_limit * 2},
        actual=len(sample),
        strategy="hydrated recent and oldest run slices",
        cap_applied=sample_limit < args.sample,
        note="project snapshot config, metric-family, user, and run-name signals come from sampled recent/oldest runs only",
    )
    if sample_limit < args.sample:
        print(f"sample_caveat=project-snapshot capped each recent/oldest hydrated slice at {sample_limit} of requested {args.sample} to avoid per-run full-summary timeouts")
    emit("state_counts", states)
    print(f"sampled_runs={len(sample)}")
    emit("job_type_counts_sample", dict(sorted(job_types.items())))
    if job_type_counts:
        emit("job_type_counts", job_type_counts)
    print("sample_users=" + ", ".join(users))
    emit("top_config_keys_sample", dict(configs.most_common(24)))
    emit("top_metric_families_sample", dict(metric_counts.most_common(24)))
    artifact_checked = bool(args.include_artifacts or run_count <= PROJECT_SNAPSHOT_ARTIFACT_RUN_COUNT_LIMIT)
    emit(
        "artifact_scope",
        {
            "command": "project-snapshot",
            "requested": bool(args.include_artifacts),
            "checked": artifact_checked,
            "project_run_count": run_count,
            "large_project_skip_threshold": PROJECT_SNAPSHOT_ARTIFACT_RUN_COUNT_LIMIT,
            "meaning": "artifact collections are included only when checked=true; otherwise artifact fields are not evidence of absence",
        },
    )
    if artifact_checked:
        artifact_rows = artifact_collection_rows(wb, path, per_page=args.artifact_per_page)
        artifact_errors = [{"type": r["type"], "error": r["error"]} for r in artifact_rows if r.get("error")]
        compact_artifacts = [{"type": r["type"], "collection_count": r["collection_count"], "sample_collections": r["samples"][:3], **({"error": r["error"]} if r.get("error") else {})} for r in artifact_rows]
        print(f"artifact_type_count={len(compact_artifacts)}")
        if artifact_errors:
            print(f"artifact_collection_error_count={len(artifact_errors)}")
            for error in artifact_errors:
                emit("artifact_collection_error", error)
        emit("artifact_type_rows", compact_artifacts)
    else:
        compact_artifacts = []
        print("artifact_collection_caveat=skipped for large project; use artifact-inventory or pass --include-artifacts when artifact collections are required")
        print("artifact_type_count=not_checked")
    config_values, names, artifact_terms = config_value_examples(sample), run_name_terms(sample), artifact_name_terms(compact_artifacts)
    emit("config_value_examples", config_values)
    emit("run_name_terms_sample", names)
    emit("artifact_name_terms_sample", artifact_terms)
    weave_counts = optional_weave_counts(path) if getattr(args, "include_weave", False) else {"checked": False}
    print("weave_counts=" + (json.dumps(weave_counts, sort_keys=True, default=str) if getattr(args, "include_weave", False) else "not_checked"))
    hints = project_workflow_hints(names, artifact_terms, dict(metric_counts), compact_artifacts, weave_counts)
    emit("workflow_hints", hints)
    signals = {"run_count": run_count, "state_counts": states, "job_types": dict(sorted(job_types.items())), "job_type_counts": job_type_counts, "users": users, "top_config_values": config_values, "top_metric_families": dict(metric_counts.most_common(16)), "artifact_types_checked": artifact_checked, "artifact_types": compact_artifacts[:12], "artifact_name_terms": artifact_terms, "workflow_hints": hints, "run_name_terms": names, "weave": weave_counts}
    emit("project_summary_signals", signals)
    artifact_answer = f"artifact_types={len(compact_artifacts)}" if artifact_checked else "artifact_types_not_checked"
    print(f"answer=runs={signals['run_count']}; states={json.dumps(states, sort_keys=True)}; sampled_recent_and_oldest={len(sample)}; {artifact_answer}; " + (f"weave_total_calls={weave_counts.get('total_calls')}; weave_root_calls={weave_counts.get('root_calls')}" if weave_counts.get("checked") else "weave_counts_not_checked"))
    print("interpretation_hint=Summarize the project purpose from project_summary_signals: use run/state counts, the exact job_type_counts (the real run-type split — ground comparisons on these, not run-name guesses), workflow_hints, config value examples, metric families, artifact collection names/types only when artifact_types_checked is true, and Weave counts when checked. Keep exact claims tied to printed evidence; do not mention Weave unless weave.checked is true; stop without extra broad scans.")
    print("stop_rule=project_summary_signals_are_answer_ready")
    for run in sample[: args.sample_rows]:
        emit("run_sample", run_snapshot_row(run, path))


def cmd_large_project(args: argparse.Namespace) -> None:
    """Dispatch bounded recipes for very large W&B run tables."""
    if getattr(args, "op", None) and getattr(args, "recipe", None) and args.op != args.recipe:
        raise SystemExit("--op and --recipe disagree; pass only one recipe")
    args.op = getattr(args, "recipe", None) or getattr(args, "op", None)
    if not args.op:
        raise SystemExit("large-project requires --recipe RECIPE or --op RECIPE")
    wb, path = ctx(args)
    entity, project = split_path(path)
    include_sweeps = args.include_sweeps
    print(f"path={path}")
    if args.op == "status":
        total = count(wb, path, include_sweeps=include_sweeps)
        finished = count(wb, path, {"state": "finished"}, include_sweeps=include_sweeps)
        crashed = count(wb, path, {"state": "crashed"}, include_sweeps=include_sweeps)
        rate = 100 * crashed / total if total else 0.0
        print(f"total={total}\nfinished={finished}\ncrashed={crashed}\ncrash_rate_pct={rate:.2f}")
        print(f"answer=total={total}; finished={finished}; crashed={crashed}; crash_rate={rate:.2f}%")
    elif args.op == "project-description":
        desc = getattr(wb.project(project, entity=entity), "description", "") or ""
        print(f"description={desc!r}\nanswer=" + (desc if desc.strip() else "No project description is set."))
    elif args.op == "run-info":
        target = getattr(args, "run_id", None) or getattr(args, "run_name", None)
        if not target:
            raise SystemExit("large-project --recipe run-info requires --run RUN_NAME_OR_ID or --run-id RUN_ID")
        run = resolve_run_ref(
            wb,
            path,
            target,
            limit=max(args.limit, 200),
            include_sweeps=include_sweeps,
            prefer_direct_id=should_probe_direct_run_id(getattr(args, "run_id", None), target),
        )
        if run is None:
            raise SystemExit(f"Could not find run matching {target!r}. Try --run with the exact display name or --run-id with the exact W&B run id.")
        row = run_row(run, include_url=True, path=path)
        emit("run", row)
        print("answer=" + "; ".join(f"{k}={row[k]}" for k in ["name", "id", "created_at", "state", "url"] if k in row and row[k]))
    elif args.op in {"recent", "finished", "most-recent", "oldest", "earliest-finished", "latest-crashed"}:
        filters = {"state": "finished"} if args.op in {"finished", "earliest-finished"} else {"state": "crashed"} if args.op == "latest-crashed" else None
        order = args.order or ("+created_at" if args.op in {"oldest", "earliest-finished"} else "-created_at")
        provided_limit = bool(getattr(args, "_provided_options", set()) & {"--sample", "--limit"})
        limit = args.limit if args.op in {"recent", "finished"} or provided_limit else 1
        sample = get_runs(wb, path, filters=filters, order=order, limit=limit, include_sweeps=include_sweeps)
        for run in sample:
            emit("run", run_row(run, path=path))
        if sample:
            row = run_row(sample[0], path=path)
            print("answer=" + "; ".join(f"{k}={row[k]}" for k in ["name", "id", "created_at", "state", "url"] if k in row and row[k]))
    elif args.op == "creator":
        sample = get_runs(wb, path, order="-created_at", limit=args.limit, include_sweeps=include_sweeps)
        users = sorted({getattr(getattr(r, "user", None), "username", None) or getattr(getattr(r, "user", None), "name", None) for r in sample} - {None})
        print("users=" + ", ".join(users))
        print("answer=" + (", ".join(users) if users else "No run creator found."))
    elif args.op == "name-pattern":
        sample = filter_names(
            sample_metadata_runs(wb, path, args.samples, include_sweeps=include_sweeps),
            args.name_include,
            args.name_exclude,
            args.name_prefix,
            args.name_regex,
        )
        names = [run_label(r) for r in sample]
        label, groups = name_pattern_summary(names)
        print(f"sampled_runs={len(names)}")
        print(f"sample_caveat=name discovery uses newest and oldest slices with per-slice limit {args.samples}")
        for key, values in groups.items(): print(f"pattern_count.{key}={len(values)}")
        print(f"pattern={label}\nanswer={label}; examples=" + ", ".join(names[:8]))
    elif args.op in {"run-config", "summary-stats", "metric-count"}:
        target = getattr(args, "run_id", None) or getattr(args, "run_name", None)
        if target:
            run = resolve_run_ref(
                wb,
                path,
                target,
                limit=max(args.limit, 200),
                include_sweeps=include_sweeps,
                prefer_direct_id=should_probe_direct_run_id(getattr(args, "run_id", None), target),
            )
            sample = [run] if run is not None else []
        else:
            sample = get_runs(
                wb,
                path,
                filters={"state": args.state} if args.state else None,
                order=args.order or "-created_at",
                limit=max(args.limit, 1),
                include_sweeps=include_sweeps,
            )
            sample = filter_names(sample, args.name_include, args.name_exclude, args.name_prefix, args.name_regex)
        if not sample:
            print("answer=no runs found")
            return
        run = sample[0]
        if args.op == "run-config":
            config = config_subset(run, args.config_key)
            for k, v in sorted(config.items()):
                print(f"{k}={jsonable(v)}")
            print("answer=" + "; ".join(f"{k}={jsonable(v)}" for k, v in sorted(config.items())))
        else:
            keys = [k for k in (run.summary_metrics or {}) if not str(k).startswith("_")]
            numeric = [(k, coerce_float(run.summary_metrics.get(k))) for k in keys if coerce_float(run.summary_metrics.get(k)) is not None]
            print(f"{'metric' if args.op == 'metric-count' else 'summary_key'}_count={len(keys)}")
            for k, v in numeric[: args.limit]:
                print(f"{k}={v}")
            print(f"answer=run={run_label(run)}; key_count={len(keys)}")
    elif args.op == "config-profiles":
        left_start, left_end = month_bounds(args.left_month) if args.left_month else (args.left_start, args.left_end)
        right_start, right_end = month_bounds(args.right_month) if args.right_month else (args.right_start, args.right_end)
        left = get_runs(
            wb,
            path,
            filters={"state": "finished", **created_at_filter(left_start, left_end)},
            order=args.left_order,
            limit=1,
            include_sweeps=include_sweeps,
        )
        right = get_runs(
            wb,
            path,
            filters={"state": "finished", **created_at_filter(right_start, right_end)},
            order=args.right_order,
            limit=1,
            include_sweeps=include_sweeps,
        )
        if not left or not right:
            print("answer=missing one or both comparison profiles")
            return
        left_run, right_run = left[0], right[0]
        left_cfg, right_cfg = config_subset(left_run, args.config_key), config_subset(right_run, args.config_key)
        emit(args.left_label or args.left_month or "left_profile", {**run_row(left_run), "config": left_cfg})
        emit(args.right_label or args.right_month or "right_profile", {**run_row(right_run), "config": right_cfg})
        keys = args.config_key or sorted(set(left_cfg) | set(right_cfg))[:20]
        ratios = {k: right / left for k in keys for left in [coerce_float(left_cfg.get(k))] for right in [coerce_float(right_cfg.get(k))] if left not in {None, 0.0} and right is not None}
        emit("ratios", ratios)
        print(f"answer={run_label(left_run)} -> {run_label(right_run)}; " + "; ".join(f"{k}: {jsonable(left_cfg.get(k))} -> {jsonable(right_cfg.get(k))}" for k in keys))
    elif args.op == "duration-range":
        shortest_candidates = get_runs(
            wb,
            path,
            filters={"state": "finished"},
            order="+summary_metrics._runtime",
            limit=args.limit,
            include_sweeps=include_sweeps,
        )
        longest_candidates = get_runs(
            wb,
            path,
            filters={"state": "finished"},
            order="-summary_metrics._runtime",
            limit=args.limit,
            include_sweeps=include_sweeps,
        )
        shortest, shortest_runtime, shortest_checked = first_run_with_runtime(shortest_candidates)
        longest, longest_runtime, longest_checked = first_run_with_runtime(longest_candidates)
        print(f"candidate_limit={args.limit}")
        print(f"shortest_candidates_checked={shortest_checked}")
        print(f"longest_candidates_checked={longest_checked}")
        if shortest:
            emit("shortest_run", {**run_row(shortest), "runtime_seconds": shortest_runtime})
        if longest:
            emit("longest_run", {**run_row(longest), "runtime_seconds": longest_runtime})
        if shortest and longest and shortest_runtime is not None and longest_runtime is not None:
            print(f"answer=shortest={run_label(shortest)} {shortest_runtime:.1f}s; longest={run_label(longest)} {longest_runtime:.1f}s")
        else:
            print("answer=no finished runs with runtime found in sampled shortest/longest candidates")
    elif args.op == "system-metrics":
        filters = {"state": args.state or "finished"}
        for run in get_runs(wb, path, filters=filters, order="-created_at", limit=args.limit, include_sweeps=include_sweeps):
            keys = sorted(str(k) for k in run.history(stream="system", samples=args.samples).columns if str(k).startswith("system."))
            if keys:
                print(f"system_metric_count={len(keys)}\nsystem_metric_keys=" + ", ".join(keys[:40]))
                print(f"answer=yes; run={run_label(run)}; system_metrics={', '.join(keys[:20])}")
                return
        print("answer=no system metrics found in sampled runs")
    elif args.op in {"tags", "tag-counts"}:
        tag_sample = [] if args.tag else sample_metadata_runs(wb, path, args.samples, include_sweeps=include_sweeps)
        tags = args.tag or sorted({t for r in tag_sample for t in (getattr(r, "tags", None) or [])})
        counts = {tag: count(wb, path, {"tags": {"$in": [tag]}}, include_sweeps=include_sweeps) for tag in tags}
        if tag_sample:
            print(f"tag_discovery_sampled_runs={len(tag_sample)}")
            print(f"tag_discovery_per_slice_limit={args.samples}")
            print("sample_caveat=tag discovery uses newest and oldest slices; absent tags are omitted, not undercounted")
        print("tags=" + ", ".join(counts))
        for tag, value in counts.items():
            print(f"tag_count.{tag}={value}")
        print("answer=" + ("; ".join(f"{tag}={value}" for tag, value in counts.items()) if counts else "No tags found in bounded sample."))
    elif args.op == "groups":
        sample, groups, counts, orders, per_slice = discover_group_counts(wb, path, args, sample_size=args.samples)
        print("groups=" + ", ".join(groups))
        print_group_counts(sample, groups, counts, orders, per_slice)
    elif args.op == "date-counts":
        start, end = month_bounds(args.month) if args.month else (args.start, args.end)
        if not start and not end:
            raise SystemExit("date-counts requires --month or --start/--end")
        label = args.label or args.month or "date_window"
        filters = created_at_filter(start, end)
        print(f"{label}_count={count(wb, path, filters, include_sweeps=include_sweeps)}")
        emit("filters", filters)
        print(f"answer={label} API date-filter count={count(wb, path, filters, include_sweeps=include_sweeps)}")
