# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import tempfile
from calendar import monthrange
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wandb_run_vocab import (
    COMPREHENSIVE_METRIC_PATTERNS,
    COMPREHENSIVE_METRIC_TOKENS,
    HPARAM_KEY_TOKENS,
    HPARAM_PRIORITY_KEYS,
    METRIC_HINT_ALIASES,
    MINIMIZE_METRIC_TOKENS,
    PERFORMANCE_METRIC_TOKENS,
)

NAME_PATTERN_DEFS = {
    "structured_date_prefix": re.compile(r"^[A-Za-z][A-Za-z0-9_-]*-\d{4}-\d{2}-\d{2}"),
    "wandb_auto_name": re.compile(r"^[a-z]+-[a-z]+-\d+$"),
    "short_slug_or_manual": re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$"),
}
RECENT_AND_OLDEST_ORDERS = ("-created_at", "+created_at")
COHORT_NAME_SELECTOR_PREFIXES = ("name_regex:", "run_name:", "name:")
COHORT_SELECTOR_HELP = "Selectors: non_sweep, sweep:any, sweep:<id>, tag:<tag>, group:<group>, name_regex:<regex>, run_name:<regex>, name:<regex>."


def api(timeout: int = 120) -> Any:
    """Return the regular W&B SDK API client."""
    import wandb

    return wandb.Api(timeout=timeout)


def resolve_path(entity: str | None = None, project: str | None = None) -> str:
    """Resolve and validate ENTITY/PROJECT from CLI args or environment."""
    if entity or project:
        if not entity or not project:
            raise SystemExit("Pass both --entity ENTITY and --project PROJECT")
        resolved = f"{entity}/{project}"
        split_path(resolved)
        return resolved

    entity, project = os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise SystemExit("Pass --entity ENTITY --project PROJECT or set WANDB_ENTITY/WANDB_PROJECT")

    resolved = f"{entity}/{project}"
    split_path(resolved)
    return resolved


def split_path(path: str) -> tuple[str, str]:
    """Split an ENTITY/PROJECT path into entity and project components."""
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"project path must be ENTITY/PROJECT with non-empty entity and project, got {path!r}")
    return parts[0], parts[1]


def run_label(run: Any) -> str:
    """Return the most user-facing run label available."""
    return (
        getattr(run, "display_name", None)
        or getattr(run, "name", None)
        or getattr(run, "id", "<unknown>")
    )


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config copy without private W&B keys."""
    return {k: v for k, v in dict(config or {}).items() if not str(k).startswith("_")}


def jsonable(value: Any) -> str:
    """Convert a value to stable JSON for containers, otherwise to a string."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def scalar_config_value(value: Any) -> Any:
    """Keep scalar config values and stringify short sequences."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 4:
        return json.dumps(value, sort_keys=True, default=str)
    return None


def parse_config_eq_specs(specs: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE config equality specs."""
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected KEY=VALUE for --config-eq, got {spec!r}")
        key, value = spec.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def filters_from_args(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    """Build W&B filters from an argparse state value and non-empty extras."""
    filters = {"state": args.state} if getattr(args, "state", None) else {}
    filters.update({k: v for k, v in extra.items() if v not in (None, {}, [])})
    for key, value in parse_config_eq_specs(getattr(args, "config_eq", None) or []).items():
        filters[f"config.{key}"] = value
    return filters


def coerce_float(value: Any) -> float | None:
    """Screen mixed W&B values for numeric use; callers must report dropped points when that affects user-facing results."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return None if math.isnan(number) else number


def numeric_direction(metric: str | None, hint: str | None = None) -> str:
    """Infer whether a metric should be minimized or maximized."""
    search_text = f"{metric or ''} {hint or ''}".lower()
    return "min" if any(t in search_text for t in MINIMIZE_METRIC_TOKENS) else "max"


def metric_hint_matches(key: str, hint: str) -> bool:
    """Return whether a metric key matches a hint or known hint alias."""
    hint_terms = [
        hint,
        *METRIC_HINT_ALIASES.get(hint, []),
        *METRIC_HINT_ALIASES.get(hint.lower(), []),
    ]
    terms = dict.fromkeys(t.lower() for t in hint_terms if t)
    lower = key.lower()

    return any(
        (
            re.search(rf"(^|[^a-z0-9]){re.escape(t)}([^a-z0-9]|$)", lower)
            if len(t) <= 3
            else t in lower
        )
        for t in terms
    )


def metric_sort_key(
    key: str, *, prefer_comprehensive: bool = False
) -> tuple[int, int, int, str]:
    """Rank metric keys by comprehensive/canonical naming and brevity."""
    lower = key.lower()
    comprehensive = any(t in lower for t in COMPREHENSIVE_METRIC_TOKENS) or any(
        re.search(p, lower) for p in COMPREHENSIVE_METRIC_PATTERNS
    )
    canonical = any(t in lower for t in ("best", "val", "valid", "eval", "metrics/"))
    return (
        0 if prefer_comprehensive and comprehensive else 1,
        0 if canonical else 1,
        len(key),
        key,
    )


def choose_metric(
    keys: list[str], hint: str, *, prefer_comprehensive: bool = False
) -> str:
    """Choose the best metric key for a hint, preferring non-loss matches."""
    hint_lower = hint.lower()
    bad_metric_terms = ("loss", "bpb")
    groups = [
        [
            k
            for k in keys
            if k.lower().endswith(hint_lower)
            and not any(s in k.lower() for s in bad_metric_terms)
        ],
        [
            k
            for k in keys
            if metric_hint_matches(k, hint)
            and not any(s in k.lower() for s in bad_metric_terms)
        ],
        [k for k in keys if metric_hint_matches(k, hint)],
    ]

    for group in groups:
        if group:
            return sorted(
                group,
                key=lambda k: metric_sort_key(
                    k, prefer_comprehensive=prefer_comprehensive
                ),
            )[0]

    raise SystemExit(f"No summary metric matching hint {hint!r}")


def choose_metric_from_runs(runs: list[Any], hint: str) -> str:
    """Choose a metric from numeric summary keys found on sampled runs."""
    keys = sorted(
        {
            k
            for r in runs
            for k, v in (r.summary_metrics or {}).items()
            if metric_hint_matches(str(k), hint) and coerce_float(v) is not None
        }
    )
    return choose_metric(keys, hint, prefer_comprehensive=True)


def sample_metric_from_hint(wb: Any, path: str, hint: str, sample_size: int) -> str:
    """Sample finished runs and choose a summary metric matching a hint."""
    runs = get_runs(
        wb,
        path,
        filters={"state": "finished"},
        limit=sample_size,
        include_sweeps=True,
    )
    return choose_metric_from_runs(runs, hint)


def likely_hparam_keys(config: dict[str, Any], explicit: list[str]) -> list[str]:
    """Select likely hyperparameter keys from explicit, priority, and token matches."""
    keys: list[str] = []
    for key in [*explicit, *HPARAM_PRIORITY_KEYS, *sorted(config)]:
        is_candidate = (
            key in explicit
            or key in HPARAM_PRIORITY_KEYS
            or any(t in key.lower() for t in HPARAM_KEY_TOKENS)
        )
        if key in config and key not in keys and is_candidate:
            keys.append(key)
        if len(keys) >= 16:
            break

    return keys


def run_search_text(run: Any, *, config_keys: list[str] | None = None) -> str:
    """Build lower-case text used for matching a run by labels and config."""
    config = clean_config(run.config)
    config = {k: config.get(k) for k in config_keys} if config_keys else config
    pieces = [
        run_label(run),
        getattr(run, "name", ""),
        getattr(run, "id", ""),
        json.dumps(config, sort_keys=True, default=str),
    ]
    return " ".join(str(x) for x in pieces if x).lower()


def unwrap_config_value(value: Any) -> Any:
    """Unwrap W&B config wrappers that only contain value and optional desc."""
    if (
        isinstance(value, dict)
        and set(value).issubset({"value", "desc"})
        and "value" in value
    ):
        return value.get("value")
    return value


def nested_config_value(config: dict[str, Any], key: str) -> Any:
    """Read a config value by exact key or dot-delimited nested path."""
    if key in config:
        return unwrap_config_value(config[key])

    current: Any = config
    for part in key.split("."):
        current = unwrap_config_value(current)
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return unwrap_config_value(current)


def merge_filters(*items: dict[str, Any] | None) -> dict[str, Any]:
    """Merge filters directly, falling back to $and for collisions/operators."""
    filters = [item for item in items if item]
    if not filters:
        return {}

    merged: dict[str, Any] = {}
    and_items: list[dict[str, Any]] = []
    for item in filters:
        if any(k in merged for k in item) or any(k.startswith("$") for k in item):
            and_items.append(item)
        else:
            merged.update(item)
    if and_items:
        return {"$and": ([merged] if merged else []) + and_items}
    return merged


def run_table_raw_search_text(row: dict[str, Any]) -> str:
    """Build search text from run-table identity, tags, and config values."""
    pieces = [
        row.get("name"),
        row.get("display_name"),
        row.get("id"),
        row.get("group"),
        row.get("job_type"),
    ]
    pieces.extend(row.get("tags") or [])
    for key, value in sorted((row.get("config") or {}).items()):
        pieces.extend([key, value])

    return " ".join(str(x) for x in pieces if x not in (None, ""))


def run_table_search_text(row: dict[str, Any]) -> str:
    """Build lower-case search text from a run-table row."""
    return run_table_raw_search_text(row).lower()


def execute_graphql(wb: Any, query: str, variables: dict[str, Any]) -> Any:
    """Execute GraphQL through the wandb SDK service API.

    Prefer ``Api._service_api.execute_graphql`` when the installed W&B SDK
    exposes it. It accepts a raw query string, returns the decoded ``data``
    payload, and raises on GraphQL errors.
    """
    service_api = getattr(wb, "_service_api", None)
    execute = getattr(service_api, "execute_graphql", None)
    if callable(execute):
        return execute(query, variables)

    client = getattr(wb, "client", None)
    client_execute = getattr(client, "execute", None)
    if not callable(client_execute):
        raise RuntimeError(
            "The installed W&B SDK exposes neither "
            "Api._service_api.execute_graphql nor Api.client.execute"
        )
    try:
        from wandb_graphql.language import parser as gql_parser
    except ImportError as exc:
        raise RuntimeError(
            "Install a current wandb package with its GraphQL dependencies"
        ) from exc
    return client_execute(gql_parser.parse(query), variable_values=variables)


RUN_TABLE_QUERY = """\
query PublicRunTableRows($project: String!, $entity: String!, $cursor: String,
           $perPage: Int!, $order: String, $filters: JSONString,
           $configKeys: [String!], $includeConfig: Boolean!) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    displayName
                    state
                    createdAt
                    group
                    jobType
                    tags
                    summaryMetrics(keys: %SUMMARY_KEYS%)
                    config(keys: $configKeys) @include(if: $includeConfig)
                }
                cursor
            }
            pageInfo {
                endCursor
                hasNextPage
            }
        }
    }
}
"""


def fetch_run_table_rows(
    wb: Any,
    path: str,
    *,
    summary_keys: list[str] | None = None,
    config_keys: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    order: str | None = None,
    limit: int = 1000,
    per_page: int = 200,
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch selected run-table rows via GraphQL and report if more remain."""
    entity, project = split_path(path)
    summary_keys = summary_keys or []
    config_keys = config_keys or []
    query = RUN_TABLE_QUERY.replace("%SUMMARY_KEYS%", json.dumps(summary_keys))

    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    has_more = False

    while len(rows) < limit:
        variables: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "perPage": max(1, min(per_page, limit - len(rows))),
            "order": order,
            "filters": json.dumps(filters or {}),
            "configKeys": config_keys,
            "includeConfig": bool(config_keys),
        }
        if cursor:
            variables["cursor"] = cursor

        data = execute_graphql(wb, query, variables)
        runs_data = data.get("project", {}).get("runs", {})
        page_info = runs_data.get("pageInfo", {})

        for edge in runs_data.get("edges", []):
            node = edge.get("node", {})
            summary = json.loads(node.get("summaryMetrics") or "{}")
            config_raw = json.loads(node.get("config") or "{}") if config_keys else {}
            row = {
                "id": node.get("name") or node.get("id"),
                "graphql_id": node.get("id"),
                "name": node.get("displayName") or node.get("name") or node.get("id"),
                "display_name": node.get("displayName"),
                "state": node.get("state"),
                "created_at": node.get("createdAt"),
                "group": node.get("group"),
                "job_type": node.get("jobType"),
                "tags": node.get("tags") or [],
                "summary": {key: summary.get(key) for key in summary_keys},
                "config": {
                    key: nested_config_value(config_raw, key) for key in config_keys
                },
            }
            rows.append(row)
            if len(rows) >= limit:
                break

        has_more = bool(page_info.get("hasNextPage"))
        if not has_more or len(rows) >= limit:
            break

        cursor = page_info.get("endCursor")

    return rows, has_more


def token_regex(token: str) -> re.Pattern[str]:
    """Compile a lower-case regex that matches a whole token."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])")


def run_scale_terms(label: str) -> list[str]:
    """Return model-size or scale tokens embedded in a run label."""
    return sorted(
        set(re.findall(r"\b\d+(?:\.\d+)?\s?(?:[kmbt]|pct|%)\b", label.lower()))
    )


def common_prefix_len(left: str, right: str) -> int:
    """Count shared leading characters case-insensitively."""
    count = 0
    for l_char, r_char in zip(left.lower(), right.lower()):
        if l_char != r_char:
            break
        count += 1
    return count


def leading_name_token(name: str) -> str:
    """Extract the leading alphanumeric run-name token."""
    match = re.match(r"([A-Za-z][A-Za-z0-9]*)", name)
    return match.group(1) if match else ""


def nearby_prefix_candidates(
    names: list[str],
    requested_prefix: str,
    *,
    min_shared: int = 4,
) -> list[dict[str, Any]]:
    """Suggest observed prefixes close to a requested run-name prefix."""
    requested = requested_prefix.lower()
    counts: defaultdict[str, list[str]] = defaultdict(list)

    for name in names:
        token = leading_name_token(name)
        if token and common_prefix_len(requested, token) >= min(
            min_shared, len(requested), len(token)
        ):
            counts[token].append(name)

    return sorted(
        ({"prefix": p, "count": len(v), "examples": v[:5]} for p, v in counts.items()),
        key=lambda r: (-r["count"], r["prefix"]),
    )


def name_pattern_summary(names: list[str]) -> tuple[str, dict[str, list[str]]]:
    """Summarize observed run-name patterns and return per-pattern matches."""
    matches = {
        label: [n for n in names if rx.match(n)]
        for label, rx in NAME_PATTERN_DEFS.items()
    }

    if matches["structured_date_prefix"]:
        prefix = matches["structured_date_prefix"][0].split("-", 1)[0]
        label = f"programmatic {prefix}-YYYY-MM-DD structured timestamp naming pattern"
    elif len(matches["wandb_auto_name"]) >= max(2, len(names) // 2):
        label = "mostly W&B auto-generated adjective-noun-number names in the sampled slices"
    else:
        label = "mixed manual or short-slug run names in the sampled slices"

    return label, matches


def parse_variant_spec(spec: str) -> tuple[str, re.Pattern[str]]:
    """Parse a variant spec as either a token or label=regex pair."""
    if "=" not in spec:
        return spec, token_regex(spec)

    label, pattern = spec.split("=", 1)
    return label.strip(), re.compile(pattern, re.I)


def match_variants(
    run: Any,
    specs: list[tuple[str, re.Pattern[str]]],
    *,
    config_keys: list[str] | None = None,
) -> list[str]:
    """Return variant names whose regexes match a run's search text."""
    text = run_search_text(run, config_keys=config_keys)
    return [name for name, rx in specs if rx.search(text)]


def get_runs(
    wb: Any,
    path: str,
    *,
    filters: dict[str, Any] | None = None,
    order: str = "-created_at",
    limit: int = 5,
    include_sweeps: bool = False,
    lazy: bool = True,
) -> list[Any]:
    """Fetch at most limit runs from a W&B project."""
    kwargs = {"filters": filters, "order": order, "per_page": max(1, limit), "include_sweeps": include_sweeps, "lazy": lazy}
    try:
        return wb.runs(path, **kwargs)[:limit]
    except Exception as exc:
        raise SystemExit(
            f"W&B run query failed for {path}: {type(exc).__name__}: {exc}. "
            f"filters={filters or {}} order={order!r} limit={limit}"
        ) from exc


def sample_run_slices(
    wb: Any,
    path: str,
    *,
    limit: int,
    orders: tuple[str, ...] | list[str] = RECENT_AND_OLDEST_ORDERS,
    filters: dict[str, Any] | None = None,
    include_sweeps: bool = False,
    lazy: bool = True,
) -> list[Any]:
    """Fetch unique runs from bounded ordered slices of a large project."""
    rows: list[Any] = []
    seen: set[Any] = set()
    for order in orders:
        for run in get_runs(wb, path, filters=filters, order=order, limit=max(1, limit), include_sweeps=include_sweeps, lazy=lazy):
            key = getattr(run, "id", None) or id(run)
            if key not in seen:
                seen.add(key)
                rows.append(run)
    return rows


def run_row(
    run: Any, *, include_url: bool = False, path: str | None = None
) -> dict[str, Any]:
    """Return a compact metadata row for a W&B run."""
    row = {
        "name": run_label(run),
        "id": run.id,
        "state": getattr(run, "state", None),
        "created_at": getattr(run, "created_at", None),
        "group": getattr(run, "group", None),
        "tags": getattr(run, "tags", None),
    }
    user = getattr(run, "user", None)
    username = getattr(user, "username", None) or getattr(user, "name", None)

    if username:
        row["user"] = username
    if include_url and path:
        row["url"] = f"https://wandb.ai/{path}/runs/{run.id}"

    return row


def runtime_seconds(run: Any) -> float | None:
    """Return the first available runtime value for a run in seconds."""
    summary = run.summary_metrics or {}
    return next(
        (
            v
            for v in (
                coerce_float(summary.get("_runtime")),
                coerce_float(summary.get("_wandb.runtime")),
                coerce_float(getattr(run, "duration", None)),
            )
            if v is not None
        ),
        None,
    )


def config_subset(run: Any, keys: list[str] | None = None) -> dict[str, Any]:
    """Return requested cleaned config keys, or the full cleaned config sorted."""
    config = clean_config(run.config)
    return (
        {key: config.get(key) for key in keys} if keys else dict(sorted(config.items()))
    )


def month_bounds(month: str) -> tuple[str, str]:
    """Return inclusive start and exclusive next-month UTC bounds for YYYY-MM."""
    match = re.fullmatch(r"(\d{4})-(\d{2})", month)
    if not match:
        raise SystemExit("--month must use YYYY-MM")

    year, month_number = int(match.group(1)), int(match.group(2))
    monthrange(year, month_number)
    end_year, end_month = (
        (year + 1, 1) if month_number == 12 else (year, month_number + 1)
    )

    return (
        f"{year:04d}-{month_number:02d}-01T00:00:00Z",
        f"{end_year:04d}-{end_month:02d}-01T00:00:00Z",
    )


def created_at_filter(start: str | None, end: str | None) -> dict[str, Any]:
    """Build a W&B created_at range filter from optional date or timestamp bounds."""
    bounds = {
        op: (v if "T" in v else f"{v}T00:00:00Z")
        for op, v in (("$gte", start), ("$lt", end))
        if v
    }
    return {"created_at": bounds} if bounds else {}


def sample_metadata_runs(wb: Any, path: str, limit: int, *, include_sweeps: bool = False, lazy: bool = True) -> list[Any]:
    """Sample unique newest and oldest runs for metadata inspection."""
    return sample_run_slices(wb, path, limit=limit, include_sweeps=include_sweeps, lazy=lazy)
