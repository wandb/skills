# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for working with W&B (Weights & Biases) training data.

Optimized for projects of any size, including large projects (10K+ runs,
1K+ metrics per run).

Key features:
- fetch_runs: Direct GraphQL with summaryMetrics field selection (15-25x
  faster than SDK iteration on large projects)
- get_api: Uses timeout=60 to prevent timeouts on large projects
- probe_project: Discovers project scale and available metrics
- runs_to_dataframe: Selective config/metric access
- diagnose_run: Configurable metric keys, uses beta_scan_history (parquet)
  for large histories
- scan_history: Auto-selects beta_scan_history for runs with 10K+ steps
- All history methods require explicit keys to avoid 502s on runs with
  thousands of metrics

Usage (in sandbox):
    import sys
    sys.path.insert(0, "skills/agent/wbagent/scripts")
    from wandb_helpers import (
        get_api,             # Create API with large-project-safe timeout
        runs_to_dataframe,   # Convert runs to a clean pandas DataFrame
        diagnose_run,        # Quick diagnostic summary of a training run
        compare_configs,     # Side-by-side config diff between two runs
        scan_history,        # Smart history scan (beta_scan_history for large runs)
    )
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# API factory
# ---------------------------------------------------------------------------

def get_api(timeout: int = 60) -> Any:
    """Create a wandb.Api with a safe timeout for large projects.

    The default wandb timeout (19s) causes frequent timeouts on projects
    with 10K+ runs or runs with 1K+ metrics.
    """
    import wandb
    return wandb.Api(timeout=timeout)


# ---------------------------------------------------------------------------
# Project probe
# ---------------------------------------------------------------------------

def probe_project(api: Any, path: str, sample_size: int = 3) -> dict[str, Any]:
    """Discover project characteristics before running queries.

    Call this FIRST on an unfamiliar project. It returns the project scale,
    available metric keys, config shape, and whether runs have step history.
    This lets you choose the right query strategy upfront instead of hitting
    timeouts or 502s by accident.

    Args:
        api: wandb.Api instance (use get_api()).
        path: "entity/project" string.
        sample_size: Number of runs to sample for metric/config inspection.

    Returns:
        Dict with: run_count_estimate, sample_metrics, sample_config_keys,
        has_step_history, recommended_per_page, warnings.
    """
    result: dict[str, Any] = {"path": path, "warnings": []}

    # Fetch a small sample of runs — avoid triggering len() or large pages
    runs = api.runs(path, filters={"state": "finished"}, order="-created_at", per_page=sample_size)
    sample = runs[:sample_size]

    if not sample:
        result["run_count_estimate"] = 0
        result["warnings"].append("No finished runs found")
        return result

    # Inspect sampled runs
    all_metric_keys: set[str] = set()
    all_config_keys: set[str] = set()
    has_history = False

    for run in sample:
        metric_keys = {k for k in run.summary_metrics.keys() if not k.startswith("_")}
        config_keys = {k for k in run.config.keys() if not k.startswith("_")}
        all_metric_keys |= metric_keys
        all_config_keys |= config_keys
        if getattr(run, "lastHistoryStep", -1) >= 0:
            has_history = True

    n_metrics = len(all_metric_keys)
    result["sample_metric_count"] = n_metrics
    result["sample_metric_keys"] = sorted(all_metric_keys)[:50]
    result["sample_config_keys"] = sorted(all_config_keys)[:50]
    result["has_step_history"] = has_history

    # Scale warnings
    if n_metrics > 500:
        result["warnings"].append(
            f"Runs have {n_metrics} metrics — ALWAYS pass keys= to history/scan_history"
        )
    if n_metrics > 5000:
        result["warnings"].append(
            f"Runs have {n_metrics} metrics — history() without keys WILL 502"
        )

    # Recommend per_page based on metric density
    if n_metrics > 1000:
        result["recommended_per_page"] = 10
    elif n_metrics > 100:
        result["recommended_per_page"] = 50
    else:
        result["recommended_per_page"] = 100

    return result


# ---------------------------------------------------------------------------
# Smart history scan
# ---------------------------------------------------------------------------

def scan_history(
    run: Any,
    keys: list[str],
    max_rows: int | None = None,
    use_beta: bool | None = None,
) -> list[dict[str, Any]]:
    """Read history rows from a run, choosing the fastest available method.

    Uses beta_scan_history (parquet-backed) for runs with large step counts
    (10K+ steps) since it avoids GraphQL pagination. Falls back to
    scan_history for smaller runs where parquet download overhead isn't worth it.

    IMPORTANT: keys is required. Never call without explicit keys on large
    projects — runs with 1K+ metrics will 502 or timeout without key filtering.

    Args:
        run: A W&B Run object.
        keys: Metric keys to fetch. REQUIRED.
        max_rows: Stop after this many rows. None = all rows.
        use_beta: Force beta_scan_history (True), force regular (False),
                  or auto-detect (None, default).

    Returns:
        List of dicts with the requested keys + _step.
    """
    if not keys:
        raise ValueError("keys is required — never scan without explicit keys on large projects")

    # Auto-detect: use beta for runs with 10K+ steps
    if use_beta is None:
        total_steps = getattr(run, "lastHistoryStep", -1)
        use_beta = total_steps >= 10_000

    rows = []
    if use_beta and hasattr(run, "beta_scan_history"):
        scanner = run.beta_scan_history(keys=keys, page_size=min(max_rows or 10_000, 10_000))
    else:
        scanner = run.scan_history(keys=keys)

    for row in scanner:
        rows.append(dict(row))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# Fast run fetcher (direct GraphQL with field selection)
# ---------------------------------------------------------------------------

_RUNS_QUERY = """\
query Runs($project: String!, $entity: String!, $cursor: String,
           $perPage: Int!, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    state
                    createdAt
                    summaryMetrics(keys: %KEYS%)
                    config
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


def fetch_runs(
    api: Any,
    path: str,
    metric_keys: list[str],
    limit: int = 200,
    filters: dict[str, Any] | None = None,
    order: str = "-created_at",
    config_keys: list[str] | None = None,
    per_page: int = 50,
) -> list[dict[str, Any]]:
    """Fetch runs using direct GraphQL with summaryMetrics field selection.

    This is DRAMATICALLY faster than iterating run objects on large projects.
    The standard SDK fetches ALL summary metrics per run (771KB+ per run on
    projects with 20K+ metrics). This function uses the GraphQL
    summaryMetrics(keys: [...]) parameter to fetch ONLY the requested metrics,
    reducing payload from 771KB to ~50 bytes per run.

    Benchmarks (wandb/large_runs_demo, 72K runs, 44K metrics/run):
        Standard SDK:  ~600ms/run (12s for 20 runs)
        This function: ~34ms/run  (0.67s for 20 runs) — 17x faster

    Args:
        api: wandb.Api instance (use get_api()).
        path: "entity/project" string.
        metric_keys: Summary metric keys to fetch. REQUIRED.
        limit: Max runs to return.
        filters: W&B filter dict (e.g., {"state": "finished"}).
        order: Sort order (e.g., "-created_at", "+summary_metrics.loss").
        config_keys: Specific config keys to extract. None = skip config.
        per_page: Runs per GraphQL page (default 50).

    Returns:
        List of flat dicts with run metadata + selected metrics + selected config.
    """
    import json as _json

    import requests

    entity, project = path.split("/", 1)

    # Build the query with specific metric keys
    keys_json = _json.dumps(metric_keys)
    query = _RUNS_QUERY.replace("%KEYS%", keys_json)

    # If we don't need config, remove it from the query to save bandwidth
    if config_keys is None:
        query = query.replace("                    config\n", "")

    filter_str = _json.dumps(filters or {})

    rows: list[dict[str, Any]] = []
    cursor = None
    remaining = limit

    while remaining > 0:
        page_size = min(per_page, remaining)
        variables: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "perPage": page_size,
            "order": order,
            "filters": filter_str,
        }
        if cursor:
            variables["cursor"] = cursor

        resp = requests.post(
            "https://api.wandb.ai/graphql",
            headers={
                "Authorization": f"Bearer {api.api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables},
            timeout=getattr(api, "_timeout", 60),
        )
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        runs_data = data.get("data", {}).get("project", {}).get("runs", {})
        edges = runs_data.get("edges", [])
        page_info = runs_data.get("pageInfo", {})

        for edge in edges:
            node = edge["node"]
            summary = _json.loads(node.get("summaryMetrics") or "{}")

            row: dict[str, Any] = {
                "id": node["id"],
                "name": node["name"],
                "state": node["state"],
                "created_at": node["createdAt"],
            }

            # Config — selective
            if config_keys is not None:
                config = _json.loads(node.get("config") or "{}")
                for k in config_keys:
                    row[f"config.{k}"] = config.get(k, {}).get("value") if isinstance(config.get(k), dict) else config.get(k)

            # Summary metrics — already filtered server-side
            for key in metric_keys:
                row[key] = summary.get(key)

            rows.append(row)

        remaining -= len(edges)
        if not page_info.get("hasNextPage") or not edges:
            break
        cursor = page_info.get("endCursor")

    return rows[:limit]


# ---------------------------------------------------------------------------
# Runs -> DataFrame (legacy wrapper, uses fetch_runs when possible)
# ---------------------------------------------------------------------------

def runs_to_dataframe(
    runs: Any,
    limit: int = 200,
    metric_keys: list[str] | None = None,
    config_keys: list[str] | None = None,
    include_all_config: bool = False,
) -> list[dict[str, Any]]:
    """Convert W&B runs to a list of flat dicts (ready for pd.DataFrame).

    For best performance on large projects, use fetch_runs() directly instead.
    This function exists for backward compatibility with code that already has
    a runs object.

    Args:
        runs: W&B Runs object from api.runs().
        limit: Max runs to process (default 200).
        metric_keys: Summary metric keys to include. If None, includes
                     "loss", "val_loss", "accuracy".
        config_keys: Specific config keys to include. None = skip config.
        include_all_config: If True, include all non-internal config keys.
                           Ignored if config_keys is set. Can be slow on
                           runs with large configs.

    Returns:
        List of dicts with run metadata + selected config + selected metrics.
    """
    if metric_keys is None:
        metric_keys = ["loss", "val_loss", "accuracy"]

    rows = []
    for run in runs[:limit]:
        row = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
        }
        # Config — selective by default
        if config_keys is not None:
            for k in config_keys:
                row[f"config.{k}"] = run.config.get(k)
        elif include_all_config:
            for k, v in run.config.items():
                if not k.startswith("_"):
                    row[f"config.{k}"] = v
        # Summary metrics — only requested keys
        for key in metric_keys:
            row[key] = run.summary_metrics.get(key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------------------------

def diagnose_run(
    run: Any,
    train_key: str = "loss",
    val_key: str | None = "val_loss",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Quick diagnostic summary of a training run.

    Checks for convergence, overfitting, NaN values, and other common
    training issues. Uses beta_scan_history for runs with large step counts.

    Args:
        run: A W&B Run object from api.run().
        train_key: Primary training metric key (default "loss").
        val_key: Validation metric key (default "val_loss"). None to skip.
        max_steps: Limit rows read. None = all.

    Returns:
        Dict with diagnostic keys. Returns {"error": ...} if the
        requested keys don't exist.
    """
    import pandas as pd

    # Verify keys exist in summary before scanning history
    available_keys = set(run.summary_metrics.keys())
    if train_key not in available_keys:
        return {"error": f"Key '{train_key}' not in run summary. Available: {sorted(k for k in available_keys if not k.startswith('_'))[:20]}"}

    keys = [train_key]
    if val_key and val_key in available_keys:
        keys.append(val_key)
    elif val_key and val_key not in available_keys:
        val_key = None  # skip val check

    rows = scan_history(run, keys=keys, max_rows=max_steps)
    if not rows:
        return {"error": "No history rows found", "summary_value": run.summary_metrics.get(train_key)}

    df = pd.DataFrame(rows)
    if train_key not in df.columns:
        return {"error": f"Key '{train_key}' not in history columns: {list(df.columns)}"}

    loss = df[train_key].dropna()

    diagnostics: dict[str, Any] = {
        "total_steps": len(loss),
        "final_value": float(loss.iloc[-1]) if len(loss) else None,
        "min_value": float(loss.min()) if len(loss) else None,
        "min_value_step": int(loss.idxmin()) if len(loss) else None,
        "has_nan": bool(df[train_key].isna().any()),
        "final_10pct_mean": float(loss.tail(max(1, len(loss) // 10)).mean())
        if len(loss)
        else None,
    }

    # Overfitting check
    if val_key and val_key in df.columns:
        val = df[val_key].dropna()
        if len(val) > 10:
            tail_size = max(1, len(val) // 5)
            train_tail = float(loss.tail(tail_size).mean())
            val_tail = float(val.tail(tail_size).mean())
            diagnostics["train_val_gap"] = round(val_tail - train_tail, 6)
            diagnostics["likely_overfit"] = val_tail > train_tail * 1.2

    # Convergence check
    if len(loss) > 100:
        last_pct = loss.tail(max(1, len(loss) // 10))
        diagnostics["converged"] = bool(last_pct.std() < last_pct.mean() * 0.01)

    return diagnostics


# ---------------------------------------------------------------------------
# Config comparison
# ---------------------------------------------------------------------------

def compare_configs(
    run_a: Any,
    run_b: Any,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Side-by-side config comparison between two W&B runs.

    Args:
        run_a: First W&B Run object.
        run_b: Second W&B Run object.
        keys: Specific config keys to compare. None = all non-internal keys.

    Returns:
        List of dicts with differing keys and their values per run.
    """
    if keys is not None:
        config_a = {k: run_a.config.get(k) for k in keys}
        config_b = {k: run_b.config.get(k) for k in keys}
    else:
        config_a = {k: v for k, v in run_a.config.items() if not k.startswith("_")}
        config_b = {k: v for k, v in run_b.config.items() if not k.startswith("_")}

    all_keys = sorted(set(config_a) | set(config_b))
    diffs = []
    for k in all_keys:
        val_a = config_a.get(k)
        val_b = config_b.get(k)
        if val_a != val_b:
            diffs.append({
                "key": k,
                run_a.name: val_a,
                run_b.name: val_b,
            })
    return diffs
