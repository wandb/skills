# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for working with W&B (Weights & Biases) training data.

The wandb SDK is for the W&B Models product — training runs, metrics history,
hyperparameter sweeps, artifacts, and system metrics. These helpers convert
run data into pandas-friendly structures for analysis.

Usage (in sandbox):
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    from wandb_helpers import (
        runs_to_dataframe,   # Convert runs to a clean pandas DataFrame
        diagnose_run,        # Quick diagnostic summary of a training run
        compare_configs,     # Side-by-side config diff between two runs
    )
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Runs -> DataFrame
# ---------------------------------------------------------------------------

def runs_to_dataframe(
    runs: Any,
    limit: int = 200,
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert W&B runs to a list of flat dicts (ready for pd.DataFrame).

    Always slices runs to avoid loading entire projects into memory.

    Args:
        runs: W&B Runs object from api.runs().
        limit: Max runs to process (default 200).
        metric_keys: Summary metric keys to include. If None, includes
                     "loss", "val_loss", "accuracy".

    Returns:
        List of dicts with run metadata + config + selected metrics.
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
        # Config (skip internal keys)
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[f"config.{k}"] = v
        # Summary metrics
        for key in metric_keys:
            row[key] = run.summary_metrics.get(key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------------------------

def diagnose_run(run: Any) -> dict[str, Any]:
    """Quick diagnostic summary of a training run.

    Loads the full loss history and checks for convergence, overfitting,
    NaN values, and other common training issues.

    Args:
        run: A W&B Run object from api.run().

    Returns:
        Dict with diagnostic keys: total_steps, final_loss, min_loss,
        min_loss_step, has_nan, final_10pct_mean, train_val_gap,
        likely_overfit, converged.
    """
    import pandas as pd

    df = pd.DataFrame(list(run.scan_history(keys=["loss", "val_loss"])))
    loss = df["loss"].dropna()

    diagnostics: dict[str, Any] = {
        "total_steps": len(loss),
        "final_loss": loss.iloc[-1] if len(loss) else None,
        "min_loss": loss.min() if len(loss) else None,
        "min_loss_step": int(loss.idxmin()) if len(loss) else None,
        "has_nan": bool(loss.isna().any()),
        "final_10pct_mean": float(loss.tail(max(1, len(loss) // 10)).mean())
        if len(loss)
        else None,
    }

    # Overfitting check (val_loss diverging from train loss)
    if "val_loss" in df.columns:
        val = df["val_loss"].dropna()
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

def compare_configs(run_a: Any, run_b: Any) -> list[dict[str, Any]]:
    """Side-by-side config comparison between two W&B runs.

    Returns only the keys that differ between the two runs.

    Args:
        run_a: First W&B Run object.
        run_b: Second W&B Run object.

    Returns:
        List of dicts with: key, run_a_name, run_a_value, run_b_name, run_b_value
    """
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
