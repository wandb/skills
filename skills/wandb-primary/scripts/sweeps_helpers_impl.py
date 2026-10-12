# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Sweep mutation helpers — wraps `upsertSweep` so the agent never
hand-writes the raw GraphQL mutation for a sweep create/update/clone/state change.
A bad upsertSweep mutation can easily have irreversible side-effects on the sweep.

See `SWEEPS.md` § Sweep Mutations for what each operation actually does and its
risk (STOP vs CANCEL are not the same thing — see `update_sweep_state` below).

Usage:
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    from sweeps_helpers import (
        SweepStateChange,   # PAUSE / RESUME / CANCEL / STOP
        update_sweep_state,
        create_sweep,
        copy_sweep,
        update_sweep,
        parameter_importance,
    )
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

import yaml
from wandb_run_ops_lib.common import execute_graphql

# Default cap on runs fetched as prior_runs when cloning a sweep — a bounded
# sample, not a promise every run is copied; raise run_limit for a small sweep.
_DEFAULT_RUN_LIMIT = 500

_SWEEP_REF_QUERY = """
query PublicSweepRef($entity: String, $project: String, $sweep: String!, $runLimit: Int!) {
    project(name: $project, entityName: $entity) {
        sweep(sweepName: $sweep) {
            id
            state
            config
            runs(first: $runLimit) { edges { node { name } } }
        }
    }
}
"""

_UPSERT_SWEEP_MUTATION = """
mutation PublicUpsertSweep(
    $id: ID,
    $config: String,
    $entityName: String,
    $projectName: String,
    $state: String,
    $priorRunsFilters: JSONString,
    $displayName: String,
) {
    upsertSweep(input: {
        id: $id,
        config: $config,
        entityName: $entityName,
        projectName: $projectName,
        state: $state,
        priorRunsFilters: $priorRunsFilters,
        displayName: $displayName,
    }) {
        sweep { id name state }
        inserted
        insertedPriorRuns
        configValidationWarnings
    }
}
"""

_PARAMETER_IMPORTANCE_QUERY = """
query PublicParameterImportance(
    $entity: String,
    $project: String!,
    $sweep: String,
    $target: String!,
    $parameters: [String!],
    $runParameters: [String!],
    $limit: Int,
    $filters: JSONString,
) {
    project(name: $project, entityName: $entity) {
        parameterImportance(
            target: $target,
            sweepName: $sweep,
            parameters: $parameters,
            runParameters: $runParameters,
            limit: $limit,
            filters: $filters,
        )
    }
}
"""


class SweepStateChange(StrEnum):
    """Target states for `update_sweep_state`.

    Values are the literal `state` strings `upsertSweep` expects (verified
    against `wandb.sdk.internal.internal_api.Api.set_sweep_state`) — not the
    display names shown in the UI.
    """

    PAUSE = "PAUSED"
    RESUME = "RUNNING"
    # STOP lets in-progress runs finish and stops assigning new ones.
    STOP = "FINISHED"
    # CANCEL kills all currently running runs immediately. The most
    # destructive option — never call without the user's express consent
    # (see SWEEPS.md § Sweep Mutations).
    CANCEL = "CANCELED"


def _load_config(config_yaml: str) -> dict[str, Any]:
    """Parse a sweep's stored YAML config string back into a dict."""
    return yaml.safe_load(config_yaml) or {}


def _sweep_ref(api: Any, path: str, *, run_limit: int) -> dict[str, Any]:
    """Resolve `path` (`entity/project/sweep_id`) to id, state, config, and runs."""
    entity, project, sweep_id = api._parse_path(path)
    response = execute_graphql(
        api,
        _SWEEP_REF_QUERY,
        {
            "entity": entity,
            "project": project,
            "sweep": sweep_id,
            "runLimit": run_limit,
        },
    )
    sweep = response["project"]["sweep"] if response["project"] else None
    if sweep is None:
        raise ValueError(f"sweep {path!r} not found")
    return {"entity": entity, "project": project, **sweep}


def _upsert(
    api: Any,
    *,
    entity: str,
    project: str,
    config: dict[str, Any] | None = None,
    obj_id: str | None = None,
    prior_runs: list[str] | None = None,
    state: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Shared `upsertSweep` call backing create/copy/update/state-change.

    `config` is a dict, YAML-encoded here for the wire; leave it `None` for a
    mutation that only changes state (the server leaves an omitted field
    unchanged rather than clearing it).
    """
    variables = {
        "id": obj_id,
        "config": yaml.dump(config) if config is not None else None,
        "entityName": entity,
        "projectName": project,
        "state": state,
        "priorRunsFilters": json.dumps({"$or": [{"name": r} for r in prior_runs]})
        if prior_runs
        else None,
        "displayName": display_name,
    }
    response = execute_graphql(api, _UPSERT_SWEEP_MUTATION, variables)
    return response["upsertSweep"]


def update_sweep_state(api: Any, path: str, change: SweepStateChange) -> dict[str, Any]:
    """Pause, resume, stop, or cancel a sweep — never a plain `upsertSweep` call.

    `change` is a `SweepStateChange`, not a raw string; STOP and CANCEL are
    different operations (see the enum's docstring) and both require the
    user's express consent before calling — this function does not itself
    ask for confirmation.
    """
    ref = _sweep_ref(api, path, run_limit=0)
    result = _upsert(
        api,
        entity=ref["entity"],
        project=ref["project"],
        obj_id=ref["id"],
        state=change.value,
    )
    return result["sweep"]


def create_sweep(
    api: Any,
    entity: str,
    project: str,
    config: dict[str, Any],
    *,
    prior_runs: list[str] | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Create a new sweep from a full config (method/metric/parameters/...).

    `prior_runs` is a list of existing run names/ids to seed as prior runs, so
    the search treats them as already-evaluated points. The training script is
    never provided here — the user runs `wandb agent` (or a launch-sweep
    scheduler, see SWEEPS.md § Sweeps on launch) against the returned sweep.
    """
    result = _upsert(
        api,
        entity=entity,
        project=project,
        config=config,
        prior_runs=prior_runs,
        display_name=display_name,
    )
    return result["sweep"]


def copy_sweep(
    api: Any, path: str, *, run_limit: int = _DEFAULT_RUN_LIMIT
) -> dict[str, Any]:
    """Clone a sweep: same config, plus its existing runs as the new sweep's prior runs.

    Prefer this over `update_sweep` for a sweep outside PENDING state —
    `upsertSweep` only allows changing an existing sweep's config while it is
    still PENDING (see SWEEPS.md § Sweep Mutations); cloning works regardless
    of the source sweep's state. `run_limit` bounds how many of the source
    sweep's runs are fetched as prior runs — raise it for a small sweep, or
    the clone silently omits later runs.
    """
    ref = _sweep_ref(api, path, run_limit=run_limit)
    prior_runs = [edge["node"]["name"] for edge in ref["runs"]["edges"]]
    result = _upsert(
        api,
        entity=ref["entity"],
        project=ref["project"],
        config=_load_config(ref["config"]),
        prior_runs=prior_runs,
    )
    return result["sweep"]


def update_sweep(
    api: Any,
    path: str,
    *,
    config: dict[str, Any] | None = None,
    prior_runs: list[str] | None = None,
) -> dict[str, Any]:
    """Change a PENDING sweep's config and/or prior runs in place.

    Raises if the sweep is not PENDING — `upsertSweep` silently no-ops that
    change server-side, so this fails loudly instead. Use `copy_sweep` for a
    RUNNING/PAUSED sweep.
    """
    ref = _sweep_ref(api, path, run_limit=0)
    if ref["state"].upper() != "PENDING":
        raise ValueError(
            f"sweep {path!r} is {ref['state']}, not PENDING — updating an "
            "already-started sweep's config/prior runs is not supported; "
            "use copy_sweep() to clone it instead"
        )
    result = _upsert(
        api,
        entity=ref["entity"],
        project=ref["project"],
        config=config,
        obj_id=ref["id"],
        prior_runs=prior_runs,
    )
    return result["sweep"]


def _flattened_config_key(name: str) -> str:
    """Map a sweep parameter name to the flattened run-config key the W&B server scores.

    Run configs wrap every value (`{"lr": {"value": ...}}`), but the backend
    looks parameters up in the *flattened* run config for parameter_importance.
    """
    return name if "." in name else f"{name}.value"


def parameter_importance(
    api: Any,
    path: str,
    target: str,
    *,
    parameters: list[str] | None = None,
    run_parameters: list[str] | None = None,
    limit: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Learned parameter importance and correlation against `target`, scoped to one sweep.

    Backs the UI's parameter importance panel — prefer this over recomputing
    correlations by hand. `target` is the summary metric to explain (e.g. the
    sweep's metric goal). `parameters` takes sweep parameter names (e.g. `lr`)
    and defaults to every top-level parameter in the sweep's own config.
    `run_parameters` restricts which run-level (non-sweep) config keys are
    scored; omit for none. Returns a dict keyed by parameter name, e.g.
    `{"learning_rate": {"importance": 0.62, "correlation": -0.41}}`.
    """
    entity, project, sweep_id = api._parse_path(path)
    if parameters is None:
        ref = _sweep_ref(api, path, run_limit=0)
        parameters = sorted(_load_config(ref["config"]).get("parameters", {}))
    flattened_to_name = {_flattened_config_key(p): p for p in parameters}
    response = execute_graphql(
        api,
        _PARAMETER_IMPORTANCE_QUERY,
        {
            "entity": entity,
            "project": project,
            "sweep": sweep_id,
            "target": target,
            "parameters": list(flattened_to_name),
            "runParameters": run_parameters,
            "limit": limit,
            "filters": json.dumps(filters) if filters else None,
        },
    )
    scores = response["project"]["parameterImportance"]
    return {flattened_to_name.get(key, key): value for key, value in scores.items()}
