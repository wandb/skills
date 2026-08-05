# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Guarded W&B workspace-view writes: field-minimal `UpsertView` with optimistic
concurrency (OCC) on `View.updatedAt` and bounded fetch/re-apply/retry.

Every helper path that persists a workspace-view spec change — the `workspace_ops.py`
CLI, hand-written raw-GraphQL scripts, and the `wandb_workspaces` SDK — should route
its write through `mutate_view_with_retry` so a concurrent user/tab edit is layered
onto rather than clobbered. The unit of work is a `mutate(spec)` callback that edits
the freshly-read spec dict in place; the helper re-reads and re-applies it on each
`VIEW_SPEC_CONFLICT` before giving up.

The write is field-minimal by default (`id` + `spec` + routing + `lastUpdatedAt`): the
resolver applies each non-nil field onto a freshly-read row, so omitting
`displayName`/`description` means a concurrent rename is never clobbered and the view
description is never blanked. `name` is never sent — it is the immutable internal
identifier the URL resolves against. A caller that genuinely intends to rename the view
or set its description passes `display_name`/`description` explicitly; those are the
only metadata columns a guarded write will ever touch, and only on request.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import parse_qs, urlparse

# Reuse the GraphQL execution primitive already vetted in wandb_run_ops_lib so we go
# through the same client surface (and the same auth path).
from wandb_run_ops_lib.common import execute_graphql

if TYPE_CHECKING:
    from collections.abc import Callable

_MutateResult = TypeVar("_MutateResult")

MAX_UPSERT_ATTEMPTS = 3
# Code and message are set in gorilla's `UpsertView` resolver
VIEW_SPEC_CONFLICT_CODE = "VIEW_SPEC_CONFLICT"
VIEW_SPEC_CONFLICT_MESSAGE = "workspace changed since you loaded it"

# Below the usual timeout=120 on purpose: view queries take seconds, and failing under
# the shell budget gives a readable exception instead of a silent shell kill.
API_TIMEOUT_SECONDS = 60

LIST_VIEWS_QUERY = """query PublicWorkspaceViewsOps($entityName: String!, $projectName: String!, $first: Int!) {
  project(name: $projectName, entityName: $entityName) {
    allViews(viewType: "project-view", first: $first) {
      edges { node { id name displayName spec updatedAt } }
    }
  }
}"""

# Narrow single-view query. On projects with many large views, the unfiltered
# LIST_VIEWS_QUERY fetches every view's full spec and can burn a whole shell budget;
# this one returns in seconds.
VIEW_BY_NAME_QUERY = """query PublicWorkspaceViewByName($entityName: String!, $projectName: String!, $viewName: String!) {
  project(name: $projectName, entityName: $entityName) {
    allViews(viewType: "project-view", viewName: $viewName) {
      edges { node { id name displayName spec updatedAt } }
    }
  }
}"""

# `displayName` / `description` are optional and nullable: the caller sends them
# only when it intends a metadata edit. Omitting a variable makes the resolver see a
# nil field and leave that column untouched, so a spec-only write never clobbers a
# concurrent rename/description change. `name` is deliberately absent — it is the
# immutable internal identifier (`nw-<token>-v`) the URL resolves against and must
# never be rewritten by the helper.
UPSERT_VIEW_MUTATION = """mutation PublicUpsertView2($id: ID!, $entityName: String!, $projectName: String!, $spec: String!, $lastUpdatedAt: DateTime!, $displayName: String, $description: String) {
  upsertView(input: {id: $id, entityName: $entityName, projectName: $projectName, spec: $spec, lastUpdatedAt: $lastUpdatedAt, displayName: $displayName, description: $description}) {
    view { id name displayName }
    inserted
  }
}"""


# ---------- path / token / URL resolution ----------------------------------


def resolve_path(path_arg: str | None, view_url: str | None) -> tuple[str, str]:
    if view_url:
        parts = [p for p in urlparse(view_url).path.split("/") if p]
        if len(parts) >= 2:
            return parts[0], parts[1]
    if path_arg:
        entity, _, project = path_arg.partition("/")
    else:
        entity = os.environ.get("WANDB_ENTITY", "")
        project = os.environ.get("WANDB_PROJECT", "")
    if not entity or not project:
        raise SystemExit("Pass --path ENTITY/PROJECT, set WANDB_ENTITY/WANDB_PROJECT, or use a full --view-url.")
    return entity, project


def token_from_url(view_url: str) -> str:
    token = (parse_qs(urlparse(view_url).query).get("nw") or [""])[0]
    if not token:
        raise SystemExit("--view-url must contain an `nw=<token>` query param.")
    return token


def token_to_internal_name(token: str) -> str:
    """Convert the URL `nw` token to the backend `name` field. User workspaces are `nw-<token>-w`; saved views are `nw-<token>-v`."""
    if token.startswith("nw-"):
        return token
    suffix = "w" if token.startswith("nwuser") else "v"
    return f"nw-{token}-{suffix}"


# ---------- view lookup ----------------------------------------------------


def list_views(api: Any, entity: str, project: str, *, limit: int = 100) -> list[dict[str, Any]]:
    resp = execute_graphql(api, LIST_VIEWS_QUERY, {"entityName": entity, "projectName": project, "first": limit})
    edges = (((resp or {}).get("project") or {}).get("allViews") or {}).get("edges") or []
    return [edge["node"] for edge in edges if edge.get("node")]


def _find_view_node(nodes: list[dict[str, Any]], *, token: str | None, display_contains: str | None) -> dict[str, Any]:
    if token:
        wanted_name = token_to_internal_name(token)
        for node in nodes:
            if node.get("name") == wanted_name:
                return node
        for node in nodes:
            if token in (node.get("name") or "") or token in (node.get("displayName") or ""):
                return node
        raise SystemExit(f"No saved view matched token {token!r}.")
    if display_contains:
        needle = display_contains.lower()
        matches = [n for n in nodes if needle in (n.get("displayName") or "").lower()]
        if not matches:
            raise SystemExit(f"No view displayName contains {display_contains!r}.")
        if len(matches) > 1:
            names = ", ".join(f"{n['displayName']!r} (id={n['id']})" for n in matches)
            raise SystemExit(f"Ambiguous: multiple views match {display_contains!r}: {names}.")
        return matches[0]
    raise SystemExit("Pass either --view-url, --view-token, or --display-contains.")


def fetch_view_by_name(api: Any, entity: str, project: str, internal_name: str) -> dict[str, Any] | None:
    """Fetch a single view by internal name. Prefer this over list_views: it stays fast on big projects."""
    resp = execute_graphql(api, VIEW_BY_NAME_QUERY, {"entityName": entity, "projectName": project, "viewName": internal_name})
    edges = (((resp or {}).get("project") or {}).get("allViews") or {}).get("edges") or []
    nodes = [edge["node"] for edge in edges if edge.get("node")]
    return nodes[0] if nodes else None


def fetch_view(api: Any, entity: str, project: str, *, view_url: str | None, view_token: str | None, display_contains: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (node, parsed_spec_dict)."""
    token = view_token or (token_from_url(view_url) if view_url else None)
    if token:
        # Narrow path first — avoids pulling every view's spec on big projects.
        node = fetch_view_by_name(api, entity, project, token_to_internal_name(token))
        if node is not None:
            return node, json.loads(node["spec"])
    nodes = list_views(api, entity, project)
    node = _find_view_node(nodes, token=token, display_contains=display_contains)
    spec = json.loads(node["spec"])
    return node, spec


# ---------- spec navigation ------------------------------------------------


def sections_of(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return spec.get("section", {}).get("panelBankConfig", {}).get("sections", [])


def section_by_name(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    needle = name.lower()
    for section in sections_of(spec):
        if (section.get("name") or "").lower() == needle:
            return section
    return None


def public_url(entity: str, project: str, internal_name: str) -> str:
    token = re.sub(r"^nw-|-[vw]$", "", internal_name)
    return f"https://wandb.ai/{entity}/{project}?nw={token}"


# ---------- guarded upsert + retry -----------------------------------------


def guarded_upsert(
    api: Any,
    node: dict[str, Any],
    spec: dict[str, Any],
    *,
    entity: str,
    project: str,
    display_name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Field-minimal `UpsertView` guarded on `node["updatedAt"]` (OCC compare-and-swap).

    Writes only the spec by default. `display_name` / `description` are sent solely
    when the caller intends that metadata edit; leaving them None omits the variable so
    the server keeps its current value (never clobbering a concurrent user rename).
    """
    last_updated_at = node.get("updatedAt")
    if not isinstance(last_updated_at, str) or not last_updated_at:
        raise SystemExit("The workspace view did not include an updatedAt timestamp; refusing an unguarded write.")
    variables: dict[str, Any] = {
        "id": node["id"],
        "entityName": entity,
        "projectName": project,
        "spec": json.dumps(spec, separators=(",", ":")),
        "lastUpdatedAt": last_updated_at,
    }
    if display_name is not None:
        variables["displayName"] = display_name
    if description is not None:
        variables["description"] = description
    resp = execute_graphql(api, UPSERT_VIEW_MUTATION, variables)
    return resp["upsertView"]


def is_view_spec_conflict(error: Exception) -> bool:
    response = getattr(error, "response", None)
    response_message = getattr(response, "message", "")
    error_text = f"{error} {response_message}"
    return (
        VIEW_SPEC_CONFLICT_CODE in error_text
        or VIEW_SPEC_CONFLICT_MESSAGE in error_text.lower()
    )


def mutate_view_with_retry(
    api: Any,
    *,
    entity: str,
    project: str,
    view_url: str | None,
    view_token: str | None,
    display_contains: str | None,
    mutate: Callable[[dict[str, Any]], _MutateResult],
    display_name: str | None = None,
    description: str | None = None,
) -> tuple[dict[str, Any], _MutateResult]:
    """Fetch (fresh spec + `updatedAt`) -> `mutate(spec)` -> guarded upsert; on
    `VIEW_SPEC_CONFLICT` re-read the latest spec and re-apply `mutate` on top of it,
    up to `MAX_UPSERT_ATTEMPTS`. Returns the `upsertView` payload and whatever
    `mutate` returned on the winning attempt. Never re-upserts a stale in-memory spec.

    Pass `display_name` / `description` only to intentionally rename the view or set its
    description; they are re-applied on every retry alongside the re-run `mutate`. Left
    None, the write touches only the spec.
    """
    for attempt in range(MAX_UPSERT_ATTEMPTS):
        node, spec = fetch_view(
            api,
            entity,
            project,
            view_url=view_url,
            view_token=view_token,
            display_contains=display_contains,
        )
        mutate_result = mutate(spec)
        try:
            return (
                guarded_upsert(
                    api,
                    node,
                    spec,
                    entity=entity,
                    project=project,
                    display_name=display_name,
                    description=description,
                ),
                mutate_result,
            )
        except Exception as error:
            if not is_view_spec_conflict(error):
                raise
            if attempt == MAX_UPSERT_ATTEMPTS - 1:
                raise SystemExit(
                    "The workspace kept changing while this edit was being applied. "
                    "No update was made; inspect the latest workspace and try again."
                ) from error

    raise AssertionError("unreachable")


def read_back_spec(api: Any, entity: str, project: str, internal_name: str) -> dict[str, Any]:
    """Fresh read-back after a mutation. The mutation response alone does not prove the spec landed."""
    node = fetch_view_by_name(api, entity, project, internal_name)
    if node is None:
        raise SystemExit(f"Verification failed: view {internal_name!r} not found on fresh read-back.")
    return json.loads(node["spec"])


# ---------- Workspaces SDK path --------------------------------------------


def workspace_spec(workspace: Any) -> dict[str, Any]:
    """Serialize a `wandb_workspaces` `Workspace` to its raw view spec — the exact bytes
    `Workspace.save()` would persist (`spec.model_dump_json(by_alias=True, exclude_none=True)`).
    """
    return json.loads(workspace._to_model().spec.model_dump_json(by_alias=True, exclude_none=True))


def save_workspace_edit_with_retry(
    api: Any,
    *,
    url: str,
    edit: Callable[[Any], Any],
    display_name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Guarded equivalent of `w = Workspace.from_url(url); edit(w); w.save()` for saved
    views. Loads the view via the SDK, applies `edit` (which mutates the `Workspace` in
    place using SDK objects), serializes its spec, and guarded-upserts on `updatedAt`
    with bounded retry — so an SDK edit layers onto, rather than clobbers, a concurrent
    user/tab change. Returns the `upsertView` payload.

    Use only for saved views. Personal `nwuser` views are rejected here (the SDK's
    `from_url` rejects them too); edit their raw spec via `mutate_view_with_retry`.

    Like `Workspace.save()`, this round-trips through the SDK object model, so any spec
    field the SDK does not model is not preserved — a property of editing via the SDK,
    not of the guard. Reach for `mutate_view_with_retry` when you must preserve fields
    the SDK cannot express.
    """
    # Deferred, optional SDK import (matches report_helpers.py); keeps this module
    # importable where only the raw-spec path is exercised.
    import wandb_workspaces.workspaces as ws

    entity, project = resolve_path(None, url)
    internal_name = token_to_internal_name(token_from_url(url))
    if internal_name.endswith("-w"):
        raise SystemExit(
            "save_workspace_edit_with_retry supports saved views only. For a personal "
            "(nwuser) view, edit the raw spec through mutate_view_with_retry instead."
        )

    def mutate(spec: dict[str, Any]) -> Any:
        # `mutate_view_with_retry` read `updatedAt` (the OCC guard) just before this;
        # loading the SDK workspace now means any write landing in between forces a
        # conflict + retry rather than a silent clobber. Re-runs fresh on every retry.
        workspace = ws.Workspace.from_url(url)
        edit(workspace)
        spec.clear()
        spec.update(workspace_spec(workspace))
        return workspace

    result, _ = mutate_view_with_retry(
        api,
        entity=entity,
        project=project,
        view_url=url,
        view_token=None,
        display_contains=None,
        mutate=mutate,
        display_name=display_name,
        description=description,
    )
    return result
