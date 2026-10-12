# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Safe raw-spec writes for existing W&B Reports.

Use this module only when ``wandb_workspaces.reports.v2.Report`` cannot safely
express the requested edit. The callback receives the exact server spec parsed as a
plain dict. Each write is field-minimal, backed up before mutation, and verified from
a fresh server read.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict
from urllib.parse import urlparse

from wandb_run_ops_lib.common import api, execute_graphql

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REPORT_BACKUP_DIR = Path("report_backups")

_REPORT_VIEW_QUERY = """\
query PublicReportView($reportId: ID!) {
  view(id: $reportId) {
    id
    name
    displayName
    description
    type
    createdAt
    updatedAt
    project { id name entityName }
    spec
  }
}
"""

_UPSERT_REPORT_MUTATION = """\
mutation PublicUpsertReport2(
  $id: ID!
  $entityName: String!
  $projectName: String!
  $spec: String!
) {
  upsertView(input: {
    id: $id
    entityName: $entityName
    projectName: $projectName
    spec: $spec
  }) {
    view { id name displayName updatedAt }
    inserted
  }
}
"""


class RawReportEditResult[MutateResult](TypedDict):
    """Outcome of a safe raw report-spec edit."""

    url: str
    verified: bool
    backup_path: str
    mutate_result: MutateResult
    upsert: dict[str, Any]
    error: str | None
    answer: str


class _GraphQLService(Protocol):
    def execute_graphql(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]: ...


class _ApiClient(Protocol):
    _service_api: _GraphQLService


def _report_id_from_url(url: str) -> str:
    """Return the padded base64 View ID embedded in a report URL."""
    path_parts = urlparse(url).path.strip("/").split("/")
    if len(path_parts) != 4 or path_parts[2] != "reports":
        raise ValueError("Expected a W&B report URL")

    slug = path_parts[3]
    separator_position = slug.rfind("--")
    if separator_position == -1:
        raise ValueError("Report URL does not contain an encoded View ID")

    encoded_id = slug[separator_position + 2 :]
    padded_id = encoded_id + ("=" * ((4 - len(encoded_id) % 4) % 4))
    try:
        decoded_id = base64.b64decode(padded_id, validate=True)
    except ValueError as exc:
        raise ValueError("Report URL contains an invalid View ID") from exc
    return base64.b64encode(decoded_id).decode("utf-8")


def _fetch_report_view(api_client: _ApiClient, url: str) -> dict[str, Any]:
    response = execute_graphql(
        api_client,
        _REPORT_VIEW_QUERY,
        {"reportId": _report_id_from_url(url)},
    )
    view = response.get("view") if isinstance(response, dict) else None
    if not isinstance(view, dict) or not isinstance(view.get("spec"), str):
        raise TypeError("GraphQL did not return a report View with a raw spec string")
    return view


def _write_backup(path: Path, url: str, view: dict[str, Any]) -> None:
    path.write_text(json.dumps({"url": url, "view": view}, indent=2), encoding="utf-8")


def _new_backup_path(backup_dir: str | Path) -> Path:
    destination = Path(backup_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return destination / f"report_before_edit_{timestamp}.json"


def backup_report_view(
    url: str,
    *,
    backup_dir: str | Path = DEFAULT_REPORT_BACKUP_DIR,
) -> Path:
    """Save the untouched server View response and return its local path."""
    view = _fetch_report_view(api(), url)
    path = _new_backup_path(backup_dir)
    _write_backup(path, url, view)
    return path


def _upsert_spec(
    api_client: _ApiClient,
    view: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    project = view.get("project")
    if not isinstance(project, dict):
        raise TypeError("Report View did not include its project")

    response = execute_graphql(
        api_client,
        _UPSERT_REPORT_MUTATION,
        {
            "id": view["id"],
            "entityName": project["entityName"],
            "projectName": project["name"],
            "spec": json.dumps(spec, separators=(",", ":")),
        },
    )
    upsert = response.get("upsertView") if isinstance(response, dict) else None
    if not isinstance(upsert, dict):
        raise TypeError("GraphQL did not return an upsertView payload")
    return upsert


def mutate_report_spec_verified[MutateResult](
    url: str,
    mutate: Callable[[dict[str, Any]], MutateResult],
    *,
    verify_readback: Callable[[dict[str, Any]], bool],
    backup_dir: str | Path = DEFAULT_REPORT_BACKUP_DIR,
) -> RawReportEditResult[MutateResult]:
    """Apply one raw-spec mutation and verify it from a fresh read."""
    api_client = api()
    backup_path = _new_backup_path(backup_dir)
    view = _fetch_report_view(api_client, url)
    _write_backup(backup_path, url, view)
    spec = json.loads(view["spec"])
    mutate_result = mutate(spec)
    upsert = _upsert_spec(api_client, view, spec)

    readback = _fetch_report_view(api_client, url)
    readback_spec = json.loads(readback["spec"])
    verified = bool(verify_readback(readback_spec))
    detail = "" if verified else " read-back verification failed"
    return RawReportEditResult[MutateResult](
        url=url,
        verified=verified,
        backup_path=str(backup_path),
        mutate_result=mutate_result,
        upsert=upsert,
        error=None if verified else "verification failed",
        answer=(
            f"answer=raw report edit saved verified={verified} "
            f"url={url} backup={backup_path}{detail}"
        ),
    )
