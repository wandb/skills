# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for AI-assisted Artifact Collection Overview generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from wandb_run_ops_lib.common import api, execute_graphql

_DEFAULT_JSON_INDENT = 2

_COLLECTION_CONTEXT_QUERY = """\
query PublicCollectionOverviewContext($artifactCollectionID: ID!) {
  artifactCollection(id: $artifactCollectionID) {
    id
    name
    description
    defaultArtifactType {
      name
    }
    project {
      id
      name
      entityName
      entity {
        name
      }
    }
  }
  artifactCollectionOverviewVersionCandidates(
    input: {artifactCollectionID: $artifactCollectionID}
  ) {
    versions {
      membershipID
      versionIndex
      createdAt
      aliases
      tags
    }
  }
}"""

_METRIC_CANDIDATES_QUERY = """\
query PublicCollectionOverviewMetricCandidates(
  $artifactCollectionID: ID!
  $membershipIDs: [ID!]!
) {
  artifactCollectionOverviewMetricCandidates(
    input: {
      artifactCollectionID: $artifactCollectionID
      membershipIDs: $membershipIDs
    }
  ) {
    metrics {
      name
    }
  }
}"""

_CREATE_DRAFT_MUTATION = """\
mutation PublicCreateCollectionOverviewDraftFromAIPlan(
  $input: GenerateArtifactCollectionOverviewAIInput!
) {
  generateArtifactCollectionOverviewDraftFromAIPlan(input: $input) {
    draftViewID
    publishedViewID
  }
}"""


def fetch_collection_overview_context(
    wb: object,
    collection_id: str,
) -> dict[str, Any]:
    """Fetch bounded metadata and version candidates for one collection."""
    response = execute_graphql(
        wb,
        _COLLECTION_CONTEXT_QUERY,
        {"artifactCollectionID": collection_id},
    )
    if not isinstance(response, dict):
        raise RuntimeError("GraphQL response was not an object.")

    collection = response.get("artifactCollection")
    if not isinstance(collection, dict):
        raise RuntimeError(f"Artifact collection not found: {collection_id}")
    candidate_packet = response.get("artifactCollectionOverviewVersionCandidates") or {}
    if not isinstance(candidate_packet, dict):
        raise RuntimeError("Version candidate response was not an object.")
    version_candidates = candidate_packet.get("versions") or []
    if not isinstance(version_candidates, list):
        raise RuntimeError("Version candidates response was not a list.")
    return {
        "collection": collection,
        "versionCandidates": version_candidates,
    }


def fetch_collection_overview_metric_candidates(
    wb: object,
    collection_id: str,
    membership_ids: list[str],
) -> dict[str, Any]:
    """Fetch bounded metric-name candidates for selected collection versions."""
    if not membership_ids:
        raise ValueError("At least one --membership-id is required.")
    response = execute_graphql(
        wb,
        _METRIC_CANDIDATES_QUERY,
        {
            "artifactCollectionID": collection_id,
            "membershipIDs": membership_ids,
        },
    )
    if not isinstance(response, dict):
        raise RuntimeError("GraphQL response was not an object.")

    packet = response.get("artifactCollectionOverviewMetricCandidates") or {}
    if not isinstance(packet, dict):
        raise RuntimeError("Metric candidate response was not an object.")
    metrics = packet.get("metrics") or []
    if not isinstance(metrics, list):
        raise RuntimeError("Metric candidates response was not a list.")
    return {
        "metrics": metrics,
    }


def create_collection_overview_draft_from_plan(
    wb: object,
    collection_id: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Submit a validated-shape AI plan to W&B for draft creation."""
    if not isinstance(plan.get("blocks"), list):
        raise ValueError("Plan must be an object with a 'blocks' list.")
    response = execute_graphql(
        wb,
        _CREATE_DRAFT_MUTATION,
        {
            "input": {
                "artifactCollectionID": collection_id,
                "plan": plan,
            },
        },
    )
    if not isinstance(response, dict):
        raise RuntimeError("GraphQL response was not an object.")

    payload = response.get("generateArtifactCollectionOverviewDraftFromAIPlan")
    if not isinstance(payload, dict):
        raise RuntimeError("W&B did not return a draft payload.")
    return payload


def _load_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan_json = cast(str | None, getattr(args, "plan_json", None))
    plan_file = cast(str | None, getattr(args, "plan_file", None))
    if plan_json:
        plan = json.loads(plan_json)
    else:
        if plan_file is None:
            raise ValueError("A plan source is required.")
        plan = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")
    return cast(dict[str, Any], plan)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="W&B Collection Overview GraphQL helper.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    context = subparsers.add_parser(
        "context",
        help="Fetch collection metadata and bounded version candidates.",
    )
    context.add_argument("--collection-id", required=True)

    metrics = subparsers.add_parser(
        "metric-candidates",
        help="Fetch bounded metric candidates for selected membership IDs.",
    )
    metrics.add_argument("--collection-id", required=True)
    metrics.add_argument("--membership-id", action="append", required=True)

    create = subparsers.add_parser(
        "create-draft",
        help="Create a Collection Overview draft from an AI plan JSON object.",
    )
    create.add_argument("--collection-id", required=True)
    plan_source = create.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--plan-json")
    plan_source.add_argument("--plan-file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    wb = api()
    command = cast(str, args.command)
    collection_id = cast(str, getattr(args, "collection_id", ""))
    try:
        if command == "context":
            result = fetch_collection_overview_context(wb, collection_id)
        elif command == "metric-candidates":
            membership_ids = cast(list[str], args.membership_id)
            result = fetch_collection_overview_metric_candidates(
                wb,
                collection_id,
                membership_ids,
            )
        elif command == "create-draft":
            result = create_collection_overview_draft_from_plan(
                wb,
                collection_id,
                _load_plan(args),
            )
        else:
            raise ValueError(f"Unknown command: {command}")
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "command": command,
                },
                indent=_DEFAULT_JSON_INDENT,
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "result": result,
            },
            indent=_DEFAULT_JSON_INDENT,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
