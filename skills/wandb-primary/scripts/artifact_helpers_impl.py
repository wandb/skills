# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Artifact list/filter helpers.

Uses the Public API (``wandb.Api``) rather than ad-hoc GraphQL so the surface
tracks SDK backwards compatibility. GraphQL projected queries remain available
for one-off composition in ``ARTIFACT_OPS.md`` when helpers are insufficient.

    H=skills/wandb-primary/scripts/artifact_helpers.py
    python $H versions --entity E --project P [filters]
    python $H --help
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from wandb_run_ops_lib.common import api, resolve_path
from wandb_run_ops_lib.core import artifact_collection_rows, emit

_DEFAULT_LIMIT = 50
_DEFAULT_SCAN_LIMIT = 500
_DEFAULT_PER_PAGE = 50
_DEFAULT_COLLECTION_PER_PAGE = 1000
_GENERATED_ARTIFACT_TYPES = frozenset({"run_table", "code", "job"})
_ORDER_CHOICES = (
    "size",
    "-size",
    "created_at",
    "-created_at",
    "name",
    "-name",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_artifact_created_at(value: Any) -> datetime | None:
    """Parse artifact created_at into an aware UTC datetime when possible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def artifact_is_generated(artifact: Any) -> bool:
    """Return whether an artifact looks system-generated."""
    flag = getattr(artifact, "is_generated", None)
    if isinstance(flag, bool):
        return flag
    typ = str(getattr(artifact, "type", "") or "")
    return typ in _GENERATED_ARTIFACT_TYPES or typ.startswith("wandb-")


def artifact_tag_names(artifact: Any) -> list[str]:
    """Normalize artifact tags to a list of tag name strings."""
    tags = getattr(artifact, "tags", None) or []
    names: list[str] = []
    for tag in tags:
        if isinstance(tag, str):
            names.append(tag)
            continue
        name = getattr(tag, "name", None)
        if name is None and isinstance(tag, dict):
            name = tag.get("name")
        if name is not None:
            names.append(str(name))
    return names


def artifact_version_row(
    artifact: Any, *, artifact_type: str, collection_name: str
) -> dict[str, Any]:
    """Return a compact metadata row for one artifact version (no download)."""
    aliases = list(getattr(artifact, "aliases", None) or [])
    return {
        "type": artifact_type,
        "collection": collection_name,
        "name": getattr(artifact, "name", None),
        "version": getattr(artifact, "version", None),
        "size_bytes": getattr(artifact, "size", None),
        "created_at": getattr(artifact, "created_at", None),
        "aliases": aliases,
        "tags": artifact_tag_names(artifact),
        "digest": getattr(artifact, "digest", None),
        "file_count": getattr(artifact, "file_count", None),
        "is_generated": artifact_is_generated(artifact),
    }


def artifact_version_matches_filters(
    row: dict[str, Any], args: argparse.Namespace, *, now: datetime
) -> bool:
    """Return whether a version row passes CLI filters."""
    size = row.get("size_bytes")
    if args.min_size_bytes is not None and (
        not isinstance(size, (int, float)) or size < args.min_size_bytes
    ):
        return False
    if args.max_size_bytes is not None and (
        not isinstance(size, (int, float)) or size > args.max_size_bytes
    ):
        return False
    created = parse_artifact_created_at(row.get("created_at"))
    if args.older_than_days is not None and (
        created is None or created > now - timedelta(days=args.older_than_days)
    ):
        return False
    if args.newer_than_days is not None and (
        created is None or created < now - timedelta(days=args.newer_than_days)
    ):
        return False
    if args.alias and args.alias not in (row.get("aliases") or []):
        return False
    if args.tag and args.tag not in (row.get("tags") or []):
        return False
    if args.exclude_generated and row.get("is_generated"):
        return False
    if args.name_regex:
        haystack = f"{row.get('collection') or ''} {row.get('name') or ''}"
        if not re.search(args.name_regex, haystack):
            return False
    return True


def sort_artifact_version_rows(
    rows: list[dict[str, Any]], order: str
) -> list[dict[str, Any]]:
    """Sort version rows by a supported client-side order key.

    Missing sort values always sort last, including for descending orders.
    """
    descending = order.startswith("-")
    key_name = order[1:] if descending else order

    def value_key(row: dict[str, Any]) -> Any | None:
        if key_name == "size":
            value = row.get("size_bytes")
            return value if isinstance(value, (int, float)) else None
        if key_name == "created_at":
            parsed = parse_artifact_created_at(row.get("created_at"))
            return parsed.timestamp() if parsed is not None else None
        return str(row.get("name") or "").lower()

    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        (present if value_key(row) is not None else missing).append(row)
    present.sort(key=value_key, reverse=descending)
    return present + missing


def cmd_versions(args: argparse.Namespace) -> None:
    """List filtered artifact versions with size/age metadata (no downloads)."""
    path = resolve_path(args.entity, args.project)
    wb = api()
    type_filter = list(args.type or [])
    collection_names = {name for name in (args.collection or []) if name}
    collection_regex = (
        re.compile(args.collection_regex) if args.collection_regex else None
    )
    now = datetime.now(UTC)
    matched: list[dict[str, Any]] = []
    scanned = 0
    collection_errors: list[dict[str, str]] = []
    scan_limit = args.scan_limit
    per_page = args.per_page

    for type_row in artifact_collection_rows(
        wb, path, type_filter or None, args.collection_per_page
    ):
        if type_row.get("error"):
            collection_errors.append(
                {"type": type_row["type"], "error": type_row["error"]}
            )
            continue
        artifact_type = str(type_row["type"])
        for collection in type_row["collections"]:
            collection_name = str(getattr(collection, "name", "") or "")
            if collection_names and collection_name not in collection_names:
                continue
            if collection_regex and not collection_regex.search(collection_name):
                continue
            try:
                # artifacts() is lazy; page fetches happen during iteration.
                for artifact in collection.artifacts(per_page=per_page):
                    if scanned >= scan_limit:
                        break
                    scanned += 1
                    row = artifact_version_row(
                        artifact,
                        artifact_type=artifact_type,
                        collection_name=collection_name,
                    )
                    if artifact_version_matches_filters(row, args, now=now):
                        matched.append(row)
            except Exception as exc:
                collection_errors.append(
                    {
                        "type": artifact_type,
                        "collection": collection_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if scanned >= scan_limit:
                break
        if scanned >= scan_limit:
            break

    ordered = sort_artifact_version_rows(matched, args.order)
    emitted = ordered[: args.limit]
    emitted_size = sum(
        int(row["size_bytes"])
        for row in emitted
        if isinstance(row.get("size_bytes"), (int, float))
    )
    matched_size = sum(
        int(row["size_bytes"])
        for row in matched
        if isinstance(row.get("size_bytes"), (int, float))
    )
    project_exhaustive = scanned < scan_limit and not collection_errors

    print(f"path={path}")
    print(f"scanned_versions={scanned}")
    print(f"matched_versions={len(matched)}")
    print(f"emitted_versions={len(emitted)}")
    print(f"order={args.order}")
    print(f"matched_size_bytes={matched_size}")
    print(f"emitted_size_bytes={emitted_size}")
    emit(
        "sample_scope",
        {
            "scanned_versions": scanned,
            "scan_limit": scan_limit,
            "matched_versions": len(matched),
            "emitted_versions": len(emitted),
            "limit": args.limit,
            "project_exhaustive": project_exhaustive,
        },
    )
    if scanned >= scan_limit:
        print(
            f"limit_caveat=versions stopped after scanning {scan_limit} versions; "
            "raise --scan-limit only when a broader scan is worth the extra API time"
        )
    if len(matched) > args.limit:
        print(
            f"limit_caveat=emitting top {args.limit} of {len(matched)} matched versions "
            f"after --order={args.order}; raise --limit to print more rows"
        )
    if collection_errors:
        print(
            "limit_caveat=one or more artifact types/collections failed to load; "
            "totals cover only successfully scanned collections"
        )
        emit("artifact_collection_errors", collection_errors[:10])
    for row in emitted:
        emit("artifact_version_row", row)
    print(
        f"answer=matched_versions={len(matched)}; emitted={len(emitted)}; "
        f"emitted_size_bytes={emitted_size}; scanned={scanned}"
    )
    print(
        "interpretation_hint=Ground claims on emitted artifact_version_row lines and "
        "matched/emitted size totals. Treat absent versions as unscanned when "
        "limit_caveat or sample_scope.project_exhaustive=false is present."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Artifact list/filter helpers (Public API / wandb.Api).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    versions = subparsers.add_parser(
        "versions",
        help="List filtered artifact versions with size, created_at, aliases, and tags (no downloads).",
    )
    versions.add_argument("--entity", help="W&B entity (or set WANDB_ENTITY).")
    versions.add_argument("--project", help="W&B project (or set WANDB_PROJECT).")
    versions.add_argument(
        "--type",
        action="append",
        default=[],
        help="Artifact type to include; repeat as needed.",
    )
    versions.add_argument(
        "--collection",
        action="append",
        default=[],
        help="Exact collection name to include; repeat as needed.",
    )
    versions.add_argument(
        "--collection-regex",
        help="Keep collections whose names match this regex.",
    )
    versions.add_argument(
        "--name-regex",
        help="Keep versions whose collection or artifact name matches this regex.",
    )
    versions.add_argument("--alias", help="Keep versions that carry this alias.")
    versions.add_argument("--tag", help="Keep versions that carry this tag.")
    versions.add_argument(
        "--min-size-bytes",
        type=_non_negative_int,
        help="Minimum artifact size in bytes.",
    )
    versions.add_argument(
        "--max-size-bytes",
        type=_non_negative_int,
        help="Maximum artifact size in bytes.",
    )
    versions.add_argument(
        "--older-than-days",
        type=_positive_int,
        help="Keep versions created at least this many days ago.",
    )
    versions.add_argument(
        "--newer-than-days",
        type=_positive_int,
        help="Keep versions created within this many days.",
    )
    versions.add_argument(
        "--exclude-generated",
        action="store_true",
        help="Drop system-generated artifact types (run_table, code, job, wandb-*).",
    )
    versions.add_argument(
        "--order",
        choices=list(_ORDER_CHOICES),
        default="-created_at",
        help="Client-side order among matched versions in the scanned slice.",
    )
    versions.add_argument(
        "--limit",
        type=_positive_int,
        default=_DEFAULT_LIMIT,
        help="Maximum matched rows to emit after ordering.",
    )
    versions.add_argument(
        "--scan-limit",
        type=_positive_int,
        default=_DEFAULT_SCAN_LIMIT,
        help="Maximum versions to scan across collections before stopping.",
    )
    versions.add_argument(
        "--per-page",
        type=_positive_int,
        default=_DEFAULT_PER_PAGE,
        help="Versions requested per collection page.",
    )
    versions.add_argument(
        "--collection-per-page",
        type=_positive_int,
        default=_DEFAULT_COLLECTION_PER_PAGE,
        help="Collections requested per type page.",
    )
    versions.set_defaults(func=cmd_versions)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        emit("error", {"type": type(exc).__name__, "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
