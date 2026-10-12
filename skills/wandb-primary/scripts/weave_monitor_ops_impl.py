# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""List and safely change Weave Monitor object activation.

This helper reads and writes serialized Monitor payloads through the trace
server. It never objectifies a Monitor or expands scorer objects, so legacy
payloads and large scorer prompts stay out of the common path.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote

logging.getLogger("weave").setLevel(logging.ERROR)

import weave
from weave.trace.refs import Ref
try:
    from weave.shared.trace_server_interface import (
        ObjCreateReq,
        ObjQueryReq,
        ObjSchema,
        ObjSchemaForInsert,
        ObjectVersionFilter,
        SortBy,
    )
except ImportError:  # weave < 0.53
    from weave.trace_server.trace_server_interface import (
        ObjCreateReq,
        ObjQueryReq,
        ObjSchema,
        ObjSchemaForInsert,
        ObjectVersionFilter,
        SortBy,
    )

_PAGE_SIZE = 100
_MONITOR_BASE_CLASS = "Monitor"
_MONITOR_LEAF_CLASS = "Monitor"
_AGENT_TURN_ENDED_OP = "weave.genai.turn_ended"
_ACTIVE_FIELD = "active"
_OBJECT_REF_NAME = re.compile(r"/object/([^/:]+):")

ClientFactory = Callable[[str], Any]
Emit = Callable[[str], None]


@dataclass(frozen=True)
class MutationResult:
    changed: int = 0
    unchanged: int = 0
    verified: int = 0


class MonitorOperationError(RuntimeError):
    """A safe-operation failure with partial progress for CLI output."""

    def __init__(self, message: str, result: MutationResult | None = None) -> None:
        super().__init__(message)
        self.result = result or MutationResult()


class ConcurrentUpdateError(MonitorOperationError):
    """The latest digest changed between planning and writing."""


class VerificationError(MonitorOperationError):
    """A fresh server read did not confirm the requested state."""


@dataclass(frozen=True)
class MonitorSummary:
    object_id: str
    display_name: str
    active: bool
    classification: str
    sampling_rate: float | None
    op_names: list[str]
    scorers: list[dict[str, str | None]]
    version: str
    digest: str


@dataclass(frozen=True)
class MutationTarget:
    obj: ObjSchema
    active: bool
    is_signal: bool


def _validate_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("--path must be exactly ENTITY/PROJECT")
    return path


def _client(path: str, client_factory: ClientFactory) -> Any:
    requested = _validate_path(path)
    client = client_factory(requested)
    if client.project_id != requested:
        raise ValueError(
            f"client project {client.project_id!r} does not match --path {requested!r}"
        )
    return client


def _serialized_val(obj: ObjSchema) -> dict[str, Any]:
    if not isinstance(obj.val, dict):
        raise MonitorOperationError(
            f"Monitor {obj.object_id!r} has a non-object serialized payload"
        )
    return obj.val


def _uri(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Ref):
        return value.uri()
    return None


def _copy_serialized(value: Any) -> Any:
    """Copy a raw value while restoring deserialized refs to their wire URI."""
    if isinstance(value, Ref):
        return value.uri()
    if isinstance(value, dict):
        return {key: _copy_serialized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_serialized(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_serialized(item) for item in value]
    return copy.deepcopy(value)


def _op_names(val: dict[str, Any]) -> list[str]:
    values = val.get("op_names")
    if values is None:
        return []
    if not isinstance(values, list):
        return []
    return [uri for value in values if (uri := _uri(value)) is not None]


def _scorer_name(ref: str | None) -> str | None:
    if ref is None:
        return None
    match = _OBJECT_REF_NAME.search(ref)
    return unquote(match.group(1)) if match else None


def _scorers(
    val: dict[str, Any], *, include_scorer_names: bool
) -> list[dict[str, str | None]]:
    values = val.get("scorers")
    if not isinstance(values, list):
        return []
    result: list[dict[str, str | None]] = []
    for value in values:
        ref = _uri(value)
        name = None
        if isinstance(value, dict):
            raw_name = value.get("name")
            name = raw_name if isinstance(raw_name, str) else None
            ref = _uri(value.get("ref"))
        item: dict[str, str | None] = {"ref": ref}
        if include_scorer_names:
            item["name"] = name or _scorer_name(ref)
        result.append(item)
    return result


def _summary(obj: ObjSchema, *, include_scorers: bool) -> MonitorSummary:
    val = _serialized_val(obj)
    raw_name = val.get("name")
    display_name = raw_name if isinstance(raw_name, str) and raw_name else obj.object_id
    raw_sampling_rate = val.get("sampling_rate")
    sampling_rate = (
        float(raw_sampling_rate)
        if isinstance(raw_sampling_rate, (int, float))
        else None
    )
    return MonitorSummary(
        object_id=obj.object_id,
        display_name=display_name,
        active=bool(val.get(_ACTIVE_FIELD, False)),
        classification=("signal" if _is_signal(obj, val) else "legacy_monitor"),
        sampling_rate=sampling_rate,
        op_names=_op_names(val),
        scorers=_scorers(val, include_scorer_names=include_scorers),
        version=f"v{obj.version_index}",
        digest=obj.digest,
    )


def _monitor_filter(*, object_ids: list[str] | None = None) -> ObjectVersionFilter:
    return ObjectVersionFilter(
        base_object_classes=[_MONITOR_BASE_CLASS],
        object_ids=object_ids,
        latest_only=True,
    )


def list_monitors(
    path: str,
    *,
    include_scorers: bool = False,
    client_factory: ClientFactory = weave.init,
) -> list[MonitorSummary]:
    """Return every latest Monitor through bounded, stable pages."""
    client = _client(path, client_factory)
    rows: list[MonitorSummary] = []
    offset = 0
    while True:
        response = client.server.objs_query(
            ObjQueryReq(
                project_id=client.project_id,
                filter=_monitor_filter(),
                limit=_PAGE_SIZE,
                offset=offset,
                sort_by=[SortBy(field="object_id", direction="asc")],
            )
        )
        page = response.objs
        if not page:
            break
        rows.extend(_summary(obj, include_scorers=include_scorers) for obj in page)
        offset += len(page)
    return rows


def _read_latest(client: Any, object_id: str) -> ObjSchema:
    response = client.server.objs_query(
        ObjQueryReq(
            project_id=client.project_id,
            filter=_monitor_filter(object_ids=[object_id]),
            limit=2,
            offset=0,
        )
    )
    if len(response.objs) != 1:
        raise MonitorOperationError(
            f"expected one latest Monitor for {object_id!r}, found {len(response.objs)}"
        )
    return response.objs[0]


def _target(obj: ObjSchema) -> MutationTarget:
    val = _serialized_val(obj)
    return MutationTarget(
        obj=obj,
        active=bool(val.get(_ACTIVE_FIELD, False)),
        is_signal=_is_signal(obj, val),
    )


def _is_signal(obj: ObjSchema, val: dict[str, Any]) -> bool:
    """Match the Monitor shape shown by the Signals UI."""
    return (
        obj.leaf_object_class == _MONITOR_LEAF_CLASS
        and _op_names(val) == [_AGENT_TURN_ENDED_OP]
    )


def _validate_class_metadata(obj: ObjSchema, val: dict[str, Any]) -> None:
    if obj.base_object_class != _MONITOR_BASE_CLASS:
        raise MonitorOperationError(
            f"{obj.object_id!r} is not stored with base class {_MONITOR_BASE_CLASS!r}"
        )
    if obj.leaf_object_class not in {"Monitor", "ClassifierMonitor"}:
        raise MonitorOperationError(
            f"unsupported Monitor leaf class for {obj.object_id!r}: "
            f"{obj.leaf_object_class!r}"
        )
    if val.get("_class_name") != obj.leaf_object_class:
        raise MonitorOperationError(
            f"serialized class metadata does not match {obj.object_id!r}"
        )
    if val.get("_type") != obj.leaf_object_class:
        raise MonitorOperationError(
            f"serialized type metadata does not match {obj.object_id!r}"
        )
    bases = val.get("_bases")
    if not isinstance(bases, list) or bases[-2:] != ["Object", "BaseModel"]:
        raise MonitorOperationError(
            f"serialized base metadata is invalid for {obj.object_id!r}"
        )
    if obj.leaf_object_class == "ClassifierMonitor" and "Monitor" not in bases[:-2]:
        raise MonitorOperationError(
            f"serialized classifier metadata is missing Monitor for {obj.object_id!r}"
        )


def _emit_plan(project_id: str, targets: Sequence[MutationTarget], emit: Emit) -> None:
    emit(f"project={project_id}")
    emit("object_ids=" + json.dumps([target.obj.object_id for target in targets]))
    for target in targets:
        emit(
            "target="
            + json.dumps(
                {
                    "active": target.active,
                    "digest": target.obj.digest,
                    "object_id": target.obj.object_id,
                    "classification": (
                        "signal" if target.is_signal else "legacy_monitor"
                    ),
                },
                sort_keys=True,
            )
        )


def _verify(
    path: str,
    object_id: str,
    *,
    active: bool,
    digest: str,
    base_object_class: str | None,
    leaf_object_class: str | None,
    client_factory: ClientFactory,
    progress: MutationResult,
) -> None:
    fresh_client = _client(path, client_factory)
    try:
        verified = _read_latest(fresh_client, object_id)
    except MonitorOperationError as exc:
        raise VerificationError(
            f"fresh read could not verify {object_id!r}: {exc}", progress
        ) from exc
    try:
        verified_val = _serialized_val(verified)
    except MonitorOperationError as exc:
        raise VerificationError(
            f"fresh read returned an invalid Monitor for {object_id!r}: {exc}",
            progress,
        ) from exc
    if verified.digest != digest or verified_val.get(_ACTIVE_FIELD) is not active:
        raise VerificationError(
            f"fresh read did not verify {object_id!r} active={active}", progress
        )
    if (
        verified.base_object_class != base_object_class
        or verified.leaf_object_class != leaf_object_class
    ):
        raise VerificationError(
            f"fresh read found changed class metadata for {object_id!r}", progress
        )


def set_active(
    path: str,
    object_ids: Sequence[str],
    *,
    active: bool,
    expected_digest: str | None = None,
    allow_signals: bool = False,
    dry_run: bool = False,
    client_factory: ClientFactory = weave.init,
    emit: Emit = print,
) -> MutationResult:
    """Set ``active`` after a digest re-read, then verify with a fresh client."""
    if not object_ids:
        raise ValueError("at least one --object-id is required")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("--object-id values must be unique")
    if expected_digest is not None and len(object_ids) != 1:
        raise ValueError("--expected-digest requires exactly one --object-id")

    client = _client(path, client_factory)
    targets = [_target(_read_latest(client, object_id)) for object_id in object_ids]
    _emit_plan(client.project_id, targets, emit)

    if expected_digest is not None and targets[0].obj.digest != expected_digest:
        raise ConcurrentUpdateError(
            f"expected digest {expected_digest!r}, found {targets[0].obj.digest!r}"
        )
    protected = [target.obj.object_id for target in targets if target.is_signal]
    if protected and not allow_signals:
        raise MonitorOperationError(
            "refusing to mutate Signal monitor(s) without --allow-signals: "
            + ", ".join(protected)
        )

    if dry_run:
        return MutationResult(
            changed=0,
            unchanged=sum(target.active is active for target in targets),
            verified=0,
        )

    progress = MutationResult()
    for planned in targets:
        current = _target(_read_latest(client, planned.obj.object_id))
        if current.obj.digest != planned.obj.digest:
            raise ConcurrentUpdateError(
                f"digest changed for {planned.obj.object_id!r}: "
                f"{planned.obj.digest!r} -> {current.obj.digest!r}",
                progress,
            )

        if current.active is active:
            progress = MutationResult(
                changed=progress.changed,
                unchanged=progress.unchanged + 1,
                verified=progress.verified,
            )
            _verify(
                path,
                current.obj.object_id,
                active=active,
                digest=current.obj.digest,
                base_object_class=current.obj.base_object_class,
                leaf_object_class=current.obj.leaf_object_class,
                client_factory=client_factory,
                progress=progress,
            )
        else:
            edited_val = _copy_serialized(_serialized_val(current.obj))
            _validate_class_metadata(current.obj, edited_val)
            edited_val[_ACTIVE_FIELD] = active
            created = client.server.obj_create(
                ObjCreateReq(
                    obj=ObjSchemaForInsert(
                        project_id=client.project_id,
                        object_id=current.obj.object_id,
                        val=edited_val,
                    )
                )
            )
            progress = MutationResult(
                changed=progress.changed + 1,
                unchanged=progress.unchanged,
                verified=progress.verified,
            )
            _verify(
                path,
                current.obj.object_id,
                active=active,
                digest=created.digest,
                base_object_class=current.obj.base_object_class,
                leaf_object_class=current.obj.leaf_object_class,
                client_factory=client_factory,
                progress=progress,
            )

        progress = MutationResult(
            changed=progress.changed,
            unchanged=progress.unchanged,
            verified=progress.verified + 1,
        )
    return progress


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List latest Monitor objects")
    list_parser.add_argument("--path", required=True, metavar="ENTITY/PROJECT")
    list_parser.add_argument("--include-scorers", action="store_true")
    list_parser.add_argument("--json", action="store_true")

    set_parser = subparsers.add_parser("set-active", help="Set Monitor active state")
    set_parser.add_argument("--path", required=True, metavar="ENTITY/PROJECT")
    set_parser.add_argument("--object-id", required=True, action="append")
    set_parser.add_argument("--active", required=True, type=_parse_bool)
    set_parser.add_argument("--expected-digest")
    set_parser.add_argument("--allow-signals", action="store_true")

    legacy_parser = subparsers.add_parser(
        "deactivate-legacy", help="Deactivate named legacy Monitor objects"
    )
    legacy_parser.add_argument("--path", required=True, metavar="ENTITY/PROJECT")
    legacy_parser.add_argument("--object-id", required=True, action="append")
    legacy_parser.add_argument("--dry-run", action="store_true")
    legacy_parser.add_argument("--allow-signals", action="store_true")
    return parser


def _emit_result(result: MutationResult, *, answer: str, emit: Emit) -> None:
    emit(f"answer={answer}")
    emit(f"changed={result.changed}")
    emit(f"unchanged={result.unchanged}")
    emit(f"verified={result.verified}")


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: ClientFactory = weave.init,
    emit: Emit = print,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            monitors = list_monitors(
                args.path,
                include_scorers=args.include_scorers,
                client_factory=client_factory,
            )
            serialized = [asdict(monitor) for monitor in monitors]
            if args.json:
                emit(json.dumps(serialized, sort_keys=True))
            else:
                for monitor in serialized:
                    emit("monitor=" + json.dumps(monitor, sort_keys=True))
            emit(f"answer=monitors={len(monitors)}")
            return 0

        result = set_active(
            args.path,
            args.object_id,
            active=(args.active if args.command == "set-active" else False),
            expected_digest=(
                args.expected_digest if args.command == "set-active" else None
            ),
            allow_signals=args.allow_signals,
            dry_run=(args.dry_run if args.command == "deactivate-legacy" else False),
            client_factory=client_factory,
            emit=emit,
        )
        action = "dry_run" if getattr(args, "dry_run", False) else "active_set"
        _emit_result(result, answer=action, emit=emit)
        return 0
    except MonitorOperationError as exc:
        _emit_result(exc.result, answer=f"error={exc}", emit=emit)
        return 1
    except Exception as exc:
        _emit_result(MutationResult(), answer=f"error={exc}", emit=emit)
        return 1


if __name__ == "__main__":
    sys.exit(main())
