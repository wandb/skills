# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Scan and preview W&B Table artifacts as EvalTables.

The safe workflow is:

    python table_artifact_to_eval_table.py scan --source-workspace entity/project
    python table_artifact_to_eval_table.py preview --source-workspace entity/project --dry-run
    python table_artifact_to_eval_table.py preview --source-workspace entity/project
    python table_artifact_to_eval_table.py verify-preview --run entity/project/preview-run --table-key table_preview

The scan and preview subcommands accept source runs, sweeps, workspaces/projects,
or individual run_table artifacts. The preview command creates one destination
run per source run in the same project and logs each table as an EvalTable under
a preview-specific key, leaving the original tables in place. It skips sources
that already have preview metadata unless --force is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import tempfile
from base64 import urlsafe_b64encode
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from zlib import crc32

MAX_ROWS = 10_000
MAX_COLUMNS = 100
MAX_TABLES_PER_BATCH = 50
DEFAULT_MAX_RUNS = 4
DEFAULT_MAX_STEPS_PER_RUN_KEY = 5
ROW_WARNING_THRESHOLD = 1_000
PREVIEW_TAG = "eval-table-preview"
DEFAULT_EVAL_TABLE_KEY_SUFFIX = "_preview"
DEFAULT_PREVIEW_RUN_SUFFIX = " preview"
ROW_INDEX_MATCH_EVAL_TABLE_KEY_SUFFIX = "_row_index_match"
ROW_INDEX_MATCH_PREVIEW_RUN_SUFFIX = " row_index_match"
MAX_RUN_NAME_LENGTH = 128
DOWNLOAD_ROOT = Path(tempfile.gettempdir()) / "wandb_eval_table_imports"
_PROGRESS_PREFIX = "[eval-table-preview]"
_Progress = Callable[[str], None]
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_BYTES_PER_KIB = 1024
_BITS_PER_MEGABIT = 1_000_000

ARTIFACT_URL_RE = re.compile(
    r"^/(?P<entity>[^/]+)/(?P<project>[^/]+)/artifacts/"
    r"(?P<artifact_type>[^/]+)/(?P<artifact_name>[^/]+)/(?P<version>[^/]+)"
    r"(?:/.*)?$"
)
RUN_URL_RE = re.compile(
    r"^/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/]+)"
    r"(?:/.*)?$"
)
SWEEP_URL_RE = re.compile(
    r"^/(?P<entity>[^/]+)/(?P<project>[^/]+)/sweeps/(?P<sweep_id>[^/]+)"
    r"(?:/.*)?$"
)
WORKSPACE_URL_RE = re.compile(
    r"^/(?P<entity>[^/]+)/(?P<project>[^/]+)"
    r"(?:/(?:workspace|overview)(?:/.*)?)?/?$"
)
VERSION_RE = re.compile(r"(?:^|:)v(?P<version>\d+)$")


def _noop_progress(_: str) -> None:
    return


def _stderr_progress(message: str) -> None:
    print(f"{_PROGRESS_PREFIX} {message}", file=sys.stderr, flush=True)


def configure_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except Exception:
            continue


def configure_wandb_console_env(progress: _Progress) -> None:
    previous_silent = os.environ.get("WANDB_SILENT")
    previous_quiet = os.environ.get("WANDB_QUIET")
    os.environ["WANDB_SILENT"] = "false"
    os.environ["WANDB_QUIET"] = "false"
    os.environ.setdefault("WANDB_CONSOLE", "wrap")
    if (
        (previous_silent or "").lower() in _TRUE_ENV_VALUES
        or (previous_quiet or "").lower() in _TRUE_ENV_VALUES
    ):
        progress("overriding W&B SDK quiet/silent console settings for visibility")


def wandb_console_settings(wandb: Any) -> Any | None:
    settings_cls = getattr(wandb, "Settings", None)
    if not callable(settings_cls):
        return None
    for kwargs in (
        {"console": "wrap", "quiet": False, "silent": False},
        {"console": "wrap", "quiet": False},
        {"console": "wrap"},
    ):
        try:
            return settings_cls(**kwargs)
        except TypeError:
            continue
    return None


def int_byte_size(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def entry_size(entry: Any) -> int | None:
    if isinstance(entry, dict):
        return int_byte_size(entry.get("size"))
    return int_byte_size(getattr(entry, "size", None))


def sum_entry_sizes(entries: Any) -> int | None:
    if isinstance(entries, dict):
        iterable = entries.values()
    elif isinstance(entries, (list, tuple)):
        iterable = entries
    else:
        return None

    total = 0
    seen = False
    for entry in iterable:
        size = entry_size(entry)
        if size is None:
            return None
        seen = True
        total += size
    return total if seen else None


def artifact_size_details(artifact: Any) -> tuple[int | None, str | None]:
    direct_size = int_byte_size(getattr(artifact, "size", None))
    if direct_size is not None:
        return direct_size, "artifact.size"

    manifest_entries = getattr(getattr(artifact, "manifest", None), "entries", None)
    manifest_size = sum_entry_sizes(manifest_entries)
    if manifest_size is not None:
        return manifest_size, "artifact.manifest.entries"

    files = getattr(artifact, "files", None)
    if callable(files):
        try:
            file_size = sum_entry_sizes(list(files()))
        except Exception:
            file_size = None
        if file_size is not None:
            return file_size, "artifact.files"

    return None, None


def directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < _BYTES_PER_KIB or unit == "TiB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= _BYTES_PER_KIB
    return f"{amount:.1f} TiB"


def format_mbps(byte_count: int, elapsed_seconds: float) -> str:
    if elapsed_seconds <= 0:
        return "unknown"
    return f"{byte_count * 8 / elapsed_seconds / _BITS_PER_MEGABIT:.2f} Mbps"


def artifact_size_summary(tables: list[ScannedTable]) -> tuple[int, int]:
    known_bytes = sum(
        table.artifact_size_bytes
        for table in tables
        if table.artifact_size_bytes is not None
    )
    unknown_count = sum(1 for table in tables if table.artifact_size_bytes is None)
    return known_bytes, unknown_count


@dataclass
class TableSource:
    artifact: Any
    source_run_path: str | None = None
    source_run_name: str | None = None
    source_run_id: str | None = None
    # A prior preview's manifest preserves these values even if artifact metadata
    # is incomplete when the retry resolves the exact artifact version.
    table_key: str | None = None
    source_step: int | None = None
    # Shape read from run history/summary at discovery time, so scan can size-filter
    # without downloading the artifact. None means "unknown" → scan downloads to measure.
    row_count: int | None = None
    column_count: int | None = None
    shape_source: str | None = None


@dataclass
class SourcePlan:
    source_kind: str
    source_label: str
    table_sources: list[TableSource]
    scanned_run_count: int = 0
    table_selection: str = "latest_per_run_key"
    max_steps_per_run_key: int | None = None


@dataclass
class ScannedTable:
    source: TableSource
    artifact_path: str
    table_key: str
    source_run_path: str | None
    source_run_name: str | None
    source_step: int | None
    row_count: int
    column_count: int
    status: str
    reasons: list[str]
    warnings: list[str]
    artifact_size_bytes: int | None = None
    artifact_size_source: str | None = None
    # Set when the table exceeds a size cap: the preview truncates it (keeps the
    # leading max_rows/max_columns) instead of skipping. None when it fits.
    truncation: dict[str, Any] | None = None
    # Populated lazily at preview time (import_scanned_table downloads on demand);
    # no longer set during scan now that sizing comes from history.
    table_json_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "table_key": self.table_key,
            "source_run_path": self.source_run_path,
            "source_run_name": self.source_run_name,
            "source_step": self.source_step,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "status": self.status,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_size": format_bytes(self.artifact_size_bytes),
            "artifact_size_source": self.artifact_size_source,
            "truncation": self.truncation,
        }


@dataclass
class ScanPlan:
    source_kind: str
    source_label: str
    target_project: str
    scanned_run_count: int
    tables: list[ScannedTable]
    max_rows: int
    max_columns: int
    max_tables: int
    table_selection: str = "latest_per_run_key"
    max_steps_per_run_key: int | None = None
    # Root the preview step downloads eligible artifacts into (download is deferred
    # out of scan); None falls back to DOWNLOAD_ROOT.
    download_root: Path | None = None
    # Set when a sweep/workspace scan fell back to the most-recent-N runs because
    # no --run-id was passed; surfaced in to_dict so the agent sees the default
    # was applied (rather than the tool silently picking the comparison set).
    run_scope: dict[str, Any] | None = None

    @property
    def eligible_tables(self) -> list[ScannedTable]:
        return [table for table in self.tables if table.status == "eligible"]

    def to_dict(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for table in self.tables:
            status_counts[table.status] = status_counts.get(table.status, 0) + 1
        warning_count = sum(1 for table in self.tables if table.warnings)
        total_known_bytes, total_unknown_count = artifact_size_summary(self.tables)
        eligible_known_bytes, eligible_unknown_count = artifact_size_summary(
            self.eligible_tables
        )
        # preview converts one key per call, so the batch cap applies per key,
        # not to the whole scan. Report per-key eligible counts and flag only the
        # keys that would exceed the cap on their own.
        eligible_by_table_key: dict[str, int] = {}
        for table in self.eligible_tables:
            eligible_by_table_key[table.table_key] = (
                eligible_by_table_key.get(table.table_key, 0) + 1
            )
        keys_over_batch_limit = sorted(
            key
            for key, count in eligible_by_table_key.items()
            if count > self.max_tables
        )
        # Run-scoped per-key summary for the plan table. A key can be logged at
        # several steps within one run (several artifact versions in that run) and,
        # separately, across several runs. Report steps *per run* and runs *across
        # the scan* so the plan never reports the raw artifact-version count (which
        # equals steps_per_run x runs) as "steps".
        steps_by_key_run: dict[str, dict[str | None, set[int | None]]] = {}
        for table in self.eligible_tables:
            steps_by_key_run.setdefault(table.table_key, {}).setdefault(
                table.source_run_path, set()
            ).add(table.source_step)
        eligible_summary_by_table_key = {
            key: {
                "runs": len(per_run),
                "steps_per_run": max(
                    (len(steps) for steps in per_run.values()), default=0
                ),
                "artifacts": sum(len(steps) for steps in per_run.values()),
            }
            for key, per_run in steps_by_key_run.items()
        }
        # Compact per-table truncation notices for the plan: the agent reads these
        # to tell the user what will be cut (and which columns) before confirming.
        truncations = [
            {
                "table_key": table.table_key,
                "source_run_name": table.source_run_name,
                "source_step": table.source_step,
                "truncation": table.truncation,
            }
            for table in self.tables
            if table.truncation is not None
        ]
        return {
            "source_kind": self.source_kind,
            "source": self.source_label,
            "target_project": self.target_project,
            "scanned_run_count": self.scanned_run_count,
            "limits": {
                "max_rows": self.max_rows,
                "max_columns": self.max_columns,
                "max_table_artifacts_per_batch": self.max_tables,
                "max_tables_per_batch": self.max_tables,
            },
            "selection": {
                "mode": self.table_selection,
                "max_steps_per_run_key": self.max_steps_per_run_key,
            },
            "run_scope": self.run_scope,
            "candidate_count": len(self.tables),
            "eligible_count": len(self.eligible_tables),
            "artifact_size": {
                "known_total_bytes": total_known_bytes,
                "known_total": format_bytes(total_known_bytes),
                "unknown_count": total_unknown_count,
                "eligible_known_total_bytes": eligible_known_bytes,
                "eligible_known_total": format_bytes(eligible_known_bytes),
                "eligible_unknown_count": eligible_unknown_count,
            },
            "eligible_by_table_key": eligible_by_table_key,
            "eligible_summary_by_table_key": eligible_summary_by_table_key,
            "truncated_count": len(truncations),
            "truncations": truncations,
            "already_previewed_count": status_counts.get("already_previewed", 0),
            "warning_count": warning_count,
            "status_counts": status_counts,
            "batch_limit_exceeded": bool(keys_over_batch_limit),
            "keys_over_batch_limit": keys_over_batch_limit,
            "candidates": [table.to_dict() for table in self.tables],
        }


@dataclass
class ImportedTable:
    artifact_path: str
    eval_table: Any
    log_key: str
    source_step: int | None
    estimated_upload_bytes: int | None = None


@dataclass
class UnspecifiedColumnConfirmation:
    accepted_columns: set[str]


def parse_artifact_path(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        match = ARTIFACT_URL_RE.match(parsed.path)
        if match is None:
            raise ValueError(f"Unsupported W&B artifact URL: {value}")
        parts = {key: unquote(part) for key, part in match.groupdict().items()}
        return (
            f"{parts['entity']}/{parts['project']}/"
            f"{parts['artifact_name']}:{parts['version']}"
        )

    return value


def parse_run_path(value: str) -> tuple[str, str, str]:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        match = RUN_URL_RE.match(parsed.path)
        if match is None:
            raise ValueError(f"Unsupported W&B run URL: {value}")
        parts = {key: unquote(part) for key, part in match.groupdict().items()}
        project_path = f"{parts['entity']}/{parts['project']}"
        return f"{project_path}/{parts['run_id']}", project_path, parts["run_id"]

    parts = [unquote(part) for part in value.split("/") if part]
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}", parts[0], parts[1]
    if len(parts) == 3:
        project_path = f"{parts[0]}/{parts[1]}"
        return f"{project_path}/{parts[2]}", project_path, parts[2]

    raise ValueError(
        "Run source must be '<project>/<run_id_or_name>', "
        "'<entity>/<project>/<run_id_or_name>', or a W&B run URL."
    )


def parse_sweep_path(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        match = SWEEP_URL_RE.match(parsed.path)
        if match is None:
            raise ValueError(f"Unsupported W&B sweep URL: {value}")
        parts = {key: unquote(part) for key, part in match.groupdict().items()}
        return (
            f"{parts['entity']}/{parts['project']}/{parts['sweep_id']}",
            f"{parts['entity']}/{parts['project']}",
        )

    parts = [unquote(part) for part in value.split("/") if part]
    if len(parts) == 2:
        return "/".join(parts), parts[0]
    if len(parts) == 3:
        return "/".join(parts), f"{parts[0]}/{parts[1]}"

    raise ValueError(
        "Sweep source must be '<project>/<sweep_id>', "
        "'<entity>/<project>/<sweep_id>', or a W&B sweep URL."
    )


def parse_workspace_path(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        match = WORKSPACE_URL_RE.match(parsed.path)
        if match is None:
            raise ValueError(f"Unsupported W&B workspace URL: {value}")
        parts = {key: unquote(part) for key, part in match.groupdict().items()}
        return f"{parts['entity']}/{parts['project']}"

    parts = [unquote(part) for part in value.split("/") if part]
    if len(parts) in {1, 2}:
        return "/".join(parts)

    raise ValueError(
        "Workspace source must be '<project>', '<entity>/<project>', "
        "or a W&B project/workspace URL."
    )


def parse_column_args(values: list[str] | None, flag_name: str) -> list[str] | None:
    if values is None:
        return None
    columns = [
        column.strip()
        for value in values
        for column in value.split(",")
        if column.strip()
    ]
    if not columns:
        raise ValueError(f"{flag_name} cannot be empty")
    return columns


def parse_table_keys(values: list[str] | None) -> set[str] | None:
    columns = parse_column_args(values, "--table-key")
    if columns is None:
        return None
    return {normalize_table_key(column) for column in columns}


def parse_run_ids(values: list[str] | None) -> list[str] | None:
    run_ids = parse_column_args(values, "--run-id")
    if run_ids is None:
        return None
    return run_ids


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def normalize_table_key(value: str) -> str:
    trimmed = value.strip()
    for prefix in ("summary:", "config:"):
        if trimmed.startswith(prefix):
            return trimmed[len(prefix) :]
    return trimmed


def infer_source_kind(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        if ARTIFACT_URL_RE.match(parsed.path):
            return "artifact"
        if RUN_URL_RE.match(parsed.path):
            return "run"
        if SWEEP_URL_RE.match(parsed.path):
            return "sweep"
        if WORKSPACE_URL_RE.match(parsed.path):
            return "workspace"
        raise ValueError(f"Unsupported W&B URL: {value}")

    if ":" in value.rsplit("/", maxsplit=1)[-1]:
        return "artifact"
    return "run"


def split_project_path(project_path: str) -> tuple[str | None, str]:
    parts = project_path.split("/")
    if len(parts) == 1:
        return None, project_path
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("Project must be '<project>' or '<entity>/<project>'.")


def project_from_artifact_source(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        match = ARTIFACT_URL_RE.match(parsed.path)
        if match is None:
            return None
        parts = {key: unquote(part) for key, part in match.groupdict().items()}
        return f"{parts['entity']}/{parts['project']}"

    parts = [unquote(part) for part in value.split("/") if part]
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    return None


def safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sanitize_artifact_name(name: str) -> str:
    """Mirror wandb's run_table artifact naming so we can match a known key forward.

    A table key can contain characters (e.g. "/") that are illegal in artifact
    names. When it does, the SDK strips them and appends a 6-char CRC suffix over
    the *original* string to keep the name unique — see
    wandb.sdk.artifacts._internal_artifact.sanitize_artifact_name. That rewrite is
    lossy, so we can't recover the original key from the sanitized artifact name;
    instead we sanitize a candidate key forward and compare (see
    table_key_from_artifact_name). Kept in sync with the SDK by copy — a stale copy
    only means special-char keys fall back to the lossy name parse, never a crash.
    """
    if (sanitized := re.sub(r"[^a-zA-Z0-9_\-.]+", "", name)) == name:
        return name
    crc = crc32(name.encode("utf-8")) & 0xFFFFFFFF
    suffix = urlsafe_b64encode(crc.to_bytes(4, byteorder="big")).rstrip(b"=").decode("ascii")
    return f"{sanitized}-{suffix}"


def artifact_version_index(artifact: Any) -> int:
    for value in (getattr(artifact, "version", None), getattr(artifact, "name", None)):
        if value is None:
            continue
        match = VERSION_RE.search(value)
        if match:
            return int(match.group("version"))
    return -1


def artifact_path(artifact: Any) -> str:
    return getattr(artifact, "source_qualified_name", None) or artifact.qualified_name


def table_key_from_artifact_name(
    artifact: Any,
    source_run_id: str | None,
    candidate_keys: set[str] | None = None,
) -> str | None:
    table_key = getattr(artifact, "table_key", None)
    if isinstance(table_key, str) and table_key:
        return normalize_table_key(table_key)

    artifact_name = getattr(artifact, "name", None)
    if not isinstance(artifact_name, str):
        return None
    name = artifact_name.rsplit("/", maxsplit=1)[-1].split(":", maxsplit=1)[0]

    # When the caller already knows which keys it wants, recover the real key by
    # sanitizing each candidate forward (wandb names the artifact
    # "run-<id>-<key>", sanitized) and matching the result against this artifact.
    # This is the only way to map a special-char key like "samples/core_full" —
    # whose artifact name "run-<id>-samplescore_full-<crc>" is lossy — back to the
    # key the user asked for, and it also keeps the returned key (used for display
    # and the "_preview" log key) equal to the original rather than the mangled form.
    if candidate_keys and source_run_id:
        for candidate in candidate_keys:
            if sanitize_artifact_name(f"run-{source_run_id}-{candidate}") == name:
                return normalize_table_key(candidate)

    prefixes = []
    if source_run_id:
        prefixes.extend([f"run-{source_run_id}-incr-", f"run-{source_run_id}-"])
    prefixes.extend(["run-", "incr-"])

    for prefix in prefixes:
        if name.startswith(prefix):
            return normalize_table_key(name[len(prefix) :])
    return None


def artifact_sort_key(artifact: Any) -> tuple[int, int, str, str]:
    step = artifact.history_step
    if step is None:
        step = sys.maxsize
    return (
        step,
        artifact_version_index(artifact),
        artifact.created_at or "",
        artifact.name,
    )


def find_table_json(download_dir: Path) -> Path:
    table_paths = sorted(download_dir.rglob("*.table.json"))
    if len(table_paths) == 1:
        return table_paths[0]
    if not table_paths:
        raise FileNotFoundError(f"No *.table.json file found under {download_dir}.")
    choices = "\n".join(f"  {path.relative_to(download_dir)}" for path in table_paths)
    raise ValueError(
        "Found multiple table JSON files; expected exactly one per table artifact.\n"
        f"{choices}"
    )


def default_log_key(table_json_path: Path) -> str:
    name = table_json_path.name
    suffix = ".table.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return table_json_path.stem


def table_shape(table_json: dict[str, Any]) -> tuple[int, int]:
    columns = table_json.get("columns")
    data = table_json.get("data")
    column_count = len(columns) if isinstance(columns, list) else 0
    row_count = len(data) if isinstance(data, list) else 0
    return row_count, column_count


def is_table_media_value(value: Any) -> bool:
    """Whether a run history/summary value is a logged-table media dict.

    Covers both regular (`table-file`) and incremental (`incremental-table-file`)
    tables — we need to recognize incrementals so we can avoid trusting their
    per-step row counts (see media_table_shape).
    """
    if not isinstance(value, dict):
        return False
    value_type = str(value.get("_type") or "").lower()
    if value_type in {"table-file", "incremental-table-file"}:
        return True
    return str(value.get("path") or "").lower().endswith(".table.json")


def media_table_shape(value: Any) -> tuple[int, int, bool] | None:
    """Extract (nrows, ncols, is_incremental) from a table media dict.

    `wandb.Table` writes `nrows`/`ncols` into the run's history/summary value, so
    we can size a table without downloading it. Returns None when the value isn't
    a table or lacks the counts.
    """
    if not is_table_media_value(value):
        return None
    nrows = value.get("nrows")
    ncols = value.get("ncols")
    if not isinstance(nrows, int) or not isinstance(ncols, int):
        return None
    is_incremental = str(value.get("_type") or "").lower() == "incremental-table-file"
    return nrows, ncols, is_incremental


def run_table_shapes(
    run: Any,
    keys: set[str],
    *,
    per_step: bool,
    min_step: int | None = None,
    max_step: int | None = None,
) -> dict[tuple[str, int | None], tuple[int, int, bool]]:
    """Read table dimensions from a run's summary/history without downloading.

    Returns `{(table_key, step): (nrows, ncols, is_incremental)}`. The latest value
    per key is also stored under `(table_key, None)` from the run summary. Best
    effort: any read failure is swallowed so the scan can fall back to downloading.
    """
    shapes: dict[tuple[str, int | None], tuple[int, int, bool]] = {}
    if not keys:
        return shapes
    try:
        summary = dict(getattr(run, "summary_metrics", None) or {})
    except Exception:
        summary = {}
    for key in keys:
        shape = media_table_shape(summary.get(key))
        if shape is not None:
            shapes[(key, None)] = shape
    if not per_step:
        return shapes
    for key in keys:
        try:
            rows = run.scan_history(keys=[key], min_step=min_step, max_step=max_step)
            for row in rows:
                shape = media_table_shape(row.get(key))
                if shape is not None:
                    shapes[(key, row.get("_step"))] = shape
        except Exception:
            continue
    return shapes


def size_truncation(
    row_count: int,
    column_count: int,
    columns: list[str] | None = None,
    max_rows: int = MAX_ROWS,
    max_columns: int = MAX_COLUMNS,
) -> dict[str, Any] | None:
    """Describe how a table exceeding the size caps will be truncated, or None.

    Oversized tables are truncated rather than skipped: the preview keeps the
    first ``max_rows`` rows and the first ``max_columns`` columns. ``columns``
    supplies the header names so the plan can tell the user exactly which columns
    get dropped; when it is None (shape came from history, not a download) only
    the dropped count is reported.
    """
    if row_count <= max_rows and column_count <= max_columns:
        return None
    truncation: dict[str, Any] = {}
    if row_count > max_rows:
        truncation["rows"] = {"kept": max_rows, "dropped": row_count - max_rows}
    if column_count > max_columns:
        column_info: dict[str, Any] = {
            "kept": max_columns,
            "dropped": column_count - max_columns,
        }
        if columns is not None:
            column_info["dropped_columns"] = list(columns[max_columns:])
        truncation["columns"] = column_info
    return truncation


def truncate_source_table(
    source_table: Any,
    max_rows: int = MAX_ROWS,
    max_columns: int = MAX_COLUMNS,
    keep_columns: set[str] | None = None,
) -> dict[str, Any] | None:
    """Truncate a loaded table in place to the size caps; return what was dropped.

    Columns are dropped before rows so the header and every row stay aligned. Rows
    keep the leading ``max_rows``. Columns keep the leading ``max_columns`` in
    original order, except that any name in ``keep_columns`` (the columns the agent
    explicitly assigned a role) is always kept regardless of position, so a named
    score/input/output column is never silently dropped — the remaining budget goes
    to the other columns in order.

    If the pinned columns alone exceed ``max_columns`` we keep all of them anyway
    (slightly over the soft cap): honoring an explicit request beats dropping a
    requested column and failing downstream. That can't happen in practice — the
    agent auto-types at most a few columns.
    """
    columns = list(source_table.columns)
    data = source_table.data
    truncation: dict[str, Any] = {}

    if len(columns) > max_columns:
        keep = keep_columns or set()
        pinned = [column for column in columns if column in keep]
        budget = max(max_columns - len(pinned), 0)
        filler = [column for column in columns if column not in keep][:budget]
        kept = set(pinned) | set(filler)
        keep_indices = [i for i, column in enumerate(columns) if column in kept]
        dropped_columns = [column for column in columns if column not in kept]
        source_table.columns = [columns[i] for i in keep_indices]
        source_table.data = [[row[i] for i in keep_indices] for row in data]
        data = source_table.data
        truncation["columns"] = {
            "kept": len(keep_indices),
            "dropped": len(dropped_columns),
            "dropped_columns": dropped_columns,
        }

    if len(data) > max_rows:
        truncation["rows"] = {"kept": max_rows, "dropped": len(data) - max_rows}
        source_table.data = data[:max_rows]

    return truncation or None


def size_warnings(
    row_count: int,
    row_warning_threshold: int = ROW_WARNING_THRESHOLD,
) -> list[str]:
    if row_count > row_warning_threshold:
        return [
            f"row_count {row_count} exceeds warning threshold "
            f"{row_warning_threshold}; preview may fail due to number of rows"
        ]
    return []


def resolve_run(api: Any, source_run: str) -> Any:
    run_path, project_path, run_id_or_name = parse_run_path(source_run)
    try:
        return api.run(run_path)
    except Exception as exact_error:
        filters = {
            "$or": [
                {"name": run_id_or_name},
                {"display_name": run_id_or_name},
                {"displayName": run_id_or_name},
            ]
        }
        matches = list(api.runs(project_path, filters=filters, per_page=100))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Could not find run {run_id_or_name!r} in {project_path!r}."
            ) from exact_error
        choices = "\n".join(
            f"  {run.entity}/{run.project}/{run.id} ({run.name})" for run in matches
        )
        raise ValueError(
            f"Found multiple runs matching {run_id_or_name!r} in {project_path!r}:\n"
            f"{choices}"
        ) from exact_error


def run_path(run: Any) -> str:
    return f"{run.entity}/{run.project}/{run.id}"


def logged_table_artifacts(
    run: Any,
    *,
    table_keys: set[str] | None = None,
    latest_only: bool = True,
    max_steps_per_run_key: int | None = DEFAULT_MAX_STEPS_PER_RUN_KEY,
) -> list[Any]:
    if max_steps_per_run_key is not None and max_steps_per_run_key < 1:
        raise ValueError("--max-steps-per-run-key must be at least 1.")

    artifacts = [
        artifact for artifact in run.logged_artifacts() if artifact.type == "run_table"
    ]
    sorted_artifacts = sorted(artifacts, key=artifact_sort_key)
    if table_keys is not None:
        sorted_artifacts = [
            artifact
            for artifact in sorted_artifacts
            if table_key_from_artifact_name(
                artifact, getattr(run, "id", None), table_keys
            )
            in table_keys
        ]
    if max_steps_per_run_key is not None:
        return latest_artifacts_per_table_key(
            sorted_artifacts,
            source_run_id=getattr(run, "id", None),
            limit=max_steps_per_run_key,
            candidate_keys=table_keys,
        )
    if not latest_only:
        return sorted_artifacts

    return latest_artifacts_per_table_key(
        sorted_artifacts,
        source_run_id=getattr(run, "id", None),
        limit=1,
        candidate_keys=table_keys,
    )


def latest_artifacts_per_table_key(
    sorted_artifacts: list[Any],
    *,
    source_run_id: str | None,
    limit: int,
    candidate_keys: set[str] | None = None,
) -> list[Any]:
    artifacts_by_key: dict[str, list[Any]] = {}
    fallback_artifacts: list[Any] = []
    for artifact in sorted_artifacts:
        table_key = table_key_from_artifact_name(
            artifact, source_run_id, candidate_keys
        )
        if table_key is None:
            fallback_artifacts.append(artifact)
            continue
        artifacts_by_key.setdefault(table_key, []).append(artifact)

    selected = [
        artifact
        for artifacts in artifacts_by_key.values()
        for artifact in artifacts[-limit:]
    ]
    return sorted(selected, key=artifact_sort_key) + fallback_artifacts


def table_selection_mode(
    *,
    latest_only: bool,
    max_steps_per_run_key: int | None,
) -> str:
    if max_steps_per_run_key is not None:
        return "latest_steps_per_run_key"
    if latest_only:
        return "latest_per_run_key"
    return "all_steps"


def table_sources_for_run(
    run: Any,
    *,
    table_keys: set[str] | None = None,
    latest_only: bool = True,
    max_steps_per_run_key: int | None = DEFAULT_MAX_STEPS_PER_RUN_KEY,
    progress: _Progress = _noop_progress,
) -> list[TableSource]:
    path = run_path(run)
    progress(f"reading logged table artifacts for run {path}")
    run_id = getattr(run, "id", None)
    artifacts = logged_table_artifacts(
        run,
        table_keys=table_keys,
        latest_only=latest_only,
        max_steps_per_run_key=max_steps_per_run_key,
    )
    progress(f"found {len(artifacts)} table artifact candidate(s) for run {path}")
    keyed = [
        (artifact, table_key_from_artifact_name(artifact, run_id, table_keys))
        for artifact in artifacts
    ]

    # Read dimensions from the run's history/summary so scan can size-filter
    # without downloading. Only walk history when more than the latest step per key
    # may be selected; otherwise the summary read alone covers it.
    per_step = (max_steps_per_run_key is None and not latest_only) or (
        max_steps_per_run_key is not None and max_steps_per_run_key > 1
    )
    steps = [a.history_step for a, _ in keyed if a.history_step is not None]
    if per_step and keyed:
        progress(f"reading table history shapes for run {path}")
    shapes = run_table_shapes(
        run,
        {key for _, key in keyed if key is not None},
        per_step=per_step,
        min_step=min(steps) if steps else None,
        max_step=(max(steps) + 1) if steps else None,
    )

    def step_rank(artifact: Any) -> int:
        return sys.maxsize if artifact.history_step is None else artifact.history_step

    latest_rank_by_key: dict[str, int] = {}
    for artifact, key in keyed:
        if key is not None:
            latest_rank_by_key[key] = max(
                latest_rank_by_key.get(key, -1), step_rank(artifact)
            )

    sources: list[TableSource] = []
    for artifact, key in keyed:
        source = TableSource(
            artifact=artifact,
            source_run_path=path,
            source_run_name=run.name,
            source_run_id=run_id,
        )
        shape: tuple[int, int, bool] | None = None
        shape_source: str | None = None
        if key is not None:
            shape = shapes.get((key, artifact.history_step))
            if shape is not None:
                shape_source = "history"
            elif step_rank(artifact) == latest_rank_by_key.get(key):
                shape = shapes.get((key, None))
                shape_source = "summary" if shape is not None else None
        if shape is not None:
            nrows, ncols, is_incremental = shape
            if is_incremental:
                # Per-step nrows is the increment, not the cumulative total, so it
                # would understate size. Leave counts unset → scan downloads to measure.
                source.shape_source = "incremental_skip"
            else:
                source.row_count = nrows
                source.column_count = ncols
                source.shape_source = shape_source
        sources.append(source)
    return sources


def build_artifact_plan(
    api: Any,
    source: str,
    *,
    progress: _Progress = _noop_progress,
) -> SourcePlan:
    source_artifact_path = parse_artifact_path(source)
    progress(f"resolving table artifact {source_artifact_path}")
    artifact = api.artifact(source_artifact_path)
    progress(f"resolved table artifact {source_artifact_path}")
    return SourcePlan(
        source_kind="artifact",
        source_label=source_artifact_path,
        table_sources=[TableSource(artifact=artifact)],
    )


def build_run_plan(
    api: Any,
    source: str,
    *,
    table_keys: set[str] | None = None,
    latest_only: bool = True,
    max_steps_per_run_key: int | None = DEFAULT_MAX_STEPS_PER_RUN_KEY,
    progress: _Progress = _noop_progress,
) -> SourcePlan:
    progress(f"resolving source run {source}")
    source_run = resolve_run(api, source)
    source_run_path = run_path(source_run)
    progress(f"resolved source run {source_run_path}")
    table_sources = table_sources_for_run(
        source_run,
        table_keys=table_keys,
        latest_only=latest_only,
        max_steps_per_run_key=max_steps_per_run_key,
        progress=progress,
    )
    progress(
        f"run plan complete for {source_run_path}: "
        f"{len(table_sources)} table artifact candidate(s)"
    )
    return SourcePlan(
        source_kind="run",
        source_label=source_run_path,
        table_sources=table_sources,
        scanned_run_count=1,
        table_selection=table_selection_mode(
            latest_only=latest_only,
            max_steps_per_run_key=max_steps_per_run_key,
        ),
        max_steps_per_run_key=max_steps_per_run_key,
    )


def resolve_project_run(api: Any, project_path: str, run_id: str) -> Any:
    return api.run(f"{project_path}/{run_id}")


def build_sweep_plan(
    api: Any,
    source: str,
    *,
    table_keys: set[str] | None = None,
    latest_only: bool = True,
    max_steps_per_run_key: int | None = DEFAULT_MAX_STEPS_PER_RUN_KEY,
    max_runs: int = DEFAULT_MAX_RUNS,
    run_ids: list[str] | None = None,
    progress: _Progress = _noop_progress,
) -> SourcePlan:
    source_sweep_path, _ = parse_sweep_path(source)
    progress(f"resolving sweep {source_sweep_path}")
    sweep = api.sweep(source_sweep_path)
    project_path = f"{sweep.entity}/{sweep.project}"
    progress(f"resolved sweep {sweep.entity}/{sweep.project}/{sweep.id}")
    if run_ids is not None:
        progress(f"resolving {len(run_ids)} requested run(s) in {project_path}")
        runs = []
        for index, run_id in enumerate(run_ids, start=1):
            progress(f"resolving requested run {index}/{len(run_ids)}: {run_id}")
            runs.append(resolve_project_run(api, project_path, run_id))
    else:
        progress(f"listing up to {max_runs} sweep run(s)")
        runs = list(islice(sweep.runs, max_runs))
    progress(f"found {len(runs)} sweep run(s) to scan")
    table_sources = []
    for index, run in enumerate(runs, start=1):
        progress(f"scanning sweep run {index}/{len(runs)}: {run_path(run)}")
        table_sources.extend(
            table_sources_for_run(
                run,
                table_keys=table_keys,
                latest_only=latest_only,
                max_steps_per_run_key=max_steps_per_run_key,
                progress=progress,
            )
        )
    progress(
        f"sweep plan complete: {len(table_sources)} table artifact candidate(s)"
    )
    return SourcePlan(
        source_kind="sweep",
        source_label=f"{sweep.entity}/{sweep.project}/{sweep.id}",
        table_sources=table_sources,
        scanned_run_count=len(runs),
        table_selection=table_selection_mode(
            latest_only=latest_only,
            max_steps_per_run_key=max_steps_per_run_key,
        ),
        max_steps_per_run_key=max_steps_per_run_key,
    )


def build_workspace_plan(
    api: Any,
    source: str,
    *,
    table_keys: set[str] | None = None,
    latest_only: bool = True,
    max_steps_per_run_key: int | None = DEFAULT_MAX_STEPS_PER_RUN_KEY,
    max_runs: int = DEFAULT_MAX_RUNS,
    run_ids: list[str] | None = None,
    progress: _Progress = _noop_progress,
) -> SourcePlan:
    source_workspace_path = parse_workspace_path(source)
    if run_ids is not None:
        progress(
            f"resolving {len(run_ids)} requested run(s) in {source_workspace_path}"
        )
        runs = []
        for index, run_id in enumerate(run_ids, start=1):
            progress(f"resolving requested run {index}/{len(run_ids)}: {run_id}")
            runs.append(resolve_project_run(api, source_workspace_path, run_id))
    else:
        # Skip our own preview runs when picking "recent runs" to convert: they
        # are outputs of this tool, not sources. Including them wastes scan slots
        # and shifts the recent-runs window every time previews are created, so
        # repeated runs never converge. Over-fetch, then take the first max_runs
        # non-preview runs.
        progress(
            f"listing up to {max_runs} recent non-preview run(s) in "
            f"{source_workspace_path}"
        )
        runs = list(
            islice(
                (
                    run
                    for run in api.runs(source_workspace_path, order="-created_at")
                    if not is_preview_run(run)
                ),
                max_runs,
            )
        )
    progress(f"found {len(runs)} workspace run(s) to scan")
    table_sources = []
    for index, run in enumerate(runs, start=1):
        progress(f"scanning workspace run {index}/{len(runs)}: {run_path(run)}")
        table_sources.extend(
            table_sources_for_run(
                run,
                table_keys=table_keys,
                latest_only=latest_only,
                max_steps_per_run_key=max_steps_per_run_key,
                progress=progress,
            )
        )
    progress(
        f"workspace plan complete: {len(table_sources)} table artifact candidate(s)"
    )
    return SourcePlan(
        source_kind="workspace",
        source_label=source_workspace_path,
        table_sources=table_sources,
        scanned_run_count=len(runs),
        table_selection=table_selection_mode(
            latest_only=latest_only,
            max_steps_per_run_key=max_steps_per_run_key,
        ),
        max_steps_per_run_key=max_steps_per_run_key,
    )


@dataclass(frozen=True)
class _SourceKind:
    """Per-kind knowledge in one place instead of three parallel switch statements:
    how to read the destination project out of a source, how to build its plan, and
    whether it accepts --run-id / owes a defaulted-runs (run_scope) disclosure.

    `build` takes the full selection knob set by keyword; kinds ignore what they
    don't use (artifact takes none; run has no runs to select), so the dispatcher
    stays uniform.
    """

    accepts_run_ids: bool
    defaults_run_scope: bool
    project_of: Callable[[str], str | None]
    build: Callable[..., SourcePlan]


def _build_artifact_source(
    api: Any,
    source: str,
    *,
    progress: _Progress,
    **_: Any,
) -> SourcePlan:
    return build_artifact_plan(api, source, progress=progress)


def _build_run_source(
    api: Any,
    source: str,
    *,
    table_keys: set[str] | None,
    latest_only: bool,
    max_steps_per_run_key: int | None,
    progress: _Progress,
    **_: Any,
) -> SourcePlan:
    return build_run_plan(
        api,
        source,
        table_keys=table_keys,
        latest_only=latest_only,
        max_steps_per_run_key=max_steps_per_run_key,
        progress=progress,
    )


def _build_sweep_source(
    api: Any,
    source: str,
    *,
    table_keys: set[str] | None,
    latest_only: bool,
    max_steps_per_run_key: int | None,
    max_runs: int,
    run_ids: list[str] | None,
    progress: _Progress,
) -> SourcePlan:
    return build_sweep_plan(
        api,
        source,
        table_keys=table_keys,
        latest_only=latest_only,
        max_steps_per_run_key=max_steps_per_run_key,
        max_runs=max_runs,
        run_ids=run_ids,
        progress=progress,
    )


def _build_workspace_source(
    api: Any,
    source: str,
    *,
    table_keys: set[str] | None,
    latest_only: bool,
    max_steps_per_run_key: int | None,
    max_runs: int,
    run_ids: list[str] | None,
    progress: _Progress,
) -> SourcePlan:
    return build_workspace_plan(
        api,
        source,
        table_keys=table_keys,
        latest_only=latest_only,
        max_steps_per_run_key=max_steps_per_run_key,
        max_runs=max_runs,
        run_ids=run_ids,
        progress=progress,
    )


# The single home for the artifact/run/sweep/workspace taxonomy. resolve_source
# reads project_of / accepts_run_ids / defaults_run_scope; build_source_plan reads
# build. Adding a kind is one entry here — no switch to keep in sync across callers.
_SOURCE_KINDS: dict[str, _SourceKind] = {
    "artifact": _SourceKind(False, False, project_from_artifact_source, _build_artifact_source),
    "run": _SourceKind(False, False, lambda s: parse_run_path(s)[1], _build_run_source),
    "sweep": _SourceKind(True, True, lambda s: parse_sweep_path(s)[1], _build_sweep_source),
    "workspace": _SourceKind(True, True, parse_workspace_path, _build_workspace_source),
}


def build_source_plan(
    api: Any,
    source: str,
    source_kind: str,
    *,
    table_keys: set[str] | None = None,
    latest_only: bool = True,
    max_steps_per_run_key: int | None = DEFAULT_MAX_STEPS_PER_RUN_KEY,
    max_runs: int = DEFAULT_MAX_RUNS,
    run_ids: list[str] | None = None,
    progress: _Progress = _noop_progress,
) -> SourcePlan:
    kind = _SOURCE_KINDS.get(source_kind)
    if kind is None:
        raise ValueError(f"Unsupported source kind: {source_kind}")
    if run_ids is not None and not kind.accepts_run_ids:
        raise ValueError("--run-id is only supported for sweep or workspace sources.")
    return kind.build(
        api,
        source,
        table_keys=table_keys,
        latest_only=latest_only,
        max_steps_per_run_key=max_steps_per_run_key,
        max_runs=max_runs,
        run_ids=run_ids,
        progress=progress,
    )


@dataclass
class ExistingPreviews:
    # Existing EvalTable previews in the target project, read from preview runs'
    # eval_table_preview config. source_paths is per source artifact (key x run x
    # version); table_keys is per key (any run). Detection is per KEY so repeated
    # /convert-eval-table runs converge: once a key has a preview, it's skipped
    # (offer --force) rather than re-offered for every un-previewed source run.
    source_paths: set[str]
    table_keys: set[str]


def parse_exact_retry_source_table(
    full_preview_run_path: str,
    source_table: object,
    table_keys: set[str] | None,
) -> tuple[str, str, int | None] | None:
    if not isinstance(source_table, dict):
        raise TypeError(
            f"{full_preview_run_path} has an invalid source-table manifest entry."
        )
    raw_artifact_path = source_table.get("artifact_path")
    raw_table_key = source_table.get("table_key")
    source_step = source_table.get("source_step")
    if not isinstance(raw_artifact_path, str) or not isinstance(raw_table_key, str):
        raise TypeError(
            f"{full_preview_run_path} has a source-table manifest entry without an "
            "artifact path and table key."
        )
    table_key = normalize_table_key(raw_table_key)
    if table_keys is not None and table_key not in table_keys:
        return None
    if not isinstance(source_step, int) and source_step is not None:
        raise TypeError(
            f"{full_preview_run_path} has a source-table manifest entry with an "
            "invalid source step."
        )
    return parse_artifact_path(raw_artifact_path), table_key, source_step


def build_exact_retry_source_plan(
    api: Any,
    preview_run_paths: list[str],
    *,
    table_keys: set[str] | None,
    progress: _Progress = _noop_progress,
) -> tuple[SourcePlan, str]:
    """Rehydrate the exact source artifacts saved by successful preview runs."""
    table_sources: list[TableSource] = []
    target_projects: set[str] = set()
    seen_sources: set[tuple[str, str | None, str, int | None]] = set()
    resolved_preview_paths: list[str] = []

    for index, preview_run_path in enumerate(dict.fromkeys(preview_run_paths), start=1):
        full_path, _, _ = parse_run_path(preview_run_path)
        progress(
            f"retry manifest {index}/{len(preview_run_paths)}: resolving "
            f"previous preview run {full_path}"
        )
        run = api.run(full_path)
        if not is_preview_run(run):
            raise ValueError(
                f"{full_path} is not an EvalTable preview run created by this helper."
            )
        resolved_preview_paths.append(run_path(run))
        target_projects.add(f"{run.entity}/{run.project}")
        config = dict(getattr(run, "config", None) or {})
        preview = config.get("eval_table_preview") or {}
        source_tables = preview.get("source_tables") or config.get(
            "eval_table_preview.source_tables"
        )
        if not isinstance(source_tables, list) or not source_tables:
            raise ValueError(
                f"{full_path} does not contain the source-table manifest needed "
                "to retry with row-order matching."
            )
        source_run_path = preview.get("source_run_path") or config.get(
            "eval_table_preview.source_run_path"
        )
        source_run_name = preview.get("source_run_name") or config.get(
            "eval_table_preview.source_run_name"
        )
        if not isinstance(source_run_path, str):
            source_run_path = None
        if not isinstance(source_run_name, str):
            source_run_name = None
        source_run_id = (
            source_run_path.rsplit("/", maxsplit=1)[-1]
            if source_run_path is not None
            else None
        )

        for source_table in source_tables:
            parsed_source = parse_exact_retry_source_table(
                full_path, source_table, table_keys
            )
            if parsed_source is None:
                continue
            source_artifact_path, table_key, source_step = parsed_source
            source_identity = (
                source_artifact_path,
                source_run_path,
                table_key,
                source_step,
            )
            if source_identity in seen_sources:
                continue
            seen_sources.add(source_identity)
            progress(
                f"retry manifest {index}/{len(preview_run_paths)}: resolving "
                f"source artifact {source_artifact_path}"
            )
            artifact = api.artifact(source_artifact_path)
            table_sources.append(
                TableSource(
                    artifact=artifact,
                    source_run_path=source_run_path,
                    source_run_name=source_run_name,
                    source_run_id=source_run_id,
                    table_key=table_key,
                    source_step=source_step,
                )
            )

    if not table_sources:
        requested_keys = sorted(table_keys) if table_keys is not None else "any key"
        raise ValueError(
            "The previous preview runs did not contain a source table matching "
            f"{requested_keys}."
        )
    if len(target_projects) != 1:
        raise ValueError(
            "Previous preview runs span multiple destination projects. Pass one "
            "project's preview runs at a time."
        )
    target_project = next(iter(target_projects))
    return (
        SourcePlan(
            source_kind="previous_preview",
            source_label="previous preview manifest",
            table_sources=table_sources,
            scanned_run_count=len(resolved_preview_paths),
            table_selection="exact_previous_preview",
        ),
        target_project,
    )


def existing_previews(
    api: Any,
    target_project: str,
    *,
    progress: _Progress = _noop_progress,
) -> ExistingPreviews:
    source_paths: set[str] = set()
    table_keys: set[str] = set()
    try:
        progress(f"checking existing preview runs in {target_project}")
        # Tags are an array field: match runs that CONTAIN the tag via the Mongo
        # $in operator. A bare {"tags": PREVIEW_TAG} does not match array-contains
        # and silently returns nothing, so previews would never be detected.
        runs = api.runs(
            target_project,
            filters={"tags": {"$in": [PREVIEW_TAG]}},
            order="-created_at",
            per_page=500,
        )
        preview_run_count = 0
        for run in runs:
            preview_run_count += 1
            config = getattr(run, "config", {}) or {}
            preview = config.get("eval_table_preview") or {}
            source_artifacts = (
                preview.get("source_artifacts")
                or config.get("eval_table_preview.source_artifacts")
                or []
            )
            for value in source_artifacts:
                if isinstance(value, str):
                    source_paths.add(value)
            keys = (
                preview.get("source_table_keys")
                or config.get("eval_table_preview.source_table_keys")
                or []
            )
            for value in keys:
                if isinstance(value, str):
                    table_keys.add(value)
        progress(
            "existing preview check complete: "
            f"{preview_run_count} preview run(s), "
            f"{len(source_paths)} source artifact(s), {len(table_keys)} table key(s)"
        )
    except Exception as e:
        print(
            f"warning: could not inspect existing previews: {e}",
            file=sys.stderr,
            flush=True,
        )
        progress("existing preview check failed; continuing without preview metadata")
    return ExistingPreviews(source_paths=source_paths, table_keys=table_keys)


def scan_source_plan(
    source_plan: SourcePlan,
    *,
    target_project: str,
    table_keys: set[str] | None,
    previewed: ExistingPreviews,
    download_root: Path,
    max_rows: int,
    max_columns: int,
    max_tables: int,
    progress: _Progress = _noop_progress,
) -> ScanPlan:
    tables: list[ScannedTable] = []
    root = download_root / safe_path_name(source_plan.source_label)
    total_sources = len(source_plan.table_sources)
    progress(
        f"scanning {total_sources} table artifact candidate(s) "
        f"for target project {target_project}"
    )
    for index, table_source in enumerate(source_plan.table_sources, start=1):
        artifact = table_source.artifact
        source_artifact_path = artifact_path(artifact)
        progress(f"scan {index}/{total_sources}: inspecting {source_artifact_path}")
        artifact_size_bytes, artifact_size_source = artifact_size_details(artifact)
        if artifact_size_bytes is None:
            progress(
                f"scan {index}/{total_sources}: artifact size unknown for "
                f"{source_artifact_path}"
            )
        else:
            progress(
                f"scan {index}/{total_sources}: artifact size "
                f"{format_bytes(artifact_size_bytes)} via {artifact_size_source}"
            )
        # Exact retry manifests preserve their key; all other sources derive it
        # from the artifact name. The downloaded JSON filename is only consulted
        # in the no-key fallback below.
        table_key = table_source.table_key or table_key_from_artifact_name(
            artifact, table_source.source_run_id, table_keys
        )
        if table_key is not None and table_keys is not None and table_key not in table_keys:
            progress(
                f"scan {index}/{total_sources}: skipping {source_artifact_path}; "
                f"table key {table_key!r} was not requested"
            )
            continue

        row_count = table_source.row_count
        column_count = table_source.column_count
        table_json_path: Path | None = None
        columns: list[str] | None = None
        # Per-key detection (plus the per-artifact fallback for the rare case
        # where the key can't be read from the artifact name): once a key has any
        # preview, its artifacts are treated as already-previewed so the plan
        # converges instead of re-offering the key for every un-previewed run.
        already_previewed = source_artifact_path in previewed.source_paths or (
            table_key is not None and table_key in previewed.table_keys
        )

        # Fast path: shape came from history. Download only when the key is unknown,
        # the shape is unavailable (artifact source, incremental, missing history),
        # or the column count is over the cap. The last case downloads purely to name
        # the dropped columns for the pre-preview warning, so skip it for tables we
        # won't preview anyway (already previewed) — no reason to materialize them.
        needs_download = (
            table_key is None
            or row_count is None
            or column_count is None
            or (column_count > max_columns and not already_previewed)
        )
        if needs_download:
            download_dir = root / safe_path_name(source_artifact_path)
            download_dir.mkdir(parents=True, exist_ok=True)
            progress(
                f"scan {index}/{total_sources}: downloading {source_artifact_path} "
                "to inspect table shape"
            )
            artifact.download(root=str(download_dir))
            table_json_path = find_table_json(download_dir)
            progress(
                f"scan {index}/{total_sources}: reading table JSON "
                f"{table_json_path}"
            )
            with table_json_path.open(encoding="utf-8") as f:
                table_json = json.load(f)
            if table_key is None:
                table_key = normalize_table_key(default_log_key(table_json_path))
                if table_keys is not None and table_key not in table_keys:
                    progress(
                        f"scan {index}/{total_sources}: skipping "
                        f"{source_artifact_path}; downloaded table key "
                        f"{table_key!r} was not requested"
                    )
                    continue
            row_count, column_count = table_shape(table_json)
            raw_columns = table_json.get("columns")
            columns = raw_columns if isinstance(raw_columns, list) else None

        truncation = size_truncation(
            row_count, column_count, columns, max_rows, max_columns
        )
        warnings = size_warnings(row_count)
        # Oversized tables are truncated, not skipped, so they stay eligible; the
        # truncation plan rides along so the agent can warn the user first.
        if already_previewed:
            status = "already_previewed"
            reasons = ["source artifact already previewed"]
        else:
            status = "eligible"
            reasons = []
        source_step = (
            table_source.source_step
            if table_source.source_step is not None
            else artifact.history_step
        )
        progress(
            f"scan {index}/{total_sources}: {status} key={table_key!r} "
            f"rows={row_count} columns={column_count} "
            f"step={source_step}"
        )
        tables.append(
            ScannedTable(
                source=table_source,
                artifact_path=source_artifact_path,
                table_key=table_key,
                source_run_path=table_source.source_run_path,
                source_run_name=table_source.source_run_name,
                source_step=source_step,
                row_count=row_count,
                column_count=column_count,
                status=status,
                reasons=reasons,
                warnings=warnings,
                artifact_size_bytes=artifact_size_bytes,
                artifact_size_source=artifact_size_source,
                truncation=truncation,
                table_json_path=table_json_path,
            )
        )
    eligible_count = sum(1 for table in tables if table.status == "eligible")
    progress(
        f"scan complete: {len(tables)} table artifact(s), "
        f"{eligible_count} eligible"
    )
    return ScanPlan(
        source_kind=source_plan.source_kind,
        source_label=source_plan.source_label,
        target_project=target_project,
        scanned_run_count=source_plan.scanned_run_count,
        tables=tables,
        max_rows=max_rows,
        max_columns=max_columns,
        max_tables=max_tables,
        table_selection=source_plan.table_selection,
        max_steps_per_run_key=source_plan.max_steps_per_run_key,
        download_root=download_root,
    )


def table_sources_by_destination_run(
    tables: list[ScannedTable],
) -> list[tuple[str, str | None, str | None, list[ScannedTable]]]:
    grouped: dict[str, tuple[str | None, str | None, list[ScannedTable]]] = {}
    for table in tables:
        group_key = (
            table.source_run_path
            or table.source.source_run_id
            or table.artifact_path
        )
        if group_key not in grouped:
            grouped[group_key] = (table.source_run_path, table.source_run_name, [])
        grouped[group_key][2].append(table)
    return [
        (group_key, source_run_path, source_run_name, group_tables)
        for group_key, (source_run_path, source_run_name, group_tables) in grouped.items()
    ]


def maybe_detach_artifact_sources(value: Any) -> None:
    if hasattr(value, "_artifact_source"):
        value._artifact_source = None
    if isinstance(value, dict):
        for item in value.values():
            maybe_detach_artifact_sources(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            maybe_detach_artifact_sources(item)


def detach_table_artifact_sources(table: Any) -> None:
    maybe_detach_artifact_sources(table)
    for row in table.data:
        maybe_detach_artifact_sources(row)


def find_duplicate_columns(columns: list[str]) -> list[str]:
    seen = set()
    duplicates = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def resolve_eval_column_groups(
    table_columns: list[str],
    input_columns: list[str] | None,
    output_columns: list[str] | None,
    score_columns: list[str] | None,
    unspecified_columns_as_output: bool,
    unspecified_column_confirmation: UnspecifiedColumnConfirmation,
) -> tuple[list[str], list[str], list[str]]:
    if input_columns is None and output_columns is None and score_columns is None:
        return [], [], []

    table_column_set = set(table_columns)
    resolved_input_columns = [
        column for column in input_columns or [] if column in table_column_set
    ]
    resolved_output_columns = [
        column for column in output_columns or [] if column in table_column_set
    ]
    resolved_score_columns = [
        column for column in score_columns or [] if column in table_column_set
    ]
    specified_columns = (
        resolved_input_columns + resolved_output_columns + resolved_score_columns
    )

    duplicate_columns = find_duplicate_columns(specified_columns)
    if duplicate_columns:
        raise ValueError(
            "Columns can only be assigned to one EvalTable group. "
            f"Duplicate columns: {duplicate_columns}"
        )

    unspecified_columns = [
        column for column in table_columns if column not in specified_columns
    ]
    if unspecified_columns:
        new_columns = [
            column
            for column in unspecified_columns
            if column not in unspecified_column_confirmation.accepted_columns
        ]
        if new_columns and not unspecified_columns_as_output:
            raise ValueError(
                "Unspecified columns require an explicit grouping choice because "
                "--no-unspecified-columns-as-output was passed. Re-run without "
                "that flag or pass all column groups."
            )
        unspecified_column_confirmation.accepted_columns.update(unspecified_columns)
        resolved_output_columns.extend(unspecified_columns)

    return resolved_input_columns, resolved_output_columns, resolved_score_columns


def validate_score_column_types(
    columns: list[str], data: list[list[Any]], score_columns: list[str]
) -> None:
    """Score columns aggregate to mean / fraction-true, so their values must be
    numeric or boolean. Reject any other type up front with a clear message rather
    than logging an EvalTable whose score aggregates are meaningless or rejected.
    None (missing) is allowed; bool is a valid numeric in Python."""
    if not score_columns:
        return
    index = {column: idx for idx, column in enumerate(columns)}
    for column in score_columns:
        idx = index[column]
        for row in data:
            value = row[idx]
            if value is not None and not isinstance(value, (bool, int, float)):
                raise ValueError(
                    f"Score column '{column}' holds a {type(value).__name__} value; "
                    "score columns must be numeric or boolean (they aggregate to "
                    "mean / fraction-true). Assign it as an input/output column "
                    "instead, or leave the table untyped."
                )


def require_eval_table_cls(wandb: Any) -> Any:
    try:
        return wandb.EvalTable
    except (AttributeError, ImportError) as e:
        raise RuntimeError(
            "This wandb install does not expose wandb.EvalTable. "
            "Run with a wandb build that includes the EvalTables API."
        ) from e


def build_eval_table(
    wandb: Any,
    source_table: Any,
    input_columns: list[str] | None,
    output_columns: list[str] | None,
    score_columns: list[str] | None,
    unspecified_columns_as_output: bool,
    unspecified_column_confirmation: UnspecifiedColumnConfirmation,
) -> Any:
    eval_table_cls = require_eval_table_cls(wandb)
    if input_columns is None and output_columns is None and score_columns is None:
        return eval_table_cls(columns=source_table.columns, data=source_table.data)

    resolved_input_columns, resolved_output_columns, resolved_score_columns = (
        resolve_eval_column_groups(
            table_columns=source_table.columns,
            input_columns=input_columns,
            output_columns=output_columns,
            score_columns=score_columns,
            unspecified_columns_as_output=unspecified_columns_as_output,
            unspecified_column_confirmation=unspecified_column_confirmation,
        )
    )
    validate_score_column_types(
        source_table.columns, source_table.data, resolved_score_columns
    )
    grouped_columns = (
        resolved_input_columns + resolved_output_columns + resolved_score_columns
    )
    column_indices = {column: idx for idx, column in enumerate(source_table.columns)}
    reordered_data = [
        [row[column_indices[column]] for column in grouped_columns]
        for row in source_table.data
    ]

    return eval_table_cls(
        input_columns=resolved_input_columns,
        output_columns=resolved_output_columns,
        score_columns=resolved_score_columns,
        data=reordered_data,
    )


def normalize_media_list_cells(source_table: Any) -> None:
    """Unwrap list-of-media cells in place so EvalTable can log them.

    TODO(sdk>0.28.0): remove once wandb.EvalTable converts list-of-media cells
    natively. Until then its media_adapters.unwrap_value converts only a single
    media value per cell; a list of media falls through unconverted and fails to
    log. We unwrap each media element (matching the per-element guidance the
    reference used to hand off to the agent), leaving non-media elements
    untouched. No-ops on wandb builds without media_adapters (the current public
    SDK), so it is inert until the runtime ships the API.
    """
    try:
        from wandb.integration.weave.media_adapters import unwrap_value
        from wandb.sdk.data_types.base_types.media import Media
    except Exception:
        return

    columns = source_table.columns
    for row in source_table.data:
        for idx, value in enumerate(row):
            if not isinstance(value, list) or not any(
                isinstance(element, Media) for element in value
            ):
                continue
            column = columns[idx] if idx < len(columns) else ""
            # Only unwrap the media elements; unwrap_value stubs non-media values,
            # so passing through plain elements avoids corrupting mixed lists.
            row[idx] = [
                unwrap_value(element, column, unsupported_media_mode="stub")
                if isinstance(element, Media)
                else element
                for element in value
            ]


def import_scanned_table(
    wandb: Any,
    table: ScannedTable,
    download_root: Path,
    input_columns: list[str] | None,
    output_columns: list[str] | None,
    score_columns: list[str] | None,
    unspecified_columns_as_output: bool,
    unspecified_column_confirmation: UnspecifiedColumnConfirmation,
    eval_table_key_suffix: str = DEFAULT_EVAL_TABLE_KEY_SUFFIX,
    max_rows: int = MAX_ROWS,
    max_columns: int = MAX_COLUMNS,
    progress: _Progress = _noop_progress,
) -> ImportedTable:
    progress(
        f"importing table artifact {table.artifact_path} "
        f"(key={table.table_key!r}, step={table.source_step})"
    )
    # Download lazily here (scan no longer downloads); reuse the scan-time download
    # if the size fallback already fetched it.
    table_json_path = table.table_json_path
    download_dir: Path | None = None
    if table_json_path is None:
        download_dir = download_root / safe_path_name(table.artifact_path)
        download_dir.mkdir(parents=True, exist_ok=True)
        progress(f"downloading table artifact {table.artifact_path}")
        table.source.artifact.download(root=str(download_dir))
        table_json_path = find_table_json(download_dir)
    else:
        download_dir = table_json_path.parent
    progress(f"reading table JSON {table_json_path}")
    with table_json_path.open(encoding="utf-8") as f:
        table_json = json.load(f)
    row_count, column_count = table_shape(table_json)
    progress(
        f"loaded table JSON for {table.table_key!r}: "
        f"{row_count} row(s), {column_count} column(s)"
    )
    source_table = wandb.Table.from_json(table_json, table.source.artifact)
    detach_table_artifact_sources(source_table)
    # Truncate oversized tables to the caps rather than refusing them; the scan
    # already surfaced what gets dropped so the user could confirm. Preserve any
    # column the agent explicitly assigned a role so column truncation never drops
    # a requested input/output/score column.
    keep_columns = {
        column
        for group in (input_columns, output_columns, score_columns)
        if group
        for column in group
    }
    truncation = truncate_source_table(
        source_table, max_rows, max_columns, keep_columns
    )
    if truncation is not None:
        progress(
            f"truncated table {table.table_key!r}: "
            f"{json.dumps(truncation, sort_keys=True)}"
        )
    progress(f"normalizing media cells for {table.table_key!r}")
    normalize_media_list_cells(source_table)
    progress(f"building EvalTable for {table.table_key!r}")
    eval_table = build_eval_table(
        wandb=wandb,
        source_table=source_table,
        input_columns=input_columns,
        output_columns=output_columns,
        score_columns=score_columns,
        unspecified_columns_as_output=unspecified_columns_as_output,
        unspecified_column_confirmation=unspecified_column_confirmation,
    )
    log_key = f"{table.table_key}{eval_table_key_suffix}"
    imported = ImportedTable(
        artifact_path=table.artifact_path,
        eval_table=eval_table,
        log_key=log_key,
        source_step=table.source_step,
        estimated_upload_bytes=(
            table.artifact_size_bytes
            if table.artifact_size_bytes is not None
            else directory_size_bytes(download_dir)
        ),
    )
    progress(f"imported {table.table_key!r} as {log_key}")
    return imported


def log_imported_tables(
    run: Any,
    imported_tables: list[ImportedTable],
    *,
    progress: _Progress = _noop_progress,
    transfer_started_at: float | None = None,
    queued_before_bytes: int = 0,
    total_known_bytes: int | None = None,
    total_unknown_count: int = 0,
) -> int:
    imported_tables = sorted(
        imported_tables,
        key=lambda table: sys.maxsize if table.source_step is None else table.source_step,
    )
    queued_bytes = queued_before_bytes
    for index, imported in enumerate(imported_tables, start=1):
        progress(
            f"logging EvalTable {index}/{len(imported_tables)}: "
            f"{imported.log_key}"
        )
        run.log({imported.log_key: imported.eval_table})
        if imported.estimated_upload_bytes is not None:
            queued_bytes += imported.estimated_upload_bytes
        progress(
            f"logged EvalTable {index}/{len(imported_tables)}: "
            f"{imported.log_key}"
        )
        if transfer_started_at is not None:
            elapsed = time.monotonic() - transfer_started_at
            total_text = format_bytes(total_known_bytes)
            unknown_text = (
                f"; {total_unknown_count} artifact(s) unknown"
                if total_unknown_count
                else ""
            )
            progress(
                "estimated upload payload logged so far: "
                f"{format_bytes(queued_bytes)} / {total_text} known"
                f"{unknown_text}; effective {format_mbps(queued_bytes, elapsed)}"
            )
    return queued_bytes


def destination_run_name(
    source_run_name: str | None,
    group_key: str,
    preview_run_suffix: str = DEFAULT_PREVIEW_RUN_SUFFIX,
) -> str:
    base = source_run_name or group_key.rsplit("/", maxsplit=1)[-1]
    base_length = MAX_RUN_NAME_LENGTH - len(preview_run_suffix)
    return f"{base[:base_length]}{preview_run_suffix}"


def preview_scan_plan(
    wandb: Any,
    scan_plan: ScanPlan,
    *,
    dry_run: bool,
    input_columns: list[str] | None,
    output_columns: list[str] | None,
    score_columns: list[str] | None,
    unspecified_columns_as_output: bool,
    no_column_map: bool = False,
    row_index_match: bool = False,
    progress: _Progress = _noop_progress,
) -> dict[str, Any]:
    if no_column_map and row_index_match:
        raise ValueError("Pass either --no-column-map or --row-index-match, not both.")
    if no_column_map and any((input_columns, output_columns, score_columns)):
        raise ValueError(
            "--no-column-map cannot be combined with input, output, or score columns"
        )
    if row_index_match and input_columns:
        raise ValueError(
            "--row-index-match cannot be combined with --input-columns; "
            "pass score or output columns only."
        )
    uses_row_index_match_name = no_column_map or row_index_match
    eval_table_key_suffix = (
        ROW_INDEX_MATCH_EVAL_TABLE_KEY_SUFFIX
        if uses_row_index_match_name
        else DEFAULT_EVAL_TABLE_KEY_SUFFIX
    )
    preview_run_suffix = (
        ROW_INDEX_MATCH_PREVIEW_RUN_SUFFIX
        if uses_row_index_match_name
        else DEFAULT_PREVIEW_RUN_SUFFIX
    )
    eligible_tables = scan_plan.eligible_tables
    progress(
        f"preview plan has {len(eligible_tables)} eligible table artifact(s); "
        f"dry_run={dry_run}"
    )
    eligible_known_bytes, eligible_unknown_count = artifact_size_summary(eligible_tables)
    progress(
        "eligible source artifact size: "
        f"{format_bytes(eligible_known_bytes)} known across "
        f"{len(eligible_tables) - eligible_unknown_count} artifact(s); "
        f"{eligible_unknown_count} artifact(s) unknown"
    )
    # Preview converts one table key per call (main enforces --table-key), so the
    # eligible tables here are all one key's artifacts across runs/steps. Assert it
    # so the batch cap below is genuinely per-key, not an aggregate across keys.
    distinct_keys = sorted({table.table_key for table in eligible_tables})
    if len(distinct_keys) > 1:
        raise ValueError(
            "preview converts one table key per call, but the eligible tables span "
            f"multiple keys: {distinct_keys}. Filter the scan to a single "
            "--table-key and preview each key in its own call."
        )
    if len(eligible_tables) > scan_plan.max_tables:
        raise ValueError(
            f"Refusing to preview {len(eligible_tables)} eligible table artifacts "
            f"for key '{distinct_keys[0]}' in one batch; limit is "
            f"{scan_plan.max_tables} per key. Narrow --run-id, use "
            "--max-steps-per-run-key for recent steps, or run multiple batches."
        )
    if not eligible_tables:
        progress("preview skipped: no eligible tables")
        return {
            "dry_run": dry_run,
            "previewed_table_count": 0,
            "previewed_run_count": 0,
            "created_runs": [],
            "message": "No eligible tables to preview.",
            "scan": scan_plan.to_dict(),
        }
    if dry_run:
        progress("preview dry run complete: no W&B runs will be created")
        return {
            "dry_run": True,
            "previewed_table_count": 0,
            "previewed_run_count": 0,
            "created_runs": [],
            "message": "Dry run only; no W&B runs were created.",
            "scan": scan_plan.to_dict(),
        }

    target_entity, target_project = split_project_path(scan_plan.target_project)
    download_root = scan_plan.download_root or DOWNLOAD_ROOT
    unspecified_column_confirmation = UnspecifiedColumnConfirmation(
        accepted_columns=set()
    )
    created_runs = []
    failed_runs = []
    previewed_table_count = 0
    transfer_started_at = time.monotonic()
    logged_payload_bytes = 0
    # Isolate failures per destination run: a download/log error on one source run
    # shouldn't abandon the runs that haven't been previewed yet. (Failures are
    # also isolated per key at the CLI level, since preview takes one --table-key.)
    groups = table_sources_by_destination_run(eligible_tables)
    progress(
        f"previewing {len(eligible_tables)} table artifact(s) into "
        f"{len(groups)} destination run(s)"
    )
    for group_index, (group_key, source_run_path, source_run_name, tables) in enumerate(
        groups, start=1
    ):
        source_label = source_run_path or group_key
        try:
            progress(
                f"destination run {group_index}/{len(groups)} for {source_label}: "
                f"importing {len(tables)} table artifact(s)"
            )
            imported_tables = []
            for table_index, table in enumerate(tables, start=1):
                progress(
                    f"destination run {group_index}/{len(groups)}: "
                    f"importing table {table_index}/{len(tables)}"
                )
                imported_tables.append(
                    import_scanned_table(
                        wandb=wandb,
                        table=table,
                        download_root=download_root,
                        input_columns=input_columns,
                        output_columns=output_columns,
                        score_columns=score_columns,
                        unspecified_columns_as_output=unspecified_columns_as_output,
                        unspecified_column_confirmation=unspecified_column_confirmation,
                        eval_table_key_suffix=eval_table_key_suffix,
                        max_rows=scan_plan.max_rows,
                        max_columns=scan_plan.max_columns,
                        progress=progress,
                    )
                )
            destination_name = destination_run_name(
                source_run_name, group_key, preview_run_suffix
            )
            config = {
                "eval_table_preview": {
                    "source_run_path": source_run_path,
                    "source_run_name": source_run_name,
                    "source_artifacts": [table.artifact_path for table in tables],
                    "source_table_keys": sorted({table.table_key for table in tables}),
                    "source_steps": {
                        table.artifact_path: table.source_step for table in tables
                    },
                    "source_tables": [
                        {
                            "artifact_path": table.artifact_path,
                            "table_key": table.table_key,
                            "source_step": table.source_step,
                        }
                        for table in tables
                    ],
                }
            }
            progress(
                f"initializing preview run {group_index}/{len(groups)}: "
                f"{destination_name!r}"
            )
            init_kwargs = {
                "entity": target_entity,
                "project": target_project,
                "job_type": PREVIEW_TAG,
                "tags": [PREVIEW_TAG],
                "config": config,
                "name": destination_name,
            }
            settings = wandb_console_settings(wandb)
            if settings is not None:
                init_kwargs["settings"] = settings
            with wandb.init(
                **init_kwargs,
            ) as run:
                run_path_value = f"{run.entity}/{run.project}/{run.id}"
                progress(
                    f"logging {len(imported_tables)} EvalTable(s) to "
                    f"{run_path_value}"
                )
                logged_payload_bytes = log_imported_tables(
                    run,
                    imported_tables,
                    progress=progress,
                    transfer_started_at=transfer_started_at,
                    queued_before_bytes=logged_payload_bytes,
                    total_known_bytes=eligible_known_bytes,
                    total_unknown_count=eligible_unknown_count,
                )
                run.summary["eval_table_preview/source_run_path"] = source_run_path
                run.summary["eval_table_preview/source_artifact_count"] = len(tables)
                run.summary["eval_table_preview/source_table_count"] = len(
                    {table.table_key for table in tables}
                )
                created_runs.append(run_path_value)
                previewed_table_count += len(tables)
                progress(
                    "waiting for W&B SDK finish/sync (uploading run history and "
                    f"logged EvalTable media/artifacts) for {run_path_value}: "
                    f"{len(imported_tables)} EvalTable(s), "
                    f"~{format_bytes(logged_payload_bytes)} of logged payload "
                    "queued; large tables/media can make this slow"
                )
            progress(f"W&B SDK finish/sync complete for {run_path_value}")
        except Exception as e:
            failed_runs.append(
                {
                    "source_run_path": source_run_path,
                    "source_run_name": source_run_name,
                    "table_keys": sorted({table.table_key for table in tables}),
                    "error": str(e),
                }
            )
            print(
                f"warning: preview failed for {source_run_path or group_key}: {e}",
                file=sys.stderr,
                flush=True,
            )
            progress(f"preview failed for {source_label}: {e}")

    progress(
        f"preview complete: {previewed_table_count} table artifact(s), "
        f"{len(created_runs)} run(s), {len(failed_runs)} failure(s)"
    )
    return {
        "dry_run": False,
        "previewed_table_count": previewed_table_count,
        "previewed_run_count": len(created_runs),
        "created_runs": created_runs,
        "failed_runs": failed_runs,
        "scan": scan_plan.to_dict(),
    }


def is_preview_run(run: Any) -> bool:
    """A run is safe to delete only if this helper created it — i.e. it carries
    the eval-table-preview tag or the eval_table_preview config block. Guards
    delete-preview from ever touching a user's real training/eval run."""
    tags = list(getattr(run, "tags", None) or [])
    config = dict(getattr(run, "config", None) or {})
    return PREVIEW_TAG in tags or bool(config.get("eval_table_preview"))


def eval_table_call_entries(run: Any) -> list[dict[str, Any]]:
    """Scan a preview run's history for the EvalTables it logged and return the
    weave evaluate_call_id recorded with each. An EvalTable logs a media value
    of ``{"_type": "eval-table", "evaluate_call_id": ...}`` under its
    preview-specific key; that call id is what deletes the backing weave
    evaluation reliably (by id, not by name)."""
    entries: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for row in run.scan_history():
        for key, value in row.items():
            if not isinstance(value, dict) or value.get("_type") != "eval-table":
                continue
            call_id = value.get("evaluate_call_id")
            if not call_id or call_id in seen_call_ids:
                continue
            seen_call_ids.add(call_id)
            entries.append(
                {
                    "key": key,
                    "evaluate_call_id": call_id,
                    "step": row.get("_step"),
                }
            )
    return entries


def weave_call_exists(client: Any, call_id: str) -> tuple[bool, str | None]:
    get_call = getattr(client, "get_call", None)
    if not callable(get_call):
        return False, "weave client does not expose get_call"
    try:
        return get_call(call_id) is not None, None
    except Exception as e:
        return False, str(e)


def verify_preview_runs(
    wandb: Any,
    run_paths: list[str],
    *,
    table_keys: set[str],
    progress: _Progress = _noop_progress,
) -> dict[str, Any]:
    """Verify preview EvalTables deterministically.

    Verification requires both signals: the preview run history contains
    eval-table media entries for the requested key(s), and each recorded
    evaluate_call_id resolves in Weave.
    """
    progress(
        f"verify-preview checking {len(run_paths)} run(s) for "
        f"{sorted(table_keys)}"
    )
    api = wandb.Api()
    progress("importing weave")
    import weave

    results: list[dict[str, Any]] = []
    for index, path in enumerate(run_paths, start=1):
        full_path, _, _ = parse_run_path(path)
        progress(f"verify-preview {index}/{len(run_paths)}: resolving {full_path}")
        result: dict[str, Any] = {
            "run": full_path,
            "requested_table_keys": sorted(table_keys),
        }
        try:
            run = api.run(full_path)
        except Exception as e:
            result.update({"status": "not_found", "error": str(e)})
            results.append(result)
            continue

        result["is_preview_run"] = is_preview_run(run)
        if not result["is_preview_run"]:
            result.update(
                {
                    "status": "skipped_not_preview",
                    "reason": (
                        "run is not tagged eval-table-preview and has no "
                        "eval_table_preview config"
                    ),
                    "history_entries": [],
                    "found_weave_call_ids": [],
                    "missing_weave_call_ids": [],
                }
            )
            results.append(result)
            continue

        entries = [
            entry
            for entry in eval_table_call_entries(run)
            if entry["key"] in table_keys
        ]
        result["history_entries"] = entries
        result["history_entry_count"] = len(entries)
        if not entries:
            result.update(
                {
                    "status": "missing_history",
                    "found_weave_call_ids": [],
                    "missing_weave_call_ids": [],
                    "reason": "no eval-table history entries found for requested keys",
                }
            )
            results.append(result)
            continue

        try:
            client = weave.init(f"{run.entity}/{run.project}")
        except Exception as e:
            call_ids = [entry["evaluate_call_id"] for entry in entries]
            result.update(
                {
                    "status": "weave_error",
                    "found_weave_call_ids": [],
                    "missing_weave_call_ids": call_ids,
                    "weave_errors": {call_id: str(e) for call_id in call_ids},
                }
            )
            results.append(result)
            continue

        found_call_ids = []
        missing_call_ids = []
        weave_errors: dict[str, str] = {}
        for entry in entries:
            call_id = entry["evaluate_call_id"]
            exists, error = weave_call_exists(client, call_id)
            if exists:
                found_call_ids.append(call_id)
            else:
                missing_call_ids.append(call_id)
                if error:
                    weave_errors[call_id] = error

        result.update(
            {
                "status": "verified" if not missing_call_ids else "missing_weave_calls",
                "found_weave_call_ids": found_call_ids,
                "missing_weave_call_ids": missing_call_ids,
            }
        )
        if weave_errors:
            result["weave_errors"] = weave_errors
        results.append(result)

    verified_runs = [result for result in results if result["status"] == "verified"]
    verified_call_count = sum(
        len(result.get("found_weave_call_ids", [])) for result in verified_runs
    )
    return {
        "verified": bool(results) and len(verified_runs) == len(results),
        "run_count": len(results),
        "verified_run_count": len(verified_runs),
        "verified_call_count": verified_call_count,
        "requested_table_keys": sorted(table_keys),
        "runs": results,
    }


def delete_preview_runs(
    wandb: Any,
    run_paths: list[str],
    *,
    dry_run: bool,
    delete_artifacts: bool,
    progress: _Progress = _noop_progress,
) -> dict[str, Any]:
    """Undo previews created by this helper: for each preview run, delete the
    weave evaluation(s) behind its EvalTable(s) by call id, then delete the run.
    Non-preview runs are refused, not deleted. Failures are isolated per run."""
    progress(
        f"delete-preview inspecting {len(run_paths)} run(s); dry_run={dry_run}"
    )
    progress("initializing W&B API client")
    api = wandb.Api()
    progress("importing weave")
    import weave

    results: list[dict[str, Any]] = []
    for index, path in enumerate(run_paths, start=1):
        full_path, _, _ = parse_run_path(path)
        progress(f"delete-preview {index}/{len(run_paths)}: resolving {full_path}")
        try:
            run = api.run(full_path)
        except Exception as e:
            progress(
                f"delete-preview {index}/{len(run_paths)}: {full_path} not found"
            )
            results.append({"run": path, "status": "not_found", "error": str(e)})
            continue

        if not is_preview_run(run):
            progress(
                f"delete-preview {index}/{len(run_paths)}: skipping "
                f"{full_path}; not a preview run"
            )
            results.append(
                {
                    "run": full_path,
                    "status": "skipped_not_preview",
                    "reason": (
                        "run is not tagged eval-table-preview and has no "
                        "eval_table_preview config; refusing to delete a run this "
                        "helper did not create"
                    ),
                }
            )
            continue

        call_entries = eval_table_call_entries(run)
        call_ids = [entry["evaluate_call_id"] for entry in call_entries]
        preview_keys = sorted({entry["key"] for entry in call_entries})
        progress(
            f"delete-preview {index}/{len(run_paths)}: found {len(call_ids)} "
            f"EvalTable call id(s) in {full_path}"
        )
        if dry_run:
            results.append(
                {
                    "run": full_path,
                    "status": "would_delete",
                    "preview_keys": preview_keys,
                    "evaluate_call_ids": call_ids,
                }
            )
            continue

        try:
            if call_ids:
                # weave.init scopes the client to this run's project so the call
                # ids resolve; deleting a call also deletes its children.
                progress(
                    f"delete-preview {index}/{len(run_paths)}: deleting "
                    f"{len(call_ids)} weave call(s) from {run.entity}/{run.project}"
                )
                client = weave.init(f"{run.entity}/{run.project}")
                client.delete_calls(call_ids)
            progress(
                f"delete-preview {index}/{len(run_paths)}: deleting run {full_path}"
            )
            run.delete(delete_artifacts=delete_artifacts)
            results.append(
                {
                    "run": full_path,
                    "status": "deleted",
                    "preview_keys": preview_keys,
                    "deleted_evaluate_call_ids": call_ids,
                }
            )
        except Exception as e:
            results.append({"run": full_path, "status": "failed", "error": str(e)})
            print(
                f"warning: delete failed for {full_path}: {e}",
                file=sys.stderr,
                flush=True,
            )
            progress(f"delete-preview failed for {full_path}: {e}")

    deleted = [r for r in results if r["status"] == "deleted"]
    progress(
        f"delete-preview complete: {len(deleted)} run(s), "
        f"{sum(len(r.get('deleted_evaluate_call_ids', [])) for r in deleted)} "
        "weave call(s) deleted"
    )
    return {
        "dry_run": dry_run,
        "deleted_run_count": len(deleted),
        "deleted_evaluate_call_count": sum(
            len(r.get("deleted_evaluate_call_ids", [])) for r in deleted
        ),
        "results": results,
    }


@dataclass(frozen=True)
class ResolvedSource:
    """What the user pointed at, normalized from the CLI flags before any Api call.

    Pure output of resolve_source — no network, no stdin. Carries the two facts the
    caller needs beyond the SourcePlan (the resolved target_project and the run
    selection) plus whether a defaulted-runs (run_scope) disclosure is owed.
    """

    source_kind: str
    source: str
    target_project: str
    run_ids: list[str] | None
    run_scope_defaulted: bool


def _select_source_and_kind(args: argparse.Namespace) -> tuple[str, str]:
    """Pick the one source the flags name (or the positional source) plus its kind,
    enforcing the mutually-exclusive flag styles."""
    has_run_flags = args.source_project is not None or args.source_run is not None
    source_flag_count = sum(
        [
            bool(has_run_flags),
            args.source_sweep is not None,
            args.source_workspace is not None,
        ]
    )
    if source_flag_count > 1:
        raise ValueError("Pass only one source flag style.")

    if has_run_flags:
        if args.source_project is None or args.source_run is None:
            raise ValueError("--source-project and --source-run must be passed together.")
        return f"{args.source_project}/{args.source_run}", "run"
    if args.source_sweep is not None:
        return args.source_sweep, "sweep"
    if args.source_workspace is not None:
        return args.source_workspace, "workspace"
    if args.source is None:
        raise ValueError("Pass a source or source flags.")
    return args.source, (args.source_kind or infer_source_kind(args.source))


def resolve_source(args: argparse.Namespace) -> ResolvedSource:
    """Interpret the source flags into a normalized ResolvedSource. Pure — no Api
    call, no stdin. The single CLI-side owner of the artifact/run/sweep/workspace
    taxonomy: flag-style selection, kind + target-project inference, and the
    defaulted-runs decision — every per-kind fact read from _SOURCE_KINDS.
    """
    source, source_kind = _select_source_and_kind(args)
    kind = _SOURCE_KINDS[source_kind]
    target_project = args.target_project or kind.project_of(source)
    if target_project is None:
        raise ValueError(
            "Could not infer the destination project. Pass --target-project."
        )
    run_ids = parse_run_ids(args.run_id)
    return ResolvedSource(
        source_kind=source_kind,
        source=source,
        target_project=target_project,
        run_ids=run_ids,
        run_scope_defaulted=kind.defaults_run_scope and run_ids is None,
    )


def build_scan_from_args(
    args: argparse.Namespace,
    wandb: Any,
    *,
    progress: _Progress = _noop_progress,
) -> ScanPlan:
    retry_preview_runs = parse_column_args(
        getattr(args, "from_preview_run", None), "--from-preview-run"
    )
    if retry_preview_runs is not None:
        has_standard_source = any(
            (
                args.source is not None,
                args.source_kind is not None,
                args.source_project is not None,
                args.source_run is not None,
                args.source_sweep is not None,
                args.source_workspace is not None,
                args.run_id is not None,
            )
        )
        if has_standard_source:
            raise ValueError(
                "--from-preview-run cannot be combined with a source or --run-id."
            )
        if not args.force:
            raise ValueError("--from-preview-run requires --force.")
        progress("initializing W&B API client")
        api = wandb.Api()
        table_keys = parse_table_keys(args.table_key)
        progress(
            "reusing the exact source-table manifest from "
            f"{len(retry_preview_runs)} previous preview run(s)"
        )
        source_plan, prior_target_project = build_exact_retry_source_plan(
            api,
            retry_preview_runs,
            table_keys=table_keys,
            progress=progress,
        )
        target_project = args.target_project or prior_target_project
        progress(
            f"retry source plan complete: table_sources={len(source_plan.table_sources)}, "
            f"target={target_project}"
        )
        scan_plan = scan_source_plan(
            source_plan,
            target_project=target_project,
            table_keys=table_keys,
            previewed=ExistingPreviews(source_paths=set(), table_keys=set()),
            download_root=Path(args.download_root),
            max_rows=args.max_rows,
            max_columns=args.max_columns,
            max_tables=args.max_tables,
            progress=progress,
        )
        progress(
            f"retry scan plan ready: tables={len(scan_plan.tables)}, "
            f"eligible={len(scan_plan.eligible_tables)}"
        )
        return scan_plan

    progress("resolving source selection")
    resolved = resolve_source(args)
    progress(
        f"resolved source: kind={resolved.source_kind}, "
        f"source={resolved.source}, target={resolved.target_project}"
    )
    progress("initializing W&B API client")
    api = wandb.Api()
    table_keys = parse_table_keys(args.table_key)
    explicit_max_steps_per_run_key = args.max_steps_per_run_key is not None
    if args.all_steps and explicit_max_steps_per_run_key:
        raise ValueError("Pass either --all-steps or --max-steps-per-run-key, not both.")
    max_steps_per_run_key = (
        None
        if args.all_steps
        else args.max_steps_per_run_key or DEFAULT_MAX_STEPS_PER_RUN_KEY
    )
    progress(
        f"building source plan: table_keys={sorted(table_keys) if table_keys else 'all'}, "
        f"max_steps_per_run_key={max_steps_per_run_key}, max_runs={args.max_runs}"
    )
    source_plan = build_source_plan(
        api,
        resolved.source,
        resolved.source_kind,
        table_keys=table_keys,
        # The CLI expresses selection via max_steps_per_run_key (default 5, or 1
        # for latest-only) and --all-steps, so latest_only stays False here.
        latest_only=False,
        max_steps_per_run_key=max_steps_per_run_key,
        max_runs=args.max_runs,
        run_ids=resolved.run_ids,
        progress=progress,
    )
    progress(
        f"source plan complete: scanned_runs={source_plan.scanned_run_count}, "
        f"table_sources={len(source_plan.table_sources)}"
    )
    previewed = (
        ExistingPreviews(source_paths=set(), table_keys=set())
        if args.force
        else existing_previews(api, resolved.target_project, progress=progress)
    )
    if args.force:
        progress("skipping existing preview metadata check because --force was passed")
    scan_plan = scan_source_plan(
        source_plan,
        target_project=resolved.target_project,
        table_keys=table_keys,
        previewed=previewed,
        download_root=Path(args.download_root),
        max_rows=args.max_rows,
        max_columns=args.max_columns,
        max_tables=args.max_tables,
        progress=progress,
    )
    # Make the fallback run selection visible: when no --run-id is given for a
    # sweep/workspace, the tool scanned the most-recent N runs rather than
    # silently owning "which runs to compare" — the agent can re-scan with
    # explicit --run-id (e.g. baseline/pinned runs).
    if resolved.run_scope_defaulted:
        scan_plan.run_scope = {
            "defaulted": True,
            "max_runs": args.max_runs,
            "note": (
                f"No runs specified; scanned the {args.max_runs} most-recent "
                "runs. Pass --run-id (repeatable) to choose specific runs, such "
                "as the baseline or pinned runs the user cares about."
            ),
        }
    progress(
        f"scan plan ready: scanned_runs={scan_plan.scanned_run_count}, "
        f"tables={len(scan_plan.tables)}, eligible={len(scan_plan.eligible_tables)}"
    )
    return scan_plan


def print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        nargs="?",
        help="run_table artifact URL/path, run URL/path, or source inferred by flags",
    )
    parser.add_argument(
        "--source-kind",
        choices=("artifact", "run", "sweep", "workspace"),
        help="override source kind for the positional source",
    )
    parser.add_argument("--source-project", help="source project for --source-run")
    parser.add_argument("--source-run", help="source run ID or display name")
    parser.add_argument("--source-sweep", help="source sweep URL/path")
    parser.add_argument("--source-workspace", help="source project/workspace URL/path")
    parser.add_argument(
        "--run-id",
        action="append",
        help="specific source run ID to scan for sweep/workspace sources; accepts comma-separated values and can repeat",
    )
    parser.add_argument(
        "--target-project",
        help="destination project, as '<project>' or '<entity>/<project>'; defaults to source project",
    )
    parser.add_argument(
        "--table-key",
        action="append",
        help="table key to include; accepts comma-separated values and can repeat",
    )
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--max-columns", type=int, default=MAX_COLUMNS)
    parser.add_argument(
        "--max-tables",
        type=positive_int,
        default=MAX_TABLES_PER_BATCH,
        help="maximum eligible table artifacts to preview in one batch",
    )
    parser.add_argument(
        "--max-runs",
        type=positive_int,
        default=DEFAULT_MAX_RUNS,
        help="maximum recent runs to scan for sweep/workspace sources",
    )
    parser.add_argument(
        "--max-steps-per-run-key",
        type=positive_int,
        help=(
            "include the latest N table artifact versions per table key/run; "
            f"defaults to {DEFAULT_MAX_STEPS_PER_RUN_KEY}, use 1 for latest-only"
        ),
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="include every table artifact version instead of the latest table per key/run",
    )
    parser.add_argument("--download-root", default=str(DOWNLOAD_ROOT))
    parser.add_argument(
        "--force",
        action="store_true",
        help="include sources even if preview metadata already exists",
    )


def add_column_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-columns", action="append", metavar="COLUMNS")
    parser.add_argument("--output-columns", action="append", metavar="COLUMNS")
    parser.add_argument("--score-columns", action="append", metavar="COLUMNS")
    parser.add_argument(
        "--unspecified-columns-as-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "when any column group is provided, assign unmentioned columns to "
            "outputs (default); pass --no-unspecified-columns-as-output to "
            "require every column to be named explicitly"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress messages on stderr; stdout remains JSON",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan and print a preflight plan")
    add_source_args(scan_parser)

    preview_parser = subparsers.add_parser(
        "preview", help="preview eligible scanned tables as EvalTables"
    )
    add_source_args(preview_parser)
    add_column_args(preview_parser)
    preview_parser.add_argument(
        "--no-column-map",
        action="store_true",
        help=(
            "log an untyped row-order retry with the _row_index_match EvalTable "
            "key suffix and a preview run name ending in row_index_match"
        ),
    )
    preview_parser.add_argument(
        "--row-index-match",
        action="store_true",
        help=(
            "match examples by row index instead of input columns; score and "
            "output columns remain available, and the retry uses the "
            "row_index_match name suffix"
        ),
    )
    preview_parser.add_argument(
        "--from-preview-run",
        action="append",
        metavar="RUN",
        help=(
            "reuse the exact source-table manifest from a prior helper-created "
            "preview run; repeat for every prior preview run in the migration"
        ),
    )
    preview_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform scan and validation without creating W&B runs",
    )

    delete_parser = subparsers.add_parser(
        "delete-preview",
        help="delete preview runs created by this helper and their weave evaluations",
    )
    delete_parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="RUN",
        help=(
            "a preview run to delete, as '<entity>/<project>/<run_id>', "
            "'<project>/<run_id>', or a run URL; repeatable"
        ),
    )
    delete_parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="also delete artifacts logged by the preview run",
    )
    delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be deleted without deleting anything",
    )

    verify_parser = subparsers.add_parser(
        "verify-preview",
        help="verify preview EvalTables exist in run history and Weave",
    )
    verify_parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="RUN",
        help=(
            "a preview run to verify, as '<entity>/<project>/<run_id>', "
            "'<project>/<run_id>', or a run URL; repeatable"
        ),
    )
    verify_parser.add_argument(
        "--table-key",
        action="append",
        required=True,
        metavar="KEY",
        help=(
            "exact logged EvalTable key to verify, such as "
            "'val_predictions_preview'; accepts comma-separated values and can repeat"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_output_streams()
    progress = _noop_progress if args.quiet else _stderr_progress

    progress(f"starting {args.command}")
    if not args.quiet:
        configure_wandb_console_env(progress)
    progress("importing wandb")
    import wandb
    progress("wandb imported")

    if args.command == "scan":
        scan_plan = build_scan_from_args(args, wandb, progress=progress)
        print_json(scan_plan.to_dict())
        return

    if args.command == "delete-preview":
        result = delete_preview_runs(
            wandb,
            parse_run_ids(args.run) or [],
            dry_run=args.dry_run,
            delete_artifacts=args.delete_artifacts,
            progress=progress,
        )
        print_json(result)
        return

    if args.command == "verify-preview":
        result = verify_preview_runs(
            wandb,
            parse_run_ids(args.run) or [],
            table_keys=parse_table_keys(args.table_key) or set(),
            progress=progress,
        )
        print_json(result)
        return

    progress("checking wandb.EvalTable availability")
    require_eval_table_cls(wandb)
    # Preview converts one table key per call so multiple keys become independent
    # invocations — isolated partial successes/failures and less work per call.
    table_keys = parse_table_keys(args.table_key)
    if table_keys is None or len(table_keys) != 1:
        raise ValueError(
            "preview converts one table key per call; pass exactly one "
            "--table-key. Run `scan` to list keys, then preview each key in its "
            "own call so failures stay isolated."
        )
    if args.from_preview_run is not None and not (
        args.no_column_map or args.row_index_match
    ):
        raise ValueError(
            "--from-preview-run requires --no-column-map or --row-index-match."
        )
    scan_plan = build_scan_from_args(args, wandb, progress=progress)
    result = preview_scan_plan(
        wandb,
        scan_plan,
        dry_run=args.dry_run,
        input_columns=parse_column_args(args.input_columns, "--input-columns"),
        output_columns=parse_column_args(args.output_columns, "--output-columns"),
        score_columns=parse_column_args(args.score_columns, "--score-columns"),
        unspecified_columns_as_output=args.unspecified_columns_as_output,
        no_column_map=args.no_column_map,
        row_index_match=args.row_index_match,
        progress=progress,
    )
    print_json(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        # Full traceback (with any chained cause) so opaque failures — e.g. an
        # API-key verification error surfaced from wandb-core — are diagnosable
        # instead of collapsing to a single str(e) line.
        traceback.print_exc()
        sys.exit(1)
