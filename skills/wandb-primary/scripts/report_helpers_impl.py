# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for creating/editing W&B Reports with verification baked in.

A clean ``report.save()`` return is NOT proof the report landed — saves can fail
silently, especially for workspace views. These helpers save and then re-read the
object from the server, so you get a trustworthy ``verified`` flag instead of
hand-rolling (and usually skipping) the read-back.

Existing-report edits also save the original GraphQL ``View`` response before any
Pydantic parsing. In particular, the ``spec`` remains the exact JSON string returned
by the server; ``model_dump()`` output is not a recovery backup.

Usage:
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    import wandb_workspaces.reports.v2 as wr
    from report_helpers import save_report_verified, edit_report_verified

    # CREATE: build blocks however you like, then save+verify in one call.
    report = wr.Report(entity=ENT, project=PROJ, title="...", width="readable",
                       blocks=[...])
    result = save_report_verified(report)        # publishes; pass draft=True for a private draft
    print(result["answer"])                      # -> answer=... verified=True url=...

    # EDIT: mutate an existing report, then save+verify.
    result = edit_report_verified(
        "https://wandb.ai/ENT/PROJ/reports/Title--ID",
        lambda r: r.blocks.extend([wr.H2(text="New section"), wr.P(text="...")]),
    )

Do NOT call ``wandb.init()`` in a report script — it mints a stray run in the user's
project. These helpers never do.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from report_write import DEFAULT_REPORT_BACKUP_DIR, backup_report_view

_SAVE_VERIFIED_OUTCOME = "report saved and verified"
_SAVE_UNVERIFIED_OUTCOME = "report saved but read-back verification FAILED"


class ReportResult(TypedDict):
    """Outcome of a verified report save/edit."""

    url: str
    verified: bool
    saved_blocks: int
    readback_blocks: int
    readback_title: str
    backup_path: str
    error: str | None
    answer: str


def _readback(url: str, *, as_model: bool = False) -> Any:
    """Re-read a report from the server. Returns the loaded Report or raises."""
    import wandb_workspaces.reports.v2 as wr

    return wr.Report.from_url(url, as_model=as_model)


def save_report_verified(
    report: Any,
    *,
    draft: bool = False,
    expect_min_blocks: int | None = None,
    expect_text: str | None = None,
    verify_readback: Callable[[Any], bool] | None = None,
) -> ReportResult:
    """Save ``report`` then re-read it and confirm it landed.

    Verification passes when the re-read report has at least ``expect_min_blocks``
    blocks (defaults to the number of blocks we tried to save), ``expect_text`` is
    present when provided, and ``verify_readback`` returns True when provided.

    Returns a ReportResult; ``verified`` is False (never raises) if the read-back
    fails or the content does not match, so the caller can report failure honestly
    instead of claiming success.
    """
    saved_blocks = len(getattr(report, "blocks", []) or [])
    want_blocks = expect_min_blocks if expect_min_blocks is not None else saved_blocks

    try:
        report.save(draft=draft)
    except Exception as exc:
        return ReportResult(
            url="",
            verified=False,
            saved_blocks=saved_blocks,
            readback_blocks=0,
            readback_title="",
            backup_path="",
            error=f"save failed: {exc}",
            answer=f"answer=report NOT saved verified=False error={exc!r}",
        )

    url = getattr(report, "url", "") or ""
    try:
        check = _readback(url)
    except Exception as exc:
        return ReportResult(
            url=url,
            verified=False,
            saved_blocks=saved_blocks,
            readback_blocks=0,
            readback_title="",
            backup_path="",
            error=f"read-back failed: {exc}",
            answer=f"answer=save returned but read-back FAILED verified=False url={url} error={exc!r}",
        )

    readback_blocks = len(getattr(check, "blocks", []) or [])
    text_ok = True
    if expect_text is not None:
        text_ok = expect_text in _blocks_text(check)

    custom_ok = True
    custom_error = ""
    if verify_readback is not None:
        try:
            custom_ok = bool(verify_readback(check))
        except Exception as exc:
            custom_ok = False
            custom_error = f": {exc}"

    verified = readback_blocks >= want_blocks > 0 and text_ok and custom_ok
    detail = ""
    if not verified:
        if readback_blocks < want_blocks:
            detail = (
                f" (read-back has {readback_blocks} blocks, expected >= {want_blocks})"
            )
        elif not text_ok:
            detail = f" (expected text {expect_text!r} not found in read-back)"
        elif not custom_ok:
            detail = f" (custom read-back verification failed{custom_error})"

    outcome = _SAVE_VERIFIED_OUTCOME if verified else _SAVE_UNVERIFIED_OUTCOME
    return ReportResult(
        url=url,
        verified=verified,
        saved_blocks=saved_blocks,
        readback_blocks=readback_blocks,
        readback_title=getattr(check, "title", "") or "",
        backup_path="",
        error=None if verified else f"verification failed{detail}",
        answer=(
            f"answer={outcome} verified={verified} "
            f"blocks={readback_blocks} url={url}{detail}"
        ),
    )


def edit_report_verified(
    url: str,
    mutate: Callable[[Any], None],
    *,
    draft: bool = False,
    expect_text: str | None = None,
    verify_readback: Callable[[Any], bool] | None = None,
    backup_dir: str | Path = DEFAULT_REPORT_BACKUP_DIR,
) -> ReportResult:
    """Back up, load, mutate, and save+verify the report at ``url``.

    ``mutate`` receives the loaded Report and should modify it in place (e.g. append
    to ``report.blocks``). Editing via ``from_url`` preserves the report ID; building
    a fresh ``wr.Report(...)`` and saving would orphan the original — don't.
    Verification requires the read-back to have at least as many blocks as the
    post-mutation report, plus any ``expect_text`` or ``verify_readback`` condition.

    The backup is fetched directly from GraphQL before the SDK parses the report.
    The report's stored width is also restored because the public SDK loader currently
    resets it to ``"readable"``.
    """
    try:
        backup_path = backup_report_view(url, backup_dir=backup_dir)
    except Exception as exc:
        return ReportResult(
            url=url,
            verified=False,
            saved_blocks=0,
            readback_blocks=0,
            readback_title="",
            backup_path="",
            error=f"backup failed: {exc}",
            answer=f"answer=report NOT edited verified=False backup_failed={exc!r}",
        )

    try:
        stored_width = _readback(url, as_model=True).spec.width
        report = _readback(url)
        report.width = stored_width
    except Exception as exc:
        return ReportResult(
            url=url,
            verified=False,
            saved_blocks=0,
            readback_blocks=0,
            readback_title="",
            backup_path=str(backup_path),
            error=f"load failed: {exc}",
            answer=f"answer=could not load report to edit verified=False url={url} error={exc!r}",
        )

    before = len(getattr(report, "blocks", []) or [])
    try:
        mutate(report)
    except Exception as exc:
        return ReportResult(
            url=url,
            verified=False,
            saved_blocks=before,
            readback_blocks=0,
            readback_title="",
            backup_path=str(backup_path),
            error=f"mutation failed before save: {exc}",
            answer=f"answer=report NOT edited verified=False mutation_failed={exc!r}",
        )
    after = len(getattr(report, "blocks", []) or [])
    result = save_report_verified(
        report,
        draft=draft,
        expect_min_blocks=max(after, before),
        expect_text=expect_text,
        verify_readback=verify_readback,
    )
    result["backup_path"] = str(backup_path)
    result["answer"] += f" backup={backup_path}"
    return result


def _blocks_text(report: Any) -> str:
    """Best-effort flatten of a report's text content for substring checks."""
    import json

    try:
        return json.dumps(
            [_block_to_plain(b) for b in (report.blocks or [])], default=str
        )
    except Exception:
        return str(getattr(report, "blocks", ""))


def _block_to_plain(block: Any) -> Any:
    for attr in ("model_dump", "to_dict"):
        fn = getattr(block, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return str(block)
