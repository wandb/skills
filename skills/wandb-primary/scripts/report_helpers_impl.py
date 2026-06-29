# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for creating/editing W&B Reports with verification baked in.

A clean ``report.save()`` return is NOT proof the report landed — saves can fail
silently, especially for workspace views. These helpers save and then re-read the
object from the server, so you get a trustworthy ``verified`` flag instead of
hand-rolling (and usually skipping) the read-back.

Usage:
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    import wandb_workspaces.reports.v2 as wr
    from report_helpers import save_report_verified, edit_report_verified

    # CREATE: build blocks however you like, then save+verify in one call.
    report = wr.Report(entity=ENT, project=PROJ, title="...", width="readable",
                       blocks=[...])
    result = save_report_verified(report)        # draft=True by default
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

from typing import Any, Callable, TypedDict


class ReportResult(TypedDict):
    """Outcome of a verified report save/edit."""

    url: str
    verified: bool
    saved_blocks: int
    readback_blocks: int
    readback_title: str
    error: str | None
    answer: str


def _readback(url: str) -> Any:
    """Re-read a report from the server. Returns the loaded Report or raises."""
    import wandb_workspaces.reports.v2 as wr

    return wr.Report.from_url(url)


def save_report_verified(
    report: Any,
    *,
    draft: bool = True,
    expect_min_blocks: int | None = None,
    expect_text: str | None = None,
) -> ReportResult:
    """Save ``report`` (draft by default) then re-read it and confirm it landed.

    Verification passes when the re-read report has at least ``expect_min_blocks``
    blocks (defaults to the number of blocks we tried to save) and, if
    ``expect_text`` is given, that string appears somewhere in the re-read blocks.

    Returns a ReportResult; ``verified`` is False (never raises) if the read-back
    fails or the content does not match, so the caller can report failure honestly
    instead of claiming success.
    """
    saved_blocks = len(getattr(report, "blocks", []) or [])
    want_blocks = expect_min_blocks if expect_min_blocks is not None else saved_blocks

    try:
        report.save(draft=draft)
    except Exception as exc:  # noqa: BLE001
        return ReportResult(
            url="", verified=False, saved_blocks=saved_blocks, readback_blocks=0,
            readback_title="", error=f"save failed: {exc}",
            answer=f"answer=report NOT saved verified=False error={exc!r}",
        )

    url = getattr(report, "url", "") or ""
    try:
        check = _readback(url)
    except Exception as exc:  # noqa: BLE001
        return ReportResult(
            url=url, verified=False, saved_blocks=saved_blocks, readback_blocks=0,
            readback_title="", error=f"read-back failed: {exc}",
            answer=f"answer=save returned but read-back FAILED verified=False url={url} error={exc!r}",
        )

    readback_blocks = len(getattr(check, "blocks", []) or [])
    text_ok = True
    if expect_text is not None:
        text_ok = expect_text in _blocks_text(check)

    verified = readback_blocks >= want_blocks > 0 and text_ok
    detail = ""
    if not verified:
        if readback_blocks < want_blocks:
            detail = f" (read-back has {readback_blocks} blocks, expected >= {want_blocks})"
        elif not text_ok:
            detail = f" (expected text {expect_text!r} not found in read-back)"

    return ReportResult(
        url=url, verified=verified, saved_blocks=saved_blocks,
        readback_blocks=readback_blocks,
        readback_title=getattr(check, "title", "") or "",
        error=None if verified else f"verification failed{detail}",
        answer=(
            f"answer=report saved and verified verified={verified} "
            f"blocks={readback_blocks} url={url}" + (detail if not verified else "")
        ),
    )


def edit_report_verified(
    url: str,
    mutate: Callable[[Any], None],
    *,
    draft: bool = True,
    expect_text: str | None = None,
) -> ReportResult:
    """Load the report at ``url``, apply ``mutate`` to it, then save+verify.

    ``mutate`` receives the loaded Report and should modify it in place (e.g. append
    to ``report.blocks``). Editing via ``from_url`` preserves the report ID; building
    a fresh ``wr.Report(...)`` and saving would orphan the original — don't.
    Verification requires the read-back to have at least as many blocks as the
    post-mutation report (and ``expect_text`` if given).
    """
    try:
        report = _readback(url)
    except Exception as exc:  # noqa: BLE001
        return ReportResult(
            url=url, verified=False, saved_blocks=0, readback_blocks=0,
            readback_title="", error=f"load failed: {exc}",
            answer=f"answer=could not load report to edit verified=False url={url} error={exc!r}",
        )

    before = len(getattr(report, "blocks", []) or [])
    mutate(report)
    after = len(getattr(report, "blocks", []) or [])
    return save_report_verified(
        report, draft=draft,
        expect_min_blocks=max(after, before),
        expect_text=expect_text,
    )


def _blocks_text(report: Any) -> str:
    """Best-effort flatten of a report's text content for substring checks."""
    import json

    try:
        return json.dumps([_block_to_plain(b) for b in (report.blocks or [])], default=str)
    except Exception:  # noqa: BLE001
        return str(getattr(report, "blocks", ""))


def _block_to_plain(block: Any) -> Any:
    for attr in ("model_dump", "to_dict"):
        fn = getattr(block, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    return str(block)
