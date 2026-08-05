# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""W&B workspace-view manipulation helpers (raw-spec path; survives user-workspace views).

All spec writes go through `workspace_write.mutate_view_with_retry` (field-minimal
`UpsertView` guarded on `View.updatedAt`, with bounded re-read/re-apply on conflict);
this module owns the CLI surface plus panel building and read-back verification.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

# Shared guarded-write engine (OCC on updatedAt + fetch/re-apply/retry). Every public
# workspace-view write routes through here so concurrent user/tab edits are layered
# onto rather than clobbered.
from workspace_write import (
    API_TIMEOUT_SECONDS,
    fetch_view,
    list_views,
    mutate_view_with_retry,
    public_url,
    read_back_spec,
    resolve_path,
    section_by_name,
    sections_of,
)

DESCRIPTION = "Programmatic W&B workspace-view manipulation: add panels, add sections, find views."


@dataclass(frozen=True)
class _MutationDetails:
    section_name: str | None = None
    section_created: bool = False


# ---------- spec mutation --------------------------------------------------


def _find_or_create_section(spec: dict[str, Any], name: str, *, create: bool, is_open: bool = True, pinned: bool = False) -> tuple[dict[str, Any], bool]:
    import wandb

    sections = spec["section"]["panelBankConfig"]["sections"]
    needle = name.lower()
    for section in sections:
        if (section.get("name") or "").lower() == needle:
            return section, False
    if not create:
        raise SystemExit(f"Section {name!r} not found. Pass --create-section to add it, or use add-section.")
    new_section = {
        "__id__": wandb.util.generate_id(11),
        "name": name,
        "isOpen": is_open,
        "sorted": 0,
        "pinned": pinned,
        "isPanelsAuto": False,
        "defaultName": name,
        "panels": [],
    }
    sections.append(new_section)
    return new_section, True


def _build_panel(*, panel_type: str, title: str, metrics: list[str], custom_expressions: list[str], x_axis: str, x_title: str | None, y_title: str | None, plot_type: str | None) -> dict[str, Any]:
    import wandb
    import wandb_workspaces.reports.v2 as wr

    if panel_type == "line":
        model = wr.LinePlot(title=title, x=x_axis, y=metrics, custom_expressions=custom_expressions or None, title_x=x_title, title_y=y_title, plot_type=plot_type)
    elif panel_type == "bar":
        if not metrics and not custom_expressions:
            raise SystemExit("--panel-type bar requires at least one --metric or --custom-expression.")
        model = wr.BarPlot(title=title, metrics=metrics, custom_expressions=custom_expressions or None, title_x=x_title, title_y=y_title)
    elif panel_type == "scalar":
        if len(metrics) > 1:
            raise SystemExit("--panel-type scalar accepts at most one --metric.")
        model = wr.ScalarChart(title=title, metric=metrics[0] if metrics else "", custom_expressions=custom_expressions or None)
    else:
        raise SystemExit(f"Unsupported --panel-type {panel_type!r}; pick one of line, bar, scalar.")

    panel = model._to_model().model_dump(by_alias=True, exclude_none=True)
    panel.pop("layout", None)  # match the shape of panels in existing user workspaces.
    panel["__id__"] = wandb.util.generate_id(11)
    return panel


def _panel_title(panel: dict[str, Any]) -> str:
    return (panel.get("config") or {}).get("chartTitle") or ""


# ---------- commands -------------------------------------------------------


def cmd_find_view(args: argparse.Namespace) -> None:
    import wandb

    entity, project = resolve_path(args.path, args.view_url)
    api = wandb.Api(timeout=API_TIMEOUT_SECONDS)

    if args.view_url or args.view_token or args.display_contains:
        # Narrow-first via fetch_view: a token match costs one single-view
        # query instead of every view's full spec.
        node, spec = fetch_view(api, entity, project, view_url=args.view_url, view_token=args.view_token, display_contains=args.display_contains)
        sections = sections_of(spec)
        print(json.dumps({
            "answer": "found",
            "id": node["id"],
            "name": node["name"],
            "displayName": node["displayName"],
            "url": public_url(entity, project, node["name"]),
            "sections": [{"name": s.get("name"), "panels": len(s.get("panels", [])), "is_open": s.get("isOpen"), "pinned": s.get("pinned")} for s in sections],
        }, indent=2))
        return

    nodes = list_views(api, entity, project, limit=args.limit)
    print(json.dumps({
        "answer": "listed",
        "path": f"{entity}/{project}",
        "view_count": len(nodes),
        "views": [{"id": n["id"], "name": n["name"], "displayName": n["displayName"], "url": public_url(entity, project, n["name"])} for n in nodes],
    }, indent=2))


def cmd_add_panel(args: argparse.Namespace) -> None:
    import wandb

    entity, project = resolve_path(args.path, args.view_url)
    api = wandb.Api(timeout=API_TIMEOUT_SECONDS)

    section_name = args.section or "Custom"
    def mutate(spec: dict[str, Any]) -> _MutationDetails:
        section, created = _find_or_create_section(
            spec,
            section_name,
            create=args.create_section or not args.section,
        )

        panel = _build_panel(
            panel_type=args.panel_type,
            title=args.title,
            metrics=args.metric or [],
            custom_expressions=args.custom_expression or [],
            x_axis=args.x,
            x_title=args.x_title,
            y_title=args.y_title,
            plot_type=args.plot_type,
        )

        if not args.allow_duplicate:
            for existing in section.get("panels", []):
                if _panel_title(existing) == args.title:
                    raise SystemExit(f"Section {section.get('name')!r} already has a panel titled {args.title!r}. Pass --allow-duplicate to add anyway, or pick a different --title.")

        section.setdefault("panels", []).append(panel)
        return _MutationDetails(
            section_name=section.get("name"),
            section_created=created,
        )
    result, mutation_details = mutate_view_with_retry(
        api,
        entity=entity,
        project=project,
        view_url=args.view_url,
        view_token=args.view_token,
        display_contains=args.display_contains,
        mutate=mutate,
    )

    # Verify against the server state, not the in-memory spec we just built.
    verified_spec = read_back_spec(api, entity, project, result["view"]["name"])
    verified_section = section_by_name(verified_spec, mutation_details.section_name or "")
    verified_titles = [_panel_title(p) for p in (verified_section or {}).get("panels", [])]
    if args.title not in verified_titles:
        raise SystemExit(f"Verification failed: panel {args.title!r} is missing from section {mutation_details.section_name!r} on fresh read-back. The mutation response claimed success; do not trust it — inspect the view and retry.")

    print(json.dumps({
        "answer": "panel_added",
        "view_id": result["view"]["id"],
        "view_name": result["view"]["name"],
        "displayName": result["view"]["displayName"],
        "url": public_url(entity, project, result["view"]["name"]),
        "section": mutation_details.section_name,
        "section_created": mutation_details.section_created,
        "panel_title": args.title,
        "panel_type": args.panel_type,
        "metrics": args.metric or [],
        "custom_expressions": args.custom_expression or [],
        "verified_by_read_back": True,
        "panels_in_section_after": len((verified_section or {}).get("panels", [])),
    }, indent=2))


def cmd_add_section(args: argparse.Namespace) -> None:
    import wandb

    entity, project = resolve_path(args.path, args.view_url)
    api = wandb.Api(timeout=API_TIMEOUT_SECONDS)

    def mutate(spec: dict[str, Any]) -> _MutationDetails:
        sections = spec["section"]["panelBankConfig"]["sections"]
        if any((s.get("name") or "").lower() == args.name.lower() for s in sections):
            raise SystemExit(f"Section {args.name!r} already exists in this view.")

        _find_or_create_section(
                spec,
                args.name,
                create=True,
                is_open=args.open,
                pinned=args.pinned,
            )
        return _MutationDetails()

    result, _ = mutate_view_with_retry(
        api,
        entity=entity,
        project=project,
        view_url=args.view_url,
        view_token=args.view_token,
        display_contains=args.display_contains,
        mutate=mutate,
    )

    # Verify against the server state, not the in-memory spec we just built.
    verified_spec = read_back_spec(api, entity, project, result["view"]["name"])
    if section_by_name(verified_spec, args.name) is None:
        raise SystemExit(f"Verification failed: section {args.name!r} is missing on fresh read-back. The mutation response claimed success; do not trust it — inspect the view and retry.")

    print(json.dumps({
        "answer": "section_added",
        "view_id": result["view"]["id"],
        "url": public_url(entity, project, result["view"]["name"]),
        "section": args.name,
        "is_open": args.open,
        "pinned": args.pinned,
        "verified_by_read_back": True,
        "section_count_after": len(sections_of(verified_spec)),
    }, indent=2))


# ---------- argparse -------------------------------------------------------


_TARGET_FLAGS: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (("--view-url",), {"help": "Workspace URL containing an ?nw=<token> query param. Resolves entity/project too."}),
    (("--view-token",), {"help": "The `nw` token alone (e.g. `nwusersrando` or `abc123`)."}),
    (("--display-contains",), {"help": "Substring of the view displayName. Must match exactly one view."}),
    (("--path",), {"help": "ENTITY/PROJECT. Defaults to WANDB_ENTITY/WANDB_PROJECT. Ignored if --view-url is given."}),
]


def _add_target_args(child: argparse.ArgumentParser) -> None:
    for flags, kwargs in _TARGET_FLAGS:
        child.add_argument(*flags, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("find-view", help="List saved workspace views or fetch one view's sections/panels.")
    _add_target_args(find)
    find.add_argument("--limit", type=int, default=100, help="Maximum saved views to fetch.")
    find.set_defaults(func=cmd_find_view)

    add_panel = sub.add_parser("add-panel", help="Add a panel to a workspace view, in place, keyed by the backend view ID. Supports user workspaces (nwuser*).")
    _add_target_args(add_panel)
    add_panel.add_argument("--panel-type", choices=["line", "bar", "scalar"], default="line")
    add_panel.add_argument("--title", required=True, help="Panel title. Avoid emoji/symbols (e.g. `×`); use ASCII like `x`.")
    add_panel.add_argument("--section", help="Target section name. If omitted, panel goes into a section named 'Custom' (created if needed).")
    add_panel.add_argument("--create-section", action="store_true", help="Create --section if it doesn't exist.")
    add_panel.add_argument("--metric", action="append", default=[], help="Logged metric to plot; repeat for multiple. Required for bar; optional for line/scalar.")
    add_panel.add_argument("--custom-expression", action="append", default=[], help="W&B expression string like `${reward/episode_mean} * 2`. Repeat for multiple. Wrap metric names with `/` in `${}`.")
    add_panel.add_argument("--x", default="_step", help="X-axis key for line plots. Defaults to `_step`.")
    add_panel.add_argument("--x-title", help="X-axis label.")
    add_panel.add_argument("--y-title", help="Y-axis label.")
    add_panel.add_argument("--plot-type", choices=["line", "stacked-area", "pct-area"], help="Line plot variant.")
    add_panel.add_argument("--allow-duplicate", action="store_true", help="Allow adding a panel whose title matches an existing panel in the section.")
    add_panel.set_defaults(func=cmd_add_panel)

    add_section = sub.add_parser("add-section", help="Add an empty section to a workspace view.")
    _add_target_args(add_section)
    add_section.add_argument("--name", required=True, help="Section name. Avoid emoji/symbols.")
    add_section.add_argument("--open", action="store_true", help="Open the section by default.")
    add_section.add_argument("--pinned", action="store_true", help="Pin the section to the top of the workspace.")
    add_section.set_defaults(func=cmd_add_section)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
