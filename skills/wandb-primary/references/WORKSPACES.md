<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Workspaces - programmatic views, sections, and panels

Reference for **workspace views**: the saved layouts of sections and panels on a
project's run page. Read this whenever a task involves inspecting or mutating a
workspace - add/rename sections, add panels (including custom-expression panels
like "loss scaled by 2"), set runset filters/groupby/order, pin columns, color or
hide runs, or save a view. Don't write workspace code from memory; the SDK has
sharp edges (see Gotchas) and the most common case (a user's *default* workspace)
is not reachable through `Workspace.from_url`.

Use `wandb-workspaces` for programmatic workspace edits. Install `wandb-workspaces`
or `wandb[workspaces]` if it is missing. Saves write **in place** - there is no
draft state - so be confident in the change before calling `save()`, and only save
after the user asks.

Required imports:

```python
import os

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr  # panel types: LinePlot, BarPlot, ScalarChart, ...
```

## View kinds: default workspace vs saved view

A workspace URL carries a `?nw=<token>` that identifies the view:

- **Default user workspace** - `?nw=nwuser<username>`. This is what a user sees by
  default on a project's runs page. `ws.Workspace.from_url(...)` **rejects** these
  URLs with `UnsupportedViewError`; mutate the default workspace through the
  raw-spec GraphQL path (last section).
- **Saved view** - `?nw=<id>` (or a URL with no `nw` token). `from_url` loads these.
  This is the clean, supported path - prefer it, or `save_as_new_view()` to snapshot
  the default workspace into a saved view you can then edit normally.

## Load and edit a saved view (primary path)

```python
ws_view = ws.Workspace.from_url("https://wandb.ai/ENTITY/PROJECT/?nw=VIEW_ID")

# Append a section with one panel.
ws_view.sections.append(
    ws.Section(
        name="Loss",
        is_open=True,
        panels=[wr.LinePlot(title="Train loss", x="Step", y=["train/loss"])],
    )
)

ws_view.save()                 # overwrite the same saved view in place
print(f"Saved: {ws_view.url}")

# Or snapshot to a brand-new saved view instead of overwriting:
# new_view = ws_view.save_as_new_view()
```

## Create a new saved view from scratch

```python
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

ws_view = ws.Workspace(
    entity=entity,
    project=project,
    name="Sweep comparison",
    sections=[
        ws.Section(
            name="Headline metrics",
            is_open=True,
            panels=[
                wr.LinePlot(title="Loss", x="Step", y=["train/loss", "val/loss"]),
                wr.BarPlot(title="Accuracy", metrics=["accuracy"]),
                wr.ScalarChart(metric="accuracy", groupby_aggfunc="max"),
            ],
        ),
    ],
)
ws_view.save()
print(f"Created: {ws_view.url}")
```

`auto_generate_panels=True` auto-populates panels for every logged key, but it can
only be set at creation and cannot be changed afterward.

## Sections and panels

`ws.Section(name=..., panels=[...], is_open=True, pinned=False)`. Panels are the
same `wr.*` types used in Reports: `wr.LinePlot`, `wr.BarPlot`, `wr.ScalarChart`,
`wr.ScatterPlot`, `wr.MediaBrowser`, etc. Use concrete metric keys when the user
names them, or `metric_regex="val/.*"` for a family.

Custom-expression panels (e.g. "loss scaled by 2") use `custom_expressions` and must
wrap any metric whose name contains `/` in `${...}`, otherwise W&B parses the slash
as division:

```python
wr.LinePlot(
    title="Loss x2",
    x="Step",
    y=[],
    custom_expressions=["${train/loss} * 2"],
)
```

## Runset settings: filters, groupby, order, columns, run colors

`ws.RunsetSettings` controls the left-hand run selector. Prefer structured filters
over raw filter strings (same `ws.Metric/Summary/Config/Tags` constructors as
`references/REPORTS.md`).

```python
ws_view.runset_settings = ws.RunsetSettings(
    filters=[
        ws.Metric("State") == "finished",
        ws.Summary("accuracy") > 0.9,
        ws.Config("optimizer") == "adam",
    ],
    groupby=[ws.Config("optimizer")],
    order=[ws.Ordering(ws.Summary("accuracy"), ascending=False)],
    pinned_columns=["summary:accuracy", "summary:loss"],
    run_settings={
        "RUN_ID_A": ws.RunSettings(color="#FF0000"),  # keys are run IDs, not names
        "RUN_ID_B": ws.RunSettings(disabled=True),     # hide a run
    },
)
ws_view.save()
```

Column names use `"run:state"`, `"summary:accuracy"`, `"config:learning_rate"`,
`"tags:__ALL__"`. For grouped logic use `And`/`Or` from `wandb_workspaces.expr`.

## Validate keys first

Before adding a panel or filter, confirm the metric/config key exists - an invented
key renders an empty chart. Discover keys from the user's project (e.g.
`wandb_helpers.probe_project()` or a small `api.runs(...)` sample) rather than
guessing.

## Save semantics

- `save()` - overwrite the loaded saved view in place. No draft, no undo.
- `save_as_new_view()` - write a new saved view, leaving the original untouched.
  Use this to turn the default workspace into an editable saved view, or to avoid
  clobbering a view you don't own.

## Gotchas

| Gotcha | Detail |
|--------|--------|
| No draft state | `save()` writes immediately. Be sure before you call it. |
| `from_url` rejects `nwuser*` | The default user workspace raises `UnsupportedViewError`; use the raw-spec path or `save_as_new_view()`. |
| Older spec shapes | `Workspace._from_model` can reject legacy specs (e.g. `panelConfigOverrides`); the raw-spec path is immune. |
| `x` (U+00D7) reads as emoji | The Pydantic validator rejects it; use ASCII `x` or `*` in titles/names. |
| Slash in metric names | Wrap in `${...}` inside custom expressions, else `a/b` parses as division. |
| `run_settings` keys are run IDs | Not display names - use `run.id`. |
| `auto_generate_panels` is set-once | Only at workspace creation; cannot be changed later. |
| Don't mass-edit on ambiguity | If a target view/section is ambiguous, surface candidates before guessing. |

## Raw-spec fallback for the default user workspace

`from_url` cannot load `?nw=nwuser<username>` views, so to edit a user's *default*
workspace, fetch the raw view spec over GraphQL, mutate the dict, and upsert it.
This needs no extra dependencies beyond `wandb`.

```python
import json
import wandb


def execute_graphql(api, query, variables):
    from wandb_graphql.language import parser as gql_parser
    return api.client.execute(gql_parser.parse(query), variable_values=variables)


entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
token = "nwuser<username>"  # the ?nw= value from the workspace URL
api = wandb.Api(timeout=120)

# 1. Fetch the raw spec via allViews. Default workspace backend name is
#    nw-<token>-w; a saved view is nw-<id>-v.
list_q = """query ($e:String!,$p:String!){
  project(entityName:$e,name:$p){ allViews(viewType:"project-view",first:100){
    edges{ node{ id name displayName spec } } } } }"""
nodes = [
    e["node"]
    for e in execute_graphql(api, list_q, {"e": entity, "p": project})[
        "project"
    ]["allViews"]["edges"]
]
node = next(n for n in nodes if n["name"] in (f"nw-{token}-w", f"nw-{token}-v"))
spec = json.loads(node["spec"])

# 2. Build a well-formed panel dict via the SDK, then mutate the spec directly -
#    don't round-trip through Workspace._from_model.
panel = (
    wr.LinePlot(title="Loss x2", x="Step", y=[], custom_expressions=["${train/loss} * 2"])
    ._to_model()
    .model_dump(by_alias=True, exclude_none=True)
)
panel.pop("layout", None)
panel["__id__"] = wandb.util.generate_id(11)

sections = spec["section"]["panelBankConfig"]["sections"]
section = next((s for s in sections if (s.get("name") or "").lower() == "loss"), None)
if section is None:
    section = {"name": "Loss", "panels": [], "isOpen": True}
    sections.append(section)
section.setdefault("panels", []).append(panel)

# 3. Upsert the mutated spec.
mutation = """mutation ($id:ID,$e:String,$p:String,$t:String,$n:String,$dn:String,$d:String,$s:String){
  upsertView(input:{id:$id,entityName:$e,projectName:$p,name:$n,displayName:$dn,description:$d,type:$t,spec:$s,createdUsing:WANDB_SDK}){
    view{ id name displayName } inserted } }"""
execute_graphql(api, mutation, {
    "id": node["id"], "e": entity, "p": project,
    "n": node["name"], "dn": node["displayName"],
    "d": "", "t": "project-view",
    "s": json.dumps(spec, separators=(",", ":")),
})
print(f"Updated workspace: https://wandb.ai/{entity}/{project}/?nw={token}")
```

## Final reply

After a mutation, give the workspace URL **and** a one-line summary of what changed
(section, panel title, expression/metric, whether a section was created). A bare URL
buries the change.
