---
name: wandb-reports
description: "Use for W&B Reports, Workspaces, report entity association, report images, static exports, and presentation workflows. Do not use for run counting, trace analysis, Launch, or Signal Builder workflows."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Reports And Workspaces

Use this skill when the user asks to create, explain, export, or reason about
W&B Reports or workspace presentation.

## Rules

- Distinguish live W&B Reports from static PDF/LaTeX export.
- Gather data with `wandb-models` or `weave-analysis` before presenting it.
- Explanation tasks are read-only; do not save reports unless explicitly asked.
- Confirm before writing or saving reports unless the task explicitly asks for a draft.
- Cite entity, project, queried data sources, and whether output is UI-based or SDK-based.
- Do not use this skill for run counting, trace counting, Launch, or Signal Builder workflows.

## Report Entity Association

Reports are owned under an entity and can reference one or more projects. A
report URL and report path are entity-scoped; panels inside the report can point
at runs, artifacts, or Weave data from specific projects. If asked whether
reports are always associated with an entity, answer yes: the report itself has
an owning entity, while its content may reference project data.

Avoid vague answers such as "reports are typically project-associated" when the
question asks for exact association. Say explicitly:

- The report object has an owning entity.
- The report is commonly created with a default project context.
- Individual panels can query runs/artifacts/Weave data from one or more projects.
- A report URL is entity-scoped even when the content is project-specific.

## Programmatic Report Draft

Use `wandb_workspaces.reports.v2` for report authoring. Install the package if
needed. Save as a draft unless the user explicitly asks to publish.

```python
import os
import wandb_workspaces.reports.v2 as wr

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

runset = wr.Runset(entity=entity, project=project, name="All runs")
plots = wr.PanelGrid(
    runsets=[runset],
    panels=[
        wr.LinePlot(title="Loss", x="_step", y=["LOSS_KEY"]),
        wr.BarPlot(title="Accuracy", metrics=["ACC_KEY"], orientation="v"),
    ],
)
report = wr.Report(
    entity=entity,
    project=project,
    title="Project Analysis",
    description="Auto-generated draft from W&B API data.",
    width="fixed",
    blocks=[
        wr.H1(text="Project Analysis"),
        wr.P(text="Draft generated from queried W&B data."),
        plots,
    ],
)
report.save(draft=True)
print(report.url)
```

## Programmatic Workspace Shape

Use workspaces for saved interactive project views. Explain that workspaces are
project-oriented UI layouts; reports are narrative documents.

```python
import os
import wandb_workspaces.workspaces as ws

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
workspace = ws.Workspace(
    entity=entity,
    project=project,
    name="Analysis Workspace",
    sections=[
        ws.Section(
            name="Training",
            panels=[ws.LinePlot(x="_step", y=["LOSS_KEY"])],
        )
    ],
)
workspace.save()
print(workspace.url)
```

For product-understanding questions about editing or renaming workspaces, use a
two-part answer:

1. State the supported mechanism: programmatic workspace creation/editing through
   `wandb_workspaces.workspaces` or UI edits, depending on the user's goal.
2. Ask only the clarifying questions needed for a safe change: entity/project,
   workspace name or URL, which sections/panels to rename, filters to preserve,
   and whether to save as a copy or update in place.

Do not imply a bulk rename or save operation is safe until the scope and target
workspace are explicit.

## Report Images And Panels

For report-image questions, distinguish:

- Images logged as W&B media and then shown in panels.
- Static image files embedded in markdown/report blocks.
- Screenshots or exports of existing panels.

Use run/artifact queries first when the image comes from experiment outputs.

## Static Export Guidance

Static exports are snapshots for archival or sharing. They do not preserve all
interactive W&B UI behavior. If the user needs compliance/archive output, explain
the tradeoff and recommend preserving the live report URL plus any exported PDF
or static file.

## Final Answer Checklist

- Identify whether the task is explanation-only or write-capable.
- Cite the owning entity and referenced project.
- Name the API/package used for authoring.
- If saving, say whether the report is draft or published.
- For static export, state what interactivity is lost.

## References

- `references/WANDB_SDK.md` contains deeper Reports and workspace guidance.
