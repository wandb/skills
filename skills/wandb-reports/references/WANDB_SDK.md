# W&B Reports And Workspaces Reference

This compact reference contains the report/workspace concepts used by the parity
`wandb-reports` skill.

## Entity And Project Association

Reports are owned under an entity. Reports commonly have a default project
context, and panels inside a report can reference project data such as runs,
artifacts, or Weave data. When answering product questions, distinguish:

- owning entity;
- default project context;
- project data referenced by panels;
- report URL or path.

## Programmatic Reports

Use `wandb_workspaces.reports.v2` for report authoring tasks.

```python
import wandb_workspaces.reports.v2 as wr

runset = wr.Runset(entity="entity", project="project", name="All runs")
report = wr.Report(
    entity="entity",
    project="project",
    title="Project Analysis",
    blocks=[wr.H1(text="Project Analysis")],
)
report.save(draft=True)
```

Save drafts unless the user explicitly asks to publish.

## Workspaces

Use `wandb_workspaces.workspaces` for saved project layouts. Ask clarifying
questions before bulk edits or in-place saves:

- target entity/project;
- workspace URL or name;
- sections or panels to modify;
- whether to save a copy or update in place.

## Static Exports

Static exports are snapshots. They do not preserve all interactivity. Preserve
the live report URL when possible.
