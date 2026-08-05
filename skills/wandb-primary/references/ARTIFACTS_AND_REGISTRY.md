<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Artifact and Registry workflow design

Use this reference when designing or changing an ML asset pipeline: versioning
datasets, checkpoints, models, and evaluation outputs; preserving lineage; or
promoting validated versions through W&B Registry. Use `WANDB_CONCEPTS.md` for
the data model, `WANDB_SDK.md` for single-object SDK mechanics, and
`ARTIFACT_OPS.md` for bounded inventory and filtering.

## End-to-end workflow

1. Consume the exact raw dataset version with `run.use_artifact()` and log the
   processed dataset as a new output Artifact.
2. Train against a pinned dataset version. Log meaningful checkpoints or the
   final/best model as `model` Artifacts; keep scalar metrics on the Run and
   structured comparison facts in Artifact metadata.
3. Evaluate exact model and test-dataset versions. Log durable predictions or
   evaluation tables as outputs.
4. After validation, link the exact passing version into an existing Registry
   collection for the durable task or use case.
5. Move lifecycle aliases only after the required checks pass. Confirm before
   moving a production or protected alias.
6. Record immutable `vN` references for reproduction. Use mutable aliases only
   when following the current lifecycle owner is intentional.

## Preserve lineage

```python
import wandb

with wandb.init(entity="ENTITY", project="PROJECT", job_type="train") as run:
    dataset = run.use_artifact("ENTITY/PROJECT/DATASET:v7")
    dataset_dir = dataset.download()

    model = wandb.Artifact(
        name="MODEL_NAME",
        type="model",
        description="Model trained for TASK.",
        metadata={"source_dataset": dataset.name},
    )
    model.add_file("CHECKPOINT_PATH")
    logged_model = run.log_artifact(model, aliases=["candidate"])
```

`run.use_artifact()` and `run.log_artifact()` create input/output lineage
edges. `wandb.Api().artifact()` is appropriate for inspection outside a Run,
but it does not record a consuming edge. Inspect existing edges with
`artifact.logged_by()` and `artifact.used_by()`.

## Design rules

- Keep one stable descriptive name for the logical asset; changed content
  becomes another immutable version.
- Use a simple pipeline-role type such as `dataset`, `model`, `predictions`, or
  `evaluation`.
- Add descriptions and structured metadata that support comparison and search.
- Treat `latest` as most recently logged or linked, never as best or approved.
- Keep exploratory versions in their project. Link versions that passed agreed
  checks into a stable Registry collection.
- Read the exact Registry and collection path before linking. A typo can create
  a new collection.
- Link from a team-owned source project in the same organization. Re-log a
  personal-entity source under a team project first.

```python
run.link_artifact(
    artifact=logged_model,
    target_path="wandb-registry-REGISTRY_NAME/COLLECTION_NAME",
    aliases=["staging"],
)
```

## Mutation checklist

Before linking, moving an alias, changing TTL, or deleting anything, verify:

- the owning team entity and source project;
- the exact version and validation evidence;
- the existing Registry/collection and allowed Artifact type;
- whether the label should be a one-version alias or a many-version tag;
- protected aliases, automations, access rules, and downstream consumers;
- explicit user authorization for the mutation.
