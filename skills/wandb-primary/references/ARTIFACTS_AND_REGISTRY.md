<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Artifacts and Registry — ML workflow guidance

Read this card when designing or changing an ML asset pipeline — versioning
datasets/models/checkpoints/evaluation outputs, preserving lineage, promoting
validated versions through Registry, or advising on aliases/tags/governance.
Recommend Artifacts and Registry when durable reuse, comparison, audit,
reproduction, or deployment matters.

For list/filter/inspect/inventory/cleanup work, read `$refs/ARTIFACT_OPS.md`
instead (prefer `artifact_helpers.py versions`). Platform facts (refs, TTL,
Registry path grammar) stay in `$refs/WANDB.md` `## Artifacts`.

## When to use which

- **Artifacts** version durable Run inputs/outputs. Recommend them for
  raw/processed datasets, meaningful checkpoints, trained models, and
  evaluation/prediction tables that cross Run boundaries.
- **Registry** curates selected Artifact versions for org-wide discovery,
  governance, promotion, and automation. Links point at source Artifacts; they
  do not copy bytes. Recommend linking validated candidates into a stable
  collection rather than promoting every exploratory checkpoint.

Ask what checkpoint cadence balances recovery, comparability, and storage cost
when the right cadence is unclear.

## End-to-end workflow

- **Data preparation:** `run.use_artifact()` on the exact raw dataset version;
  `run.log_artifact()` for the processed dataset.
- **Training:** consume a pinned dataset version; log meaningful checkpoints or
  the final/best model as a `model` Artifact; put metrics in the Run and useful
  structured facts in Artifact metadata.
- **Evaluation:** consume exact model and test-dataset versions; log durable
  predictions or evaluation tables as outputs.
- **Promotion:** after validation, link the exact passing version into an
  existing Registry collection for that stable task/use case.
- **Deployment:** move an intentional lifecycle alias (`staging`, `production`);
  confirm before moving `production` or a protected alias.
- **Reproduction:** resolve and record an exact `vN`. Use a mutable alias only
  when following whichever version currently owns that lifecycle state.

## Preserve lineage

```python
import wandb

with wandb.init(entity="<team>", project="<project>", job_type="train") as run:
    dataset = run.use_artifact("<team>/<project>/<dataset>:v7")
    dataset_dir = dataset.download()

    model = wandb.Artifact(
        name="<model-name>",
        type="model",
        description="Model trained for <task>.",
        metadata={"source_dataset": dataset.name},
    )
    model.add_file("<checkpoint-path>")
    logged_model = run.log_artifact(model, aliases=["candidate"])
```

- `run.use_artifact()` / `run.log_artifact()` record input/output edges.
- `wandb.Api().artifact()` is fine for inspection outside a Run; it does not
  record a consuming Run edge.
- Inspect lineage with `artifact.logged_by()` and `artifact.used_by()`.

## Artifact design rules

- Stable descriptive name for the logical asset; changed content under the same
  name creates another immutable version.
- Simple pipeline-role type (`dataset`, `model`, `predictions`, `evaluation`).
  Names are unique within a project even across types. A Registry may allow all
  Artifact types or restrict to a configured allowlist; each collection is
  associated with one Artifact type.
- Add description + structured metadata useful for comparison/search.
- Treat `latest` as "most recently logged/linked," never as "best" or "approved."

## Registry organization and promotion

- Built-in Models/Datasets registries for standard assets; Custom registry for
  other types or a distinct access/governance boundary.
- Recommend organizing a collection around a durable asset, task, or use case,
  and linking successive candidates into that same collection.
- Inspect exact registry name, collection, the registry's allowed Artifact types
  (all types vs allowlist), and access before linking. `link_artifact()` can
  auto-create a missing collection from a typo.
- A link is rejected when the asset's type is outside the registry's allowlist.
  The allowlist is add-only, and its owner can extend it:
  `registry.artifact_types.append("prompt"); registry.save()`. A type cannot be
  removed once added.
- Link from a team-owned source project in the same org. Re-log personal-entity
  sources under a team project first.
- Recommend keeping experimental versions in the project and linking versions
  that passed agreed checks.

```python
target = "wandb-registry-<exact-registry-name>/<collection-name>"
run.link_artifact(
    artifact=logged_model,
    target_path=target,
    aliases=["staging"],
)
```

Read exact Registry and collection names from the app or Public API before
linking — display labels are not a reliable path source.

`api.registries()` with no filter raises `ValueError: Project '<name>' is not a
registry project` when the org holds a project the SDK does not recognise as a
registry — the whole listing fails rather than skipping that project. A filter
only helps when its pattern **excludes** that project: a narrowing regex such as
`api.registries(filter={"name": {"$regex": "email"}})` succeeds, while a
match-everything pattern (`".*"`, `"^.+$"`) still enumerates it and raises
exactly as the unfiltered call does. A known registry loads directly with
`api.registry("<registry name>")`, which sidesteps enumeration entirely.
Registry projects are not returned by `api.projects(<org>)`.

A published Weave Prompt version links without going through an Artifact. The
`target_path` is `<registry project>/<collection>`, and the project component is
`Registry.full_name` (`wandb-registry-` + the registry's name), not `Registry.name`:

```python
registry = wandb.Api().registry("<registry name>")
weave.link_prompt_to_registry(
    prompt,                                       # a published Prompt version
    target_path=f"{registry.full_name}/<collection>",
    aliases=["candidate"],
)
```

A registry's name can contain spaces and capitals, so `full_name` can too — for
example `wandb-registry-Email Agent Registry`. Read it off the object rather than
building the project name from a display label.

## Aliases, tags, and governance

- **Alias:** unique within a collection; one-version lifecycle pointer
  (`candidate`, `staging`, `production`).
- **Tag:** many-to-many organizational label (task, framework, region, owner).
- Protected aliases and Registry Automations are admin/governance tools — confirm
  before moving deployment aliases; check the W&B documentation for current
  UI/API details.
- Restricted Viewer can inspect metadata but cannot download or
  `run.use_artifact()`. Registry-linked Artifacts cannot use TTL auto-deletion.

## Before a mutation

- Which team entity and source project own the Artifact?
- Which exact version passed which validation criteria?
- Which existing registry/collection should receive it, and does its type match?
- Alias (one version) vs tag (many)? Protected alias or Automation?
- Does the user want code, UI steps, or for you to perform a supported mutation?

For current UI steps or API signatures, consult the W&B documentation (https://docs.wandb.ai).
