<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Bounded Artifact operations

Use this reference for nontrivial Artifact listing, filtering, inventory,
lineage, cleanup analysis, and storage accounting. Single known-version SDK
mechanics live in `WANDB_SDK.md`; pipeline and promotion design lives in
`ARTIFACTS_AND_REGISTRY.md`.

```bash
R=skills/wandb-primary/scripts/wandb_run_ops.py
H=skills/wandb-primary/scripts/artifact_helpers.py
```

## Route by task

| Need | Use |
|---|---|
| One known version, lineage, or download | `api.artifact("ENTITY/PROJECT/NAME:vN")` |
| Project type/collection shape | `python "$R" artifact-inventory --entity E --project P` |
| Filter/rank versions by size, age, type, alias, or tag | `python "$H" versions --entity E --project P ...` |
| Summarize selected manifest entries and bounded file contents | `python "$R" artifact-file-summary ...` |
| Locate checkpoint outputs for selected runs | `python "$R" checkpoint-locations ...` |
| Delete or change TTL | Confirm first, then use the SDK mechanics in `WANDB_CONCEPTS.md` |

## Scale posture

- Inspect metadata before downloading bytes. Size, creation time, aliases,
  tags, digest, file count, and manifest entry paths/sizes are available
  without `download()`.
- Prefer the bundled helpers to hand-written type-to-collection-to-version
  walks. They emit sample/cap caveats that must appear in the conclusion.
- Paginate and project only required fields. Do not materialize an entire large
  project to answer a bounded question.
- Treat a partial page as evidence for that page, not a project-wide total.
- Use `artifacts(first: 0) { totalCount }` when only a count is required.

## Helper recipes

```bash
uv run python "$R" artifact-inventory --entity E --project P
uv run python "$H" versions --entity E --project P \
  --type model --order=-size --limit 50 --scan-limit 500
uv run python "$R" artifact-file-summary --entity E --project P \
  --artifact NAME:VERSION --file-regex 'metrics|manifest' --max-files 10
uv run python "$R" checkpoint-locations --entity E --project P \
  --run-name-include TOKEN
```

`artifact-inventory` summarizes types and collections without downloading file
contents. `artifact_helpers versions` emits filtered version rows from the
Public API. `artifact-file-summary` downloads only selected entries after
examining the manifest.

## Projected GraphQL

When the helpers do not expose a required join, query only the fields needed.
The shared `execute_graphql` helper in `wandb_run_ops_lib.common` accepts a raw
query string and returns the decoded data payload.

```graphql
query CollectionVersions($id: ID!, $first: Int!, $after: String) {
  artifactCollection(id: $id) {
    name
    artifactMemberships(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      edges {
        node {
          versionIndex
          createdAt
          aliases { alias }
          artifact { id size createdAt isGenerated tags { name } }
        }
      }
    }
  }
}
```

For project collection search without nested version bodies, use
`artifactCollectionsSearch(..., artifactsLimit: 0)`. Download only after the
metadata identifies the exact required version and files.
