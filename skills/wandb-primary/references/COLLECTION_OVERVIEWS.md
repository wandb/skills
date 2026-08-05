<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Artifact Collection Overview drafts

Use this workflow when the user asks to generate a W&B Artifact Collection
Overview draft. The helper fetches bounded collection evidence, validates
selected membership IDs and metric names through W&B, and creates the draft.

```bash
H=skills/wandb-primary/scripts/collection_overview_helpers.py
uv run --with wandb python "$H" --help
```

Creating a draft is an external write. Fetching context and metric candidates
is read-only; run `create-draft` only when the user requested creation.

## Plan contract

Send an ordered `blocks` list:

```json
{
  "blocks": [
    {"kind": "HEADING", "text": "Collection Overview"},
    {"kind": "PARAGRAPH", "text": "Short evidence-grounded summary."},
    {"kind": "BULLET_LIST", "items": ["Strength", "Limitation"]},
    {
      "kind": "METRIC_GRID",
      "membershipIDs": ["MEMBERSHIP_ID"],
      "panels": [
        {
          "metricName": "eval/accuracy",
          "chartType": "LINE",
          "chartTitle": "Accuracy by version"
        }
      ]
    }
  ]
}
```

Supported prose blocks are `HEADING`, `PARAGRAPH`, and `BULLET_LIST`.
`METRIC_GRID` panels require a returned `metricName`; `SCATTER` also requires a
returned `xMetricName`. Chart types are `LINE`, `BAR`, and `SCATTER`. Optional
axis bounds must be numeric; use log scale only when justified by the evidence.

## Workflow

1. Obtain the exact collection ID and optional focus from the user's request or
   available W&B context.
2. Fetch bounded collection metadata and version candidates:

   ```bash
   uv run python "$H" context --collection-id COLLECTION_ID
   ```

3. Select at most six returned `membershipID` values. Prefer relevant aliases,
   tags, and recent versions.
4. Fetch metric candidates for only those versions:

   ```bash
   uv run python "$H" metric-candidates \
     --collection-id COLLECTION_ID \
     --membership-id MEMBERSHIP_ID_1 \
     --membership-id MEMBERSHIP_ID_2
   ```

5. Build a concise plan from the collection type, metadata, aliases, tags,
   returned metrics, and user guidance. Save it to a temporary JSON file.
6. Create the draft and report the returned identifiers without exposing raw
   layout payloads unless the user asks to debug them:

   ```bash
   uv run python "$H" create-draft \
     --collection-id COLLECTION_ID --plan-file PLAN.json
   ```

## Evidence rules

- Use only returned membership IDs and metric names.
- Do not invent metric values; W&B resolves them during validation.
- Keep prose independent from optional metric grids because invalid or empty
  grids may be omitted.
- Prefer one or two focused grids to a generic dashboard.
- Use model-card framing only when the collection evidence indicates a model.
  Dataset, prompt, table, and other collection types need different framing.
- Omit unsupported claims. Do not infer safety, compliance, bias, privacy,
  production readiness, intended use, or deployment recommendations from
  metric names alone.
- Keep review notes, TODOs, open questions, and publishing instructions outside
  the draft itself.
- If validation rejects a plan, change it before one bounded retry. Otherwise
  report the error.
