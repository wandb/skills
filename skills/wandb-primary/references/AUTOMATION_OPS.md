<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Automation querying & creation

Read this card before listing or mutating W&B Automations (triggers), and
when the user asks to set alerts, Slack/email/push notifications, or to be
notified when something happens. That is an Automation. Do not lead with
`run.alert()` or User Settings → Alerts (scriptable run alerts); those are
the older path.

Use the helper for list and create:

`H=skills/wandb-primary/scripts/automation_helpers.py`

Confirm with the user before create / update / delete: the next matching event
will fire. GraphQL `createFilterTrigger` is the API; do not tell the user
Automations are UI-only.

An Automation = **scope** + **event** + optional **filter** + **action**.
Team workspaces only (not personal). `scopeID` is the GraphQL `id` of the
project, collection, or entity, not `entity/project`. `RUN_STATE` /
`RUN_METRIC*` are Multi-tenant Cloud and Dedicated Cloud only (not
self-managed).

## When to use what

| Need | Prefer |
|---|---|
| List project automations | `python $H list --entity E --project P` |
| List registry automations | `python $H list --registry NAME --org ORG` |
| List one collection | `python $H list --collection-id ID` (exact scope only) |
| Slack / webhook destination IDs | `python $H integrations --entity TEAM` |
| Placeholders for a prompt / payload | `python $H template-vars --event ADD_ARTIFACT_ALIAS` |
| Create | `python $H create …` (re-reads by id; project lists can lag) |
| Fetch one by id | `python $H get --id ID` |
| Custom composition | GraphQL via this helper's `execute_graphql` below (not `api._service_api`) |

`python $H --help` / `python $H <sub> --help` for flags.

## Registry vs project

Registry-wide automations are **PROJECT-scoped** on the backing project
`wandb-registry-{EXACT_REGISTRY_NAME}` under the **org entity**. They are not
on individual collections.

On `/orgs/{org}/registry/{registry}` (Automations tab), the UI lists
`project(name: "wandb-registry-{registry}", entityName: orgEntity) {
  triggers(projectOnly: true) }`. That is `python $H list --registry NAME --org ORG`.

Do not list a registry by iterating collections. A collection query returns only
automations whose scope is that collection; it misses registry-wide rules
(the usual "alias added to any collection in this registry" case).

`--include-collections` on a registry list also returns collection-scoped rows
(`projectOnly: false`). The Registry Automations tab does not.

If `WANDB_PROJECT` is already `wandb-registry-*`, `python $H list` is enough.

## Helpers

```bash
python $H list --entity E --project P
python $H list --registry nice-overview --org ORG
python $H list --collection-id COLLECTION_GQL_ID
python $H integrations --entity TEAM
python $H template-vars --event ADD_ARTIFACT_ALIAS
python $H create --entity E --project P --name NAME \
  --event ADD_ARTIFACT_ALIAS --alias-regex prod \
  --action NOTIFICATION --slack-integration-id ID \
  --title 'New prod alias' --message '${artifact_version_string} in ${project_name}'
python $H create --registry NAME --org ORG --name NAME \
  --event ADD_ARTIFACT_ALIAS --alias-regex staging \
  --action PUSH_NOTIFICATION --title 'Staged' --body '${artifact_version_string}'
python $H create --entity E --project P --name NAME \
  --event RUN_STATE --run-state FINISHED --sweep SWEEP_ID \
  --action GENERIC_WEBHOOK --webhook-integration-id ID \
  --payload '{"run":"${run_name}","status":"${run_status}"}'
python $H get --id TRIGGER_GQL_ID
```

Create flags for common events: `--alias-regex`, `--tag-regex`, `--run-state`
(repeatable: `RUNNING` / `FINISHED` / `FAILED`), `--sweep`, `--metric-name` +
`--threshold` (`RUN_METRIC`). Raw `--event-filter` for anything else.
Actions: `--action NOTIFICATION|GENERIC_WEBHOOK|PUSH_NOTIFICATION`.

After create, trust `python $H get --id` (or the helper's own re-read). Do not
retry `create` because a project list missed the new row.

## List / create (GraphQL, when the helper is not enough)

`eventFilter` is always a JSON **string**. Artifact events use a filter tree.
Run events nest a **string** `run_filter` plus a typed filter object.

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("skills/wandb-primary/scripts").resolve()))
from wandb_run_ops_lib.common import api
from automation_helpers import execute_graphql

wb = api()
EVENT_FILTER = json.dumps({"$or": [{"$and": [{"alias": {"$regex": "production"}}]}]})
m = """
mutation CreateAutomation($input: CreateFilterTriggerInput!) {
  createFilterTrigger(input: $input) { trigger { id name enabled } }
}
"""
execute_graphql(wb, m, {"input": {
    "name": NAME,
    "triggeringEventType": "ADD_ARTIFACT_ALIAS",
    "scopeType": "PROJECT",
    "scopeID": PROJECT_GQL_ID,
    "eventFilter": EVENT_FILTER,
    "triggeredActionType": "NOTIFICATION",
    "triggeredActionConfig": {"notificationActionInput": {
        "integrationID": SLACK_INTEGRATION_ID, "title": "...", "message": "...",
    }},
    "enabled": True,
}})
```

Empty artifact filter: `{"$or":[{"$and":[]}]}`. Tag: `"tag"` instead of
`"alias"`. `CREATE_ARTIFACT` / `LINK_MODEL` / `UNLINK_ARTIFACT`: empty artifact
filter.

**Run status** (`RUN_STATE`). States: `RUNNING`, `FINISHED`, `FAILED`
(`FAILED` includes crashed / killed / preempted). Fires once per matching run.
`--sweep` / `run_filter` `{"sweep": ID}` selects that sweep's child runs.

```python
EVENT_FILTER = json.dumps({
    "run_filter": json.dumps({"$and": []}),
    "run_state_filter": {"states": ["FINISHED"]},
})
```

**Metric threshold** (`RUN_METRIC`). `cmp_op`: `$gt` `$gte` `$lt` `$lte`
`$eq` `$ne`. `agg_op`: `AVERAGE` `MIN` `MAX`.

```python
EVENT_FILTER = json.dumps({
    "run_filter": json.dumps({"$and": []}),
    "run_metric_filter": {
        "threshold_filter": {
            "name": "loss", "threshold": 1.5, "window_size": 5,
            "cmp_op": "$gt", "agg_op": "AVERAGE",
        }
    },
})
```

**Metric change** (`RUN_METRIC_CHANGE`): `change_filter` with
`prior_window_size`, `current_window_size`, `change_amount` (relative is a
fraction: `0.1` = 10%), `change_type` `ABSOLUTE`/`RELATIVE`, `change_dir`
`INCREASE`/`DECREASE`/`ANY`. **Z-score** (`RUN_METRIC_ZSCORE`):
`zscore_filter` `{name, window_size, threshold, change_dir}`. Weave: copy
`weave_metric_filter` from an existing automation.

`updateFilterTrigger(input: { id, … })`. `deleteTrigger(input: { triggerID })`.

## Events

`CREATE_ARTIFACT` · `ADD_ARTIFACT_ALIAS` · `ADD_ARTIFACT_TAG` ·
`REMOVE_ARTIFACT_TAG` · `ADD_COLLECTION_TAG` · `REMOVE_COLLECTION_TAG` ·
`LINK_MODEL` · `UNLINK_ARTIFACT` · `RUN_STATE` · `RUN_METRIC` ·
`RUN_METRIC_CHANGE` · `RUN_METRIC_ZSCORE` · `WEAVE_METRIC_THRESHOLD`

Advertised on this instance: `serverInfo { features { name isEnabled } }`
(`AUTOMATION_EVENT_*`).

Do not create `UPDATE_ARTIFACT_ALIAS`. Default scope `PROJECT`; collection
events may use `ARTIFACT_COLLECTION`. History states: `PENDING`, `FINISHED`,
`FAILED`, `SKIPPED`.
