---
name: wandb-models
description: "Use for W&B Models: runs, sweeps, artifacts, configs, metrics, run history, team-member API, best-run selection, model scaling, status breakdowns, and large-project model analysis. Do not use for report authoring, Launch, Signal Builder, or broad project overviews."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# W&B Models, Runs, Sweeps, And Artifacts

Use this skill for read-only analysis of W&B runs, metrics, configs, sweeps,
artifacts, tags, groups, team members, and run histories.

## Fast Rules

- Use `wandb.Api(timeout=60)` or `get_api()` for every W&B API task.
- For exact counts, use `len(api.runs(..., per_page=1, include_sweeps=False, lazy=True))`.
- Do not use `len(list(runs))`, unbounded `list(api.runs(...))`, or history reads without explicit `keys=[...]`.
- Skip metric probing for pure count, status, recent/oldest, team-member, tag, group, and creator tasks.
- Use `lazy=False` only when you need config values; lazy run objects can have empty `run.config`.
- Discover project-specific metric keys before assuming names like `loss` or `accuracy`.
- Cite entity/project, filters, ordering, count method, and any uncertainty in the final answer.
- For hybrid prompts that ask about both W&B runs and Weave evals/traces, answer
  the run half here and use the most specific W&B skill for first-hop coordination.
- Do not use this skill alone for Weave trace/eval counts, report authoring,
  Launch, Signal Builder, or mixed project-overview tasks.

## Exact Artifact Count Protocol

Use this for artifact count, artifact type, collection count, or artifact version
count tasks. Do not infer artifacts from run summaries or files shown in recent
runs.

1. Resolve `entity/project`.
2. Use W&B artifact APIs, not run history, to list collections or artifacts.
3. Count collections and versions separately when the prompt is ambiguous.
4. Print compact totals first. Do not print every collection before the totals,
   because long per-collection output can be truncated and make the final
   numbers look unsupported.
5. Prefer `artifact_collection_summary()` so collection/version semantics are
   counted consistently.

```python
import os
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import artifact_collection_summary, get_api

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
summary = artifact_collection_summary(api, path)
print("Project:", summary["path"])
print("Artifact collections:", summary["collection_count"])
print("Total artifact versions:", summary["version_count"])
print("Artifact types:", summary["type_counts"])
print("Multi-version collections:", summary["multi_version_collections"][:10])
```

Final answer template:

```text
I counted artifacts for {entity}/{project} from the W&B artifact API, not from sampled runs.

- artifact collections: {collection_count}
- artifact versions: {version_count}
- artifact types observed: {artifact_types}
- method: {api_or_helper_method}
```

## Run-Counting Protocol

Use this exact protocol for total runs, status breakdowns, failed/crashed counts,
and failure rates. This protocol is intentionally direct SDK code, not a helper
workflow.

1. Resolve `entity/project` from the prompt or `WANDB_ENTITY` / `WANDB_PROJECT`.
2. Use the W&B Runs API even if the project also has Weave traces.
3. Query exact counts with `per_page=1`, `include_sweeps=False`, and `lazy=True`.
4. Always count `finished`, `failed`, `crashed`, and `running` for status-breakdown prompts.
5. Treat `failed` and `crashed` as distinct W&B states, but if the user says
   "failed runs" generically, also report `failed_or_crashed = failed + crashed`.
6. Compute `failure_rate = failed_or_crashed / total * 100` for generic failure-rate prompts.
7. Before answering, verify the arithmetic and mention any unreported statuses as `other`.
8. Never answer count/status questions from sampled rows or recent runs.

## Helper Imports

```python
import sys
sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import (
    get_api,
    artifact_collection_summary,
    probe_project,
    fetch_runs,
    scan_history,
    compare_configs,
    diagnose_run,
    runs_to_dataframe,
)
```

## Hybrid Runs Plus Weave Prompts

Use the most specific W&B skill first for hybrid routing. If you are already here,
produce the W&B run evidence and clearly mark it as the run half.

Run evidence sequence:

```python
# Part 1: W&B run evidence
import os
import wandb

api = wandb.Api(timeout=60)
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
run_total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))

print({"source": "wandb_runs", "total": run_total, "finished": finished})
```

Final answer template for hybrid tasks after the Weave half is collected:

```text
For {entity}/{project}:
- W&B runs: {run_total} total, {finished} finished (method: wandb.Api runs count).
- Weave evaluations: {eval_count} Evaluation.evaluate calls (method: calls_query_stats with op filter).
- These are different data surfaces; I did not infer one from the other.
```

## Exact Run Counts And Status Recipe

Copy this for run-counting tasks. It is optimized for large projects and for
questions that ask "how many failed" in natural language.

```python
import os
import wandb

api = wandb.Api(timeout=60)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
failed = len(api.runs(path, filters={"state": "failed"}, per_page=1, include_sweeps=False, lazy=True))
crashed = len(api.runs(path, filters={"state": "crashed"}, per_page=1, include_sweeps=False, lazy=True))
running = len(api.runs(path, filters={"state": "running"}, per_page=1, include_sweeps=False, lazy=True))

failed_or_crashed = failed + crashed
known_statuses = finished + failed + crashed + running
other = max(total - known_statuses, 0)
failure_rate = (failed_or_crashed / total * 100) if total else 0.0

print(f"Project: {path}")
print(f"Total: {total}")
print(f"Finished: {finished}")
print(f"Failed: {failed}")
print(f"Crashed: {crashed}")
print(f"Failed or crashed: {failed_or_crashed}")
print(f"Running: {running}")
print(f"Other statuses: {other}")
print(f"Failure rate: {failure_rate:.1f}%")

assert total >= known_statuses or other == 0
assert 0.0 <= failure_rate <= 100.0
```

Final-answer checklist for count/status tasks:

- State the project path and exact count method.
- Include `total`, `finished`, `failed`, `crashed`, `running`, and `other`.
- If the prompt says "failed" generically, include both `failed` and `crashed`,
  then explicitly map `failed_or_crashed = failed + crashed`.
- Compute `failure_rate = failed_or_crashed / total * 100` from the same
  numerator you report.
- Mention `other` if known statuses do not sum to total.

Use this exact final-answer template:

```text
I queried W&B runs for {entity}/{project} with exact `api.runs(..., per_page=1, lazy=True)` counts.

- total: {total}
- finished: {finished}
- failed: {failed}
- crashed: {crashed}
- running: {running}
- other: {other}
- failed_or_crashed: {failed_or_crashed}
- failure_rate: {failure_rate:.1f}% using failed_or_crashed / total
```

## Team Members API

Use this when the user asks how to list members of a W&B team through the API.
The access pattern is `api.team(<team_name>).members`.

```python
import wandb

api = wandb.Api(timeout=60)
team = api.team("TEAM_NAME")
for member in team.members:
    print({
        "name": getattr(member, "name", None),
        "username": getattr(member, "username", None),
        "email": getattr(member, "email", None),
        "admin": getattr(member, "admin", None),
    })
```

## Recent, Oldest, And Creator Lookups

Use bounded ordering and slicing. Do not materialize the full run collection.

```python
import os
import wandb

api = wandb.Api(timeout=60)
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"

recent = api.runs(path, order="-created_at", per_page=10)[:10]
oldest = api.runs(path, order="created_at", per_page=1)[:1]

for run in recent:
    print(run.id, run.name, run.state, run.created_at, getattr(run, "user", None))
if oldest:
    print("Oldest:", oldest[0].id, oldest[0].name, oldest[0].created_at)
```

## Metric Keys And Run Tables

Only probe metric/config keys when the task needs them. Use `probe_project()` to
inspect structure, then fetch only the columns needed for the answer.

```python
import os
import pandas as pd
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import get_api, probe_project, fetch_runs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
info = probe_project(api, path, sample_size=5)
print("Sample metrics:", info.get("sample_metrics", [])[:50])
print("Sample config keys:", info.get("sample_config_keys", [])[:50])

rows = fetch_runs(
    api,
    path,
    metric_keys=["LOSS_KEY", "ACC_KEY"],
    filters={"state": "finished"},
    limit=100,
)
df = pd.DataFrame(rows)
print(df.describe(include="all"))
```

## Best Run By Summary Metric

Use this for "best model", "best run", "highest mAP", "lowest loss", or any
argmax/argmin question. Do not assume the metric key; discover it first.

```python
import os
import pandas as pd
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import get_api, probe_project, fetch_runs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
info = probe_project(api, path, sample_size=10)
print("Candidate metrics:", info.get("sample_metric_keys", [])[:80])
print("Candidate config keys:", info.get("sample_config_keys", [])[:80])

metric_key = "SUMMARY_METRIC_KEY"  # e.g. mAP, val/mAP, accuracy, eval/score
rows = fetch_runs(
    api,
    path,
    metric_keys=[metric_key],
    config_keys=["MODEL_OR_ARCH_KEY", "LEARNING_RATE_KEY"],
    filters={"state": "finished"},
    order=f"-summary_metrics.{metric_key}",  # use + for lower-is-better metrics
    limit=20,
)
df = pd.DataFrame(rows)
print(df[["id", "name", metric_key, "MODEL_OR_ARCH_KEY"]].head(10))
```

Final answer requirements:

- Name the winning run by `id` and `name`.
- State the metric key and value used for ranking.
- Include the relevant config fields that explain why it won.
- If the metric direction is ambiguous, say how you interpreted "best".

For object-detection projects, explicitly look for mAP keys such as `mAP@.5`,
`mAP@.5:.95`, or project-specific validation mAP names. Report the winning run,
architecture or model family, mAP value, and key training settings from the
queried run config.

## Model Scaling And Cohort Tables

Use this for model-size ladders, architecture comparisons, training-best
questions, and "how does performance change as the model gets bigger?"

```python
import os
import pandas as pd
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import get_api, fetch_runs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
rows = fetch_runs(
    api,
    path,
    metric_keys=["MAP_KEY", "PRECISION_KEY", "RECALL_KEY"],
    config_keys=["MODEL_SIZE_KEY", "ARCHITECTURE_KEY"],
    filters={"state": "finished"},
    limit=500,
)
df = pd.DataFrame(rows)
summary = (
    df.groupby(["MODEL_SIZE_KEY", "ARCHITECTURE_KEY"], dropna=False)
    .agg(
        runs=("id", "count"),
        best_map=("MAP_KEY", "max"),
        best_precision=("PRECISION_KEY", "max"),
        best_recall=("RECALL_KEY", "max"),
    )
    .sort_values("best_map", ascending=False)
)
print(summary.to_string())
```

Explain the tradeoff: best metric, number of runs per cohort, and whether larger
models improved quality enough to justify cost or complexity.

## Distinct Config Values With Counts

Use this for architecture or variant histograms. It is safer than guessing from
run names.

```python
import os
from collections import Counter
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import get_api, fetch_runs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
config_key = "ARCHITECTURE_OR_MODEL_KEY"
rows = fetch_runs(
    api,
    path,
    metric_keys=[],
    config_keys=[config_key],
    filters={"state": "finished"},
    limit=1000,
)
counts = Counter(row.get(config_key) or "<missing>" for row in rows)
for value, count in counts.most_common():
    print(value, count)
```

In the final answer, cite the config key used and call out missing/unknown
values separately.

## Config And Run Comparison

Use `lazy=False` when reading config. For two known runs, prefer
`compare_configs()` and direct `summary_metrics.get()`.

For prompts asking for all finished-run config differences, fetch all finished
runs with `lazy=False`, compare only non-internal config keys, and present only
keys that actually vary. Always state that failed/crashed runs were excluded.

```python
import os
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import get_api, compare_configs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
run_a = api.run(f"{path}/RUN_A_ID")
run_b = api.run(f"{path}/RUN_B_ID")

for diff in compare_configs(run_a, run_b):
    print(diff)

for key in ["LOSS_KEY", "ACC_KEY"]:
    print(key, run_a.summary_metrics.get(key), run_b.summary_metrics.get(key))
```

## Run History

Never call `history()` or `scan_history()` without explicit keys. Use the helper
so large histories use the safest available path.

```python
import os
import sys

sys.path.insert(0, "skills/wandb-models/scripts")
from wandb_helpers import get_api, scan_history

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
run = api.run(f"{path}/RUN_ID")
rows = scan_history(run, keys=["LOSS_KEY", "VAL_LOSS_KEY"], max_rows=50_000)
print(f"Rows: {len(rows)}")
print(rows[:3])
```

## Artifacts And Sweeps

Use the W&B API for project artifact collections and sweep relationships. Keep
counts explicit and cite whether you counted collections, versions, or runs.

```python
import os
import wandb

api = wandb.Api(timeout=60)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

artifact_types = api.artifact_types(project, entity_name=entity)
for artifact_type in artifact_types:
    collections = list(artifact_type.collections())
    print(artifact_type.name, "collections:", len(collections))

sweeps = api.project(project, entity).sweeps()
for sweep in sweeps:
    print(sweep.id, sweep.name, len(sweep.runs))
```

## Exact Artifact Collection And Version Counts

Use this for artifact-count tasks. Do not infer artifact counts from run rows.

```python
import os
import wandb

api = wandb.Api(timeout=60)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

collection_total = 0
version_total = 0
by_type = []

for artifact_type in api.artifact_types(project, entity_name=entity):
    collections = list(artifact_type.collections())
    collection_total += len(collections)
    type_versions = 0
    for collection in collections:
        versions = list(collection.versions())
        type_versions += len(versions)
    version_total += type_versions
    by_type.append((artifact_type.name, len(collections), type_versions))

print("Project:", f"{entity}/{project}")
print("Artifact collections:", collection_total)
print("Artifact versions:", version_total)
for type_name, collections, versions in by_type:
    print(type_name, "collections=", collections, "versions=", versions)
```

Final answer template:

```text
I queried W&B artifacts for {entity}/{project}, not run rows.
- artifact collections: {collection_total}
- artifact versions: {version_total}
- by type: {type_table}
```

## Distinct Tags And Groups

For large projects, use server-side GraphQL fields instead of scanning every run.

```python
import os
import wandb
from wandb_graphql.language import parser as gql_parser

api = wandb.Api(timeout=60)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
doc = gql_parser.parse("""
query ProjectTagsAndGroups($entity: String!, $project: String!) {
  project(entityName: $entity, name: $project) {
    tagCounts { name count }
    groupedRuns(groupKeys: ["group"], first: 100) {
      ... on GroupedRunConnection {
        edges { node { group totalRuns } }
      }
    }
  }
}
""")
result = api.client.execute(doc, variable_values={"entity": entity, "project": project})
project_data = result["project"]
print("Tags:", sorted((t["name"], t["count"]) for t in project_data["tagCounts"]))
print("Groups:", [
    (edge["node"]["group"], edge["node"]["totalRuns"])
    for edge in project_data["groupedRuns"]["edges"]
    if edge["node"]["group"]
])
```

## References

- `references/WANDB_SDK.md` contains the deeper W&B SDK surface.
