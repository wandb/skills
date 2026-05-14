# W&B SDK Reference

This compact reference contains the W&B SDK surfaces used by the parity
`wandb-models` skill and common W&B tasks.

## API Client

```python
import wandb

api = wandb.Api(timeout=60)
```

Use `timeout=60` for live project queries. Never fabricate counts when API calls
fail.

## Exact Run Counts

```python
path = "entity/project"
total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
failed = len(api.runs(path, filters={"state": "failed"}, per_page=1, include_sweeps=False, lazy=True))
crashed = len(api.runs(path, filters={"state": "crashed"}, per_page=1, include_sweeps=False, lazy=True))
```

For user-facing failure rates, report `failed`, `crashed`, and
`failed_or_crashed = failed + crashed`.

## Run Tables

Use helper-backed selective fetching for large projects:

```python
from wandb_helpers import fetch_runs, probe_project

info = probe_project(api, path, sample_size=10)
rows = fetch_runs(
    api,
    path,
    metric_keys=["metric_key"],
    config_keys=["config_key"],
    filters={"state": "finished"},
    limit=500,
)
```

Discover metric and config keys before assuming names.

## Team Members

```python
team = api.team("team_name")
members = team.members
```

Each member may expose `name`, `username`, `email`, and `admin`.
