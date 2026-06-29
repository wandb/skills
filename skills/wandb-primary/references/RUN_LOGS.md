<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Run Logs (GraphQL)

Reading W&B run console logs. Two surfaces expose log data, backed by
different storage.

## Two log surfaces

1. **Logs tab** — backed by GraphQL `Run.logLines`. Streams log lines
   during execution. Available for running, crashed, and resumed runs
   even when no downloadable file exists.
2. **Files tab** — `output.log` (written on successful completion) or
   multipart log files under `logs/` (uploaded incrementally when
   `console_multipart=True`). Not created if the run crashes before
   flushing.

## State-specific behavior

- **Running runs:** Logs visible in the Logs tab before `output.log` exists.
- **Crashed runs:** Streamed lines may exist in the Logs tab, but the final
  buffered lines before the crash may be missing (never flushed/uploaded).
- **Resumed runs:** Files-tab logs may be overwritten by the resumed
  execution. Check `logLines` for the full history.
- **Large logs (>100k lines):** The Logs tab has a retention window. Oldest
  lines may be unavailable via both the UI and `logLines`.

## Console modes

- **`console_multipart=False` (default):** One `output.log` uploaded on
  completion. Nothing downloadable while running.
- **`console_multipart=True`:** Incremental uploads during execution. Earlier
  parts downloadable while running. Full history preserved across parts for
  large logs.

### Multipart file layout

When `console_multipart=True`, log parts are stored under a `logs/` directory
with timestamped filenames:

```text
logs/output_YYYYMMDD_HHMMSS_NNNNNNNNN.log
```

Example from a run with two parts:
```text
logs/output_20251219_162652_408917000.log
logs/output_20251219_162655_920170000.log
```

Rollover is controlled by two settings (whichever triggers first):
- `console_chunk_max_bytes` — size-based rollover threshold (bytes)
- `console_chunk_max_seconds` — time-based rollover threshold (seconds)

To list multipart files via the API:

```python
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
log_files = [f for f in run.files() if f.name.startswith("logs/")]
```

To download and stitch them in order:

```python
import os, tempfile
with tempfile.TemporaryDirectory() as d:
    for f in sorted(log_files, key=lambda f: f.name):
        f.download(root=d, replace=True)
    parts = sorted(os.listdir(os.path.join(d, "logs")))
    full_log = []
    for p in parts:
        with open(os.path.join(d, "logs", p)) as fh:
            full_log.extend(fh.readlines())
```

`logLines` (GraphQL) returns the unified log regardless of console mode.
The Logs tab UI reads from `logLines`, not from these files.

## Fetch all lines (paginated)

```python
import wandb
from wandb_gql import gql

api = wandb.Api()

query = gql("""
query RunLogs($entity: String!, $project: String!, $run: String!,
              $first: Int, $after: String) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      state
      logLineCount
      logLines(first: $first, after: $after) {
        edges {
          cursor
          node { number timestamp level label line }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
""")

after = None
lines = []
while True:
    res = api.client.execute(query, variable_values={
        "entity": ENTITY, "project": PROJECT, "run": RUN_ID,
        "first": 5000, "after": after,
    })
    conn = res["project"]["run"]["logLines"]
    lines.extend(edge["node"] for edge in conn["edges"])
    if not conn["pageInfo"]["hasNextPage"]:
        break
    after = conn["pageInfo"]["endCursor"]
```

## Fetch tail only

For crash diagnosis or quick checks:

```python
query = gql("""
query RunLogTail($entity: String!, $project: String!, $run: String!,
                 $last: Int) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      state
      logLineCount
      logLines(last: $last) {
        edges { node { number timestamp level label line } }
        pageInfo { hasPreviousPage startCursor }
      }
    }
  }
}
""")

res = api.client.execute(query, variable_values={
    "entity": ENTITY, "project": PROJECT, "run": RUN_ID,
    "last": 100,
})
```

## LogLine fields

| Field | Description |
|---|---|
| `number` | Line number (0-indexed) |
| `timestamp` | ISO timestamp |
| `level` | Log level (`info`, `error`, etc.) |
| `label` | Source label (e.g. `stdout`, `stderr`) |
| `line` | The log line content |

## Useful Run fields

| Field | Description |
|---|---|
| `logLineCount` | Total number of log lines available |
| `logLines(first:, after:, last:, before:)` | Paginated log connection |

## Search and navigation

The UI Logs tab search only covers the currently loaded chunk (~10k lines).
`logLines` supports pagination across the full log:

- `first` / `after` — forward pagination from the start
- `last` — fetch the tail
- Cursor pagination to sample middle ranges

## Gotchas

1. **`run.files()` absence ≠ no logs.** The Logs tab can have data when
   `output.log` is not downloadable.
2. **Default API timeout is 19s.** This is fine for short logs but may
   time out on large fetches. `wandb.Api(timeout=...)` accepts a custom
   timeout in seconds.
3. **Crashed run logs may be incomplete.** Final buffered lines may never
   reach W&B. Absence of an exception in `logLines` does not prove
   there was no crash.
4. **Tail fetch is available for large logs.** `logLines(last: N)` returns
   the last N lines without paginating from the start.
5. **Progress bars / dynamic output.** Carriage-return-style terminal output
   may not render correctly in downloaded multipart files. `logLines`
   matches what the Logs tab shows.
