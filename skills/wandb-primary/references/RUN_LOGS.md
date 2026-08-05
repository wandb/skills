<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Run Logs

Reading W&B run console logs. Two surfaces expose log data, backed by
different storage.

## Two log surfaces

1. **`logLines` surface** — the streamed `logLines` view (read via GraphQL). Streams
   during execution and is available for running and (usually) crashed runs
   even when no downloadable file exists — the most reliable first read for
   crash/running-run diagnosis.
2. **downloadable run files** — `output.log` is rebuilt after finalization and
   usually exists for finished and crashed runs; multipart files under `logs/`
   upload incrementally when `console_multipart=True`. A hard kill before
   flushing may leave no downloadable log.

## State-specific behavior

- **Running runs:** Logs visible in the `logLines` surface before `output.log` exists.
- **Crashed runs:** Streamed lines may exist in the `logLines` surface, but the final
  buffered lines before the crash may be missing (never flushed/uploaded).
- **Resumed runs:** On resume, the default console overwrites the prior segment's
  `output.log` and `logLines`, leaving only the latest segment; with
  `console_multipart=True` each segment is a separate timestamped part and all are
  kept. Read the multipart parts for a resumed run's full history — or confirm
  `logLines` covers the segment you need.
- **Large logs (>100k lines):** The `logLines` surface has a retention window.
  Oldest lines may be unavailable from the service.

## Console modes

- **`console_multipart=False` (default):** A single `output.log`, uploaded only
  on completion — nothing is downloadable while the run is in progress, and a
  **resume overwrites it** — only the latest segment is kept, in both the file
  and `logLines`.
- **`console_multipart=True`:** The console is uploaded incrementally as
  timestamped `logs/output_*.log` parts, downloadable while the run is still
  running. Parts accumulate — across size/time rollover *and* resumes — so the
  full history is preserved across the parts.

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

Without rollover settings, a single part is written per run segment. Because a
resumed run never overwrites earlier parts, a resumed run still yields one file
per segment even when no rollover threshold is configured.

To list multipart files via the API:

```python
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
log_files = [f for f in run.files() if f.name.startswith("logs/")]
```

To download and stitch them in order (parallelize the downloads with a thread
pool when there are many parts):

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

## Reading the log (recommended order)

W&B rebuilds `output.log` server-side from the streamed console, so it *usually*
exists for finished and crashed runs alike (normal exit, uncaught exception, most
hard kills). So read the file first, and fall back to `logLines` only when there
is no file: a **still-running** run (not yet finalized), or a run **killed
without graceful cleanup** (e.g. an OOM kill that dies before the console is
flushed) — which can leave no `output.log`.

1. **Tail (crash diagnosis)** — download `output.log` and read the end; the
   traceback / OOM / final stderr lives there. For a multipart run (no single
   `output.log`), download the newest `logs/output_*.log` part instead.
2. **Full scan** — download `output.log` (or the parts) and grep locally.
3. **No `output.log`?** — a still-running run, or one killed before flush (OOM):
   fall back to `logLines` (below).

### Tail from output.log (default for crash diagnosis)

```python
import tempfile
import wandb

api = wandb.Api(timeout=120)
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
with tempfile.TemporaryDirectory() as d:
    path = run.file("output.log").download(root=d, replace=True).name
    tail = open(path).read().splitlines()[-200:]
```

For a multipart run, download the newest `logs/output_*.log` part for the tail
(see *Multipart file layout* above for listing/stitching all parts).

### Reading logLines (running runs, or crashes with no output.log)

Use `logLines` when there is no `output.log` to read: a still-running run, or a
run killed before its console was flushed (e.g. OOM). When `output.log` exists
(most finished/crashed runs), read that instead — it's a single static file.

`logLineCount` is a cheap way to size the log before fetching and pick a strategy:

| `logLineCount` | Strategy |
|---|---|
| ≤ 5,000 | single page (`first: 5000`) |
| 5,000–100,000 | paginate forward (`first: 5000` + `after` cursor) |
| > 100,000 | tail first (`last: 100`), paginate only if needed |
| `None` | unknown size — tail first |

If the information you want is likely near the **end** of the log (a crash, final
metrics), start from the tail (`last: N`) instead of paging from the start. You
can also page in **reverse** — `last: 5000` with `before: <cursor>` — to walk
backwards from the end.

```python
import sys
import wandb

sys.path.insert(0, "skills/wandb-primary/scripts")
from wandb_run_ops_lib.common import execute_graphql

api = wandb.Api(timeout=120)

tail_query = """
query RunLogTail($entity: String!, $project: String!, $run: String!, $last: Int) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      state
      logLineCount
      logLines(last: $last, useImprovedPagination: true) {
        edges { node { number timestamp level label line } }
      }
    }
  }
}
"""

res = execute_graphql(api, tail_query, {
    "entity": ENTITY, "project": PROJECT, "run": RUN_ID, "last": 200,
})
```

For a bounded full scan of a running run's stream (rarely needed), paginate with
hard limits:

```python
full_query = """
query RunLogs($entity: String!, $project: String!, $run: String!,
              $first: Int, $after: String) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      logLines(first: $first, after: $after, useImprovedPagination: true) {
        edges { node { number timestamp level label line } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""

def fetch_log_lines(api, *, entity, project, run_id,
                    page_size=1000, max_pages=20, max_lines=20_000):
    after, last_cursor, lines = None, None, []
    for _ in range(max_pages):
        conn = execute_graphql(api, full_query, {
            "entity": entity, "project": project, "run": run_id,
            "first": page_size, "after": after,
        })["project"]["run"]["logLines"]
        edges = conn.get("edges") or []
        if not edges:
            break
        lines.extend(e["node"] for e in edges)
        info = conn["pageInfo"]
        after = info.get("endCursor")
        if (not info.get("hasNextPage") or after is None
                or after == last_cursor or len(lines) >= max_lines):
            break
        last_cursor = after
    return lines[:max_lines]
```

`fetch_log_lines` pages forward (`first`/`after`). To page **backwards** from the
end — cheaper when you only need the last few thousand lines of a huge log — use
`last`/`before` instead: fetch `last: 5000`, then pass each response's
`pageInfo.startCursor` as `before` (select `pageInfo { hasPreviousPage startCursor }`).

## `logLines` surface fields

Each line shown in the streamed `logLines` surface carries:

| Field | Description |
|---|---|
| `number` | Line number (0-indexed) |
| `timestamp` | ISO timestamp |
| `level` | Log level (`info`, `error`, etc.) |
| `label` | Writer label distinguishing concurrent writers to the same run in shared mode — not the stdout/stderr stream |
| `line` | The log line content |

## Searching a long log

The streamed `logLines` surface search only covers the currently loaded chunk (~10k lines).
For a full-log search, prefer the downloaded files when they exist (`output.log`
or the multipart parts) and grep locally; otherwise use **bounded** `logLines`
pagination with a regex filter (see above), never an unbounded scan.

## Gotchas

1. **`run.files()` absence ≠ no logs.** The `logLines` surface can have data when
   `output.log` is not downloadable.
2. **Default API timeout is short.** Log queries can fail on large runs or slow
   backend responses. `wandb.Api(timeout=...)` and the `timeout` parameter on
   `execute_graphql(...)` accept a custom timeout in seconds. `logLines(last: N)`
   fetches only the tail, and smaller `first:` values reduce page size.
3. **Crashed run logs may be incomplete.** Final buffered lines may never
   reach W&B. Absence of an exception in `logLines` does not prove
   there was no crash.
4. **Progress bars / dynamic output.** Carriage-return-style terminal output
   may not render correctly in downloaded multipart files. `logLines`
   matches what the `logLines` surface shows.
