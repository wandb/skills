---
name: wandb-leet
description: "Inspect W&B runs from a coding agent with the wandb leet JSON commands (runs, history, summary, console, system, session) and the wandb mcp server: which runs exist, how a run is doing, what a metric did, why it crashed, what the console says, for runs in a local wandb directory or on the W&B server by URL, and steer the LEET terminal UI a user has open. Use when the user trains with wandb and asks about a run, its metrics, its logs or its system metrics."
---
<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# W&B LEET for agents

`wandb leet` is LEET, the W&B terminal UI (Lightweight Experiment Exploration
Tool). Its query verbs answer the same questions as JSON: one compact object
per call on stdout, read from the `.wandb` files of a local wandb directory or
from the W&B server when PATH is a run or project URL. Every query error is
one compact `leet.error/1` object on stderr with exit code 1, flag mistakes
included:

```
{"error":"path_not_found","message":"path does not exist: ./nope","schema":"leet.error/1"}
```

`wandb leet --help` must list `runs`, `history`, `summary`, `console`,
`system` and `session`; a release without them lists only `config`,
`inspect`, `run` and `symon`.

## Rules

- Use the JSON verbs. Never launch the interactive TUI from an agent unless
  the user asks to see it: `wandb leet`, `wandb leet <path-or-url>`,
  `wandb leet run` and `wandb leet symon` take over the terminal and do not
  return. `wandb leet inspect` prints every raw record as text when piped;
  use `summary` instead.
- Read the envelope before the data: `schema`, `state`, `exit_code`,
  `truncated`, `has_more`, `next_cursor`, `warnings`, `skipped_keys`.
- Poll with `--after`; do not `--follow`. `--follow` blocks until the run
  exits or `--idle-timeout` (default 10m) passes, and works for local runs
  only. If the user wants it, pass a short `--idle-timeout`.
- Ask for bins, not rows, for anything longer than a few hundred steps: one
  call, up to N bins, `min`, `max`, `avg`, `last` and `has_nan` per bin.
- The objects are compact. Shape them with `jq` or a short Python snippet
  and report the numbers; do not paste whole objects.

## Verbs

PATH is a wandb directory, a run directory, a `.wandb` file or a W&B project
or run page URL. Without PATH the configured wandb directory is used
(`./wandb` in the working directory or `$WANDB_DIR`); a missing one is
`path_not_found`. `--run RUN` picks a run in a directory by its directory
name, ID or display name; the default is the newest run. A run URL needs no
`--run`.

| Verb | Answers | Most useful flags |
|---|---|---|
| `runs [PATH]` | which runs exist: `run_id`, `name`, `state`, `state_detail`, `exit_code`, `step`, `tags`, `notes`, `last_write`; a project URL lists the server's runs | `--filter QUERY` (the LEET sidebar language, e.g. `'name:resnet cfg.lr>=0.001'`), `--glob`, `--summary-keys 'val/loss,train/*'` (local only), `--limit N` (200) |
| `history --keys K [PATH]` | what a metric did: rows or bins per key | `--keys 'train/loss,val/*'` (required), `--bins N`, `--last N`, `--min-step N --max-step N`, `--x-axis _runtime`, `--limit N` (1000), `--after CURSOR` |
| `summary [PATH]` | how the run is doing: `state`, `exit_code`, `step`, `summary`, `config`, `history_keys` (`count`, `last`, `last_step` per logged key) and the last 50 console lines | `--keys 'val/*'` |
| `console [PATH]` | what the logs say, ANSI stripped, a progress bar redrawn in place is one line | `--last N` (200), `--grep REGEX`, `--since 10m` or an RFC 3339 time, `--limit N` with `--after CURSOR` to page from the start |
| `system [PATH]` | CPU, memory, GPU, disk and network over `_timestamp`; keys carry the `system.` prefix | `--keys 'system.gpu.*.gpu,system.memory*'`, `--bins N` (required for a URL), `--last N`, `--since 10m` |
| `session [PATH] METHOD` | steer the LEET a user has open on PATH | `--params JSON` |

Start with `summary`: `history_keys` says which keys the run logged and how
often, so you can pick `--keys` for `history` without reading the series. A
glob never matches keys starting with `_`; a glob that matches nothing leaves
`keys` empty with a `no key matches` warning.

## Envelope

- `schema`: `leet.runs/1`, `leet.history/1`, `leet.summary/1`,
  `leet.console/1` or `leet.system/1`. Adding a field never bumps the version.
- `source`: `kind` `local` or `remote`, `path`, `url` (the API URL the
  launcher resolved, not the page URL you passed) and, for one run, `run_id`,
  `project` and `entity`.
- `state`: `running`, `finished` (exit code 0), `failed` (an exit code
  other than 0 or 254), `crashed` (exit code 254, or a local run with no exit
  record and no write for 10 minutes) or `unknown`. `state_detail` carries
  `exit code 1` locally and the server's own word remotely, where `exit_code`
  is null.
- `truncated`: a limit cut something. `has_more` with `next_cursor`: more
  pages exist now, call again with `--after`. `warnings`: strings saying what
  was left out and why.
- `history` and `system` add `keys` (what came back), `requested_keys`,
  `skipped_keys` (matched keys that never held a number), `coverage`
  (`first_x`, `last_x` and the requested window), `retrieval_method`
  (`local_scan`, `remote_history` or `remote_bucketed`), `exact` (false when
  corrupt data was skipped or the server aggregated) and `rows_scanned`.
- `history` alone adds `axes` (the x axis of each key) and `total_steps`
  (null until a local page reaches the end of the file; a URL reports it on
  every page). `system` is always on `_timestamp` (`x_axis`) and its
  `coverage` has `requested_since`.
- `console` adds `total_lines` and `matched_lines`; `runs` adds `total`.
- NaN and infinities are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`.

## Rows, bins and polling

Rows are flat objects with `_step`, `_runtime`, `_timestamp` and the selected
keys, 1000 per call. `--last N` keeps the last N rows (for a URL, the rows of
the last N steps); `--min-step` and `--max-step` bound the steps inclusively;
`--x-axis _runtime` puts every key on one axis and drops rows without a finite
value for it.

`--bins N` returns `series[].bins[]` with `x0`, `x1`, `count`, `min`, `max`,
`avg`, `last` and `has_nan`. Locally the bins are equal-width with one set of
edges per axis, so bin i lines up across keys; empty bins are left out, so
fewer than N can come back. For a URL the server chooses the edges: about N
bins come back, `x0` and `x1` are the first and last x it observed, and
`exact` is false. A bin where a key held only NaN has `has_nan` true and null
aggregates.

Paging and live runs share one loop. `has_more` true: call again with
`--after` now. `state` `running` with nothing more: sleep, then call again
with `--after`. While the run runs the cursor is never null, so a quiet poll
continues where it left off; it is null once the run exited and everything up
to its exit record was returned.

```bash
cur=
while out=$(wandb leet history --keys train/loss ${cur:+--after "$cur"} ./wandb); do
  jq -c '.rows[]' <<<"$out"                             # use the rows
  cur=$(jq -r '.next_cursor // empty' <<<"$out")
  jq -e '.has_more' <<<"$out" >/dev/null && continue     # another page now
  [ "$(jq -r .state <<<"$out")" = running ] || break     # exited: done
  sleep 30
done
```

The same loop pages `console --limit N` and remote `history`. `--last` is a
tail with no cursor (`truncated` says it cut lines); only `--limit` pages. A
cursor belongs to one run: a cursor from another run, or a local cursor given
with a URL, is `bad_request`.

## Remote runs

Pass the W&B page URL as PATH: `https://wandb.ai/<entity>/<project>` for
`runs`, `https://wandb.ai/<entity>/<project>/runs/<run-id>` for the per-run
verbs (a run URL given to `runs` is `bad_request`). Credentials come from
`wandb login`; there is nothing to pass. A request the server does not answer
fails after 60 s. The objects keep the local shape, with `source.kind`
`remote` and `source.url` on the API host the page URL resolved to:

- `runs`: `run_key`, `file`, `exit_code` and `step` are null; `start_time` is
  the creation time and `last_write` the last heartbeat; `--summary-keys` is
  `not_available` (read `summary <run-url>`); `truncated` when `--limit`
  stopped the listing, plus a `more runs may exist` warning when the server
  had further pages.
- `summary`: `history_keys` is null with a warning; `step` is the summary's
  `_step`.
- `history`: rows come from the run's export and live history
  (`remote_history`, `exact` true); `--bins` is aggregated by the server
  (`remote_bucketed`). Every key is on `_step`, with a warning when the run
  defines other axes.
- `console`: `--last` is a tail; `--grep` and `--since` on a tail filter only
  the last N lines the server returned and say so in `warnings`. Page with
  `--limit` and `--after` to search the whole log.

## Steering an open LEET

When the user has LEET open on a directory, `wandb leet session PATH METHOD`
sends one JSON-RPC request to it and prints the reply; PATH comes before
METHOD. Call `status` first:

```
$ wandb leet session ./wandb status
{"id":1,"jsonrpc":"2.0","result":{"cursor_run":"offline-run-20260623_185811-y8k00lv3","focused_metric":null,"metrics_filter":{"mode":"regex","query":""},"mode":"workspace","pending_input":null,"pinned_run":"offline-run-20260623_185811-y8k00lv3","runs_filter":{...},"selected_runs":["offline-run-20260623_185811-y8k00lv3"],"system_metrics_filter":{...},"wandb_dir":"/abs/wandb"}}
```

| Method | `--params` | Effect |
|---|---|---|
| `status` | none | `mode` (`workspace`, `run` or `inspect`), `cursor_run`, `selected_runs`, `pinned_run`, the three filters, `focused_metric`, `pending_input` |
| `runs.select` | `{"filter": "name:sim", "exclusive": true}` or `{"keys": ["0skytt9t"]}` | select by sidebar filter, or by directory name, ID or display name; `exclusive` deselects the rest; adds `unindexed` |
| `runs.deselect` | `{"keys": ["0skytt9t"]}` or `{"all": true}` | `all` keeps the pinned run |
| `runs.pin`, `runs.open` | `{"key": "0skytt9t"}` | pin a run; open its run view (`mode` becomes `run`). `runs.close` returns to the workspace |
| `filter.set` | `{"target": "metrics", "query": "loss"}` | `target` is `runs`, `metrics` or `system`; optional `mode` |
| `metrics.focus` | `{"key": "train/loss"}` | page to the chart and focus it; `-32602` when the filter hides it |
| `step.goto` | `{"step": 500}` | crosshair on a step; in the run view also the console line logged nearest to it; adds `step` |
| `action`, `actions` | `{"name": "workspace.next_page", "key": "n"}` | press a key binding by name; `actions` lists the bindings of the current view |

Every method that changes the view replies with the full `status` object plus
its own fields, so you see the effect. A `name:` filter matches the display
name; use `keys` for a run ID or directory name. Errors are JSON-RPC error
objects on stdout with exit code 1: `-32602` for bad params, a filter that
matches no run, or an `action` that would start typing (use `filter.set` or
`runs.select`); `-32001` `busy` while the user is typing a filter or answering
a prompt (`status` still answers, with `pending_input` set; wait and retry).
A `no_session` error on stderr means no LEET shows that directory: do not
open one, answer from the query verbs instead. Remote workspaces cannot be
steered: `session` with a URL is `bad_request`.

## MCP

`wandb mcp` serves the same verbs to an MCP host over stdio:

```json
{"mcpServers": {"wandb": {"command": "wandb", "args": ["mcp"]}}}
```

Tools: `leet_runs`, `leet_history`, `leet_summary`, `leet_console`,
`leet_system`, `leet_session`, `leet_view`. Inputs are the flags with
underscores (`min_step`, `x_axis`, `summary_keys`); list flags are arrays
(`"keys": ["train/loss"]`); `leet_history` requires `keys`, `leet_session`
requires `method` and takes `params` as an object; there is no `follow`.
`path` takes the same directories, files and URLs as the CLI and defaults to
`$WANDB_DIR` or the wandb directory in the working directory. A result's
`structuredContent` is the object the CLI prints and `content[0].text` is its
JSON; an error is the `leet.error/1` object with `isError` true, including an
input that fails the schema (`unknown field "bogus"` is `bad_request`). URLs
must be on the server you are logged in to (`wandb login`, then restart the
server); another host is `bad_request`. `leet_view` opens the LEET panel
(`ui://leet/panel`: the runs of a local directory and one chart per key,
refreshed every 15 s while a run runs) in a host that renders MCP Apps;
elsewhere it returns `leet.view/1` with the run list, and the numbers come
from `leet_history`.

```
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"leet_history","arguments":{"path":"./wandb","run":"2w7tm747","keys":["train/loss"],"bins":2}}}
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"axes\":...}"}],"isError":false,"structuredContent":{"axes":{"train/loss":"_step"},...,"series":[{"bins":[{"avg":8.399200213778201,"count":500,...,"x0":1,"x1":500.5},{...}],"count":1000,"key":"train/loss","x_axis":"_step"}],"state":"finished",...}}}
```

## Troubleshooting

| `error` | Means | Do |
|---|---|---|
| `path_not_found` | PATH does not exist, or there is no `wandb` directory where you ran it | pass the directory |
| `bad_request` | a flag or param: an unknown option, a negative N, a regex that does not compile, `history` without `--keys`, a cursor from another run, a run URL given to `runs`, or in `wandb mcp` a URL on another host than the configured server | fix the call; the message says what |
| `not_available` | `--follow` on a URL; `system` on a URL without `--bins`; `--summary-keys` on a project URL; a URL with no stored API key; a remote history page whose export takes over 60 s | poll with `--after`; add `--bins N`; read `summary <run-url>`; `wandb login` (then restart `wandb mcp`) |
| `server_too_old` | `console --limit` and `--after` paging on a W&B server older than 0.77 | use `--last` |
| `run_not_found` | `--run` matched nothing; the server does not know the run or project, or the key cannot read it | check `runs` |
| `internal` | a read or IO failure | retry once, then report it |

## Worked examples

Paths are shortened to `/abs/wandb` and `...` marks cut output.

### Which runs exist, and how is one doing

```
$ wandb leet runs --filter 'cfg.lr>=1e-3' --summary-keys val/loss --limit 2 ./wandb
{"runs":[{...,"exit_code":0,"last_write":"2026-06-24T06:52:32Z","name":"sim-unstable-L32-1782283724-2",...,"run_id":"2w7tm747",...,"state":"finished","state_detail":"exit code 0","step":1000,"summary":{"val/loss":5.785150990445261},...},{...}],"schema":"leet.runs/1","source":{"kind":"local","path":"/abs/wandb","url":null},"total":12,"truncated":true,"warnings":[]}
```

12 runs match and 2 are shown (`truncated`). Then the one you care about:

```
$ wandb leet summary --keys 'val/*' --run 2w7tm747 ./wandb
{...,"exit_code":0,...,"history_keys":{"val/loss":{"count":1000,"last":5.785150990445261,"last_step":1000},"val/perplexity":{...}},...,"schema":"leet.summary/1",...,"state":"finished","state_detail":"exit code 0","step":1000,"summary":{"val/loss":5.785150990445261,"val/perplexity":13.640432726091793},...,"warnings":[]}
```

Answer: finished with exit code 0 at step 1000, val/loss 5.79. Without
`--keys`, `history_keys` lists every logged key (1009 for this run).

### What a metric did

```
$ wandb leet history --keys train/loss --bins 4 --run 2w7tm747 ./wandb
{"axes":{"train/loss":"_step"},...,"exact":true,"exit_code":0,"has_more":false,"keys":["train/loss"],"next_cursor":null,...,"retrieval_method":"local_scan","rows_scanned":1000,...,"series":[{"bins":[{"avg":9.863586562002087,"count":250,"has_nan":false,"last":10.720389491199562,"max":36.37471971924943,"min":5.801817763700558,"x0":1,"x1":250.75},{"avg":6.934813865554327,...},{...},{"avg":4.7224391711406,...,"x0":750.25,"x1":1000}],"count":1000,"key":"train/loss","x_axis":"_step"}],"skipped_keys":{},"source":{...},"state":"finished","total_steps":1000,"truncated":false,"warnings":[],"x_axis":null}
```

Answer: the average fell from 9.86 to 4.72 over the four quarters; the `max`
of 36.4 in the first quarter is a spike. To look at it, read rows with
`--min-step` and `--max-step`, or ask the console.

### Why it failed

```
$ wandb leet summary --run y8k00lv3 ./wandb
{...,"exit_code":1,...,"name":"sim-stellar-L32-1782266291-0",...,"schema":"leet.summary/1",...,"state":"failed","state_detail":"exit code 1","step":159,...}

$ wandb leet console --last 3 --run y8k00lv3 ./wandb
{"exit_code":1,"has_more":false,"lines":[{"content":"    time.sleep(step_interval)","label":"","level":"error","number":14,"timestamp":"2026-06-24T01:59:34Z"},{...},{"content":"KeyboardInterrupt",...,"level":"error","number":16,...}],"matched_lines":17,"next_cursor":null,"run_id":"y8k00lv3","schema":"leet.console/1","source":{...},"state":"failed","total_lines":17,"truncated":true,"warnings":[]}
```

Answer: `failed` with exit code 1 at step 159; the traceback on stderr
(`level` `error`) ends in `KeyboardInterrupt`, so the process was interrupted.
For a longer tail use `console --last 50`, or `--grep 'Traceback|nan'` to
jump to the lines that matter; a run that died without an exit record is
`crashed` instead.

### A run on the W&B server

```
$ wandb leet history --keys 'loss,acc' --last 2 https://wandb.ai/trustmebro/leet-live-sysmetrics/runs/e947k3ph
{"axes":{"acc":"_step","loss":"_step"},"coverage":{"first_x":70,"last_x":71,...},"exact":true,"exit_code":null,"has_more":false,"keys":["acc","loss"],"next_cursor":null,...,"retrieval_method":"remote_history","rows":[{"_runtime":350.63694925,"_step":70,"_timestamp":1792536107.058923,"acc":0.9698026165776815,"loss":0.0301973834223185},{...,"_step":71,...,"loss":0.028724639654239433}],"rows_scanned":2,"run_id":"e947k3ph",...,"schema":"leet.history/1","skipped_keys":{},"source":{...},"state":"finished","state_detail":"finished","total_steps":71,"truncated":false,"warnings":[],"x_axis":null}

$ wandb leet system --keys 'system.cpu,system.memory*' --bins 2 https://wandb.ai/trustmebro/leet-live-sysmetrics/runs/e947k3ph
{"coverage":{"first_x":1792535758.531505,"last_x":1792536116.5019703,"requested_since":null},"exact":false,"exit_code":null,"has_more":false,"keys":["system.cpu","system.memory.used",...],...,"retrieval_method":"remote_bucketed","rows_scanned":181,...,"schema":"leet.system/1","series":[{"bins":[{"avg":0.375993549823761,"count":1,...},{"avg":0.020667347979421418,"count":24,...},{...},{...}],"count":181,"key":"system.cpu","title":"Process CPU","unit":"%","x_axis":"_timestamp"},{...}],"skipped_keys":{},"source":{...},"state":"finished","state_detail":"finished","truncated":false,"warnings":[],"x_axis":"_timestamp"}
```

`--bins 2` came back as 4 bins: the server chose the edges (`exact` false).
