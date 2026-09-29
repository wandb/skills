# Trace fidelity

Applies to explicit spans, auto-instrumentation, and existing OTel pipelines.
Backend arrival and correct operation names do not prove fidelity.

## Boundaries

Before editing, identify:
- Each span producer (main agent, delegated agent, worker) and its exporter/version.
- The provider/proxy usage contract for each producer, including streaming counters.
- The turn owner across requests and the client/server tool dispatcher.
- Whether delegated jobs are awaited, detached, or independent.

Use code and actual request/response records; ask when ownership is unclear.

## Turns

- One turn covers one human submission through all model/tool steps to delivery,
  failure, or cancellation. Create the owner **outside the internal loop**.
- Preserve turn identity and trace context across client roundtrips. Use supported
  SDK context/batch APIs without holding requests open or changing behavior.
  Restore the original turn on continuation; do not start another `invoke_agent`
  root merely because a new server request arrived.
- **Cross-request option:** persist the originating `trace_id` and root `span_id`
  alongside the application run ID. Parent continuation spans to that root using
  supported context/batch APIs; emit/finalize the root once at delivery, failure,
  or cancellation, retaining its original start time. Do not reopen an ended root
  or emit a copy per step.
- Verify turn IDs remain stable across continuations and change per submission.
  Missing/reused IDs leave grouping unresolved. Timestamps, roles, and equal text
  cannot establish identity; grouping attributes do not repair parentage.

## Tools and delegation

- Record actual dispatch/completion with the original timestamps, context, and
  tool-call ID. History replay or delayed acknowledgment is not a new execution.
  Record receipt separately; unknown execution timing stays unknown.
- **Client tools:** instrument the actual client dispatcher, or export its original
  execution records with tool-call IDs and timestamps. Receipt events alone do
  not verify execution coverage or latency. If dispatcher access is unavailable,
  report that gap; do not recreate synthetic execution spans to fill it.
- Preserve available exit codes, structured error results, and attempt identity
  when converting tool results to receipt events. Map outcomes using the tool
  contract; missing status is **unknown**, not success.
- Attribute failures to original attempts. Preserve retries and real duplicate
  executions; never deduplicate by error text. Missing provenance means uncertain
  attribution.
- **Awaited jobs:** carry originating trace/span context in the job envelope,
  restore it in the worker, and finish children within the parent lifetime.
- **Detached jobs:** retain a causal link to the launcher; linked traces may not
  merge into a turn view. Independent jobs need no invented parent.
- Correlate late results by originating job/call ID, not the next turn. Do not
  delay responses. Bound look-ahead; acceptance is not completion, and an expired
  observation window leaves completion pending/unknown.

## Usage

- Record final per-request usage once, on the LLM span. Do not sum cumulative
  stream updates or copy usage into additive parent fields. Missing is not zero.
- OTel input totals include cached tokens. Check scope, units, and bucket overlap;
  add cache counts only when the provider contract says they are disjoint.
  Provider name or count magnitude alone cannot determine the conversion.
- **Synthetic example:** `uncached=3`, `cache_read=80`, `cache_creation=20`
  gives `input_total=103`. An already-inclusive `input_total=103` stays `103`.
- Flag/reject impossible totals, such as `cache_read + cache_creation > input_total`.
  Compare with provider responses; nonnegative cost alone proves nothing.
  Preserve raw evidence rather than clamping costs or rewriting history.
  Tracing estimates are not bills. Validate each producer separately: correct
  main-agent usage does not establish correct delegated-agent or worker usage.
- Retain response ID, actual response model, finish reasons, and instrumentation
  version when available.

## Messages and answers

- Respect content suppression/redaction. When recording content, preserve exact
  model inputs and structured outputs, including tool calls/results. Use
  synthetic data for payload tests.
- **Tool-only `respond`:** preserve the raw call; record the delivered answer
  separately on the enclosing turn using supported APIs. Do not invent assistant
  text for rendering. If the provider emitted both text and a call, retain both.
- Establish sends from response/call identity and delivery events, not equal text
  or a `respond` call alone. Preserve repeated submissions and actual duplicate sends.
- Normalize proven tool-result envelopes losslessly with matching call IDs;
  preserve the original envelope and client version. Never relabel genuine user
  text or delegated instructions to hide bubbles. Moving context to
  `system`/`developer` requires a separately tested model-request change.
- Report renderer gaps. Exclude proven history replay from new `repetition`,
  `context_loss`, or `tool_failure` evidence without hiding fresh output.

## Conformance check

Test the real instrumentation/serializer with an in-memory exporter or local
OTLP receiver. Use independently recorded provider responses, execution events,
and deliveries as ground truth, not the exported spans themselves.

Cover applicable cases:
- **Turns:** two submissions with identical text; multiple model calls and a
  client-tool roundtrip within the first. Assert identity and parentage across
  every continuation, not just within one well-formed step trace. Verify one root
  finalization for each terminal outcome: delivery, failure, and cancellation.
- **Tools:** match each independently recorded dispatch/attempt to its execution
  span and timing. A receipt-only event is not a match. Replay history without
  creating execution; preserve real retries and unknown timing. Test success,
  failure, and missing status, including outcome preservation in receipt events.
- **Usage:** inclusive, exclusive, cumulative-stream, and missing counters;
  account for each request once, without parent double-counting. Exercise every
  producer, including delegated agents, rather than only the main path.
- **Messages:** tool-only `respond`, provider text plus `respond`, legacy tool
  envelopes, and genuine user text. Match raw parts and actual deliveries.
- **Delegation:** awaited, detached, and independent jobs; late completion;
  overlapping conversations without context leakage.

Known-bad controls must fail the corresponding assertions: per-step turns,
history-as-execution, missing execution spans, dropped available error codes,
doubled cache counts, copied parent usage, invented assistant text, and dropped
delegation context.

Report cases run, inapplicable, or blocked. Verify backend receipt and rendered
turns separately when authorized; local tests do not prove production delivery
or UI behavior.
