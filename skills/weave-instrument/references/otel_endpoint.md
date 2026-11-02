# OpenTelemetry export

Use this for an existing pipeline, or for an application-owned provider that
exports without `weave.init`: from settings the application already keeps, and
without contacting the server at startup. The Session SDK emits through the
global `TracerProvider`, so its spans reach Weave through this exporter.

Export OTLP/HTTP to `https://trace.wandb.ai/agents/otel/v1/traces`, or the
configured deployment endpoint. Use HTTP Basic authentication with username
`api` and the runtime W&B API key as password. Construct the Authorization
header in-process; never print the key or its base64 encoding.

Set resource attributes `wandb.entity` and `wandb.project`. Agent spans need
`gen_ai.operation.name` values `invoke_agent`, `chat`, or `execute_tool` with
correct parentage; changing the endpoint does not reshape flat traces.

Create spans with the Session SDK even with your own provider: it writes the
message, usage, and tool-result attributes the Agents view reads. Hand-written
`gen_ai.*` attributes are easy to get almost right (`gen_ai.output` is not
`gen_ai.output.messages`), and the view then shows empty messages and zero
usage.

For an environment-configured exporter, set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`,
`OTEL_EXPORTER_OTLP_TRACES_HEADERS`, and `OTEL_RESOURCE_ATTRIBUTES` through the
application's secret/configuration mechanism.

For an application-owned Python provider, add a
`BatchSpanProcessor(OTLPSpanExporter(endpoint=..., headers=...))` using
`opentelemetry.sdk.trace.export` and
`opentelemetry.exporter.otlp.proto.http.trace_exporter`. Preserve existing
processors and set the routing attributes on the provider's Resource.
`weave.trace.urls.otel_traces_endpoint()` resolves the configured Weave endpoint.

`weave.init()` does not attach an exporter to an existing global provider.
Flush the configured pipeline on process exit and verify backend receipt.
