# Existing OpenTelemetry pipelines

Export OTLP/HTTP to `https://trace.wandb.ai/agents/otel/v1/traces`, or the
configured deployment endpoint. Use HTTP Basic authentication with username
`api` and the runtime W&B API key as password. Construct the Authorization
header in-process; never print the key or its base64 encoding.

Set resource attributes `wandb.entity` and `wandb.project`. Agent spans need
`gen_ai.operation.name` values `invoke_agent`, `chat`, or `execute_tool` with
correct parentage; changing the endpoint does not reshape flat traces.

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
