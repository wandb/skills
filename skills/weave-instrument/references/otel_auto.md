# Auto-instrumentation caveats

Verify these against the installed Weave version and actual emitted spans.

| Condition | Action |
| --- | --- |
| Python plain `openai`, default `use_otel_v2=True` | Call `weave.integrations.patch_openai()` after init for flat calls, or use Session SDK LLM spans for agent traces. `WEAVE_USE_OTEL_V2=false` is another capture mode. |
| Python Google ADK | Import ADK before init, or call `patch_google_adk()`; the root-module hook misses a later `google.adk` import. |
| Existing global `TracerProvider` | Init backs off. Add [Weave export](otel_endpoint.md) to that provider; do not assume init attached an exporter. |
| Node ESM auto-capture | Launch with `node --import=weave/instrument your-entry.mjs`. This flag is unnecessary for explicit Session SDK spans. |
| Auto already emits the desired tree | Do not wrap the same operations manually. |

Python `weave.session.agent_name_override("name")` is a context manager for
renaming auto-generated agent spans. It creates no span; the override takes
precedence over the framework's native name and integration default.

If traces are absent, check runtime auth, init order, provider ownership,
and the language-specific cases above before adding more instrumentation.
