# Signal Prompt Examples

Use these patterns when drafting Weave signal prompts. Keep prompts binary,
bounded, and explicit about false positives.

## Low-Quality LLM Response

```text
Question: Does the model response fail to answer the user's request?

IS:
- The response refuses without a valid safety reason.
- The response is unrelated to the user's prompt.
- The response claims success but omits the requested artifact or answer.
- The response contains obvious hallucinated project, run, or trace details.

NOT:
- The response asks a necessary clarification.
- The response gives a partial answer and clearly states remaining uncertainty.
- The response is short but directly answers the request.

Default: false. Only mark true when the failure is visible in the trace fields.
```

## Failed Tool Or Trace Outcome

```text
Question: Does this trace represent a failed execution that should be reviewed?

IS:
- `summary.weave.status` is error or descendant_error.
- The output contains an exception, traceback, timeout, or failed external call.
- A scorer or feedback field marks the result as failed.

NOT:
- The trace is successful but low confidence.
- The trace is incomplete because it is still running.
- The trace has a warning that did not affect the final result.

Default: false. Prefer status/scorer fields over free-text guesses.
```

## Monitor-Ready Structure

Every signal prompt should state:

- The op name or monitor filter.
- The interpolated fields, such as `{output}` or `{output[response]}`.
- The binary decision label or score.
- The false-positive exclusions.
- Whether the signal is explanation-only, create-new, or update-existing.
