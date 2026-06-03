# Chat Completions via Responses API Design

## Summary

`/openai/v1/chat/completions` remains a Chat Completions-compatible client endpoint, but the proxy no longer forwards it to upstream `/chat/completions`. Instead, it translates the client request to an OpenAI Responses API request, forces `store=false`, sends it to upstream `/responses`, and translates the upstream result back to Chat Completions format.

This keeps existing OpenAI SDK clients working while aligning with Bedrock's future Responses-only upstream surface.

## Scope

In scope:

- `POST /openai/v1/chat/completions` non-streaming requests.
- `POST /openai/v1/chat/completions` streaming requests.
- Existing model mapping, per-key provider endpoint resolution, Bedrock header passthrough, upstream error handling, and usage tracking.
- Token usage preservation, including cache-read tokens and reasoning tokens when upstream includes them.

Out of scope:

- Changing `/openai/v1/responses` behavior.
- Adding a runtime feature flag for the old direct `/chat/completions` upstream path.
- Supporting OpenAI endpoint families beyond Chat Completions and Responses.

## Request Translation

The chat-completions route converts the incoming request body to a Responses API body:

- `model` is resolved through existing model mapping before translation.
- `messages` becomes Responses `input`.
- `max_tokens` becomes `max_output_tokens` when present.
- Common generation and tool fields are copied when present: `temperature`, `top_p`, `tools`, `tool_choice`, `parallel_tool_calls`, `metadata`, `reasoning`, `response_format`, `stream`.
- `store` is always set to `False` after copying compatible fields, even if the client supplies `store=True`.

The upstream target path is `/responses` for both streaming and non-streaming requests.

## Response Translation

Non-streaming upstream Responses bodies are converted back to Chat Completions bodies:

- `object` becomes `chat.completion`.
- `id`, `created`, and `model` are preserved when present.
- Response output message content is flattened into `choices[0].message.content`.
- Response function-call output is mapped into Chat Completions `tool_calls` where possible.
- Response terminal state maps to `finish_reason`, defaulting to `stop`.
- Responses usage is converted to Chat Completions usage:
  - `input_tokens` -> `prompt_tokens`
  - `output_tokens` -> `completion_tokens`
  - `total_tokens` preserved or computed
  - `input_tokens_details.cached_tokens` -> `prompt_tokens_details.cached_tokens`
  - `output_tokens_details.reasoning_tokens` -> `completion_tokens_details.reasoning_tokens`

Streaming upstream Responses SSE is converted into Chat Completions data-only SSE:

- `response.created` starts a `chat.completion.chunk`.
- `response.output_text.delta` emits a `choices[0].delta.content` chunk.
- Function-call deltas are converted when upstream carries function-call events.
- `response.completed` emits the final chunk including usage when available, followed by `data: [DONE]`.
- No `event:` lines are emitted to Chat Completions clients.

## Usage Tracking

Usage recording must not regress:

- Usage is extracted from Responses-shaped upstream usage.
- Usage is recorded with `api_surface="chat_completions"` to preserve reporting dimensions.
- `cached_tokens` is populated from `input_tokens_details.cached_tokens`.
- Normal input token accounting continues subtracting cached tokens from input tokens via the existing normalization path.
- `reasoning_tokens` is populated from `output_tokens_details.reasoning_tokens` when present.
- If a streaming upstream does not provide usage, the route does not synthesize usage.

## Error Handling

Upstream non-2xx Responses errors are returned verbatim with the upstream status code and JSON body, matching the existing passthrough contract. Timeouts and connection failures continue to use the existing upstream error mapping.

## Testing

Tests must prove:

- `/openai/v1/chat/completions` calls upstream `/responses`, not `/chat/completions`.
- Upstream request contains `store: false`.
- Non-streaming clients receive Chat Completions-formatted responses.
- Streaming clients receive Chat Completions data-only SSE without `event:` lines.
- Usage tracking keeps `api_surface="chat_completions"` and preserves input, output, cached, and reasoning tokens.
- Provider-specific endpoint routing uses `/responses`.
