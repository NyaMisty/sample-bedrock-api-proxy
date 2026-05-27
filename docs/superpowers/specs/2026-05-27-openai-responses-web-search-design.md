# OpenAI Responses Web Search Compatibility - Design

**Status:** Draft for review
**Date:** 2026-05-27
**Scope:** `/openai/v1/responses` only

## Summary

Add proxy-side compatibility for OpenAI Responses API requests that include the hosted `web_search` tool. The feature will intercept only `POST /openai/v1/responses` requests whose `tools` array contains `{"type": "web_search"}` or legacy `{"type": "web_search_preview"}`. Those requests will run through the existing proxy-managed web search capability used by the Anthropic Messages API, then return OpenAI Responses-shaped JSON or SSE.

All other OpenAI passthrough routes remain unchanged. In particular, `/openai/v1/chat/completions` stays pure passthrough and is explicitly out of scope.

## Goals

- Support `web_search` for OpenAI Responses clients against Bedrock-backed models that do not natively expose OpenAI hosted web search.
- Reuse the existing web search provider stack: Tavily/Brave provider selection, domain filtering, max result limits, and feature toggles.
- Preserve current passthrough behavior for Responses requests that do not include web search.
- Support both non-streaming and `stream: true` Responses requests.
- Keep the OpenAI wire contract as close as practical: Responses-shaped JSON, Responses SSE event names, OpenAI-style text annotations, and normalized usage tracking.

## Non-Goals

- No Chat Completions web search support in this change.
- No implementation of OpenAI search-preview Chat Completions models.
- No persisted Responses state for locally synthesized web-search responses. Existing passthrough CRUD endpoints still forward to bedrock-mantle.
- No offline/cache-only web search mode. Requests that set `external_web_access: false` will be rejected in the local compatibility path.
- No support for all future OpenAI web search parameters. Unsupported parameters will either be ignored when harmless or rejected when they imply behavior the proxy cannot provide.

## Existing Context

The OpenAI passthrough router currently forwards `POST /openai/v1/responses` to bedrock-mantle and records usage from the returned JSON or final SSE event. Streaming Responses already has compatibility code that synthesizes `event:` lines for data-only upstream frames.

The existing Anthropic `WebSearchService` detects Anthropic web search tools, replaces them with a normal `web_search` custom tool, calls Bedrock in an agentic loop, executes searches server-side, injects tool results back into the conversation, and post-processes citation markers into citation metadata.

The new design should reuse the loop and provider behavior, but not leak Anthropic response block shapes (`server_tool_use`, `web_search_tool_result`, `web_search_result_location`) to OpenAI clients.

## Triggering Behavior

`POST /openai/v1/responses` will choose between two paths:

1. **Default passthrough path:** no web search tool present. Existing behavior is unchanged.
2. **Local web search path:** at least one tool has `type` equal to `web_search` or `web_search_preview`.

The request still goes through the existing proxy auth, rate limit, budget, and model mapping flow before execution. Model mapping continues to resolve the requested model to the Bedrock model id used internally.

## Request Compatibility

### Supported Inputs

The first implementation supports the common Responses request shapes already accepted by bedrock-mantle passthrough:

- `model`
- `input` as a string
- `input` as an array of role/content items
- `instructions`
- `temperature`
- `top_p`
- `max_output_tokens`
- `stream`
- `tools`
- `tool_choice` values compatible with `auto` or a web search selection

The adapter will convert these into an internal Anthropic `MessageRequest`:

- `model` maps directly after model-id resolution.
- `instructions` becomes `system`.
- `max_output_tokens` becomes `max_tokens`.
- `input` becomes `messages`.
- The OpenAI web search tool becomes an Anthropic `web_search_20250305` tool marker.

### Search Tool Options

Supported option mapping:

- `filters.allowed_domains` maps to `allowed_domains`.
- `filters.blocked_domains` maps to `blocked_domains`.
- top-level or tool-level `user_location` maps to the existing approximate `UserLocation` schema where fields overlap.
- `search_context_size` is accepted but treated as advisory. Actual result volume remains governed by `WEB_SEARCH_MAX_RESULTS`.

Unsupported or constrained options:

- `external_web_access: false` returns HTTP 400 with an OpenAI-style error because the existing provider path performs live external search.
- `return_token_budget` is not supported in the first implementation. If present, return HTTP 400 rather than silently providing misleading behavior.
- Multiple web search tools in one request are treated as one effective search capability. Conflicting filters return HTTP 400.

## Execution Architecture

Create an OpenAI Responses web search adapter in `app/api/openai_passthrough/web_search.py` with four responsibilities:

1. Detect whether a raw Responses body requires local web search handling.
2. Convert the raw OpenAI Responses body into an internal `MessageRequest`.
3. Invoke the existing `WebSearchService` with `BedrockService`.
4. Convert the resulting `MessageResponse` or Anthropic SSE stream into OpenAI Responses JSON/SSE.

The router should stay thin:

- Parse body.
- Resolve model id.
- If no web search tool, use existing passthrough code.
- If web search tool, call the adapter.
- Record usage using the existing OpenAI passthrough usage path, with `api_surface="responses"`.

The preferred internal call is the existing `WebSearchService.handle_request()` and `handle_request_streaming()` so behavior stays aligned with `/v1/messages`.

## Response Mapping

### Non-Streaming JSON

The adapter returns a Responses-shaped object:

```json
{
  "id": "resp_<id>",
  "object": "response",
  "created_at": 1779840000,
  "status": "completed",
  "model": "<resolved-model>",
  "output": [
    {
      "id": "ws_<id>",
      "type": "web_search_call",
      "status": "completed"
    },
    {
      "id": "msg_<id>",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "answer text",
          "annotations": [
            {
              "type": "url_citation",
              "url": "https://example.com",
              "title": "Example",
              "start_index": 0,
              "end_index": 12
            }
          ]
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  }
}
```

`output_text` should be populated for SDK convenience when feasible.

Anthropic citation objects are converted to OpenAI-style `url_citation` annotations. If exact text offsets cannot be derived reliably, annotations should use conservative offsets around the cited sentence or the full text span rather than invalid indexes.

Tool-result detail blocks from the Anthropic response should not be exposed as raw `web_search_tool_result` blocks. A compact `web_search_call` output item is enough for OpenAI-compatible clients.

### Streaming SSE

For `stream: true`, return `text/event-stream` with Responses-style events. The implementation will use the existing hybrid approach: call the local web search loop internally and emit mapped events as blocks are available.

Minimum event sequence:

- `response.created`
- `response.output_item.added` for a `web_search_call` item when the first search happens
- `response.output_item.done` for the `web_search_call`
- `response.output_item.added` for the assistant message
- `response.content_part.added`
- one or more `response.output_text.delta`
- `response.content_part.done`
- `response.output_item.done`
- `response.completed`

If an error happens before headers are committed, return an HTTP JSON error. If an error happens after streaming starts, emit `response.failed` with an OpenAI-style error payload and close the stream.

## Usage Tracking

The adapter will record usage with `api_surface="responses"` using the same `_record_usage()` helper as the passthrough route.

Usage fields map from the internal Anthropic `Usage` object:

- `input_tokens` -> Responses `usage.input_tokens`
- `output_tokens` -> Responses `usage.output_tokens`
- total is the sum of input and output
- `server_tool_use.web_search_requests` is copied to top-level `metadata.web_search_requests` on the synthesized Responses object and logged for debugging.

Search provider fees are not modeled separately in the existing usage schema. That remains unchanged.

## Errors

Use OpenAI-style error bodies:

```json
{
  "error": {
    "message": "Invalid web search configuration",
    "type": "invalid_request_error"
  }
}
```

Expected mappings:

- `ENABLE_WEB_SEARCH=false` -> 400 `invalid_request_error`
- Missing provider API key or provider unavailable -> 503 `upstream_error`
- Unsupported `external_web_access=false` -> 400 `invalid_request_error`
- Unsupported `return_token_budget` -> 400 `invalid_request_error`
- Invalid or conflicting filters -> 400 `invalid_request_error`
- Bedrock/model errors -> follow existing proxy error conventions where possible

## Testing

Add focused tests under `tests/integration/test_openai_passthrough/` and unit tests for any adapter helpers:

- Non-web-search Responses request still forwards to upstream unchanged.
- Responses request with `tools: [{"type":"web_search"}]` does not call upstream `/responses`.
- Non-streaming local web search returns Responses-shaped JSON with `web_search_call`, assistant message, `output_text`, annotations, and usage.
- Streaming local web search returns Responses SSE event names and a terminal `response.completed`.
- `external_web_access: false` returns 400.
- `return_token_budget` returns 400.
- Domain filters map to existing `WebSearchToolDefinition`.
- Chat Completions with web search-shaped fields remains unchanged and forwards upstream.

Tests should mock `WebSearchService` or the search provider/Bedrock boundary, not perform live search.

## Rollout

No new feature flag is required for the first version. The behavior is gated by the existing `ENABLE_OPENAI_PASSTHROUGH` route mount and `ENABLE_WEB_SEARCH` search feature flag.

Documentation should mention that OpenAI Responses web search compatibility is proxy-managed and currently supports live search only.

## Open Questions Resolved

- Scope is Responses API only.
- Streaming is included in the first implementation.
- Chat Completions remains pure passthrough.
- Existing Message API web search remains the source of implementation behavior.
