# Chat Completions via Responses API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `/openai/v1/chat/completions` through upstream `/responses` with `store=false` while preserving Chat Completions client compatibility and usage accounting.

**Architecture:** Add a small translator module beside the OpenAI passthrough router. The router translates chat requests to Responses requests, reuses the existing upstream client and streaming opener, then translates Responses bodies/events back to Chat Completions format. Usage tracking continues to record the route as `chat_completions` while normalizing Responses-shaped usage.

**Tech Stack:** FastAPI, httpx, pytest, respx, existing OpenAI passthrough helpers.

---

## File Structure

- Create `app/api/openai_passthrough/chat_responses_adapter.py`: request/body and SSE translation helpers.
- Modify `app/api/openai_passthrough/router.py`: route `/chat/completions` upstream calls to `/responses`, call adapter helpers, and record Responses-shaped usage as `chat_completions`.
- Modify `app/api/openai_passthrough/usage_extractor.py`: allow `normalize_usage(..., "chat_completions")` to accept Responses-shaped usage when the route internally uses Responses.
- Modify `tests/integration/test_openai_passthrough/test_chat_completions.py`: replace direct passthrough assertions with Responses-upstream assertions.
- Modify `tests/integration/test_openai_passthrough/test_provider_endpoint.py`: provider chat-completions tests expect upstream `/responses`.

### Task 1: Failing Non-Streaming Tests

**Files:**
- Modify: `tests/integration/test_openai_passthrough/test_chat_completions.py`
- Modify: `tests/integration/test_openai_passthrough/test_provider_endpoint.py`

- [ ] **Step 1: Update non-streaming route tests**

Change the non-streaming chat completion tests so `respx_mock.post("/responses")` is expected, the upstream body includes `input` and `store is False`, the client response remains Chat Completions format, and usage is still recorded with `api_surface="chat_completions"`.

- [ ] **Step 2: Update provider endpoint tests**

Change provider-specific chat completion tests so the provider and default upstream routes are `/responses` instead of `/chat/completions`.

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_chat_completions.py::test_non_streaming_chat_completions_forwards_and_logs_usage tests/integration/test_openai_passthrough/test_provider_endpoint.py -q
```

Expected: failures show the router is still calling upstream `/chat/completions`.

### Task 2: Non-Streaming Implementation

**Files:**
- Create: `app/api/openai_passthrough/chat_responses_adapter.py`
- Modify: `app/api/openai_passthrough/router.py`
- Modify: `app/api/openai_passthrough/usage_extractor.py`

- [ ] **Step 1: Add adapter helpers**

Implement `chat_request_to_response_request()`, `responses_usage_to_chat_usage()`, and `response_to_chat_completion()`.

- [ ] **Step 2: Route non-streaming chat completions to `/responses`**

Use the adapter before the upstream call, send the translated body to `/responses`, translate successful response JSON back to Chat Completions, and record raw Responses usage with `api_surface="chat_completions"`.

- [ ] **Step 3: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_chat_completions.py::test_non_streaming_chat_completions_forwards_and_logs_usage tests/integration/test_openai_passthrough/test_provider_endpoint.py -q
```

Expected: selected tests pass.

### Task 3: Streaming Tests and Implementation

**Files:**
- Modify: `tests/integration/test_openai_passthrough/test_chat_completions.py`
- Modify: `app/api/openai_passthrough/chat_responses_adapter.py`
- Modify: `app/api/openai_passthrough/router.py`

- [ ] **Step 1: Update streaming tests**

Change streaming chat tests so upstream emits Responses SSE from `/responses`, downstream remains Chat Completions data-only SSE, and usage is recorded from `response.completed.response.usage`.

- [ ] **Step 2: Run streaming tests to verify RED**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_chat_completions.py::test_streaming_chat_completions_forwards_sse_and_records_usage tests/integration/test_openai_passthrough/test_chat_completions.py::test_streaming_chat_completions_does_not_inject_event_lines -q
```

Expected: failures show the router/streamer still expects Chat Completions SSE.

- [ ] **Step 3: Add streaming adapter**

Implement an async generator that consumes Responses SSE lines and yields Chat Completions SSE chunks.

- [ ] **Step 4: Wire streaming route**

Use `open_upstream_stream("POST", "/responses", translated_body, ...)` and return the streaming adapter.

- [ ] **Step 5: Run streaming tests to verify GREEN**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_chat_completions.py -q
```

Expected: chat-completions integration tests pass.

### Task 4: Full Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused OpenAI passthrough tests**

Run:

```bash
uv run pytest tests/unit/test_openai_passthrough tests/integration/test_openai_passthrough -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run lint on touched app files**

Run:

```bash
uv run ruff check app/api/openai_passthrough
```

Expected: no lint errors.
