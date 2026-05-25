# OpenAI Passthrough — Design Document

**Status:** Approved (design)
**Author:** River Xie
**Date:** 2026-05-25

## Summary

Add new client-facing endpoints that accept OpenAI-native API formats (Chat Completions and Responses) and forward them to AWS Bedrock's `bedrock-mantle` endpoint. Existing Anthropic-format endpoints (`/v1/messages`) are untouched.

This is **distinct from** the existing `ENABLE_OPENAI_COMPAT` feature, which converts Anthropic-format requests on `/v1/messages` into OpenAI calls. The new feature exposes OpenAI-format directly so OpenAI SDK clients can hit the proxy without translation.

## Motivation

- OpenAI SDK users want to access non-Claude Bedrock models (gpt-oss-120b, etc.) through their existing OpenAI SDK code with minimal changes.
- The Responses API offers stateful conversation chaining (`previous_response_id`, `store=true`) that has no Anthropic equivalent and is awkward to expose through `/v1/messages`.
- Centralizing all model traffic through one proxy gives unified API key auth, budget tracking, rate limits, usage analytics, and pricing — regardless of wire format.

## Non-Goals

- Cross-format translation: OpenAI-in → Anthropic-out is not a goal. Both directions are OpenAI-format end-to-end on these new endpoints.
- OpenAI features that bedrock-mantle doesn't support (e.g. assistants API).
- OTEL tracing for the new endpoints (deferred to v2).

## Design Decisions

The following were resolved during brainstorming:

| # | Decision | Choice |
|---|---|---|
| 1 | Integration depth | **Full integration** — same proxy API key, budget, rate limit, usage tracking |
| 2 | Model scope | **Allow any model bedrock-mantle accepts** (no Claude-block) |
| 3 | Responses API surface | **Full CRUD** (POST + GET + DELETE + cancel + list_input_items) |
| 4 | URL routing | **`/openai/v1/...` prefix** (matches AWS bedrock-runtime convention) |
| 5 | Request handling | **Raw httpx passthrough** (no Pydantic schemas for OpenAI types) |
| 6 | Model ID mapping | **Apply mapping if exists, else passthrough** |
| 7 | Usage tracking | **Normalize into existing Anthropic-shaped schema** + new `api_surface` and `reasoning_tokens` columns |

## High-Level Architecture

### Module Layout

```
app/api/openai_passthrough/
├── __init__.py          # exposes APIRouter
├── router.py            # FastAPI routes (chat, responses, models, CRUD)
├── client.py            # httpx async client to bedrock-mantle (singleton)
├── usage_extractor.py   # parse usage from JSON body or final SSE event
└── streaming.py         # SSE passthrough + usage extraction tee
```

### Mounting

The router mounts at `/openai/v1` only when the feature flag is enabled, in `app/main.py`:

```python
if settings.enable_openai_passthrough:
    from app.api.openai_passthrough import router as openai_router
    app.include_router(openai_router, prefix="/openai/v1", tags=["OpenAI Passthrough"])
```

### Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/chat/completions` | Streaming + non-streaming |
| POST | `/responses` | Streaming + non-streaming + background |
| GET | `/responses/{response_id}` | Retrieve stored response |
| DELETE | `/responses/{response_id}` | Delete stored response |
| GET | `/responses/{response_id}/input_items` | List input items |
| POST | `/responses/{response_id}/cancel` | Cancel background response |
| GET | `/models` | List models from Mantle |

### Request Flow (POST chat/completions or responses)

1. `verify_api_key` middleware (extended to read `Authorization: Bearer` + existing `x-api-key`)
2. Rate limit check (existing token bucket per API key)
3. Budget check (existing)
4. Parse request body as `dict` (no Pydantic validation)
5. Apply model mapping if exists
6. Forward via httpx to `{OPENAI_BASE_URL}/{path}` with proxy's Bedrock API key in `Authorization`
7. Stream/return response
8. Extract usage → log to `anthropic-proxy-usage` + `anthropic-proxy-usage-stats` with `api_surface` column

### Request Flow (CRUD on /responses/{id})

1. `verify_api_key`
2. Rate limit check (shared bucket with POST)
3. **Skip** budget check and usage logging (no tokens consumed)
4. Forward verbatim to Mantle
5. Return verbatim

## Auth & Middleware Changes

### Header Acceptance

`app/middleware/auth.py::verify_api_key` is extended to accept either `x-api-key` or `Authorization: Bearer`:

```python
async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> ApiKeyInfo:
    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:].strip()
    if not api_key:
        raise HTTPException(401, "Missing API key (x-api-key or Authorization: Bearer)")
    # ... existing lookup logic unchanged
```

This is **backwards compatible**. If both headers are present, `x-api-key` wins (deterministic).

### Rate Limiting

No change. The existing rate limiter is keyed by `api_key_id`; once auth resolves, all endpoints (Anthropic and OpenAI) share the same per-key bucket. A client cannot dodge limits by switching API surfaces.

### Budget

Same. The budget check is per-key and surface-agnostic. POST endpoints check + update; non-POST endpoints (GET/DELETE/list/cancel) skip both since they are free operations.

### Bedrock-mantle Auth (Proxy → AWS)

The proxy uses `OPENAI_API_KEY` (the Bedrock API key, already configured for the existing `ENABLE_OPENAI_COMPAT` feature) as `Authorization: Bearer` to bedrock-mantle:

```python
headers = {
    "Authorization": f"Bearer {settings.openai_api_key}",
    "Content-Type": "application/json",
}
```

### Error Contract

Mantle errors (4xx/5xx) are returned to the client **as-is** — same status code, same JSON body. No wrapping or rewriting. This preserves OpenAI-SDK error semantics so `OpenAIError` subclasses raise correctly client-side.

The only proxy-injected errors are:
- `401` — bad proxy API key
- `429` — proxy rate limit
- `402` — budget exceeded

### Proxy-Injected Headers (upstream)

- `User-Agent: bedrock-api-proxy/<version>`
- `X-Proxy-Request-ID: <uuid>` for log correlation

Both are zero-cost, useful for debugging, and ignored by Mantle.

## Passthrough Client & Streaming

### httpx Client (Singleton)

```python
# app/api/openai_passthrough/client.py
import httpx
from app.core.config import settings

_client: httpx.AsyncClient | None = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=httpx.Timeout(settings.bedrock_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _client
```

### Non-Streaming POST

```python
async def chat_completions(request: Request, api_key_info: ApiKeyInfo = Depends(verify_api_key)):
    body = await request.json()
    body["model"] = resolve_model_id(body.get("model", ""))

    if body.get("stream"):
        return StreamingResponse(
            stream_passthrough("/chat/completions", body, api_key_info, api_surface="chat_completions"),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        "/chat/completions", json=body,
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
    )
    if resp.status_code >= 400:
        return JSONResponse(resp.json(), status_code=resp.status_code)

    data = resp.json()
    log_usage_async(api_key_info, data.get("usage", {}), body["model"], "chat_completions")
    return JSONResponse(data)
```

### Streaming Passthrough

For SSE, we forward bytes line-by-line and *tee* to extract the final `usage` chunk.

- **Chat Completions stream**: requires `stream_options: {"include_usage": true}` from the client. If sent, the second-to-last chunk has `usage`. If not, no usage extracted (proxy logs zero — documented behavior).
- **Responses API stream**: usage is on the `response.completed` event. Always present.

```python
async def stream_passthrough(path, body, api_key_info, api_surface):
    usage_holder: dict = {}
    async with get_client().stream(
        "POST", path, json=body,
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
    ) as resp:
        async for raw_line in resp.aiter_lines():
            yield (raw_line + "\n").encode()
            try_extract_usage(raw_line, usage_holder, api_surface)
    if usage_holder:
        log_usage_async(api_key_info, usage_holder, body["model"], api_surface)
```

`try_extract_usage` is small (~30 LOC) — pattern matches `data: {...}` lines, JSON-parses, looks for `usage` field on completion events.

### CRUD Endpoints

Pure passthrough — forward method, path, body, query params; return status + body unchanged. ~20 LOC for all four combined:

```python
@router.api_route("/responses/{response_id}", methods=["GET", "DELETE"])
async def responses_crud(response_id: str, request: Request, _=Depends(verify_api_key)):
    resp = await get_client().request(
        request.method, f"/responses/{response_id}",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))
```

### Edge Case — `store=true` and Pricing

Mantle stores conversations for 30 days for free per the docs (no separate storage cost). We do not bill for it. If AWS adds a storage charge later, a feature flag can force `store=false`.

## Usage Tracking

### Normalization

```python
# app/api/openai_passthrough/usage_extractor.py
def normalize_usage(raw: dict, api_surface: str) -> dict:
    """Return Anthropic-shaped usage dict + reasoning_tokens."""
    if api_surface == "chat_completions":
        in_tok = raw.get("prompt_tokens", 0)
        out_tok = raw.get("completion_tokens", 0)
        cached = raw.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        reasoning = raw.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    else:  # responses
        in_tok = raw.get("input_tokens", 0)
        out_tok = raw.get("output_tokens", 0)
        cached = raw.get("input_tokens_details", {}).get("cached_tokens", 0)
        reasoning = raw.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    return {
        "input_tokens": in_tok - cached,         # subtract: cache hits billed separately
        "output_tokens": out_tok,                # reasoning already included per spec
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,        # OpenAI APIs don't expose this
        "reasoning_tokens": reasoning,           # new optional column
    }
```

### DDB Schema Additions

Added to `anthropic-proxy-usage`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `api_surface` | string | `"messages"` | One of `messages`, `chat_completions`, `responses` |
| `reasoning_tokens` | integer | `0` | Optional, sparse |

Both are sparse attributes — DynamoDB will not reject existing rows. **No migration required**; old rows simply will not have these fields when read.

### Pricing Lookup

Existing `anthropic-proxy-model-pricing` is keyed by Bedrock model ID. After model mapping, we have the Bedrock ID, so pricing works unchanged. Models missing from the pricing table log usage with `cost=0` and emit a warning (existing behavior).

## Configuration

### New Env Var

Added to `app/core/config.py`:

```python
enable_openai_passthrough: bool = Field(
    default=False, alias="ENABLE_OPENAI_PASSTHROUGH",
    description="Mount /openai/v1/* endpoints (Chat Completions + Responses passthrough to bedrock-mantle)"
)
```

### Reused Vars

- `OPENAI_API_KEY` — Bedrock API key for bedrock-mantle (already exists)
- `OPENAI_BASE_URL` — Mantle endpoint URL (already exists)

### Flag Interaction

`ENABLE_OPENAI_COMPAT` (existing) and `ENABLE_OPENAI_PASSTHROUGH` (new) are **independent** and can be enabled together. They affect different endpoints:

- `ENABLE_OPENAI_COMPAT=True`: routes non-Claude traffic on `/v1/messages` through bedrock-mantle (Anthropic↔OpenAI conversion)
- `ENABLE_OPENAI_PASSTHROUGH=True`: mounts `/openai/v1/*` endpoints (no conversion, raw forward)

## Testing Strategy

### Unit Tests (`tests/unit/test_openai_passthrough/`)

- `test_usage_extractor.py` — normalize chat_completions and responses usage shapes (incl. missing/zero fields, cached tokens, reasoning tokens)
- `test_model_mapping.py` — passthrough when no mapping exists, substitution when it does
- `test_auth.py` — `Authorization: Bearer` resolves to API key correctly; both-headers precedence

### Integration Tests (`tests/integration/test_openai_passthrough/`)

`respx` mocks bedrock-mantle. Tests cover:

- POST chat/completions non-streaming → usage logged correctly
- POST chat/completions streaming with `include_usage=true` → usage logged from second-to-last chunk
- POST chat/completions streaming **without** `include_usage` → request succeeds, usage logged as zero
- POST responses streaming → usage logged from `response.completed` event
- POST responses non-streaming → usage logged from response body
- GET /responses/{id} forwards correctly
- DELETE /responses/{id} forwards correctly
- POST /responses/{id}/cancel forwards correctly
- GET /responses/{id}/input_items forwards correctly
- 4xx from Mantle returned verbatim (status + body)
- Rate limit on shared bucket triggers across both surfaces (mix `/v1/messages` and `/openai/v1/chat/completions` traffic)
- Budget exhaustion blocks POST endpoints but not CRUD endpoints

## Documentation Updates

- `CLAUDE.md` — new "Features" entry: "OpenAI Passthrough"
- `docs/architecture/features.md` — detailed feature doc with examples
- `env.example` — new flag with comment
- `README.md` / `README_ZH.md` — usage example with OpenAI SDK pointing at `/openai/v1`

## Open Items (Deferred)

1. **OTEL tracing** — additive, deferred to v2.
2. **Admin portal `api_surface` filter** — existing dashboards aggregate fine; add filter when needed.
3. **Guardrails passthrough** — Mantle Chat Completions supports guardrails via `X-Amzn-Bedrock-GuardrailIdentifier` headers. Recommend whitelisting `X-Amzn-Bedrock-*` headers in the passthrough on initial implementation. Trivial addition (~5 LOC), high value for guardrail-using customers. **Confirm before implementation.**

## Implementation Sequence

Once approved:

1. Schema/config skeleton: feature flag, DDB column additions to usage manager, normalization function with unit tests
2. Auth middleware extension (`Authorization: Bearer` support) with unit tests
3. httpx client singleton + non-streaming chat/completions endpoint + integration test
4. Streaming chat/completions + usage tee + integration test
5. Responses API POST (streaming + non-streaming) + integration tests
6. Responses CRUD endpoints (GET, DELETE, cancel, list_input_items) + integration tests
7. `/models` passthrough endpoint
8. Documentation updates (CLAUDE.md, features.md, env.example, READMEs)
