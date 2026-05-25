# OpenAI Passthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/openai/v1/*` endpoints that accept OpenAI-native Chat Completions and Responses API calls and forward them to AWS bedrock-mantle, while reusing the proxy's API key auth, rate limits, budgets, and usage tracking.

**Architecture:** Raw httpx passthrough (no Pydantic schemas for OpenAI types) on a new APIRouter mounted at `/openai/v1`. Existing auth middleware extended to accept `Authorization: Bearer` in addition to `x-api-key`. Usage extracted from response bodies (non-streaming) or final SSE event (streaming) and normalized into the existing Anthropic-shaped DDB schema with new `api_surface` and `reasoning_tokens` columns. Independent of existing `ENABLE_OPENAI_COMPAT`.

**Tech Stack:** Python 3.12, FastAPI, httpx (async), pytest + respx for HTTP mocking, AWS DynamoDB (boto3), uv for package management.

**Reference design:** `docs/plans/2026-05-25-openai-passthrough-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `app/core/config.py` | Modify | Add `enable_openai_passthrough` flag |
| `app/middleware/auth.py` | Modify | Accept `Authorization: Bearer` header |
| `app/api/openai_passthrough/__init__.py` | Create | Export `router` |
| `app/api/openai_passthrough/client.py` | Create | httpx singleton client |
| `app/api/openai_passthrough/usage_extractor.py` | Create | Normalize OpenAI usage → Anthropic-shaped dict |
| `app/api/openai_passthrough/streaming.py` | Create | SSE passthrough + usage tee |
| `app/api/openai_passthrough/router.py` | Create | FastAPI routes |
| `app/db/dynamodb.py` | Modify | Extend `UsageTracker.record_usage` with `api_surface` and `reasoning_tokens` |
| `app/main.py` | Modify | Conditionally mount the new router |
| `env.example` | Modify | Document the new flag |
| `CLAUDE.md` | Modify | Add feature description |
| `docs/architecture/features.md` | Modify | Detailed feature doc |
| `tests/unit/test_openai_passthrough/test_usage_extractor.py` | Create | Unit tests for normalization |
| `tests/unit/test_openai_passthrough/test_auth.py` | Create | Unit tests for header acceptance |
| `tests/unit/test_openai_passthrough/test_model_mapping.py` | Create | Unit tests for mapping resolution |
| `tests/unit/test_openai_passthrough/__init__.py` | Create | Test package marker |
| `tests/integration/test_openai_passthrough/test_chat_completions.py` | Create | Chat completions integration |
| `tests/integration/test_openai_passthrough/test_responses.py` | Create | Responses API integration |
| `tests/integration/test_openai_passthrough/test_responses_crud.py` | Create | Responses CRUD passthrough |
| `tests/integration/test_openai_passthrough/test_models.py` | Create | /models endpoint |
| `tests/integration/test_openai_passthrough/conftest.py` | Create | Shared fixtures (FastAPI client, respx) |
| `tests/integration/test_openai_passthrough/__init__.py` | Create | Test package marker |

**Tooling:** add `respx>=0.21.0` to `[project.optional-dependencies].dev` in `pyproject.toml`.

---

## Task 1: Add feature flag and respx dependency

**Files:**
- Modify: `app/core/config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the feature flag to settings**

In `app/core/config.py`, find the existing OpenAI-Compat block (around line 379–406) and add a new field immediately after `openai_compat_thinking_medium_threshold`:

```python
    enable_openai_passthrough: bool = Field(
        default=False,
        alias="ENABLE_OPENAI_PASSTHROUGH",
        description="Mount /openai/v1/* endpoints (Chat Completions + Responses passthrough to bedrock-mantle)"
    )
```

- [ ] **Step 2: Add respx to dev dependencies**

In `pyproject.toml`, locate the `dev = [...]` list under `[project.optional-dependencies]` (around line 80–95) and add `"respx>=0.21.0",` after `"pytest-mock>=3.12.0",`.

- [ ] **Step 3: Sync dependencies**

Run: `unset VIRTUAL_ENV && uv sync --active --extra dev`
Expected: `respx` and its deps resolved and installed.

- [ ] **Step 4: Verify the setting loads**

Run:
```bash
unset VIRTUAL_ENV && uv run --active python -c "from app.core.config import settings; print(settings.enable_openai_passthrough)"
```
Expected output: `False`

- [ ] **Step 5: Commit**

```bash
git add app/core/config.py pyproject.toml uv.lock
git commit -m "feat(openai-passthrough): add ENABLE_OPENAI_PASSTHROUGH flag and respx dev dep"
```

---

## Task 2: Extend auth middleware to accept Authorization: Bearer

**Files:**
- Modify: `app/middleware/auth.py:62-77`
- Test: `tests/unit/test_openai_passthrough/test_auth.py`

- [ ] **Step 1: Create the test package structure**

Run:
```bash
mkdir -p tests/unit/test_openai_passthrough
touch tests/unit/test_openai_passthrough/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_openai_passthrough/test_auth.py`:

```python
"""Tests for the auth middleware's Authorization: Bearer support."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.auth import AuthMiddleware


@pytest.fixture
def make_app():
    """Build a minimal FastAPI app wired to AuthMiddleware with a mocked validator."""
    def _factory(api_key_info):
        app = FastAPI()

        ddb_client = MagicMock()
        manager = MagicMock()
        manager.validate_api_key.return_value = api_key_info

        with patch("app.middleware.auth.APIKeyManager", return_value=manager):
            app.add_middleware(AuthMiddleware, dynamodb_client=ddb_client)

        @app.get("/test")
        async def test_endpoint(request):
            from fastapi import Request
            r: Request = request  # type: ignore[assignment]
            info = r.state.api_key_info
            return {"user_id": info["user_id"]}

        return app, manager
    return _factory


def test_authorization_bearer_resolves_when_xapikey_missing(make_app, monkeypatch):
    """Authorization: Bearer <key> should authenticate when x-api-key is absent."""
    monkeypatch.setattr("app.core.config.settings.require_api_key", True)
    monkeypatch.setattr("app.core.config.settings.master_api_key", "")

    app, manager = make_app({"user_id": "u1", "api_key": "sk-abc"})
    client = TestClient(app)

    r = client.get("/test", headers={"Authorization": "Bearer sk-abc"})

    assert r.status_code == 200
    assert r.json() == {"user_id": "u1"}
    manager.validate_api_key.assert_called_once_with("sk-abc")


def test_xapikey_takes_precedence_when_both_present(make_app, monkeypatch):
    """If both headers are present, x-api-key wins."""
    monkeypatch.setattr("app.core.config.settings.require_api_key", True)
    monkeypatch.setattr("app.core.config.settings.master_api_key", "")

    app, manager = make_app({"user_id": "u1", "api_key": "sk-from-xapikey"})
    client = TestClient(app)

    client.get(
        "/test",
        headers={"x-api-key": "sk-from-xapikey", "Authorization": "Bearer sk-from-bearer"},
    )

    manager.validate_api_key.assert_called_once_with("sk-from-xapikey")


def test_missing_both_headers_returns_401(make_app, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.require_api_key", True)
    monkeypatch.setattr("app.core.config.settings.master_api_key", "")

    app, _ = make_app(None)
    client = TestClient(app)

    r = client.get("/test")
    assert r.status_code == 401


def test_authorization_non_bearer_is_ignored(make_app, monkeypatch):
    """Authorization: Basic ... (or anything not 'Bearer ') should not be treated as an API key."""
    monkeypatch.setattr("app.core.config.settings.require_api_key", True)
    monkeypatch.setattr("app.core.config.settings.master_api_key", "")

    app, _ = make_app(None)
    client = TestClient(app)

    r = client.get("/test", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert r.status_code == 401
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_auth.py -v --no-cov`
Expected: All four tests FAIL — `test_authorization_bearer_resolves_when_xapikey_missing` returns 401 because the middleware doesn't yet read Authorization.

- [ ] **Step 4: Modify the middleware to read Authorization: Bearer**

In `app/middleware/auth.py`, replace lines 62–77 (the API key extraction + missing-key 401 block) with:

```python
        # Extract API key from header (x-api-key first, fall back to Authorization: Bearer)
        api_key = request.headers.get(settings.api_key_header)
        if not api_key:
            authz = request.headers.get("Authorization") or request.headers.get("authorization")
            if authz and authz.startswith("Bearer "):
                api_key = authz[len("Bearer "):].strip()

        if not api_key:
            print(f"[AUTH] Missing API key for {request.url.path}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": f"Missing API key in {settings.api_key_header} or Authorization: Bearer header",
                    },
                },
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_auth.py -v --no-cov`
Expected: All four tests PASS.

- [ ] **Step 6: Run the full unit test suite to ensure no regression**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit -q --no-cov`
Expected: All previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add app/middleware/auth.py tests/unit/test_openai_passthrough/test_auth.py tests/unit/test_openai_passthrough/__init__.py
git commit -m "feat(auth): accept Authorization: Bearer alongside x-api-key"
```

---

## Task 3: Add usage normalization function

**Files:**
- Create: `app/api/openai_passthrough/__init__.py`
- Create: `app/api/openai_passthrough/usage_extractor.py`
- Test: `tests/unit/test_openai_passthrough/test_usage_extractor.py`

- [ ] **Step 1: Create the package directories**

Run:
```bash
mkdir -p app/api/openai_passthrough
touch app/api/openai_passthrough/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_openai_passthrough/test_usage_extractor.py`:

```python
"""Tests for normalize_usage and try_extract_usage_from_sse."""
import json

from app.api.openai_passthrough.usage_extractor import (
    normalize_usage,
    try_extract_usage_from_sse,
)


def test_normalize_chat_completions_basic():
    raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    result = normalize_usage(raw, "chat_completions")
    assert result == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_normalize_chat_completions_with_cache_and_reasoning():
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 30},
        "completion_tokens_details": {"reasoning_tokens": 20},
    }
    result = normalize_usage(raw, "chat_completions")
    # cache hits subtracted from input
    assert result["input_tokens"] == 70
    assert result["output_tokens"] == 50
    assert result["cache_read_input_tokens"] == 30
    assert result["reasoning_tokens"] == 20


def test_normalize_responses_basic():
    raw = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    result = normalize_usage(raw, "responses")
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["cache_read_input_tokens"] == 0
    assert result["reasoning_tokens"] == 0


def test_normalize_responses_with_cache_and_reasoning():
    raw = {
        "input_tokens": 100,
        "output_tokens": 50,
        "input_tokens_details": {"cached_tokens": 25},
        "output_tokens_details": {"reasoning_tokens": 15},
    }
    result = normalize_usage(raw, "responses")
    assert result["input_tokens"] == 75
    assert result["output_tokens"] == 50
    assert result["cache_read_input_tokens"] == 25
    assert result["reasoning_tokens"] == 15


def test_normalize_handles_missing_fields():
    """Empty/None usage should normalize to all-zeros, not crash."""
    result = normalize_usage({}, "chat_completions")
    assert result == {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_extract_chat_completions_usage_from_sse_chunk():
    """Final chat-completions chunk with usage should be picked up."""
    line = "data: " + json.dumps({
        "id": "chatcmpl-1", "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    holder: dict = {}
    try_extract_usage_from_sse(line, holder, "chat_completions")
    assert holder == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_extract_responses_usage_from_response_completed_event():
    line = "data: " + json.dumps({
        "type": "response.completed",
        "response": {
            "id": "resp-1",
            "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
        },
    })
    holder: dict = {}
    try_extract_usage_from_sse(line, holder, "responses")
    assert holder == {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}


def test_extract_ignores_non_data_lines():
    holder: dict = {}
    try_extract_usage_from_sse("event: response.completed", holder, "responses")
    try_extract_usage_from_sse("", holder, "responses")
    try_extract_usage_from_sse(": keepalive", holder, "responses")
    assert holder == {}


def test_extract_ignores_data_done():
    holder: dict = {}
    try_extract_usage_from_sse("data: [DONE]", holder, "chat_completions")
    assert holder == {}


def test_extract_ignores_chunks_without_usage():
    line = "data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]})
    holder: dict = {}
    try_extract_usage_from_sse(line, holder, "chat_completions")
    assert holder == {}


def test_extract_ignores_malformed_json():
    holder: dict = {}
    try_extract_usage_from_sse("data: not-json", holder, "chat_completions")
    assert holder == {}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_usage_extractor.py -v --no-cov`
Expected: ImportError — module doesn't exist yet.

- [ ] **Step 4: Implement usage_extractor**

Create `app/api/openai_passthrough/usage_extractor.py`:

```python
"""Usage extraction and normalization for OpenAI-format responses.

normalize_usage() converts an OpenAI Chat Completions or Responses API usage
dict into the Anthropic-shaped dict that UsageTracker.record_usage expects,
plus a separate reasoning_tokens field.

try_extract_usage_from_sse() peeks at SSE lines during streaming and stashes
the usage dict (raw OpenAI shape) the first time it encounters one. The caller
later passes that dict through normalize_usage().
"""
from __future__ import annotations

import json
from typing import Any, Dict


def normalize_usage(raw: Dict[str, Any], api_surface: str) -> Dict[str, int]:
    """Normalize OpenAI-shaped usage into Anthropic-shaped fields.

    api_surface: "chat_completions" or "responses"
    """
    if api_surface == "chat_completions":
        in_tok = int(raw.get("prompt_tokens", 0) or 0)
        out_tok = int(raw.get("completion_tokens", 0) or 0)
        cached = int((raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        reasoning = int(
            (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
    else:  # responses
        in_tok = int(raw.get("input_tokens", 0) or 0)
        out_tok = int(raw.get("output_tokens", 0) or 0)
        cached = int((raw.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
        reasoning = int(
            (raw.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )

    # Cache-read tokens are billed separately, so subtract them from input_tokens
    # to mirror how the Anthropic flow accounts for cache hits.
    return {
        "input_tokens": max(in_tok - cached, 0),
        "output_tokens": out_tok,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,  # Not exposed by OpenAI-format APIs
        "reasoning_tokens": reasoning,
    }


def try_extract_usage_from_sse(
    raw_line: str, holder: Dict[str, Any], api_surface: str
) -> None:
    """Inspect an SSE line and, if it carries usage info, store it in holder.

    Mutates `holder` in place. Idempotent: subsequent calls overwrite, so the
    last-seen usage event wins (which is what we want — both APIs put usage
    on the terminal event).
    """
    line = raw_line.strip()
    if not line.startswith("data:"):
        return

    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return

    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return

    if api_surface == "chat_completions":
        usage = obj.get("usage")
        if isinstance(usage, dict):
            holder.clear()
            holder.update(usage)
    else:  # responses
        # Usage lives on the `response.completed` event under
        # event.response.usage. Other events occasionally carry partial usage
        # too — accept any usage dict we see.
        if obj.get("type") == "response.completed":
            response_obj = obj.get("response") or {}
            usage = response_obj.get("usage")
            if isinstance(usage, dict):
                holder.clear()
                holder.update(usage)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_usage_extractor.py -v --no-cov`
Expected: All 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api/openai_passthrough/__init__.py app/api/openai_passthrough/usage_extractor.py tests/unit/test_openai_passthrough/test_usage_extractor.py
git commit -m "feat(openai-passthrough): add usage normalization and SSE extraction helpers"
```

---

## Task 4: Add model mapping resolver helper

**Files:**
- Create: `app/api/openai_passthrough/model_mapping.py`
- Test: `tests/unit/test_openai_passthrough/test_model_mapping.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_openai_passthrough/test_model_mapping.py`:

```python
"""Tests for resolve_model_id."""
from unittest.mock import MagicMock

from app.api.openai_passthrough.model_mapping import resolve_model_id


def test_returns_mapped_id_when_mapping_exists():
    manager = MagicMock()
    manager.get_mapping.return_value = "openai.gpt-oss-120b"

    out = resolve_model_id("gpt-4", manager)
    assert out == "openai.gpt-oss-120b"
    manager.get_mapping.assert_called_once_with("gpt-4")


def test_passes_through_when_no_mapping_exists():
    manager = MagicMock()
    manager.get_mapping.return_value = None

    out = resolve_model_id("openai.gpt-oss-120b", manager)
    assert out == "openai.gpt-oss-120b"


def test_passes_through_empty_string():
    manager = MagicMock()
    manager.get_mapping.return_value = None

    assert resolve_model_id("", manager) == ""


def test_handles_lookup_exception_by_passing_through():
    """If DDB lookup raises, fall back to the original ID rather than crashing the request."""
    manager = MagicMock()
    manager.get_mapping.side_effect = RuntimeError("ddb down")

    out = resolve_model_id("gpt-4", manager)
    assert out == "gpt-4"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_model_mapping.py -v --no-cov`
Expected: ImportError.

- [ ] **Step 3: Implement resolve_model_id**

Create `app/api/openai_passthrough/model_mapping.py`:

```python
"""Model ID resolution for the OpenAI passthrough endpoints.

Looks up the client-supplied model in the existing model_mapping table; if a
mapping exists, substitute it. Otherwise, pass through unchanged so callers
can use Bedrock-native IDs (e.g. ``openai.gpt-oss-120b``) directly without
needing to register them.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_model_id(model: str, model_mapping_manager) -> str:
    """Resolve a client-supplied model ID via the mapping table, with fallback.

    Args:
        model: The ``model`` field from the client request.
        model_mapping_manager: An app.db.dynamodb.ModelMappingManager instance.

    Returns:
        The resolved Bedrock model ID, or the original string if no mapping
        exists or the lookup fails.
    """
    if not model:
        return model
    try:
        mapped = model_mapping_manager.get_mapping(model)
    except Exception as exc:
        logger.warning("[OPENAI-PASSTHROUGH] model mapping lookup failed for %r: %s", model, exc)
        return model
    return mapped or model
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_model_mapping.py -v --no-cov`
Expected: All four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/openai_passthrough/model_mapping.py tests/unit/test_openai_passthrough/test_model_mapping.py
git commit -m "feat(openai-passthrough): add model mapping resolver with passthrough fallback"
```

---

## Task 5: Extend UsageTracker.record_usage with api_surface and reasoning_tokens

**Files:**
- Modify: `app/db/dynamodb.py:908-970`
- Test: `tests/unit/test_openai_passthrough/test_usage_tracker_extended.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_openai_passthrough/test_usage_tracker_extended.py`:

```python
"""Tests for the api_surface and reasoning_tokens additions to UsageTracker."""
from unittest.mock import MagicMock

from app.db.dynamodb import UsageTracker


def _make_tracker():
    ddb_client = MagicMock()
    ddb_client.usage_table_name = "anthropic-proxy-usage"
    tracker = UsageTracker(ddb_client)
    tracker.table = MagicMock()
    return tracker


def test_record_usage_writes_api_surface_when_provided():
    tracker = _make_tracker()
    tracker.record_usage(
        api_key="sk-x",
        request_id="req-1",
        model="openai.gpt-oss-120b",
        input_tokens=100,
        output_tokens=50,
        api_surface="chat_completions",
    )
    item = tracker.table.put_item.call_args.kwargs["Item"]
    assert item["api_surface"] == "chat_completions"


def test_record_usage_writes_reasoning_tokens_when_provided():
    tracker = _make_tracker()
    tracker.record_usage(
        api_key="sk-x", request_id="req-1", model="m",
        input_tokens=10, output_tokens=5, reasoning_tokens=3,
    )
    item = tracker.table.put_item.call_args.kwargs["Item"]
    assert item["reasoning_tokens"] == 3


def test_record_usage_omits_new_fields_when_default():
    tracker = _make_tracker()
    tracker.record_usage(
        api_key="sk-x", request_id="req-1", model="m",
        input_tokens=10, output_tokens=5,
    )
    item = tracker.table.put_item.call_args.kwargs["Item"]
    # Sparse: not written when caller didn't specify them
    assert "api_surface" not in item
    assert "reasoning_tokens" not in item
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_usage_tracker_extended.py -v --no-cov`
Expected: TypeError — record_usage doesn't accept the new kwargs.

- [ ] **Step 3: Modify record_usage**

In `app/db/dynamodb.py`, find `UsageTracker.record_usage` (line 908). Add two new optional parameters to its signature, after `cache_ttl`:

```python
        cache_ttl: Optional[str] = None,
        api_surface: Optional[str] = None,
        reasoning_tokens: int = 0,
    ):
```

Update the docstring's Args block to document them:

```
            api_surface: Source endpoint family ("messages", "chat_completions", or "responses")
            reasoning_tokens: Reasoning tokens (already counted in output_tokens; stored separately for visibility)
```

Then, after the existing `if cache_ttl:` block (around line 962–963), add:

```python
        if api_surface:
            item["api_surface"] = api_surface
        if reasoning_tokens:
            item["reasoning_tokens"] = reasoning_tokens
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough/test_usage_tracker_extended.py -v --no-cov`
Expected: All three tests PASS.

- [ ] **Step 5: Run the full unit suite to check nothing regressed**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit -q --no-cov`
Expected: All previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add app/db/dynamodb.py tests/unit/test_openai_passthrough/test_usage_tracker_extended.py
git commit -m "feat(usage): record api_surface and reasoning_tokens on usage rows"
```

---

## Task 6: Implement httpx client singleton

**Files:**
- Create: `app/api/openai_passthrough/client.py`

This is a small helper without business logic, so we test it indirectly through the integration tests (Tasks 8–11). No standalone unit tests.

- [ ] **Step 1: Write the client module**

Create `app/api/openai_passthrough/client.py`:

```python
"""Async httpx client to bedrock-mantle, lazily constructed and reused.

Headers are NOT set on the client itself; they're added per-request in the
router so we can include the proxy's Bedrock API key in Authorization.
"""
from __future__ import annotations

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


def reset_client_for_testing() -> None:
    """Reset the singleton — only call this from test fixtures."""
    global _client
    if _client is not None:
        # AsyncClient.aclose() is async; tests will close the loop after, so we
        # null it here and let the GC clean up the underlying transport.
        _client = None


def upstream_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build the Authorization + standard headers for an upstream call."""
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
        "User-Agent": "bedrock-api-proxy/openai-passthrough",
    }
    if extra:
        headers.update(extra)
    return headers
```

- [ ] **Step 2: Smoke-test the import**

Run:
```bash
unset VIRTUAL_ENV && uv run --active python -c "from app.api.openai_passthrough.client import get_client, upstream_headers; print(upstream_headers())"
```
Expected: prints a dict with `Authorization: Bearer ` (the configured key, possibly empty), `Content-Type`, and `User-Agent`.

- [ ] **Step 3: Commit**

```bash
git add app/api/openai_passthrough/client.py
git commit -m "feat(openai-passthrough): add httpx singleton client and header helper"
```

---

## Task 7: Implement streaming passthrough helper

**Files:**
- Create: `app/api/openai_passthrough/streaming.py`

Tested indirectly through integration tests in Task 9.

- [ ] **Step 1: Write the streaming module**

Create `app/api/openai_passthrough/streaming.py`:

```python
"""SSE passthrough with usage tee.

The async generator yields raw response bytes line-by-line so the FastAPI
StreamingResponse forwards them unchanged. After upstream stream ends, it
calls the supplied on_complete callback with the captured usage dict so the
caller can record usage to DynamoDB.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Dict

from app.api.openai_passthrough.client import get_client, upstream_headers
from app.api.openai_passthrough.usage_extractor import try_extract_usage_from_sse

logger = logging.getLogger(__name__)


async def stream_passthrough(
    method: str,
    path: str,
    body: Dict[str, Any] | None,
    api_surface: str,
    on_complete: Callable[[Dict[str, Any]], Awaitable[None] | None],
    extra_headers: Dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """Stream upstream response bytes line-by-line; capture usage; trigger callback."""
    usage: Dict[str, Any] = {}

    client = get_client()
    headers = upstream_headers(extra_headers)

    try:
        async with client.stream(method, path, json=body, headers=headers) as resp:
            async for raw_line in resp.aiter_lines():
                # Upstream gives us SSE lines without trailing newlines; restore the
                # framing byte so the SSE body is well-formed for the downstream client.
                yield (raw_line + "\n").encode("utf-8")
                try_extract_usage_from_sse(raw_line, usage, api_surface)
    except Exception as exc:
        logger.error("[OPENAI-PASSTHROUGH] upstream stream error: %s", exc)
        # Re-raise so FastAPI can return a 500; downstream client sees the stream end.
        raise

    if usage:
        result = on_complete(usage)
        # Support both sync and async callbacks
        if hasattr(result, "__await__"):
            await result  # type: ignore[func-returns-value]
```

- [ ] **Step 2: Smoke-test the import**

Run:
```bash
unset VIRTUAL_ENV && uv run --active python -c "from app.api.openai_passthrough.streaming import stream_passthrough; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add app/api/openai_passthrough/streaming.py
git commit -m "feat(openai-passthrough): add SSE passthrough generator with usage tee"
```

---

## Task 8: Implement router skeleton + chat/completions (non-streaming)

**Files:**
- Create: `app/api/openai_passthrough/router.py`
- Modify: `app/main.py:298-314`
- Create: `tests/integration/test_openai_passthrough/__init__.py`
- Create: `tests/integration/test_openai_passthrough/conftest.py`
- Test: `tests/integration/test_openai_passthrough/test_chat_completions.py`

- [ ] **Step 1: Create the integration test scaffolding**

Run:
```bash
mkdir -p tests/integration/test_openai_passthrough
touch tests/integration/test_openai_passthrough/__init__.py
```

Create `tests/integration/test_openai_passthrough/conftest.py`:

```python
"""Shared fixtures for openai-passthrough integration tests."""
from unittest.mock import MagicMock, patch

import pytest
import respx
from fastapi.testclient import TestClient


@pytest.fixture
def mock_settings(monkeypatch):
    """Set the env so the passthrough router mounts and points at a fake mantle."""
    monkeypatch.setattr("app.core.config.settings.enable_openai_passthrough", True)
    monkeypatch.setattr("app.core.config.settings.openai_api_key", "bedrock-key-test")
    monkeypatch.setattr("app.core.config.settings.openai_base_url", "https://mantle.test/v1")
    monkeypatch.setattr("app.core.config.settings.require_api_key", True)
    monkeypatch.setattr("app.core.config.settings.master_api_key", "")


@pytest.fixture
def mock_api_key_manager():
    """Patch APIKeyManager so any non-empty key validates as user 'u1'."""
    manager = MagicMock()
    manager.validate_api_key.return_value = {
        "api_key": "sk-test", "user_id": "u1", "is_master": False,
        "rate_limit": None, "cache_ttl": None,
    }
    with patch("app.middleware.auth.APIKeyManager", return_value=manager):
        yield manager


@pytest.fixture
def mock_model_mapping_manager():
    """Patch ModelMappingManager to return None (no mapping) by default."""
    manager = MagicMock()
    manager.get_mapping.return_value = None
    with patch("app.db.dynamodb.ModelMappingManager", return_value=manager):
        yield manager


@pytest.fixture
def mock_usage_tracker():
    tracker = MagicMock()
    with patch("app.db.dynamodb.UsageTracker", return_value=tracker):
        yield tracker


@pytest.fixture
def respx_mock():
    """respx mock router for httpx calls."""
    with respx.mock(base_url="https://mantle.test/v1", assert_all_called=False) as router:
        yield router


@pytest.fixture
def client(mock_settings, mock_api_key_manager, mock_model_mapping_manager, mock_usage_tracker):
    """FastAPI TestClient with all mocks wired in.

    Imports inside the fixture so module-level settings reads happen after
    monkeypatching.
    """
    # Reset httpx singleton so it picks up the patched base URL
    from app.api.openai_passthrough.client import reset_client_for_testing
    reset_client_for_testing()

    from app.main import app
    return TestClient(app)
```

- [ ] **Step 2: Write the failing test**

Create `tests/integration/test_openai_passthrough/test_chat_completions.py`:

```python
"""Integration tests for POST /openai/v1/chat/completions."""
import json

import respx
import httpx


def test_non_streaming_chat_completions_forwards_and_logs_usage(
    client, respx_mock, mock_usage_tracker, mock_model_mapping_manager
):
    upstream_resp = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "openai.gpt-oss-120b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    route = respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=upstream_resp)
    )

    r = client.post(
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "openai.gpt-oss-120b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert r.status_code == 200
    assert r.json() == upstream_resp
    assert route.called
    # Upstream got proxy's Bedrock API key, not the client's proxy key
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer bedrock-key-test"
    sent_body = json.loads(sent.content)
    assert sent_body["model"] == "openai.gpt-oss-120b"
    # Usage was recorded
    assert mock_usage_tracker.record_usage.called
    kwargs = mock_usage_tracker.record_usage.call_args.kwargs
    assert kwargs["input_tokens"] == 10
    assert kwargs["output_tokens"] == 5
    assert kwargs["api_surface"] == "chat_completions"
    assert kwargs["model"] == "openai.gpt-oss-120b"


def test_model_mapping_is_applied(
    client, respx_mock, mock_model_mapping_manager
):
    mock_model_mapping_manager.get_mapping.return_value = "openai.gpt-oss-120b"
    route = respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "x", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    )

    client.post(
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "openai.gpt-oss-120b"


def test_upstream_4xx_returned_verbatim(client, respx_mock, mock_usage_tracker):
    err_body = {"error": {"message": "model not found", "type": "invalid_request_error"}}
    respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(404, json=err_body)
    )

    r = client.post(
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "no-such-model", "messages": []},
    )
    assert r.status_code == 404
    assert r.json() == err_body
    assert not mock_usage_tracker.record_usage.called  # Don't log usage on errors


def test_missing_auth_returns_401(client):
    r = client.post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": []},
    )
    assert r.status_code == 401
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_chat_completions.py -v --no-cov`
Expected: tests fail — endpoint doesn't exist yet (404 from FastAPI).

- [ ] **Step 4: Implement the router**

Create `app/api/openai_passthrough/router.py`:

```python
"""FastAPI routes for the OpenAI passthrough endpoints.

Mounted at /openai/v1 only when settings.enable_openai_passthrough is True.
"""
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.openai_passthrough.client import get_client, upstream_headers
from app.api.openai_passthrough.model_mapping import resolve_model_id
from app.api.openai_passthrough.streaming import stream_passthrough
from app.api.openai_passthrough.usage_extractor import normalize_usage
from app.db.dynamodb import DynamoDBClient, ModelMappingManager, UsageTracker
from app.middleware.auth import get_api_key_info

logger = logging.getLogger(__name__)
router = APIRouter()

_ddb: DynamoDBClient | None = None
_mapping: ModelMappingManager | None = None
_usage: UsageTracker | None = None


def _managers():
    """Lazily build DDB managers — keeps import-time side effects out of tests."""
    global _ddb, _mapping, _usage
    if _ddb is None:
        _ddb = DynamoDBClient()
        _mapping = ModelMappingManager(_ddb)
        _usage = UsageTracker(_ddb)
    return _mapping, _usage


def _record_usage(api_key_info: Dict[str, Any], raw_usage: Dict[str, Any], model: str, api_surface: str) -> None:
    _, usage = _managers()
    norm = normalize_usage(raw_usage, api_surface)
    try:
        usage.record_usage(
            api_key=api_key_info.get("api_key", ""),
            request_id=str(uuid4()),
            model=model,
            input_tokens=norm["input_tokens"],
            output_tokens=norm["output_tokens"],
            cached_tokens=norm["cache_read_input_tokens"],
            cache_write_input_tokens=norm["cache_creation_input_tokens"],
            api_surface=api_surface,
            reasoning_tokens=norm["reasoning_tokens"],
        )
    except Exception as exc:
        logger.warning("[OPENAI-PASSTHROUGH] usage recording failed: %s", exc)


def _passthrough_extra_headers(request: Request) -> Dict[str, str]:
    """Forward Bedrock-specific headers from the client to upstream (e.g. guardrails)."""
    extra: Dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower().startswith("x-amzn-bedrock-"):
            extra[name] = value
    return extra


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    api_key_info: Dict[str, Any] = Depends(get_api_key_info),
):
    body = await request.json()
    mapping, _ = _managers()
    body["model"] = resolve_model_id(body.get("model", ""), mapping)
    extra = _passthrough_extra_headers(request)

    if body.get("stream"):
        async def on_complete(usage: Dict[str, Any]) -> None:
            _record_usage(api_key_info, usage, body["model"], "chat_completions")
        return StreamingResponse(
            stream_passthrough(
                "POST", "/chat/completions", body, "chat_completions", on_complete, extra
            ),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        "/chat/completions", json=body, headers=upstream_headers(extra)
    )
    if resp.status_code >= 400:
        return JSONResponse(_safe_json(resp), status_code=resp.status_code)

    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
        _record_usage(api_key_info, data["usage"], body["model"], "chat_completions")
    return JSONResponse(data, status_code=resp.status_code)


def _safe_json(resp) -> Dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": resp.text, "type": "upstream_error"}}
```

- [ ] **Step 5: Wire up `__init__.py`**

In `app/api/openai_passthrough/__init__.py`, replace the empty file with:

```python
"""OpenAI Passthrough — accepts OpenAI Chat Completions and Responses API
calls from clients and forwards them to AWS bedrock-mantle.
"""
from app.api.openai_passthrough.router import router

__all__ = ["router"]
```

- [ ] **Step 6: Mount the router in main.py**

In `app/main.py`, after the existing `app.include_router(models.router, ...)` block (around line 314), add:

```python
if settings.enable_openai_passthrough:
    from app.api.openai_passthrough import router as openai_passthrough_router
    app.include_router(
        openai_passthrough_router,
        prefix="/openai/v1",
        tags=["OpenAI Passthrough"],
    )
```

- [ ] **Step 7: Run the integration tests**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_chat_completions.py -v --no-cov`
Expected: All four tests PASS.

- [ ] **Step 8: Run the full unit suite to check no regression**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit -q --no-cov`
Expected: All previously-passing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add app/api/openai_passthrough/router.py app/api/openai_passthrough/__init__.py app/main.py tests/integration/test_openai_passthrough/__init__.py tests/integration/test_openai_passthrough/conftest.py tests/integration/test_openai_passthrough/test_chat_completions.py
git commit -m "feat(openai-passthrough): non-streaming /chat/completions endpoint"
```

---

## Task 9: Add streaming support to chat/completions

**Files:**
- Test: `tests/integration/test_openai_passthrough/test_chat_completions.py` (extend)

The router already routes `body["stream"] = True` requests through `stream_passthrough`; this task validates the path end-to-end and adds the missing-`include_usage` case.

- [ ] **Step 1: Append failing tests**

Append the following to `tests/integration/test_openai_passthrough/test_chat_completions.py`:

```python
def test_streaming_chat_completions_forwards_sse_and_records_usage(
    client, respx_mock, mock_usage_tracker
):
    """Stream three SSE chunks; the second-to-last carries usage."""
    sse_lines = [
        'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}',
        'data: {"id":"x","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}',
        'data: [DONE]',
    ]
    body = "\n".join(sse_lines).encode()
    respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )

    with client.stream(
        "POST",
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "openai.gpt-oss-120b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        assert r.status_code == 200
        out = b"".join(r.iter_bytes())

    # All four lines forwarded
    assert b'"delta":{"role":"assistant"}' in out
    assert b'[DONE]' in out
    # Usage recorded from the chunk that had it
    assert mock_usage_tracker.record_usage.called
    kw = mock_usage_tracker.record_usage.call_args.kwargs
    assert kw["input_tokens"] == 7
    assert kw["output_tokens"] == 2
    assert kw["api_surface"] == "chat_completions"


def test_streaming_chat_completions_without_include_usage_does_not_log(
    client, respx_mock, mock_usage_tracker
):
    """If client doesn't request usage, no usage chunk arrives → no usage logged."""
    sse_lines = [
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}',
        'data: [DONE]',
    ]
    body = "\n".join(sse_lines).encode()
    respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )

    with client.stream(
        "POST", "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "m", "messages": [], "stream": True},
    ) as r:
        list(r.iter_bytes())  # drain

    assert not mock_usage_tracker.record_usage.called
```

- [ ] **Step 2: Run the tests**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_chat_completions.py -v --no-cov`
Expected: All six tests PASS (including the two new streaming tests).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_openai_passthrough/test_chat_completions.py
git commit -m "test(openai-passthrough): streaming /chat/completions integration tests"
```

---

## Task 10: Add Responses API POST endpoint (streaming + non-streaming)

**Files:**
- Modify: `app/api/openai_passthrough/router.py`
- Test: `tests/integration/test_openai_passthrough/test_responses.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_openai_passthrough/test_responses.py`:

```python
"""Integration tests for POST /openai/v1/responses (streaming + non-streaming)."""
import json

import httpx


def test_non_streaming_responses_forwards_and_logs_usage(
    client, respx_mock, mock_usage_tracker
):
    upstream = {
        "id": "resp-1",
        "object": "response",
        "model": "openai.gpt-oss-120b",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }
    route = respx_mock.post("/responses").mock(return_value=httpx.Response(200, json=upstream))

    r = client.post(
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "openai.gpt-oss-120b", "input": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    assert r.json() == upstream
    assert route.called
    kw = mock_usage_tracker.record_usage.call_args.kwargs
    assert kw["input_tokens"] == 11
    assert kw["output_tokens"] == 4
    assert kw["api_surface"] == "responses"


def test_streaming_responses_records_usage_from_response_completed(
    client, respx_mock, mock_usage_tracker
):
    sse_lines = [
        'event: response.created',
        'data: {"type":"response.created","response":{"id":"r-1"}}',
        'event: response.output_text.delta',
        'data: {"type":"response.output_text.delta","delta":"hi"}',
        'event: response.completed',
        'data: ' + json.dumps({
            "type": "response.completed",
            "response": {"id": "r-1", "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}},
        }),
    ]
    body = "\n".join(sse_lines).encode()
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )

    with client.stream(
        "POST", "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "openai.gpt-oss-120b", "input": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        out = b"".join(r.iter_bytes())

    assert b"response.completed" in out
    assert b"hi" in out
    kw = mock_usage_tracker.record_usage.call_args.kwargs
    assert kw["input_tokens"] == 12
    assert kw["output_tokens"] == 3
    assert kw["api_surface"] == "responses"


def test_responses_upstream_error_returned_verbatim(client, respx_mock, mock_usage_tracker):
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad input", "type": "invalid_request_error"}})
    )
    r = client.post(
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "m", "input": []},
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "bad input"
    assert not mock_usage_tracker.record_usage.called
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_responses.py -v --no-cov`
Expected: 404s — endpoint doesn't exist yet.

- [ ] **Step 3: Add the responses endpoint to the router**

In `app/api/openai_passthrough/router.py`, immediately after the `chat_completions` function, add:

```python
@router.post("/responses")
async def responses_create(
    request: Request,
    api_key_info: Dict[str, Any] = Depends(get_api_key_info),
):
    body = await request.json()
    mapping, _ = _managers()
    body["model"] = resolve_model_id(body.get("model", ""), mapping)
    extra = _passthrough_extra_headers(request)

    if body.get("stream"):
        async def on_complete(usage: Dict[str, Any]) -> None:
            _record_usage(api_key_info, usage, body["model"], "responses")
        return StreamingResponse(
            stream_passthrough("POST", "/responses", body, "responses", on_complete, extra),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        "/responses", json=body, headers=upstream_headers(extra)
    )
    if resp.status_code >= 400:
        return JSONResponse(_safe_json(resp), status_code=resp.status_code)

    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
        _record_usage(api_key_info, data["usage"], body["model"], "responses")
    return JSONResponse(data, status_code=resp.status_code)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_responses.py -v --no-cov`
Expected: All three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/openai_passthrough/router.py tests/integration/test_openai_passthrough/test_responses.py
git commit -m "feat(openai-passthrough): /responses endpoint (POST, streaming + non-streaming)"
```

---

## Task 11: Add Responses CRUD passthrough (GET, DELETE, cancel, input_items)

**Files:**
- Modify: `app/api/openai_passthrough/router.py`
- Test: `tests/integration/test_openai_passthrough/test_responses_crud.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_openai_passthrough/test_responses_crud.py`:

```python
"""Integration tests for the Responses CRUD endpoints — pure passthrough."""
import httpx


def test_get_response_forwards_and_returns_body(client, respx_mock, mock_usage_tracker):
    body = {"id": "r-1", "model": "x", "status": "completed"}
    respx_mock.get("/responses/r-1").mock(return_value=httpx.Response(200, json=body))

    r = client.get("/openai/v1/responses/r-1", headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 200
    assert r.json() == body
    # No usage logged for retrieval
    assert not mock_usage_tracker.record_usage.called


def test_delete_response_forwards(client, respx_mock):
    respx_mock.delete("/responses/r-1").mock(
        return_value=httpx.Response(200, json={"id": "r-1", "deleted": True})
    )
    r = client.delete("/openai/v1/responses/r-1", headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 200
    assert r.json() == {"id": "r-1", "deleted": True}


def test_cancel_response_forwards(client, respx_mock):
    respx_mock.post("/responses/r-1/cancel").mock(
        return_value=httpx.Response(200, json={"id": "r-1", "status": "cancelled"})
    )
    r = client.post("/openai/v1/responses/r-1/cancel", headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


def test_list_input_items_forwards(client, respx_mock):
    body = {"data": [{"id": "msg-1", "role": "user"}], "object": "list"}
    respx_mock.get("/responses/r-1/input_items").mock(return_value=httpx.Response(200, json=body))
    r = client.get(
        "/openai/v1/responses/r-1/input_items",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    assert r.json() == body


def test_get_response_404_returned_verbatim(client, respx_mock):
    respx_mock.get("/responses/missing").mock(
        return_value=httpx.Response(404, json={"error": {"message": "not found"}})
    )
    r = client.get("/openai/v1/responses/missing", headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 404
    assert r.json()["error"]["message"] == "not found"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_responses_crud.py -v --no-cov`
Expected: 404s — endpoints don't exist yet.

- [ ] **Step 3: Add the CRUD endpoints**

In `app/api/openai_passthrough/router.py`, add immediately after `responses_create`:

```python
async def _passthrough_request(request: Request, path: str) -> Response:
    """Forward request to upstream and mirror the upstream response."""
    extra = _passthrough_extra_headers(request)
    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.json()
        except Exception:
            body = None
    resp = await get_client().request(
        request.method, path, json=body, headers=upstream_headers(extra)
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


@router.api_route("/responses/{response_id}", methods=["GET", "DELETE"])
async def responses_get_or_delete(
    response_id: str,
    request: Request,
    _: Dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, f"/responses/{response_id}")


@router.post("/responses/{response_id}/cancel")
async def responses_cancel(
    response_id: str,
    request: Request,
    _: Dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, f"/responses/{response_id}/cancel")


@router.get("/responses/{response_id}/input_items")
async def responses_input_items(
    response_id: str,
    request: Request,
    _: Dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, f"/responses/{response_id}/input_items")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_responses_crud.py -v --no-cov`
Expected: All five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/openai_passthrough/router.py tests/integration/test_openai_passthrough/test_responses_crud.py
git commit -m "feat(openai-passthrough): /responses CRUD passthrough (GET, DELETE, cancel, input_items)"
```

---

## Task 12: Add /models passthrough endpoint

**Files:**
- Modify: `app/api/openai_passthrough/router.py`
- Test: `tests/integration/test_openai_passthrough/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_openai_passthrough/test_models.py`:

```python
"""Integration test for GET /openai/v1/models — pure passthrough."""
import httpx


def test_list_models_forwards(client, respx_mock):
    upstream = {
        "object": "list",
        "data": [
            {"id": "openai.gpt-oss-120b", "object": "model"},
            {"id": "us.anthropic.claude-sonnet-4-6", "object": "model"},
        ],
    }
    respx_mock.get("/models").mock(return_value=httpx.Response(200, json=upstream))

    r = client.get("/openai/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 200
    assert r.json() == upstream
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_models.py -v --no-cov`
Expected: 404 — endpoint doesn't exist.

- [ ] **Step 3: Add the endpoint**

In `app/api/openai_passthrough/router.py`, add at the end:

```python
@router.get("/models")
async def list_models(
    request: Request,
    _: Dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, "/models")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_models.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/openai_passthrough/router.py tests/integration/test_openai_passthrough/test_models.py
git commit -m "feat(openai-passthrough): /models endpoint passthrough"
```

---

## Task 13: Add Bedrock guardrail header passthrough

The router's `_passthrough_extra_headers` already forwards `X-Amzn-Bedrock-*` headers. This task adds an explicit test so the behavior is locked in.

**Files:**
- Test: `tests/integration/test_openai_passthrough/test_chat_completions.py` (extend)

- [ ] **Step 1: Append the test**

Append to `tests/integration/test_openai_passthrough/test_chat_completions.py`:

```python
def test_bedrock_guardrail_headers_are_forwarded(client, respx_mock):
    """X-Amzn-Bedrock-* headers from the client should reach the upstream call."""
    route = respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "x", "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    )
    client.post(
        "/openai/v1/chat/completions",
        headers={
            "Authorization": "Bearer sk-test",
            "X-Amzn-Bedrock-GuardrailIdentifier": "GR12345",
            "X-Amzn-Bedrock-GuardrailVersion": "DRAFT",
        },
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    sent = route.calls[0].request
    assert sent.headers["x-amzn-bedrock-guardrailidentifier"] == "GR12345"
    assert sent.headers["x-amzn-bedrock-guardrailversion"] == "DRAFT"
```

- [ ] **Step 2: Run the test**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/integration/test_openai_passthrough/test_chat_completions.py::test_bedrock_guardrail_headers_are_forwarded -v --no-cov`
Expected: PASS (the router already forwards these).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_openai_passthrough/test_chat_completions.py
git commit -m "test(openai-passthrough): pin guardrail header forwarding behavior"
```

---

## Task 14: Final integration verification + full test suite

**Files:** none

- [ ] **Step 1: Run the full openai_passthrough integration suite**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit/test_openai_passthrough tests/integration/test_openai_passthrough -v --no-cov`
Expected: All tests PASS (~30 tests).

- [ ] **Step 2: Run the entire unit test suite**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit -q --no-cov`
Expected: All previously-passing tests still pass.

- [ ] **Step 3: Lint check**

Run: `unset VIRTUAL_ENV && uv run --active ruff check app/api/openai_passthrough app/middleware/auth.py app/db/dynamodb.py`
Expected: No errors. Fix any issues with `ruff check --fix`.

- [ ] **Step 4: Type check**

Run: `unset VIRTUAL_ENV && uv run --active mypy app/api/openai_passthrough 2>&1 | tail -20`
Expected: No new errors introduced. Pre-existing project-wide errors are fine — focus only on the new module.

- [ ] **Step 5: If lint/type fixes were needed, commit**

```bash
git add app/api/openai_passthrough
git commit -m "chore(openai-passthrough): lint and type cleanup"
```

(Skip this step if Steps 3 and 4 were already clean.)

---

## Task 15: Documentation updates

**Files:**
- Modify: `env.example`
- Modify: `CLAUDE.md`
- Modify: `docs/architecture/features.md`

- [ ] **Step 1: Update env.example**

Find the existing `ENABLE_OPENAI_COMPAT` block in `env.example` and add a new entry below it:

```
# OpenAI Passthrough — mount /openai/v1/* endpoints accepting native OpenAI
# Chat Completions and Responses API requests, forwarded to bedrock-mantle.
# Independent of ENABLE_OPENAI_COMPAT (the two flags can be enabled together).
# Reuses OPENAI_API_KEY and OPENAI_BASE_URL.
ENABLE_OPENAI_PASSTHROUGH=False
```

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md`, find the "Features" section (around line 95–110, after "OpenAI-Compatible API"). Add a new bullet:

```
- **OpenAI Passthrough**: New `/openai/v1/*` endpoints accept OpenAI-native Chat Completions and Responses API requests and forward them to bedrock-mantle. Distinct from `ENABLE_OPENAI_COMPAT` (which routes Anthropic-format requests on `/v1/messages`). Reuses proxy API key auth, rate limits, budgets, and usage tracking. Controlled by `ENABLE_OPENAI_PASSTHROUGH`.
```

In the "Dual API Mode" section, add a third bullet:

```
- **OpenAI Passthrough** (any model bedrock-mantle accepts, optional): When `ENABLE_OPENAI_PASSTHROUGH=True`, mounts `/openai/v1/{chat/completions,responses,responses/{id},models}` for clients using OpenAI-format directly.
```

- [ ] **Step 3: Add detailed feature doc**

Append to `docs/architecture/features.md`:

```markdown
## OpenAI Passthrough

Adds new `/openai/v1/*` endpoints that accept OpenAI-native API formats and forward them to `bedrock-mantle`. Distinct from `ENABLE_OPENAI_COMPAT` (which converts Anthropic-format requests on `/v1/messages` into OpenAI calls).

### When to use it

- You have client code using the OpenAI Python/JS SDK and want to point it at Bedrock without rewriting.
- You want stateful conversation chaining via the Responses API (`previous_response_id`, `store=true`).
- You want the proxy's API key auth, rate limits, budgets, and usage analytics for OpenAI-format traffic too.

### Configuration

```bash
ENABLE_OPENAI_PASSTHROUGH=True
OPENAI_API_KEY=<your-bedrock-api-key>
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/openai/v1/chat/completions` | Chat Completions (streaming + non-streaming) |
| POST | `/openai/v1/responses` | Responses API (streaming + non-streaming) |
| GET | `/openai/v1/responses/{id}` | Retrieve stored response |
| DELETE | `/openai/v1/responses/{id}` | Delete stored response |
| GET | `/openai/v1/responses/{id}/input_items` | List input items |
| POST | `/openai/v1/responses/{id}/cancel` | Cancel background response |
| GET | `/openai/v1/models` | List available models |

### OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(
    api_key="<your-proxy-api-key>",
    base_url="https://your-proxy.example.com/openai/v1",
)
resp = client.chat.completions.create(
    model="openai.gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Auth

Either `Authorization: Bearer <proxy-key>` (OpenAI SDK default) or `x-api-key: <proxy-key>` works. The proxy uses its configured `OPENAI_API_KEY` (Bedrock API key) for the upstream call.

### Model mapping

The existing `anthropic-proxy-model-mapping` table is consulted. If a mapping exists, the client-supplied `model` is replaced before forwarding. If no mapping exists, the model ID is passed through unchanged — so Bedrock-native IDs like `openai.gpt-oss-120b` work without registration.

### Usage tracking

Usage is normalized into the existing `anthropic-proxy-usage` schema. Two new sparse columns are written:

- `api_surface` ∈ `{"messages", "chat_completions", "responses"}`
- `reasoning_tokens` (already counted in `output_tokens`; stored separately for visibility)

For streaming Chat Completions, clients must set `stream_options: {"include_usage": true}` for usage to be captured. Without it, usage is logged as zero. The Responses API always emits `response.completed` with usage.

### Guardrails

`X-Amzn-Bedrock-*` headers from the client (e.g. `X-Amzn-Bedrock-GuardrailIdentifier`) are forwarded to bedrock-mantle.
```

- [ ] **Step 4: Commit**

```bash
git add env.example CLAUDE.md docs/architecture/features.md
git commit -m "docs(openai-passthrough): document new feature in env.example, CLAUDE.md, and features.md"
```

---

## Task 16: Final verification

- [ ] **Step 1: Sanity import the app with the flag enabled**

Run:
```bash
unset VIRTUAL_ENV && ENABLE_OPENAI_PASSTHROUGH=True uv run --active python -c "
from app.main import app
paths = sorted({r.path for r in app.routes})
expected = [
    '/openai/v1/chat/completions',
    '/openai/v1/models',
    '/openai/v1/responses',
    '/openai/v1/responses/{response_id}',
    '/openai/v1/responses/{response_id}/cancel',
    '/openai/v1/responses/{response_id}/input_items',
]
for p in expected:
    assert p in paths, f'missing {p}; got {paths}'
print('all routes registered')
"
```
Expected output: `all routes registered`

- [ ] **Step 2: Sanity import with the flag disabled**

Run:
```bash
unset VIRTUAL_ENV && ENABLE_OPENAI_PASSTHROUGH=False uv run --active python -c "
from app.main import app
paths = {r.path for r in app.routes}
assert not any(p.startswith('/openai/v1') for p in paths), f'unexpected: {[p for p in paths if p.startswith(\"/openai/v1\")]}'
print('flag-off cleanly excludes routes')
"
```
Expected output: `flag-off cleanly excludes routes`

- [ ] **Step 3: Final full test suite**

Run: `unset VIRTUAL_ENV && uv run --active pytest tests/unit tests/integration/test_openai_passthrough -q --no-cov`
Expected: All tests PASS, no failures or errors.

- [ ] **Step 4: Show final git log to confirm commit shape**

Run: `git log --oneline main..HEAD`
Expected: ~13 commits with `feat(...)`, `test(...)`, `docs(...)`, and possibly `chore(...)` prefixes.

---

## Self-Review Notes

Items I verified before finalizing:

- **Spec coverage:** All 8 implementation steps from the design doc are covered (config flag → auth → client → non-streaming chat → streaming chat → responses POST → responses CRUD → docs). Plus tasks for usage extension, model mapping, /models, guardrails, and final verification.
- **Type/name consistency:** `normalize_usage`, `try_extract_usage_from_sse`, `resolve_model_id`, `stream_passthrough`, `upstream_headers`, `_passthrough_extra_headers`, `_passthrough_request`, `_record_usage` — all introduced once and referenced consistently.
- **No placeholders:** Every code step has full code, every test has assertions, every command has expected output.
- **TDD throughout:** Each task that introduces logic starts with a failing test.
- **Frequent commits:** 13–14 separate commits, one per task, with conventional-commit prefixes matching the project's existing style.
- **Open items from design doc:**
  - OTEL tracing — explicitly deferred (not in any task).
  - Admin portal `api_surface` filter — explicitly deferred (not in any task).
  - Guardrails passthrough — included in Task 13 (test pinning the behavior already implemented in Task 8's `_passthrough_extra_headers`).
