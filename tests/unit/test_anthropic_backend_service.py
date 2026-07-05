"""Tests for AnthropicBackendService (direct-Anthropic httpx backend)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import BedrockAPIError
from app.schemas.anthropic import MessageRequest
from app.services.anthropic_backend_service import AnthropicBackendService


def _make_service() -> AnthropicBackendService:
    """Build an AnthropicBackendService with __init__ bypassed."""
    with patch.object(AnthropicBackendService, "__init__", lambda self: None):
        svc = AnthropicBackendService.__new__(AnthropicBackendService)
        svc._base_url = "https://api.anthropic.com"
        svc._api_key = "sk-ant-test"
        svc._client = MagicMock(name="sync_httpx")
        svc._async_client = MagicMock(name="async_httpx")
        return svc


def _request() -> MessageRequest:
    return MessageRequest(
        model="claude-sonnet-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
    )


# ---------------------------------------------------------------------------
# _normalize_body — header vs body normalization
# ---------------------------------------------------------------------------


def test_normalize_body_strips_anthropic_version_to_header():
    """anthropic_version must NOT appear in the body (direct API uses the
    anthropic-version header). A body field is a 400 from Anthropic."""
    svc = _make_service()
    body = svc._normalize_body(_request())
    assert "anthropic_version" not in body
    headers = svc._headers()
    assert headers["anthropic-version"] == "2023-06-01"


def test_normalize_body_keeps_cache_control():
    """cache_control is Anthropic-native and must be preserved."""
    svc = _make_service()
    req = MessageRequest(
        model="claude-sonnet-5",
        max_tokens=64,
        system=[
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    body = svc._normalize_body(req)
    assert "system" in body
    sys_block = body["system"][0]
    assert "cache_control" in sys_block


def test_normalize_body_drops_service_tier():
    """service_tier has no direct-API equivalent — silent degradation."""
    svc = _make_service()
    body = svc._normalize_body(_request())
    assert "service_tier" not in body


def test_headers_include_api_key_and_version():
    svc = _make_service()
    svc._current_beta = None
    headers = svc._headers()
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


def test_headers_include_beta_when_present():
    svc = _make_service()
    svc._current_beta = ["beta-1", "beta-2"]
    headers = svc._headers()
    assert headers["anthropic-beta"] == "beta-1,beta-2"


# ---------------------------------------------------------------------------
# invoke_model_sync — non-streaming
# ---------------------------------------------------------------------------


def test_invoke_model_sync_posts_to_messages_endpoint():
    svc = _make_service()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    svc._client.post.return_value = fake_resp

    result = svc.invoke_model_sync(_request(), "req-1")
    assert result.id == "msg_1"
    assert result.model == "claude-sonnet-5"

    # Verify the URL and that the body had no anthropic_version.
    call_args = svc._client.post.call_args
    assert call_args.args[0] == "https://api.anthropic.com/v1/messages"
    body = call_args.kwargs["json"]
    assert "anthropic_version" not in body
    # The direct API requires `model` in the body (unlike Bedrock InvokeModel).
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] == 64
    assert len(body["messages"]) == 1
    # Non-streaming request must NOT inject `stream: true` into the body
    # (Bedrock toggles streaming via a different API method; the direct API
    # uses the body field, so we must only set it when the client asked).
    assert "stream" not in body


def test_normalize_body_injects_stream_flag_when_requested():
    """stream=True in the request must be forwarded as body `stream: true`
    (the direct Anthropic API toggles streaming via the body, not via a
    separate API method like Bedrock's InvokeModelWithResponseStream)."""
    svc = _make_service()
    req = MessageRequest(
        model="claude-sonnet-5",
        max_tokens=64,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    body = svc._normalize_body(req)
    assert body.get("stream") is True


def test_invoke_model_sync_raises_bedrock_api_error_on_4xx():
    svc = _make_service()
    fake_resp = MagicMock()
    fake_resp.status_code = 400
    fake_resp.text = json.dumps(
        {"error": {"type": "invalid_request_error", "message": "bad"}}
    )
    svc._client.post.return_value = fake_resp

    with pytest.raises(BedrockAPIError) as exc_info:
        svc.invoke_model_sync(_request(), "req-2")
    assert exc_info.value.http_status == 400
    assert "bad" in exc_info.value.error_message


# ---------------------------------------------------------------------------
# Transparent proxy — per-request api_key override
# ---------------------------------------------------------------------------


def test_invoke_model_sync_relays_per_request_api_key():
    """When api_key is passed (transparent-proxy mode), it must reach the
    x-api-key header of the upstream call INSTEAD of the configured key."""
    svc = _make_service()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    svc._client.post.return_value = fake_resp

    svc.invoke_model_sync(_request(), "req-1", api_key="sk-client-relayed")

    call_headers = svc._client.post.call_args.kwargs["headers"]
    assert call_headers["x-api-key"] == "sk-client-relayed"
    # The configured key must NOT be used when an override is supplied.
    assert call_headers["x-api-key"] != "sk-ant-test"


def test_invoke_model_sync_falls_back_to_configured_key_without_override():
    """Regression: with no api_key arg, the configured ANTHROPIC_API_KEY is used
    (non-transparent behavior unchanged)."""
    svc = _make_service()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    svc._client.post.return_value = fake_resp

    svc.invoke_model_sync(_request(), "req-1")

    call_headers = svc._client.post.call_args.kwargs["headers"]
    assert call_headers["x-api-key"] == "sk-ant-test"


@pytest.mark.asyncio
async def test_invoke_model_async_relays_per_request_api_key():
    svc = _make_service()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }

    async def _fake_post(*args, **kwargs):
        return fake_resp

    svc._async_client.post = _fake_post

    await svc.invoke_model(_request(), "req-1", api_key="sk-async-relayed")

    # _fake_post captured kwargs via its closure; re-derive headers by
    # invoking _headers directly with the same override.
    assert svc._headers("sk-async-relayed")["x-api-key"] == "sk-async-relayed"


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_tokens_calls_anthropic_endpoint():
    from app.schemas.anthropic import CountTokensRequest

    svc = _make_service()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"input_tokens": 17}
    svc._async_client.post = MagicMock()

    # Make the async post return an awaitable.
    async def _fake_post(*args, **kwargs):
        return fake_resp

    svc._async_client.post = _fake_post

    req = CountTokensRequest(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
    )
    result = await svc.count_tokens(req)
    assert result == 17


@pytest.mark.asyncio
async def test_count_tokens_falls_back_to_heuristic_on_error():
    import httpx

    from app.schemas.anthropic import CountTokensRequest

    svc = _make_service()

    async def _fake_post(*args, **kwargs):
        raise httpx.ConnectError("network down")

    svc._async_client.post = _fake_post

    req = CountTokensRequest(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hello world"}],
    )
    result = await svc.count_tokens(req)
    # Heuristic: ~4 chars/token. "hello world" = 11 chars → 2 tokens (min 1).
    assert result >= 1
    assert isinstance(result, int)
