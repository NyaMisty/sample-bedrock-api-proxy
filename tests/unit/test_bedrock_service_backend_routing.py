"""Tests for BedrockService backend short-circuit routing.

Verifies that BACKEND_MODE dispatches ALL models (including claude-*) to the
selected non-Bedrock backend BEFORE the _is_claude_model branch, and that
bedrock mode preserves the existing routing (claude → native, non-claude →
OpenAI compat / Converse).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.anthropic import MessageRequest
from app.services.bedrock_service import BedrockService


def _make_service(backend_mode: str) -> BedrockService:
    """Build a BedrockService with __init__ bypassed and backend mode set."""
    with patch.object(BedrockService, "__init__", lambda self: None):
        svc = BedrockService.__new__(BedrockService)
        svc._backend_mode = backend_mode
        svc._openai_use_responses = False
        svc._anthropic_backend_service = None
        svc._openai_compat_service = None
        return svc


def _request(model: str) -> MessageRequest:
    return MessageRequest(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
    )


# ---------------------------------------------------------------------------
# Anthropic backend short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_mode_routes_claude_model_to_anthropic_backend():
    svc = _make_service("anthropic")
    mock_backend = MagicMock()
    mock_backend.invoke_model = AsyncMock(return_value="anthropic-response")
    svc._anthropic_backend_service = mock_backend

    result = await svc.invoke_model(_request("claude-sonnet-5"), "req-1")

    assert result == "anthropic-response"
    mock_backend.invoke_model.assert_awaited_once()
    # Verify the request passed through unchanged.
    passed_req = mock_backend.invoke_model.await_args.args[0]
    assert passed_req.model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_anthropic_mode_routes_non_claude_model_to_anthropic_backend():
    svc = _make_service("anthropic")
    mock_backend = MagicMock()
    mock_backend.invoke_model = AsyncMock(return_value="anthropic-response")
    svc._anthropic_backend_service = mock_backend

    await svc.invoke_model(_request("gpt-5.5"), "req-2")

    # Non-Claude model must NOT fall through to the _is_claude_model /
    # _openai_compat_service branch — it must hit the anthropic backend.
    mock_backend.invoke_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_mode_count_tokens_uses_anthropic_backend():
    from app.schemas.anthropic import CountTokensRequest

    svc = _make_service("anthropic")
    mock_backend = MagicMock()
    mock_backend.count_tokens = AsyncMock(return_value=42)
    svc._anthropic_backend_service = mock_backend

    req = CountTokensRequest(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
    )
    result = await svc.count_tokens(req)
    assert result == 42
    mock_backend.count_tokens.assert_awaited_once()


# ---------------------------------------------------------------------------
# OpenAI backend short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_mode_routes_claude_model_to_openai_compat():
    """The key gap-fix: claude-* models MUST go to OpenAI when mode=openai."""
    svc = _make_service("openai")
    mock_compat = MagicMock()
    mock_compat.invoke_model = AsyncMock(return_value="openai-response")
    svc._openai_compat_service = mock_compat

    result = await svc.invoke_model(_request("claude-sonnet-5"), "req-3")
    assert result == "openai-response"
    mock_compat.invoke_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_mode_count_tokens_falls_back_to_heuristic():
    from app.schemas.anthropic import CountTokensRequest

    svc = _make_service("openai")
    svc._estimate_token_count = MagicMock(return_value=99)

    req = CountTokensRequest(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
    )
    result = await svc.count_tokens(req)
    assert result == 99
    svc._estimate_token_count.assert_called_once()


# ---------------------------------------------------------------------------
# Bedrock mode regression — must NOT short-circuit to the new backends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bedrock_mode_does_not_call_anthropic_or_openai_backends():
    """In bedrock mode the short-circuits must be no-ops: neither the
    anthropic backend nor the openai compat service is invoked. We patch
    the sync inner method (the entry point that runs after the
    short-circuits) to assert the request reaches the Bedrock path."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    import app.services.bedrock_service as bs

    svc = _make_service("bedrock")
    svc._anthropic_backend_service = None
    svc._openai_compat_service = None  # ensure no Converse→OpenAI fallthrough

    # Patch the inner sync method so we don't actually call boto3. It's
    # reached only when the short-circuits are skipped (i.e. bedrock mode).
    captured = {"called": False}

    def _fake_inner(self, request, request_id, *args, **kwargs):
        captured["called"] = True
        captured["model"] = request.model
        return "bedrock-response"

    def _fake_invoke_model_sync(self, request, *args, **kwargs):
        # _invoke_model_sync is sync (runs in thread pool). Signature:
        # (self, request, request_id, service_tier, anthropic_beta,
        #  otel_ctx, cache_ttl, provider_id).
        return _fake_inner(self, request, args[0] if args else None)

    real_executor = ThreadPoolExecutor(max_workers=1)
    with (
        patch.object(bs.BedrockService, "_invoke_model_sync_inner", _fake_inner),
        patch.object(bs.BedrockService, "_invoke_model_sync", _fake_invoke_model_sync),
        patch(
            "app.services.bedrock_service._get_semaphore",
            return_value=asyncio.Semaphore(1),
        ),
        patch(
            "app.services.bedrock_service._get_executor",
            return_value=real_executor,
        ),
    ):
        result = await svc.invoke_model(_request("claude-sonnet-5"), "req-4")
        assert result == "bedrock-response"
        assert captured["called"] is True
        assert captured["model"] == "claude-sonnet-5"
