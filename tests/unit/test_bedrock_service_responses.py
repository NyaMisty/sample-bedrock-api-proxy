"""Tests for BedrockService Responses-mode dispatch and provider-aware compat client.

The OpenAI-compat service is imported lazily INSIDE BedrockService.__init__
via `from app.services.openai_compat_service import OpenAICompatService`, so we
patch it at its source module `app.services.openai_compat_service.OpenAICompatService`.

We also pass a MagicMock dynamodb_client so __init__ does not hit AWS for
DynamoDB. The boto3 bedrock-runtime client construction is patched out.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.anthropic import Message, MessageRequest, MessageResponse


def _make_request(model: str) -> MessageRequest:
    return MessageRequest(
        model=model,
        messages=[Message(role="user", content="hi")],
        max_tokens=16,
    )


def _make_response() -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "ok"}],
        model="m",
        stop_reason="end_turn",
        usage={"input_tokens": 1, "output_tokens": 1},
    )


def _build_service(**kwargs):
    """Build BedrockService with boto3 + compat-service patched out.

    Returns (service, compat_mock, compat_cls_mock).
    """
    compat_mock = MagicMock()
    sentinel = _make_response()
    compat_mock.invoke_responses = AsyncMock(return_value=sentinel)
    compat_mock.invoke_model = AsyncMock(return_value=sentinel)

    with patch("boto3.client", return_value=MagicMock()), patch(
        "app.services.openai_compat_service.OpenAICompatService",
        return_value=compat_mock,
    ) as compat_cls:
        from app.services.bedrock_service import BedrockService

        svc = BedrockService(dynamodb_client=MagicMock(), **kwargs)
    return svc, compat_mock, compat_cls, sentinel


async def test_responses_mode_dispatches_to_invoke_responses():
    svc, compat_mock, _cls, sentinel = _build_service(
        openai_base_url="https://prov.test/openai/v1",
        openai_api_key="k",
        openai_use_responses=True,
    )

    req = _make_request("amazon.nova-pro-v1:0")
    result = await svc.invoke_model(req)

    compat_mock.invoke_responses.assert_awaited_once()
    compat_mock.invoke_model.assert_not_called()
    assert result is sentinel


async def test_default_mode_uses_chat_completions():
    svc, compat_mock, _cls, sentinel = _build_service(
        openai_base_url="https://prov.test/openai/v1",
        openai_api_key="k",
        openai_use_responses=False,
    )

    req = _make_request("amazon.nova-pro-v1:0")
    result = await svc.invoke_model(req)

    compat_mock.invoke_model.assert_awaited_once()
    compat_mock.invoke_responses.assert_not_called()
    assert result is sentinel


def test_provider_override_enables_compat_when_global_empty(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "", raising=False)
    monkeypatch.setattr(settings, "enable_openai_compat", False, raising=False)

    svc, _compat_mock, compat_cls, _sentinel = _build_service(
        openai_base_url="https://prov.test/openai/v1",
        openai_api_key="k",
        openai_use_responses=True,
    )

    assert svc._openai_compat_service is not None
    compat_cls.assert_called_once_with(
        base_url="https://prov.test/openai/v1",
        api_key="k",
    )


async def test_claude_model_not_routed_to_compat():
    svc, compat_mock, _cls, _sentinel = _build_service(
        openai_base_url="https://prov.test/openai/v1",
        openai_api_key="k",
        openai_use_responses=True,
    )

    req = _make_request("claude-sonnet-4-5")

    # Claude model must not be dispatched to the compat service. We don't care
    # what the boto3 path does (it's mocked / may error); we only assert the
    # compat methods were never invoked. Patch the sync path so we get a clean
    # return instead of a boto3 error after the dispatch decision.
    with patch.object(svc, "_invoke_model_sync", return_value=_make_response()):
        await svc.invoke_model(req)

    compat_mock.invoke_responses.assert_not_called()
    compat_mock.invoke_model.assert_not_called()
