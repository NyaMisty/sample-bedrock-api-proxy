"""Web-search branch of the OpenAI passthrough Responses endpoint.

Regression tests for the bug where the web-search agentic loop ignored the
per-key provider endpoint (hitting the global us-east-1 mantle, producing
"model does not exist") and used Chat Completions instead of the Responses
API (which Responses-only models like ``openai.gpt-5.5`` don't support).

The fix routes the loop's model calls to the per-key provider endpoint and
drives non-Claude models via the OpenAI Responses API (``store=False``) by
constructing ``BedrockService(openai_base_url=..., openai_api_key=...,
openai_use_responses=True)``.
"""

from unittest.mock import MagicMock, patch

from app.schemas.anthropic import MessageResponse, TextContent, Usage


def _message_response() -> MessageResponse:
    return MessageResponse(
        id="msg_websearch",
        model="claude",
        role="assistant",
        content=[TextContent(type="text", text="done")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _provider_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.get_provider.return_value = {
        "provider_id": "prov-east2",
        "is_active": True,
        "auth_type": "bearer_token",
        "endpoint_url": "https://prov.test/openai/v1",
    }
    mgr.get_decrypted_credentials.return_value = {"bearer_token": "prov-bearer"}
    return mgr


def test_websearch_branch_builds_provider_aware_responses_bedrock_service(
    client, mock_api_key_manager, mock_web_search_service, mock_bedrock_service
):
    # Key is associated with a provider pointing at a region-specific endpoint.
    mock_api_key_manager.validate_api_key.return_value = {
        "api_key": "sk-test",
        "user_id": "u1",
        "is_master": False,
        "rate_limit": None,
        "cache_ttl": None,
        "provider_id": "prov-east2",
    }
    mock_web_search_service.handle_request.return_value = _message_response()

    with patch(
        "app.api.openai_passthrough.router.ProviderManager",
        return_value=_provider_manager(),
        create=True,
    ):
        r = client.post(
            "/openai/v1/responses",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "openai.gpt-5.5",
                "input": "latest AWS news",
                "tools": [{"type": "web_search"}],
            },
        )

    assert r.status_code == 200, r.text
    # The web-search loop's BedrockService is wired to the per-key provider
    # endpoint AND driven through the Responses API.
    mock_bedrock_service.constructor_mock.assert_called_once_with(
        openai_base_url="https://prov.test/openai/v1",
        openai_api_key="prov-bearer",
        openai_use_responses=True,
    )


def test_websearch_without_provider_uses_global_responses_mode(
    client, mock_api_key_manager, mock_web_search_service, mock_bedrock_service
):
    # No provider on the key → fall back to globals, but still Responses mode.
    mock_web_search_service.handle_request.return_value = _message_response()

    r = client.post(
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "openai.gpt-5.5",
            "input": "latest AWS news",
            "tools": [{"type": "web_search"}],
        },
    )

    assert r.status_code == 200, r.text
    mock_bedrock_service.constructor_mock.assert_called_once_with(
        openai_base_url=None,
        openai_api_key=None,
        openai_use_responses=True,
    )
