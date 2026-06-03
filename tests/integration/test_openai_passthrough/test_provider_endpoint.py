"""Per-key provider endpoint override for the OpenAI passthrough.

Regression test for the bug where an API key configured with a model provider
(``endpoint_url=https://bedrock-mantle.us-east-2.api.aws/openai/v1``) was
ignored: the passthrough still forwarded to the global ``MANTLE_ENDPOINT_URL``
default, producing "model id does not exist" for region-specific models.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import respx


def _provider_manager_for(endpoint_url: str, bearer: str) -> MagicMock:
    mgr = MagicMock()
    mgr.get_provider.return_value = {
        "provider_id": "prov-east2",
        "is_active": True,
        "auth_type": "bearer_token",
        "endpoint_url": endpoint_url,
    }
    mgr.get_decrypted_credentials.return_value = {"bearer_token": bearer}
    return mgr


def test_chat_completions_uses_per_key_provider_endpoint(
    client, mock_api_key_manager, mock_model_mapping_manager
):
    # The key is associated with a provider pointing at the us-east-2 mantle
    # OpenAI endpoint — distinct from the global default (https://mantle.test/v1).
    mock_api_key_manager.validate_api_key.return_value = {
        "api_key": "sk-test",
        "user_id": "u1",
        "is_master": False,
        "rate_limit": None,
        "cache_ttl": None,
        "provider_id": "prov-east2",
    }
    provider_endpoint = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
    provider_mgr = _provider_manager_for(provider_endpoint, "east2-bedrock-key")

    upstream_resp = {
        "id": "resp-1",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    with patch(
        "app.api.openai_passthrough.router.ProviderManager",
        return_value=provider_mgr,
        create=True,
    ), respx.mock(assert_all_called=False) as rmock:
        provider_route = rmock.post(f"{provider_endpoint}/responses").mock(
            return_value=httpx.Response(200, json=upstream_resp)
        )
        default_route = rmock.post("https://mantle.test/v1/responses").mock(
            return_value=httpx.Response(200, json=upstream_resp)
        )

        r = client.post(
            "/openai/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert r.status_code == 200
    # Must hit the per-key provider endpoint, never the global default.
    assert provider_route.called, "request should go to the per-key provider endpoint"
    assert not default_route.called, "must not fall back to the default endpoint"
    sent = provider_route.calls[0].request
    # Provider's own credential is used, not the global Bedrock API key.
    assert sent.headers["authorization"] == "Bearer east2-bedrock-key"
    assert json.loads(sent.content)["model"] == "gpt-5.5"
    assert json.loads(sent.content)["store"] is False


def test_chat_completions_without_provider_uses_default_endpoint(
    client, mock_api_key_manager, mock_model_mapping_manager
):
    # No provider_id on the key → behaviour is unchanged (global default).
    upstream_resp = {
        "id": "resp-1",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    route = None
    with respx.mock(base_url="https://mantle.test/v1", assert_all_called=False) as rmock:
        route = rmock.post("/responses").mock(
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
    assert route.called
    assert route.calls[0].request.headers["authorization"] == "Bearer bedrock-key-test"
