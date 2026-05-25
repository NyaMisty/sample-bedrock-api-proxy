"""Integration tests for POST /openai/v1/chat/completions."""
import json

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
