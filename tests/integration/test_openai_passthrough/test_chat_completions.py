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


def test_streaming_upstream_timeout_yields_clean_sse_error(
    client, respx_mock, mock_usage_tracker
):
    """Upstream timeout during streaming should produce a structured SSE error event, not crash the stream."""
    respx_mock.post("/chat/completions").mock(
        side_effect=httpx.ReadTimeout("upstream took too long")
    )

    with client.stream(
        "POST",
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "m", "messages": [], "stream": True},
    ) as r:
        out = b"".join(r.iter_bytes())

    assert b'"upstream_error"' in out, f"expected structured error, got: {out}"
    assert b"[DONE]" in out
    # No usage logged when the stream errored before any usage event arrived
    assert not mock_usage_tracker.record_usage.called
