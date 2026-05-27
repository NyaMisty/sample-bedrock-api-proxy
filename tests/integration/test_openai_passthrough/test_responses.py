"""Integration tests for POST /openai/v1/responses (streaming + non-streaming)."""
import json

import httpx

from app.schemas.anthropic import MessageResponse, Usage


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


def test_streaming_responses_synthesizes_event_lines_for_data_only_upstream(
    client, respx_mock,
):
    """Bedrock-mantle's Responses API emits data-only SSE (no `event:` lines).
    Strict clients (e.g. Codex CLI) require `event: <type>` per OpenAI spec, so
    the proxy must synthesize them from each frame's JSON `type` field.
    """
    sse_lines = [
        'data: {"type":"response.created","response":{"id":"r-1"}}',
        '',
        'data: {"type":"response.output_text.delta","delta":"hi"}',
        '',
        'data: ' + json.dumps({
            "type": "response.completed",
            "response": {"id": "r-1", "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
        }),
        '',
    ]
    body = "\n".join(sse_lines).encode()
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )

    with client.stream(
        "POST", "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "m", "input": [], "stream": True},
    ) as r:
        out = b"".join(r.iter_bytes()).decode()

    # Each data: frame with a `type` field should be preceded by an event: line
    assert "event: response.created\ndata: " in out
    assert "event: response.output_text.delta\ndata: " in out
    assert "event: response.completed\ndata: " in out


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


def test_streaming_responses_upstream_4xx_returns_json_not_sse(
    client, respx_mock, mock_usage_tracker
):
    """When upstream rejects a streaming request with 4xx, the proxy must
    surface a real JSON 4xx response — NOT a fake 200 text/event-stream
    that wraps the error body. Strict SSE clients (codex) hang waiting
    for response.completed if we send the error as event-stream.
    """
    err = {"error": {"message": "tools[13].type=namespace not allowed", "type": "validation_error"}}
    respx_mock.post("/responses").mock(return_value=httpx.Response(400, json=err))

    r = client.post(
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "m", "input": [], "stream": True, "tools": [{"type": "namespace"}]},
    )
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json"), \
        f"expected JSON content-type, got {r.headers['content-type']}"
    assert r.json() == err
    assert not mock_usage_tracker.record_usage.called


def test_non_streaming_responses_web_search_uses_local_adapter_not_upstream(
    client,
    respx_mock,
    mock_usage_tracker,
    mock_web_search_service,
):
    mock_web_search_service.handle_request.return_value = MessageResponse(
        id="msg-local",
        type="message",
        role="assistant",
        model="m",
        stop_reason="end_turn",
        content=[{"type": "text", "text": "answer"}],
        usage=Usage(
            input_tokens=3,
            output_tokens=2,
            server_tool_use={"web_search_requests": 1},
        ),
    )
    route = respx_mock.post("/responses").mock(
        return_value=httpx.Response(500, json={"error": {"message": "should not call"}})
    )

    r = client.post(
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "m",
            "input": "Search the web",
            "tools": [{"type": "web_search"}],
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "response"
    assert data["output"][0]["type"] == "web_search_call"
    assert data["output_text"] == "answer"
    assert data["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert not route.called
    assert mock_web_search_service.handle_request.called
    kw = mock_usage_tracker.record_usage.call_args.kwargs
    assert kw["api_surface"] == "responses"
    assert kw["input_tokens"] == 3
    assert kw["output_tokens"] == 2
