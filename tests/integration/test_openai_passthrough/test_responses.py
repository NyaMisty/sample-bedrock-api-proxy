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
