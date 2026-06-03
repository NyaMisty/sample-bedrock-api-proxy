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
    assert result["input_tokens"] == 100
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
    assert result["input_tokens"] == 100
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
