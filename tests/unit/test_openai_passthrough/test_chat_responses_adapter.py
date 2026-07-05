"""Tests for Chat Completions -> Responses API request adaptation."""

import pytest

from app.api.openai_passthrough.chat_responses_adapter import (
    UNSUPPORTED_PARAM_CACHE_TTL_SECONDS,
    chat_request_to_response_request,
    pop_unsupported_parameter,
    reset_unsupported_param_cache_for_testing,
    strip_learned_unsupported_params,
)


@pytest.fixture(autouse=True)
def _clean_unsupported_param_cache():
    reset_unsupported_param_cache_for_testing()
    yield
    reset_unsupported_param_cache_for_testing()


class TestChatRequestToResponseRequest:
    def test_string_content_passes_through(self):
        body = {
            "model": "openai.gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = chat_request_to_response_request(body)
        assert result["input"] == [{"role": "user", "content": "hello"}]

    def test_user_text_parts_converted_to_input_text(self):
        """Chat Completions {"type": "text"} parts must become input_text.

        Regression: upstream Responses API rejects Chat Completions part
        types with 400 "Invalid 'input': value did not match any expected
        variant".
        """
        body = {
            "model": "openai.gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ],
        }
        result = chat_request_to_response_request(body)
        assert result["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "part one"},
                    {"type": "input_text", "text": "part two"},
                ],
            }
        ]

    def test_assistant_text_parts_converted_to_output_text(self):
        """Assistant history text parts must become output_text with annotations."""
        body = {
            "model": "openai.gpt-5.5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hello there", "annotations": []}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "what model"}]},
            ],
        }
        result = chat_request_to_response_request(body)
        assert result["input"][1] == {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "hello there", "annotations": []}
            ],
        }

    def test_image_url_part_converted_to_input_image(self):
        body = {
            "model": "openai.gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/cat.png",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        result = chat_request_to_response_request(body)
        assert result["input"][0]["content"] == [
            {"type": "input_text", "text": "describe"},
            {
                "type": "input_image",
                "image_url": "https://example.com/cat.png",
                "detail": "high",
            },
        ]

    def test_already_native_parts_pass_through(self):
        body = {
            "model": "openai.gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "native"}],
                }
            ],
        }
        result = chat_request_to_response_request(body)
        assert result["input"][0]["content"] == [
            {"type": "input_text", "text": "native"}
        ]

    def test_unsupported_parameter_dropped_from_body(self):
        """xai.grok-4.3 rejects 'temperature' on /responses; drop it and retry."""
        body = {"model": "xai.grok-4.3", "temperature": 1, "input": []}
        error = {
            "error": {
                "code": "unsupported_parameter",
                "message": (
                    "Unsupported parameter: 'temperature' is not supported "
                    "with this model."
                ),
                "param": "temperature",
                "type": "invalid_request_error",
            }
        }
        assert pop_unsupported_parameter(body, 400, error) == "temperature"
        assert "temperature" not in body

    def test_unknown_parameter_code_also_dropped(self):
        """grok Responses uses code=unknown_parameter for stop/seed."""
        body = {"model": "xai.grok-4.3", "stop": ["x"], "input": []}
        error = {
            "error": {
                "code": "unknown_parameter",
                "message": "Unknown parameter: 'stop'.",
                "param": "stop",
                "type": "invalid_request_error",
            }
        }
        assert pop_unsupported_parameter(body, 400, error) == "stop"
        assert "stop" not in body

    def test_unsupported_parameter_ignores_other_errors(self):
        body = {"model": "m", "temperature": 1}
        validation_error = {
            "error": {
                "code": "validation_error",
                "message": "invalid request body",
                "param": None,
                "type": "invalid_request_error",
            }
        }
        assert pop_unsupported_parameter(body, 400, validation_error) is None
        assert pop_unsupported_parameter(body, 500, validation_error) is None
        assert pop_unsupported_parameter(body, 400, "not json") is None
        assert pop_unsupported_parameter(body, 400, {"error": "str"}) is None
        assert body["temperature"] == 1

    def test_unsupported_parameter_absent_from_body_returns_none(self):
        """Don't retry if the named param isn't in our request (avoid loops)."""
        body = {"model": "m"}
        error = {
            "error": {
                "code": "unsupported_parameter",
                "param": "temperature",
                "type": "invalid_request_error",
            }
        }
        assert pop_unsupported_parameter(body, 400, error) is None

    def test_learned_param_stripped_proactively_on_next_request(self):
        """After one 400, later requests for the same model skip the round-trip."""
        error = {
            "error": {
                "code": "unsupported_parameter",
                "param": "temperature",
                "type": "invalid_request_error",
            }
        }
        first = {"model": "xai.grok-4.3", "temperature": 1, "input": []}
        assert pop_unsupported_parameter(first, 400, error) == "temperature"

        second = {"model": "xai.grok-4.3", "temperature": 0.5, "top_p": 0.9}
        assert strip_learned_unsupported_params(second) == ["temperature"]
        assert "temperature" not in second
        assert second["top_p"] == 0.9  # only learned params stripped

        # A different model is unaffected
        other = {"model": "openai.gpt-5.5", "temperature": 1}
        assert strip_learned_unsupported_params(other) == []
        assert other["temperature"] == 1

    def test_learned_param_cache_expires_after_ttl(self, monkeypatch):
        error = {
            "error": {
                "code": "unsupported_parameter",
                "param": "temperature",
                "type": "invalid_request_error",
            }
        }
        pop_unsupported_parameter(
            {"model": "xai.grok-4.3", "temperature": 1}, 400, error
        )

        import app.api.openai_passthrough.chat_responses_adapter as adapter

        real_monotonic = adapter.time.monotonic
        monkeypatch.setattr(
            adapter.time,
            "monotonic",
            lambda: real_monotonic() + UNSUPPORTED_PARAM_CACHE_TTL_SECONDS + 1,
        )
        body = {"model": "xai.grok-4.3", "temperature": 1}
        assert strip_learned_unsupported_params(body) == []
        assert body["temperature"] == 1  # expired entry no longer strips

    def test_system_and_tool_messages_unaffected(self):
        body = {
            "model": "openai.gpt-5.5",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "run the tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
        }
        result = chat_request_to_response_request(body)
        assert result["instructions"] == "be helpful"
        assert result["input"] == [
            {"role": "user", "content": "run the tool"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "f",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
