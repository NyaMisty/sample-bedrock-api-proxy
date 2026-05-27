"""Unit tests for OpenAI Responses API web search adapter helpers."""

import pytest

from app.api.openai_passthrough.web_search import (
    OpenAIResponsesWebSearchError,
    build_message_request,
    extract_web_search_options,
    is_responses_web_search_request,
)
from app.schemas.anthropic import MessageRequest


def _message_text(req: MessageRequest, index: int) -> str:
    content = req.messages[index].model_dump()["content"]
    assert isinstance(content, list)
    first = content[0]
    assert isinstance(first, dict)
    text = first["text"]
    assert isinstance(text, str)
    return text


def test_is_responses_web_search_request_detects_current_and_preview_tools():
    assert is_responses_web_search_request({"tools": [{"type": "web_search"}]})
    assert is_responses_web_search_request({"tools": [{"type": "web_search_preview"}]})
    assert not is_responses_web_search_request({"tools": [{"type": "function", "name": "x"}]})
    assert not is_responses_web_search_request({"input": "hi"})


def test_extract_web_search_options_maps_filters_and_location():
    options = extract_web_search_options(
        {
            "tools": [
                {
                    "type": "web_search",
                    "filters": {
                        "allowed_domains": ["docs.python.org"],
                        "blocked_domains": ["example.com"],
                    },
                    "user_location": {
                        "type": "approximate",
                        "city": "Seattle",
                        "region": "WA",
                        "country": "US",
                        "timezone": "America/Los_Angeles",
                    },
                    "search_context_size": "medium",
                }
            ]
        }
    )

    assert options.allowed_domains == ["docs.python.org"]
    assert options.blocked_domains == ["example.com"]
    assert options.user_location is not None
    assert options.user_location.city == "Seattle"
    assert options.search_context_size == "medium"


def test_extract_web_search_options_rejects_external_web_access_false():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        extract_web_search_options(
            {"tools": [{"type": "web_search", "external_web_access": False}]}
        )

    assert exc.value.status_code == 400
    assert exc.value.error_type == "invalid_request_error"
    assert "external_web_access" in exc.value.message


def test_extract_web_search_options_rejects_return_token_budget():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        extract_web_search_options(
            {"tools": [{"type": "web_search", "return_token_budget": 1200}]}
        )

    assert exc.value.status_code == 400
    assert "return_token_budget" in exc.value.message


def test_extract_web_search_options_rejects_present_non_object_filters():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        extract_web_search_options({"tools": [{"type": "web_search", "filters": []}]})

    assert "filters" in exc.value.message


def test_extract_web_search_options_rejects_conflicting_multiple_tools():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        extract_web_search_options(
            {
                "tools": [
                    {"type": "web_search", "filters": {"allowed_domains": ["a.com"]}},
                    {"type": "web_search", "filters": {"allowed_domains": ["b.com"]}},
                ]
            }
        )

    assert exc.value.status_code == 400
    assert "Conflicting" in exc.value.message


def test_build_message_request_converts_string_input_and_instructions():
    req = build_message_request(
        {
            "model": "openai.gpt-oss-120b",
            "instructions": "Be concise.",
            "input": "What changed in Python 3.13?",
            "max_output_tokens": 777,
            "temperature": 0.2,
            "top_p": 0.9,
            "tools": [{"type": "web_search"}],
        }
    )

    assert isinstance(req, MessageRequest)
    assert req.model == "openai.gpt-oss-120b"
    assert req.max_tokens == 777
    assert req.system is not None
    assert req.messages[0].role == "user"
    assert _message_text(req, 0) == "What changed in Python 3.13?"
    assert req.temperature == 0.2
    assert req.top_p == 0.9
    assert req.tools == [{"type": "web_search_20250305", "name": "web_search"}]


def test_build_message_request_converts_responses_input_array_and_filters():
    req = build_message_request(
        {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Find current news"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "What topic?"}],
                },
                {
                    "role": "user",
                    "content": "AI infrastructure",
                },
            ],
            "tools": [
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": ["example.com"]},
                }
            ],
        }
    )

    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    assert _message_text(req, 0) == "Find current news"
    assert _message_text(req, 1) == "What topic?"
    assert _message_text(req, 2) == "AI infrastructure"
    assert req.tools == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["example.com"],
        }
    ]


def test_build_message_request_rejects_missing_input():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        build_message_request({"model": "m", "tools": [{"type": "web_search"}]})

    assert exc.value.status_code == 400
    assert "input" in exc.value.message


def test_build_message_request_maps_web_search_tool_choice_dict():
    req = build_message_request(
        {
            "model": "m",
            "input": "Find news",
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "web_search"},
        }
    )

    assert req.tool_choice == {"type": "tool", "name": "web_search"}


def test_build_message_request_maps_function_tool_choice_dict():
    req = build_message_request(
        {
            "model": "m",
            "input": "Find news",
            "tools": [{"type": "web_search"}],
            "tool_choice": {"function": {"name": "web_search"}},
        }
    )

    assert req.tool_choice == {"type": "tool", "name": "web_search"}


def test_build_message_request_rejects_zero_max_output_tokens():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        build_message_request(
            {
                "model": "m",
                "input": "Find news",
                "max_output_tokens": 0,
                "tools": [{"type": "web_search"}],
            }
        )

    assert "max_output_tokens" in exc.value.message


def test_build_message_request_rejects_invalid_max_output_tokens():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        build_message_request(
            {
                "model": "m",
                "input": "Find news",
                "max_output_tokens": "abc",
                "tools": [{"type": "web_search"}],
            }
        )

    assert "max_output_tokens" in exc.value.message


def test_build_message_request_rejects_invalid_user_location_field_type():
    with pytest.raises(OpenAIResponsesWebSearchError) as exc:
        build_message_request(
            {
                "model": "m",
                "input": "Find news",
                "tools": [
                    {
                        "type": "web_search",
                        "user_location": {"type": "approximate", "city": 123},
                    }
                ],
            }
        )

    assert "user_location" in exc.value.message
