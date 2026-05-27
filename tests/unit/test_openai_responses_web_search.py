"""Unit tests for OpenAI Responses API web search adapter helpers."""

import pytest

from app.api.openai_passthrough.web_search import (
    OpenAIResponsesWebSearchError,
    extract_web_search_options,
    is_responses_web_search_request,
)


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
