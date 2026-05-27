"""OpenAI Responses API web search compatibility.

This module adapts OpenAI Responses requests that use the hosted web_search
tool to the proxy's existing Anthropic Messages web search implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.web_search import UserLocation

OPENAI_WEB_SEARCH_TOOL_TYPES = {"web_search", "web_search_preview"}


class OpenAIResponsesWebSearchError(Exception):
    """Error that should be returned as an OpenAI-style JSON error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type

    def to_error_body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.error_type}}


@dataclass(frozen=True)
class OpenAIWebSearchOptions:
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: UserLocation | None = None
    search_context_size: str | None = None


def _web_search_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [
        tool
        for tool in tools
        if isinstance(tool, dict) and tool.get("type") in OPENAI_WEB_SEARCH_TOOL_TYPES
    ]


def is_responses_web_search_request(body: dict[str, Any]) -> bool:
    return bool(_web_search_tools(body))


def _as_str_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise OpenAIResponsesWebSearchError(f"{field_name} must be an array of strings")
    return value


def _parse_user_location(raw: Any) -> UserLocation | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise OpenAIResponsesWebSearchError("user_location must be an object")
    if raw.get("type", "approximate") != "approximate":
        raise OpenAIResponsesWebSearchError("Only approximate user_location is supported")
    return UserLocation(
        type="approximate",
        city=raw.get("city"),
        region=raw.get("region"),
        country=raw.get("country"),
        timezone=raw.get("timezone"),
    )


def _options_signature(options: OpenAIWebSearchOptions) -> tuple[Any, ...]:
    return (
        tuple(options.allowed_domains or []),
        tuple(options.blocked_domains or []),
        options.user_location.model_dump() if options.user_location else None,
        options.search_context_size,
    )


def extract_web_search_options(body: dict[str, Any]) -> OpenAIWebSearchOptions:
    tools = _web_search_tools(body)
    if not tools:
        raise OpenAIResponsesWebSearchError("No web_search tool found")

    parsed: list[OpenAIWebSearchOptions] = []
    for tool in tools:
        if tool.get("external_web_access") is False:
            raise OpenAIResponsesWebSearchError(
                "external_web_access=false is not supported by this proxy"
            )
        if "return_token_budget" in tool:
            raise OpenAIResponsesWebSearchError(
                "return_token_budget is not supported by this proxy"
            )

        filters = tool.get("filters")
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise OpenAIResponsesWebSearchError("filters must be an object")

        location = tool.get("user_location", body.get("user_location"))
        parsed.append(
            OpenAIWebSearchOptions(
                allowed_domains=_as_str_list(
                    filters.get("allowed_domains"), "filters.allowed_domains"
                ),
                blocked_domains=_as_str_list(
                    filters.get("blocked_domains"), "filters.blocked_domains"
                ),
                user_location=_parse_user_location(location),
                search_context_size=tool.get("search_context_size"),
            )
        )

    first = parsed[0]
    first_sig = _options_signature(first)
    if any(_options_signature(item) != first_sig for item in parsed[1:]):
        raise OpenAIResponsesWebSearchError(
            "Conflicting web_search tool definitions are not supported"
        )
    return first
