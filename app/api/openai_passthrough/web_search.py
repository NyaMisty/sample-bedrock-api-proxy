"""OpenAI Responses API web search compatibility.

This module adapts OpenAI Responses requests that use the hosted web_search
tool to the proxy's existing Anthropic Messages web search implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.anthropic import MessageRequest
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


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"input_text", "output_text", "text"}:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


def _convert_input_to_messages(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        if not input_value.strip():
            raise OpenAIResponsesWebSearchError("input must not be empty")
        return [{"role": "user", "content": [{"type": "text", "text": input_value}]}]

    if not isinstance(input_value, list) or not input_value:
        raise OpenAIResponsesWebSearchError("input is required for web_search requests")

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": [{"type": "text", "text": item}]})
            continue
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _content_text(item.get("content", ""))
        if text:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if not messages:
        raise OpenAIResponsesWebSearchError("input must contain at least one message")
    return messages


def _tool_choice(body: dict[str, Any]) -> Any:
    choice = body.get("tool_choice")
    if choice in (None, "auto"):
        return None
    if choice in {"required", "web_search"}:
        return {"type": "tool", "name": "web_search"}
    if isinstance(choice, dict):
        if choice.get("type") in OPENAI_WEB_SEARCH_TOOL_TYPES:
            return {"type": "tool", "name": "web_search"}
        function = choice.get("function")
        if isinstance(function, dict) and function.get("name") == "web_search":
            return {"type": "tool", "name": "web_search"}
    raise OpenAIResponsesWebSearchError("Only auto or web_search tool_choice is supported")


def build_message_request(body: dict[str, Any]) -> MessageRequest:
    options = extract_web_search_options(body)
    max_tokens = int(body.get("max_output_tokens") or body.get("max_tokens") or 4096)
    if max_tokens < 1:
        raise OpenAIResponsesWebSearchError("max_output_tokens must be greater than 0")

    web_search_tool: dict[str, Any] = {
        "type": "web_search_20250305",
        "name": "web_search",
    }
    if options.allowed_domains:
        web_search_tool["allowed_domains"] = options.allowed_domains
    if options.blocked_domains:
        web_search_tool["blocked_domains"] = options.blocked_domains
    if options.user_location:
        web_search_tool["user_location"] = options.user_location.model_dump(
            exclude_none=True
        )

    return MessageRequest(
        model=str(body.get("model") or ""),
        messages=_convert_input_to_messages(body.get("input")),
        max_tokens=max_tokens,
        system=body.get("instructions"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stream=False,
        tools=[web_search_tool],
        tool_choice=_tool_choice(body),
    )
