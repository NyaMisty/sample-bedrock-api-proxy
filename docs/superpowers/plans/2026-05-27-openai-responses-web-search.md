# OpenAI Responses Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add proxy-managed `web_search` compatibility for `/openai/v1/responses`, including non-streaming and streaming Responses API output.

**Architecture:** Add a focused adapter at `app/api/openai_passthrough/web_search.py` that detects OpenAI Responses web search requests, converts them to internal Anthropic `MessageRequest`, runs the existing `WebSearchService`, and maps results back to OpenAI Responses JSON/SSE. Keep `router.py` thin: detect local web search before passthrough, call the adapter, and reuse existing usage recording.

**Tech Stack:** FastAPI, httpx, Pydantic v2, pytest, respx, existing `WebSearchService`, existing Anthropic schemas.

---

## File Structure

- Create: `app/api/openai_passthrough/web_search.py`
  - Owns detection, request conversion, response conversion, stream event formatting, and local execution wrapper for Responses web search.
- Modify: `app/api/openai_passthrough/router.py`
  - Adds local web-search branch inside `responses_create()` only.
- Modify: `tests/integration/test_openai_passthrough/conftest.py`
  - Adds fixtures for mock `BedrockService` and mock `WebSearchService`.
- Modify: `tests/integration/test_openai_passthrough/test_responses.py`
  - Adds route-level tests that prove Responses web search does not passthrough and streaming emits Responses SSE.
- Create: `tests/unit/test_openai_responses_web_search.py`
  - Adds focused adapter tests for detection, conversion, response mapping, and streaming event formatting.
- Modify: `README.md` and `README_ZH.md`
  - Documents Responses-only OpenAI web search compatibility and constraints.

---

## Task 1: Adapter Detection And Validation

**Files:**
- Create: `tests/unit/test_openai_responses_web_search.py`
- Create: `app/api/openai_passthrough/web_search.py`

- [ ] **Step 1: Write failing unit tests for detection and unsupported options**

Add this file:

```python
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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import errors for `app.api.openai_passthrough.web_search`.

- [ ] **Step 3: Implement detection and option extraction**

Create `app/api/openai_passthrough/web_search.py` with this initial content:

```python
"""OpenAI Responses API web search compatibility.

This module adapts OpenAI Responses requests that use the hosted web_search
tool to the proxy's existing Anthropic Messages web search implementation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

from app.core.config import settings
from app.schemas.anthropic import MessageRequest, MessageResponse
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

        filters = tool.get("filters") or {}
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
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: PASS for the five tests added in this task.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py
git commit -m "feat: detect openai responses web search requests"
```

---

## Task 2: Request Conversion To Anthropic MessageRequest

**Files:**
- Modify: `tests/unit/test_openai_responses_web_search.py`
- Modify: `app/api/openai_passthrough/web_search.py`

- [ ] **Step 1: Add failing conversion tests**

Append these tests to `tests/unit/test_openai_responses_web_search.py`:

```python
from app.api.openai_passthrough.web_search import build_message_request


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
    assert req.messages[0].content[0].text == "What changed in Python 3.13?"
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
    assert req.messages[0].content[0].text == "Find current news"
    assert req.messages[1].content[0].text == "What topic?"
    assert req.messages[2].content[0].text == "AI infrastructure"
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
```

Also add this import near the top if the file does not already have it:

```python
from app.schemas.anthropic import MessageRequest
```

- [ ] **Step 2: Run tests to verify conversion tests fail**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: FAIL with `ImportError` for `build_message_request`.

- [ ] **Step 3: Implement request conversion**

Append these helpers to `app/api/openai_passthrough/web_search.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify conversion passes**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py
git commit -m "feat: convert responses web search requests"
```

---

## Task 3: Non-Streaming Response Mapping

**Files:**
- Modify: `tests/unit/test_openai_responses_web_search.py`
- Modify: `app/api/openai_passthrough/web_search.py`

- [ ] **Step 1: Add failing response mapping tests**

Append these imports and tests:

```python
from app.api.openai_passthrough.web_search import build_response_json
from app.schemas.anthropic import MessageResponse, Usage


def test_build_response_json_maps_text_annotations_and_usage():
    msg = MessageResponse(
        id="msg-local",
        type="message",
        role="assistant",
        model="m",
        stop_reason="end_turn",
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtoolu_123",
                "name": "web_search",
                "input": {"query": "Python 3.13"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_123",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://docs.python.org/3/whatsnew/3.13.html",
                        "title": "What\u2019s New In Python 3.13",
                        "encrypted_content": "eA==",
                    }
                ],
            },
            {
                "type": "text",
                "text": "Python 3.13 added a new interactive interpreter.",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://docs.python.org/3/whatsnew/3.13.html",
                        "title": "What\u2019s New In Python 3.13",
                        "cited_text": "new interactive interpreter",
                    }
                ],
            },
        ],
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            server_tool_use={"web_search_requests": 1},
        ),
    )

    data = build_response_json(msg, original_model="m")

    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["model"] == "m"
    assert data["output"][0]["type"] == "web_search_call"
    assert data["output"][0]["status"] == "completed"
    message = data["output"][1]
    assert message["type"] == "message"
    assert message["content"][0]["type"] == "output_text"
    assert data["output_text"] == "Python 3.13 added a new interactive interpreter."
    ann = message["content"][0]["annotations"][0]
    assert ann["type"] == "url_citation"
    assert ann["url"] == "https://docs.python.org/3/whatsnew/3.13.html"
    assert 0 <= ann["start_index"] <= ann["end_index"] <= len(data["output_text"])
    assert data["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert data["metadata"]["web_search_requests"] == 1
```

- [ ] **Step 2: Run tests to verify response mapping fails**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py::test_build_response_json_maps_text_annotations_and_usage -q
```

Expected: FAIL with `ImportError` for `build_response_json`.

- [ ] **Step 3: Implement non-streaming response mapping**

Append this code to `app/api/openai_passthrough/web_search.py`:

```python
def _block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return {}


def _extract_search_count(response: MessageResponse) -> int:
    server_tool_use = response.usage.server_tool_use if response.usage else None
    if not isinstance(server_tool_use, dict):
        return 0
    return int(server_tool_use.get("web_search_requests", 0) or 0)


def _text_and_annotations(content: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    output_parts: list[str] = []
    annotations: list[dict[str, Any]] = []

    for block in content:
        block_dict = _block_dict(block)
        if block_dict.get("type") != "text":
            continue
        text = str(block_dict.get("text") or "")
        if not text:
            continue
        start = sum(len(part) for part in output_parts)
        if output_parts:
            output_parts.append("\n")
            start += 1
        output_parts.append(text)
        end = start + len(text)
        for citation in block_dict.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            if citation.get("type") != "web_search_result_location":
                continue
            cited_text = str(citation.get("cited_text") or "")
            cited_start = text.find(cited_text) if cited_text else -1
            if cited_start >= 0:
                ann_start = start + cited_start
                ann_end = ann_start + len(cited_text)
            else:
                ann_start = start
                ann_end = end
            annotations.append(
                {
                    "type": "url_citation",
                    "url": citation.get("url", ""),
                    "title": citation.get("title", ""),
                    "start_index": ann_start,
                    "end_index": ann_end,
                }
            )

    return "".join(output_parts), annotations


def build_response_json(
    response: MessageResponse,
    *,
    original_model: str,
    response_id: str | None = None,
) -> dict[str, Any]:
    response_id = response_id or f"resp_{uuid4().hex[:24]}"
    output_text, annotations = _text_and_annotations(response.content)
    search_count = _extract_search_count(response)

    output: list[dict[str, Any]] = []
    if search_count > 0:
        output.append(
            {
                "id": f"ws_{uuid4().hex[:24]}",
                "type": "web_search_call",
                "status": "completed",
            }
        )

    output.append(
        {
            "id": response.id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": output_text,
                    "annotations": annotations,
                }
            ],
        }
    )

    input_tokens = int(response.usage.input_tokens if response.usage else 0)
    output_tokens = int(response.usage.output_tokens if response.usage else 0)
    data: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": original_model,
        "output": output,
        "output_text": output_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if search_count > 0:
        data["metadata"] = {"web_search_requests": search_count}
    return data
```

- [ ] **Step 4: Run tests to verify response mapping passes**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py
git commit -m "feat: map responses web search output"
```

---

## Task 4: Streaming Event Mapping

**Files:**
- Modify: `tests/unit/test_openai_responses_web_search.py`
- Modify: `app/api/openai_passthrough/web_search.py`

- [ ] **Step 1: Add failing streaming formatter test**

Append this test:

```python
import json

from app.api.openai_passthrough.web_search import stream_response_events


@pytest.mark.asyncio
async def test_stream_response_events_emits_responses_sse():
    msg = MessageResponse(
        id="msg-local",
        type="message",
        role="assistant",
        model="m",
        stop_reason="end_turn",
        content=[{"type": "text", "text": "hello"}],
        usage=Usage(input_tokens=2, output_tokens=1, server_tool_use={"web_search_requests": 1}),
    )

    chunks = [chunk async for chunk in stream_response_events(msg, original_model="m")]
    out = b"".join(chunks).decode("utf-8")

    assert "event: response.created\n" in out
    assert "event: response.output_item.added\n" in out
    assert "event: response.output_text.delta\n" in out
    assert "event: response.completed\n" in out
    completed_lines = [
        line for line in out.splitlines() if line.startswith("data: ") and "response.completed" in line
    ]
    assert completed_lines
    payload = json.loads(completed_lines[-1][len("data: "):])
    assert payload["response"]["usage"]["input_tokens"] == 2
```

- [ ] **Step 2: Run the streaming formatter test to verify it fails**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py::test_stream_response_events_emits_responses_sse -q
```

Expected: FAIL with `ImportError` for `stream_response_events`.

- [ ] **Step 3: Implement SSE event formatting**

Append this code:

```python
def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


async def stream_response_events(
    response: MessageResponse,
    *,
    original_model: str,
    response_id: str | None = None,
) -> AsyncIterator[bytes]:
    data = build_response_json(
        response,
        original_model=original_model,
        response_id=response_id,
    )
    response_stub = {
        key: data[key]
        for key in ("id", "object", "created_at", "status", "model")
        if key in data
    }

    yield _sse("response.created", {"type": "response.created", "response": response_stub})

    for index, item in enumerate(data["output"]):
        yield _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": item,
            },
        )
        if item.get("type") == "message":
            content = item.get("content") or []
            if content:
                yield _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "part": content[0],
                    },
                )
                text = content[0].get("text", "")
                if text:
                    yield _sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": item["id"],
                            "output_index": index,
                            "content_index": 0,
                            "delta": text,
                        },
                    )
                yield _sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "part": content[0],
                    },
                )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": index,
                "item": item,
            },
        )

    completed = dict(data)
    completed["status"] = "completed"
    yield _sse("response.completed", {"type": "response.completed", "response": completed})
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py
git commit -m "feat: stream responses web search events"
```

---

## Task 5: Router Integration For Non-Streaming Responses

**Files:**
- Modify: `tests/integration/test_openai_passthrough/conftest.py`
- Modify: `tests/integration/test_openai_passthrough/test_responses.py`
- Modify: `app/api/openai_passthrough/router.py`
- Modify: `app/api/openai_passthrough/web_search.py`

- [ ] **Step 1: Add mock fixtures for local web search execution**

Modify the import at the top of `tests/integration/test_openai_passthrough/conftest.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
```

Then add these fixtures after `mock_usage_tracker()`:

```python
@pytest.fixture
def mock_web_search_service():
    service = MagicMock()
    service.handle_request = AsyncMock()
    with patch(
        "app.api.openai_passthrough.router.get_web_search_service",
        return_value=service,
        create=True,
    ):
        yield service


@pytest.fixture
def mock_bedrock_service():
    service = MagicMock()
    with patch(
        "app.api.openai_passthrough.router.BedrockService",
        return_value=service,
        create=True,
    ):
        yield service
```

Update the `client` fixture signature to include both fixtures so patches are active when `app.main` reloads:

```python
def client(
    mock_settings,
    mock_api_key_manager,
    mock_model_mapping_manager,
    mock_usage_tracker,
    mock_web_search_service,
    mock_bedrock_service,
):
```

- [ ] **Step 2: Add failing integration test for non-streaming local path**

Append to `tests/integration/test_openai_passthrough/test_responses.py`:

```python
from app.schemas.anthropic import MessageResponse, Usage


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
```

- [ ] **Step 3: Run the new integration test to verify it fails**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_responses.py::test_non_streaming_responses_web_search_uses_local_adapter_not_upstream -q
```

Expected: FAIL because `router.py` still forwards web search requests upstream.

- [ ] **Step 4: Add local execution helper**

Append to `app/api/openai_passthrough/web_search.py`:

```python
async def handle_non_streaming_web_search(
    body: dict[str, Any],
    *,
    web_search_service: Any,
    bedrock_service: Any,
    request_id: str,
    service_tier: str,
) -> dict[str, Any]:
    if not settings.enable_web_search:
        raise OpenAIResponsesWebSearchError("Web search is disabled")
    message_request = build_message_request(body)
    response = await web_search_service.handle_request(
        request=message_request,
        bedrock_service=bedrock_service,
        request_id=request_id,
        service_tier=service_tier,
        anthropic_beta=None,
    )
    return build_response_json(response, original_model=body.get("model", ""))
```

- [ ] **Step 5: Wire router local branch**

Modify imports in `app/api/openai_passthrough/router.py`:

```python
from app.api.openai_passthrough.web_search import (
    OpenAIResponsesWebSearchError,
    handle_non_streaming_web_search,
    is_responses_web_search_request,
)
from app.services.bedrock_service import BedrockService
from app.services.web_search_service import get_web_search_service
```

Inside `responses_create()`, after model resolution and `extra = _passthrough_extra_headers(request)`, insert:

```python
    if is_responses_web_search_request(body):
        request_id = f"resp-{uuid4().hex}"
        service_tier = api_key_info.get("service_tier", "default")
        try:
            data = await handle_non_streaming_web_search(
                body,
                web_search_service=get_web_search_service(),
                bedrock_service=BedrockService(),
                request_id=request_id,
                service_tier=service_tier,
            )
        except OpenAIResponsesWebSearchError as exc:
            return JSONResponse(exc.to_error_body(), status_code=exc.status_code)
        if isinstance(data.get("usage"), dict):
            _record_usage(api_key_info, data["usage"], body["model"], "responses")
        return JSONResponse(data, status_code=200)
```

This is intentionally before the existing `if body.get("stream"):` block.

- [ ] **Step 6: Run integration test to verify it passes**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_responses.py::test_non_streaming_responses_web_search_uses_local_adapter_not_upstream -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add app/api/openai_passthrough/router.py app/api/openai_passthrough/web_search.py tests/integration/test_openai_passthrough/conftest.py tests/integration/test_openai_passthrough/test_responses.py
git commit -m "feat: route responses web search locally"
```

---

## Task 6: Router Integration For Streaming And Error Responses

**Files:**
- Modify: `tests/integration/test_openai_passthrough/test_responses.py`
- Modify: `app/api/openai_passthrough/router.py`
- Modify: `app/api/openai_passthrough/web_search.py`

- [ ] **Step 1: Add failing streaming and validation integration tests**

Append:

```python
def test_streaming_responses_web_search_emits_local_responses_sse(
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
        content=[{"type": "text", "text": "streamed answer"}],
        usage=Usage(
            input_tokens=4,
            output_tokens=3,
            server_tool_use={"web_search_requests": 1},
        ),
    )
    route = respx_mock.post("/responses").mock(
        return_value=httpx.Response(500, json={"error": {"message": "should not call"}})
    )

    with client.stream(
        "POST",
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "m",
            "input": "Search the web",
            "stream": True,
            "tools": [{"type": "web_search"}],
        },
    ) as r:
        out = b"".join(r.iter_bytes()).decode("utf-8")

    assert "event: response.created" in out
    assert "event: response.output_text.delta" in out
    assert "streamed answer" in out
    assert "event: response.completed" in out
    assert not route.called
    kw = mock_usage_tracker.record_usage.call_args.kwargs
    assert kw["api_surface"] == "responses"
    assert kw["input_tokens"] == 4
    assert kw["output_tokens"] == 3


def test_responses_web_search_rejects_external_web_access_false(
    client,
    respx_mock,
    mock_usage_tracker,
):
    route = respx_mock.post("/responses").mock(
        return_value=httpx.Response(500, json={"error": {"message": "should not call"}})
    )

    r = client.post(
        "/openai/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "m",
            "input": "Search the web",
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
    )

    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "external_web_access" in r.json()["error"]["message"]
    assert not route.called
    assert not mock_usage_tracker.record_usage.called
```

- [ ] **Step 2: Run tests to verify streaming path fails**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_responses.py::test_streaming_responses_web_search_emits_local_responses_sse tests/integration/test_openai_passthrough/test_responses.py::test_responses_web_search_rejects_external_web_access_false -q
```

Expected: streaming test fails because router returns JSON for local web search even when `stream` is true.

- [ ] **Step 3: Add router imports for streaming mapping**

Modify imports in `app/api/openai_passthrough/router.py` to include the adapter helpers used by both local paths:

```python
from typing import Any, AsyncIterator, cast
```

Add these imports with the other app imports:

```python
from app.api.openai_passthrough.web_search import (
    OpenAIResponsesWebSearchError,
    build_message_request,
    build_response_json,
    handle_non_streaming_web_search,
    is_responses_web_search_request,
    stream_response_events,
)
```

Also add:

```python
from app.services.bedrock_service import BedrockService
from app.services.web_search_service import get_web_search_service
```

- [ ] **Step 4: Split router local branch into streaming and non-streaming paths**

Modify the local branch in `responses_create()` to:

```python
    if is_responses_web_search_request(body):
        request_id = f"resp-{uuid4().hex}"
        service_tier = api_key_info.get("service_tier", "default")
        web_search_service = get_web_search_service()
        bedrock_service = BedrockService()

        if body.get("stream"):
            try:
                message_request = build_message_request(body)
            except OpenAIResponsesWebSearchError as exc:
                return JSONResponse(exc.to_error_body(), status_code=exc.status_code)

            async def on_local_stream_complete() -> AsyncIterator[bytes]:
                response = await web_search_service.handle_request(
                    request=message_request,
                    bedrock_service=bedrock_service,
                    request_id=request_id,
                    service_tier=service_tier,
                    anthropic_beta=None,
                )
                data = build_response_json(
                    response,
                    original_model=body.get("model", ""),
                    response_id=request_id,
                )
                if isinstance(data.get("usage"), dict):
                    _record_usage(api_key_info, data["usage"], body["model"], "responses")
                async for chunk in stream_response_events(
                    response,
                    original_model=body.get("model", ""),
                    response_id=request_id,
                ):
                    yield chunk

            return StreamingResponse(
                on_local_stream_complete(),
                media_type="text/event-stream",
            )

        try:
            data = await handle_non_streaming_web_search(
                body,
                web_search_service=web_search_service,
                bedrock_service=bedrock_service,
                request_id=request_id,
                service_tier=service_tier,
            )
        except OpenAIResponsesWebSearchError as exc:
            return JSONResponse(exc.to_error_body(), status_code=exc.status_code)
        if isinstance(data.get("usage"), dict):
            _record_usage(api_key_info, data["usage"], body["model"], "responses")
        return JSONResponse(data, status_code=200)
```

- [ ] **Step 5: Run streaming and validation tests**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_responses.py::test_streaming_responses_web_search_emits_local_responses_sse tests/integration/test_openai_passthrough/test_responses.py::test_responses_web_search_rejects_external_web_access_false -q
```

Expected: PASS.

- [ ] **Step 6: Run all OpenAI passthrough integration tests**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add app/api/openai_passthrough/router.py app/api/openai_passthrough/web_search.py tests/integration/test_openai_passthrough/test_responses.py
git commit -m "feat: support streaming responses web search"
```

---

## Task 7: Preserve Chat Completions Passthrough And Add Documentation

**Files:**
- Modify: `tests/integration/test_openai_passthrough/test_chat_completions.py`
- Modify: `README.md`
- Modify: `README_ZH.md`

- [ ] **Step 1: Add regression test proving Chat Completions is unchanged**

Append to `tests/integration/test_openai_passthrough/test_chat_completions.py`:

```python
def test_chat_completions_web_search_shape_still_passthrough(client, respx_mock):
    route = respx_mock.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    r = client.post(
        "/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search"}],
        },
    )

    assert r.status_code == 200
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"] == [{"type": "web_search"}]
```

- [ ] **Step 2: Run regression test to verify it passes**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough/test_chat_completions.py::test_chat_completions_web_search_shape_still_passthrough -q
```

Expected: PASS.

- [ ] **Step 3: Update English README**

In `README.md`, add this bullet under the OpenAI Passthrough feature description:

```markdown
- **Responses API Web Search Compatibility**: `POST /openai/v1/responses` can execute `tools: [{"type": "web_search"}]` proxy-side using the existing Tavily/Brave web search providers. This applies only to Responses API; Chat Completions remains passthrough. Live search is supported; `external_web_access: false` and `return_token_budget` are rejected by the proxy-managed path.
```

- [ ] **Step 4: Update Chinese README**

In `README_ZH.md`, add this bullet under the OpenAI Passthrough feature description:

```markdown
- **Responses API Web Search 兼容**：`POST /openai/v1/responses` 可以通过现有 Tavily/Brave 搜索提供商在代理侧执行 `tools: [{"type": "web_search"}]`。该能力仅适用于 Responses API；Chat Completions 继续保持透传。当前支持实时搜索；`external_web_access: false` 和 `return_token_budget` 会被代理侧路径拒绝。
```

- [ ] **Step 5: Commit Task 7**

Run:

```bash
git add README.md README_ZH.md tests/integration/test_openai_passthrough/test_chat_completions.py
git commit -m "docs: document responses web search compatibility"
```

---

## Task 8: Full Verification

**Files:**
- No source files should be edited in this task unless a verification failure reveals a defect.

- [ ] **Step 1: Run adapter unit tests**

Run:

```bash
uv run pytest tests/unit/test_openai_responses_web_search.py -q
```

Expected: PASS.

- [ ] **Step 2: Run OpenAI passthrough integration tests**

Run:

```bash
uv run pytest tests/integration/test_openai_passthrough -q
```

Expected: PASS.

- [ ] **Step 3: Run formatter/linter on touched Python files**

Run:

```bash
uv run black app/api/openai_passthrough/router.py app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py tests/integration/test_openai_passthrough/test_responses.py tests/integration/test_openai_passthrough/test_chat_completions.py tests/integration/test_openai_passthrough/conftest.py
uv run ruff check app/api/openai_passthrough/router.py app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py tests/integration/test_openai_passthrough/test_responses.py tests/integration/test_openai_passthrough/test_chat_completions.py tests/integration/test_openai_passthrough/conftest.py
```

Expected: black reports either unchanged or reformatted files; ruff exits 0.

- [ ] **Step 4: Run focused full suite**

Run:

```bash
uv run pytest tests/unit/test_openai_passthrough tests/unit/test_openai_responses_web_search.py tests/integration/test_openai_passthrough -q
```

Expected: PASS.

- [ ] **Step 5: Commit verification formatting changes when present**

Run:

```bash
git status --short
git add app/api/openai_passthrough/router.py app/api/openai_passthrough/web_search.py tests/unit/test_openai_responses_web_search.py tests/integration/test_openai_passthrough/test_responses.py tests/integration/test_openai_passthrough/test_chat_completions.py tests/integration/test_openai_passthrough/conftest.py README.md README_ZH.md
git commit -m "test: verify responses web search compatibility"
```

Expected: Commit succeeds when black changed files. When black changed nothing, `git commit` prints `nothing to commit`; continue with the existing task commits.

---

## Self-Review

- Spec coverage: Responses-only scope is covered by Tasks 5 and 6; Chat Completions non-scope regression is covered by Task 7; unsupported options are covered by Tasks 1 and 6; streaming is covered by Tasks 4 and 6; usage tracking is covered by Tasks 5 and 6; docs are covered by Task 7.
- Type consistency: The plan consistently uses `OpenAIResponsesWebSearchError`, `is_responses_web_search_request`, `extract_web_search_options`, `build_message_request`, `build_response_json`, and `stream_response_events` from `app/api/openai_passthrough/web_search.py`.
- Test boundary: Tests mock the local service boundary and never perform live search.
