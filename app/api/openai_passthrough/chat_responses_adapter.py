"""Adapt Chat Completions requests to upstream Responses API calls.

The public endpoint remains Chat Completions-compatible. These helpers translate
only the proxy-to-upstream leg and then map Responses output back to Chat
Completions shape for existing OpenAI SDK clients.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

_CHAT_TO_RESPONSES_COPY_FIELDS = {
    "temperature",
    "top_p",
    "stream",
    "metadata",
    "reasoning",
    "parallel_tool_calls",
    "service_tier",
    "user",
}


def chat_request_to_response_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions body into a Responses API body."""
    result: dict[str, Any] = {"model": body.get("model", "")}

    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        input_items.extend(_convert_chat_message(message, instructions))

    if instructions:
        result["instructions"] = "\n".join(part for part in instructions if part)
    result["input"] = input_items

    max_output_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if max_output_tokens is not None:
        result["max_output_tokens"] = max_output_tokens

    for field in _CHAT_TO_RESPONSES_COPY_FIELDS:
        if field in body:
            result[field] = body[field]

    if "tools" in body:
        result["tools"] = [_convert_tool(tool) for tool in body.get("tools") or []]
    if "tool_choice" in body:
        result["tool_choice"] = _convert_tool_choice(body["tool_choice"])
    if "response_format" in body:
        result["response_format"] = body["response_format"]
    if "stop" in body:
        result["stop"] = body["stop"]

    return result


def response_to_chat_completion(
    response: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Convert a non-streaming Responses API body to Chat Completions shape."""
    content = _extract_response_text(response)
    tool_calls = _extract_response_tool_calls(response)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content or not tool_calls else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    result: dict[str, Any] = {
        "id": response.get("id") or f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(response.get("created") or time.time()),
        "model": response.get("model") or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(response, tool_calls),
            }
        ],
    }
    usage = response.get("usage")
    if isinstance(usage, dict):
        result["usage"] = responses_usage_to_chat_usage(usage)
    return result


def responses_usage_to_chat_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses usage fields to Chat Completions usage fields."""
    prompt_tokens = int(raw.get("input_tokens", 0) or 0)
    completion_tokens = int(raw.get("output_tokens", 0) or 0)
    total_tokens = int(raw.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    input_details = raw.get("input_tokens_details")
    if isinstance(input_details, dict):
        prompt_details: dict[str, Any] = {}
        if "cached_tokens" in input_details:
            prompt_details["cached_tokens"] = int(
                input_details.get("cached_tokens") or 0
            )
        if prompt_details:
            result["prompt_tokens_details"] = prompt_details

    output_details = raw.get("output_tokens_details")
    if isinstance(output_details, dict):
        completion_details: dict[str, Any] = {}
        if "reasoning_tokens" in output_details:
            completion_details["reasoning_tokens"] = int(
                output_details.get("reasoning_tokens") or 0
            )
        if completion_details:
            result["completion_tokens_details"] = completion_details

    return result


async def stream_responses_as_chat_completions(
    resp: httpx.Response,
    *,
    model: str,
    on_complete: Callable[[dict[str, Any]], Awaitable[None] | None],
) -> AsyncIterator[bytes]:
    """Convert an upstream Responses SSE stream to Chat Completions SSE."""
    response_id = f"chatcmpl-{int(time.time())}"
    response_model = model
    created = int(time.time())
    usage: dict[str, Any] = {}
    done_sent = False

    try:
        async for raw_line in resp.aiter_lines():
            payload = _load_sse_data(raw_line)
            if payload is None:
                continue

            event_type = payload.get("type")
            if event_type == "response.created":
                response_obj = payload.get("response") or {}
                if isinstance(response_obj, dict):
                    response_id = response_obj.get("id") or response_id
                    response_model = response_obj.get("model") or response_model
                    created = int(response_obj.get("created") or created)
                yield _chat_sse(
                    _chat_chunk(
                        response_id,
                        response_model,
                        created,
                        [{"index": 0, "delta": {"role": "assistant"}}],
                    )
                )

            elif event_type == "response.output_text.delta":
                delta = payload.get("delta")
                if isinstance(delta, str) and delta:
                    yield _chat_sse(
                        _chat_chunk(
                            response_id,
                            response_model,
                            created,
                            [{"index": 0, "delta": {"content": delta}}],
                        )
                    )

            elif event_type == "response.completed":
                response_obj = payload.get("response") or {}
                tool_calls = (
                    _extract_response_tool_calls(response_obj)
                    if isinstance(response_obj, dict)
                    else []
                )
                yield _chat_sse(
                    _chat_chunk(
                        response_id,
                        response_model,
                        created,
                        [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": _finish_reason(
                                    response_obj if isinstance(response_obj, dict) else {},
                                    tool_calls,
                                ),
                            }
                        ],
                    )
                )
                if isinstance(response_obj, dict):
                    raw_usage = response_obj.get("usage")
                    if isinstance(raw_usage, dict):
                        usage.clear()
                        usage.update(raw_usage)
                        usage_chunk = _chat_chunk(
                            response_id,
                            response_model,
                            created,
                            [],
                        )
                        usage_chunk["usage"] = responses_usage_to_chat_usage(raw_usage)
                        yield _chat_sse(usage_chunk)
                yield b"data: [DONE]\n\n"
                done_sent = True

        if not done_sent:
            yield b"data: [DONE]\n\n"
    finally:
        await resp.aclose()

    if usage:
        result = on_complete(usage)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]


def _convert_chat_message(
    message: dict[str, Any],
    instructions: list[str],
) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content", "")
    if role in {"system", "developer"}:
        text = _content_to_text(content)
        if text:
            instructions.append(text)
        return []

    if role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": _content_to_text(content),
            }
        ]

    items: list[dict[str, Any]] = []
    if content not in ("", None):
        items.append({"role": role or "user", "content": content})

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        items.append(
            {
                "type": "function_call",
                "call_id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", ""),
            }
        )
    return items


def _convert_tool(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool
    if tool.get("type") != "function":
        return tool
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool
    converted = {
        "type": "function",
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": _normalize_null_required_fields(function.get("parameters", {})),
    }
    return {key: value for key, value in converted.items() if value not in (None, "")}


def _normalize_null_required_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                []
                if key == "required" and item is None
                else _normalize_null_required_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_null_required_fields(item) for item in value]
    return value


def _convert_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "function":
        return tool_choice
    function = tool_choice.get("function")
    if isinstance(function, dict):
        return {"type": "function", "name": function.get("name", "")}
    return tool_choice


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _extract_response_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for entry in item.get("content") or []:
                if isinstance(entry, dict) and entry.get("type") in {
                    "output_text",
                    "text",
                }:
                    texts.append(str(entry.get("text", "")))
        elif item.get("type") == "output_text":
            texts.append(str(item.get("text", "")))
    return "".join(texts)


def _extract_response_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = item.get("call_id") or item.get("id") or f"call_{len(tool_calls)}"
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "",
                },
            }
        )
    return tool_calls


def _finish_reason(response: dict[str, Any], tool_calls: list[dict[str, Any]]) -> str:
    if tool_calls:
        return "tool_calls"
    if response.get("status") == "incomplete":
        return "length"
    return "stop"


def _load_sse_data(raw_line: str) -> dict[str, Any] | None:
    line = raw_line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _chat_chunk(
    response_id: str,
    model: str,
    created: int,
    choices: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }


def _chat_sse(payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"data: {data}\n\n".encode()
