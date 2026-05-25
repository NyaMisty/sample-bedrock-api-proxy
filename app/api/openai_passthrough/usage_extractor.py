"""Usage extraction and normalization for OpenAI-format responses.

normalize_usage() converts an OpenAI Chat Completions or Responses API usage
dict into the Anthropic-shaped dict that UsageTracker.record_usage expects,
plus a separate reasoning_tokens field.

try_extract_usage_from_sse() peeks at SSE lines during streaming and stashes
the usage dict (raw OpenAI shape) the first time it encounters one. The caller
later passes that dict through normalize_usage().
"""
from __future__ import annotations

import json
from typing import Any, Dict


def normalize_usage(raw: Dict[str, Any], api_surface: str) -> Dict[str, int]:
    """Normalize OpenAI-shaped usage into Anthropic-shaped fields.

    api_surface: "chat_completions" or "responses"
    """
    if api_surface == "chat_completions":
        in_tok = int(raw.get("prompt_tokens", 0) or 0)
        out_tok = int(raw.get("completion_tokens", 0) or 0)
        cached = int((raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        reasoning = int(
            (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
    else:  # responses
        in_tok = int(raw.get("input_tokens", 0) or 0)
        out_tok = int(raw.get("output_tokens", 0) or 0)
        cached = int((raw.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
        reasoning = int(
            (raw.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )

    # Cache-read tokens are billed separately, so subtract them from input_tokens
    # to mirror how the Anthropic flow accounts for cache hits.
    return {
        "input_tokens": max(in_tok - cached, 0),
        "output_tokens": out_tok,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,  # Not exposed by OpenAI-format APIs
        "reasoning_tokens": reasoning,
    }


def try_extract_usage_from_sse(
    raw_line: str, holder: Dict[str, Any], api_surface: str
) -> None:
    """Inspect an SSE line and, if it carries usage info, store it in holder.

    Mutates `holder` in place. Idempotent: subsequent calls overwrite, so the
    last-seen usage event wins (which is what we want — both APIs put usage
    on the terminal event).
    """
    line = raw_line.strip()
    if not line.startswith("data:"):
        return

    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return

    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return

    if api_surface == "chat_completions":
        usage = obj.get("usage")
        if isinstance(usage, dict):
            holder.clear()
            holder.update(usage)
    else:  # responses
        # Usage lives on the `response.completed` event under
        # event.response.usage. Other events occasionally carry partial usage
        # too — accept any usage dict we see.
        if obj.get("type") == "response.completed":
            response_obj = obj.get("response") or {}
            usage = response_obj.get("usage")
            if isinstance(usage, dict):
                holder.clear()
                holder.update(usage)
