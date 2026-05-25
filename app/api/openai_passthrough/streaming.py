"""SSE passthrough with usage tee.

The async generator yields raw response bytes line-by-line so the FastAPI
StreamingResponse forwards them unchanged. After upstream stream ends, it
calls the supplied on_complete callback with the captured usage dict so the
caller can record usage to DynamoDB.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from app.api.openai_passthrough.client import get_client, upstream_headers, upstream_url
from app.api.openai_passthrough.usage_extractor import try_extract_usage_from_sse

logger = logging.getLogger(__name__)


async def stream_passthrough(
    method: str,
    path: str,
    body: dict[str, Any] | None,
    api_surface: str,
    on_complete: Callable[[dict[str, Any]], Awaitable[None] | None],
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """Stream upstream response bytes line-by-line; capture usage; trigger callback."""
    usage: dict[str, Any] = {}

    client = get_client()
    headers = upstream_headers(extra_headers)

    try:
        async with client.stream(method, upstream_url(path), json=body, headers=headers) as resp:
            async for raw_line in resp.aiter_lines():
                # Upstream gives us SSE lines without trailing newlines; restore the
                # framing byte so the SSE body is well-formed for the downstream client.
                yield (raw_line + "\n").encode("utf-8")
                try_extract_usage_from_sse(raw_line, usage, api_surface)
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        # Upstream connection/timeout failure during streaming. OpenAI SDK clients
        # expect a clean SSE termination, not an abruptly closed stream.
        logger.error("[OPENAI-PASSTHROUGH] upstream stream connection error: %s", exc)
        err = {
            "error": {
                "message": f"upstream connection failed: {type(exc).__name__}",
                "type": "upstream_error",
            }
        }
        yield ("data: " + json.dumps(err) + "\n\n").encode("utf-8")
        yield b"data: [DONE]\n\n"
        return
    except Exception as exc:
        # Unexpected error — re-raise so FastAPI can convert to 500.
        logger.error("[OPENAI-PASSTHROUGH] upstream stream error: %s", exc)
        raise

    if usage:
        result = on_complete(usage)
        # Support both sync and async callbacks
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]
