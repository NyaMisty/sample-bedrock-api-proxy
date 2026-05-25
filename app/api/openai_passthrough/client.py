"""Async httpx client to bedrock-mantle, lazily constructed and reused.

Headers are NOT set on the client itself; they're added per-request in the
router so we can include the proxy's Bedrock API key in Authorization.
"""
from __future__ import annotations

import httpx

from app.core.config import settings

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=httpx.Timeout(settings.bedrock_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _client


def reset_client_for_testing() -> None:
    """Reset the singleton — only call this from test fixtures."""
    global _client
    if _client is not None:
        # AsyncClient.aclose() is async; tests will close the loop after, so we
        # null it here and let the GC clean up the underlying transport.
        _client = None


def upstream_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build the Authorization + standard headers for an upstream call."""
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
        "User-Agent": "bedrock-api-proxy/openai-passthrough",
    }
    if extra:
        headers.update(extra)
    return headers
