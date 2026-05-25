"""Async httpx client to bedrock-mantle, lazily constructed and reused.

Headers are NOT set on the client itself; they're added per-request in the
router so we can include the proxy's Bedrock API key in Authorization.

URL building note: we deliberately do NOT set ``base_url`` on the AsyncClient.
httpx follows RFC 3986 path-merging, which means a request path starting with
``/`` REPLACES the path component of the base_url. With
``OPENAI_BASE_URL=https://bedrock-mantle.us-west-2.api.aws/v1``, calling
``client.post("/chat/completions")`` would produce
``https://bedrock-mantle.us-west-2.api.aws/chat/completions`` (the ``/v1`` is
dropped). To avoid this footgun we build full URLs explicitly via
``upstream_url(path)``.
"""
from __future__ import annotations

import httpx

from app.core.config import settings

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
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


def upstream_url(path: str) -> str:
    """Build a full upstream URL by appending ``path`` to ``OPENAI_BASE_URL``.

    Avoids httpx's RFC 3986 path-replacement behaviour by always producing a
    fully-qualified URL.

    Examples:
        OPENAI_BASE_URL=https://bedrock-mantle.us-west-2.api.aws/v1
        upstream_url("/chat/completions")  -> https://bedrock-mantle.us-west-2.api.aws/v1/chat/completions
        upstream_url("models")             -> https://bedrock-mantle.us-west-2.api.aws/v1/models
    """
    base = settings.openai_base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


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
