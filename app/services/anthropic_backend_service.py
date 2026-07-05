"""
Direct-Anthropic backend service.

Used when ``BACKEND_MODE=anthropic``. Calls the native Anthropic Messages
API (``POST /v1/messages``) via httpx — no boto3, no Bedrock protocol.
The proxy's request/response schemas are already Anthropic-format, so this
is mostly passthrough with light body/header normalization.

Mirrors :class:`OpenAICompatService`'s method surface so
:class:`BedrockService` can dispatch to either backend identically via
short-circuits at the top of ``invoke_model`` / ``invoke_model_stream`` /
``count_tokens``.

Silent degradation (per project decision):
- ``cache_control``: passed through natively (Anthropic supports it).
- ``service_tier``: dropped (no equivalent on the direct API).
- ``count_tokens``: calls Anthropic's ``/v1/messages/count_tokens``; on
  failure falls back to the BedrockService heuristic.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from uuid import uuid4

import httpx

from app.core.config import settings
from app.core.exceptions import BedrockAPIError
from app.schemas.anthropic import MessageRequest, MessageResponse

logger = logging.getLogger(__name__)


# Anthropic API version stamp. The direct API takes this as the
# `anthropic-version` HTTP header (unlike Bedrock InvokeModel, which takes
# `anthropic_version` in the request body).
_ANTHROPIC_VERSION_HEADER = "2023-06-01"


class AnthropicBackendService:
    """Direct-Anthropic backend using httpx.

    Constructed by :class:`BedrockService` when ``settings.backend_mode ==
    'anthropic'``. All model IDs (including ``claude-*``) route here.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        resolved_base_url = (base_url or settings.anthropic_base_url).rstrip("/")
        self._base_url = resolved_base_url
        self._api_key = api_key or settings.anthropic_api_key
        self._client = httpx.Client(timeout=settings.bedrock_timeout)
        self._async_client = httpx.AsyncClient(timeout=settings.bedrock_timeout)
        print(f"[ANTHROPIC-BACKEND] Initialized with base_url={self._base_url}")

    # ------------------------------------------------------------------
    # Body / header normalization
    # ------------------------------------------------------------------
    def _normalize_body(self, request: MessageRequest) -> Dict[str, Any]:
        """Build the native Anthropic request body and strip Bedrock-isms.

        - Reuses :func:`build_native_anthropic_body` with the direct-API
          version stamp.
        - Pops ``anthropic_version`` out of the body (direct API uses the
          ``anthropic-version`` header instead — a body field is a 400).
        - Pops ``anthropic_beta`` out of the body (direct API uses the
          ``anthropic-beta`` header).
        - Drops ``service_tier`` (no direct-API equivalent — silent
          degradation).
        - ``cache_control`` is preserved end-to-end (Anthropic native).
        """
        # Lazy import to avoid circular dependency at module load time.
        from app.services.bedrock_service import build_native_anthropic_body

        body = build_native_anthropic_body(
            request,
            anthropic_version=_ANTHROPIC_VERSION_HEADER,
        )
        # The direct Anthropic API requires `model` in the body (unlike
        # Bedrock InvokeModel, which carries the model in the URL path).
        # build_native_anthropic_body omits it, so add it here. Use the
        # client-supplied model ID verbatim — the direct endpoint resolves it.
        body["model"] = request.model
        # Bedrock toggles streaming via a different API method
        # (InvokeModelWithResponseStream); the direct Anthropic API toggles
        # it via `"stream": true` in the body. Forward the client's flag.
        if getattr(request, "stream", False):
            body["stream"] = True
        # anthropic_version belongs in the header, not the body.
        body.pop("anthropic_version", None)
        # anthropic_beta belongs in the anthropic-beta header; the body
        # field is Bedrock-InvokeModel-specific.
        beta_from_body = body.pop("anthropic_beta", None)
        # service_tier has no direct-API equivalent — drop silently.
        body.pop("service_tier", None)

        # Stash beta for header construction. _resolve_beta merges the
        # body-derived beta (after DynamoDB blocklist/mapping) with any
        # caller-supplied header beta.
        self._current_beta = beta_from_body
        return body

    def _headers(self) -> Dict[str, str]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION_HEADER,
            "content-type": "application/json",
        }
        beta = getattr(self, "_current_beta", None)
        if beta:
            headers["anthropic-beta"] = ",".join(beta)
        return headers

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def invoke_model_sync(
        self, request: MessageRequest, request_id: Optional[str] = None
    ) -> MessageResponse:
        """Synchronously call ``POST /v1/messages`` and return the response.

        The Anthropic response is already in native ``MessageResponse`` shape,
        so no converter is needed — just validate.
        """
        body = self._normalize_body(request)
        url = f"{self._base_url}/v1/messages"
        try:
            resp = self._client.post(url, json=body, headers=self._headers())
        except httpx.HTTPError as e:
            raise BedrockAPIError(
                error_code="AnthropicBackendRequestError",
                error_message=f"HTTP transport error: {e}",
                http_status=502,
                error_type="api_error",
            )
        if resp.status_code >= 400:
            raise self._map_error(resp.status_code, resp.text)
        data = resp.json()
        try:
            return MessageResponse.model_validate(data)
        except Exception as e:
            raise BedrockAPIError(
                error_code="AnthropicBackendResponseError",
                error_message=f"Failed to parse Anthropic response: {e}",
                http_status=502,
                error_type="api_error",
            )

    async def invoke_model(
        self, request: MessageRequest, request_id: Optional[str] = None
    ) -> MessageResponse:
        """Asynchronously call ``POST /v1/messages``.

        Uses the async httpx client directly (no thread-pool bridge needed —
        httpx is natively async).
        """
        body = self._normalize_body(request)
        url = f"{self._base_url}/v1/messages"
        try:
            resp = await self._async_client.post(
                url, json=body, headers=self._headers()
            )
        except httpx.HTTPError as e:
            raise BedrockAPIError(
                error_code="AnthropicBackendRequestError",
                error_message=f"HTTP transport error: {e}",
                http_status=502,
                error_type="api_error",
            )
        if resp.status_code >= 400:
            raise self._map_error(resp.status_code, resp.text)
        data = resp.json()
        try:
            return MessageResponse.model_validate(data)
        except Exception as e:
            raise BedrockAPIError(
                error_code="AnthropicBackendResponseError",
                error_message=f"Failed to parse Anthropic response: {e}",
                http_status=502,
                error_type="api_error",
            )

    # ------------------------------------------------------------------
    # Streaming — pure SSE relay
    # ------------------------------------------------------------------
    async def invoke_model_stream(
        self, request: MessageRequest, message_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream ``POST /v1/messages`` SSE events.

        Anthropic's streaming response is ALREADY in the target SSE format
        the proxy emits (``event: <type>\\ndata: <json>\\n\\n``), so this is
        a pure relay — no event conversion. Bytes are forwarded as-is to
        preserve SSE framing exactly (``aiter_lines`` would strip the
        ``\\n`` separators and risk corrupting event boundaries).
        """
        body = self._normalize_body(request)
        url = f"{self._base_url}/v1/messages"
        try:
            async with self._async_client.stream(
                "POST", url, json=body, headers=self._headers()
            ) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    yield self._format_sse_event(
                        self._error_event(
                            resp.status_code, text.decode("utf-8", errors="replace")
                        )
                    )
                    return
                # Forward raw bytes — SSE framing must be preserved verbatim.
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8", errors="replace")
        except httpx.HTTPError as e:
            yield self._format_sse_event(
                self._error_event(502, f"HTTP transport error: {e}")
            )

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------
    async def count_tokens(self, request: MessageRequest) -> int:
        """Call Anthropic's ``/v1/messages/count_tokens`` endpoint.

        Accepts either a ``MessageRequest`` or a ``CountTokensRequest``
        (the latter lacks ``max_tokens``; ``build_native_anthropic_body``
        defaults it to 1, which we pop before sending).

        On any error, fall back to a local heuristic (silent degradation)
        so the count_tokens endpoint never hard-fails.
        """
        body = self._normalize_body(request)
        # count_tokens does not require max_tokens; pop it to be safe.
        body.pop("max_tokens", None)
        url = f"{self._base_url}/v1/messages/count_tokens"
        try:
            resp = await self._async_client.post(
                url, json=body, headers=self._headers()
            )
            if resp.status_code == 200:
                data = resp.json()
                return int(data.get("input_tokens", 0))
            # Non-200 → fall through to heuristic.
        except Exception as e:
            logger.warning(
                "[ANTHROPIC-BACKEND] count_tokens failed, falling back to heuristic: %s",
                e,
            )
        return self._heuristic_token_count(request)

    def _heuristic_token_count(self, request: MessageRequest) -> int:
        """Local fallback when the Anthropic count_tokens endpoint is unavailable.

        Mirrors BedrockService._estimate_token_count's intent (rough char/4
        heuristic) without depending on the Bedrock converter. Good enough
        for billing-approximation purposes; never used on the happy path.
        """
        chars = 0
        for msg in request.messages:
            if isinstance(msg.content, str):
                chars += len(msg.content)
            elif msg.content:
                for block in msg.content:
                    text = ""
                    if isinstance(block, dict):
                        text = block.get("text", "") or json.dumps(
                            block.get("input", {}), default=str
                        )
                    elif hasattr(block, "text"):
                        text = getattr(block, "text", "") or ""
                    elif hasattr(block, "model_dump"):
                        text = json.dumps(block.model_dump(), default=str)
                    chars += len(text)
        if request.system:
            if isinstance(request.system, str):
                chars += len(request.system)
            else:
                for part in request.system:
                    if hasattr(part, "text"):
                        chars += len(getattr(part, "text", ""))
                    elif isinstance(part, dict):
                        chars += len(part.get("text", ""))
        # ~4 chars per token is the standard rough heuristic.
        return max(1, chars // 4)

    # ------------------------------------------------------------------
    # Error / SSE helpers
    # ------------------------------------------------------------------
    def _map_error(self, status_code: int, body: str) -> BedrockAPIError:
        """Map an Anthropic HTTP error to BedrockAPIError (caught by messages.py)."""
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            err_type = (
                err.get("type", "api_error") if isinstance(err, dict) else "api_error"
            )
            err_msg = err.get("message", body) if isinstance(err, dict) else body
        except Exception:
            err_type = "api_error"
            err_msg = body[:500]

        http_status = status_code
        if status_code == 429:
            err_type = err_type or "rate_limit_error"
        elif status_code == 401 or status_code == 403:
            err_type = "permission_error"
        elif status_code == 400:
            err_type = err_type or "invalid_request_error"
        elif status_code == 404:
            err_type = err_type or "not_found_error"

        return BedrockAPIError(
            error_code=f"AnthropicBackend_{status_code}",
            error_message=err_msg,
            http_status=http_status,
            error_type=err_type,
        )

    def _error_event(self, status_code: int, message: str) -> Dict[str, Any]:
        """Build an Anthropic-shaped SSE error event for the streaming path."""
        return {
            "type": "error",
            "error": {
                "type": "api_error" if status_code >= 500 else "invalid_request_error",
                "message": message,
            },
        }

    def _format_sse_event(self, event: Dict[str, Any]) -> str:
        """Format an event dict as an SSE string (matches Bedrock native path)."""
        event_type = event.get("type", "unknown")
        event_data = json.dumps(event)
        return f"event: {event_type}\ndata: {event_data}\n\n"
