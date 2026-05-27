"""FastAPI routes for the OpenAI passthrough endpoints.

Mounted at /openai/v1 only when settings.enable_openai_passthrough is True.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.openai_passthrough.client import get_client, upstream_headers, upstream_url
from app.api.openai_passthrough.context_store import (
    ResponseContextNotFound,
    ResponseContextTooLarge,
    get_response_context_store,
)
from app.api.openai_passthrough.model_mapping import resolve_model_id
from app.api.openai_passthrough.streaming import (
    UpstreamConnectionError,
    open_upstream_stream,
    stream_passthrough_response,
)
from app.api.openai_passthrough.usage_extractor import normalize_usage
from app.api.openai_passthrough.web_search import (
    OpenAIResponsesWebSearchError,
    build_message_request,
    build_response_json,
    ensure_web_search_enabled,
    handle_non_streaming_web_search,
    is_responses_web_search_request,
    stream_response_events,
)
from app.db.dynamodb import DynamoDBClient, ModelMappingManager, UsageTracker
from app.middleware.auth import get_api_key_info
from app.services.bedrock_service import BedrockService
from app.services.web_search_service import get_web_search_service

logger = logging.getLogger(__name__)
router = APIRouter()

_ddb: DynamoDBClient | None = None
_mapping: ModelMappingManager | None = None
_usage: UsageTracker | None = None
_context_store: Any | None = None


def _managers() -> tuple[ModelMappingManager, UsageTracker, Any]:
    """Lazily build DDB managers — keeps import-time side effects out of tests."""
    global _ddb, _mapping, _usage, _context_store
    if _ddb is None or _mapping is None or _usage is None or _context_store is None:
        _ddb = DynamoDBClient()
        _mapping = ModelMappingManager(_ddb)
        _usage = UsageTracker(_ddb)
        _context_store = get_response_context_store(_ddb)
    return _mapping, _usage, _context_store


def _record_usage(
    api_key_info: dict[str, Any],
    raw_usage: dict[str, Any],
    model: str,
    api_surface: str,
) -> None:
    _, usage, _ = _managers()
    norm = normalize_usage(raw_usage, api_surface)
    try:
        usage.record_usage(
            api_key=api_key_info.get("api_key", ""),
            request_id=str(uuid4()),
            model=model,
            input_tokens=norm["input_tokens"],
            output_tokens=norm["output_tokens"],
            cached_tokens=norm["cache_read_input_tokens"],
            cache_write_input_tokens=norm["cache_creation_input_tokens"],
            api_surface=api_surface,
            reasoning_tokens=norm["reasoning_tokens"],
        )
    except Exception as exc:
        logger.warning("[OPENAI-PASSTHROUGH] usage recording failed: %s", exc)


def _api_error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": str(exc), "type": "api_error"}},
        status_code=500,
    )


def _passthrough_extra_headers(request: Request) -> dict[str, str]:
    """Forward Bedrock-specific headers from the client to upstream (e.g. guardrails)."""
    extra: dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower().startswith("x-amzn-bedrock-"):
            extra[name] = value
    return extra


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    api_key_info: dict[str, Any] = Depends(get_api_key_info),
):
    body = await request.json()
    mapping, _, _ = _managers()
    body["model"] = resolve_model_id(body.get("model", ""), mapping)
    extra = _passthrough_extra_headers(request)

    if body.get("stream"):
        try:
            upstream_resp, error_body = await open_upstream_stream(
                "POST", "/chat/completions", body, extra
            )
        except UpstreamConnectionError as exc:
            return JSONResponse(
                {"error": {"message": exc.message, "type": "upstream_error"}},
                status_code=exc.status_code,
            )
        if error_body is not None:
            return JSONResponse(
                _decode_error_body(error_body),
                status_code=upstream_resp.status_code,
            )

        async def on_complete(usage: dict[str, Any]) -> None:
            _record_usage(api_key_info, usage, body["model"], "chat_completions")

        return StreamingResponse(
            stream_passthrough_response(upstream_resp, "chat_completions", on_complete),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        upstream_url("/chat/completions"), json=body, headers=upstream_headers(extra)
    )
    if resp.status_code >= 400:
        return JSONResponse(_safe_json(resp), status_code=resp.status_code)

    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
        _record_usage(api_key_info, data["usage"], body["model"], "chat_completions")
    return JSONResponse(data, status_code=resp.status_code)


@router.post("/responses")
async def responses_create(
    request: Request,
    api_key_info: dict[str, Any] = Depends(get_api_key_info),
):
    body = await request.json()
    mapping, _, context_store = _managers()
    body["model"] = resolve_model_id(body.get("model", ""), mapping)
    extra = _passthrough_extra_headers(request)

    if is_responses_web_search_request(body):
        request_id = f"resp-{uuid4().hex}"
        service_tier = api_key_info.get("service_tier", "default")
        api_key = api_key_info.get("api_key", "")
        previous_messages = None
        previous_response_id = body.get("previous_response_id")
        if previous_response_id is not None:
            if not isinstance(previous_response_id, str):
                return JSONResponse(
                    {
                        "error": {
                            "message": "previous_response_id must be a string",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )
            try:
                previous_messages = context_store.load(
                    previous_response_id,
                    api_key=api_key,
                )
            except ResponseContextNotFound:
                return JSONResponse(
                    {
                        "error": {
                            "message": (
                                f"previous_response_id {previous_response_id!r} "
                                "was not found"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=404,
                )

        try:
            ensure_web_search_enabled()
            message_request = build_message_request(
                body,
                previous_messages=previous_messages,
            )
        except OpenAIResponsesWebSearchError as exc:
            return JSONResponse(exc.to_error_body(), status_code=exc.status_code)

        try:
            web_search_service = get_web_search_service()
            bedrock_service = BedrockService()
        except Exception as exc:
            return _api_error_response(exc)

        if body.get("stream"):
            try:
                response = await web_search_service.handle_request(
                    request=message_request,
                    bedrock_service=bedrock_service,
                    request_id=request_id,
                    service_tier=service_tier,
                    anthropic_beta=None,
                )
            except Exception as exc:
                return _api_error_response(exc)

            data = build_response_json(
                response,
                original_model=body.get("model", ""),
                response_id=request_id,
            )
            try:
                context_store.save(
                    response_id=data["id"],
                    api_key=api_key,
                    request=message_request,
                    response_data=data,
                )
            except ResponseContextTooLarge as exc:
                logger.warning("[OPENAI-PASSTHROUGH] context not stored: %s", exc)
            except Exception as exc:
                logger.warning(
                    "[OPENAI-PASSTHROUGH] context storage failed: %s",
                    exc,
                )
            if isinstance(data.get("usage"), dict):
                _record_usage(api_key_info, data["usage"], body["model"], "responses")

            return StreamingResponse(
                stream_response_events(
                    response,
                    original_model=body.get("model", ""),
                    response_id=request_id,
                    response_data=data,
                ),
                media_type="text/event-stream",
            )

        try:
            data = await handle_non_streaming_web_search(
                body,
                message_request=message_request,
                web_search_service=web_search_service,
                bedrock_service=bedrock_service,
                request_id=request_id,
                service_tier=service_tier,
            )
        except OpenAIResponsesWebSearchError as exc:
            return JSONResponse(exc.to_error_body(), status_code=exc.status_code)
        except Exception as exc:
            return _api_error_response(exc)
        try:
            context_store.save(
                response_id=data["id"],
                api_key=api_key,
                request=message_request,
                response_data=data,
            )
        except ResponseContextTooLarge as exc:
            logger.warning("[OPENAI-PASSTHROUGH] context not stored: %s", exc)
        except Exception as exc:
            logger.warning("[OPENAI-PASSTHROUGH] context storage failed: %s", exc)
        if isinstance(data.get("usage"), dict):
            _record_usage(api_key_info, data["usage"], body["model"], "responses")
        return JSONResponse(data, status_code=200)

    if body.get("stream"):
        try:
            upstream_resp, error_body = await open_upstream_stream(
                "POST", "/responses", body, extra
            )
        except UpstreamConnectionError as exc:
            return JSONResponse(
                {"error": {"message": exc.message, "type": "upstream_error"}},
                status_code=exc.status_code,
            )
        if error_body is not None:
            return JSONResponse(
                _decode_error_body(error_body),
                status_code=upstream_resp.status_code,
            )

        async def on_complete(usage: dict[str, Any]) -> None:
            _record_usage(api_key_info, usage, body["model"], "responses")

        return StreamingResponse(
            stream_passthrough_response(upstream_resp, "responses", on_complete),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        upstream_url("/responses"), json=body, headers=upstream_headers(extra)
    )
    if resp.status_code >= 400:
        return JSONResponse(_safe_json(resp), status_code=resp.status_code)

    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
        _record_usage(api_key_info, data["usage"], body["model"], "responses")
    return JSONResponse(data, status_code=resp.status_code)


async def _passthrough_request(request: Request, path: str) -> Response:
    """Forward request to upstream and mirror the upstream response."""
    extra = _passthrough_extra_headers(request)
    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.json()
        except Exception:
            body = None
    resp = await get_client().request(
        request.method, upstream_url(path), json=body, headers=upstream_headers(extra)
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


@router.api_route("/responses/{response_id}", methods=["GET", "DELETE"])
async def responses_get_or_delete(
    response_id: str,
    request: Request,
    _: dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, f"/responses/{response_id}")


@router.post("/responses/{response_id}/cancel")
async def responses_cancel(
    response_id: str,
    request: Request,
    _: dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, f"/responses/{response_id}/cancel")


@router.get("/responses/{response_id}/input_items")
async def responses_input_items(
    response_id: str,
    request: Request,
    _: dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, f"/responses/{response_id}/input_items")


@router.get("/models")
async def list_models(
    request: Request,
    _: dict[str, Any] = Depends(get_api_key_info),
):
    return await _passthrough_request(request, "/models")


def _safe_json(resp) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], resp.json())
    except ValueError:
        return {"error": {"message": resp.text, "type": "upstream_error"}}


def _decode_error_body(body: bytes) -> dict[str, Any]:
    """Parse a non-2xx upstream body as JSON, falling back to a wrapped string."""
    import json as _json

    try:
        decoded = _json.loads(body)
    except (ValueError, TypeError):
        return {
            "error": {
                "message": body.decode("utf-8", "replace"),
                "type": "upstream_error",
            }
        }
    if isinstance(decoded, dict):
        return cast(dict[str, Any], decoded)
    return {"error": {"message": str(decoded), "type": "upstream_error"}}
