"""FastAPI routes for the OpenAI passthrough endpoints.

Mounted at /openai/v1 only when settings.enable_openai_passthrough is True.
"""
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.openai_passthrough.client import get_client, upstream_headers
from app.api.openai_passthrough.model_mapping import resolve_model_id
from app.api.openai_passthrough.streaming import stream_passthrough
from app.api.openai_passthrough.usage_extractor import normalize_usage
from app.db.dynamodb import DynamoDBClient, ModelMappingManager, UsageTracker
from app.middleware.auth import get_api_key_info

logger = logging.getLogger(__name__)
router = APIRouter()

_ddb: DynamoDBClient | None = None
_mapping: ModelMappingManager | None = None
_usage: UsageTracker | None = None


def _managers() -> tuple[ModelMappingManager, UsageTracker]:
    """Lazily build DDB managers — keeps import-time side effects out of tests."""
    global _ddb, _mapping, _usage
    if _ddb is None or _mapping is None or _usage is None:
        _ddb = DynamoDBClient()
        _mapping = ModelMappingManager(_ddb)
        _usage = UsageTracker(_ddb)
    return _mapping, _usage


def _record_usage(api_key_info: Dict[str, Any], raw_usage: Dict[str, Any], model: str, api_surface: str) -> None:
    _, usage = _managers()
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


def _passthrough_extra_headers(request: Request) -> Dict[str, str]:
    """Forward Bedrock-specific headers from the client to upstream (e.g. guardrails)."""
    extra: Dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower().startswith("x-amzn-bedrock-"):
            extra[name] = value
    return extra


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    api_key_info: Dict[str, Any] = Depends(get_api_key_info),
):
    body = await request.json()
    mapping, _ = _managers()
    body["model"] = resolve_model_id(body.get("model", ""), mapping)
    extra = _passthrough_extra_headers(request)

    if body.get("stream"):
        async def on_complete(usage: Dict[str, Any]) -> None:
            _record_usage(api_key_info, usage, body["model"], "chat_completions")
        return StreamingResponse(
            stream_passthrough(
                "POST", "/chat/completions", body, "chat_completions", on_complete, extra
            ),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        "/chat/completions", json=body, headers=upstream_headers(extra)
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
    api_key_info: Dict[str, Any] = Depends(get_api_key_info),
):
    body = await request.json()
    mapping, _ = _managers()
    body["model"] = resolve_model_id(body.get("model", ""), mapping)
    extra = _passthrough_extra_headers(request)

    if body.get("stream"):
        async def on_complete(usage: Dict[str, Any]) -> None:
            _record_usage(api_key_info, usage, body["model"], "responses")
        return StreamingResponse(
            stream_passthrough("POST", "/responses", body, "responses", on_complete, extra),
            media_type="text/event-stream",
        )

    resp = await get_client().post(
        "/responses", json=body, headers=upstream_headers(extra)
    )
    if resp.status_code >= 400:
        return JSONResponse(_safe_json(resp), status_code=resp.status_code)

    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
        _record_usage(api_key_info, data["usage"], body["model"], "responses")
    return JSONResponse(data, status_code=resp.status_code)


def _safe_json(resp) -> Dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": resp.text, "type": "upstream_error"}}
