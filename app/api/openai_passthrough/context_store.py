"""Shared state storage for proxy-managed OpenAI Responses requests."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import time
from typing import Any

from app.core.config import settings
from app.schemas.anthropic import Message, MessageRequest


class ResponseContextNotFound(Exception):
    """Raised when a previous response context cannot be loaded."""


class ResponseContextTooLarge(Exception):
    """Raised when context exceeds configured storage limits."""


def _api_key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _message_to_dict(message: Message) -> dict[str, str]:
    dumped = message.model_dump()
    return {
        "role": str(dumped["role"]),
        "content": _content_text(dumped.get("content")),
    }


def _messages_from_context(value: Any) -> list[Message]:
    if not isinstance(value, list):
        raise ResponseContextNotFound("Stored response context is malformed")
    messages: list[Message] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content:
            messages.append(Message(role=role, content=content))
    if not messages:
        raise ResponseContextNotFound("Stored response context is empty")
    return messages


class ShardedResponseContextStore:
    """Store compressed response context as multiple DynamoDB chunk items."""

    def __init__(
        self,
        table: Any,
        *,
        ttl_seconds: int | None = None,
        chunk_size_bytes: int | None = None,
        max_context_bytes: int | None = None,
        max_chunks: int | None = None,
    ) -> None:
        self.table = table
        self.ttl_seconds = ttl_seconds or settings.response_context_ttl_seconds
        self.chunk_size_bytes = (
            chunk_size_bytes or settings.response_context_chunk_size_bytes
        )
        self.max_context_bytes = (
            max_context_bytes or settings.response_context_max_bytes
        )
        self.max_chunks = max_chunks or settings.response_context_max_chunks

    def _context_messages(
        self,
        request: MessageRequest,
        response_data: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages = [_message_to_dict(message) for message in request.messages]
        output_text = response_data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            messages.append({"role": "assistant", "content": output_text})
        return messages

    def _encode(self, messages: list[dict[str, str]]) -> str:
        while messages:
            raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            payload = base64.b64encode(gzip.compress(raw)).decode("ascii")
            if len(payload.encode("utf-8")) <= self.max_context_bytes:
                return payload
            if len(messages) == 1:
                break
            messages = messages[1:]
        raise ResponseContextTooLarge("response context exceeds max storage size")

    def save(
        self,
        *,
        response_id: str,
        api_key: str,
        request: MessageRequest,
        response_data: dict[str, Any],
    ) -> None:
        payload = self._encode(self._context_messages(request, response_data))
        chunks = [
            payload[index : index + self.chunk_size_bytes]
            for index in range(0, len(payload), self.chunk_size_bytes)
        ]
        if not chunks or len(chunks) > self.max_chunks:
            raise ResponseContextTooLarge("response context exceeds max chunk count")

        expires_at = int(time.time()) + self.ttl_seconds
        self.table.put_item(
            Item={
                "response_id": response_id,
                "chunk_id": "META",
                "api_key_hash": _api_key_hash(api_key),
                "encoding": "gzip+base64",
                "chunk_count": len(chunks),
                "total_bytes": len(payload.encode("utf-8")),
                "expires_at": expires_at,
                "created_at": int(time.time()),
            }
        )
        for index, chunk in enumerate(chunks):
            self.table.put_item(
                Item={
                    "response_id": response_id,
                    "chunk_id": f"CHUNK#{index:06d}",
                    "payload": chunk,
                    "expires_at": expires_at,
                }
            )

    def load(self, response_id: str, *, api_key: str) -> list[Message]:
        meta = self.table.get_item(
            Key={"response_id": response_id, "chunk_id": "META"}
        ).get("Item")
        if not meta or meta.get("api_key_hash") != _api_key_hash(api_key):
            raise ResponseContextNotFound("previous_response_id was not found")

        chunk_count = int(meta.get("chunk_count") or 0)
        if chunk_count < 1 or chunk_count > self.max_chunks:
            raise ResponseContextNotFound("previous_response_id context is incomplete")

        result = self.table.query(
            KeyConditionExpression=(
                "response_id = :response_id AND begins_with(chunk_id, :chunk_prefix)"
            ),
            ExpressionAttributeValues={
                ":response_id": response_id,
                ":chunk_prefix": "CHUNK#",
            },
        )
        chunks = sorted(result.get("Items") or [], key=lambda item: item["chunk_id"])
        if len(chunks) != chunk_count:
            raise ResponseContextNotFound("previous_response_id context is incomplete")

        payload = "".join(str(chunk.get("payload") or "") for chunk in chunks)
        try:
            raw = gzip.decompress(base64.b64decode(payload.encode("ascii")))
            decoded = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ResponseContextNotFound(
                "previous_response_id context is malformed"
            ) from exc
        return _messages_from_context(decoded)


def get_response_context_store(dynamodb_client: Any) -> ShardedResponseContextStore:
    table = dynamodb_client.dynamodb.Table(dynamodb_client.response_context_table_name)
    return ShardedResponseContextStore(table)
