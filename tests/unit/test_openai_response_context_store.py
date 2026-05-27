"""Tests for sharded DynamoDB storage of Responses API context."""

import hashlib

import pytest

from app.api.openai_passthrough.context_store import (
    ResponseContextNotFound,
    ResponseContextTooLarge,
    ShardedResponseContextStore,
)
from app.schemas.anthropic import Message, MessageRequest


def _text(message: Message) -> str:
    content = message.model_dump()["content"]
    assert isinstance(content, list)
    first = content[0]
    assert isinstance(first, dict)
    text = first["text"]
    assert isinstance(text, str)
    return text


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}

    def put_item(self, *, Item: dict) -> None:
        self.items[(Item["response_id"], Item["chunk_id"])] = dict(Item)

    def get_item(self, *, Key: dict) -> dict:
        item = self.items.get((Key["response_id"], Key["chunk_id"]))
        return {"Item": dict(item)} if item else {}

    def query(self, **kwargs) -> dict:
        response_id = kwargs["ExpressionAttributeValues"][":response_id"]
        chunks = [
            dict(item)
            for (pk, sk), item in self.items.items()
            if pk == response_id and sk.startswith("CHUNK#")
        ]
        return {"Items": sorted(chunks, key=lambda item: item["chunk_id"])}


def _message_request(text: str) -> MessageRequest:
    return MessageRequest(
        model="m",
        messages=[Message(role="user", content=text)],
        max_tokens=128,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )


def test_sharded_context_store_round_trips_messages():
    table = FakeTable()
    store = ShardedResponseContextStore(
        table,
        ttl_seconds=3600,
        chunk_size_bytes=48,
        max_context_bytes=2048,
        max_chunks=20,
    )

    store.save(
        response_id="resp_1",
        api_key="proxy-key",
        request=_message_request("Find a positive AI infrastructure story."),
        response_data={"output_text": "The answer mentioned Example Robotics."},
    )

    loaded = store.load("resp_1", api_key="proxy-key")

    assert [message.role for message in loaded] == ["user", "assistant"]
    assert _text(loaded[0]) == "Find a positive AI infrastructure story."
    assert _text(loaded[1]) == "The answer mentioned Example Robotics."
    assert table.items[("resp_1", "META")]["chunk_count"] > 1
    assert table.items[("resp_1", "META")]["encoding"] == "gzip+base64"


def test_sharded_context_store_rejects_wrong_api_key():
    table = FakeTable()
    store = ShardedResponseContextStore(table)
    store.save(
        response_id="resp_1",
        api_key="owner-key",
        request=_message_request("Find news."),
        response_data={"output_text": "Answer"},
    )

    with pytest.raises(ResponseContextNotFound):
        store.load("resp_1", api_key="other-key")


def test_sharded_context_store_uses_keyed_owner_digest():
    table = FakeTable()
    store = ShardedResponseContextStore(table)
    store.save(
        response_id="resp_1",
        api_key="owner-key",
        request=_message_request("Find news."),
        response_data={"output_text": "Answer"},
    )

    stored_hash = table.items[("resp_1", "META")]["api_key_hash"]
    bare_sha256 = hashlib.sha256(b"owner-key").hexdigest()

    assert stored_hash != bare_sha256


def test_sharded_context_store_rejects_missing_chunk():
    table = FakeTable()
    store = ShardedResponseContextStore(table)
    store.save(
        response_id="resp_1",
        api_key="proxy-key",
        request=_message_request("Find news."),
        response_data={"output_text": "Answer"},
    )
    del table.items[("resp_1", "CHUNK#000000")]

    with pytest.raises(ResponseContextNotFound):
        store.load("resp_1", api_key="proxy-key")


def test_sharded_context_store_rejects_context_that_cannot_fit_one_message():
    table = FakeTable()
    store = ShardedResponseContextStore(
        table,
        max_context_bytes=24,
        max_chunks=1,
    )

    with pytest.raises(ResponseContextTooLarge):
        store.save(
            response_id="resp_1",
            api_key="proxy-key",
            request=_message_request("x" * 1000),
            response_data={"output_text": "y" * 1000},
        )
