"""Tests for the api_surface and reasoning_tokens additions to UsageTracker."""
from unittest.mock import MagicMock

from app.db.dynamodb import UsageTracker


def _make_tracker():
    ddb_client = MagicMock()
    ddb_client.usage_table_name = "anthropic-proxy-usage"
    tracker = UsageTracker(ddb_client)
    tracker.table = MagicMock()
    return tracker


def test_record_usage_writes_api_surface_when_provided():
    tracker = _make_tracker()
    tracker.record_usage(
        api_key="sk-x",
        request_id="req-1",
        model="openai.gpt-oss-120b",
        input_tokens=100,
        output_tokens=50,
        api_surface="chat_completions",
    )
    item = tracker.table.put_item.call_args.kwargs["Item"]
    assert item["api_surface"] == "chat_completions"


def test_record_usage_writes_reasoning_tokens_when_provided():
    tracker = _make_tracker()
    tracker.record_usage(
        api_key="sk-x", request_id="req-1", model="m",
        input_tokens=10, output_tokens=5, reasoning_tokens=3,
    )
    item = tracker.table.put_item.call_args.kwargs["Item"]
    assert item["reasoning_tokens"] == 3


def test_record_usage_omits_new_fields_when_default():
    tracker = _make_tracker()
    tracker.record_usage(
        api_key="sk-x", request_id="req-1", model="m",
        input_tokens=10, output_tokens=5,
    )
    item = tracker.table.put_item.call_args.kwargs["Item"]
    # Sparse: not written when caller didn't specify them
    assert "api_surface" not in item
    assert "reasoning_tokens" not in item
