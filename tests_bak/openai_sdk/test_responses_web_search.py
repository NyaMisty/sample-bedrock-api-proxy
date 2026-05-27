#!/usr/bin/env python3
"""
OpenAI SDK smoke test for proxy-managed Responses API web_search.

Usage:
    cd tests_bak/openai_sdk
    python test_responses_web_search.py
    python test_responses_web_search.py --non-stream
    python test_responses_web_search.py --stream
    python test_responses_web_search.py --model openai.gpt-oss-120b

Configuration:
    Loads ../.env by default and uses:
      BASE_URL  - proxy root URL, for example https://...cloudfront.net
      API_KEY   - proxy API key

    You may also override with:
      OPENAI_PROXY_BASE_URL
      OPENAI_PROXY_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from openai import OpenAI

warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - convenience for ad-hoc runs
    load_dotenv = None


DEFAULT_MODEL = "openai.gpt-oss-120b"
DEFAULT_QUERY = (
    "Search the web for one current positive technology news story from today. "
    "Answer in two concise bullet points and mention the source title if available."
)


def _load_env() -> None:
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)


def _openai_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.rstrip("/")
    if base_url.endswith("/openai/v1"):
        return base_url
    if base_url.endswith("/openai"):
        return f"{base_url}/v1"
    return f"{base_url}/openai/v1"


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    return {"raw": str(value)}


def _response_to_dict(response: Any) -> dict[str, Any]:
    return _to_dict(response)


def _output_items(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    output = response_data.get("output")
    if not isinstance(output, list):
        return []
    return [_to_dict(item) for item in output]


def _output_text(response: Any, response_data: dict[str, Any]) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    text = response_data.get("output_text")
    if isinstance(text, str):
        return text
    return ""


def _assert_non_streaming_response(response: Any) -> dict[str, Any]:
    data = _response_to_dict(response)
    items = _output_items(data)
    item_types = [item.get("type") for item in items]
    text = _output_text(response, data)

    if data.get("object") != "response":
        raise AssertionError(f"expected object=response, got {data.get('object')!r}")
    if data.get("status") != "completed":
        raise AssertionError(f"expected status=completed, got {data.get('status')!r}")
    if "web_search_call" not in item_types:
        raise AssertionError(f"expected a web_search_call output item, got {item_types!r}")
    if "message" not in item_types:
        raise AssertionError(f"expected a message output item, got {item_types!r}")
    if not text.strip():
        raise AssertionError("expected non-empty output_text")

    return {
        "id": data.get("id"),
        "status": data.get("status"),
        "output_types": item_types,
        "usage": data.get("usage"),
        "output_text": text,
    }


def run_non_streaming(client: OpenAI, model: str, query: str) -> None:
    print("\n=== Responses web_search: non-streaming ===")
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=query,
    )
    summary = _assert_non_streaming_response(response)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def _event_to_dict(event: Any) -> dict[str, Any]:
    data = _to_dict(event)
    if "type" not in data:
        event_type = getattr(event, "type", None)
        if event_type:
            data["type"] = event_type
    return data


def run_streaming(client: OpenAI, model: str, query: str) -> None:
    print("\n=== Responses web_search: streaming ===")
    seen: set[str] = set()
    deltas: list[str] = []
    completed_response: dict[str, Any] | None = None

    stream = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=query,
        stream=True,
    )

    for event in stream:
        data = _event_to_dict(event)
        event_type = data.get("type")
        if event_type:
            seen.add(str(event_type))

        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.completed":
            response_data = data.get("response")
            if isinstance(response_data, dict):
                completed_response = response_data

        print(f"event: {event_type}")

    if "response.created" not in seen:
        raise AssertionError(f"missing response.created event; saw {sorted(seen)!r}")
    if "response.output_text.delta" not in seen:
        raise AssertionError(f"missing response.output_text.delta event; saw {sorted(seen)!r}")
    if "response.completed" not in seen:
        raise AssertionError(f"missing response.completed event; saw {sorted(seen)!r}")
    if not "".join(deltas).strip():
        raise AssertionError("expected non-empty streamed output_text delta")
    if not completed_response:
        raise AssertionError("expected response.completed to include response data")

    item_types = [
        item.get("type")
        for item in _output_items(completed_response)
    ]
    if "web_search_call" not in item_types:
        raise AssertionError(f"expected completed response to include web_search_call, got {item_types!r}")

    print(
        json.dumps(
            {
                "events": sorted(seen),
                "output_types": item_types,
                "usage": completed_response.get("usage"),
                "output_text": "".join(deltas),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test OpenAI Responses API web_search through the proxy."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--non-stream", action="store_true", help="Run only non-streaming test")
    mode.add_argument("--stream", action="store_true", help="Run only streaming test")
    parser.add_argument("--model", default=os.getenv("OPENAI_TEST_MODEL", DEFAULT_MODEL))
    parser.add_argument("--query", default=os.getenv("OPENAI_WEB_SEARCH_QUERY", DEFAULT_QUERY))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_PROXY_BASE_URL") or os.getenv("BASE_URL"),
        help="Proxy root URL or /openai/v1 URL. Defaults to OPENAI_PROXY_BASE_URL or BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_PROXY_API_KEY") or os.getenv("API_KEY"),
        help="Proxy API key. Defaults to OPENAI_PROXY_API_KEY or API_KEY.",
    )
    return parser.parse_args()


def main() -> int:
    _load_env()
    args = parse_args()

    if not args.base_url:
        print("Missing BASE_URL or OPENAI_PROXY_BASE_URL", file=sys.stderr)
        return 2
    if not args.api_key:
        print("Missing API_KEY or OPENAI_PROXY_API_KEY", file=sys.stderr)
        return 2

    base_url = _openai_base_url(args.base_url)
    client = OpenAI(base_url=base_url, api_key=args.api_key)
    print(f"Base URL: {base_url}")
    print(f"Model:    {args.model}")

    try:
        if args.stream:
            run_streaming(client, args.model, args.query)
        elif args.non_stream:
            run_non_streaming(client, args.model, args.query)
        else:
            run_non_streaming(client, args.model, args.query)
            run_streaming(client, args.model, args.query)
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print("\nOK: Responses API web_search smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
