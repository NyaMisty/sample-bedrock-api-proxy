#!/usr/bin/env python3
"""End-to-end OpenAI SDK checks for /openai/v1 passthrough.

This script targets a deployed proxy and validates both OpenAI client surfaces:

- ``client.chat.completions.create()`` against ``/openai/v1/chat/completions``.
  The proxy should keep the Chat Completions response shape even though it now
  calls upstream Responses API with ``store=false`` internally.
- ``client.responses.create()`` against ``/openai/v1/responses``.

Configuration is loaded from ``tests_bak/.env`` via ``tests_bak/config.py`` and
can be overridden from the CLI.

Run examples:

    python tests_bak/openai_sdk_passthrough_test.py --base-url https://... --api-key sk-...
    python tests_bak/openai_sdk_passthrough_test.py --test chat_stream
    python tests_bak/openai_sdk_passthrough_test.py --model openai.gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from typing import Any

import httpx
from openai import APIStatusError, OpenAI

from config import API_KEY, BASE_URL

DEFAULT_MODEL = "openai.gpt-oss-120b"


def _openai_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.rstrip("/")
    if base_url.endswith("/openai/v1"):
        return base_url
    if base_url.endswith("/openai"):
        return f"{base_url}/v1"
    return f"{base_url}/openai/v1"


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=_openai_base_url(base_url))


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    return {"raw": str(value)}


def _usage_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return _to_dict(value)


def _assert_positive_usage(usage: Any, *, surface: str) -> dict[str, Any]:
    data = _usage_dict(usage)
    if surface == "chat":
        in_key = "prompt_tokens"
        out_key = "completion_tokens"
    else:
        in_key = "input_tokens"
        out_key = "output_tokens"
    if int(data.get(in_key) or 0) <= 0:
        raise AssertionError(f"{surface} usage missing positive {in_key}: {data!r}")
    if int(data.get(out_key) or 0) <= 0:
        raise AssertionError(f"{surface} usage missing positive {out_key}: {data!r}")
    return data


def _print_json(label: str, value: Any) -> None:
    print(f"{label}=" + json.dumps(value, ensure_ascii=False, indent=2, default=str))


def test_health(base_url: str) -> None:
    print("=" * 72)
    print("GET /health")
    print("=" * 72)
    response = httpx.get(base_url.rstrip("/") + "/health", timeout=30)
    print(f"status_code={response.status_code}")
    print(response.text[:1000])
    response.raise_for_status()


def test_models_list(client: OpenAI) -> None:
    print("=" * 72)
    print("GET /openai/v1/models")
    print("=" * 72)
    models = client.models.list()
    ids = [model.id for model in models.data]
    print(f"received {len(ids)} model(s); first 10: {ids[:10]}")
    if not ids:
        raise AssertionError("models list returned empty")


def test_chat_non_stream(client: OpenAI, model: str) -> None:
    print("=" * 72)
    print(f"POST /openai/v1/chat/completions non-streaming model={model}")
    print("=" * 72)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer concisely. Return only the requested result.",
            },
            {"role": "user", "content": "What is 17 + 25?"},
        ],
        max_completion_tokens=128,
    )
    data = _to_dict(completion)
    message = completion.choices[0].message
    usage = _assert_positive_usage(completion.usage, surface="chat")

    print(f"id={completion.id}")
    print(f"object={completion.object}")
    print(f"model={completion.model}")
    print(f"finish_reason={completion.choices[0].finish_reason}")
    print(f"content={message.content!r}")
    _print_json("usage", usage)

    if completion.object != "chat.completion":
        raise AssertionError(f"expected chat.completion object, got {completion.object!r}")
    if not message.content:
        raise AssertionError("chat non-streaming returned empty content")
    if "choices" not in data:
        raise AssertionError(f"chat response missing choices: {data!r}")


def test_chat_stream(client: OpenAI, model: str) -> None:
    print("=" * 72)
    print(f"POST /openai/v1/chat/completions streaming model={model}")
    print("=" * 72)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 5, separated by commas.",
            }
        ],
        max_completion_tokens=128,
        stream=True,
        stream_options={"include_usage": True},
    )

    parts: list[str] = []
    final_usage = None
    for chunk in stream:
        if chunk.usage is not None:
            final_usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                parts.append(delta.content)
    print()

    if not "".join(parts).strip():
        raise AssertionError("chat streaming returned no text deltas")
    usage = _assert_positive_usage(final_usage, surface="chat")
    _print_json("final_usage", usage)


def test_responses_non_stream(client: OpenAI, model: str) -> str:
    print("=" * 72)
    print(f"POST /openai/v1/responses non-streaming model={model}")
    print("=" * 72)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": "Say hello in English, Spanish, and Japanese.",
            }
        ],
        max_output_tokens=160,
    )
    data = _to_dict(response)
    usage = _assert_positive_usage(response.usage, surface="responses")
    output_text = getattr(response, "output_text", None)

    print(f"id={response.id}")
    print(f"object={data.get('object')}")
    print(f"status={response.status}")
    print(f"output_text={output_text!r}")
    _print_json("usage", usage)

    if data.get("object") != "response":
        raise AssertionError(f"expected object=response, got {data.get('object')!r}")
    if not response.id:
        raise AssertionError("responses non-streaming returned no id")
    return response.id


def test_responses_stream(client: OpenAI, model: str) -> str:
    print("=" * 72)
    print(f"POST /openai/v1/responses streaming model={model}")
    print("=" * 72)
    stream = client.responses.create(
        model=model,
        input=[{"role": "user", "content": "Write one short sentence about AWS."}],
        max_output_tokens=160,
        stream=True,
    )

    response_id = ""
    deltas: list[str] = []
    completed_usage = None
    seen_events: set[str] = set()

    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type:
            seen_events.add(event_type)
        if event_type == "response.created":
            response_id = event.response.id
            print(f"[response.created] id={response_id}")
        elif event_type == "response.output_text.delta":
            sys.stdout.write(event.delta)
            sys.stdout.flush()
            deltas.append(event.delta)
        elif event_type == "response.completed":
            completed_usage = event.response.usage
            print(f"\n[response.completed] usage={completed_usage}")

    print()
    if "response.created" not in seen_events:
        raise AssertionError(f"missing response.created event; saw {sorted(seen_events)!r}")
    if "response.output_text.delta" not in seen_events:
        raise AssertionError(
            f"missing response.output_text.delta event; saw {sorted(seen_events)!r}"
        )
    if "response.completed" not in seen_events:
        raise AssertionError(f"missing response.completed event; saw {sorted(seen_events)!r}")
    if not "".join(deltas).strip():
        raise AssertionError("responses streaming returned no text deltas")
    _assert_positive_usage(completed_usage, surface="responses")
    return response_id


def test_responses_retrieve_delete(client: OpenAI, response_ids: Iterable[str]) -> None:
    for response_id in response_ids:
        if not response_id:
            continue
        print("=" * 72)
        print(f"GET /openai/v1/responses/{response_id}")
        print("=" * 72)
        response = client.responses.retrieve(response_id)
        print(f"id={response.id} status={response.status} model={response.model}")
        if response.id != response_id:
            raise AssertionError(f"retrieve returned {response.id!r}, expected {response_id!r}")

        print("=" * 72)
        print(f"DELETE /openai/v1/responses/{response_id}")
        print("=" * 72)
        try:
            deleted = client.responses.delete(response_id)
        except APIStatusError as exc:
            body = getattr(exc.response, "text", "")
            if exc.status_code in {401, 403} and (
                "DeleteInference" in body or "access_denied" in body
            ):
                print(
                    "skip delete: upstream Bedrock API key lacks "
                    "bedrock-mantle:DeleteInference"
                )
                continue
            raise
        print(deleted)


TESTS = {
    "health": lambda client, model, base_url: test_health(base_url),
    "models": lambda client, model, base_url: test_models_list(client),
    "chat_non_stream": lambda client, model, base_url: test_chat_non_stream(client, model),
    "chat_stream": lambda client, model, base_url: test_chat_stream(client, model),
    "responses_non_stream": lambda client, model, base_url: test_responses_non_stream(client, model),
    "responses_stream": lambda client, model, base_url: test_responses_stream(client, model),
}


def run_all(client: OpenAI, model: str, base_url: str) -> None:
    test_health(base_url)
    print()
    test_models_list(client)
    print()
    test_chat_non_stream(client, model)
    print()
    test_chat_stream(client, model)
    print()
    non_stream_id = test_responses_non_stream(client, model)
    print()
    stream_id = test_responses_stream(client, model)
    print()
    time.sleep(1)
    test_responses_retrieve_delete(client, [non_stream_id, stream_id])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id")
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="all",
        help="which test to run",
    )
    parser.add_argument("--api-key", default=API_KEY, help="proxy API key")
    parser.add_argument("--base-url", default=BASE_URL, help="proxy root or /openai/v1 URL")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Missing API key. Set API_KEY in tests_bak/.env or pass --api-key.")
    if not args.base_url:
        sys.exit("Missing base URL. Set BASE_URL in tests_bak/.env or pass --base-url.")

    print(f"proxy root/base = {args.base_url.rstrip('/')}")
    print(f"openai base    = {_openai_base_url(args.base_url)}")
    print(f"model          = {args.model}")
    print()

    client = make_client(args.base_url, args.api_key)

    if args.test == "all":
        run_all(client, args.model, args.base_url)
    else:
        TESTS[args.test](client, args.model, args.base_url)

    print("\nAll requested passthrough checks passed.")


if __name__ == "__main__":
    main()
