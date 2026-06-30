"""AgentCore Gateway web search provider tests."""

import json

import httpx
import pytest

from app.core.config import Settings


def test_settings_reads_agentcore_gateway_url(monkeypatch):
    monkeypatch.setenv("AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    monkeypatch.setenv("AGENTCORE_GATEWAY_REGION", "us-east-1")

    settings = Settings(_env_file=None)

    assert settings.agentcore_gateway_url == "https://gateway.example/mcp"
    assert settings.agentcore_gateway_region == "us-east-1"


def test_create_search_provider_supports_agentcore_without_api_key(monkeypatch):
    from app.core.config import settings
    from app.services.web_search.providers import (
        AgentCoreSearchProvider,
        create_search_provider,
    )

    monkeypatch.setattr(settings, "web_search_provider", "agentcore", raising=False)
    monkeypatch.setattr(
        settings,
        "agentcore_gateway_url",
        "https://gateway.example/mcp",
        raising=False,
    )
    monkeypatch.setattr(
        settings, "agentcore_gateway_region", "us-east-1", raising=False
    )
    monkeypatch.setattr(settings, "web_search_api_key", None, raising=False)

    provider = create_search_provider()

    assert isinstance(provider, AgentCoreSearchProvider)
    assert provider.gateway_url == "https://gateway.example/mcp"


def _tools_list_response(tool_names):
    """Build a JSON-RPC tools/list response exposing the given tool names."""
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": "list-1",
            "result": {
                "tools": [
                    {
                        "name": name,
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "maxResults": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    }
                    for name in tool_names
                ]
            },
        },
    )


class RoutingFakeClient:
    """Fake httpx client that routes by JSON-RPC method.

    Records every request and returns a tools/list response advertising the
    given (target-prefixed) tool name, then a tools/call response.
    """

    def __init__(self, tool_names, call_response):
        self.tool_names = tool_names
        self.call_response = call_response
        self.requests = []
        self.list_calls = 0

    async def post(self, url, *, content, headers):
        body = json.loads(content.decode("utf-8"))
        self.requests.append(
            {"url": url, "content": content, "headers": headers, "body": body}
        )
        if body["method"] == "tools/list":
            self.list_calls += 1
            return _tools_list_response(self.tool_names)
        if body["method"] == "tools/call":
            return self.call_response(body)
        raise AssertionError(f"unexpected method {body['method']}")


async def test_agentcore_search_discovers_prefixed_tool_name_and_maps_results(
    monkeypatch,
):
    from app.services.web_search.providers import AgentCoreSearchProvider

    def call_response(body):
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "req-1",
                "result": {
                    "isError": False,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "id": "search-1",
                                    "results": [
                                        {
                                            "title": "AWS AgentCore",
                                            "url": "https://aws.example/agentcore",
                                            "text": "AgentCore web search snippet",
                                            "publishedDate": "2026-06-19",
                                        }
                                    ],
                                }
                            ),
                        }
                    ],
                },
            },
        )

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp",
        region="us-east-1",
    )
    fake_client = RoutingFakeClient(
        tool_names=["web-search-tool___WebSearch"], call_response=call_response
    )
    provider._client = fake_client
    monkeypatch.setattr(
        provider, "_signed_headers", lambda payload: {"Authorization": "signed"}
    )

    results = await provider.search("agentcore web search", max_results=3)

    assert len(results) == 1
    assert results[0].title == "AWS AgentCore"
    assert results[0].url == "https://aws.example/agentcore"
    assert results[0].content == "AgentCore web search snippet"
    assert results[0].page_age == "2026-06-19"

    # First request discovers the tool name, second invokes it.
    assert fake_client.requests[0]["body"]["method"] == "tools/list"
    call_request = fake_client.requests[1]
    assert call_request["url"] == "https://gateway.example/mcp"
    assert call_request["headers"]["Authorization"] == "signed"
    assert call_request["body"]["method"] == "tools/call"
    assert call_request["body"]["params"] == {
        "name": "web-search-tool___WebSearch",
        "arguments": {"query": "agentcore web search", "maxResults": 3},
    }


async def test_agentcore_caches_discovered_tool_name_across_searches(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    def call_response(body):
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "req",
                "result": {"isError": False, "content": []},
            },
        )

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp", region="us-east-1"
    )
    fake_client = RoutingFakeClient(
        tool_names=["web-search-tool___WebSearch"], call_response=call_response
    )
    provider._client = fake_client
    monkeypatch.setattr(provider, "_signed_headers", lambda payload: {})

    await provider.search("first")
    await provider.search("second")

    # tools/list must only be issued once; the name is cached afterwards.
    assert fake_client.list_calls == 1
    methods = [r["body"]["method"] for r in fake_client.requests]
    assert methods == ["tools/list", "tools/call", "tools/call"]


async def test_agentcore_uses_exact_tool_name_when_not_prefixed(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    def call_response(body):
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "req",
                "result": {"isError": False, "content": []},
            },
        )

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp", region="us-east-1"
    )
    fake_client = RoutingFakeClient(
        tool_names=["WebSearch"], call_response=call_response
    )
    provider._client = fake_client
    monkeypatch.setattr(provider, "_signed_headers", lambda payload: {})

    await provider.search("query")

    assert fake_client.requests[1]["body"]["params"]["name"] == "WebSearch"


async def test_agentcore_falls_back_to_single_tool(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    def call_response(body):
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "req",
                "result": {"isError": False, "content": []},
            },
        )

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp", region="us-east-1"
    )
    fake_client = RoutingFakeClient(
        tool_names=["custom-target___SomethingElse"], call_response=call_response
    )
    provider._client = fake_client
    monkeypatch.setattr(provider, "_signed_headers", lambda payload: {})

    await provider.search("query")

    assert (
        fake_client.requests[1]["body"]["params"]["name"]
        == "custom-target___SomethingElse"
    )


async def test_agentcore_raises_when_no_websearch_tool_available(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    def call_response(body):  # pragma: no cover - should never be reached
        raise AssertionError("tools/call must not be issued when discovery fails")

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp", region="us-east-1"
    )
    fake_client = RoutingFakeClient(
        tool_names=["target-a___ToolA", "target-b___ToolB"],
        call_response=call_response,
    )
    provider._client = fake_client
    monkeypatch.setattr(provider, "_signed_headers", lambda payload: {})

    with pytest.raises(ValueError, match="WebSearch"):
        await provider.search("query")


async def test_agentcore_search_parses_sse_jsonrpc_response(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    class FakeClient:
        async def post(self, url, *, content, headers):
            del url, headers
            body = json.loads(content.decode("utf-8"))
            if body["method"] == "tools/list":
                return _tools_list_response(["web-search-tool___WebSearch"])
            return httpx.Response(
                200,
                text=(
                    "event: message\n"
                    'data: {"jsonrpc":"2.0","id":"1","result":{"isError":false,'
                    '"content":[{"type":"text","text":"{\\"results\\":[{\\"title\\":'
                    '\\"SSE\\",\\"url\\":\\"https://example.com\\",\\"text\\":'
                    '\\"from sse\\"}]}"}]}}\n\n'
                ),
            )

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp",
        region="us-east-1",
    )
    provider._client = FakeClient()
    monkeypatch.setattr(provider, "_signed_headers", lambda payload: {})

    results = await provider.search("query")

    assert len(results) == 1
    assert results[0].title == "SSE"
    assert results[0].content == "from sse"


async def test_agentcore_search_clamps_query_to_agentcore_limit(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    class FakeClient:
        def __init__(self):
            self.call_payload = None

        async def post(self, url, *, content, headers):
            del url, headers
            body = json.loads(content.decode("utf-8"))
            if body["method"] == "tools/list":
                return _tools_list_response(["web-search-tool___WebSearch"])
            self.call_payload = body
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "req-1",
                    "result": {"isError": False, "content": []},
                },
            )

    provider = AgentCoreSearchProvider(
        gateway_url="https://gateway.example/mcp",
        region="us-east-1",
    )
    fake_client = FakeClient()
    provider._client = fake_client
    monkeypatch.setattr(provider, "_signed_headers", lambda payload: {})

    await provider.search("x" * 250)

    sent_query = fake_client.call_payload["params"]["arguments"]["query"]
    assert sent_query == "x" * 200
