"""AgentCore Gateway web search provider tests."""

import json

import httpx

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


async def test_agentcore_search_invokes_websearch_tool_and_maps_results(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    class FakeClient:
        def __init__(self):
            self.requests = []

        async def post(self, url, *, content, headers):
            self.requests.append({"url": url, "content": content, "headers": headers})
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
    fake_client = FakeClient()
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

    request = fake_client.requests[0]
    assert request["url"] == "https://gateway.example/mcp"
    assert request["headers"]["Authorization"] == "signed"
    payload = json.loads(request["content"].decode("utf-8"))
    assert payload["method"] == "tools/call"
    assert payload["params"] == {
        "name": "WebSearch",
        "arguments": {"query": "agentcore web search", "maxResults": 3},
    }


async def test_agentcore_search_parses_sse_jsonrpc_response(monkeypatch):
    from app.services.web_search.providers import AgentCoreSearchProvider

    class FakeClient:
        async def post(self, url, *, content, headers):
            del url, content, headers
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
            self.payload = None

        async def post(self, url, *, content, headers):
            del url, headers
            self.payload = json.loads(content.decode("utf-8"))
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

    sent_query = fake_client.payload["params"]["arguments"]["query"]
    assert sent_query == "x" * 200
