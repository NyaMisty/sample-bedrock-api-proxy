"""
Search provider interface and implementations.

Supports Tavily, Brave Search, and AgentCore Gateway Web Search providers.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result from any provider."""

    url: str
    title: str
    content: str  # Page content/snippet
    page_age: str | None = None


class SearchProvider(ABC):
    """Abstract search provider interface."""

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict | None = None,
    ) -> list[SearchResult]:
        """
        Execute a web search.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            allowed_domains: Only include results from these domains
            blocked_domains: Exclude results from these domains
            user_location: Optional location dict for localized results

        Returns:
            List of SearchResult objects
        """
        pass


class TavilySearchProvider(SearchProvider):
    """
    Tavily search provider.

    Tavily is designed for AI applications and returns clean, structured content.
    Uses the tavily-python SDK.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy-initialize Tavily client."""
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict | None = None,
    ) -> list[SearchResult]:
        """Execute search via Tavily API."""
        import asyncio

        kwargs = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_raw_content": False,
        }

        if allowed_domains:
            kwargs["include_domains"] = allowed_domains
        if blocked_domains:
            kwargs["exclude_domains"] = blocked_domains

        logger.info(
            f"[WebSearch/Tavily] Searching: {query!r} (max_results={max_results})"
        )

        # Tavily SDK is synchronous, run in thread pool
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.search(**kwargs)
        )

        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    page_age=item.get("published_date"),
                )
            )

        logger.info(f"[WebSearch/Tavily] Got {len(results)} results")
        return results


class BraveSearchProvider(SearchProvider):
    """
    Brave Search provider.

    Uses the Brave Search API via httpx.
    """

    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Any | None = None

    @property
    def client(self):
        """Lazy-initialize httpx client for connection reuse."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict | None = None,
    ) -> list[SearchResult]:
        """Execute search via Brave Search API."""
        # Build query with domain filtering via site: prefix
        search_query = query
        if allowed_domains:
            site_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
            search_query = f"({site_filter}) {query}"

        params = {
            "q": search_query,
            "count": max_results,
        }

        if user_location and user_location.get("country"):
            params["country"] = user_location["country"]

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        logger.info(
            f"[WebSearch/Brave] Searching: {search_query!r} (count={max_results})"
        )

        response = await self.client.get(self.ENDPOINT, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Domain filtering is handled by DomainFilter post-processing
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=item.get("description", ""),
                    page_age=item.get("page_age"),
                )
            )

        logger.info(f"[WebSearch/Brave] Got {len(results)} results")
        return results[:max_results]


class AgentCoreSearchProvider(SearchProvider):
    """
    Amazon Bedrock AgentCore Gateway Web Search provider.

    Invokes the managed AgentCore WebSearch MCP tool through an IAM-authenticated
    Gateway endpoint. AgentCore currently supports the Web Search connector in
    us-east-1 only.
    """

    TOOL_NAME = "WebSearch"
    SERVICE_NAME = "bedrock-agentcore"

    def __init__(self, gateway_url: str, region: str = "us-east-1"):
        if not gateway_url:
            raise ValueError(
                "AgentCore Gateway URL is required. Set AGENTCORE_GATEWAY_URL."
            )
        self.gateway_url = gateway_url
        self.region = region
        self._client: Any | None = None

    @property
    def client(self):
        """Lazy-initialize httpx client for connection reuse."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _signed_headers(self, payload: bytes) -> dict[str, str]:
        """Create SigV4 headers for an AgentCore Gateway MCP request."""
        import botocore.session
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("AWS credentials are required for AgentCore web search")

        request = AWSRequest(
            method="POST",
            url=self.gateway_url,
            data=payload,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        )
        SigV4Auth(
            credentials.get_frozen_credentials(), self.SERVICE_NAME, self.region
        ).add_auth(request)
        return dict(request.headers.items())

    @staticmethod
    def _json_from_response_text(text: str) -> dict[str, Any]:
        """Parse direct JSON or an SSE stream carrying JSON-RPC data frames."""
        stripped = text.strip()
        if not stripped:
            raise ValueError("AgentCore Gateway returned an empty response")

        if stripped.startswith("{"):
            return json.loads(stripped)

        data_frames = []
        for line in stripped.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                value = line.removeprefix("data:").strip()
                if value and value != "[DONE]":
                    data_frames.append(value)

        if not data_frames:
            raise ValueError("AgentCore Gateway response did not contain JSON data")
        return json.loads(data_frames[-1])

    @staticmethod
    def _extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract WebSearch results from a JSON-RPC MCP response."""
        if "error" in payload:
            message = payload.get("error", {}).get("message", "unknown MCP error")
            raise ValueError(f"AgentCore Gateway web search failed: {message}")

        result = payload.get("result", payload)
        if result.get("isError"):
            raise ValueError("AgentCore Gateway WebSearch returned an error result")

        for item in result.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text", "")
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            results = parsed.get("results")
            if isinstance(results, list):
                return [r for r in results if isinstance(r, dict)]

        return []

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict | None = None,
    ) -> list[SearchResult]:
        """Execute search via AgentCore Gateway WebSearch MCP tool."""
        del allowed_domains, blocked_domains, user_location

        search_query = query[:200]
        if len(query) > 200:
            logger.info(
                "[WebSearch/AgentCore] Truncated query from %s to 200 characters",
                len(query),
            )

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": self.TOOL_NAME,
                "arguments": {
                    "query": search_query,
                    "maxResults": max(1, min(max_results, 25)),
                },
            },
        }
        payload = json.dumps(request).encode("utf-8")
        headers = self._signed_headers(payload)

        logger.info(
            f"[WebSearch/AgentCore] Searching: {query!r} (max_results={max_results})"
        )
        response = await self.client.post(
            self.gateway_url,
            content=payload,
            headers=headers,
        )
        if response.status_code >= 400:
            response.raise_for_status()

        data = self._json_from_response_text(response.text)
        raw_results = self._extract_results(data)
        results = [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("text")
                or item.get("content")
                or item.get("snippet")
                or "",
                page_age=item.get("publishedDate") or item.get("page_age"),
            )
            for item in raw_results
        ]

        logger.info(f"[WebSearch/AgentCore] Got {len(results)} results")
        return results[:max_results]


def create_search_provider(
    provider: str | None = None,
    api_key: str | None = None,
) -> SearchProvider:
    """
    Create a search provider instance based on configuration.

    Args:
        provider: Provider name ('tavily' or 'brave'). Defaults to settings.
        api_key: API key. Defaults to settings.

    Returns:
        SearchProvider instance

    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    provider = (provider or settings.web_search_provider).lower()

    if provider == "agentcore":
        return AgentCoreSearchProvider(
            gateway_url=settings.agentcore_gateway_url or "",
            region=settings.agentcore_gateway_region,
        )

    api_key = api_key or settings.web_search_api_key
    if not api_key:
        raise ValueError(
            "Web search API key is required. Set WEB_SEARCH_API_KEY environment variable."
        )

    if provider == "tavily":
        return TavilySearchProvider(api_key=api_key)
    elif provider == "brave":
        return BraveSearchProvider(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown search provider: {provider}. Use 'tavily', 'brave', or 'agentcore'."
        )
