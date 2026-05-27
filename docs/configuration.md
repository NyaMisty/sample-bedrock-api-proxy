# Configuration Reference

Configuration is managed through environment variables. See `.env.example` for all options.

## Application Settings

```bash
APP_NAME=Anthropic-Bedrock API Proxy
ENVIRONMENT=development  # development, staging, production
LOG_LEVEL=INFO
```

## AWS Settings

```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

## Authentication

```bash
REQUIRE_API_KEY=True
MASTER_API_KEY=sk-your-master-key
API_KEY_HEADER=x-api-key
```

## Rate Limiting

```bash
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=1000  # requests per window
RATE_LIMIT_WINDOW=60     # window in seconds
```

## Feature Flags

```bash
ENABLE_TOOL_USE=True
ENABLE_EXTENDED_THINKING=True
ENABLE_DOCUMENT_SUPPORT=True
PROMPT_CACHING_ENABLED=False
ENABLE_PROGRAMMATIC_TOOL_CALLING=True  # Requires Docker
ENABLE_WEB_SEARCH=True                # Requires search provider API key
ENABLE_WEB_FETCH=True                 # Default: enabled, uses httpx (no key needed)
ENABLE_OPENAI_COMPAT=True            # Use OpenAI Chat Completions API (non-Claude models)
ENABLE_OPENAI_PASSTHROUGH=True       # Mount /openai/v1/* passthrough endpoints
DEFAULT_CACHE_TTL=1h                  # Proxy default cache TTL (optional: '5m' or '1h')
```

## OpenAI-Compatible API

```bash
# Enable OpenAI-compatible API (only affects non-Claude models; default: False)
ENABLE_OPENAI_COMPAT=True

# Enable OpenAI SDK-compatible passthrough endpoints under /openai/v1/* (default: False)
ENABLE_OPENAI_PASSTHROUGH=True

# Bedrock Mantle API Key
BEDROCK_API_KEY=your-bedrock-api-key

# Bedrock Mantle endpoint URL
MANTLE_ENDPOINT_URL=https://bedrock-mantle.us-east-1.api.aws/v1

# thinking → reasoning mapping thresholds
OPENAI_COMPAT_THINKING_HIGH_THRESHOLD=10000    # budget_tokens >= this → effort=high
OPENAI_COMPAT_THINKING_MEDIUM_THRESHOLD=4000   # budget_tokens >= this → effort=medium
```

`cdk/scripts/deploy.sh` requires `BEDROCK_API_KEY` when `ENABLE_OPENAI_COMPAT=true`.

### OpenAI Passthrough Usage

When `ENABLE_OPENAI_PASSTHROUGH=True`, OpenAI SDK clients can use the proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-proxy.example.com/openai/v1",
    api_key="sk-your-proxy-api-key",
)

response = client.responses.create(
    model="openai.gpt-oss-120b",
    tools=[{"type": "web_search"}],
    input="Search the web for one current positive technology news story.",
)
print(response.output_text)
```

### Responses API State Storage

For proxy-managed Responses API web search, `previous_response_id` is stored in DynamoDB so ECS deployments with multiple tasks can continue stateful conversations across task boundaries.

```bash
# Optional Responses web_search state settings
DYNAMODB_RESPONSE_CONTEXT_TABLE=anthropic-proxy-response-context
RESPONSE_CONTEXT_TTL_SECONDS=3600
RESPONSE_CONTEXT_CHUNK_SIZE_BYTES=262144
RESPONSE_CONTEXT_MAX_BYTES=1048576
RESPONSE_CONTEXT_MAX_CHUNKS=8
```

## Web Search

```bash
ENABLE_WEB_SEARCH=True
WEB_SEARCH_PROVIDER=tavily          # 'tavily' (recommended) or 'brave'
WEB_SEARCH_API_KEY=tvly-your-api-key
WEB_SEARCH_MAX_RESULTS=5            # Max results per search query (default: 5)
WEB_SEARCH_DEFAULT_MAX_USES=10      # Default max searches per request (default: 10)
```

### Usage Example

```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-your-api-key", base_url="http://localhost:8000")

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
            "allowed_domains": ["python.org", "docs.python.org"],
        }
    ],
    messages=[{"role": "user", "content": "What are the new features in Python 3.13?"}]
)
```

### Search Provider Comparison

| Provider | Features | API Key |
|----------|----------|---------|
| **Tavily** (recommended) | AI-optimized, returns structured content | [tavily.com](https://tavily.com) |
| **Brave Search** | General-purpose search API | [brave.com/search/api](https://brave.com/search/api/) |

### Tool Type Comparison

| Tool Type | Description | Requires Docker |
|-----------|-------------|----------------|
| `web_search_20250305` | Basic web search | No |
| `web_search_20260209` | Dynamic filtering (Claude can write code to filter results) | **Yes** (EC2 launch type) |

## Web Fetch

```bash
ENABLE_WEB_FETCH=True                           # Default: enabled
WEB_FETCH_DEFAULT_MAX_USES=20                   # Default max fetches per request
WEB_FETCH_DEFAULT_MAX_CONTENT_TOKENS=100000     # Default max content tokens per fetch
```

### Usage Example

```python
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    tools=[
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 5,
            "max_content_tokens": 50000,
        }
    ],
    messages=[{"role": "user", "content": "Fetch https://docs.python.org/3/whatsnew/3.13.html and summarize"}],
    extra_headers={"anthropic-beta": "web-fetch-2025-09-10"},
)
```

### Web Search vs Web Fetch

| Dimension | Web Search | Web Fetch |
|-----------|-----------|-----------|
| **Input** | Search keywords (`query`) | Specific URL (`url`) |
| **Output** | Multiple search result snippets | Full page content of a single URL |
| **Provider** | Tavily / Brave (API key required) | httpx direct fetch (**no key needed**) |
| **PDF Support** | No | Yes (base64 passthrough) |

## Programmatic Tool Calling (PTC)

```bash
ENABLE_PROGRAMMATIC_TOOL_CALLING=True
PTC_SANDBOX_IMAGE=python:3.11-slim      # Docker sandbox image
PTC_SESSION_TIMEOUT=270                   # Session timeout (seconds)
PTC_EXECUTION_TIMEOUT=60                  # Code execution timeout (seconds)
PTC_MEMORY_LIMIT=256m                     # Container memory limit
PTC_NETWORK_DISABLED=True                 # Disable network in container (security)
```

## Prompt Cache TTL

```bash
# Proxy-level default cache TTL (optional, defaults to Anthropic's 5m if not set)
DEFAULT_CACHE_TTL=1h
```

### TTL Priority

| Priority | Source | Description |
|----------|--------|-------------|
| 1 (Highest) | API Key `cache_ttl` | Forced override, rewrites ALL `cache_control` blocks |
| 2 | Client request `cache_control.ttl` | TTL specified by client in request |
| 3 | `DEFAULT_CACHE_TTL` env var | Proxy-level default |
| 4 (Lowest) | No TTL | Anthropic/Bedrock default (5 minutes) |

### Billing

| TTL | Cache Write Price | Description |
|-----|------------------|-------------|
| 5m (default) | 1.25x input price | Standard cache write rate |
| 1h | 2.0x input price | Extended caching requires higher write cost |

## OpenTelemetry Tracing

```bash
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-otel-endpoint
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic xxxxx
OTEL_SERVICE_NAME=anthropic-bedrock-proxy
OTEL_TRACE_CONTENT=false
OTEL_TRACE_SAMPLING_RATIO=1.0
```

See [OpenTelemetry Tracing Guide](./otel-tracing.md) for detailed setup instructions.

## Beta Header Mapping

The proxy supports automatic mapping of Anthropic beta headers to Bedrock beta headers.

**Default Mapping:**

| Anthropic Beta Header | Bedrock Beta Headers |
|----------------------|---------------------|
| `advanced-tool-use-2025-11-20` | `tool-examples-2025-10-29`, `tool-search-tool-2025-10-19` |

**Supported Models:** Claude Opus 4.5 (`claude-opus-4-5-20251101`)

**Configuration:** Modify `BETA_HEADER_MAPPING` in `.env` or `app/core/config.py`. Add keywords to `BETA_HEADER_SUPPORTED_MODELS` (substring, case-insensitive match).

## Service Tier

```bash
DEFAULT_SERVICE_TIER=default  # 'default', 'flex', 'priority', 'reserved'
```

See [Service Tier Guide](./service-tier.md) for detailed information.
