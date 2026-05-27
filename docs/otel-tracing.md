# OpenTelemetry Distributed Tracing (LLM Observability)

The proxy has built-in OpenTelemetry tracing support, exporting detailed LLM call information to any OTEL-compatible observability backend for:

- **Token Usage Tracking**: Input/output/cache token statistics per request
- **Latency Analysis**: End-to-end latency, Bedrock API call latency, streaming response duration
- **Session Correlation**: Correlate multiple requests in the same conversation via `x-session-id` header
- **Tool Call Tracing**: Record each tool call's name and ID
- **PTC Code Execution Tracing**: Track Programmatic Tool Calling execution flow
- **Error Diagnostics**: Automatic exception recording and error status

## Trace Hierarchy (Turn-Based Agent Loop)

```
Trace "chat claude-sonnet-4-5-20250929"
  ├── Turn 1 (input=user_msg, output=assistant_response)
  │     ├── gen_ai.chat (model, tokens, usage)
  │     ├── Tool: Read (input=tool_input)
  │     └── Tool: Edit (input=tool_input)
  ├── Turn 2
  │     ├── gen_ai.chat
  │     └── Tool: Bash
  └── Turn 3
        └── gen_ai.chat (final text response, no tools)
```

Each HTTP request in an agent loop maps to a **Turn** span containing:
- A `gen_ai.chat` generation span with model, token usage, and latency
- Tool spans for each tool_use block in the response
- Structured input/output attributes for Langfuse rendering

## Key Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.request.model` | Request model | `claude-sonnet-4-5-20250929` |
| `gen_ai.usage.input_tokens` | Input tokens | `1500` |
| `gen_ai.usage.output_tokens` | Output tokens | `350` |
| `gen_ai.response.finish_reasons` | Stop reason | `["end_turn"]` |
| `gen_ai.conversation.id` | Session ID | `session-abc123` |
| `langfuse.observation.usage_details` | Full usage JSON with cache tokens | `{"input":1500,"output":350,"cache_read_input_tokens":800}` |
| `proxy.api_key_hash` | API key hash (privacy-safe) | `a1b2c3d4...` |

## Connecting to Langfuse Cloud

[Langfuse](https://langfuse.com) is an open-source LLM observability platform with native OTEL support.

**1. Get Langfuse Credentials**

Log in to [Langfuse Cloud](https://us.cloud.langfuse.com), go to project Settings → API Keys to get your Public Key and Secret Key.

**2. Generate Base64 Auth String**

```bash
echo -n "your-public-key:your-secret-key" | base64
```

**3. Configure Environment Variables**

```bash
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://us.cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64-string-from-step-2>
OTEL_SERVICE_NAME=anthropic-bedrock-proxy
OTEL_TRACE_CONTENT=true
```

**4. Start Service and Send Requests**

```bash
# Start service
uv run uvicorn app.main:app --reload

# Send request (with session ID for trace correlation)
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-key" \
  -H "x-session-id: my-test-session" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**5. View Traces in Langfuse**

Log in to Langfuse Cloud and navigate to the Traces page to see:
- Complete span hierarchy and timeline
- Token usage and cache hit statistics
- Conversations grouped by Session ID
- Model, latency, and cost metrics

## Connecting to Other OTEL Backends

**Jaeger (Local Debugging):**

```bash
# Start Jaeger
docker run -d -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one

# Configure proxy
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=anthropic-bedrock-proxy

# View traces: http://localhost:16686
```

**Grafana Tempo:**

```bash
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-tempo-endpoint
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <credentials>
```

## Content Tracing (Optional)

By default, tracing **does not record** actual request and response content (as it may contain sensitive information). To enable content tracing for debugging:

```bash
# Enable content tracing (records prompt and completion content, beware of PII risks)
OTEL_TRACE_CONTENT=true
```

When enabled, trace data will include:
- Structured trace input as JSON (system prompt, tools with schemas, user message)
- Current turn's messages only (not full history) in gen_ai.chat spans
- Response text and tool call details

## CDK Deployment with Tracing

When deploying to ECS via CDK, you can enable tracing via environment variables at deploy time — **no code changes required**:

```bash
# Example with Langfuse
ENABLE_TRACING=true \
OTEL_EXPORTER_OTLP_ENDPOINT=https://us.cloud.langfuse.com/api/public/otel \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $(echo -n 'pk-xxx:sk-xxx' | base64)" \
OTEL_SERVICE_NAME=anthropic-bedrock-proxy-prod \
OTEL_TRACE_CONTENT=true \
OTEL_TRACE_SAMPLING_RATIO=1.0 \
./scripts/deploy.sh -e prod -r us-west-2 -p arm64 -l ec2
```

## Environment Variables

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ENABLE_TRACING` | Enable tracing | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP export endpoint | none |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Protocol (`http/protobuf` / `grpc`) | `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers | none |
| `OTEL_SERVICE_NAME` | Service name | none |
| `OTEL_TRACE_CONTENT` | Record prompt/completion content | `false` |
| `OTEL_TRACE_SAMPLING_RATIO` | Sampling ratio (0.0-1.0) | `1.0` |

> **Priority**: Environment variables > `cdk/config/config.ts` settings > defaults

## Sampling Configuration

For high-traffic scenarios, control trace data volume with sampling:

```bash
# 50% sampling (sample 1 out of every 2 requests)
OTEL_TRACE_SAMPLING_RATIO=0.5

# 10% sampling (for high-traffic production)
OTEL_TRACE_SAMPLING_RATIO=0.1

# Full sampling (default, for development and low-traffic environments)
OTEL_TRACE_SAMPLING_RATIO=1.0
```
