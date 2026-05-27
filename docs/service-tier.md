# Service Tier Configuration

The Bedrock Service Tier feature allows you to balance between cost and latency. This proxy service fully supports this feature with flexible configuration options.

## Available Tiers

| Tier | Description | Latency | Cost | Claude Support |
|------|-------------|---------|------|----------------|
| `default` | Standard service tier | Standard | Standard | ✅ |
| `flex` | Flexible tier for batch processing | Higher (up to 24h) | Lower | ❌ |
| `priority` | Priority tier for real-time apps | Lower | Higher | ❌ |
| `reserved` | Reserved capacity tier | Stable | Prepaid | ✅ |

## Configuration Methods

### Per API Key Configuration

System default is `default`. You can create API keys with different service tiers for different users or purposes:

```bash
# Create an API key with flex tier (for non-real-time batch processing)
./scripts/create-api-key.sh -u batch-user -n "Batch Processing Key" -t flex

# Create an API key with priority tier (for real-time applications)
./scripts/create-api-key.sh -u realtime-user -n "Realtime App Key" -t priority
```

### Priority Rules

Service tier is determined by the following priority:
1. **API Key Configuration** (highest priority) - if the API key has a specified service tier
2. **System Default** - `default`

## Automatic Fallback Mechanism

When the specified service tier is not supported by the target model, the proxy service will **automatically fall back** to `default` tier and retry the request:

```
Request (flex tier) → Claude model → flex not supported → Auto fallback to default → Success
```

This ensures that requests will not fail even if an incompatible service tier is configured.

## Usage Recommendations

| Scenario | Recommended Tier | Description |
|----------|-----------------|-------------|
| Real-time chat/conversation | `default` or `priority` | Requires low latency response |
| Batch data processing | `flex` | Can tolerate higher latency, saves cost |
| Code generation/dev tools | `default` | Balance between latency and cost |
| Production critical apps | `reserved` | Requires stable capacity guarantee |

## Model Compatibility

| Model | default | flex | priority | reserved |
|-------|---------|------|----------|----------|
| Claude Series | ✅ | ❌ | ❌ | ✅ |
| Qwen Series | ✅ | ✅ | ✅ | ✅ |
| DeepSeek Series | ✅ | ✅ | ✅ | ✅ |
| Nova Series | ✅ | ✅ | ✅ | ✅ |
| MiniMax Series | ✅ | ✅ | ✅ | ✅ |

> **Note**: Specific model support for service tiers may change with AWS Bedrock updates. Please refer to the [AWS Official Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-service-tiers.html) for the latest information.

## Environment Variable

```bash
# Default service tier: 'default', 'flex', 'priority', 'reserved'
DEFAULT_SERVICE_TIER=default
```
