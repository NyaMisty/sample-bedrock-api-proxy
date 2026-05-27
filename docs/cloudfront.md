# CloudFront HTTPS Encryption

The proxy includes a built-in CloudFront distribution that provides HTTPS encryption for all API traffic. It uses AWS-managed `*.cloudfront.net` certificates — **no custom domain or ACM certificate required**.

## Architecture

```
Client (Anthropic SDK)
    │
    ▼ HTTPS (443)
CloudFront (*.cloudfront.net)
    │  - AWS-managed TLS certificate
    │  - Attaches X-CloudFront-Secret header
    │  - HSTS security response header
    │
    ▼ HTTP (80, internal)
ALB (existing)
    │  - Validates X-CloudFront-Secret
    │  - Rejects direct access (returns 403)
    │
    ▼ HTTP (8000)
ECS Tasks (unchanged)
```

## Enabling CloudFront

CloudFront is **disabled by default** for both `dev` and `prod` environments. Enable it via environment variable:

```bash
# Enable CloudFront HTTPS distribution
ENABLE_CLOUDFRONT=true ./scripts/deploy.sh -e prod -r us-west-2 -p arm64

# Deployment output
# Access URLs:
#   API Proxy (HTTPS): https://d1234567890.cloudfront.net
#   Admin Portal (HTTPS): https://d1234567890.cloudfront.net/admin/
```

## Client Configuration

With CloudFront enabled, update `ANTHROPIC_BASE_URL` to the HTTPS URL:

```bash
export CLAUDE_CODE_USE_BEDROCK=0
export ANTHROPIC_BASE_URL=https://d1234567890.cloudfront.net
export ANTHROPIC_API_KEY=sk-xxxx
```

## Security Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **HTTPS Encryption** | End-to-end TLS encryption from client to CloudFront, protecting API keys and request data |
| **ALB Access Control** | ALB only accepts requests with the `X-CloudFront-Secret` header, rejects direct access |
| **HSTS** | Forces browsers to use HTTPS (`Strict-Transport-Security: max-age=31536000`) |
| **Auto-Generated Secret** | Secrets Manager automatically generates a 32-character random validation key |

## Streaming vs Non-Streaming Considerations

| Mode | CloudFront Behavior | Recommendation |
|------|---------------------|----------------|
| **Streaming** (`"stream": true`) | CloudFront natively supports SSE, forwards in real-time. Timeout only affects time-to-first-byte (`message_start` typically arrives within seconds) | **Recommended** |
| **Non-streaming** | Timeout covers the entire response generation time. Default 60 seconds, returns 504 on timeout | Switch to streaming for long responses |

> **Tip**: To support non-streaming requests longer than 60 seconds, request a CloudFront Origin Read Timeout quota increase (up to 180 seconds) via the AWS Support Console.

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enableCloudFront` | boolean | `false` | Enable CloudFront HTTPS distribution |
| `cloudFrontOriginReadTimeout` | number | `60` | Origin read timeout (seconds), default max 60s, up to 180s with quota increase |

## Disabling CloudFront

Set `enableCloudFront: false` (or `ENABLE_CLOUDFRONT=false`) and redeploy to fall back to HTTP-only direct ALB access.
