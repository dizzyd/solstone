# Provider Configuration

Route LLM API calls to different providers based on context prefixes.

## Config File

Location: `JOURNAL_PATH/config/providers.json`

If the file doesn't exist, the system uses defaults (Google Gemini Flash for everything).

## Structure

```json
{
  "defaults": {
    "provider": "google",
    "model": "gemini-3-flash-preview"
  },
  "prefixes": {
    "insight.activity*": {
      "provider": "digitalocean",
      "model": "openai-gpt-oss-120b"
    },
    "describe.*": {
      "provider": "google",
      "model": "gemini-2.5-flash-lite"
    }
  }
}
```

**Note:** Use `insight.activity*` (not `insight.activity.*`) to match both `insight.activity` and `insight.activity.markdown`.

## Context Matching

Contexts are dot-separated strings like `insight.activity.markdown` or `describe.frame`.

Matching priority:
1. Exact match on full context string
2. Glob pattern match (most specific wins - longer base pattern preferred)
3. Fall back to `defaults`

## Available Providers

| Provider | ID | Notes |
|----------|-----|-------|
| Google Gemini | `google` | Supports thinking, caching, vision |
| OpenAI | `openai` | Supports thinking, vision |
| DigitalOcean Gradient | `digitalocean` | OpenAI-compatible, no thinking/vision |
| Amazon Bedrock | `bedrock` | Supports vision (Claude/Nova models) |

## Available Models

### Google (`google`)
- `gemini-3-flash-preview` - Fast, default
- `gemini-3-pro-preview` - More capable
- `gemini-2.5-flash-lite` - Cheapest

### OpenAI (`openai`)
- `gpt-5.2` - Most capable
- `gpt-5-mini` - Balanced
- `gpt-5-nano` - Fastest/cheapest

### DigitalOcean (`digitalocean`)
- `openai-gpt-oss-120b` - GPT-OSS 120B
- `openai-gpt-oss-20b` - GPT-OSS 20B
- `llama3.3-70b-instruct` - Llama 3.3 70B
- `deepseek-r1-distill-llama-70b` - DeepSeek R1
- `alibaba-qwen3-32b` - Qwen3 32B

### Bedrock (`bedrock`)
- `anthropic.claude-3-5-sonnet-20241022-v2:0` - Claude 3.5 Sonnet
- `amazon.nova-pro-v1:0` - Nova Pro
- `amazon.nova-lite-v1:0` - Nova Lite
- `meta.llama3-1-70b-instruct-v1:0` - Llama 3.1 70B

## Common Context Prefixes

| Context Pattern | Used By |
|-----------------|---------|
| `insight.<type>.markdown` | think-insight (activity, screen, meetings, etc.) |
| `describe.frame` | observe-describe |
| `enrich.*` | observe enrich pipeline |
| `agent.<persona>.*` | muse-agents |

## Code References

- Provider abstraction: `think/providers.py`
- Provider implementations: `think/providers_*.py`
- Model constants: `think/models.py`
- Unified generate API: `think/models.py:generate()` and `agenerate()`

## Applying Changes

The config is cached on first load. To apply changes without restarting:

```python
from think.providers import clear_config_cache
clear_config_cache()
```
