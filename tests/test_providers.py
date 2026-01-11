# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.providers module."""

import pytest

from think.providers import (
    DEFAULT_CONFIG,
    clear_config_cache,
    get_provider,
    load_provider_config,
    resolve_provider,
)


@pytest.fixture
def temp_journal(tmp_path, monkeypatch):
    """Create a temporary journal directory with config."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    config_dir = journal_path / "config"
    config_dir.mkdir()

    # Patch get_journal in think.utils (where it's imported from)
    monkeypatch.setattr("think.utils.get_journal", lambda: str(journal_path))

    # Clear any cached config from previous tests
    clear_config_cache()

    yield journal_path

    # Clean up cache after test
    clear_config_cache()


@pytest.fixture
def sample_config():
    """Sample provider configuration."""
    return {
        "defaults": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prefixes": {
            "describe.*": {"provider": "google", "model": "gemini-2.5-flash-lite"},
            "describe.meeting": {"provider": "openai", "model": "gpt-5-mini"},
            "describe.code": {"provider": "openai", "model": "gpt-5.2"},
            "insight.*": {"provider": "google", "model": "gemini-3-pro-preview"},
            "detect.*": {"provider": "google", "model": "gemini-3-flash-preview"},
        },
    }


class TestLoadProviderConfig:
    """Tests for load_provider_config function."""

    def test_returns_defaults_when_no_file(self, temp_journal):
        """Should return default config when providers.json doesn't exist."""
        config = load_provider_config()

        assert config == DEFAULT_CONFIG
        assert config["defaults"]["provider"] == "google"
        assert config["defaults"]["model"] == "gemini-3-flash-preview"

    def test_loads_config_from_file(self, temp_journal, sample_config):
        """Should load config from providers.json."""
        import json

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(json.dumps(sample_config))

        config = load_provider_config()

        assert config["defaults"]["provider"] == "google"
        assert "describe.*" in config["prefixes"]
        assert config["prefixes"]["describe.meeting"]["provider"] == "openai"

    def test_handles_invalid_json(self, temp_journal):
        """Should return defaults on invalid JSON."""
        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text("not valid json {{{")

        config = load_provider_config()

        assert config == DEFAULT_CONFIG

    def test_adds_missing_keys(self, temp_journal):
        """Should add missing defaults/prefixes keys."""
        import json

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(
            json.dumps({"prefixes": {"test.*": {"provider": "openai"}}})
        )

        config = load_provider_config()

        assert "defaults" in config
        assert "prefixes" in config


class TestResolveProvider:
    """Tests for resolve_provider function."""

    def test_exact_match(self, temp_journal, sample_config):
        """Should match exact context string."""
        import json

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(json.dumps(sample_config))

        result = resolve_provider("describe.meeting")

        assert result.provider == "openai"
        assert result.model == "gpt-5-mini"
        assert result.context == "describe.meeting"

    def test_glob_match(self, temp_journal, sample_config):
        """Should match glob patterns."""
        import json

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(json.dumps(sample_config))

        result = resolve_provider("describe.frame")

        assert result.provider == "google"
        assert result.model == "gemini-2.5-flash-lite"

    def test_most_specific_wins(self, temp_journal, sample_config):
        """Exact match should win over glob pattern."""
        import json

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(json.dumps(sample_config))

        # describe.code is exact match, should beat describe.*
        result = resolve_provider("describe.code")

        assert result.provider == "openai"
        assert result.model == "gpt-5.2"

    def test_falls_back_to_defaults(self, temp_journal, sample_config):
        """Should fall back to defaults when no match."""
        import json

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(json.dumps(sample_config))

        result = resolve_provider("unknown.context")

        assert result.provider == "google"
        assert result.model == "gemini-3-flash-preview"

    def test_returns_defaults_without_config(self, temp_journal):
        """Should return defaults when no config file exists."""
        result = resolve_provider("any.context")

        assert result.provider == "google"
        assert result.model == "gemini-3-flash-preview"

    def test_handles_nested_wildcards(self, temp_journal):
        """Should handle nested patterns correctly."""
        import json

        config = {
            "defaults": {"provider": "google", "model": "default-model"},
            "prefixes": {
                "insight.*": {"provider": "google", "model": "insight-model"},
                "insight.*.markdown": {
                    "provider": "openai",
                    "model": "markdown-model",
                },
            },
        }

        config_path = temp_journal / "config" / "providers.json"
        config_path.write_text(json.dumps(config))

        # More specific pattern should win
        result = resolve_provider("insight.decisions.markdown")
        assert result.model == "markdown-model"

        # Less specific should match
        result = resolve_provider("insight.todos")
        assert result.model == "insight-model"


class TestGetProvider:
    """Tests for get_provider function."""

    def test_returns_google_provider(self):
        """Should return GoogleProvider for 'google'."""
        from think.providers_google import GoogleProvider

        provider = get_provider("google")

        assert isinstance(provider, GoogleProvider)

    def test_returns_openai_provider(self):
        """Should return OpenAIProvider for 'openai'."""
        from think.providers_openai import OpenAIProvider

        provider = get_provider("openai")

        assert isinstance(provider, OpenAIProvider)

    def test_returns_digitalocean_provider(self):
        """Should return DigitalOceanProvider for 'digitalocean'."""
        from think.providers_digitalocean import DigitalOceanProvider

        provider = get_provider("digitalocean")

        assert isinstance(provider, DigitalOceanProvider)

    def test_raises_for_unknown(self):
        """Should raise ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown")


class TestProviderCapabilities:
    """Tests for provider capability methods."""

    def test_google_supports_vision(self):
        """Google provider should support vision."""
        provider = get_provider("google")
        assert provider.supports_vision() is True

    def test_google_supports_thinking(self):
        """Google provider should support thinking."""
        provider = get_provider("google")
        assert provider.supports_thinking() is True

    def test_google_supports_caching(self):
        """Google provider should support caching."""
        provider = get_provider("google")
        assert provider.supports_caching() is True

    def test_openai_supports_vision(self):
        """OpenAI provider should support vision."""
        provider = get_provider("openai")
        assert provider.supports_vision() is True

    def test_openai_no_thinking(self):
        """OpenAI provider should not support thinking_budget."""
        provider = get_provider("openai")
        assert provider.supports_thinking() is False

    def test_openai_no_caching(self):
        """OpenAI provider should not support cached_content."""
        provider = get_provider("openai")
        assert provider.supports_caching() is False

    def test_digitalocean_no_vision(self):
        """DigitalOcean provider should not support vision (conservative)."""
        provider = get_provider("digitalocean")
        assert provider.supports_vision() is False

    def test_digitalocean_no_thinking(self):
        """DigitalOcean provider should not support thinking_budget."""
        provider = get_provider("digitalocean")
        assert provider.supports_thinking() is False

    def test_digitalocean_no_caching(self):
        """DigitalOcean provider should not support cached_content."""
        provider = get_provider("digitalocean")
        assert provider.supports_caching() is False


class TestClearConfigCache:
    """Tests for clear_config_cache function."""

    def test_cache_is_cleared(self, temp_journal):
        """Should clear the cached config."""
        import json

        config_path = temp_journal / "config" / "providers.json"

        # Write initial config
        config_path.write_text(
            json.dumps(
                {
                    "defaults": {"provider": "google", "model": "model-v1"},
                    "prefixes": {},
                }
            )
        )

        result1 = resolve_provider("any.context")
        assert result1.model == "model-v1"

        # Update config
        config_path.write_text(
            json.dumps(
                {
                    "defaults": {"provider": "openai", "model": "model-v2"},
                    "prefixes": {},
                }
            )
        )

        # Should still return cached value
        result2 = resolve_provider("any.context")
        assert result2.model == "model-v1"

        # Clear cache
        clear_config_cache()

        # Now should return new value
        result3 = resolve_provider("any.context")
        assert result3.model == "model-v2"


class TestContextWindow:
    """Tests for get_context_window method."""

    def test_google_context_window(self):
        """Google provider should return correct context windows."""
        provider = get_provider("google")
        assert provider.get_context_window("gemini-3-flash-preview") == 1000000
        assert provider.get_context_window("gemini-3-pro-preview") == 2000000
        # Unknown model should return default
        assert provider.get_context_window("unknown-model") == 1000000

    def test_openai_context_window(self):
        """OpenAI provider should return correct context windows."""
        provider = get_provider("openai")
        assert provider.get_context_window("gpt-5") == 128000
        assert provider.get_context_window("o1") == 200000
        # Unknown model should return default
        assert provider.get_context_window("unknown-model") == 128000

    def test_digitalocean_context_window(self):
        """DigitalOcean provider should return correct context windows."""
        provider = get_provider("digitalocean")
        assert provider.get_context_window("openai-gpt-oss-120b") == 131072
        assert provider.get_context_window("llama3.3-70b-instruct") == 128000
        # Unknown model should return default
        assert provider.get_context_window("unknown-model") == 128000


class TestTokenEstimation:
    """Tests for estimate_tokens and check_content_fits methods."""

    def test_estimate_tokens_string(self):
        """Should estimate tokens for string content."""
        provider = get_provider("google")
        # 400 characters should be ~100 tokens
        content = "a" * 400
        assert provider.estimate_tokens(content) == 100

    def test_estimate_tokens_list(self):
        """Should estimate tokens for list content."""
        provider = get_provider("google")
        content = ["a" * 400, "b" * 800]  # 100 + 200 tokens
        assert provider.estimate_tokens(content) == 300

    def test_estimate_tokens_empty(self):
        """Should return 0 for empty content."""
        provider = get_provider("google")
        assert provider.estimate_tokens("") == 0
        assert provider.estimate_tokens([]) == 0

    def test_check_content_fits_true(self):
        """Should return True when content fits."""
        provider = get_provider("google")
        content = "a" * 4000  # ~1000 tokens
        fits, estimated, available = provider.check_content_fits(
            content, "gemini-3-flash-preview"
        )
        assert fits is True
        assert estimated == 1000
        assert available == 999000  # 1M - 1000 buffer

    def test_check_content_fits_false(self):
        """Should return False when content exceeds limit."""
        provider = get_provider("digitalocean")
        # Create content larger than 128K context window
        content = "a" * (130000 * 4)  # ~130K tokens
        fits, estimated, available = provider.check_content_fits(
            content, "llama3-8b-instruct"
        )
        assert fits is False
        assert estimated == 130000
        assert available == 127000  # 128K - 1000 buffer
