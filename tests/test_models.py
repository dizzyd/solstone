"""Tests for think.models module."""

from think.models import (
    CLAUDE_HAIKU_4,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    GEMINI_FLASH,
    GEMINI_LITE,
    GEMINI_PRO,
    GPT_5,
    GPT_5_MINI,
    GPT_5_NANO,
    get_model_provider,
)


def test_get_model_provider_gemini():
    """Test provider detection for Gemini models."""
    assert get_model_provider(GEMINI_PRO) == "google"
    assert get_model_provider(GEMINI_FLASH) == "google"
    assert get_model_provider(GEMINI_LITE) == "google"


def test_get_model_provider_gpt():
    """Test provider detection for GPT models."""
    assert get_model_provider(GPT_5) == "openai"
    assert get_model_provider(GPT_5_MINI) == "openai"
    assert get_model_provider(GPT_5_NANO) == "openai"


def test_get_model_provider_claude():
    """Test provider detection for Claude models."""
    assert get_model_provider(CLAUDE_OPUS_4) == "anthropic"
    assert get_model_provider(CLAUDE_SONNET_4) == "anthropic"
    assert get_model_provider(CLAUDE_HAIKU_4) == "anthropic"


def test_get_model_provider_case_insensitive():
    """Test that provider detection is case-insensitive."""
    assert get_model_provider("GPT-5") == "openai"
    assert get_model_provider("Gemini-2.5-Flash") == "google"
    assert get_model_provider("CLAUDE-SONNET-4-5") == "anthropic"


def test_get_model_provider_unknown():
    """Test that unknown models return 'unknown'."""
    assert get_model_provider("random-model-xyz") == "unknown"
    assert get_model_provider("llama-3") == "unknown"
    assert get_model_provider("") == "unknown"
