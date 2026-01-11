# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Provider abstraction layer for LLM API calls.

Enables context prefix-based routing to different providers (Google Gemini, OpenAI)
via configuration in JOURNAL_PATH/config/providers.json.

Example config:
    {
        "defaults": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prefixes": {
            "describe.*": {"provider": "google", "model": "gemini-2.5-flash-lite"},
            "describe.meeting": {"provider": "openai", "model": "gpt-5-mini"}
        }
    }

Usage:
    from think.providers import resolve_provider, get_provider

    config = resolve_provider("describe.meeting")
    provider = get_provider(config.provider)
    result = provider.generate(contents, model=config.model, context="describe.meeting")
"""

from __future__ import annotations

import fnmatch
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Default configuration when no config file exists
DEFAULT_CONFIG = {
    "defaults": {"provider": "google", "model": "gemini-3-flash-preview"},
    "prefixes": {},
}


@dataclass
class ProviderConfig:
    """Configuration resolved for a specific context."""

    provider: str  # "google" or "openai"
    model: str
    context: str  # The original context string


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        contents: Union[str, List[Any]],
        model: str,
        temperature: float = 0.3,
        max_output_tokens: int = 8192 * 2,
        system_instruction: Optional[str] = None,
        json_output: bool = False,
        thinking_budget: Optional[int] = None,
        timeout_s: Optional[float] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Async generation with unified interface."""
        pass

    @abstractmethod
    def generate(
        self,
        contents: Union[str, List[Any]],
        model: str,
        temperature: float = 0.3,
        max_output_tokens: int = 8192 * 2,
        system_instruction: Optional[str] = None,
        json_output: bool = False,
        thinking_budget: Optional[int] = None,
        timeout_s: Optional[float] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Sync generation with unified interface."""
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports vision/image inputs."""
        pass

    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this provider supports thinking_budget parameter."""
        pass

    def supports_caching(self) -> bool:
        """Whether this provider supports cached_content parameter."""
        return False

    @abstractmethod
    def get_context_window(self, model: str) -> int:
        """Get the context window (max input tokens) for a model.

        Parameters
        ----------
        model : str
            Model identifier

        Returns
        -------
        int
            Maximum number of input tokens the model supports
        """
        pass

    def estimate_tokens(self, contents: Union[str, List[Any]]) -> int:
        """Estimate token count for content.

        Uses a simple heuristic of ~4 characters per token, which is
        reasonable for English text. Subclasses can override with
        more accurate tokenizer-based estimation.

        Parameters
        ----------
        contents : Union[str, List[Any]]
            Content to estimate - string or list of content items

        Returns
        -------
        int
            Estimated token count
        """
        if isinstance(contents, str):
            # ~4 characters per token for English
            return len(contents) // 4
        elif isinstance(contents, list):
            total = 0
            for item in contents:
                if isinstance(item, str):
                    total += len(item) // 4
                elif hasattr(item, "save"):
                    # PIL Image - estimate ~1000 tokens for image
                    total += 1000
                elif isinstance(item, bytes):
                    # Raw image bytes - estimate ~1000 tokens
                    total += 1000
                else:
                    # Unknown type - stringify and estimate
                    total += len(str(item)) // 4
            return total
        return 0

    def check_content_fits(
        self, contents: Union[str, List[Any]], model: str, buffer: int = 1000
    ) -> tuple[bool, int, int]:
        """Check if content fits within model's context window.

        Parameters
        ----------
        contents : Union[str, List[Any]]
            Content to check
        model : str
            Model identifier
        buffer : int
            Reserve this many tokens for output (default 1000)

        Returns
        -------
        tuple[bool, int, int]
            (fits, estimated_tokens, available_tokens)
            - fits: True if content fits within context window minus buffer
            - estimated_tokens: Estimated input token count
            - available_tokens: Context window minus buffer
        """
        estimated = self.estimate_tokens(contents)
        context_window = self.get_context_window(model)
        available = context_window - buffer
        return (estimated <= available, estimated, available)


def load_provider_config() -> Dict[str, Any]:
    """
    Load providers.json from journal config directory.

    Returns default config if file doesn't exist, allowing the system
    to function without explicit configuration (backward compatible).

    Returns
    -------
    dict
        Provider configuration with 'defaults' and 'prefixes' keys
    """
    from think.utils import get_journal

    config_path = Path(get_journal()) / "config" / "providers.json"
    if not config_path.exists():
        logger.debug(f"No providers.json found at {config_path}, using defaults")
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path) as f:
            config = json.load(f)
        # Ensure required keys exist
        if "defaults" not in config:
            config["defaults"] = DEFAULT_CONFIG["defaults"].copy()
        if "prefixes" not in config:
            config["prefixes"] = {}
        return config
    except Exception as e:
        logger.warning(f"Failed to load providers.json: {e}, using defaults")
        return DEFAULT_CONFIG.copy()


@lru_cache(maxsize=1)
def _get_cached_config() -> Dict[str, Any]:
    """Cache the config to avoid repeated file reads."""
    return load_provider_config()


def clear_config_cache() -> None:
    """Clear the cached config (useful for testing or after config changes)."""
    _get_cached_config.cache_clear()


def resolve_provider(context: str) -> ProviderConfig:
    """
    Resolve which provider/model to use for a given context prefix.

    Matching logic:
    1. Exact match on full context string
    2. Glob pattern match (most specific wins - longer base pattern preferred)
    3. Fall back to defaults

    Parameters
    ----------
    context : str
        Context prefix (e.g., "describe.frame", "insight.meetings.markdown")

    Returns
    -------
    ProviderConfig
        Resolved provider and model for this context
    """
    config = _get_cached_config()
    prefixes = config.get("prefixes", {})
    defaults = config.get("defaults", DEFAULT_CONFIG["defaults"])

    # Try exact match first
    if context in prefixes:
        prefix_config = prefixes[context]
        return ProviderConfig(
            provider=prefix_config.get("provider", defaults["provider"]),
            model=prefix_config.get("model", defaults["model"]),
            context=context,
        )

    # Try glob patterns, preferring longer (more specific) matches
    matches = []
    for pattern, prefix_config in prefixes.items():
        if fnmatch.fnmatch(context, pattern):
            # Score by specificity: count non-wildcard characters
            specificity = len(pattern.replace("*", "").replace("?", ""))
            matches.append((specificity, pattern, prefix_config))

    if matches:
        # Sort by specificity (higher = more specific)
        matches.sort(reverse=True, key=lambda x: x[0])
        _, _, prefix_config = matches[0]
        return ProviderConfig(
            provider=prefix_config.get("provider", defaults["provider"]),
            model=prefix_config.get("model", defaults["model"]),
            context=context,
        )

    # Fall back to defaults
    return ProviderConfig(
        provider=defaults["provider"],
        model=defaults["model"],
        context=context,
    )


def get_provider(provider_name: str) -> LLMProvider:
    """
    Factory function to get a provider instance.

    Parameters
    ----------
    provider_name : str
        Provider identifier ("google", "openai", "digitalocean", or "bedrock")

    Returns
    -------
    LLMProvider
        Provider instance

    Raises
    ------
    ValueError
        If provider_name is not recognized
    """
    if provider_name == "google":
        from think.providers_google import GoogleProvider

        return GoogleProvider()
    elif provider_name == "openai":
        from think.providers_openai import OpenAIProvider

        return OpenAIProvider()
    elif provider_name == "digitalocean":
        from think.providers_digitalocean import DigitalOceanProvider

        return DigitalOceanProvider()
    elif provider_name == "bedrock":
        from think.providers_bedrock import BedrockProvider

        return BedrockProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


__all__ = [
    "ProviderConfig",
    "LLMProvider",
    "load_provider_config",
    "clear_config_cache",
    "resolve_provider",
    "get_provider",
]
