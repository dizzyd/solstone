# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Google Gemini provider implementation.

Wraps existing gemini_generate() and gemini_agenerate() functions
to conform to the LLMProvider interface.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from think.providers import LLMProvider

# Context windows for Gemini models
# https://ai.google.dev/gemini-api/docs/models/gemini
GEMINI_CONTEXT_WINDOWS = {
    "gemini-3-flash-preview": 1000000,
    "gemini-3-pro-preview": 2000000,
    "gemini-2.5-flash-lite": 1000000,
    "gemini-2.5-flash": 1000000,
    "gemini-2.5-pro": 2000000,
    "gemini-2.0-flash": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.5-pro": 2000000,
}
GEMINI_DEFAULT_CONTEXT_WINDOW = 1000000


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self):
        """Initialize Google provider."""
        pass

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
        cached_content: Optional[str] = None,
        client=None,
        **kwargs,
    ) -> str:
        """
        Async generation via Gemini.

        Delegates to gemini_agenerate() with full parameter passthrough.
        """
        from think.models import gemini_agenerate

        return await gemini_agenerate(
            contents=contents,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_output=json_output,
            thinking_budget=thinking_budget,
            cached_content=cached_content,
            client=client,
            timeout_s=timeout_s,
            context=context,
        )

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
        cached_content: Optional[str] = None,
        client=None,
        **kwargs,
    ) -> str:
        """
        Sync generation via Gemini.

        Delegates to gemini_generate() with full parameter passthrough.
        """
        from think.models import gemini_generate

        return gemini_generate(
            contents=contents,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_output=json_output,
            thinking_budget=thinking_budget,
            cached_content=cached_content,
            client=client,
            timeout_s=timeout_s,
            context=context,
        )

    def supports_vision(self) -> bool:
        """Gemini supports vision inputs."""
        return True

    def supports_thinking(self) -> bool:
        """Gemini supports thinking_budget parameter."""
        return True

    def supports_caching(self) -> bool:
        """Gemini supports cached_content parameter."""
        return True

    def get_context_window(self, model: str) -> int:
        """Get context window for a Gemini model.

        Parameters
        ----------
        model : str
            Model identifier (e.g., "gemini-3-flash-preview")

        Returns
        -------
        int
            Maximum input tokens for the model
        """
        return GEMINI_CONTEXT_WINDOWS.get(model.lower(), GEMINI_DEFAULT_CONTEXT_WINDOW)


__all__ = [
    "GoogleProvider",
    "GEMINI_CONTEXT_WINDOWS",
    "GEMINI_DEFAULT_CONTEXT_WINDOW",
]
