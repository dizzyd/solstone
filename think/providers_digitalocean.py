# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
DigitalOcean Gradient provider implementation.

Implements the LLMProvider interface for DigitalOcean's Gradient Serverless
Inference API, which hosts GPT-OSS models (gpt-oss-20b, gpt-oss-120b).

The API is OpenAI-compatible, using the same request/response format
but with a different endpoint (https://inference.do-ai.run).

Requires DO_API_KEY environment variable for authentication.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from think.providers import LLMProvider

logger = logging.getLogger(__name__)

# DigitalOcean Gradient Serverless Inference endpoint
DO_INFERENCE_BASE_URL = "https://inference.do-ai.run/v1"

# Context windows for DigitalOcean models
# Note: API enforces 96K limit for gpt-oss models despite docs claiming higher
DO_CONTEXT_WINDOWS = {
    "openai-gpt-oss-120b": 96000,
    "openai-gpt-oss-20b": 96000,
    "llama3.3-70b-instruct": 96000,
    "llama3-8b-instruct": 96000,
    "mistral-nemo-instruct-2407": 96000,
    "deepseek-r1-distill-llama-70b": 96000,
    "alibaba-qwen3-32b": 96000,
}
DO_DEFAULT_CONTEXT_WINDOW = 96000


class DigitalOceanProvider(LLMProvider):
    """DigitalOcean Gradient provider implementation for GPT-OSS models."""

    def __init__(self):
        """Initialize DigitalOcean provider."""
        load_dotenv()
        self._api_key = os.getenv("DO_API_KEY")
        self._sync_client = None
        self._async_client = None

    @property
    def default_model(self) -> str:
        """Return the default model for DigitalOcean."""
        return "openai-gpt-oss-120b"

    def _get_sync_client(self):
        """Get or create sync OpenAI-compatible client for DigitalOcean."""
        if self._sync_client is None:
            from openai import OpenAI

            if not self._api_key:
                raise ValueError("DO_API_KEY not found in environment")
            self._sync_client = OpenAI(
                api_key=self._api_key,
                base_url=DO_INFERENCE_BASE_URL,
            )
        return self._sync_client

    def _get_async_client(self):
        """Get or create async OpenAI-compatible client for DigitalOcean."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            if not self._api_key:
                raise ValueError("DO_API_KEY not found in environment")
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=DO_INFERENCE_BASE_URL,
            )
        return self._async_client

    def _build_messages(
        self,
        contents: Union[str, List[Any]],
        system_instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert contents to OpenAI message format.

        Handles text, PIL Images, and raw image bytes.

        Parameters
        ----------
        contents : Union[str, List[Any]]
            Content to send - can be string, list of strings,
            PIL Images, or raw bytes
        system_instruction : str, optional
            System message to prepend

        Returns
        -------
        List[Dict[str, Any]]
            OpenAI messages array
        """
        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        # Handle different content types
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            user_content = []
            for item in contents:
                if isinstance(item, str):
                    user_content.append({"type": "text", "text": item})
                elif hasattr(item, "save"):
                    # PIL Image - convert to base64
                    b64 = self._image_to_base64(item)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                elif isinstance(item, bytes):
                    # Raw image bytes
                    b64 = base64.b64encode(item).decode()
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                else:
                    # Unknown type - stringify
                    user_content.append({"type": "text", "text": str(item)})
            messages.append({"role": "user", "content": user_content})

        return messages

    def _image_to_base64(self, img) -> str:
        """Convert PIL Image to base64 string."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _build_request_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        json_output: bool,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build kwargs for OpenAI-compatible API request."""
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        if json_output:
            request_kwargs["response_format"] = {"type": "json_object"}

        if timeout_s:
            request_kwargs["timeout"] = timeout_s

        return request_kwargs

    def _extract_usage(self, response) -> Dict[str, Any]:
        """Extract usage data from OpenAI-compatible response."""
        if not response.usage:
            return {}

        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return usage

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
        """
        Sync generation via DigitalOcean Gradient Serverless Inference.

        Parameters
        ----------
        contents : Union[str, List[Any]]
            Content to send (text, images, etc.)
        model : str
            Model name (e.g., "gpt-oss-20b", "gpt-oss-120b")
        temperature : float
            Temperature for generation (default: 0.3)
        max_output_tokens : int
            Maximum tokens for response (default: 8192 * 2)
        system_instruction : str, optional
            System message
        json_output : bool
            Whether to request JSON response format (default: False)
        thinking_budget : int, optional
            Ignored for DigitalOcean (not supported)
        timeout_s : float, optional
            Request timeout in seconds
        context : str, optional
            Context string for token usage logging

        Returns
        -------
        str
            Response text from the model
        """
        from think.models import log_token_usage

        client = self._get_sync_client()
        messages = self._build_messages(contents, system_instruction)
        request_kwargs = self._build_request_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_output=json_output,
            timeout_s=timeout_s,
        )

        response = client.chat.completions.create(**request_kwargs)

        text = response.choices[0].message.content or ""

        # Log token usage
        usage = self._extract_usage(response)
        if usage:
            log_token_usage(model=model, usage=usage, context=context)

        return text

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
        """
        Async generation via DigitalOcean Gradient Serverless Inference.

        Parameters are the same as generate().
        """
        from think.models import log_token_usage

        client = self._get_async_client()
        messages = self._build_messages(contents, system_instruction)
        request_kwargs = self._build_request_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_output=json_output,
            timeout_s=timeout_s,
        )

        response = await client.chat.completions.create(**request_kwargs)

        text = response.choices[0].message.content or ""

        # Log token usage
        usage = self._extract_usage(response)
        if usage:
            log_token_usage(model=model, usage=usage, context=context)

        return text

    def supports_vision(self) -> bool:
        """GPT-OSS models may support vision depending on the variant."""
        # Conservative default - check DO docs for specific model capabilities
        return False

    def supports_thinking(self) -> bool:
        """DigitalOcean doesn't have a thinking_budget parameter."""
        return False

    def supports_caching(self) -> bool:
        """DigitalOcean doesn't support cached_content parameter."""
        return False

    def get_context_window(self, model: str) -> int:
        """Get context window for a DigitalOcean model.

        Parameters
        ----------
        model : str
            Model identifier (e.g., "openai-gpt-oss-120b")

        Returns
        -------
        int
            Maximum input tokens for the model
        """
        return DO_CONTEXT_WINDOWS.get(model.lower(), DO_DEFAULT_CONTEXT_WINDOW)


__all__ = [
    "DigitalOceanProvider",
    "DO_INFERENCE_BASE_URL",
    "DO_CONTEXT_WINDOWS",
    "DO_DEFAULT_CONTEXT_WINDOW",
]
