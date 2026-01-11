# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Amazon Bedrock provider implementation.

Implements the LLMProvider interface for Amazon Bedrock's inference API,
supporting Claude, Llama, Mistral, and Titan model families.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from think.providers import LLMProvider

logger = logging.getLogger(__name__)

# Context windows for Bedrock models
BEDROCK_CONTEXT_WINDOWS = {
    # Claude 3.5 (vision + thinking support)
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    # Claude 3 (vision support)
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    # Amazon Nova 1.0 (vision support for Pro/Lite)
    "amazon.nova-pro-v1:0": 300000,
    "amazon.nova-lite-v1:0": 300000,
    "amazon.nova-micro-v1:0": 128000,
    # Amazon Nova 2.0 (vision support for Pro/Lite)
    "amazon.nova-pro-v2:0": 300000,
    "amazon.nova-lite-v2:0": 300000,
    # Meta Llama 3.1 (text-only, large context)
    "meta.llama3-1-70b-instruct-v1:0": 128000,
    "meta.llama3-1-8b-instruct-v1:0": 128000,
    # Meta Llama 3 (text-only)
    "meta.llama3-70b-instruct-v1:0": 8000,
    "meta.llama3-8b-instruct-v1:0": 8000,
    # Mistral (text-only)
    "mistral.mistral-large-2407-v1:0": 128000,
    "mistral.mistral-large-2501-v1:0": 128000,
    "mistral.mistral-small-2402-v1:0": 32000,
    # Amazon Titan (text-only)
    "amazon.titan-text-premier-v1:0": 32000,
    "amazon.titan-text-express-v1": 8000,
}
BEDROCK_DEFAULT_CONTEXT_WINDOW = 128000

# Models that support vision input
BEDROCK_VISION_MODELS = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v2:0",
    "amazon.nova-lite-v2:0",
}

# Models that support extended thinking
BEDROCK_THINKING_MODELS = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


class BedrockProvider(LLMProvider):
    """Amazon Bedrock provider implementation."""

    def __init__(self):
        """Initialize Bedrock provider."""
        load_dotenv()
        self._region = os.getenv("AWS_REGION", "us-east-1")
        self._sync_client = None
        self._async_client = None
        self._current_model = None

    @property
    def default_model(self) -> str:
        """Return the default model for Bedrock."""
        return "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def _get_sync_client(self):
        """Get or create sync Bedrock Runtime client."""
        if self._sync_client is None:
            import boto3

            self._sync_client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
            )
        return self._sync_client

    def _get_async_client(self):
        """Get or create async Bedrock Runtime client.

        Note: boto3 doesn't have native async support, so we use
        the sync client in an executor for async operations.
        """
        # For now, return sync client - async will use run_in_executor
        return self._get_sync_client()

    def _get_model_family(self, model: str) -> str:
        """Determine the model family from model ID."""
        if model.startswith("anthropic.claude"):
            return "claude"
        elif model.startswith("amazon.nova"):
            return "nova"
        elif model.startswith("meta.llama"):
            return "llama"
        elif model.startswith("mistral."):
            return "mistral"
        elif model.startswith("amazon.titan"):
            return "titan"
        else:
            # Default to Claude format for unknown models
            logger.warning(
                f"Unknown model family for {model}, defaulting to Claude format"
            )
            return "claude"

    def _image_to_base64(self, img) -> str:
        """Convert PIL Image to base64 string."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _build_claude_request(
        self,
        contents: Union[str, List[Any]],
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build request body for Claude models (Anthropic Messages API)."""
        # Build message content
        if isinstance(contents, str):
            message_content = [{"type": "text", "text": contents}]
        elif isinstance(contents, list):
            message_content = []
            for item in contents:
                if isinstance(item, str):
                    message_content.append({"type": "text", "text": item})
                elif hasattr(item, "save"):
                    # PIL Image
                    b64 = self._image_to_base64(item)
                    message_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        }
                    )
                elif isinstance(item, bytes):
                    # Raw image bytes
                    b64 = base64.b64encode(item).decode()
                    message_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        }
                    )
                else:
                    message_content.append({"type": "text", "text": str(item)})
        else:
            message_content = [{"type": "text", "text": str(contents)}]

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": message_content}],
            "temperature": temperature,
        }

        if system_instruction:
            request_body["system"] = system_instruction

        # Add thinking budget for supported models
        if thinking_budget and self._current_model in BEDROCK_THINKING_MODELS:
            request_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        return request_body

    def _build_llama_request(
        self,
        contents: Union[str, List[Any]],
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build request body for Llama models."""
        # Flatten contents to text
        if isinstance(contents, str):
            user_content = contents
        elif isinstance(contents, list):
            text_parts = [str(item) for item in contents if isinstance(item, str)]
            user_content = "\n".join(text_parts)
        else:
            user_content = str(contents)

        # Build Llama prompt format
        if system_instruction:
            prompt = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_content} [/INST]"
        else:
            prompt = f"<s>[INST] {user_content} [/INST]"

        return {
            "prompt": prompt,
            "max_gen_len": max_output_tokens,
            "temperature": temperature,
        }

    def _build_mistral_request(
        self,
        contents: Union[str, List[Any]],
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build request body for Mistral models."""
        # Flatten contents to text
        if isinstance(contents, str):
            user_content = contents
        elif isinstance(contents, list):
            text_parts = [str(item) for item in contents if isinstance(item, str)]
            user_content = "\n".join(text_parts)
        else:
            user_content = str(contents)

        # Build Mistral prompt format
        if system_instruction:
            prompt = f"<s>[INST] {system_instruction}\n\n{user_content} [/INST]"
        else:
            prompt = f"<s>[INST] {user_content} [/INST]"

        return {
            "prompt": prompt,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }

    def _build_titan_request(
        self,
        contents: Union[str, List[Any]],
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build request body for Amazon Titan models."""
        # Flatten contents to text
        if isinstance(contents, str):
            user_content = contents
        elif isinstance(contents, list):
            text_parts = [str(item) for item in contents if isinstance(item, str)]
            user_content = "\n".join(text_parts)
        else:
            user_content = str(contents)

        # Build input text
        if system_instruction:
            input_text = f"{system_instruction}\n\n{user_content}"
        else:
            input_text = user_content

        return {
            "inputText": input_text,
            "textGenerationConfig": {
                "maxTokenCount": max_output_tokens,
                "temperature": temperature,
            },
        }

    def _build_nova_request(
        self,
        contents: Union[str, List[Any]],
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build request body for Amazon Nova models.

        Nova uses a messages-based API similar to Claude but with
        Amazon's specific schema.
        """
        # Build message content
        if isinstance(contents, str):
            message_content = [{"text": contents}]
        elif isinstance(contents, list):
            message_content = []
            for item in contents:
                if isinstance(item, str):
                    message_content.append({"text": item})
                elif hasattr(item, "save"):
                    # PIL Image
                    b64 = self._image_to_base64(item)
                    message_content.append(
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": b64},
                            }
                        }
                    )
                elif isinstance(item, bytes):
                    # Raw image bytes
                    b64 = base64.b64encode(item).decode()
                    message_content.append(
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": b64},
                            }
                        }
                    )
                else:
                    message_content.append({"text": str(item)})
        else:
            message_content = [{"text": str(contents)}]

        request_body = {
            "messages": [{"role": "user", "content": message_content}],
            "inferenceConfig": {
                "maxTokens": max_output_tokens,
                "temperature": temperature,
            },
        }

        if system_instruction:
            request_body["system"] = [{"text": system_instruction}]

        return request_body

    def _extract_claude_response(
        self, response_body: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text and usage from Claude response."""
        text_parts = []
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        text = "\n".join(text_parts)

        usage = {}
        if "usage" in response_body:
            usage_data = response_body["usage"]
            usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
                "total_tokens": (
                    usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0)
                ),
            }

        return text, usage

    def _extract_llama_response(
        self, response_body: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text and usage from Llama response."""
        text = response_body.get("generation", "")

        usage = {}
        if (
            "prompt_token_count" in response_body
            or "generation_token_count" in response_body
        ):
            input_tokens = response_body.get("prompt_token_count", 0)
            output_tokens = response_body.get("generation_token_count", 0)
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        return text, usage

    def _extract_mistral_response(
        self, response_body: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text and usage from Mistral response."""
        outputs = response_body.get("outputs", [])
        text = outputs[0].get("text", "") if outputs else ""

        # Mistral doesn't provide detailed token counts in standard response
        usage = {}

        return text, usage

    def _extract_titan_response(
        self, response_body: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text and usage from Titan response."""
        results = response_body.get("results", [])
        text = results[0].get("outputText", "") if results else ""

        usage = {}
        if "inputTextTokenCount" in response_body:
            input_tokens = response_body.get("inputTextTokenCount", 0)
            output_tokens = sum(r.get("tokenCount", 0) for r in results)
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        return text, usage

    def _extract_nova_response(
        self, response_body: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text and usage from Nova response."""
        output = response_body.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        text_parts = []
        for block in content:
            if "text" in block:
                text_parts.append(block["text"])

        text = "\n".join(text_parts)

        usage = {}
        if "usage" in response_body:
            usage_data = response_body["usage"]
            usage = {
                "input_tokens": usage_data.get("inputTokens", 0),
                "output_tokens": usage_data.get("outputTokens", 0),
                "total_tokens": usage_data.get("totalTokens", 0),
            }

        return text, usage

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
        Sync generation via Amazon Bedrock.

        Parameters
        ----------
        contents : Union[str, List[Any]]
            Content to send (text, images for Claude models)
        model : str
            Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
        temperature : float
            Temperature for generation (default: 0.3)
        max_output_tokens : int
            Maximum tokens for response (default: 8192 * 2)
        system_instruction : str, optional
            System message
        json_output : bool
            Whether to request JSON response format (default: False)
        thinking_budget : int, optional
            Thinking budget for Claude 3.5 models
        timeout_s : float, optional
            Request timeout in seconds (not directly supported by boto3)
        context : str, optional
            Context string for token usage logging

        Returns
        -------
        str
            Response text from the model
        """
        from think.models import log_token_usage

        self._current_model = model
        client = self._get_sync_client()
        family = self._get_model_family(model)

        # Build request based on model family
        if family == "claude":
            request_body = self._build_claude_request(
                contents,
                temperature,
                max_output_tokens,
                system_instruction,
                thinking_budget,
            )
        elif family == "nova":
            request_body = self._build_nova_request(
                contents, temperature, max_output_tokens, system_instruction
            )
        elif family == "llama":
            request_body = self._build_llama_request(
                contents, temperature, max_output_tokens, system_instruction
            )
        elif family == "mistral":
            request_body = self._build_mistral_request(
                contents, temperature, max_output_tokens, system_instruction
            )
        elif family == "titan":
            request_body = self._build_titan_request(
                contents, temperature, max_output_tokens, system_instruction
            )
        else:
            raise ValueError(f"Unsupported model family: {family}")

        # Invoke the model
        response = client.invoke_model(
            modelId=model,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        # Parse response
        response_body = json.loads(response["body"].read())

        # Extract text and usage based on model family
        if family == "claude":
            text, usage = self._extract_claude_response(response_body)
        elif family == "nova":
            text, usage = self._extract_nova_response(response_body)
        elif family == "llama":
            text, usage = self._extract_llama_response(response_body)
        elif family == "mistral":
            text, usage = self._extract_mistral_response(response_body)
        elif family == "titan":
            text, usage = self._extract_titan_response(response_body)
        else:
            text = str(response_body)
            usage = {}

        # Log token usage
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
        Async generation via Amazon Bedrock.

        Note: boto3 doesn't have native async support, so this uses
        asyncio.to_thread to run the sync client in an executor.

        Parameters are the same as generate().
        """
        import asyncio

        # Run sync generate in executor
        return await asyncio.to_thread(
            self.generate,
            contents=contents,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_output=json_output,
            thinking_budget=thinking_budget,
            timeout_s=timeout_s,
            context=context,
            **kwargs,
        )

    def supports_vision(self) -> bool:
        """Check if current model supports vision.

        Returns True if any Claude model is being used, as they all support vision.
        For model-specific checks, use the BEDROCK_VISION_MODELS set.
        """
        return True  # Conservative: vision available via Claude models

    def supports_thinking(self) -> bool:
        """Check if current model supports thinking budget.

        Only Claude 3.5 Sonnet currently supports extended thinking on Bedrock.
        """
        return True  # Conservative: available via Claude 3.5 Sonnet

    def supports_caching(self) -> bool:
        """Bedrock doesn't support cached_content parameter."""
        return False

    def get_context_window(self, model: str) -> int:
        """Get context window for a Bedrock model.

        Parameters
        ----------
        model : str
            Model identifier (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")

        Returns
        -------
        int
            Maximum input tokens for the model
        """
        return BEDROCK_CONTEXT_WINDOWS.get(model, BEDROCK_DEFAULT_CONTEXT_WINDOW)


__all__ = [
    "BedrockProvider",
    "BEDROCK_CONTEXT_WINDOWS",
    "BEDROCK_DEFAULT_CONTEXT_WINDOW",
    "BEDROCK_VISION_MODELS",
    "BEDROCK_THINKING_MODELS",
]
