# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import inspect
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.utils import get_journal


def _register_digitalocean_pricing() -> None:
    """Register DigitalOcean Gradient model pricing with genai-prices.

    Called once at module load to inject DO pricing into the genai-prices
    snapshot, enabling cost calculation for DO models.
    """
    try:
        from genai_prices.data_snapshot import (
            DataSnapshot,
            get_snapshot,
            set_custom_snapshot,
        )
        from genai_prices.types import ClauseEquals, ModelInfo, ModelPrice, Provider

        snapshot = get_snapshot()

        # Check if already registered
        if any(p.id == "digitalocean" for p in snapshot.providers):
            return

        # DigitalOcean Gradient pricing (USD per million tokens)
        # Source: https://docs.digitalocean.com/products/gradient-ai-platform/details/pricing/
        do_provider = Provider(
            id="digitalocean",
            name="DigitalOcean Gradient",
            api_pattern=None,
            pricing_urls=[
                "https://docs.digitalocean.com/products/gradient-ai-platform/details/pricing/"
            ],
            description="DigitalOcean Gradient Serverless Inference",
            price_comments=None,
            model_match=None,
            provider_match=None,
            extractors=None,
            models=[
                ModelInfo(
                    id="openai-gpt-oss-120b",
                    match=ClauseEquals(equals="openai-gpt-oss-120b"),
                    name="GPT-OSS 120B",
                    description="Open source 120B parameter model",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.10"),
                        output_mtok=Decimal("0.70"),
                    ),
                ),
                ModelInfo(
                    id="openai-gpt-oss-20b",
                    match=ClauseEquals(equals="openai-gpt-oss-20b"),
                    name="GPT-OSS 20B",
                    description="Open source 20B parameter model",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.05"),
                        output_mtok=Decimal("0.45"),
                    ),
                ),
                ModelInfo(
                    id="llama3.3-70b-instruct",
                    match=ClauseEquals(equals="llama3.3-70b-instruct"),
                    name="Llama 3.3 70B Instruct",
                    description="Meta Llama 3.3 70B",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.65"),
                        output_mtok=Decimal("0.65"),
                    ),
                ),
                ModelInfo(
                    id="llama3-8b-instruct",
                    match=ClauseEquals(equals="llama3-8b-instruct"),
                    name="Llama 3.1 8B Instruct",
                    description="Meta Llama 3.1 8B",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.198"),
                        output_mtok=Decimal("0.198"),
                    ),
                ),
                ModelInfo(
                    id="mistral-nemo-instruct-2407",
                    match=ClauseEquals(equals="mistral-nemo-instruct-2407"),
                    name="Mistral NeMo",
                    description="Mistral NeMo 12B",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.30"),
                        output_mtok=Decimal("0.30"),
                    ),
                ),
                ModelInfo(
                    id="deepseek-r1-distill-llama-70b",
                    match=ClauseEquals(equals="deepseek-r1-distill-llama-70b"),
                    name="DeepSeek R1 Distill Llama 70B",
                    description="DeepSeek R1 distilled to Llama 70B",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.99"),
                        output_mtok=Decimal("0.99"),
                    ),
                ),
                ModelInfo(
                    id="alibaba-qwen3-32b",
                    match=ClauseEquals(equals="alibaba-qwen3-32b"),
                    name="Qwen3 32B",
                    description="Alibaba Qwen3 32B",
                    context_window=131072,
                    price_comments=None,
                    prices=ModelPrice(
                        input_mtok=Decimal("0.25"),
                        output_mtok=Decimal("0.55"),
                    ),
                ),
            ],
        )

        new_providers = list(snapshot.providers) + [do_provider]
        new_snapshot = DataSnapshot(
            providers=new_providers,
            from_auto_update=False,
            timestamp=snapshot.timestamp,
        )
        set_custom_snapshot(new_snapshot)

    except Exception:
        # Silently fail - pricing is optional
        pass


# Register DO pricing on module load
_register_digitalocean_pricing()

GEMINI_FLASH = "gemini-3-flash-preview"
GEMINI_PRO = "gemini-3-pro-preview"
GEMINI_LITE = "gemini-2.5-flash-lite"

# Mapping from config string names to model constants
GEMINI_MODEL_NAMES = {
    "lite": GEMINI_LITE,
    "flash": GEMINI_FLASH,
    "pro": GEMINI_PRO,
}

GPT_5 = "gpt-5.2"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

CLAUDE_OPUS_4 = "claude-opus-4-5"
CLAUDE_SONNET_4 = "claude-sonnet-4-5"
CLAUDE_HAIKU_4 = "claude-haiku-4-5"

# DigitalOcean Gradient models
DO_GPT_OSS_120B = "openai-gpt-oss-120b"
DO_GPT_OSS_20B = "openai-gpt-oss-20b"
DO_LLAMA_70B = "llama3.3-70b-instruct"
DO_LLAMA_8B = "llama3-8b-instruct"
DO_MISTRAL_NEMO = "mistral-nemo-instruct-2407"
DO_DEEPSEEK_70B = "deepseek-r1-distill-llama-70b"
DO_QWEN_32B = "alibaba-qwen3-32b"


def get_or_create_client(client: Optional[genai.Client] = None) -> genai.Client:
    """Get existing client or create new one.

    Parameters
    ----------
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one
        using GOOGLE_API_KEY from environment.

    Returns
    -------
    genai.Client
        The provided client or a newly created one.
    """
    if client is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        client = genai.Client(api_key=api_key)
    return client


def _normalize_contents(
    contents: Union[str, List[Any], List[types.Content]],
) -> List[Any]:
    """Normalize contents to list format."""
    if isinstance(contents, str):
        return [contents]
    return contents


def _build_generate_config(
    temperature: float,
    max_output_tokens: int,
    system_instruction: Optional[str],
    json_output: bool,
    thinking_budget: Optional[int],
    cached_content: Optional[str],
    timeout_s: Optional[float] = None,
) -> types.GenerateContentConfig:
    """Build the GenerateContentConfig.

    Note: Gemini's max_output_tokens is actually the total budget (thinking + output).
    We compute this internally: total = max_output_tokens + thinking_budget.
    """
    # Compute total tokens: output + thinking budget
    total_tokens = max_output_tokens + (thinking_budget or 0)

    config_args = {
        "temperature": temperature,
        "max_output_tokens": total_tokens,
    }

    if system_instruction:
        config_args["system_instruction"] = system_instruction

    if json_output:
        config_args["response_mime_type"] = "application/json"

    if thinking_budget:
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    if cached_content:
        config_args["cached_content"] = cached_content

    if timeout_s:
        # Convert seconds to milliseconds for the SDK
        timeout_ms = int(timeout_s * 1000)
        config_args["http_options"] = types.HttpOptions(timeout=timeout_ms)

    return types.GenerateContentConfig(**config_args)


def _validate_response(
    response, max_output_tokens: int, thinking_budget: Optional[int] = None
) -> str:
    """Validate response and extract text."""
    if response is None or response.text is None:
        # Try to extract text from candidates if available
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Check for finish reason to understand why we got no text
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)
                if "MAX_TOKENS" in finish_reason:
                    total_tokens = max_output_tokens + (thinking_budget or 0)
                    raise ValueError(
                        f"Model hit token limit ({total_tokens} total = {max_output_tokens} output + "
                        f"{thinking_budget or 0} thinking) before producing output. "
                        f"Try increasing max_output_tokens or reducing thinking_budget."
                    )
                elif "SAFETY" in finish_reason:
                    raise ValueError(
                        f"Response blocked by safety filters: {finish_reason}"
                    )
                elif "STOP" not in finish_reason:
                    raise ValueError(f"Response failed with reason: {finish_reason}")

            # Try to extract text from parts if available
            if (
                hasattr(candidate, "content")
                and hasattr(candidate.content, "parts")
                and candidate.content.parts
            ):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text

        # If we still don't have text, raise an error with details
        error_msg = "No text in response"
        if response:
            if hasattr(response, "candidates") and not response.candidates:
                error_msg = "No candidates in response"
            elif hasattr(response, "prompt_feedback"):
                error_msg = f"Response issue: {response.prompt_feedback}"
        raise ValueError(error_msg)

    return response.text


def log_token_usage(
    model: str,
    usage: Union[Dict[str, Any], Any],
    context: Optional[str] = None,
    segment: Optional[str] = None,
) -> None:
    """Log token usage to journal with unified schema.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash")
    usage : dict or response object
        Usage data in provider-specific format, OR a Gemini response object.
        Dict formats supported:
        - OpenAI format: {input_tokens, output_tokens, total_tokens,
                         details: {input: {cached_tokens}, output: {reasoning_tokens}}}
        - Gemini format: {prompt_token_count, candidates_token_count,
                         cached_content_token_count, thoughts_token_count, total_token_count}
        - Unified format: {input_tokens, output_tokens, total_tokens,
                          cached_tokens, reasoning_tokens, requests}
        Response objects: Gemini GenerateContentResponse with usage_metadata attribute
    context : str, optional
        Context string (e.g., "module.function:123" or "agent.persona.id").
        If None, auto-detects from call stack.
    segment : str, optional
        Segment key (e.g., "143022_300") for attribution.
        If None, falls back to SEGMENT_KEY environment variable.
    """
    try:
        journal = get_journal()

        # Extract from Gemini response object if needed
        if hasattr(usage, "usage_metadata"):
            try:
                metadata = usage.usage_metadata
                usage = {
                    "prompt_token_count": getattr(metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(
                        metadata, "candidates_token_count", 0
                    ),
                    "cached_content_token_count": getattr(
                        metadata, "cached_content_token_count", 0
                    ),
                    "thoughts_token_count": getattr(
                        metadata, "thoughts_token_count", 0
                    ),
                    "total_token_count": getattr(metadata, "total_token_count", 0),
                }
            except Exception:
                return  # Can't extract, fail silently

        # Auto-detect calling context if not provided
        if context is None:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None

            # Skip frames that contain "gemini" in function name
            while caller_frame and "gemini" in caller_frame.f_code.co_name.lower():
                caller_frame = caller_frame.f_back

            if caller_frame:
                module_name = caller_frame.f_globals.get("__name__", "unknown")
                func_name = caller_frame.f_code.co_name
                line_num = caller_frame.f_lineno

                # Clean up module name
                for prefix in ["think.", "observe.", "convey.", "muse."]:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix) :]
                        break

                context = f"{module_name}.{func_name}:{line_num}"

        # Normalize usage data to unified schema
        normalized_usage: Dict[str, int] = {}

        # Handle OpenAI format with nested details
        if "input_tokens" in usage or "output_tokens" in usage:
            normalized_usage["input_tokens"] = usage.get("input_tokens", 0)
            normalized_usage["output_tokens"] = usage.get("output_tokens", 0)
            normalized_usage["total_tokens"] = usage.get("total_tokens", 0)

            # Extract nested details
            details = usage.get("details", {})
            if details:
                input_details = details.get("input", {})
                if input_details and input_details.get("cached_tokens"):
                    normalized_usage["cached_tokens"] = input_details["cached_tokens"]

                output_details = details.get("output", {})
                if output_details and output_details.get("reasoning_tokens"):
                    normalized_usage["reasoning_tokens"] = output_details[
                        "reasoning_tokens"
                    ]

            # Optional requests field for OpenAI
            if "requests" in usage and usage["requests"] is not None:
                normalized_usage["requests"] = usage["requests"]

            # Pass through Anthropic cache fields if present
            if usage.get("cached_tokens"):
                normalized_usage["cached_tokens"] = usage["cached_tokens"]
            if usage.get("cache_creation_tokens"):
                normalized_usage["cache_creation_tokens"] = usage[
                    "cache_creation_tokens"
                ]

        # Handle Gemini format
        elif "prompt_token_count" in usage or "candidates_token_count" in usage:
            normalized_usage["input_tokens"] = usage.get("prompt_token_count", 0)
            normalized_usage["output_tokens"] = usage.get("candidates_token_count", 0)
            normalized_usage["total_tokens"] = usage.get("total_token_count", 0)

            if usage.get("cached_content_token_count"):
                normalized_usage["cached_tokens"] = usage["cached_content_token_count"]
            if usage.get("thoughts_token_count"):
                normalized_usage["reasoning_tokens"] = usage["thoughts_token_count"]

        # Already in unified format
        else:
            normalized_usage = {k: v for k, v in usage.items() if isinstance(v, int)}

        # Build token log entry
        token_data = {
            "timestamp": time.time(),
            "model": model,
            "context": context,
            "usage": normalized_usage,
        }

        # Add segment: prefer parameter, fallback to env (set by think/insight, observe handlers)
        segment_key = segment or os.getenv("SEGMENT_KEY")
        if segment_key:
            token_data["segment"] = segment_key

        # Save to journal/tokens/<YYYYMMDD>.jsonl (one file per day)
        tokens_dir = Path(journal) / "tokens"
        tokens_dir.mkdir(exist_ok=True)

        filename = time.strftime("%Y%m%d.jsonl")
        filepath = tokens_dir / filename

        # Atomic append - safe for parallel writers
        with open(filepath, "a") as f:
            f.write(json.dumps(token_data) + "\n")

    except Exception:
        # Silently fail - logging shouldn't break the main flow
        pass


def get_model_provider(model: str) -> str:
    """Get the provider name from a model identifier.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash", "claude-sonnet-4-5")

    Returns
    -------
    str
        Provider name: "openai", "google", "anthropic", "digitalocean", or "unknown"
    """
    model_lower = model.lower()

    # Check for DigitalOcean models (GPT-OSS, Llama, Mistral, DeepSeek, Alibaba)
    if model_lower.startswith("openai-gpt-oss"):
        return "digitalocean"
    elif model_lower.startswith(("llama", "mistral", "deepseek", "alibaba")):
        return "digitalocean"
    elif model_lower.startswith("gpt"):
        return "openai"
    elif model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("claude"):
        return "anthropic"
    else:
        return "unknown"


def calc_token_cost(token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calculate cost for a token usage record.

    Parameters
    ----------
    token_data : dict
        Token usage record from journal logs with structure:
        {
            "model": "gemini-2.5-flash",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 500,
                "cached_tokens": 800,
                "reasoning_tokens": 200,
                ...
            }
        }

    Returns
    -------
    dict or None
        Cost breakdown:
        {
            "total_cost": 0.00123,
            "input_cost": 0.00075,
            "output_cost": 0.00048,
            "currency": "USD"
        }
        Returns None if pricing unavailable or calculation fails.
    """
    try:
        from genai_prices import Usage, calc_price

        model = token_data.get("model")
        usage_data = token_data.get("usage", {})

        if not model or not usage_data:
            return None

        # Get provider ID
        provider_id = get_model_provider(model)
        if provider_id == "unknown":
            return None

        # Map our token fields to genai_prices Usage format
        # Note: Gemini reports reasoning_tokens separately, but they're billed at
        # output token rates. genai-prices doesn't have a separate field for reasoning,
        # so we add them to output_tokens for correct pricing.
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cached_tokens = usage_data.get("cached_tokens", 0)
        reasoning_tokens = usage_data.get("reasoning_tokens", 0)

        # Add reasoning tokens to output for pricing (Gemini bills them as output)
        total_output_tokens = output_tokens + reasoning_tokens

        # Create Usage object
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=total_output_tokens,
            cache_read_tokens=cached_tokens if cached_tokens > 0 else None,
        )

        # Calculate price
        result = calc_price(
            usage=usage,
            model_ref=model,
            provider_id=provider_id,
        )

        # Return simplified cost breakdown
        return {
            "total_cost": float(result.total_price),
            "input_cost": float(result.input_price),
            "output_cost": float(result.output_price),
            "currency": "USD",
        }

    except Exception:
        # Silently fail if pricing unavailable
        return None


def gemini_generate(
    contents: Union[str, List[Any], List[types.Content]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    cached_content: Optional[str] = None,
    client: Optional[genai.Client] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
) -> str:
    """
    Simplified wrapper for genai.models.generate_content with common defaults.

    Parameters
    ----------
    contents : str, List, or List[types.Content]
        The content to send to the model. Can be:
        - A string (will be converted to a list with one string)
        - A list of strings, types.Part objects, or mixed content
        - A list of types.Content objects for complex conversations
    model : str
        Model name to use (default: GEMINI_FLASH)
    temperature : float
        Temperature for generation (default: 0.3)
    max_output_tokens : int
        Maximum tokens for the model's response output (default: 8192 * 2).
        Note: This is the output budget only. The total token budget sent to
        Gemini's API is computed as max_output_tokens + thinking_budget.
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking. When set, the total token budget
        becomes max_output_tokens + thinking_budget.
    cached_content : str, optional
        Name of cached content to use
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one.
    timeout_s : float, optional
        Request timeout in seconds.
    context : str, optional
        Context string for token usage logging (e.g., "insight.decisions.markdown").
        If not provided, auto-detects from call stack.

    Returns
    -------
    str
        Response text from the model
    """
    client = get_or_create_client(client)
    contents = _normalize_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        cached_content=cached_content,
        timeout_s=timeout_s,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens, thinking_budget)
    log_token_usage(model=model, usage=response, context=context)
    return text


async def gemini_agenerate(
    contents: Union[str, List[Any], List[types.Content]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    cached_content: Optional[str] = None,
    client: Optional[genai.Client] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
) -> str:
    """
    Async wrapper for genai.aio.models.generate_content with common defaults.

    Parameters
    ----------
    contents : str, List, or List[types.Content]
        The content to send to the model. Can be:
        - A string (will be converted to a list with one string)
        - A list of strings, types.Part objects, or mixed content
        - A list of types.Content objects for complex conversations
    model : str
        Model name to use (default: GEMINI_FLASH)
    temperature : float
        Temperature for generation (default: 0.3)
    max_output_tokens : int
        Maximum tokens for the model's response output (default: 8192 * 2).
        Note: This is the output budget only. The total token budget sent to
        Gemini's API is computed as max_output_tokens + thinking_budget.
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking. When set, the total token budget
        becomes max_output_tokens + thinking_budget.
    cached_content : str, optional
        Name of cached content to use
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one.
    timeout_s : float, optional
        Request timeout in seconds.
    context : str, optional
        Context string for token usage logging (e.g., "insight.decisions.markdown").
        If not provided, auto-detects from call stack.

    Returns
    -------
    str
        Response text from the model
    """
    client = get_or_create_client(client)
    contents = _normalize_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        cached_content=cached_content,
        timeout_s=timeout_s,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens, thinking_budget)
    log_token_usage(model=model, usage=response, context=context)
    return text


def generate(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs,
) -> str:
    """
    Unified sync generation that routes based on context prefix.

    Routes to the appropriate provider (Google Gemini, OpenAI) based on
    configuration in JOURNAL_PATH/config/providers.json.

    Parameters
    ----------
    contents : str or List
        The content to send to the model
    context : str
        Context prefix (e.g., "describe.frame", "insight.meetings.markdown")
        Used to determine provider and model from config.
    temperature : float
        Temperature for generation (default: 0.3)
    max_output_tokens : int
        Maximum tokens for response (default: 8192 * 2)
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking (only supported by some providers)
    timeout_s : float, optional
        Request timeout in seconds
    **kwargs
        Additional provider-specific arguments

    Returns
    -------
    str
        Response text from the model
    """
    from think.providers import get_provider, resolve_provider

    config = resolve_provider(context)
    provider = get_provider(config.provider)

    # Only pass thinking_budget if provider supports it
    effective_thinking_budget = (
        thinking_budget if provider.supports_thinking() else None
    )

    return provider.generate(
        contents=contents,
        model=config.model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=effective_thinking_budget,
        timeout_s=timeout_s,
        context=context,
        **kwargs,
    )


async def agenerate(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs,
) -> str:
    """
    Unified async generation that routes based on context prefix.

    Routes to the appropriate provider (Google Gemini, OpenAI) based on
    configuration in JOURNAL_PATH/config/providers.json.

    Parameters
    ----------
    contents : str or List
        The content to send to the model
    context : str
        Context prefix (e.g., "describe.frame", "insight.meetings.markdown")
        Used to determine provider and model from config.
    temperature : float
        Temperature for generation (default: 0.3)
    max_output_tokens : int
        Maximum tokens for response (default: 8192 * 2)
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking (only supported by some providers)
    timeout_s : float, optional
        Request timeout in seconds
    **kwargs
        Additional provider-specific arguments

    Returns
    -------
    str
        Response text from the model
    """
    from think.providers import get_provider, resolve_provider

    config = resolve_provider(context)
    provider = get_provider(config.provider)

    # Only pass thinking_budget if provider supports it
    effective_thinking_budget = (
        thinking_budget if provider.supports_thinking() else None
    )

    return await provider.agenerate(
        contents=contents,
        model=config.model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=effective_thinking_budget,
        timeout_s=timeout_s,
        context=context,
        **kwargs,
    )


__all__ = [
    "GEMINI_PRO",
    "GEMINI_FLASH",
    "GEMINI_LITE",
    "GEMINI_MODEL_NAMES",
    "GPT_5",
    "GPT_5_MINI",
    "GPT_5_NANO",
    "CLAUDE_OPUS_4",
    "CLAUDE_SONNET_4",
    "CLAUDE_HAIKU_4",
    "GPT_OSS_20B",
    "GPT_OSS_120B",
    "get_or_create_client",
    "gemini_generate",
    "gemini_agenerate",
    "generate",
    "agenerate",
    "log_token_usage",
    "get_model_provider",
    "calc_token_cost",
]
