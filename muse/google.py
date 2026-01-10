#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Gemini backed agent implementation.

This module provides the Google Gemini backend for the ``muse-agents`` CLI.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, Optional

from google import genai
from google.genai import types

from think.utils import create_mcp_client, get_model_for

from .agents import JSONEventCallback, ThinkingEvent

_DEFAULT_MAX_TOKENS = 8192


def _get_default_model() -> str:
    """Return the configured default model for agents."""
    return get_model_for("agents")


class ToolLoggingHooks:
    """Wrap ``session.call_tool`` to emit events."""

    def __init__(
        self,
        writer: JSONEventCallback,
        agent_id: str | None = None,
        persona: str | None = None,
    ) -> None:
        self.writer = writer
        self._counter = 0
        self.session = None
        self.agent_id = agent_id
        self.persona = persona

    def attach(self, session: Any) -> None:
        self.session = session
        original = session.call_tool

        async def wrapped(name: str, arguments: dict | None = None, **kwargs) -> Any:
            self._counter += 1
            call_id = f"{name}-{self._counter}"
            self.writer.emit(
                {
                    "event": "tool_start",
                    "tool": name,
                    "args": arguments,
                    "call_id": call_id,
                }
            )

            # Build _meta dict for passing agent identity
            meta = {}
            if self.agent_id:
                meta["agent_id"] = self.agent_id
            if self.persona:
                meta["persona"] = self.persona

            result = await original(
                name=name,
                arguments=arguments,
                meta=meta,
                **kwargs,
            )

            # Extract content from CallToolResult if needed
            if hasattr(result, "content"):
                # MCP CallToolResult object - extract text from TextContent objects
                if isinstance(result.content, list):
                    # Handle array of content items
                    extracted_content = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            # TextContent object - extract the text
                            extracted_content.append(item.text)
                        else:
                            # Other content types - keep as is
                            extracted_content.append(item)
                    # If single text content, return as string, otherwise as list
                    result_data = (
                        extracted_content[0]
                        if len(extracted_content) == 1
                        else extracted_content
                    )
                else:
                    result_data = result.content
            else:
                # Direct result (dict, string, etc.)
                result_data = result

            self.writer.emit(
                {
                    "event": "tool_end",
                    "tool": name,
                    "args": arguments,
                    "result": result_data,
                    "call_id": call_id,
                }
            )
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


async def run_agent(
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Run a single prompt through the Google Gemini agent and return the response.

    Args:
        config: Complete configuration dictionary including prompt, instruction, model, etc.
        on_event: Optional event callback
    """
    # Extract values from unified config
    prompt = config.get("prompt", "")
    if not prompt:
        raise ValueError("Missing 'prompt' in config")

    model = config.get("model") or _get_default_model()
    max_tokens = config.get("max_tokens", _DEFAULT_MAX_TOKENS)
    disable_mcp = config.get("disable_mcp", False)
    persona = config.get("persona", "default")

    callback = JSONEventCallback(on_event)

    try:
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": persona,
                "model": model,
                "backend": "google",
            }
        )

        # Extract instruction and extra_context from config
        system_instruction = config.get("instruction", "")
        first_user = config.get("extra_context", "")

        # Build history - check for continuation first
        continue_from = config.get("continue_from")
        if continue_from:
            # Load previous conversation history using shared function
            from .agents import parse_agent_events_to_turns

            turns = parse_agent_events_to_turns(continue_from)
            # Convert to Google's format
            history = []
            for turn in turns:
                role = "model" if turn["role"] == "assistant" else turn["role"]
                history.append(
                    types.Content(role=role, parts=[types.Part(text=turn["content"])])
                )
        else:
            # Fresh conversation
            history = []
            if first_user:
                history.append(
                    types.Content(role="user", parts=[types.Part(text=first_user)])
                )

        # Create client
        client = genai.Client(api_key=api_key)

        # Create fresh chat session
        chat = client.aio.chats.create(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            history=history,
        )

        # Configure tools based on disable_mcp flag
        if not disable_mcp:
            mcp_url = config.get("mcp_server_url")
            if not mcp_url:
                raise RuntimeError("MCP server URL not provided in config")

            # Create MCP client and attach hooks
            async with create_mcp_client(str(mcp_url)) as mcp:
                # Attach tool logging hooks to the MCP session
                agent_id = config.get("agent_id")
                tool_hooks = ToolLoggingHooks(
                    callback, agent_id=agent_id, persona=persona
                )
                tool_hooks.attach(mcp.session)

                # Extract allowed tools from config
                allowed_tools = config.get("tools", None)

                # For now, use the MCP session directly
                # Tool filtering for Google requires more complex implementation
                # that would need to intercept function calls and validate against allowed list
                if allowed_tools and isinstance(allowed_tools, list):
                    logging.getLogger(__name__).info(
                        f"Tool filtering requested for Google backend with tools: {allowed_tools}"
                    )
                    logging.getLogger(__name__).warning(
                        "Tool filtering for Google backend is not yet fully implemented - using all available tools"
                    )

                cfg = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    tools=[mcp.session],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                    thinking_config=(
                        types.ThinkingConfig(
                            include_thoughts=True,
                            thinking_budget=-1,  # Enable dynamic thinking
                        )
                        if hasattr(types, "ThinkingConfig")
                        else None
                    ),
                )

                response = await chat.send_message(prompt, config=cfg)
        else:
            # No MCP tools - just basic config
            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                thinking_config=(
                    types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=-1,  # Enable dynamic thinking
                    )
                    if hasattr(types, "ThinkingConfig")
                    else None
                ),
            )

            response = await chat.send_message(prompt, config=cfg)

        # Extract thinking content from response (works for both MCP and non-MCP cases)
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                # Check for thinking content in candidate
                if hasattr(candidate, "thought") and candidate.thought:
                    thinking_event: ThinkingEvent = {
                        "event": "thinking",
                        "ts": int(time.time() * 1000),
                        "summary": candidate.thought,
                        "model": model,
                    }
                    callback.emit(thinking_event)

        # Also check for thinking at the response level
        if hasattr(response, "thought") and response.thought:
            thinking_event: ThinkingEvent = {
                "event": "thinking",
                "ts": int(time.time() * 1000),
                "summary": response.thought,
                "model": model,
            }
            callback.emit(thinking_event)

        text = response.text
        if not text:
            raise RuntimeError("Model returned empty response")

        # Extract usage from response
        usage_dict = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            usage_dict = {
                "input_tokens": getattr(metadata, "prompt_token_count", 0),
                "output_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0),
            }
            # Only include optional fields if non-zero
            cached = getattr(metadata, "cached_content_token_count", 0)
            if cached:
                usage_dict["cached_tokens"] = cached
            reasoning = getattr(metadata, "thoughts_token_count", 0)
            if reasoning:
                usage_dict["reasoning_tokens"] = reasoning

        callback.emit(
            {
                "event": "finish",
                "result": text,
                "usage": usage_dict,
                "ts": int(time.time() * 1000),
            }
        )
        return text
    except Exception as exc:
        callback.emit(
            {
                "event": "error",
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise


__all__ = [
    "run_agent",
]
