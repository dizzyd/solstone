#!/usr/bin/env python3
"""Gemini backed agent implementation.

This module exposes :class:`AgentSession` for interacting with Google's Gemini
API. It is utilised by the unified ``think-agents`` CLI.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Callable, Optional

from google import genai
from google.genai import types

from .agents import BaseAgentSession, JSONEventCallback, ThinkingEvent
from .models import GEMINI_FLASH
from .utils import agent_instructions, create_mcp_client

DEFAULT_MODEL = GEMINI_FLASH
DEFAULT_MAX_TOKENS = 8192


def setup_logging(verbose: bool) -> logging.Logger:
    """Return app logger configured for ``verbose``."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return logging.getLogger(__name__)


class ToolLoggingHooks:
    """Wrap ``session.call_tool`` to emit events."""

    def __init__(self, writer: JSONEventCallback) -> None:
        self.writer = writer

    def attach(self, session: Any) -> None:
        original = session.call_tool

        async def wrapped(name: str, arguments: dict | None = None, **kwargs) -> Any:
            self.writer.emit({"event": "tool_start", "tool": name, "args": arguments})
            result = await original(name=name, arguments=arguments, **kwargs)
            self.writer.emit({"event": "tool_end", "tool": name, "result": result})
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


class AgentSession(BaseAgentSession):
    """Context manager running Gemini with MCP tools."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_event: Optional[Callable[[dict], None]] = None,
        persona: str = "default",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._callback = JSONEventCallback(on_event)
        self._history: list[dict[str, str]] = []
        self.persona = persona

    async def __aenter__(self) -> "AgentSession":
        return self

    @property
    def history(self) -> list[dict[str, str]]:
        """Return the accumulated chat history as ``role``/``content`` dicts."""

        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Record a message to the chat history."""
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through Gemini and return the result."""
        try:
            # Check API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set")

            self._callback.emit(
                {
                    "event": "start",
                    "prompt": prompt,
                    "persona": self.persona,
                    "model": self.model,
                }
            )

            # Create fresh client and MCP for each run (prevents event loop issues)
            async with create_mcp_client("fastmcp") as mcp:
                client = genai.Client(api_key=api_key)

                # Get system instruction and build history
                system_instruction, first_user, _ = agent_instructions(self.persona)

                # Build history for chat
                history = []
                if first_user:
                    history.append(
                        types.Content(role="user", parts=[types.Part(text=first_user)])
                    )

                # Add existing conversation history
                for msg in self._history:
                    role = msg["role"]
                    # Google genai uses "model" instead of "assistant"
                    if role == "assistant":
                        role = "model"
                    content = msg["content"]
                    history.append(
                        types.Content(role=role, parts=[types.Part(text=content)])
                    )

                # Create fresh chat session
                chat = client.aio.chats.create(
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    history=history,
                )

                # Attach tool logging hooks to the MCP session
                ToolLoggingHooks(self._callback).attach(mcp.session)

                cfg = types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
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

                # Extract thinking content from response
                if hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        # Check for thinking content in candidate
                        if hasattr(candidate, "thought") and candidate.thought:
                            thinking_event: ThinkingEvent = {
                                "event": "thinking",
                                "ts": int(time.time() * 1000),
                                "summary": candidate.thought,
                                "model": self.model,
                            }
                            self._callback.emit(thinking_event)

                # Also check for thinking at the response level
                if hasattr(response, "thought") and response.thought:
                    thinking_event: ThinkingEvent = {
                        "event": "thinking",
                        "ts": int(time.time() * 1000),
                        "summary": response.thought,
                        "model": self.model,
                    }
                    self._callback.emit(thinking_event)

                text = response.text
                self._callback.emit({"event": "finish", "result": text})
                self._history.append({"role": "user", "content": prompt})
                self._history.append({"role": "assistant", "content": text})
                return text
        except Exception as exc:
            self._callback.emit(
                {
                    "event": "error",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
            setattr(exc, "_evented", True)
            raise


async def run_prompt(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` with a temporary :class:`AgentSession`."""

    async with AgentSession(
        model, max_tokens=max_tokens, on_event=on_event, persona=persona
    ) as ag:
        return await ag.run(prompt)


__all__ = [
    "AgentSession",
    "run_prompt",
    "setup_logging",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
]
