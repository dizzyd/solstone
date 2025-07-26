#!/usr/bin/env python3
"""think.agent

CLI utility launching an OpenAI agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-agent [TASK_FILE] [--model MODEL] [--max-tokens N] [-v] [-o OUT]

When ``TASK_FILE`` is omitted, an interactive ``chat>`` prompt is started.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents import (
    Agent,
    AgentHooks,
    ModelSettings,
    RunConfig,
    Runner,
    SQLiteSession,
    enable_verbose_stdout_logging,
    set_default_openai_key,
)
from agents.mcp import MCPServerStdio

from think.utils import get_topics, setup_cli

AGENT_PATH = Path(__file__).with_name("agent.txt")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration for agent and tools."""

    if verbose:
        enable_verbose_stdout_logging()

        openai_agents_logger = logging.getLogger("openai.agents")
        openai_agents_logger.setLevel(logging.DEBUG)
        if not openai_agents_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            openai_agents_logger.addHandler(handler)

        tracing_logger = logging.getLogger("openai.agents.tracing")
        tracing_logger.setLevel(logging.DEBUG)
        if not tracing_logger.handlers:
            tracing_logger.addHandler(handler)

    app_logger = logging.getLogger(__name__)
    if verbose:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)

    return app_logger


class JSONEventWriter:
    """Write JSONL events to stdout and optional file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.file = None
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.file = open(path, "a", encoding="utf-8")
            except Exception as exc:  # pragma: no cover - display only
                logging.error("Failed to open %s: %s", path, exc)

    def emit(self, data: dict) -> None:
        line = json.dumps(data, ensure_ascii=False)
        print(line)
        if self.file:
            try:
                self.file.write(line + "\n")
                self.file.flush()
            except Exception as exc:  # pragma: no cover - display only
                logging.error("Failed to write event to %s: %s", self.path, exc)

    def close(self) -> None:
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass


class ToolLoggingHooks(AgentHooks):
    """Custom hooks to log tool calls and other agent events."""

    def __init__(self, writer: JSONEventWriter) -> None:
        self.writer = writer

    def on_tool_call_start(self, context, tool_name: str, arguments: dict) -> None:
        self.writer.emit({"event": "tool_start", "tool": tool_name, "args": arguments})

    def on_tool_call_end(self, context, tool_name: str, result) -> None:
        self.writer.emit({"event": "tool_end", "tool": tool_name, "result": result})

    def on_agent_start(self, context, agent_name: str) -> None:
        self.writer.emit({"event": "agent_start", "agent": agent_name})

    def on_agent_end(self, context, agent_name: str, result) -> None:
        self.writer.emit({"event": "agent_end", "agent": agent_name})


def agent_instructions() -> tuple[str, str]:
    """Return system instruction and initial user message."""

    system_instruction = AGENT_PATH.read_text(encoding="utf-8")

    extra_parts: list[str] = []
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        ent_path = Path(journal) / "entities.md"
        if ent_path.is_file():
            entities = ent_path.read_text(encoding="utf-8").strip()
            if entities:
                extra_parts.append("## Well-Known Entities\n" + entities)

    topics = get_topics()
    if topics:
        lines = [
            "## Topics",
            "These are the topics available for use in tool and resource requests:",
        ]
        for name, info in sorted(topics.items()):
            desc = str(info.get("contains", ""))
            lines.append(f"* Topic: `{name}`: {desc}")
        extra_parts.append("\n".join(lines))

    now = datetime.now()

    # Try to get local timezone
    try:
        local_tz = zoneinfo.ZoneInfo(str(now.astimezone().tzinfo))
        now_local = now.astimezone(local_tz)
        time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception:
        # Fallback without timezone
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

    extra_parts.append(f"## Current Date and Time\n{time_str}")

    extra_context = "\n\n".join(extra_parts).strip()
    return system_instruction, extra_context


async def main_async():
    """Main async entry point."""
    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with the request, '-' for stdin or omit for interactive",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for the final response",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Direct prompt text for single query mode",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="File path to write the final result or error to",
    )

    args = setup_cli(parser)
    out_path = args.out

    app_logger = setup_logging(args.verbose)
    event_writer = JSONEventWriter(out_path)

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        set_default_openai_key(api_key)

    # Get task/prompt
    if args.query:
        user_prompt = args.query
    elif args.task_file is None:
        user_prompt = None  # Interactive mode
    elif args.task_file == "-":
        user_prompt = sys.stdin.read()
    else:
        if not os.path.isfile(args.task_file):
            parser.error(f"Task file not found: {args.task_file}")
        app_logger.info("Loading task file %s", args.task_file)
        user_prompt = Path(args.task_file).read_text(encoding="utf-8")

    # Configure MCP server
    journal_path = os.getenv("JOURNAL_PATH", "journal")
    mcp_server = MCPServerStdio(
        params={
            "command": sys.executable,
            "args": ["-m", "think.mcp_server"],
            "env": {
                "JOURNAL_PATH": journal_path,
                "PYTHONPATH": os.pathsep.join([os.getcwd()] + sys.path),
            },
        },
        name="Sunstone MCP Server",
    )

    # Configure run settings
    run_config = RunConfig()

    # Connect to MCP server and run
    try:
        async with mcp_server:
            system_instruction, extra_context = agent_instructions()
            # Create agent with connected server and model configuration
            agent = Agent(
                name="SunstoneCLI",
                instructions=f"{system_instruction}\n\n{extra_context}",
                model=args.model,
                model_settings=ModelSettings(
                    max_tokens=args.max_tokens, temperature=0.2
                ),
                mcp_servers=[mcp_server],
                hooks=ToolLoggingHooks(event_writer),
            )

            if user_prompt is None:
                # Interactive mode with session management
                app_logger.info("Starting interactive mode with model %s", args.model)
                session = SQLiteSession("sunstone_cli_session")

                try:
                    while True:
                        try:
                            loop = asyncio.get_event_loop()
                            prompt = await loop.run_in_executor(
                                None, lambda: input("chat> ")
                            )

                            if not prompt:
                                continue

                            event_writer.emit({"event": "start", "prompt": prompt})
                            result = await Runner.run(
                                agent, prompt, session=session, run_config=run_config
                            )
                            event_writer.emit(
                                {"event": "finish", "result": result.final_output}
                            )

                        except EOFError:
                            break
                except KeyboardInterrupt:
                    pass
            else:
                # Single prompt mode
                app_logger.debug("Task contents: %s", user_prompt)
                app_logger.info("Running agent with model %s", args.model)

                event_writer.emit({"event": "start", "prompt": user_prompt})
                result = await Runner.run(agent, user_prompt, run_config=run_config)
                event_writer.emit({"event": "finish", "result": result.final_output})
    except Exception as exc:
        event_writer.emit({"event": "error", "error": str(exc)})
        raise
    finally:
        event_writer.close()


def main():
    """Entry point that runs the async main."""
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
