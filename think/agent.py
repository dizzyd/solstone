#!/usr/bin/env python3
"""think.agent

CLI utility launching an OpenAI agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-agent [TASK_FILE] [--model MODEL] [--max-tokens N] [-v]

When ``TASK_FILE`` is omitted, an interactive ``chat>`` prompt is started.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_key
from agents.mcp import MCPServerStdio

from think.utils import setup_cli


def build_agent(model: str, max_tokens: int) -> tuple[Agent, RunConfig]:
    """Return configured OpenAI agent and run configuration."""

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        set_default_openai_key(api_key)

    run_config = RunConfig(
        model=model,
        model_settings=ModelSettings(max_tokens=max_tokens, temperature=0.2),
    )

    mcp_servers = []
    if MCPServerStdio is not None:
        mcp_servers.append(
            MCPServerStdio(
                {"command": "python", "args": ["think/mcp_server.py", "--stdio"]}
            )
        )

    agent = Agent(
        name="SunstoneCLI",
        instructions=(
            "You are the Sunstone journal assistant. Use the provided tools to "
            "search ponder summaries, search occurrences, and read markdown files."
        ),
        mcp_servers=mcp_servers,
    )
    return agent, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with the request, '-' for stdin or omit for interactive",
    )
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for the final response",
    )

    args = setup_cli(parser)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openai.agents").setLevel(logging.DEBUG)

    if args.task_file is None:
        user_prompts = None
    elif args.task_file == "-":
        user_prompts = [sys.stdin.read()]
    else:
        if not os.path.isfile(args.task_file):
            parser.error(f"Task file not found: {args.task_file}")

        logging.info("Loading task file %s", args.task_file)
        user_prompts = [Path(args.task_file).read_text(encoding="utf-8")]

    logging.info("Building agent with model %s", args.model)
    agent, run_config = build_agent(args.model, args.max_tokens)

    if user_prompts is None:
        # interactive mode
        try:
            while True:
                try:
                    prompt = input("chat> ")
                except EOFError:
                    break
                if not prompt:
                    continue
                result = Runner.run_sync(agent, prompt, run_config=run_config)
                print(result.final_output)
        except KeyboardInterrupt:
            pass
    else:
        user_prompt = user_prompts[0]
        logging.debug("Task contents: %s", user_prompt)
        logging.info("Running agent")
        result = Runner.run_sync(agent, user_prompt, run_config=run_config)
        print(result.final_output)


if __name__ == "__main__":
    main()
