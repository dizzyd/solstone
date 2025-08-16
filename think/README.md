# sunstone-think

Post-processing utilities for clustering, summarising and repairing captured data. The tools leverage the Gemini API to analyse transcriptions and screenshots. All commands work with a **journal** directory that holds daily folders in `YYYYMMDD` format.

## Installation

```bash
pip install -e .
```

All dependencies are listed in `pyproject.toml`.

## Usage

The package exposes several commands:

- `think-ponder` builds a Markdown summary of a day's recordings using a Gemini prompt.
- `think-cluster` groups audio and screen JSON files into report sections. Use `--start` and
  `--length` to limit the report to a specific time range.
- `see-describe` and `hear-transcribe` include a `--repair` option to process
  any missing screenshot or audio descriptions for a day.
- `think-entity-roll` collects entities across days and writes a rollup file.
- `think-process-day` runs the above tools for a single day.
- `think-supervisor` monitors hear and see heartbeats. Use `--no-runners` to skip starting them automatically.
- `think-mcp-tools` starts an MCP server exposing search capabilities for both ponder text and raw transcripts.
- `think-cortex` starts a WebSocket API server for managing AI agent instances.

```bash
think-ponder YYYYMMDD [-f PROMPT] [-p] [-c] [--force] [-v]
think-cluster YYYYMMDD [--start HHMMSS --length MINUTES]
think-entity-roll
think-process-day [--day YYYYMMDD] [--force] [--repair] [--rebuild]
think-supervisor [--no-runners]
think-mcp-tools [--transport http] [--port PORT] [--path PATH]
think-cortex [--host HOST] [--port PORT] [--path PATH]
```

`-p` is a switch enabling the Gemini Pro model. Use `-c` to count tokens only,
`--force` to overwrite existing files and `-v` for verbose logs.

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.
`JOURNAL_PATH` and `GOOGLE_API_KEY` can also be provided in a `.env` file which
is loaded automatically by most commands.

## Service Discovery

When HTTP services start up, they write their active URIs to files in the journal's `agents/` directory for automated discovery:

- `think-mcp-tools --transport http` writes to `<journal>/agents/mcp.uri` (default: `http://127.0.0.1:6270/mcp/`)
- `think-cortex` writes to `<journal>/agents/cortex.uri` (default: `ws://127.0.0.1:2468/ws/cortex`)

These URI files allow other components to automatically discover running services without hardcoded addresses.

## Automating daily processing

The `think-process-day` command can be triggered by a systemd timer. Below is a
minimal service and timer that process yesterday's folder every morning at
06:00:

```ini
[Unit]
Description=Process sunstone journal

[Service]
Type=oneshot
ExecStart=/usr/local/bin/think-process-day --repair

[Install]
WantedBy=multi-user.target
```

```ini
[Unit]
Description=Run think-process-day daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true
Unit=think-process-day.service

[Install]
WantedBy=timers.target
```

## CLI Agent

The single ``think-agents`` command works with OpenAI, Gemini or Claude via the
``--backend`` option:

```bash
think-agents [TASK_FILE] [--backend PROVIDER] [--model MODEL] [--max-tokens N] [-o OUT_FILE]
```

The provider can be ``openai`` (default), ``google`` or ``anthropic``. The CLI
starts a local MCP server so tools
like topic search are available during a run. If `TASK_FILE` is omitted an
interactive prompt is started.

Set the corresponding API key environment variable (`OPENAI_API_KEY`,
`GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`) along with `JOURNAL_PATH` so the agent can
query your journal index.
The agent prints its
final answer to `stdout`; `-o` or `--out` writes all JSON events to a file.

### Common interface

The `AgentSession` context manager powers all the CLIs. Use
`think.openai.AgentSession`, `think.google.AgentSession` or
`think.anthropic.AgentSession` depending on the backend. The shared
`BaseAgentSession` interface lives in `think.agents`:

```python
async with AgentSession() as agent:
    agent.add_history("user", "previous message")
    result = await agent.run("new request")
    print(agent.history)
```

`run()` returns the final text result. `add_history()` queues prior messages to
provide context and `history` exposes all messages seen during the session. The
same code works with any implementation, allowing you to choose between OpenAI,
Gemini or Claude at runtime.

## Topic map keys

`think.utils.get_topics()` reads the prompt files under `think/topics` and
returns a dictionary keyed by topic name. Each entry contains:

- `path` – the prompt text file path
- `color` – UI color hex string
- `mtime` – modification time of the `.txt` file
- Any additional keys from the matching `<topic>.json` metadata file such as
  `title`, `description` or `occurrences`

## Cortex JSON Event Structures

The think-agents system emits structured JSON events during agent execution. These events are stored in `<journal>/agents/<timestamp>.jsonl` files and can be consumed via the cortex WebSocket API or directly from the files.

### Event Types

All events include:
- `event`: The event type string
- `ts`: Unix timestamp in milliseconds

#### start
Emitted when an agent run begins.
```json
{
  "event": "start",
  "ts": 1234567890123,
  "prompt": "User's request text",
  "persona": "default",
  "model": "gpt-4o"
}
```

#### tool_start
Emitted when a tool execution begins.
```json
{
  "event": "tool_start",
  "ts": 1234567890123,
  "tool": "search_summaries",
  "args": {"query": "search terms", "limit": 10},
  "call_id": "unique_call_id"
}
```

#### tool_end
Emitted when a tool execution completes.
```json
{
  "event": "tool_end",
  "ts": 1234567890123,
  "tool": "search_summaries",
  "args": {"query": "search terms"},
  "result": ["result", "array", "or", "object"],
  "call_id": "unique_call_id"
}
```

#### thinking
Emitted when the model produces reasoning/thinking content (model-dependent).
```json
{
  "event": "thinking",
  "ts": 1234567890123,
  "summary": "Model's internal reasoning about the task...",
  "model": "o1-mini"
}
```

#### agent_updated
Emitted when control is handed off to a different agent (multi-agent scenarios).
```json
{
  "event": "agent_updated",
  "ts": 1234567890123,
  "agent": "SpecializedAgent"
}
```

#### finish
Emitted when the agent run completes successfully.
```json
{
  "event": "finish",
  "ts": 1234567890123,
  "result": "Final response text to the user"
}
```

#### error
Emitted when an error occurs during execution.
```json
{
  "event": "error",
  "ts": 1234567890123,
  "error": "Error message",
  "trace": "Full stack trace..."
}
```

### Tool Call Tracking

Tool events use `call_id` to pair `tool_start` and `tool_end` events. This allows tracking:
- Which tools are currently running
- Tool execution duration
- Tool inputs and outputs

The frontend uses this to show real-time status updates as tools execute.

### WebSocket API

The cortex server (`think-cortex`) provides a WebSocket API for real-time agent monitoring:

- **spawn**: Create a new agent instance
- **attach**: Subscribe to live events from a running agent
- **detach**: Unsubscribe from agent events
- **list**: Get paginated list of historical and running agents

Events are forwarded through the WebSocket with type `agent_event`:
```json
{
  "type": "agent_event",
  "event": {
    "event": "tool_start",
    "tool": "search_summaries",
    ...
  }
}
```
