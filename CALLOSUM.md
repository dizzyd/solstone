# Callosum Protocol

Callosum is a JSON-per-line message bus for real-time event distribution across Sunstone services.

## Protocol

**Transport:** Unix domain socket at `$JOURNAL_PATH/health/callosum.sock`

**Format:** Newline-delimited JSON. Broadcast to all connected clients.

**Message Structure:**
```json
{
  "tract": "source_subsystem",
  "event": "event_type",
  "ts": 1234567890123,
  // ... tract-specific fields
}
```

**Required Fields:**
- `tract` - Source subsystem identifier (string)
- `event` - Event type within tract (string)
- `ts` - Timestamp in milliseconds (auto-added if missing)

**Behavior:**
- All connections are bidirectional (can emit and receive)
- No routing, no filtering - all messages broadcast to all clients
- Clients should drain socket continuously to prevent backpressure

---

## Tract Registry

### `cortex` - Agent execution events
**Source:** `muse/cortex.py`
**Events:** `request`, `start`, `thinking`, `tool_start`, `tool_end`, `finish`, `error`, `agent_updated`, `info`
**Details:** See [CORTEX.md](CORTEX.md) for agent lifecycle, personas, and event schemas

### `task` - Generic task execution
**Source:** `think/supervisor.py`
**Events:** `request`, `start`, `finish`, `error`
**Fields:** `task_id`, `cmd` (command array), `exit_code`, `error`
**Details:** See `think/supervisor.py` implementation

### `logs` - Process output streaming
**Source:** `think/runner.py`
**Events:** `exec`, `line`, `exit`
**Fields:** `process`, `name`, `pid`, `cmd`, `stream`, `line`, `exit_code`, `duration_ms`, `log_path`
**Details:** See `think/runner.py` `ManagedProcess` class

---

## Implementation

**Client Library:** `think/callosum.py` `CallosumConnection` class
**Server:** `think/callosum.py` `CallosumServer` class

See code documentation for usage patterns and examples.
