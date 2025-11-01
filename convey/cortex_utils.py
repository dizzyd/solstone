"""Flask utilities for Cortex agent interactions and event streaming."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from think.callosum import CallosumConnection

from . import state
from .push import push_server

logger = logging.getLogger(__name__)

_WATCH_LOCK = threading.Lock()
_CALLOSUM_CONNECTION: Optional[CallosumConnection] = None


def _ensure_journal_env() -> None:
    if state.journal_root and not os.environ.get("JOURNAL_PATH"):
        os.environ["JOURNAL_PATH"] = state.journal_root


def _event_identifier(event: Dict[str, Any]) -> str:
    agent_id = event.get("agent_id") or ""
    event_type = event.get("event") or ""
    call_id = event.get("call_id") or ""
    tool = event.get("tool") or ""
    ts = event.get("ts") or ""
    # Ensure deterministic string so duplicates can be filtered client-side
    return f"{agent_id}:{event_type}:{call_id}:{tool}:{ts}"


def build_cortex_event_payload(
    event: Dict[str, Any], *, source: str = "cortex", view: str = "chat"
) -> Dict[str, Any]:
    payload = dict(event)
    payload.setdefault("view", view)
    payload["source"] = source
    payload["event_id"] = payload.get("event_id") or _event_identifier(payload)
    return payload


def _broadcast_cortex_event(message: Dict[str, Any]) -> None:
    """Broadcast Cortex event from Callosum to all connected clients."""
    # Filter for cortex tract
    if message.get("tract") != "cortex":
        return

    # Broadcast to all views
    for view in ["chat", "entities", "domains"]:
        payload = build_cortex_event_payload(message, view=view)
        try:
            push_server.push(payload)
        except Exception:  # pragma: no cover - defensive against socket errors
            logger.exception("Failed to broadcast Cortex event to view %s", view)


def start_cortex_event_watcher() -> None:
    """Start listening for Cortex events via Callosum."""
    global _CALLOSUM_CONNECTION
    with _WATCH_LOCK:
        if _CALLOSUM_CONNECTION:
            return

        # Ensure JOURNAL_PATH is set
        _ensure_journal_env()

        # Create Callosum connection with callback
        try:
            _CALLOSUM_CONNECTION = CallosumConnection(callback=_broadcast_cortex_event)
            _CALLOSUM_CONNECTION.connect()
            logger.info("Cortex event watcher connected to Callosum")
        except Exception as e:
            logger.warning(f"Failed to start Cortex watcher: {e}")
            _CALLOSUM_CONNECTION = None


def stop_cortex_event_watcher(timeout: float = 5.0) -> None:
    """Stop listening for Cortex events."""
    global _CALLOSUM_CONNECTION
    with _WATCH_LOCK:
        if _CALLOSUM_CONNECTION:
            _CALLOSUM_CONNECTION.close()
            _CALLOSUM_CONNECTION = None
            logger.info("Cortex event watcher stopped")
