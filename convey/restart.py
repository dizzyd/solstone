#!/usr/bin/env python3
"""Restart utility for Convey web service."""

from __future__ import annotations

import argparse
import sys
import threading
import time

from think.callosum import CallosumConnection
from think.utils import setup_cli


def wait_for_convey_restart(
    timeout: float = 5.0,
) -> tuple[bool, list[tuple[float, str, str]]]:
    """Restart convey service and wait for it to be ready.

    Connects to CALLOSUM, sends restart request to supervisor, and monitors
    events until service successfully restarts or fails.

    Args:
        timeout: Maximum seconds to wait for restart (default: 5.0)

    Returns:
        (success, logs): True if service started successfully within timeout,
                        list of (timestamp, stream, line) tuples collected during restart

    Failure conditions:
        - Service restarts twice (crash detected)
        - Timeout expires before successful start
        - Connection to CALLOSUM fails
    """
    state = {
        "start_count": 0,
        "ref": None,
        "logs": [],  # List of (timestamp, stream, line)
        "started": False,
        "failed": False,
        "lock": threading.Lock(),
    }

    def handle_event(message: dict) -> None:
        """Process incoming CALLOSUM messages."""
        tract = message.get("tract")
        event = message.get("event")
        service = message.get("service")

        with state["lock"]:
            # Track supervisor events for convey service
            if tract == "supervisor" and service == "convey":
                if event == "restarting":
                    print("Restarting convey service...", file=sys.stderr)

                elif event == "stopped":
                    exit_code = message.get("exit_code", "?")
                    print(f"Convey stopped (exit code: {exit_code})", file=sys.stderr)

                elif event == "started":
                    state["start_count"] += 1
                    ref = message.get("ref")
                    pid = message.get("pid")

                    if state["start_count"] == 1:
                        # First start - this is expected
                        state["ref"] = ref
                        state["started"] = True
                        print(
                            f"Convey started (pid: {pid}, ref: {ref})", file=sys.stderr
                        )
                    elif state["start_count"] >= 2:
                        # Second start - service crashed and restarted
                        state["failed"] = True
                        print(
                            f"ERROR: Convey crashed and restarted (attempt {state['start_count']})",
                            file=sys.stderr,
                        )

            # Collect logs from convey process
            elif tract == "logs" and event == "line":
                name = message.get("name", "")
                # Match both "convey" service and any convey-related processes
                if "convey" in name.lower():
                    timestamp = message.get("ts", time.time() * 1000) / 1000
                    stream = message.get("stream", "stdout")
                    line = message.get("line", "")
                    state["logs"].append((timestamp, stream, line))

    # Connect to CALLOSUM
    callosum = CallosumConnection()
    try:
        callosum.start(callback=handle_event)
    except Exception as e:
        print(f"ERROR: Failed to connect to CALLOSUM: {e}", file=sys.stderr)
        return False, []

    # Give connection a moment to establish
    time.sleep(0.2)

    # Send restart request
    print("Sending restart request to supervisor...", file=sys.stderr)
    if not callosum.emit("supervisor", "restart", service="convey"):
        print("ERROR: Failed to send restart request", file=sys.stderr)
        callosum.stop()
        return False, []

    # Wait for restart with timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        with state["lock"]:
            if state["failed"]:
                # Service crashed during restart
                callosum.stop()
                return False, state["logs"]

            if state["started"]:
                # Service restarted successfully
                callosum.stop()
                # Small additional delay for Flask to fully bind port
                time.sleep(1.0)
                return True, state["logs"]

        # Check again in 100ms
        time.sleep(0.1)

    # Timeout expired
    print(f"ERROR: Timeout waiting for convey to restart ({timeout}s)", file=sys.stderr)
    callosum.stop()
    return False, state["logs"]


def main() -> None:
    """CLI entry point for convey restart utility."""
    parser = argparse.ArgumentParser(
        description="Restart the Convey web service via supervisor"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for restart (default: 5.0)",
    )

    args = setup_cli(parser)

    success, logs = wait_for_convey_restart(timeout=args.timeout)

    if not success:
        print("\nERROR: Convey service failed to restart", file=sys.stderr)
        if logs:
            print("\nCollected logs:", file=sys.stderr)
            print("-" * 60, file=sys.stderr)
            for timestamp, stream, line in logs:
                # Format timestamp as readable time
                time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                stream_prefix = "ERR" if stream == "stderr" else "OUT"
                print(f"[{time_str}] [{stream_prefix}] {line}", file=sys.stderr)
            print("-" * 60, file=sys.stderr)
        sys.exit(1)

    # Success - print the URL
    print("Convey running at http://localhost:8000/")


if __name__ == "__main__":
    main()
