#!/usr/bin/env python3
"""
gnome_active.py — GNOME activity detection via D-Bus

Checks if the desktop is currently active based on:
- Screen lock status
- Power save mode
- Idle time
- Recent audio activity

Requirements:
  pip install dbus-next

Examples:
  # Check if system is active
  gnome-active && echo "Active" || echo "Inactive"

  # Use in scripts
  if gnome-active; then
      echo "System is active, proceeding..."
  fi

Exit codes:
  0 - Active (not locked, not idle, or recent activity detected)
  1 - Inactive (locked, power save, or idle with no recent activity)
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from observe.gnome.dbus import (
    get_idle_time_ms,
    is_power_save_active,
    is_screen_locked,
)
from think.utils import day_path, setup_cli


def recent_audio_activity(window: int = 120) -> bool:
    """Return True if an *_audio.jsonl file was modified in the last ``window`` seconds."""
    day_dir = day_path()  # Uses today by default, creates if needed, returns Path
    if not day_dir.exists():
        return False
    cutoff = time.time() - window
    for name in os.listdir(day_dir):
        if not name.endswith("_audio.jsonl"):
            continue
        path = day_dir / name
        try:
            if os.path.getmtime(path) >= cutoff:
                return True
        except OSError:
            continue
    return False


async def check_activity(
    idle_threshold_ms: int = 60000, audio_window_s: int = 120
) -> bool:
    """
    Check if the desktop is currently active.

    Args:
        idle_threshold_ms: Idle time threshold in milliseconds (default: 60000 = 60s)
        audio_window_s: Audio activity window in seconds (default: 120 = 2min)

    Returns:
        True if active, False if inactive
    """
    bus = await MessageBus(bus_type=BusType.SESSION).connect()

    # Check if screen is locked
    locked = await is_screen_locked(bus)
    if locked:
        logging.info("Screen is locked")
        return False

    # Check if in power save mode
    power_save = await is_power_save_active(bus)
    if power_save:
        logging.info("Screen is in power save mode")
        return False

    # Check idle time and audio activity
    recent_audio = recent_audio_activity(window=audio_window_s)
    idle_ms = await get_idle_time_ms(bus)

    if not recent_audio and idle_ms > idle_threshold_ms:
        logging.info(
            f"No recent audio activity and desktop idle for {idle_ms/1000:.0f}s"
        )
        return False

    # System is active
    logging.debug(
        f"System active: idle={idle_ms/1000:.0f}s, recent_audio={recent_audio}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check if GNOME desktop is currently active."
    )
    parser.add_argument(
        "--since",
        type=int,
        metavar="SECONDS",
        default=60,
        help="Idle time threshold in seconds (default: 60).",
    )
    parser.add_argument(
        "--audio-window",
        type=int,
        metavar="SECONDS",
        default=120,
        help="Audio activity window in seconds (default: 120 = 2min).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed activity information.",
    )

    args = setup_cli(parser)

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        is_active = asyncio.run(
            check_activity(
                idle_threshold_ms=args.since * 1000,
                audio_window_s=args.audio_window,
            )
        )
        print(str(is_active))
        sys.exit(0 if is_active else 1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
