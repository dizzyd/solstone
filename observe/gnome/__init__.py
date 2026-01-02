# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""GNOME-specific activity detection library."""

from observe.gnome.activity import (
    get_idle_time_ms,
    get_monitor_geometries,
    is_power_save_active,
    is_screen_locked,
)

__all__ = [
    "get_idle_time_ms",
    "get_monitor_geometries",
    "is_power_save_active",
    "is_screen_locked",
]
