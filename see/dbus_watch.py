#!/usr/bin/env python3
"""Simple helpers to observe GNOME Shell DBus activity.

The script attaches to three sets of events and prints a short line whenever
something happens:
  * idle and user activity from ``org.gnome.Mutter.IdleMonitor``;
  * window changes from ``org.gnome.Shell.Introspect``;
  * all desktop notifications via ``org.freedesktop.Notifications``.

Run it directly to watch everything or cherry pick the helpers for your own
experiments.
"""

import asyncio

from dbus_next.aio import MessageBus
from dbus_next.constants import MessageType
from dbus_next.message import Message

IDLE_IFACE = "org.gnome.Mutter.IdleMonitor"
IDLE_PATH = "/org/gnome/Mutter/IdleMonitor/Core"
SHELL_IFACE = "org.gnome.Shell.Introspect"
SHELL_PATH = "/org/gnome/Shell/Introspect"
SHELL_NAME = "org.gnome.Shell"


async def watch_idle() -> None:
    bus = await MessageBus().connect()
    proxy = await bus.introspect(SHELL_NAME, IDLE_PATH)
    idle = bus.get_proxy_object(SHELL_NAME, IDLE_PATH, proxy).get_interface(IDLE_IFACE)

    idle_watch = await idle.call_add_idle_watch(30_000)
    active_watch = await idle.call_add_user_active_watch()

    def on_watch_fired(watch_id: int) -> None:
        if watch_id == idle_watch:
            print("[idle] Desktop idle for 30 s")
        elif watch_id == active_watch:
            print("[idle] User became active")

    idle.on_watch_fired(on_watch_fired)

    while True:
        t = await idle.call_get_idletime()
        print(f"[idle] Current idle time {t / 1000:.1f} s")
        await asyncio.sleep(5)


async def watch_windows() -> None:
    bus = await MessageBus().connect()
    proxy = await bus.introspect(SHELL_NAME, SHELL_PATH)
    intro = bus.get_proxy_object(SHELL_NAME, SHELL_PATH, proxy).get_interface(SHELL_IFACE)

    windows = await intro.call_get_windows()
    titles = [props.get("title") for props in windows.values()]
    print("[shell] Existing windows:", titles)

    # Try to subscribe to individual window events, fall back to general changes
    try:
        intro.on_window_added(lambda wid, props: print("[shell] Window added", props.get("title")))
        intro.on_window_removed(lambda wid: print("[shell] Window removed", wid))
        intro.on_window_changed(
            lambda wid, props: print("[shell] Window changed", props.get("title"))
        )
        print("[shell] Watching individual window events")
    except:
        intro.on_windows_changed(lambda: print("[shell] Windows list changed"))
        print("[shell] Watching general window changes")

    await bus.wait_for_disconnect()


async def watch_notifications() -> None:
    bus = await MessageBus().connect()

    try:
        introspection = await bus.introspect("org.freedesktop.Notifications", "/org/freedesktop/Notifications")
        proxy_obj = bus.get_proxy_object("org.freedesktop.Notifications", "/org/freedesktop/Notifications", introspection)
        notifications = proxy_obj.get_interface("org.freedesktop.Notifications")

        # Try to subscribe to available notification signals
        try:
            notifications.on_notification_closed(
                lambda nid, reason: print(f"[notify] Notification {nid} closed (reason: {reason})")
            )
        except:
            pass

        try:
            notifications.on_action_invoked(
                lambda nid, action: print(f"[notify] Action {action} invoked on notification {nid}")
            )
        except:
            pass

        print("[notify] Watching for notifications...")
        await bus.wait_for_disconnect()

    except Exception as e:
        print(f"[notify] Could not watch notifications: {e}")
        await bus.wait_for_disconnect()


async def main() -> None:
    await asyncio.gather(watch_idle(), watch_windows(), watch_notifications())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
