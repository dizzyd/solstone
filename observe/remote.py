# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Remote observer support and sync client.

This module provides:
- ObserverBackend: Local-only backend for observers (Callosum events)
- RemoteClient: HTTP client for uploading segments and relaying events (used by sync.py)
- Staging path management for observers
"""

from __future__ import annotations

import logging
import platform
import queue
import socket
import threading
import time
from pathlib import Path

import requests

from think.callosum import CallosumConnection
from think.utils import get_journal

logger = logging.getLogger(__name__)

# Host identification (captured once at module load)
# Exported for use by platform observers
HOST = socket.gethostname()
PLATFORM = platform.system().lower()  # "linux", "darwin", "windows"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 5, 15]  # seconds
UPLOAD_TIMEOUT = 300  # 5 minutes for large files


class RemoteClient:
    """Client for uploading segment files and events to a remote server."""

    def __init__(self, remote_url: str):
        """Initialize remote client.

        Args:
            remote_url: Full URL to remote ingest endpoint (including key)
                       e.g., "https://server:5000/app/remote/ingest/abc123..."
        """
        self.remote_url = remote_url.rstrip("/")
        self.event_url = f"{self.remote_url}/event"
        self.session = requests.Session()

        # Event queue for async sending
        self._event_queue: queue.Queue = queue.Queue(maxsize=100)
        self._event_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background event sender thread."""
        if self._event_thread and self._event_thread.is_alive():
            return

        self._stop_event.clear()
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()

    def stop(self) -> None:
        """Stop background event sender thread."""
        self._stop_event.set()
        if self._event_thread:
            self._event_thread.join(timeout=2.0)

    def _event_loop(self) -> None:
        """Background loop to send queued events."""
        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=1.0)
                self._send_event_sync(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Event send error: {e}")

    def _send_event_sync(self, event: dict) -> bool:
        """Send a single event to remote server."""
        for attempt, delay in enumerate(RETRY_BACKOFF):
            try:
                response = self.session.post(
                    self.event_url,
                    json=event,
                    timeout=10,
                )
                if response.status_code == 200:
                    return True
                logger.warning(
                    f"Event send failed: {response.status_code} {response.text}"
                )
            except requests.RequestException as e:
                logger.warning(f"Event send attempt {attempt + 1} failed: {e}")

            if attempt < len(RETRY_BACKOFF) - 1:
                time.sleep(delay)

        logger.error(f"Event send failed after {MAX_RETRIES} attempts")
        return False

    def emit(self, tract: str, event: str, **fields) -> bool:
        """Queue an event to be sent to remote server.

        Args:
            tract: Event tract (e.g., "observe")
            event: Event name (e.g., "status")
            **fields: Additional event fields

        Returns:
            True if queued successfully, False if queue full
        """
        message = {"tract": tract, "event": event, **fields}
        try:
            self._event_queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning(f"Event queue full, dropping: {tract}/{event}")
            return False

    def upload_segment(
        self,
        day: str,
        segment: str,
        files: list[Path],
    ) -> bool:
        """Upload segment files to remote server.

        Args:
            day: Day string (YYYYMMDD)
            segment: Segment key (HHMMSS_LEN)
            files: List of file paths to upload

        Returns:
            True if upload succeeded, False otherwise
        """
        if not files:
            logger.warning("No files to upload")
            return False

        for attempt, delay in enumerate(RETRY_BACKOFF):
            # Open file handles and ensure they're closed
            file_handles = []
            files_data = []
            try:
                # Build files list for requests
                for path in files:
                    if not path.exists():
                        logger.warning(f"File not found, skipping: {path}")
                        continue
                    fh = open(path, "rb")
                    file_handles.append(fh)
                    files_data.append(
                        ("files", (path.name, fh, "application/octet-stream"))
                    )

                if not files_data:
                    logger.error("No valid files to upload")
                    return False

                # Send request with host/platform for event emission
                response = self.session.post(
                    self.remote_url,
                    data={
                        "day": day,
                        "segment": segment,
                        "host": HOST,
                        "platform": PLATFORM,
                    },
                    files=files_data,
                    timeout=UPLOAD_TIMEOUT,
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"Uploaded {len(result.get('files', []))} files "
                        f"({result.get('bytes', 0)} bytes) for {day}/{segment}"
                    )
                    return True

                logger.warning(f"Upload failed: {response.status_code} {response.text}")

            except requests.RequestException as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")

            finally:
                # Always close file handles
                for fh in file_handles:
                    try:
                        fh.close()
                    except Exception:
                        pass

            if attempt < len(RETRY_BACKOFF) - 1:
                logger.info(f"Retrying upload in {delay}s...")
                time.sleep(delay)

        logger.error(f"Upload failed after {MAX_RETRIES} attempts: {day}/{segment}")
        return False

    def upload_and_cleanup(
        self,
        day: str,
        segment: str,
        files: list[Path],
    ) -> bool:
        """Upload segment files and delete local copies on success.

        Args:
            day: Day string (YYYYMMDD)
            segment: Segment key (HHMMSS_LEN)
            files: List of file paths to upload

        Returns:
            True if upload succeeded and files deleted, False otherwise
        """
        if self.upload_segment(day, segment, files):
            # Delete local files on success
            for path in files:
                try:
                    if path.exists():
                        path.unlink()
                        logger.debug(f"Deleted local file: {path}")
                except OSError as e:
                    logger.warning(f"Failed to delete {path}: {e}")
            return True
        return False


class ObserverBackend:
    """Backend for observer segment completion and status events.

    Handles local Callosum events for observers. For remote sync, use
    the separate sync.py service which uses RemoteClient directly.
    """

    def __init__(self):
        """Initialize the observer backend."""
        self._callosum: CallosumConnection | None = None
        self._staging_path: Path | None = None

    @property
    def staging_path(self) -> Path:
        """Get the staging path for segment files.

        Uses get_journal() which auto-creates a platform-default path if needed.
        """
        if self._staging_path is not None:
            return self._staging_path

        self._staging_path = Path(get_journal())
        return self._staging_path

    def start(self) -> None:
        """Start the backend (connects to Callosum)."""
        self._callosum = CallosumConnection()
        self._callosum.start()
        logger.info("Backend started (Callosum)")

    def stop(self) -> None:
        """Stop the backend."""
        if self._callosum:
            self._callosum.stop()
            self._callosum = None

    def emit(self, tract: str, event: str, **fields) -> bool:
        """Emit an event (status, observing, etc).

        Automatically adds host and platform fields.

        Args:
            tract: Event tract (e.g., "observe")
            event: Event name (e.g., "status")
            **fields: Additional event fields

        Returns:
            True if emitted successfully
        """
        # Always include host/platform
        fields.setdefault("host", HOST)
        fields.setdefault("platform", PLATFORM)

        if self._callosum:
            self._callosum.emit(tract, event, **fields)
            return True
        return False

    def segment_complete(
        self,
        day: str,
        segment: str,
        file_paths: list[Path],
    ) -> bool:
        """Handle segment completion - emit observe.observing event.

        Args:
            day: Day string (YYYYMMDD)
            segment: Segment key (HHMMSS_LEN)
            file_paths: List of file paths in the segment

        Returns:
            True if successful
        """
        if not file_paths:
            return True

        if self._callosum:
            file_names = [f.name for f in file_paths]
            self._callosum.emit(
                "observe",
                "observing",
                day=day,
                segment=segment,
                files=file_names,
                host=HOST,
                platform=PLATFORM,
            )
            logger.info(f"Segment observing: {segment} ({len(file_names)} files)")
            return True

        return False


def staging_day_path(base: Path, day: str) -> Path:
    """Create and return day directory under staging base.

    Args:
        base: Staging base path (from ObserverBackend.staging_path)
        day: Day string (YYYYMMDD)

    Returns:
        Path to day directory (created if needed)
    """
    day_dir = base / day
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir
