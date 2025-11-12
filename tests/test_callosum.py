"""Unit tests for the Callosum message bus.

These tests use mocks to test logic in isolation without real I/O.
"""

import json
import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from think.callosum import CallosumConnection, CallosumServer


@pytest.fixture
def journal_path(tmp_path):
    """Set up a temporary journal path."""
    journal = tmp_path / "journal"
    journal.mkdir()
    os.environ["JOURNAL_PATH"] = str(journal)
    yield journal
    # Cleanup
    if "JOURNAL_PATH" in os.environ:
        del os.environ["JOURNAL_PATH"]


def test_server_broadcast_validates_tract_field():
    """Test that messages without tract field are rejected."""
    server = CallosumServer()
    server.clients = [Mock()]

    # Message without tract should be rejected
    invalid_msg = {"event": "test"}
    server.broadcast(invalid_msg)

    # Client should not receive anything
    server.clients[0].sendall.assert_not_called()


def test_server_broadcast_validates_event_field():
    """Test that messages without event field are rejected."""
    server = CallosumServer()
    server.clients = [Mock()]

    # Message without event should be rejected
    invalid_msg = {"tract": "test"}
    server.broadcast(invalid_msg)

    # Client should not receive anything
    server.clients[0].sendall.assert_not_called()


def test_server_broadcast_adds_timestamp():
    """Test that server adds timestamp if not present."""
    server = CallosumServer()
    mock_client = Mock()
    server.clients = [mock_client]

    # Valid message without timestamp
    msg = {"tract": "test", "event": "hello"}

    with patch("time.time", return_value=1234567.890):
        server.broadcast(msg)

    # Should have called sendall with message including timestamp
    mock_client.sendall.assert_called_once()
    sent_data = mock_client.sendall.call_args[0][0]
    sent_msg = json.loads(sent_data.decode("utf-8"))

    assert sent_msg["tract"] == "test"
    assert sent_msg["event"] == "hello"
    assert sent_msg["ts"] == 1234567890  # milliseconds


def test_server_broadcast_preserves_custom_timestamp():
    """Test that custom timestamp in message is preserved."""
    server = CallosumServer()
    mock_client = Mock()
    server.clients = [mock_client]

    custom_ts = 9999999999
    msg = {"tract": "test", "event": "hello", "ts": custom_ts}

    server.broadcast(msg)

    # Should preserve custom timestamp
    sent_data = mock_client.sendall.call_args[0][0]
    sent_msg = json.loads(sent_data.decode("utf-8"))
    assert sent_msg["ts"] == custom_ts


def test_server_broadcast_removes_dead_clients():
    """Test that broadcast removes clients that fail to receive."""
    server = CallosumServer()

    # Create mock clients - one working, one dead
    working_client = Mock()
    dead_client = Mock()
    dead_client.sendall.side_effect = Exception("Connection broken")

    server.clients = [working_client, dead_client]

    msg = {"tract": "test", "event": "hello"}
    server.broadcast(msg)

    # Dead client should be removed
    assert working_client in server.clients
    assert dead_client not in server.clients
    assert len(server.clients) == 1

    # Dead client socket should be closed
    dead_client.close.assert_called_once()


def test_client_emit_returns_false_when_not_started():
    """Test that emit() returns False and logs warning if start() not called yet."""
    client = CallosumConnection()

    # emit() should return False and log when thread not started
    with patch("think.callosum.logger") as mock_logger:
        result = client.emit("test", "hello")
        assert result is False
        mock_logger.warning.assert_called_once()
        assert "Thread not running" in mock_logger.warning.call_args[0][0]


def test_client_emit_queues_message():
    """Test that emit() queues message when thread is running."""
    client = CallosumConnection()

    # Setup running thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    client.thread = mock_thread

    result = client.emit("test", "hello", data="world", count=42)

    assert result is True
    # Message should be in queue
    assert client.send_queue.qsize() == 1
    msg = client.send_queue.get_nowait()
    assert msg["tract"] == "test"
    assert msg["event"] == "hello"
    assert msg["data"] == "world"
    assert msg["count"] == 42


def test_client_emit_returns_false_when_queue_full():
    """Test that emit() returns False when queue is full."""
    client = CallosumConnection()

    # Setup running thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    client.thread = mock_thread

    # Fill the queue
    for i in range(1000):
        client.send_queue.put({"tract": "test", "event": f"msg{i}"})

    # Next emit should fail
    with patch("think.callosum.logger") as mock_logger:
        result = client.emit("test", "overflow")
        assert result is False
        mock_logger.warning.assert_called()
        assert "Queue full" in mock_logger.warning.call_args[0][0]


def test_client_start_creates_thread():
    """Test that start() creates and starts background thread."""
    client = CallosumConnection()

    def callback(msg):
        pass

    client.start(callback=callback)

    assert client.thread is not None
    assert client.thread.is_alive()
    assert client.callback is callback

    # Cleanup
    client.stop()


def test_client_start_idempotent():
    """Test that calling start() multiple times is safe."""
    client = CallosumConnection()

    client.start()
    first_thread = client.thread

    # Call start again
    client.start()

    # Should still have same thread (not restarted)
    assert client.thread is first_thread

    # Cleanup
    client.stop()


def test_client_stop_stops_thread():
    """Test that stop() stops the background thread."""
    client = CallosumConnection()

    # Setup running thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = False
    client.thread = mock_thread

    client.stop()

    # Should set stop event and join thread
    assert client.stop_event.is_set()
    mock_thread.join.assert_called_once_with(timeout=0.5)


def test_server_socket_path_from_env(journal_path):
    """Test that server uses JOURNAL_PATH env var for socket path."""
    server = CallosumServer()

    expected_path = journal_path / "health" / "callosum.sock"
    assert server.socket_path == expected_path


def test_server_socket_path_custom():
    """Test that server accepts custom socket path."""
    custom_path = Path("/tmp/custom.sock")
    server = CallosumServer(socket_path=custom_path)

    assert server.socket_path == custom_path


def test_client_socket_path_from_env(journal_path):
    """Test that client uses JOURNAL_PATH env var for socket path."""
    client = CallosumConnection()

    expected_path = journal_path / "health" / "callosum.sock"
    assert client.socket_path == expected_path


def test_client_socket_path_custom():
    """Test that client accepts custom socket path."""
    custom_path = Path("/tmp/custom.sock")
    client = CallosumConnection(socket_path=custom_path)

    assert client.socket_path == custom_path


def test_callosum_send_without_journal_path():
    """Test that callosum_send() returns False when JOURNAL_PATH not set."""
    from think.callosum import callosum_send

    # Temporarily remove JOURNAL_PATH
    old_path = os.environ.get("JOURNAL_PATH")
    if "JOURNAL_PATH" in os.environ:
        del os.environ["JOURNAL_PATH"]

    try:
        result = callosum_send("test", "event", data="value")
        assert result is False
    finally:
        # Restore JOURNAL_PATH
        if old_path:
            os.environ["JOURNAL_PATH"] = old_path


def test_callosum_send_with_custom_path():
    """Test that callosum_send() accepts custom socket path."""
    from think.callosum import callosum_send

    # Use non-existent socket - should return False but not crash
    custom_path = Path("/tmp/nonexistent_callosum.sock")
    result = callosum_send("test", "event", socket_path=custom_path, data="value")

    # Should fail gracefully (no server listening)
    assert result is False
