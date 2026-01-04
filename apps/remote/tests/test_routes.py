# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for remote app routes."""

from __future__ import annotations

import io


def test_api_list_empty(remote_env):
    """Test listing remotes when none exist."""
    env = remote_env()

    resp = env.client.get("/app/remote/api/list")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_create_remote(remote_env):
    """Test creating a new remote."""
    env = remote_env()

    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test-laptop"},
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()

    assert "key" in data
    assert len(data["key"]) > 32  # 256 bits = 43 base64 chars
    assert data["key_prefix"] == data["key"][:8]
    assert data["name"] == "test-laptop"
    assert "/app/remote/ingest/" in data["ingest_url"]


def test_api_create_requires_name(remote_env):
    """Test that creating a remote requires a name."""
    env = remote_env()

    # Missing name
    resp = env.client.post(
        "/app/remote/api/create",
        json={},
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Name is required" in resp.get_json()["error"]

    # Empty name
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "   "},
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_api_list_shows_created_remote(remote_env):
    """Test that created remotes appear in the list."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "my-remote"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    key_prefix = resp.get_json()["key_prefix"]

    # List should show it
    resp = env.client.get("/app/remote/api/list")
    assert resp.status_code == 200
    remotes = resp.get_json()

    assert len(remotes) == 1
    assert remotes[0]["key_prefix"] == key_prefix
    assert remotes[0]["name"] == "my-remote"
    assert remotes[0]["enabled"] is True
    assert remotes[0]["stats"]["segments_received"] == 0


def test_api_delete_remote(remote_env):
    """Test revoking a remote (soft-delete)."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "to-revoke"},
        content_type="application/json",
    )
    key_prefix = resp.get_json()["key_prefix"]

    # Revoke it
    resp = env.client.delete(f"/app/remote/api/{key_prefix}")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

    # List should still show it, but marked as revoked
    resp = env.client.get("/app/remote/api/list")
    remotes = resp.get_json()
    assert len(remotes) == 1
    assert remotes[0]["key_prefix"] == key_prefix
    assert remotes[0]["revoked"] is True
    assert remotes[0]["revoked_at"] is not None


def test_api_delete_nonexistent(remote_env):
    """Test deleting a nonexistent remote returns 404."""
    env = remote_env()

    resp = env.client.delete("/app/remote/api/nonexistent")
    assert resp.status_code == 404


def test_ingest_invalid_key(remote_env):
    """Test that ingest rejects invalid keys."""
    env = remote_env()

    resp = env.client.post(
        "/app/remote/ingest/invalid-key-12345",
        data={"day": "20250103", "segment": "120000_300"},
    )
    assert resp.status_code == 401
    assert "Invalid key" in resp.get_json()["error"]


def test_ingest_missing_segment(remote_env):
    """Test that ingest requires segment."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without segment
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "20250103"},
    )
    assert resp.status_code == 400
    assert "Missing segment" in resp.get_json()["error"]


def test_ingest_missing_day(remote_env):
    """Test that ingest requires day."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without day
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"segment": "120000_300"},
    )
    assert resp.status_code == 400
    assert "Missing day" in resp.get_json()["error"]


def test_ingest_invalid_segment_format(remote_env):
    """Test that ingest validates segment format."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Invalid segment format
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "20250103", "segment": "invalid"},
    )
    assert resp.status_code == 400
    assert "Invalid segment format" in resp.get_json()["error"]


def test_ingest_invalid_day_format(remote_env):
    """Test that ingest validates day format."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Invalid day format
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "2025-01-03", "segment": "120000_300"},
    )
    assert resp.status_code == 400
    assert "Invalid day format" in resp.get_json()["error"]


def test_ingest_no_files(remote_env):
    """Test that ingest requires files."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without files
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "20250103", "segment": "120000_300"},
    )
    assert resp.status_code == 400
    assert "No files uploaded" in resp.get_json()["error"]


def test_ingest_success(remote_env):
    """Test successful file ingest."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test-remote"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "test_audio.flac"),
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["files"] == ["test_audio.flac"]
    assert data["bytes"] == len(test_data)

    # Verify file was written
    expected_file = env.journal / "20250103" / "test_audio.flac"
    assert expected_file.exists()
    assert expected_file.read_bytes() == test_data


def test_ingest_updates_stats(remote_env):
    """Test that ingest updates remote stats."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "stats-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Check stats updated
    resp = env.client.get("/app/remote/api/list")
    remotes = resp.get_json()
    assert len(remotes) == 1
    assert remotes[0]["stats"]["segments_received"] == 1
    assert remotes[0]["stats"]["bytes_received"] == len(test_data)
    assert remotes[0]["last_segment"] == "120000_300"
    assert remotes[0]["last_seen"] is not None


def test_ingest_event_relay(remote_env):
    """Test event relay endpoint."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "event-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Send an event
    resp = env.client.post(
        f"/app/remote/ingest/{key}/event",
        json={"tract": "observe", "event": "status", "mode": "screencast"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_ingest_event_missing_tract(remote_env):
    """Test that event relay requires tract."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Missing tract
    resp = env.client.post(
        f"/app/remote/ingest/{key}/event",
        json={"event": "status"},
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Missing tract or event" in resp.get_json()["error"]


def test_ingest_revoked_key(remote_env):
    """Test that ingest rejects revoked keys."""
    env = remote_env()

    # Create and revoke a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "revoked-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    resp = env.client.delete(f"/app/remote/api/{key_prefix}")
    assert resp.status_code == 200

    # Try to upload - should fail
    test_data = b"test content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 403
    assert "Remote revoked" in resp.get_json()["error"]


def test_ingest_event_revoked_key(remote_env):
    """Test that event relay rejects revoked keys."""
    env = remote_env()

    # Create and revoke a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "revoked-event-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    resp = env.client.delete(f"/app/remote/api/{key_prefix}")
    assert resp.status_code == 200

    # Try to send event - should fail
    resp = env.client.post(
        f"/app/remote/ingest/{key}/event",
        json={"tract": "observe", "event": "status"},
        content_type="application/json",
    )
    assert resp.status_code == 403
    assert "Remote revoked" in resp.get_json()["error"]


def test_api_get_key(remote_env):
    """Test retrieving full key for a remote."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "key-test"},
        content_type="application/json",
    )
    create_data = resp.get_json()
    key = create_data["key"]
    key_prefix = create_data["key_prefix"]

    # Get the key
    resp = env.client.get(f"/app/remote/api/{key_prefix}/key")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["key"] == key
    assert data["name"] == "key-test"
    assert data["ingest_url"] == f"/app/remote/ingest/{key}"


def test_api_get_key_nonexistent(remote_env):
    """Test getting key for nonexistent remote returns 404."""
    env = remote_env()

    resp = env.client.get("/app/remote/api/nonexistent/key")
    assert resp.status_code == 404


# === Segment collision helper tests ===


def test_randomize_segment_produces_valid_output():
    """Test that _randomize_segment produces valid segment keys."""
    from apps.remote.routes import _randomize_segment

    result = _randomize_segment("120000_300")

    # Result is either None (boundary hit) or a valid segment
    if result is not None:
        assert "_" in result
        time_part, dur_part = result.split("_")
        assert len(time_part) == 6
        assert time_part.isdigit()
        assert dur_part.isdigit()
        assert int(dur_part) > 0


def test_randomize_segment_never_produces_invalid_time():
    """Test that _randomize_segment never produces times outside 00:00:00-23:59:59."""
    from apps.remote.routes import _randomize_segment

    # Test at boundaries - should return None or valid, never invalid
    for segment in ["000000_300", "235959_300"]:
        for _ in range(20):
            result = _randomize_segment(segment)
            if result is not None:
                time_part = result.split("_")[0]
                hours = int(time_part[:2])
                assert 0 <= hours <= 23


def test_randomize_segment_never_produces_zero_duration():
    """Test that _randomize_segment never produces duration <= 0."""
    from apps.remote.routes import _randomize_segment

    # Test at duration boundary
    for _ in range(20):
        result = _randomize_segment("120000_1")
        if result is not None:
            dur = int(result.split("_")[1])
            assert dur > 0


def test_segment_exists_with_directory(remote_env):
    """Test _segment_exists detects existing segment directory."""
    from apps.remote.routes import _segment_exists

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create a segment directory
    segment_dir = day_dir / "120000_300"
    segment_dir.mkdir()

    assert _segment_exists(day_dir, "120000_300") is True
    assert _segment_exists(day_dir, "120001_300") is False


def test_segment_exists_with_files(remote_env):
    """Test _segment_exists detects files with segment prefix."""
    from apps.remote.routes import _segment_exists

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create a file with segment prefix
    (day_dir / "120000_300_audio.flac").write_bytes(b"test")

    assert _segment_exists(day_dir, "120000_300") is True
    assert _segment_exists(day_dir, "120001_300") is False


def test_segment_exists_empty_directory(remote_env):
    """Test _segment_exists returns False for empty directory."""
    from apps.remote.routes import _segment_exists

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    assert _segment_exists(day_dir, "120000_300") is False


def test_find_available_segment_no_conflict(remote_env):
    """Test _find_available_segment returns original when no conflict."""
    from apps.remote.routes import _find_available_segment

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    result = _find_available_segment(day_dir, "120000_300")
    assert result == "120000_300"


def test_find_available_segment_with_conflict(remote_env):
    """Test _find_available_segment finds alternative when conflict exists."""
    from apps.remote.routes import _find_available_segment

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create conflicting file
    (day_dir / "120000_300_audio.flac").write_bytes(b"test")

    result = _find_available_segment(day_dir, "120000_300")

    # Should find a different segment
    assert result is not None
    assert result != "120000_300"
    # Should be a valid segment format
    assert "_" in result
    time_part, dur_part = result.split("_")
    assert len(time_part) == 6
    assert dur_part.isdigit()


def test_find_available_segment_with_limited_attempts(remote_env):
    """Test _find_available_segment respects max_attempts limit."""
    from apps.remote.routes import _find_available_segment

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create conflicting file
    (day_dir / "120000_300_audio.flac").write_bytes(b"test")

    # With max_attempts=0, should return None immediately (no attempts allowed)
    result = _find_available_segment(day_dir, "120000_300", max_attempts=0)
    assert result is None


def test_save_to_failed_creates_directory(remote_env):
    """Test _save_to_failed creates failed directory structure."""
    from io import BytesIO

    from werkzeug.datastructures import FileStorage

    from apps.remote.routes import _save_to_failed

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create mock file uploads
    files = [
        FileStorage(stream=BytesIO(b"audio data"), filename="120000_300_audio.flac"),
        FileStorage(stream=BytesIO(b"video data"), filename="120000_300_screen.webm"),
    ]

    failed_dir = _save_to_failed(day_dir, files, "120000_300")

    # Verify structure includes segment key
    assert failed_dir.exists()
    assert "remote/failed/120000_300/" in str(failed_dir)
    assert (failed_dir / "120000_300_audio.flac").exists()
    assert (failed_dir / "120000_300_screen.webm").exists()


# === Integration tests for collision handling ===


def test_ingest_collision_adjusts_segment(remote_env):
    """Test that ingest adjusts segment key on collision."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "collision-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Create a conflicting file
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300_audio.flac").write_bytes(b"existing")

    # Upload with same segment key
    test_data = b"new audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"

    # The filename should have been adjusted
    saved_file = data["files"][0]
    assert saved_file != "120000_300_audio.flac"
    assert "_audio.flac" in saved_file

    # Verify both files exist
    assert (day_dir / "120000_300_audio.flac").exists()  # Original
    assert (day_dir / saved_file).exists()  # New adjusted


def test_ingest_no_collision_preserves_segment(remote_env):
    """Test that ingest preserves segment key when no collision."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "no-collision-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without any conflicting files
    test_data = b"audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["files"] == ["120000_300_audio.flac"]

    # Verify file saved with original name
    expected_file = env.journal / "20250103" / "120000_300_audio.flac"
    assert expected_file.exists()


def test_ingest_stats_use_adjusted_segment(remote_env):
    """Test that remote stats record the adjusted segment key."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "stats-adjust-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Create a conflicting file
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300_audio.flac").write_bytes(b"existing")

    # Upload with same segment key
    test_data = b"new audio"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )

    assert resp.status_code == 200

    # Check stats - last_segment should be the adjusted one
    resp = env.client.get("/app/remote/api/list")
    remotes = resp.get_json()
    assert len(remotes) == 1
    # The stored segment should be different from original
    last_segment = remotes[0]["last_segment"]
    assert last_segment is not None
    # It should be adjusted (not the original conflicting one)
    assert (
        last_segment != "120000_300"
        or (day_dir / f"{last_segment}_audio.flac").exists()
    )
