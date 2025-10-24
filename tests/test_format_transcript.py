"""Tests for transcript text formatting utility."""

import json
from pathlib import Path

import pytest

from observe.utils import format_transcript_text


def test_format_transcript_basic(tmp_path):
    """Test basic transcript formatting with metadata and entries."""
    # Create test transcript file
    day_dir = tmp_path / "20250615"
    day_dir.mkdir()
    transcript_file = day_dir / "100500_audio.jsonl"

    # Write JSONL with metadata and entries
    metadata = {"topics": ["meeting", "planning"], "setting": "work"}
    entries = [
        {"start": "00:01:23", "source": "mic", "speaker": 1, "text": "Hello world"},
        {"start": "00:01:25", "source": "sys", "speaker": 2, "text": "Hi there"},
        {"start": "00:01:30", "source": "mic", "speaker": 1, "text": "How are you?"},
    ]

    lines = [json.dumps(metadata)]
    lines.extend(json.dumps(entry) for entry in entries)
    transcript_file.write_text("\n".join(lines) + "\n")

    # Format transcript
    result = format_transcript_text(transcript_file)

    # Verify output
    assert "Start: 2025-06-15 10:05am" in result
    assert "Topics: meeting, planning" in result
    assert "Setting: work" in result
    assert "[00:01:23] (mic) Speaker 1: Hello world" in result
    assert "[00:01:25] (sys) Speaker 2: Hi there" in result
    assert "[00:01:30] (mic) Speaker 1: How are you?" in result


def test_format_transcript_minimal(tmp_path):
    """Test formatting with minimal metadata."""
    day_dir = tmp_path / "20250615"
    day_dir.mkdir()
    transcript_file = day_dir / "100500_audio.jsonl"

    # Minimal metadata (empty dict)
    metadata = {}
    entries = [
        {"start": "00:00:01", "text": "Simple text"},
        {"start": "00:00:05", "source": "mic", "text": "With source"},
    ]

    lines = [json.dumps(metadata)]
    lines.extend(json.dumps(entry) for entry in entries)
    transcript_file.write_text("\n".join(lines) + "\n")

    result = format_transcript_text(transcript_file)

    assert "Start: 2025-06-15 10:05am" in result
    assert "[00:00:01] Simple text" in result
    assert "[00:00:05] (mic) With source" in result


def test_format_transcript_imported(tmp_path):
    """Test formatting of imported transcript with domain metadata."""
    day_dir = tmp_path / "20250615"
    day_dir.mkdir()
    transcript_file = day_dir / "100500_imported_audio.jsonl"

    metadata = {
        "imported": {"id": "abc123", "domain": "uavionix"},
        "topics": ["discussion"],
    }
    entries = [
        {
            "start": "00:00:10",
            "source": "mic",
            "speaker": 1,
            "text": "Imported content",
        }
    ]

    lines = [json.dumps(metadata)]
    lines.extend(json.dumps(entry) for entry in entries)
    transcript_file.write_text("\n".join(lines) + "\n")

    result = format_transcript_text(transcript_file)

    assert "Start: 2025-06-15 10:05am" in result
    assert "Topics: discussion" in result
    assert "Domain: uavionix" in result
    assert "Import ID: abc123" in result
    assert "[00:00:10] (mic) Speaker 1: Imported content" in result


def test_format_transcript_no_speaker(tmp_path):
    """Test formatting entries without speaker information."""
    day_dir = tmp_path / "20250615"
    day_dir.mkdir()
    transcript_file = day_dir / "100500_audio.jsonl"

    metadata = {"setting": "personal"}
    entries = [
        {"start": "00:00:01", "source": "mic", "text": "No speaker here"},
        {"start": "00:00:05", "text": "No source or speaker"},
    ]

    lines = [json.dumps(metadata)]
    lines.extend(json.dumps(entry) for entry in entries)
    transcript_file.write_text("\n".join(lines) + "\n")

    result = format_transcript_text(transcript_file)

    assert "[00:00:01] (mic) No speaker here" in result
    assert "[00:00:05] No source or speaker" in result


def test_format_transcript_error(tmp_path):
    """Test error handling for missing file."""
    nonexistent = tmp_path / "20250615" / "nonexistent.jsonl"
    result = format_transcript_text(nonexistent)

    assert "Error loading transcript:" in result
    assert "not found" in result.lower()


def test_format_transcript_no_day_in_path(tmp_path):
    """Test formatting when day can't be parsed from path."""
    # File not in a YYYYMMDD directory
    transcript_file = tmp_path / "100500_audio.jsonl"

    metadata = {"setting": "test"}
    entries = [{"start": "00:00:01", "text": "Test"}]

    lines = [json.dumps(metadata)]
    lines.extend(json.dumps(entry) for entry in entries)
    transcript_file.write_text("\n".join(lines) + "\n")

    result = format_transcript_text(transcript_file)

    # Should still work, just without "Start:" header
    assert "Setting: test" in result
    assert "[00:00:01] Test" in result
    # Start header should not be present since we couldn't parse the day
    assert "Start:" not in result
