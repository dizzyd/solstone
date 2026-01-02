# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.transcribe validation logic."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from observe.transcribe import Transcriber, validate_transcription


class _MockTranscriber:
    """Mock transcriber for testing _prepare_audio without API key."""

    def _prepare_audio(self, raw_path):
        return Transcriber._prepare_audio(self, raw_path)


def _extract_metadata_and_items(result: list) -> tuple[dict, list]:
    """Return metadata dict and transcript items following transcribe output rules."""
    metadata = {}
    transcript_items = result
    if result and isinstance(result[-1], dict):
        last_item = result[-1]
        if "start" not in last_item and (
            "topics" in last_item or "setting" in last_item
        ):
            metadata = last_item
            transcript_items = result[:-1]
    return metadata, transcript_items


def test_validate_empty_result():
    """Empty result should be valid."""
    result = []
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_only_metadata():
    """Result with only metadata (no 'start' field) should be valid."""
    result = [{"topics": "test, demo", "setting": "personal"}]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_with_transcript_and_metadata():
    """Valid transcript with metadata should pass."""
    result = [
        {
            "start": "00:00:01",
            "source": "mic",
            "speaker": 1,
            "text": "Hello world",
            "description": "friendly",
        },
        {"topics": "greeting", "setting": "personal"},
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_without_metadata():
    """Valid transcript without metadata should pass."""
    result = [
        {
            "start": "00:00:01",
            "source": "mic",
            "speaker": 1,
            "text": "Hello world",
            "description": "friendly",
        }
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_missing_start_field():
    """Transcript item missing 'start' should fail."""
    result = [
        {
            "source": "mic",
            "speaker": 1,
            "text": "Hello world",
            "description": "friendly",
        }
    ]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "missing 'start' field" in error


def test_validate_unpadded_timestamp_format():
    """Unpadded timestamp format should pass."""
    result = [
        {
            "start": "1:2:3",  # Should be HH:MM:SS
            "source": "mic",
            "text": "Hello",
        }
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid  # This should pass - we only check format, not padding


def test_validate_invalid_timestamp_not_string():
    """Timestamp that's not a string should fail."""
    result = [{"start": 123, "source": "mic", "text": "Hello"}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "'start' is not a string" in error


def test_validate_invalid_timestamp_format_not_three_parts():
    """Timestamp without three parts should fail."""
    result = [{"start": "00:00", "source": "mic", "text": "Hello"}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "not in HH:MM:SS format" in error


def test_validate_missing_text_field():
    """Transcript item missing 'text' should fail."""
    result = [{"start": "00:00:01", "source": "mic", "speaker": 1}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "missing 'text' field" in error


def test_validate_text_not_string():
    """Text field that's not a string should fail."""
    result = [{"start": "00:00:01", "source": "mic", "text": 123}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "'text' is not a string" in error


def test_validate_result_not_list():
    """Result that's not a list should fail."""
    result = {"start": "00:00:01"}
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "not a list" in error


def test_validate_item_not_dict():
    """Item that's not a dict should fail."""
    result = ["invalid", {"topics": "test"}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "not a dictionary" in error


def test_validate_multiple_transcript_items():
    """Multiple valid transcript items should pass."""
    result = [
        {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "First"},
        {"start": "00:00:05", "source": "sys", "speaker": 2, "text": "Second"},
        {"start": "00:00:10", "source": "mic", "speaker": 1, "text": "Third"},
        {"topics": "conversation", "setting": "personal"},
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_string_speaker_labels():
    """String speaker labels (from diarization) should pass."""
    result = [
        {
            "start": "00:00:01",
            "speaker": "Speaker 1",
            "text": "Hello world",
            "description": "friendly",
        },
        {
            "start": "00:00:05",
            "speaker": "Speaker 2",
            "text": "Hi there",
            "description": "casual",
        },
        {"topics": "greeting", "setting": "personal"},
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_jsonl_format_with_metadata():
    """Test JSONL format with metadata first."""
    result = [
        {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "Hello"},
        {"start": "00:00:05", "source": "sys", "speaker": 2, "text": "Hi"},
        {"topics": "greeting", "setting": "personal"},
    ]

    # Extract metadata and transcript items (mimics _transcribe logic)
    metadata, transcript_items = _extract_metadata_and_items(result)

    # Write JSONL format
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(item) for item in transcript_items)
    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Verify format
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 3

    # First line should be metadata
    first = json.loads(lines[0])
    assert first == {"topics": "greeting", "setting": "personal"}

    # Remaining lines should be transcript items
    second = json.loads(lines[1])
    assert second["start"] == "00:00:01"
    assert second["text"] == "Hello"

    third = json.loads(lines[2])
    assert third["start"] == "00:00:05"
    assert third["text"] == "Hi"


def test_jsonl_format_without_metadata():
    """Test JSONL format with empty metadata when none provided."""
    result = [
        {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "Hello"},
    ]

    # Extract metadata and transcript items (mimics _transcribe logic)
    metadata, transcript_items = _extract_metadata_and_items(result)

    # Write JSONL format
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(item) for item in transcript_items)
    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Verify format
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 2

    # First line should be empty metadata
    first = json.loads(lines[0])
    assert first == {}

    # Second line should be transcript item
    second = json.loads(lines[1])
    assert second["start"] == "00:00:01"
    assert second["text"] == "Hello"


def test_jsonl_format_empty_result():
    """Test JSONL format with empty result (only metadata line)."""
    result = []

    # Extract metadata and transcript items (mimics _transcribe logic)
    metadata, transcript_items = _extract_metadata_and_items(result)

    # Write JSONL format
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(item) for item in transcript_items)
    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Verify format
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 1

    # Only line should be empty metadata
    first = json.loads(lines[0])
    assert first == {}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_prepare_audio_multi_track_m4a():
    """Test that _prepare_audio mixes multiple M4A audio streams together."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two mono FLAC files to combine into multi-track M4A
        track0_path = Path(tmpdir) / "track0.flac"
        track1_path = Path(tmpdir) / "track1.flac"
        m4a_path = Path(tmpdir) / "test.m4a"

        # Track 0: silence (system audio - no content)
        # Track 1: 440Hz sine wave (microphone - has voice)
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        track0_data = np.zeros_like(t)  # Silence
        track1_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone

        sf.write(track0_path, track0_data, sample_rate, format="FLAC")
        sf.write(track1_path, track1_data, sample_rate, format="FLAC")

        # Use ffmpeg to create multi-track M4A (same structure as sck-cli output)
        # This creates an M4A with 2 separate mono audio streams
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(track0_path),
                "-i",
                str(track1_path),
                "-map",
                "0:a",
                "-map",
                "1:a",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(m4a_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ffmpeg failed: {result.stderr}"

        transcriber = _MockTranscriber()
        temp_flac = transcriber._prepare_audio(m4a_path)

        try:
            assert temp_flac.exists()
            assert temp_flac.suffix == ".flac"

            # Read the output and verify both streams were mixed
            mixed_data, sr = sf.read(temp_flac, dtype="float32")

            # The mixed audio should have content from track 1 (the sine wave)
            # AAC compression affects amplitude, so use loose threshold
            rms = np.sqrt(np.mean(mixed_data**2))
            assert rms > 0.1, f"Mixed audio should contain signal, got RMS={rms}"

            # Verify sample rate matches expected
            assert sr == 16000
        finally:
            if temp_flac.exists():
                temp_flac.unlink()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_prepare_audio_single_stream_m4a():
    """Test that _prepare_audio handles single-stream M4A correctly."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        track_path = Path(tmpdir) / "track.flac"
        m4a_path = Path(tmpdir) / "single.m4a"

        # Single track with 440Hz sine wave
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        track_data = 0.5 * np.sin(2 * np.pi * 440 * t)

        sf.write(track_path, track_data, sample_rate, format="FLAC")

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(track_path),
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(m4a_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ffmpeg failed: {result.stderr}"

        transcriber = _MockTranscriber()
        temp_flac = transcriber._prepare_audio(m4a_path)

        try:
            assert temp_flac.exists()

            mixed_data, sr = sf.read(temp_flac, dtype="float32")
            rms = np.sqrt(np.mean(mixed_data**2))
            # Single stream should preserve the signal
            assert rms > 0.3, f"Single stream should have strong signal, got RMS={rms}"
        finally:
            if temp_flac.exists():
                temp_flac.unlink()


def test_prepare_audio_flac_passthrough():
    """Test that _prepare_audio returns FLAC files unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        flac_path = Path(tmpdir) / "test.flac"

        sample_rate = 16000
        data = np.zeros(sample_rate, dtype=np.float32)
        sf.write(flac_path, data, sample_rate, format="FLAC")

        transcriber = _MockTranscriber()
        result = transcriber._prepare_audio(flac_path)

        # FLAC should be returned as-is (not converted)
        assert result == flac_path
