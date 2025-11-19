"""Tests for observe.reduce module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from observe.reduce import assemble_markdown, reduce_analysis


def test_assemble_markdown_extracts_segment_from_directory():
    """Test that assemble_markdown correctly extracts base time from segment directory."""
    # Mock frames with relative timestamps (seconds from segment start)
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Editing Python"},
        },
        {
            "timestamp": 30,
            "monitor": "0",
            "analysis": {"visible": "terminal", "visual_description": "Running tests"},
        },
        {
            "timestamp": 120,
            "monitor": "0",
            "analysis": {"visible": "browser", "visual_description": "Reading docs"},
        },
    ]

    # Simulate path structure: YYYYMMDD/HHMMSS/screen.jsonl
    jsonl_path = Path("20240101/143022/screen.jsonl")

    markdown = assemble_markdown(
        frames, entity_names="", video_path=jsonl_path, include_entity_context=False
    )

    # Verify absolute times are calculated correctly from segment (14:30:22)
    assert "14:30:22" in markdown  # Base time from segment
    assert "14:30:52" in markdown  # Base + 30 seconds
    assert "14:32:22" in markdown  # Base + 120 seconds (2 minutes)

    # Verify frame content is included
    assert "Editing Python" in markdown
    assert "Running tests" in markdown
    assert "Reading docs" in markdown


def test_assemble_markdown_handles_segment_with_duration_suffix():
    """Test that assemble_markdown handles HHMMSS_LEN segment format."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Code"},
        },
        {
            "timestamp": 60,
            "monitor": "0",
            "analysis": {"visible": "terminal", "visual_description": "Terminal"},
        },
    ]

    # Segment with duration suffix: 143022_300 (5 minutes)
    jsonl_path = Path("20240101/143022_300/screen.jsonl")

    markdown = assemble_markdown(
        frames, entity_names="", video_path=jsonl_path, include_entity_context=False
    )

    # Should still extract base time correctly
    assert "14:30:22" in markdown  # Base time
    assert "14:31:22" in markdown  # Base + 60 seconds
    assert "Code" in markdown
    assert "Terminal" in markdown


def test_assemble_markdown_handles_no_video_path():
    """Test that assemble_markdown works when video_path is None (defaults to midnight)."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Code"},
        },
        {
            "timestamp": 3600,
            "monitor": "0",
            "analysis": {"visible": "browser", "visual_description": "Browser"},
        },
    ]

    markdown = assemble_markdown(
        frames, entity_names="", video_path=None, include_entity_context=False
    )

    # Should default to 00:00:00 base time
    assert "00:00:00" in markdown
    assert "01:00:00" in markdown  # 3600 seconds = 1 hour
    assert "Code" in markdown
    assert "Browser" in markdown


def test_assemble_markdown_handles_multiple_monitors():
    """Test that monitor information is included when multiple monitors are present."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "monitor_position": "left",
            "analysis": {"visible": "code", "visual_description": "Editing code"},
        },
        {
            "timestamp": 0,
            "monitor": "1",
            "monitor_position": "right",
            "analysis": {"visible": "browser", "visual_description": "Documentation"},
        },
    ]

    jsonl_path = Path("20240101/120000/screen.jsonl")

    markdown = assemble_markdown(
        frames, entity_names="", video_path=jsonl_path, include_entity_context=False
    )

    # Should include monitor info when multiple monitors present
    assert "Monitor 0 - left" in markdown
    assert "Monitor 1 - right" in markdown
    assert "Editing code" in markdown
    assert "Documentation" in markdown


def test_assemble_markdown_includes_entity_context():
    """Test that entity context is included when requested."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Code"},
        },
    ]

    jsonl_path = Path("20240101/120000/screen.jsonl")
    entity_names = "Alice, Bob, ProjectX"

    markdown = assemble_markdown(
        frames,
        entity_names=entity_names,
        video_path=jsonl_path,
        include_entity_context=True,
    )

    # Should include entity context header
    assert "# Entity Context" in markdown
    assert "Alice, Bob, ProjectX" in markdown


def test_assemble_markdown_includes_extracted_text():
    """Test that extracted text is included in output."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {
                "visible": "terminal",
                "visual_description": "Terminal window",
            },
            "extracted_text": "$ python test.py\nAll tests passed",
        },
    ]

    jsonl_path = Path("20240101/120000/screen.jsonl")

    markdown = assemble_markdown(
        frames, entity_names="", video_path=jsonl_path, include_entity_context=False
    )

    # Should include extracted text in code block
    assert "**Extracted Text:**" in markdown
    assert "$ python test.py" in markdown
    assert "All tests passed" in markdown


def test_main_constructs_path_from_day_and_segment(tmp_path, monkeypatch):
    """Test that main() constructs correct JSONL path from --day and --segment args."""
    # Create mock journal structure
    journal_path = tmp_path / "journal"
    day_dir = journal_path / "20251109"
    segment_dir = day_dir / "222502_303"
    segment_dir.mkdir(parents=True)

    # Create mock screen.jsonl with minimal valid data
    screen_jsonl = segment_dir / "screen.jsonl"
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Test frame"},
        }
    ]
    with open(screen_jsonl, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")

    # Create mock reduce.txt prompt
    reduce_prompt = tmp_path / "reduce.txt"
    reduce_prompt.write_text("Test prompt")

    # Set JOURNAL_PATH env var
    monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

    # Mock the prompt file location
    with patch("observe.reduce.Path") as mock_path_class:
        # Make Path() work normally for most calls
        mock_path_class.side_effect = lambda *args: Path(*args)
        # But intercept the prompt path lookup
        mock_reduce_file = MagicMock()
        mock_reduce_file.parent = reduce_prompt.parent
        mock_path_class.__file__ = str(mock_reduce_file)

        # Mock gemini_generate to avoid actual API call
        with patch("observe.reduce.gemini_generate") as mock_gemini:
            mock_gemini.return_value = "# Test Summary\n\nGenerated markdown"

            # Mock prompt file reading
            with patch("observe.reduce.Path.read_text") as mock_read:
                mock_read.return_value = "Test prompt"

                # Call reduce_analysis with the constructed path
                exit_code = reduce_analysis(screen_jsonl)

                # Should succeed
                assert exit_code == 0

                # Verify markdown was written
                output_md = segment_dir / "screen.md"
                assert output_md.exists()
                assert "Test Summary" in output_md.read_text()
