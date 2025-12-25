"""Tests for observe/describe.py category prompt discovery."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_category_prompts_discovered():
    """Test that category prompts are discovered on import."""
    from observe.describe import CATEGORY_PROMPTS

    # Should have discovered some category prompts
    assert len(CATEGORY_PROMPTS) > 0
    # Meeting should be one of them
    assert "meeting" in CATEGORY_PROMPTS


def test_category_prompts_have_required_fields():
    """Test that discovered categories have required metadata."""
    from observe.describe import CATEGORY_PROMPTS

    for category, metadata in CATEGORY_PROMPTS.items():
        # Each category should have 'output' and 'prompt' fields
        assert "output" in metadata, f"Category {category} missing 'output' field"
        assert "prompt" in metadata, f"Category {category} missing 'prompt' field"
        # Output should be 'json' or 'markdown'
        assert metadata["output"] in (
            "json",
            "markdown",
        ), f"Category {category} has invalid output: {metadata['output']}"
        # Prompt should be non-empty string
        assert isinstance(metadata["prompt"], str)
        assert len(metadata["prompt"]) > 0


def test_meeting_category_is_json():
    """Test that meeting category outputs JSON."""
    from observe.describe import CATEGORY_PROMPTS

    assert "meeting" in CATEGORY_PROMPTS
    assert CATEGORY_PROMPTS["meeting"]["output"] == "json"


def test_text_categories_are_markdown():
    """Test that text-based categories output markdown."""
    from observe.describe import CATEGORY_PROMPTS

    text_categories = ["messaging", "browsing", "reading", "productivity"]
    for category in text_categories:
        if category in CATEGORY_PROMPTS:
            assert (
                CATEGORY_PROMPTS[category]["output"] == "markdown"
            ), f"Category {category} should output markdown"


def test_discover_category_prompts_with_missing_dir(tmp_path):
    """Test that discovery handles missing directory gracefully."""
    from observe.describe import _discover_category_prompts

    with patch("observe.describe.Path") as mock_path:
        # Mock to point to non-existent directory
        mock_describe_dir = tmp_path / "nonexistent"
        mock_path.return_value.parent.__truediv__.return_value = mock_describe_dir

        result = _discover_category_prompts()
        assert result == {}


def test_discover_category_prompts_with_valid_dir(tmp_path):
    """Test that discovery works with valid category files."""
    from observe.describe import _discover_category_prompts

    # Create test category directory
    describe_dir = tmp_path / "describe"
    describe_dir.mkdir()

    # Create test category files
    (describe_dir / "test.json").write_text('{"output": "markdown"}')
    (describe_dir / "test.txt").write_text("Test prompt content")

    with patch("observe.describe.Path") as mock_path:
        mock_path.return_value.parent.__truediv__.return_value = describe_dir

        result = _discover_category_prompts()
        assert "test" in result
        assert result["test"]["output"] == "markdown"
        assert result["test"]["prompt"] == "Test prompt content"


def test_discover_category_prompts_skips_incomplete(tmp_path):
    """Test that discovery skips categories without matching txt file."""
    from observe.describe import _discover_category_prompts

    # Create test category directory
    describe_dir = tmp_path / "describe"
    describe_dir.mkdir()

    # Create JSON without matching txt
    (describe_dir / "incomplete.json").write_text('{"output": "json"}')

    with patch("observe.describe.Path") as mock_path:
        mock_path.return_value.parent.__truediv__.return_value = describe_dir

        result = _discover_category_prompts()
        assert "incomplete" not in result
