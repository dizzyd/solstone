"""Tests for observe/utils.py functions."""

import pytest

from observe.utils import parse_monitor_metadata


class TestParseMonitorMetadata:
    """Test monitor metadata parsing from video title strings."""

    def test_single_monitor_with_metadata(self):
        """Test parsing a single monitor from title string."""
        title = "DP-3:center,0,0,1920,1080"
        result = parse_monitor_metadata(title, 1920, 1080)

        assert len(result) == 1
        assert "DP-3" in result
        assert result["DP-3"] == {
            "name": "DP-3",
            "position": "center",
            "x1": 0,
            "y1": 0,
            "x2": 1920,
            "y2": 1080,
        }

    def test_multiple_monitors(self):
        """Test parsing multiple monitors from title string."""
        title = "DP-3:center,1920,0,5360,1440 HDMI-4:right,5360,219,7280,1299"
        result = parse_monitor_metadata(title, 7280, 1440)

        assert len(result) == 2
        assert "DP-3" in result
        assert "HDMI-4" in result

        assert result["DP-3"] == {
            "name": "DP-3",
            "position": "center",
            "x1": 1920,
            "y1": 0,
            "x2": 5360,
            "y2": 1440,
        }

        assert result["HDMI-4"] == {
            "name": "HDMI-4",
            "position": "right",
            "x1": 5360,
            "y1": 219,
            "x2": 7280,
            "y2": 1299,
        }

    def test_empty_title_returns_default(self):
        """Test that empty title returns default single monitor."""
        result = parse_monitor_metadata("", 1920, 1080)

        assert len(result) == 1
        assert "0" in result
        assert result["0"] == {
            "name": "0",
            "position": "unknown",
            "x1": 0,
            "y1": 0,
            "x2": 1920,
            "y2": 1080,
        }

    def test_invalid_title_returns_default(self):
        """Test that unparseable title returns default single monitor."""
        title = "garbage-data-not-parseable"
        result = parse_monitor_metadata(title, 1920, 1080)

        assert len(result) == 1
        assert "0" in result
        assert result["0"]["position"] == "unknown"
        assert result["0"]["x2"] == 1920
        assert result["0"]["y2"] == 1080

    def test_monitor_with_hyphen_in_name(self):
        """Test parsing monitor names with hyphens."""
        title = "DP-3:left,0,0,1920,1080"
        result = parse_monitor_metadata(title, 1920, 1080)

        assert "DP-3" in result
        assert result["DP-3"]["name"] == "DP-3"

    def test_monitor_with_underscore_in_name(self):
        """Test parsing monitor names with underscores."""
        title = "HDMI_1:right,1920,0,3840,1080"
        result = parse_monitor_metadata(title, 3840, 1080)

        assert "HDMI_1" in result
        assert result["HDMI_1"]["name"] == "HDMI_1"

    def test_partial_invalid_title_skips_bad_entries(self):
        """Test that partially valid title parses good entries and skips bad ones."""
        title = "DP-3:center,0,0,1920,1080 invalid-entry HDMI-4:right,1920,0,3840,1080"
        result = parse_monitor_metadata(title, 3840, 1080)

        # Should parse the two valid entries
        assert len(result) == 2
        assert "DP-3" in result
        assert "HDMI-4" in result

    def test_different_video_dimensions(self):
        """Test with various video dimensions."""
        title = ""
        result = parse_monitor_metadata(title, 3840, 2160)

        assert result["0"]["x2"] == 3840
        assert result["0"]["y2"] == 2160

    def test_monitor_position_labels(self):
        """Test various position labels are preserved."""
        positions = ["left", "center", "right", "top", "bottom", "primary"]

        for pos in positions:
            title = f"MON-1:{pos},0,0,1920,1080"
            result = parse_monitor_metadata(title, 1920, 1080)
            assert result["MON-1"]["position"] == pos
