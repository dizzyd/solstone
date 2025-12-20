"""Tests for observe/utils.py functions."""

import pytest

from observe.utils import (
    assign_monitor_positions,
    parse_monitor_metadata,
    parse_screen_filename,
)


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


class TestAssignMonitorPositions:
    """Test monitor position assignment algorithm."""

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert assign_monitor_positions([]) == []

    def test_single_monitor(self):
        """Single monitor always gets 'center'."""
        monitors = [{"id": "DP-1", "box": [0, 0, 1920, 1080]}]
        result = assign_monitor_positions(monitors)

        assert len(result) == 1
        assert result[0]["position"] == "center"

    def test_two_side_by_side(self):
        """Two side-by-side monitors get 'left' and 'right'."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "left"
        assert positions["DP-2"] == "right"

    def test_two_stacked_vertically(self):
        """Two stacked monitors get 'top' and 'bottom'."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [0, 1080, 1920, 2160]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "top"
        assert positions["DP-2"] == "bottom"

    def test_three_in_a_row(self):
        """Three monitors in a row get 'left', 'center', 'right'."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
            {"id": "DP-3", "box": [3840, 0, 5760, 1080]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "left"
        assert positions["DP-2"] == "center"
        assert positions["DP-3"] == "right"

    def test_2x2_grid(self):
        """2x2 grid gets corner positions."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
            {"id": "DP-3", "box": [0, 1080, 1920, 2160]},
            {"id": "DP-4", "box": [1920, 1080, 3840, 2160]},
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        assert positions["DP-1"] == "left-top"
        assert positions["DP-2"] == "right-top"
        assert positions["DP-3"] == "left-bottom"
        assert positions["DP-4"] == "right-bottom"

    def test_offset_dual_monitors(self):
        """Offset dual monitors (different sizes) get distinct positions."""
        # Larger monitor on right, smaller on left offset down
        monitors = [
            {"id": "DP-1", "box": [0, 200, 1920, 1280]},  # 1920x1080, offset down
            {"id": "DP-2", "box": [1920, 0, 4480, 1440]},  # 2560x1440
        ]
        result = assign_monitor_positions(monitors)

        positions = {m["id"]: m["position"] for m in result}
        # DP-1 center: (960, 740), DP-2 center: (3200, 720)
        # Union: (0,0) to (4480,1440), midlines: (2240, 720)
        # DP-1: center_x=960 < 2240 → left, center_y=740 > 720+epsilon → bottom
        # DP-2: center_x=3200 > 2240 → right, center_y=720 = midline → center
        assert positions["DP-1"] == "left-bottom"
        assert positions["DP-2"] == "right"

    def test_no_position_collisions_side_by_side(self):
        """Verify no collisions in typical dual side-by-side setup."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
        ]
        result = assign_monitor_positions(monitors)

        positions = [m["position"] for m in result]
        # All positions should be unique
        assert len(positions) == len(set(positions))

    def test_no_position_collisions_2x2(self):
        """Verify no collisions in 2x2 grid."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080]},
            {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
            {"id": "DP-3", "box": [0, 1080, 1920, 2160]},
            {"id": "DP-4", "box": [1920, 1080, 3840, 2160]},
        ]
        result = assign_monitor_positions(monitors)

        positions = [m["position"] for m in result]
        assert len(positions) == len(set(positions))

    def test_preserves_existing_fields(self):
        """Extra fields in monitor dicts are preserved."""
        monitors = [
            {"id": "DP-1", "box": [0, 0, 1920, 1080], "extra": "data"},
        ]
        result = assign_monitor_positions(monitors)

        assert result[0]["extra"] == "data"
        assert result[0]["position"] == "center"


class TestParseScreenFilename:
    """Test screen filename parsing for per-monitor files."""

    def test_standard_format(self):
        """Parse standard per-monitor filename."""
        position, connector = parse_screen_filename("143022_300_center_DP-3_screen")
        assert position == "center"
        assert connector == "DP-3"

    def test_left_position(self):
        """Parse left position filename."""
        position, connector = parse_screen_filename("120000_600_left_HDMI-1_screen")
        assert position == "left"
        assert connector == "HDMI-1"

    def test_compound_position(self):
        """Parse compound position like left-top."""
        position, connector = parse_screen_filename("090000_300_left-top_DP-1_screen")
        assert position == "left-top"
        assert connector == "DP-1"

    def test_simple_screen_filename(self):
        """Simple screen filename without position returns unknown."""
        position, connector = parse_screen_filename("143022_300_screen")
        assert position == "unknown"
        assert connector == "unknown"

    def test_audio_filename(self):
        """Audio filename returns unknown."""
        position, connector = parse_screen_filename("143022_300_audio")
        assert position == "unknown"
        assert connector == "unknown"

    def test_post_move_format(self):
        """Parse post-move filename (in segment directory, no HHMMSS_LEN prefix)."""
        position, connector = parse_screen_filename("center_DP-3_screen")
        assert position == "center"
        assert connector == "DP-3"

    def test_post_move_left_top(self):
        """Parse post-move filename with compound position."""
        position, connector = parse_screen_filename("left-top_HDMI-2_screen")
        assert position == "left-top"
        assert connector == "HDMI-2"

    def test_plain_screen(self):
        """Plain 'screen' filename returns unknown."""
        position, connector = parse_screen_filename("screen")
        assert position == "unknown"
        assert connector == "unknown"
