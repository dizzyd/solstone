"""Tests for observe.see utilities."""

import pytest
from PIL import Image

from observe.see import (
    decode_frames,
    draw_bounding_box,
    image_to_jpeg_bytes,
)


def test_draw_bounding_box():
    """Test drawing bounding box on image."""
    img = Image.new("RGB", (100, 100), color="white")

    # Draw a box - should not crash
    box_2d = [10, 20, 30, 40]  # y_min, x_min, y_max, x_max
    draw_bounding_box(img, box_2d, color="red", width=3)

    # Image should still be same size (mutated in place)
    assert img.size == (100, 100)


def test_image_to_jpeg_bytes():
    """Test converting image to JPEG bytes."""
    img = Image.new("RGB", (50, 50), color="green")

    # Convert to JPEG
    jpeg_bytes = image_to_jpeg_bytes(img, quality=85)

    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0
    # JPEG files start with FF D8 magic bytes
    assert jpeg_bytes[:2] == b"\xff\xd8"


def test_decode_frames_empty_list():
    """Test decode_frames with empty frame list."""
    result = decode_frames("dummy.mp4", [])
    assert result == []


def test_decode_frames_missing_frame_id():
    """Test decode_frames raises error when frame_id is missing."""
    frames = [{"timestamp": 1.0}]

    with pytest.raises(ValueError, match="must have 'frame_id' field"):
        decode_frames("dummy.mp4", frames)
