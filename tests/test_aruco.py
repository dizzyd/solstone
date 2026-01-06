# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.aruco ArUco marker detection and masking."""

import cv2
import numpy as np
from PIL import Image

from observe.aruco import (
    CORNER_TAG_IDS,
    detect_convey_region,
    mask_convey_region,
    polygon_area,
)


def test_corner_tag_ids():
    """Test that corner tag IDs match expected values."""
    assert CORNER_TAG_IDS == {2, 4, 6, 7}


def test_polygon_area_square():
    """Test polygon area calculation for a square."""
    # 100x100 square
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert polygon_area(polygon) == 10000.0


def test_polygon_area_triangle():
    """Test polygon area calculation for a triangle."""
    # Right triangle with legs 10 and 20
    polygon = [(0, 0), (10, 0), (0, 20)]
    assert polygon_area(polygon) == 100.0  # (10 * 20) / 2


def test_polygon_area_empty():
    """Test polygon area with insufficient points."""
    assert polygon_area([]) == 0.0
    assert polygon_area([(0, 0)]) == 0.0
    assert polygon_area([(0, 0), (1, 1)]) == 0.0


def test_detect_convey_region_no_markers():
    """Test detection returns None when no markers are present."""
    # Plain white image - no markers
    img = Image.new("RGB", (640, 480), color="white")
    result = detect_convey_region(img)
    assert result is None


def test_detect_convey_region_grayscale():
    """Test detection works with grayscale input."""
    # Grayscale image - should handle conversion
    img = Image.new("L", (640, 480), color=128)
    result = detect_convey_region(img)
    assert result is None  # No markers, but shouldn't crash


def test_mask_convey_region():
    """Test masking fills polygon with black."""
    img = Image.new("RGB", (100, 100), color="white")

    # Define a square polygon in the center
    polygon = [(25, 25), (75, 25), (75, 75), (25, 75)]
    mask_convey_region(img, polygon)

    # Check corners are still white
    assert img.getpixel((0, 0)) == (255, 255, 255)
    assert img.getpixel((99, 99)) == (255, 255, 255)

    # Check center is black
    assert img.getpixel((50, 50)) == (0, 0, 0)


def test_mask_convey_region_triangle():
    """Test masking works with non-rectangular polygon."""
    img = Image.new("RGB", (100, 100), color="white")

    # Triangle
    polygon = [(50, 10), (90, 90), (10, 90)]
    mask_convey_region(img, polygon)

    # Center should be black (inside triangle)
    assert img.getpixel((50, 60)) == (0, 0, 0)

    # Top corners should still be white (outside triangle)
    assert img.getpixel((5, 5)) == (255, 255, 255)
    assert img.getpixel((95, 5)) == (255, 255, 255)


def test_detect_convey_region_with_real_markers():
    """Test detection with actual ArUco markers rendered into image."""
    # Create a test image
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Generate and place the 4 corner markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 50

    # Generate markers
    markers = {}
    for tag_id in [6, 7, 4, 2]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)
        # Convert to 3-channel
        markers[tag_id] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)

    # Place markers at corners (with some padding)
    pad = 20
    # TL - tag 6
    img_array[pad : pad + marker_size, pad : pad + marker_size] = markers[6]
    # TR - tag 7
    img_array[pad : pad + marker_size, 640 - pad - marker_size : 640 - pad] = markers[7]
    # BL - tag 4
    img_array[480 - pad - marker_size : 480 - pad, pad : pad + marker_size] = markers[4]
    # BR - tag 2
    img_array[
        480 - pad - marker_size : 480 - pad, 640 - pad - marker_size : 640 - pad
    ] = markers[2]

    # Convert to PIL
    pil_img = Image.fromarray(img_array)

    # Detect
    result = detect_convey_region(pil_img)

    # Should find all 4 markers and return polygon
    assert result is not None
    assert len(result) == 4

    # Polygon should roughly bound the marker positions
    # Each point should be a tuple of numeric values
    for point in result:
        assert len(point) == 2
        assert np.issubdtype(type(point[0]), np.number) or isinstance(
            point[0], (int, float)
        )
        assert np.issubdtype(type(point[1]), np.number) or isinstance(
            point[1], (int, float)
        )


def test_detect_convey_region_partial_markers():
    """Test detection returns None when only some markers present."""
    # Create a test image
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Generate and place only 2 corner markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 50
    pad = 20

    # Only place TL and TR markers
    for tag_id, pos in [(6, (pad, pad)), (7, (pad, 640 - pad - marker_size))]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)
        marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        y, x = pos
        img_array[y : y + marker_size, x : x + marker_size] = marker_rgb

    pil_img = Image.fromarray(img_array)

    # Should return None - only 2 of 4 markers found
    result = detect_convey_region(pil_img)
    assert result is None
