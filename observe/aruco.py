# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
ArUco marker detection for Convey UI masking.

Detects the 4 corner fiducial tags (ArUco DICT_4X4_50, IDs 2,4,6,7) used in the
Convey web interface to identify and mask self-referential UI regions in
screencast frames before vision processing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Corner tag IDs from convey/static/tags/
# Tag positions: 6=TL, 7=TR, 4=BL, 2=BR
CORNER_TAG_IDS = {6, 7, 4, 2}

# Singleton detector instance (created on first use)
_detector: Optional[cv2.aruco.ArucoDetector] = None


def _get_detector() -> cv2.aruco.ArucoDetector:
    """Get or create the ArUco detector singleton."""
    global _detector
    if _detector is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        # Tuned parameters for small markers
        params.minMarkerPerimeterRate = 0.002
        params.maxMarkerPerimeterRate = 8.0
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detector = cv2.aruco.ArucoDetector(dictionary, params)
    return _detector


def detect_convey_region(image: Image.Image) -> Optional[list[tuple[float, float]]]:
    """
    Detect Convey UI region by finding all 4 corner fiducial tags.

    Parameters
    ----------
    image : Image.Image
        PIL Image to scan for ArUco markers

    Returns
    -------
    Optional[list[tuple[float, float]]]
        Polygon coordinates [(x,y), ...] in order [TL, TR, BR, BL] if all 4
        corner tags are detected, None otherwise.
    """
    # Convert PIL to numpy array
    img_array = np.array(image)

    # Convert to grayscale for detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Detect markers
    detector = _get_detector()
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    # Build map of detected tag ID -> corner points
    id_to_corners = {}
    for tag_id, pts in zip(ids.flatten().tolist(), corners):
        id_to_corners[tag_id] = pts

    # Check if all 4 corner tags are present
    if not CORNER_TAG_IDS.issubset(id_to_corners.keys()):
        return None

    # Extract outer corners from each tag to form the bounding polygon
    # ArUco corner order within each marker: [TL, TR, BR, BL]
    tl = id_to_corners[6].reshape(4, 2)[0]  # TL tag, TL corner
    tr = id_to_corners[7].reshape(4, 2)[1]  # TR tag, TR corner
    br = id_to_corners[2].reshape(4, 2)[2]  # BR tag, BR corner
    bl = id_to_corners[4].reshape(4, 2)[3]  # BL tag, BL corner

    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def mask_convey_region(image: Image.Image, polygon: list[tuple[float, float]]) -> None:
    """
    Mask Convey UI region by filling polygon with black.

    Mutates the image in place.

    Parameters
    ----------
    image : Image.Image
        PIL Image to mask (modified in place)
    polygon : list[tuple[float, float]]
        Polygon coordinates from detect_convey_region()
    """
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, fill=(0, 0, 0))


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """
    Calculate area of a polygon using the shoelace formula.

    Parameters
    ----------
    polygon : list[tuple[float, float]]
        List of (x, y) coordinates

    Returns
    -------
    float
        Area in square pixels
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


__all__ = [
    "CORNER_TAG_IDS",
    "detect_convey_region",
    "mask_convey_region",
    "polygon_area",
]
