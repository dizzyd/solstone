#!/usr/bin/env python3
"""
Generate ArUco marker PNG for corner tag detection.

Creates DICT_4X4_50 markers at 2px/bit with a 2px transparent border.
Output is 16x16px (6 bits × 2px = 12px core + 2px border on each side).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_tag(marker_id: int, output_path: Path) -> None:
    """Generate a single ArUco marker PNG.

    Args:
        marker_id: ArUco marker ID (0-49 for DICT_4X4_50)
        output_path: Path to save the PNG file
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # 6 bits per side (4 data + 1 border each side), 2px per bit = 12px
    cell_size = 2
    bits_per_side = 6  # 4 data bits + 2 border bits
    core_size = bits_per_side * cell_size  # 12px

    # Generate the marker (white=255, black=0)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, core_size, borderBits=1)

    # Add 2px transparent border on all sides -> 16x16px final
    border = 2
    final_size = core_size + 2 * border  # 16px

    # Create RGBA image with transparent background
    img = np.zeros((final_size, final_size, 4), dtype=np.uint8)

    # Copy marker into center, black pixels become opaque black, white stays transparent
    for y in range(core_size):
        for x in range(core_size):
            if marker[y, x] == 0:  # Black pixel in marker
                img[y + border, x + border] = [0, 0, 0, 255]  # Opaque black
            # White pixels stay transparent (already zeros)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker PNG for corner tag detection."
    )
    parser.add_argument("id", type=int, help="ArUco marker ID (0-49)")
    parser.add_argument("output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    if not 0 <= args.id < 50:
        parser.error("Marker ID must be 0-49 for DICT_4X4_50")

    generate_tag(args.id, args.output)
    print(f"Generated tag-{args.id} -> {args.output}")


if __name__ == "__main__":
    main()
