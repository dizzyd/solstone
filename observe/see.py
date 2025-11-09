from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Historical note: This border detection function was previously used in describe.py
# to detect and handle blue borders in screencast frames. The implementation would:
# 1. Fast pre-check for pure blue (0,0,255) pixels in each frame
# 2. If found, call detect_border() to get border bounding box
# 3. Check if change box still met 400x400px threshold after subtracting border area
# 4. Censor (black out) the border region in frames sent to vision analysis
# 5. Store censor_coords in output for reference
# This feature was removed in favor of simpler processing - the complexity of tracking
# border coordinates through the processing pipeline and qualification logic outweighed
# the benefits. If re-implementing, consider: frame-level caching of border detection,
# intersection calculation between change boxes and border boxes, and threading
# censor_coords through _frame_to_bytes(), process_with_vision(), and output.


def detect_border(
    im: Image.Image,
    color: tuple[int, int, int],
    *,
    min_length: int = 100,
    border: int = 3,
    tolerance: int = 0,
) -> tuple[int, int, int, int]:
    """Detect a coloured border and return the bounding box coordinates.

    Parameters
    ----------
    im : Image.Image
        Image to analyse.
    color : tuple[int, int, int]
        RGB values of the border colour to detect.
    min_length : int, optional
        Minimum number of matching pixels per side, by default 100.
    border : int, optional
        Expected thickness of the border in pixels, by default 3.
    tolerance : int, optional
        Allowed deviation per channel for colour matching, by default 0.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box as ``(y_min, x_min, y_max, x_max)``.
    """
    arr = np.asarray(im)
    r, g, b = color
    mask = np.logical_and.reduce(
        [
            np.abs(arr[..., 0] - r) <= tolerance,
            np.abs(arr[..., 1] - g) <= tolerance,
            np.abs(arr[..., 2] - b) <= tolerance,
        ]
    )

    col_hits = mask.sum(0)
    row_hits = mask.sum(1)

    cols = np.where(col_hits >= min_length)[0]
    rows = np.where(row_hits >= min_length)[0]
    if cols.size == 0 or rows.size == 0:
        raise ValueError("No border detected")

    def first_last(groups):
        groups = np.split(groups, np.where(np.diff(groups) != 1)[0] + 1)
        groups = [g for g in groups if g.size == border]
        if not groups:
            raise ValueError("Border not thick enough")
        return groups[0][0], groups[-1][-1]

    x_min, x_max = first_last(cols)
    y_min, y_max = first_last(rows)

    return int(y_min), int(x_min), int(y_max), int(x_max)


def crop_frame_to_monitor(image: Image.Image, monitor_bounds: dict) -> Image.Image:
    """Crop PIL image to monitor bounds.

    Args:
        image: Full multi-monitor PIL image
        monitor_bounds: Dict with x1, y1, x2, y2 keys

    Returns:
        Cropped PIL image (new image, original unchanged)
    """
    x1 = monitor_bounds.get("x1", 0)
    y1 = monitor_bounds.get("y1", 0)
    x2 = monitor_bounds.get("x2", image.width)
    y2 = monitor_bounds.get("y2", image.height)
    return image.crop((x1, y1, x2, y2))


def draw_bounding_box(
    image: Image.Image,
    box_2d: list[int],
    color: str = "red",
    width: int = 3,
) -> None:
    """Draw bounding box on image (mutates in place).

    Args:
        image: PIL image to draw on (modified in place)
        box_2d: [y_min, x_min, y_max, x_max] coordinates
        color: Box color (default "red")
        width: Line width in pixels (default 3)
    """
    from PIL import ImageDraw

    y_min, x_min, y_max, x_max = box_2d
    draw = ImageDraw.Draw(image)
    for i in range(width):
        draw.rectangle(
            [x_min - i, y_min - i, x_max + i, y_max + i],
            outline=color,
        )


def image_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Convert PIL image to JPEG bytes.

    Args:
        image: PIL image
        quality: JPEG quality 1-100 (default 85)

    Returns:
        JPEG encoded bytes
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_frames(
    video_path: str | Path,
    frames: list[dict],
    annotate_boxes: bool = True,
) -> list[Image.Image | None]:
    """
    Decode and process video frames with monitor cropping and optional annotation.

    Takes frame metadata from a screen.jsonl file and returns corresponding
    PIL images cropped to their monitor bounds with change regions annotated.

    Args:
        video_path: Path to the raw video file
        frames: List of frame dicts from screen.jsonl, each containing:
            - frame_id (int): Sequential frame number from video
            - monitor (str): Monitor ID
            - box_2d (list[int], optional): [y_min, x_min, y_max, x_max] change region
            Additional fields (timestamp, etc.) are preserved but not used
            Note: Duplicate frame_ids are normal when multiple monitors qualify the same frame
        annotate_boxes: Draw red borders around box_2d regions (default True)

    Returns:
        List of PIL Images in same order as input frames.
        None for frames that couldn't be matched/decoded.
        Each image is cropped to its monitor bounds.

    Raises:
        ValueError: If frames are missing frame_id field

    Example:
        >>> from observe.utils import load_analysis_frames
        >>> from observe.see import decode_frames
        >>> all_frames = load_analysis_frames("20250101/092152/screen.jsonl")
        >>> # Filter to actual frames (skip header)
        >>> frames = [f for f in all_frames if "frame_id" in f]
        >>> # Get first 10 frames
        >>> images = decode_frames("20250101/092152/screen.webm", frames[:10])
        >>> images[0].show()  # Display first frame
    """
    # Early validation before imports
    if not frames:
        return []

    # Validate frames have frame_id field
    for frame in frames:
        if frame.get("frame_id") is None:
            raise ValueError("All frames must have 'frame_id' field")

    # Import heavy dependencies only after validation
    import pathlib

    import av

    from observe.utils import parse_monitor_metadata

    # Build a map of frame_id -> (index, frame_dict) for quick lookup
    frame_map = {f["frame_id"]: (i, f) for i, f in enumerate(frames)}

    # Initialize result list with Nones
    results: list[Image.Image | None] = [None] * len(frames)

    # Open video and parse monitor metadata
    video_path = pathlib.Path(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        # Parse monitor metadata from video
        title = container.metadata.get("title", "")
        width = stream.width
        height = stream.height
        monitors = parse_monitor_metadata(title, width, height)

        # Decode video and process requested frames
        frame_count = 0
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue

            # Check if this frame_id is requested
            if frame_count in frame_map:
                idx, frame_dict = frame_map[frame_count]

                # Convert to PIL Image (full multi-monitor frame)
                arr = av_frame.to_ndarray(format="rgb24")
                full_img = Image.fromarray(arr)

                # Get monitor bounds
                monitor_id = frame_dict.get("monitor", "0")
                monitor_bounds = monitors.get(monitor_id, {})

                # Crop to monitor
                img = crop_frame_to_monitor(full_img, monitor_bounds)
                full_img.close()

                # Draw bounding box if requested and present
                if annotate_boxes and "box_2d" in frame_dict:
                    box_2d = frame_dict["box_2d"]
                    draw_bounding_box(img, box_2d)

                results[idx] = img

            frame_count += 1

            # Early exit if we've processed all requested frames
            if all(r is not None for r in results):
                break

    return results


__all__ = [
    "detect_border",
    "crop_frame_to_monitor",
    "draw_bounding_box",
    "image_to_jpeg_bytes",
    "decode_frames",
]
