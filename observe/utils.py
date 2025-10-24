"""Utilities for working with screencasts and video files."""

import json
import logging
import re
from datetime import datetime
from math import ceil
from pathlib import Path

import av
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def load_analysis_frames(jsonl_path: Path) -> list[dict]:
    """
    Load and parse analysis JSONL, filtering out error frames.

    Parameters
    ----------
    jsonl_path : Path
        Path to analysis JSONL file

    Returns
    -------
    list[dict]
        List of valid frame analysis results
    """
    frames = []
    try:
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                    # Skip frames with errors
                    if "error" not in frame:
                        frames.append(frame)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON at line {line_num} in {jsonl_path}: {e}"
                    )
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {jsonl_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading {jsonl_path}: {e}")
        return []

    return frames


def format_transcript_text(file_path: str | Path) -> str:
    """Format a JSONL audio transcript file into friendly human-readable text.

    Parses the transcript file and outputs a formatted text version with a header
    containing start time and metadata (setting, topics, domain, etc.), followed by
    timestamped entries showing speaker and text.

    Parameters
    ----------
    file_path : str | Path
        Path to the JSONL transcript file (e.g., "20250615/100500_audio.jsonl")

    Returns
    -------
    str
        Formatted text with structure:
        Start: 2024-06-15 10:05a Setting: personal Topics: topic1
        [00:01:23] (mic) Speaker 1: Hello world
        [00:01:25] (sys) Speaker 2: Hi there
        ...

    Examples
    --------
    >>> text = format_transcript_text("20250615/100500_audio.jsonl")
    >>> print(text)
    Start: 2025-06-15 10:05a Setting: work Topics: meeting
    [00:00:15] (mic) Speaker 1: Good morning everyone
    [00:00:18] (sys) Speaker 2: Morning!
    """
    from observe.hear import load_transcript

    path = Path(file_path)

    # Parse day and time from filename
    # Expected format: YYYYMMDD/HHMMSS_audio.jsonl or YYYYMMDD/HHMMSS_imported_audio.jsonl
    parts = path.parts
    day_str = None
    time_str = None

    # Try to find YYYYMMDD in path
    for part in reversed(parts):
        if re.match(r"^\d{8}$", part):
            day_str = part
            break

    # Parse time from filename
    filename = path.name
    time_match = re.match(r"^(\d{6}).*_audio\.jsonl$", filename)
    if time_match:
        time_str = time_match.group(1)

    # Load transcript
    metadata, entries = load_transcript(path)
    if entries is None:
        error_msg = metadata.get("error", "Unknown error")
        return f"Error loading transcript: {error_msg}"

    # Build header line
    header_parts = []

    # Add start time if we could parse it
    if day_str and time_str:
        try:
            dt = datetime.strptime(f"{day_str}{time_str}", "%Y%m%d%H%M%S")
            # Format as "2024-06-15 10:05a"
            time_formatted = dt.strftime("%Y-%m-%d %I:%M%p").lower()
            header_parts.append(f"Start: {time_formatted}")
        except ValueError:
            pass

    # Add metadata fields (excluding special fields)
    skip_fields = {"error", "raw", "imported"}

    for key, value in metadata.items():
        if key in skip_fields:
            continue

        # Format the value
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)

        if value_str:
            header_parts.append(f"{key.capitalize()}: {value_str}")

    # Handle imported metadata specially
    if "imported" in metadata and isinstance(metadata["imported"], dict):
        imported = metadata["imported"]
        if "domain" in imported:
            header_parts.append(f"Domain: {imported['domain']}")
        if "id" in imported:
            header_parts.append(f"Import ID: {imported['id']}")

    # Build output
    lines = []
    if header_parts:
        lines.append(" ".join(header_parts))

    # Format entries
    for entry in entries:
        entry_parts = []

        # Timestamp
        start = entry.get("start", "")
        if start:
            entry_parts.append(f"[{start}]")

        # Source (mic/sys)
        source = entry.get("source", "")
        if source:
            entry_parts.append(f"({source})")

        # Speaker
        speaker = entry.get("speaker")
        if speaker is not None:
            entry_parts.append(f"Speaker {speaker}:")
        else:
            entry_parts.append("")

        # Text
        text = entry.get("text", "")

        # Combine and add to output
        prefix = " ".join(entry_parts).strip()
        if prefix:
            lines.append(f"{prefix} {text}" if text else prefix)
        elif text:
            lines.append(text)

    return "\n".join(lines)


def get_frames(container: av.container.Container) -> list[tuple[float, int]]:
    """
    Get frames sorted by compressed packet size from a video container.

    Larger packets typically indicate more complex/detailed frames with more
    visual change. This method is fast since it reads packets without decoding.
    Samples frames at 1.0 second intervals.

    Args:
        container: PyAV container opened for reading

    Returns:
        List of (timestamp, packet_size) tuples sorted by packet size descending
    """
    # Scan video packets and collect frame data
    frame_data = []  # List of (timestamp, packet_size)
    sample_interval = 1.0
    last_sampled = -sample_interval

    for packet in container.demux(video=0):
        if packet.pts is None:
            continue

        timestamp = float(packet.pts * packet.time_base)

        # Sample at 1 second intervals
        if timestamp - last_sampled >= sample_interval:
            frame_data.append((timestamp, packet.size))
            last_sampled = timestamp

    # Sort by packet size descending
    frame_data.sort(key=lambda x: x[1], reverse=True)

    return frame_data


def compare_frames(
    frame1: av.VideoFrame,
    frame2: av.VideoFrame,
    block_size: int = 64,
    ssim_threshold: float = 0.90,
    margin: int = 5,
) -> list[dict]:
    """
    Compare two PyAV video frames and return bounding boxes of changed regions.

    Uses block-based SSIM on Y-plane (luma) for efficient detection of perceptual
    changes without full RGB decoding. Optimized to work directly with PyAV frames
    for ~3-5x faster performance.

    Args:
        frame1: First video frame
        frame2: Second video frame
        block_size: Size of comparison blocks in pixels (default 64)
        ssim_threshold: SSIM threshold below which blocks are marked as changed (default 0.90)
        margin: Pixel margin to add around bounding boxes (default 5)

    Returns:
        List of dicts with 'box_2d' key containing [y_min, x_min, y_max, x_max] coordinates
    """
    # Extract Y-plane (luma) directly - equivalent to LAB L-channel
    y_plane1 = frame1.to_ndarray(format="gray")
    y_plane2 = frame2.to_ndarray(format="gray")

    height, width = y_plane1.shape
    grid_rows = ceil(height / block_size)
    grid_cols = ceil(width / block_size)
    changed = [[False] * grid_cols for _ in range(grid_rows)]

    # Compute SSIM for each block
    for i in range(grid_rows):
        for j in range(grid_cols):
            y0 = i * block_size
            x0 = j * block_size
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            block1 = y_plane1[y0:y1, x0:x1]
            block2 = y_plane2[y0:y1, x0:x1]
            score, _ = ssim(block1, block2, full=True)
            if score < ssim_threshold:
                changed[i][j] = True

    # Group contiguous changed blocks using DFS
    groups = _group_changed_blocks(changed, grid_rows, grid_cols)

    # Convert groups to bounding boxes
    boxes = _blocks_to_boxes(groups, block_size, width, height, margin)

    return boxes


def _group_changed_blocks(changed, grid_rows, grid_cols):
    """Group contiguous changed blocks using iterative DFS."""
    groups = []
    visited = [[False] * grid_cols for _ in range(grid_rows)]

    def dfs(i, j, group):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= grid_rows or cj < 0 or cj >= grid_cols:
                continue
            if visited[ci][cj] or not changed[ci][cj]:
                continue
            visited[ci][cj] = True
            group.append((ci, cj))
            for ni, nj in [(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)]:
                stack.append((ni, nj))

    for i in range(grid_rows):
        for j in range(grid_cols):
            if changed[i][j] and not visited[i][j]:
                group = []
                dfs(i, j, group)
                groups.append(group)

    return groups


def _blocks_to_boxes(groups, block_size, width, height, margin):
    """Convert groups of changed blocks to bounding boxes."""
    boxes = []
    for group in groups:
        # Calculate bounding box in pixel coordinates
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0
        for i, j in group:
            x0 = j * block_size
            y0 = i * block_size
            x1 = min(x0 + block_size, width)
            y1 = min(y0 + block_size, height)
            min_x = min(min_x, x0)
            min_y = min(min_y, y0)
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)
        # Add margin
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(width, max_x + margin)
        max_y = min(height, max_y + margin)
        # Format as [y_min, x_min, y_max, x_max]
        boxes.append({"box_2d": [min_y, min_x, max_y, max_x]})
    return boxes
