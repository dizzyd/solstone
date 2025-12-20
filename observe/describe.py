#!/usr/bin/env python3
"""
Describe screencast videos by detecting significant frame changes.

Processes per-monitor .webm screencast files, detects changes using block-based SSIM,
and qualifies frames that meet the 400x400 threshold for Gemini processing.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

import av
import numpy as np
from PIL import Image

from think.callosum import callosum_send
from think.utils import setup_cli

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Type of vision analysis request."""

    DESCRIBE_JSON = "describe_json"
    DESCRIBE_TEXT = "describe_text"
    DESCRIBE_MEETING = "describe_meeting"


def _load_config() -> dict:
    """
    Load describe.json configuration file.

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    SystemExit
        If config file is missing or invalid
    """
    config_path = Path(__file__).parent / "describe.json"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise SystemExit(1)

    try:
        with open(config_path) as f:
            config = json.load(f)
        logger.debug(f"Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise SystemExit(1)


# Load configuration at module level
CONFIG = _load_config()


class VideoProcessor:
    """Process per-monitor screencast videos and detect significant frame changes."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        # Get video dimensions
        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            self.width = stream.width
            self.height = stream.height
        # Store qualified frames as simple list
        self.qualified_frames: List[dict] = []
        # Load entity names for vision analysis context
        from think.entities import load_entity_names

        self.entity_names = load_entity_names()

    def process(self) -> List[dict]:
        """
        Process video and return qualified frames.

        Returns:
            List of qualified frames with timestamp, frame data (as bytes),
            and change boxes.
        """
        last_qualified: Optional[np.ndarray] = None

        try:
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                stream.codec_context.thread_count = 0

                frame_count = 0
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue

                    timestamp = frame.time if frame.time is not None else 0.0
                    frame_count += 1

                    # First frame: always qualify with full frame bounds
                    if last_qualified is None:
                        box_2d = [0, 0, self.height, self.width]

                        # Convert to PIL for bytes conversion
                        arr_rgb = frame.to_ndarray(format="rgb24")
                        pil_img = Image.fromarray(arr_rgb)

                        # Convert frame to bytes immediately
                        crop_bytes, full_bytes = self._frame_to_bytes(pil_img, box_2d)

                        # Clean up PIL image and RGB array
                        pil_img.close()
                        del arr_rgb

                        self.qualified_frames.append(
                            {
                                "frame_id": frame_count,
                                "timestamp": timestamp,
                                "box_2d": box_2d,
                                "crop_bytes": crop_bytes,
                                "full_bytes": full_bytes,
                            }
                        )

                        # Store grayscale numpy array for comparison
                        last_qualified = frame.to_ndarray(format="gray")

                        logger.debug(f"First frame at {timestamp:.2f}s")
                        continue

                    # Compare current frame with last qualified
                    frame_gray = frame.to_ndarray(format="gray")
                    boxes = self._compare_frames(last_qualified, frame_gray)

                    if not boxes:
                        continue

                    # Find largest box by area
                    largest_box = max(
                        boxes,
                        key=lambda b: (b["box_2d"][2] - b["box_2d"][0])
                        * (b["box_2d"][3] - b["box_2d"][1]),
                    )

                    y_min, x_min, y_max, x_max = largest_box["box_2d"]
                    box_width = x_max - x_min
                    box_height = y_max - y_min

                    # Qualify if largest box meets threshold
                    if box_width >= 400 and box_height >= 400:
                        arr = frame.to_ndarray(format="rgb24")
                        frame_pil = Image.fromarray(arr)
                        del arr

                        # Convert frame to bytes immediately
                        crop_bytes, full_bytes = self._frame_to_bytes(
                            frame_pil, largest_box["box_2d"]
                        )

                        frame_pil.close()

                        self.qualified_frames.append(
                            {
                                "frame_id": frame_count,
                                "timestamp": timestamp,
                                "box_2d": largest_box["box_2d"],
                                "crop_bytes": crop_bytes,
                                "full_bytes": full_bytes,
                            }
                        )

                        # Store grayscale numpy array for comparison
                        last_qualified = frame_gray

                        logger.debug(
                            f"Qualified frame at {timestamp:.2f}s "
                            f"(box: {box_width}x{box_height})"
                        )

                logger.info(
                    f"Processed {frame_count} frames from {self.video_path.name}, "
                    f"{len(self.qualified_frames)} qualified"
                )

        except Exception as e:
            logger.error(
                f"Error processing video {self.video_path}: {e}", exc_info=True
            )
            raise

        return self.qualified_frames

    def _frame_to_bytes(
        self,
        img: Image.Image,
        box_2d: list,
    ) -> tuple[bytes, bytes]:
        """
        Convert frame to bytes - both crop region and full frame.

        Parameters
        ----------
        img : Image.Image
            PIL Image to convert (full frame)
        box_2d : list
            Change box [y_min, x_min, y_max, x_max]

        Returns
        -------
        tuple[bytes, bytes]
            (crop_bytes, full_frame_bytes) as PNG bytes
        """
        y_min, x_min, y_max, x_max = box_2d

        # Expand bounds by 50px in all directions where possible
        img_width, img_height = img.size
        expanded_x_min = max(0, x_min - 50)
        expanded_y_min = max(0, y_min - 50)
        expanded_x_max = min(img_width, x_max + 50)
        expanded_y_max = min(img_height, y_max + 50)

        # Crop to expanded region
        cropped = img.crop(
            (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max)
        )

        # Convert to PNG bytes (compress_level=1 for speed)
        crop_io = io.BytesIO()
        cropped.save(crop_io, format="PNG", compress_level=1)
        crop_bytes = crop_io.getvalue()
        crop_io.close()
        cropped.close()

        # Full frame bytes
        full_io = io.BytesIO()
        img.save(full_io, format="PNG", compress_level=1)
        full_bytes = full_io.getvalue()
        full_io.close()

        # img is closed by caller

        return crop_bytes, full_bytes

    def _compare_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mad_threshold: float = 1.5,
    ) -> List[dict]:
        """
        Compare two grayscale frames and return changed region boxes.

        Parameters
        ----------
        frame1 : np.ndarray
            First grayscale frame
        frame2 : np.ndarray
            Second grayscale frame
        mad_threshold : float
            Mean Absolute Difference threshold for early bailout (default: 1.5)
            If downsampled MAD is below this, skip expensive SSIM computation

        Returns
        -------
        List[dict]
            Boxes with 'box_2d' key containing [y_min, x_min, y_max, x_max]
        """
        # Fast pre-filter: compute MAD on 1/4-scale downsampled images
        # This is extremely fast (pure NumPy) and eliminates unchanged frames
        small1 = frame1[::4, ::4].astype(np.int16)
        small2 = frame2[::4, ::4].astype(np.int16)
        mad = np.abs(small1 - small2).mean()

        if mad < mad_threshold:
            # No significant change detected - skip expensive SSIM
            return []

        return self._compute_ssim_boxes(frame1, frame2)

    def _compute_ssim_boxes(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        block_size: int = 160,
        ssim_threshold: float = 0.90,
        margin: int = 5,
        downsample_factor: int = 4,
    ) -> List[dict]:
        """
        Compare two grayscale frames using block-based SSIM.

        Downsamples before SSIM for speed, computes full SSIM map once,
        then pools to blocks. Much faster than 200+ per-block SSIM calls.
        """
        from math import ceil

        from skimage.metrics import structural_similarity as ssim

        # Store original dimensions for final boxing
        H_orig, W_orig = frame1.shape

        # 1) Downsample by factor (e.g., 4x) for faster SSIM
        small1 = frame1[::downsample_factor, ::downsample_factor]
        small2 = frame2[::downsample_factor, ::downsample_factor]

        # 2) Convert to float32 in [0, 1] to avoid float64 upcasting
        small1 = small1.astype(np.float32) / 255.0
        small2 = small2.astype(np.float32) / 255.0

        # 3) Compute SSIM map once on downsampled images
        _, ssim_map = ssim(
            small1,
            small2,
            data_range=1.0,  # float32 in [0, 1]
            full=True,
            gaussian_weights=False,  # uniform window, matches previous behavior
            use_sample_covariance=False,  # speed boost
            channel_axis=None,  # grayscale
        )

        H, W = ssim_map.shape
        # Block size in downsampled space
        block_size_down = block_size // downsample_factor
        rows = ceil(H / block_size_down)
        cols = ceil(W / block_size_down)

        # 4) Pad to block grid for clean vectorized pooling
        pad_h = rows * block_size_down - H
        pad_w = cols * block_size_down - W
        if pad_h or pad_w:
            ssim_map = np.pad(ssim_map, ((0, pad_h), (0, pad_w)), mode="edge")

        # 5) Vectorized block mean pooling
        block_means = ssim_map.reshape(
            rows, block_size_down, cols, block_size_down
        ).mean(axis=(1, 3))
        changed = (block_means < ssim_threshold).tolist()

        # 6) Reuse shared grouping and boxing logic from utils (uses original dimensions)
        from observe.utils import _blocks_to_boxes, _group_changed_blocks

        groups = _group_changed_blocks(changed, rows, cols)
        return _blocks_to_boxes(groups, block_size, W_orig, H_orig, margin)

    def _user_contents(self, prompt: str, image, entities: bool = False) -> list:
        """Build contents list with optional entity context."""
        contents = [prompt]
        if entities and self.entity_names:
            contents.append(
                f"These are some frequently used names that you may encounter "
                f"and can be helpful when transcribing for accuracy: {self.entity_names}"
            )
        contents.append(image)
        return contents

    def _move_to_segment(self, media_path: Path) -> Path:
        """Move media file to its segment and return new path."""
        from observe.utils import extract_descriptive_suffix
        from think.utils import segment_key

        segment = segment_key(media_path.stem)
        if segment is None:
            raise ValueError(f"Invalid media filename: {media_path.stem}")
        suffix = extract_descriptive_suffix(media_path.stem)
        segment_dir = media_path.parent / segment
        try:
            segment_dir.mkdir(exist_ok=True)
            # Preserve the original extension
            ext = media_path.suffix
            new_path = segment_dir / f"{suffix}{ext}"
            media_path.rename(new_path)
            logger.info(f"Moved {media_path} to {segment_dir}")
            return new_path
        except Exception as exc:
            logger.error(f"Failed to move {media_path} to segment: {exc}")
            return media_path

    async def process_with_vision(
        self,
        use_prompt: str = "describe_json.txt",
        max_concurrent: int = 5,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Process video and write vision analysis results to file.

        Parameters
        ----------
        use_prompt : str
            Prompt template filename to use (default: describe_json.txt)
        max_concurrent : int
            Maximum number of concurrent API requests (default: 5)
        output_path : Optional[Path]
            Path to write JSONL output (default: {video_stem}.jsonl)
        """
        from think.batch import GeminiBatch
        from think.models import GEMINI_LITE

        # Load prompt templates
        prompt_path = Path(__file__).parent / use_prompt
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        system_instruction = prompt_path.read_text()

        # Load text extraction prompt
        text_prompt_path = Path(__file__).parent / "describe_text.txt"
        if not text_prompt_path.exists():
            raise FileNotFoundError(f"Text prompt not found: {text_prompt_path}")

        text_system_instruction = text_prompt_path.read_text()

        # Load meeting analysis prompt
        meeting_prompt_path = Path(__file__).parent / "describe_meeting.txt"
        if not meeting_prompt_path.exists():
            raise FileNotFoundError(f"Meeting prompt not found: {meeting_prompt_path}")

        meeting_system_instruction = meeting_prompt_path.read_text()

        # Process video to get qualified frames (synchronous)
        qualified_frames = self.process()

        # Create batch processor
        batch = GeminiBatch(max_concurrent=max_concurrent)

        # Open output file if specified
        output_file = open(output_path, "w") if output_path else None

        # Write metadata header to JSONL file with actual video filename
        if output_file:
            from observe.utils import extract_descriptive_suffix

            suffix = extract_descriptive_suffix(self.video_path.stem)
            metadata = {"raw": f"{suffix}{self.video_path.suffix}"}
            output_file.write(json.dumps(metadata) + "\n")
            output_file.flush()

        # Create vision requests for all qualified frames
        for frame_data in qualified_frames:
            # Load crop image from bytes - keep it open until request completes
            crop_img = Image.open(io.BytesIO(frame_data["crop_bytes"]))

            req = batch.create(
                contents=self._user_contents(
                    "Analyze this screenshot frame from a screencast recording.",
                    crop_img,
                ),
                model=GEMINI_LITE,
                system_instruction=system_instruction,
                json_output=True,
                temperature=0.7,
                max_output_tokens=3072,
                thinking_budget=2048,
            )

            # Attach metadata for tracking (store bytes, not PIL images)
            req.frame_id = frame_data["frame_id"]
            req.timestamp = frame_data["timestamp"]
            req.box_2d = frame_data["box_2d"]
            req.retry_count = 0
            req.crop_bytes = frame_data["crop_bytes"]  # Store bytes for reuse
            req.full_bytes = frame_data["full_bytes"]  # Store for meeting analysis
            req.request_type = RequestType.DESCRIBE_JSON
            req.json_analysis = None  # Will store the JSON analysis result
            req.meeting_analysis = None  # Will store meeting analysis if applicable
            req.requests = []  # Track all requests for this frame
            req.initial_image = crop_img  # Keep reference to close after completion

            batch.add(req)

        # Clear qualified_frames now that all requests are created
        # Bytes are already referenced in request objects, so this allows them
        # to be freed incrementally as requests complete rather than all at the end
        self.qualified_frames.clear()

        # Track success/failure for all frames
        total_frames = 0
        failed_frames = 0

        # Stream results as they complete, with retry logic
        async for req in batch.drain_batch():
            total_frames += 1
            # Check for errors
            has_error = bool(req.error)
            error_msg = req.error

            # Handle based on request type
            if not has_error:
                if req.request_type == RequestType.DESCRIBE_JSON:
                    # Parse JSON analysis
                    try:
                        analysis = json.loads(req.response)
                        req.json_analysis = analysis  # Store for follow-up analysis
                    except json.JSONDecodeError as e:
                        has_error = True
                        error_msg = f"Invalid JSON response: {e}"
                elif req.request_type == RequestType.DESCRIBE_MEETING:
                    # Parse meeting analysis
                    try:
                        meeting_data = json.loads(req.response)
                        req.meeting_analysis = meeting_data  # Store meeting analysis
                    except json.JSONDecodeError as e:
                        has_error = True
                        error_msg = f"Invalid JSON response: {e}"

            # Retry logic (up to 5 attempts total, so 4 retries)
            if has_error and req.retry_count < 4:
                req.retry_count += 1
                batch.add(req)
                logger.info(
                    f"Retrying frame {req.frame_id} (attempt {req.retry_count + 1}/5): {error_msg}"
                )
                continue  # Don't output, wait for retry result

            # Track failure after all retries exhausted
            if has_error:
                failed_frames += 1

            # Record this request's result (after retries are done)
            request_record = {
                "type": req.request_type.value,
                "model": req.model_used,
                "duration": req.duration,
            }
            if req.retry_count > 0:
                request_record["retries"] = req.retry_count

            req.requests.append(request_record)

            # Check if we should trigger follow-up analysis
            should_process_further = (
                not has_error
                and req.request_type == RequestType.DESCRIBE_JSON
                and req.json_analysis
            )

            if should_process_further:
                visible_category = req.json_analysis.get("visible", "")

                # Check for meeting analysis
                if visible_category == "meeting":
                    logger.info(f"Frame {req.frame_id}: Triggering meeting analysis")
                    # Load full frame from cached bytes
                    full_image = Image.open(io.BytesIO(req.full_bytes))

                    batch.update(
                        req,
                        contents=self._user_contents(
                            "Analyze this meeting screenshot.",
                            full_image,
                            entities=True,
                        ),
                        model=GEMINI_LITE,
                        system_instruction=meeting_system_instruction,
                        json_output=True,
                        max_output_tokens=10240,
                        thinking_budget=6144,
                    )
                    # Don't close yet - batch needs it for encoding
                    # Store reference for cleanup later
                    req.meeting_image = full_image

                    # Close initial image since DESCRIBE_JSON is complete
                    if hasattr(req, "initial_image") and req.initial_image:
                        req.initial_image.close()
                        req.initial_image = None

                    req.request_type = RequestType.DESCRIBE_MEETING
                    req.retry_count = 0
                    continue  # Don't output yet, wait for meeting analysis

                # Check for text extraction
                text_categories = CONFIG.get("text_extraction_categories", [])
                if visible_category in text_categories:
                    logger.info(
                        f"Frame {req.frame_id}: Triggering text extraction for category '{visible_category}'"
                    )
                    # Load crop image from cached bytes
                    crop_img = Image.open(io.BytesIO(req.crop_bytes))

                    # Update request for text extraction and re-add
                    batch.update(
                        req,
                        contents=self._user_contents(
                            "Extract text from this screenshot frame.",
                            crop_img,
                            entities=True,
                        ),
                        model=GEMINI_LITE,
                        system_instruction=text_system_instruction,
                        json_output=False,
                        max_output_tokens=8192,
                        thinking_budget=4096,
                    )
                    # Don't close yet - batch needs it for encoding
                    # Store reference for cleanup later
                    req.text_image = crop_img

                    # Close initial image since DESCRIBE_JSON is complete
                    if hasattr(req, "initial_image") and req.initial_image:
                        req.initial_image.close()
                        req.initial_image = None

                    req.request_type = RequestType.DESCRIBE_TEXT
                    req.retry_count = 0
                    continue  # Don't output yet, wait for text extraction

            # Final output - this frame is complete
            result = {
                "frame_id": req.frame_id,
                "timestamp": req.timestamp,
                "box_2d": req.box_2d,
                "requests": req.requests,
            }

            # Add error at top level if any request failed
            if has_error:
                result["error"] = error_msg

            # Add analysis if we have it
            if req.json_analysis:
                result["analysis"] = req.json_analysis

            # Add meeting analysis if we have it (from DESCRIBE_MEETING)
            if req.meeting_analysis:
                result["meeting_analysis"] = req.meeting_analysis

            # Add extracted text if we have it (from DESCRIBE_TEXT)
            if req.request_type == RequestType.DESCRIBE_TEXT and req.response:
                result["extracted_text"] = req.response

            # Write to file and optionally to stdout
            result_line = json.dumps(result)
            if output_file:
                output_file.write(result_line + "\n")
                output_file.flush()
            if logger.isEnabledFor(logging.DEBUG):
                print(result_line, flush=True)

            # Close all PIL Images associated with this request
            if hasattr(req, "initial_image") and req.initial_image:
                req.initial_image.close()
                req.initial_image = None
            if hasattr(req, "meeting_image") and req.meeting_image:
                req.meeting_image.close()
                req.meeting_image = None
            if hasattr(req, "text_image") and req.text_image:
                req.text_image.close()
                req.text_image = None

            # Aggressively clear heavy fields now that request is finalized
            req.crop_bytes = None
            req.full_bytes = None
            req.json_analysis = None
            req.meeting_analysis = None

        # Close output file
        if output_file:
            output_file.close()

        # Check if all frames failed
        all_failed = total_frames > 0 and failed_frames == total_frames

        if all_failed:
            # Don't move video to segment - leave for retry
            error_detail = (
                f"Error details in {output_path}" if output_path else "No output file"
            )
            logger.error(
                f"All {total_frames} frame(s) failed processing. "
                f"Video left in place for retry. {error_detail}"
            )
            # Clear qualified_frames to free memory before raising
            self.qualified_frames.clear()
            raise RuntimeError(
                f"All {total_frames} frame(s) failed vision analysis after retries"
            )
        else:
            # At least some frames succeeded - move to segment
            if failed_frames > 0:
                logger.warning(
                    f"{failed_frames}/{total_frames} frame(s) failed processing. "
                    f"Moving video to segment anyway."
                )
            if output_path:
                self._move_to_segment(self.video_path)

        # Clear qualified_frames to free memory
        self.qualified_frames.clear()


def output_qualified_frames(
    processor: VideoProcessor, qualified_frames: List[dict]
) -> None:
    """Output qualified frames as JSON."""
    output = {
        "video": str(processor.video_path.name),
        "width": processor.width,
        "height": processor.height,
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
                "box_2d": frame["box_2d"],
            }
            for frame in qualified_frames
        ],
    }

    print(json.dumps(output, indent=2))


async def async_main():
    """Async CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Describe screencast videos with vision analysis"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="describe_json.txt",
        help="Prompt template to use (default: describe_json.txt)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=10,
        help="Max concurrent vision API requests (default: 10)",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="Only output frame metadata without vision analysis",
    )
    args = setup_cli(parser)

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Video file not found: {video_path}")

    # Determine output path and warn if overwriting
    output_path = None
    segment = None
    suffix = None
    if not args.frames_only:
        # Extract segment and suffix for output naming
        from observe.utils import extract_descriptive_suffix
        from think.utils import segment_key

        segment = segment_key(video_path.stem)
        if segment is None:
            parser.error(
                f"Invalid video filename: {video_path.stem} (must be HHMMSS_LEN format)"
            )
        suffix = extract_descriptive_suffix(video_path.stem)
        segment_dir = video_path.parent / segment
        segment_dir.mkdir(exist_ok=True)
        # Output JSONL matches input filename pattern (e.g., center_DP-3_screen.jsonl)
        output_path = segment_dir / f"{suffix}.jsonl"
        if output_path.exists():
            logger.warning(f"Overwriting existing analysis file: {output_path}")

    logger.info(f"Processing video: {video_path}")

    start_time = time.time()

    try:
        processor = VideoProcessor(video_path)

        if args.frames_only:
            # Original behavior: just output frame metadata
            qualified_frames = processor.process()
            output_qualified_frames(processor, qualified_frames)
        else:
            # New behavior: process with vision analysis
            await processor.process_with_vision(
                use_prompt=args.prompt,
                max_concurrent=args.jobs,
                output_path=output_path,
            )

            # Emit completion event
            if output_path and output_path.exists():
                journal_path = Path(os.getenv("JOURNAL_PATH", ""))
                # Moved path is in segment: YYYYMMDD/HHMMSS_LEN/suffix.webm
                moved_path = (
                    video_path.parent / segment / f"{suffix}{video_path.suffix}"
                )

                try:
                    rel_input = moved_path.relative_to(journal_path)
                    rel_output = output_path.relative_to(journal_path)
                except ValueError:
                    rel_input = moved_path
                    rel_output = output_path

                duration_ms = int((time.time() - start_time) * 1000)

                callosum_send(
                    "observe",
                    "described",
                    input=str(rel_input),
                    output=str(rel_output),
                    duration_ms=duration_ms,
                )
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}", exc_info=True)
        raise


def main():
    """CLI entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
