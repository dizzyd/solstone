#!/usr/bin/env python3
import argparse
import json
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

API_BASE = "https://api.rev.ai/speechtotext/v1"

def die(msg, code=1):
    logging.error(msg)
    sys.exit(code)

def submit_job(token: str, media_path: Path, language: str, model: str,
               diarization_type: str, forced_alignment: bool,
               speakers_count: int | None, speaker_channels_count: int | None,
               remove_disfluencies: bool, filter_profanity: bool, skip_punctuation: bool):
    url = f"{API_BASE}/jobs"
    headers = {"Authorization": f"Bearer {token}"}

    # multipart/form-data upload of local file (recommended for typical files)
    files = {
        "media": (media_path.name, open(media_path, "rb"),
                  mimetypes.guess_type(media_path.name)[0] or "application/octet-stream")
    }

    # JSON fields go in 'options' part for multipart
    # See docs for fields: transcriber, language, skip_diarization, diarization_type, etc.
    options = {
        "transcriber": model,                   # "fusion" for best ASR
        "skip_diarization": False,              # we want diarization
        "diarization_type": diarization_type,   # "premium" when available
        "language": language,
        "forced_alignment": forced_alignment,
        "remove_disfluencies": remove_disfluencies,
        "filter_profanity": filter_profanity,
        "skip_punctuation": skip_punctuation,
    }
    if speakers_count is not None:
        options["speakers_count"] = speakers_count
    if speaker_channels_count is not None:
        options["speaker_channels_count"] = speaker_channels_count

    data = {"options": json.dumps(options)}

    logging.info("Submitting job to Rev AI: %s", json.dumps(options))
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if resp.status_code >= 300:
        die(f"Job submission failed ({resp.status_code}): {resp.text}")
    return resp.json()["id"]

def get_job(token: str, job_id: str):
    url = f"{API_BASE}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code >= 300:
        die(f"Get job failed ({resp.status_code}): {resp.text}")
    return resp.json()

def get_transcript_json(token: str, job_id: str):
    url = f"{API_BASE}/jobs/{job_id}/transcript"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.rev.transcript.v1.0+json",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code >= 300:
        die(f"Get transcript failed ({resp.status_code}): {resp.text}")
    return resp.json()

def convert_revai_to_sunstone(revai_json: dict) -> list:
    """Convert Rev.ai transcript format to Sunstone transcript format.
    
    Args:
        revai_json: Dict with Rev.ai transcript structure (monologues with elements)
        
    Returns:
        List of transcript entries in Sunstone format
    """
    result = []
    
    if "monologues" not in revai_json:
        return result
    
    for monologue in revai_json["monologues"]:
        speaker = monologue.get("speaker", 0) + 1  # Rev uses 0-based, we use 1-based
        elements = monologue.get("elements", [])
        
        # Build sentences from elements
        current_text = ""
        start_ts = None
        confidences = []
        
        for i, elem in enumerate(elements):
            if elem["type"] == "text":
                # Track first timestamp
                if start_ts is None and elem.get("ts") is not None:
                    start_ts = elem["ts"]
                
                # Add word
                current_text += elem["value"]
                
                # Track confidence
                if elem.get("confidence") is not None:
                    confidences.append(elem["confidence"])
                    
            elif elem["type"] == "punct":
                # Add punctuation
                current_text += elem["value"]
                
                # If sentence-ending punctuation, create entry
                if elem["value"] in [".", "!", "?"] and current_text.strip():
                    # Format timestamp as HH:MM:SS
                    if start_ts is not None:
                        start_seconds = int(start_ts)
                        hours = start_seconds // 3600
                        minutes = (start_seconds % 3600) // 60
                        seconds = start_seconds % 60
                        start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        start_time = "00:00:00"
                    
                    # Create entry
                    entry = {
                        "start": start_time,
                        "source": "mic",
                        "speaker": speaker,
                        "text": current_text.strip()
                    }
                    
                    # Add description based on confidence
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence < 0.7:
                            entry["description"] = "low confidence transcription"
                        elif avg_confidence > 0.95:
                            entry["description"] = "clear and confident speech"
                    
                    result.append(entry)
                    
                    # Reset for next sentence
                    current_text = ""
                    start_ts = None
                    confidences = []
        
        # Handle any remaining text without sentence-ending punctuation
        if current_text.strip():
            if start_ts is not None:
                start_seconds = int(start_ts)
                hours = start_seconds // 3600
                minutes = (start_seconds % 3600) // 60
                seconds = start_seconds % 60
                start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                start_time = "00:00:00"
            
            entry = {
                "start": start_time,
                "source": "mic",
                "speaker": speaker,
                "text": current_text.strip()
            }
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < 0.7:
                    entry["description"] = "low confidence transcription"
                elif avg_confidence > 0.95:
                    entry["description"] = "clear and confident speech"
            
            result.append(entry)
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Rev AI transcription CLI (high-quality + diarization). If a .json file is provided, converts it to Sunstone format instead of transcribing."
    )
    parser.add_argument("media", help="Path to audio/video file or Rev AI JSON file to convert")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("-o", "--output", help="Write JSON to this path (default: stdout)")
    parser.add_argument("--language", default="en", help="ISO code (default: en)")
    parser.add_argument("--model", default="fusion",
                        choices=["fusion", "machine", "low_cost", "human"],
                        help='Rev transcriber (default: "fusion" for highest quality)')
    parser.add_argument("--diarization-type", default="premium",
                        choices=["standard", "premium"],
                        help='Diarization type (default: "premium")')
    parser.add_argument("--forced-alignment", action="store_true",
                        help="Improve per-word timestamps where supported")
    parser.add_argument("--speakers-count", type=int, default=None,
                        help="If known, hint total unique speakers (improves diarization)")
    parser.add_argument("--speaker-channels-count", type=int, default=None,
                        help="If multichannel file with distinct speakers per channel (extra cost)")
    parser.add_argument("--remove-disfluencies", action="store_true",
                        help="Remove ums/uhs + atmospherics (English/Spanish only)")
    parser.add_argument("--filter-profanity", action="store_true",
                        help="Replace profanities with asterisks")
    parser.add_argument("--skip-punctuation", action="store_true",
                        help="Disable punctuation in output")
    parser.add_argument("--poll-interval", type=float, default=2.5,
                        help="Seconds between status polls (default: 2.5)")
    parser.add_argument("--timeout", type=float, default=60*30,
                        help="Overall timeout in seconds (default: 30m)")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    media_path = Path(args.media).expanduser().resolve()
    if not media_path.exists() or not media_path.is_file():
        die(f"File not found: {media_path}")

    # Check if input is a JSON file - if so, convert instead of transcribe
    if media_path.suffix.lower() == ".json":
        logging.info("Detected JSON input - converting Rev AI format to Sunstone format")
        
        # Load the Rev AI JSON
        try:
            with open(media_path, "r", encoding="utf-8") as f:
                revai_data = json.load(f)
        except json.JSONDecodeError as e:
            die(f"Invalid JSON file: {e}")
        
        # Convert to Sunstone format
        sunstone_transcript = convert_revai_to_sunstone(revai_data)
        
        # Output the result
        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            out_path.write_text(json.dumps(sunstone_transcript, indent=2, ensure_ascii=False), encoding="utf-8")
            logging.info("Wrote converted transcript to %s", out_path)
        else:
            print(json.dumps(sunstone_transcript, indent=2, ensure_ascii=False))
        
        return

    # Otherwise, do normal transcription
    load_dotenv()
    token = os.getenv("REVAI_ACCESS_TOKEN") or os.getenv("REV_ACCESS_TOKEN")
    if not token:
        die("Missing REVAI_ACCESS_TOKEN in .env")

    job_id = submit_job(
        token=token,
        media_path=media_path,
        language=args.language,
        model=args.model,
        diarization_type=args.diarization_type,
        forced_alignment=args.forced_alignment,
        speakers_count=args.speakers_count,
        speaker_channels_count=args.speaker_channels_count,
        remove_disfluencies=args.remove_disfluencies,
        filter_profanity=args.filter_profanity,
        skip_punctuation=args.skip_punctuation,
    )
    logging.info("Job submitted: %s", job_id)

    # Poll
    start = time.time()
    status = None
    while True:
        job = get_job(token, job_id)
        new_status = job.get("status")
        if new_status != status:
            logging.info("Status: %s", new_status)
            status = new_status

        if new_status in ("transcribed", "completed"):
            break
        if new_status in ("failed", "error"):
            die(f"Job failed: {json.dumps(job, indent=2)}")

        if time.time() - start > args.timeout:
            die("Timed out waiting for transcription")

        time.sleep(args.poll_interval)

    transcript = get_transcript_json(token, job_id)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Wrote %s", out_path)
    else:
        print(json.dumps(transcript, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
