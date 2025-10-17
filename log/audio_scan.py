#!/usr/bin/env python3
"""
Scan all YYYYMMDD/*_audio.json files in the journal and categorize them by format.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Format type definitions
FORMATS = {
    "simple_text": "Simple object with 'text' string field only",
    "null_text": "Object with 'text' set to null (empty/failed)",
    "text_object_meta": "Object with 'text' as metadata object",
    "buffering": "Object with 'buffering' flag (incomplete/in-progress)",
    "array_basic": "Array with basic objects (at/speaker/text only)",
    "array_desc": "Array with objects using 'description' field",
    "array_sentiment": "Array with objects using 'sentiment' field",
    "array_meta_only": "Array with only metadata (no transcript entries)",
    "imported": "Object with 'imported' and 'entries' keys",
    "text_array_at": "Object with 'text' array using 'at' timestamp",
    "text_array_start": "Object with 'text' array using 'start' timestamp",
    "text_array_meta_only": "Object with 'text' array containing only metadata",
    "text_array_empty": "Object with empty 'text' array",
}


def detect_format(data, filepath):
    """Detect the format of an audio transcript JSON file."""

    # Format 1: simple_text - {"text": "..."}
    if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
        return "simple_text"

    # Format 1b: null_text - {"text": null}
    if isinstance(data, dict) and "text" in data and data["text"] is None:
        return "null_text"

    # Format 1c: text_object_meta - {"text": {...metadata...}}
    if isinstance(data, dict) and "text" in data and isinstance(data["text"], dict):
        return "text_object_meta"

    # Format 1d: buffering - {"buffering": true}
    if isinstance(data, dict) and "buffering" in data:
        return "buffering"

    # Format 4: imported - {"imported": {...}, "entries": [...]}
    if isinstance(data, dict) and "imported" in data and "entries" in data:
        return "imported"

    # Format 5 & 6 & 7 & 8: text_array_* - {"text": [...]}
    if isinstance(data, dict) and "text" in data and isinstance(data["text"], list):
        if len(data["text"]) == 0:
            return "text_array_empty"
        if len(data["text"]) > 0:
            # Check if entries use 'at' or 'start' or are metadata only
            first_entry = data["text"][0]
            if isinstance(first_entry, dict):
                if "start" in first_entry or "speaker" in first_entry:
                    # Has transcript entries
                    if "start" in first_entry:
                        return "text_array_start"
                    elif "at" in first_entry:
                        return "text_array_at"
                    # Fallback if has speaker but no timestamp
                    return "text_array_at"
                elif "source" in first_entry or "topics" in first_entry:
                    # Only metadata, no transcript entries
                    return "text_array_meta_only"

    # Format 2 & 3 & 4: array_* - [{"at": ..., "speaker": ..., ...}, ...]
    if isinstance(data, list) and len(data) > 0:
        # Look at non-metadata entries (skip last entry which might be metadata)
        has_transcript_entries = False
        has_metadata = False
        for entry in data:
            if isinstance(entry, dict):
                if "speaker" in entry and "text" in entry:
                    has_transcript_entries = True
                    # Check for specific annotation fields
                    if "description" in entry:
                        return "array_desc"
                    elif "sentiment" in entry:
                        return "array_sentiment"
                elif "source" in entry or "topics" in entry or "setting" in entry:
                    has_metadata = True

        # If we found transcript entries but no special fields, it's basic array
        if has_transcript_entries:
            return "array_basic"
        # If we only found metadata and no transcript entries
        elif has_metadata:
            return "array_meta_only"

    # Unknown format
    return None


def scan_journal(journal_path):
    """Scan all audio JSON files and categorize by format."""

    journal = Path(journal_path)
    if not journal.exists():
        print(f"Error: Journal path does not exist: {journal_path}", file=sys.stderr)
        sys.exit(1)

    # Find all *_audio.json files
    audio_files = sorted(journal.glob("*/*_audio.json"))
    print(f"Found {len(audio_files)} audio files to scan\n")

    format_counts = Counter()
    unknown_files = []

    for filepath in audio_files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            fmt = detect_format(data, filepath)

            if fmt is None:
                print(f"UNKNOWN FORMAT: {filepath}")
                print(f"Exiting to allow examination...")
                print(f"\nCurrent tally:")
                for fmt_name, count in sorted(format_counts.items()):
                    print(f"  {fmt_name}: {count}")
                sys.exit(1)
            else:
                format_counts[fmt] += 1

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {filepath}: {e}", file=sys.stderr)
            unknown_files.append(str(filepath))
        except Exception as e:
            print(f"ERROR: Failed to read {filepath}: {e}", file=sys.stderr)
            unknown_files.append(str(filepath))

    # Print final tally
    print("\n" + "="*60)
    print("FINAL TALLY")
    print("="*60)

    total = sum(format_counts.values())
    for fmt_name in sorted(format_counts.keys()):
        count = format_counts[fmt_name]
        percentage = (count / total * 100) if total > 0 else 0
        description = FORMATS.get(fmt_name, "Unknown")
        print(f"\n{fmt_name}: {count} files ({percentage:.1f}%)")
        print(f"  {description}")

    print(f"\n{'='*60}")
    print(f"Total files scanned: {total}")

    if unknown_files:
        print(f"\nFiles with errors: {len(unknown_files)}")
        for f in unknown_files:
            print(f"  {f}")

    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        journal_path = sys.argv[1]
    else:
        journal_path = "/home/jer/Pictures/Eri"

    scan_journal(journal_path)
