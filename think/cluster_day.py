import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict


from typing import Optional, Tuple


def cluster_day(folder_path: str, date_str: Optional[str] = None) -> Tuple[str, int]:
    """Return Markdown summary for one day's JSON files and the number processed.

    ``folder_path`` may point directly at the ``YYYYMMDD`` folder or at the
    parent directory containing day folders. If ``date_str`` is omitted it will
    be derived from the trailing path component when it looks like ``YYYYMMDD``.

    The function understands the new file layout used by the ``hear`` and
    ``see`` packages where files are organised under ``<base>/<YYYYMMDD>`` and
    named ``<HHMMSS>[_suffix]_<prefix>.json``.
    """

    # Determine which directory actually holds the day's files.
    if date_str is None:
        base = os.path.basename(os.path.normpath(folder_path))
        if re.fullmatch(r"\d{8}", base):
            date_str = base
            day_dir = folder_path
        else:
            raise ValueError(
                "date_str must be provided when folder_path does not end with YYYYMMDD"
            )
    else:
        candidate = os.path.join(folder_path, date_str)
        day_dir = candidate if os.path.isdir(candidate) else folder_path

    # Capture the optional time suffix (e.g. ``_mic``) and the trailing prefix
    # such as ``audio`` or ``monitor_1_diff``.
    filename_pattern = re.compile(r"^(\d{6}(?:_[^_]+)*)_(.+)\.json$")

    collected_files_data = []
    
    for filename in os.listdir(day_dir):
        match = filename_pattern.match(filename)
        if match:
            time_part, prefix = match.groups()
            time_digits = time_part.split("_")[0]
            try:
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(time_digits[0:2])
                minute = int(time_digits[2:4])
                second = int(time_digits[4:6])
                timestamp = datetime(year, month, day, hour, minute, second)
                full_path = os.path.join(day_dir, filename)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Warning: Could not read file {filename}: {e}", file=sys.stderr)
                    continue
                collected_files_data.append({
                    "filepath": full_path,
                    "basename": filename,
                    "timestamp": timestamp,
                    "prefix": prefix,
                    "content": content,
                })
            except ValueError:
                print(f"Warning: Could not parse time from filename {filename}. Skipping.", file=sys.stderr)
    
    # Sort all collected files by their precise timestamp
    collected_files_data.sort(key=lambda x: x['timestamp'])

    # Group files into 5-minute intervals
    grouped_files = defaultdict(list)
    for file_data in collected_files_data:
        ts = file_data['timestamp']
        interval_minute = ts.minute - (ts.minute % 5)
        interval_start_time = ts.replace(minute=interval_minute, second=0, microsecond=0)
        grouped_files[interval_start_time].append(file_data)

    lines = []
    sorted_interval_keys = sorted(grouped_files.keys())

    if not sorted_interval_keys:
        return f"No JSON files found for date {date_str} in {day_dir}.", 0

    for interval_start in sorted_interval_keys:
        interval_end = interval_start + timedelta(minutes=5)
        lines.append(f"## {interval_start.strftime('%Y-%m-%d %H:%M')} - {interval_end.strftime('%H:%M')}")
        lines.append("")

        files_in_group = grouped_files[interval_start]
        for file_data in files_in_group:
            lines.append(f"### {file_data['prefix']} ({file_data['basename']})")
            lines.append("```json")
            lines.append(file_data['content'].strip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines), len(collected_files_data)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report for a day's JSON files grouped by 5-minute intervals."
    )
    parser.add_argument(
        "folder_path",
        help="Directory containing the day's files or its parent directory",
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Day folder (YYYYMMDD). If omitted, derived from folder_path",
    )

    args = parser.parse_args()

    # Validate folder_path argument
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder not found at specified path: {args.folder_path}", file=sys.stderr)
        sys.exit(1)

    if args.date and not re.fullmatch(r"\d{8}", args.date):
        print("Error: Date argument format must be YYYYMMDD (e.g., 20250524).", file=sys.stderr)
        sys.exit(1)

    markdown, _ = cluster_day(args.folder_path, args.date)
    print(markdown)

if __name__ == "__main__":
    main()
