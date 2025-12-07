"""Event indexing and search functionality."""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

from .core import _scan_files, get_index, sanitize_fts_query
from .insights import find_event_files


def _index_events(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    """Index events from a JSONL file.

    Path format: facets/{facet}/events/YYYYMMDD.jsonl
    Each line is a self-contained event with 'occurred' field.
    """
    logger = logging.getLogger(__name__)

    # Extract facet and day from path: facets/{facet}/events/YYYYMMDD.jsonl
    parts = rel.split(os.sep)
    path_facet = parts[1] if len(parts) >= 4 else ""
    filename = os.path.basename(rel)
    file_day = os.path.splitext(filename)[0]  # YYYYMMDD from filename

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if verbose:
        logger.info("  indexed %s events", len(events))

    for idx, event in enumerate(events):
        occurred = 1 if event.get("occurred", True) else 0
        topic = event.get("topic", "")
        facet = event.get("facet", path_facet)

        conn.execute(
            "INSERT INTO events_text(content, path, day, idx) VALUES (?, ?, ?, ?)",
            (
                json.dumps(event, ensure_ascii=False),
                rel,
                file_day,
                idx,
            ),
        )
        conn.execute(
            "INSERT INTO event_match(path, day, idx, topic, facet, start, end, occurred) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rel,
                file_day,
                idx,
                topic,
                facet,
                event.get("start") or "",
                event.get("end") or "",
                occurred,
            ),
        )


def scan_events(journal: str, verbose: bool = False) -> bool:
    """Index event JSONL files from facets/*/events/*.jsonl."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="events", journal=journal)

    delete_sql = [
        "DELETE FROM events_text WHERE path=?",
        "DELETE FROM event_match WHERE path=?",
    ]

    files = find_event_files(journal)
    if files:
        logger.info("\nIndexing %s event files...", len(files))
    changed = _scan_files(conn, files, delete_sql, _index_events, verbose)

    if changed:
        conn.commit()
    conn.close()
    return changed


def search_events(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    facet: str | None = None,
    start: str | None = None,
    end: str | None = None,
    topic: str | None = None,
    occurred: bool | None = None,
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the events index and return total count and results.

    Parameters
    ----------
    query : str
        FTS search query (empty string for no text search).
    limit : int
        Maximum results to return.
    offset : int
        Number of results to skip.
    day : str, optional
        Filter by day (YYYYMMDD). For anticipations, this is the event date.
    facet : str, optional
        Filter by facet identifier.
    start : str, optional
        Filter events ending at or after this time (HH:MM:SS).
    end : str, optional
        Filter events starting at or before this time (HH:MM:SS).
    topic : str, optional
        Filter by insight topic.
    occurred : bool, optional
        Filter by occurred status. True for occurrences, False for anticipations.
        None returns both.
    """
    conn, _ = get_index(index="events")

    # Build WHERE clause and parameters
    params: list = []

    # Only use FTS MATCH if query is non-empty
    if query:
        sanitized = sanitize_fts_query(query)
        where_clause = f"events_text MATCH '{sanitized}'"
    else:
        # No search query, just filter by metadata
        where_clause = "1=1"

    if day:
        where_clause += " AND m.day=?"
        params.append(day)
    if facet:
        where_clause += " AND m.facet=?"
        params.append(facet)
    if topic:
        where_clause += " AND m.topic=?"
        params.append(topic)
    if start:
        where_clause += " AND m.end>=?"
        params.append(start)
    if end:
        where_clause += " AND m.start<=?"
        params.append(end)
    if occurred is not None:
        where_clause += " AND m.occurred=?"
        params.append(1 if occurred else 0)

    # Get total count
    total = conn.execute(
        f"""
        SELECT count(*)
        FROM events_text t JOIN event_match m ON t.path=m.path AND t.idx=m.idx
        WHERE {where_clause}
        """,
        params,
    ).fetchone()[0]

    # Get results with limit and offset, ordered by day and start time (newest first)
    sql = f"""
        SELECT t.content,
               m.path, m.day, m.idx, m.topic, m.facet, m.start, m.end, m.occurred,
               bm25(events_text) as rank
        FROM events_text t JOIN event_match m ON t.path=m.path AND t.idx=m.idx
        WHERE {where_clause}
        ORDER BY m.day DESC, m.start DESC LIMIT ? OFFSET ?
    """

    cursor = conn.execute(sql, params + [limit, offset])
    results = []
    for row in cursor.fetchall():
        (
            content,
            path,
            day_label,
            idx,
            topic_label,
            facet_val,
            start_val,
            end_val,
            occurred_val,
            rank,
        ) = row
        try:
            occ_obj = json.loads(content)
        except Exception:
            occ_obj = {}
        text = (
            occ_obj.get("title")
            or occ_obj.get("summary")
            or occ_obj.get("subject")
            or occ_obj.get("details")
            or content
        )
        results.append(
            {
                "id": f"{path}:{idx}",
                "text": text,
                "metadata": {
                    "day": day_label,
                    "path": path,
                    "index": idx,
                    "topic": topic_label,
                    "facet": facet_val,
                    "start": start_val,
                    "end": end_val,
                    "occurred": bool(occurred_val),
                    "participants": occ_obj.get("participants"),
                },
                "score": rank,
                "event": occ_obj,
            }
        )
    conn.close()
    return total, results


def format_events(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format event JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (one event per line)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting facet name and day)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of {"timestamp": int, "markdown": str} dicts, one per event
            - meta: Dict with optional "header" and "error" keys
    """
    import re
    from datetime import datetime
    from pathlib import Path

    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    skipped_count = 0

    # Extract facet name and day from path
    facet_name = "unknown"
    day_str: str | None = None

    if file_path:
        file_path = Path(file_path)

        # Extract facet name from path: facets/{facet}/events/YYYYMMDD.jsonl
        path_str = str(file_path)
        facet_match = re.search(r"facets/([^/]+)/events", path_str)
        if facet_match:
            facet_name = facet_match.group(1)

        # Extract day from filename
        if file_path.stem.isdigit() and len(file_path.stem) == 8:
            day_str = file_path.stem

    # Calculate base timestamp (midnight of the event day) in milliseconds
    base_ts = 0
    if day_str:
        try:
            dt = datetime.strptime(day_str, "%Y%m%d")
            base_ts = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Count occurrences vs anticipations for header
    occurred_count = sum(1 for e in entries if e.get("occurred", True))
    anticipated_count = len(entries) - occurred_count

    # Build header
    if day_str:
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        header_title = f"# Events: {facet_name} ({formatted_day})"
    else:
        header_title = f"# Events: {facet_name}"

    event_count = len(entries)
    event_word = "event" if event_count == 1 else "events"
    status_parts = []
    if occurred_count:
        status_parts.append(f"{occurred_count} occurred")
    if anticipated_count:
        status_parts.append(f"{anticipated_count} anticipated")

    if status_parts:
        meta["header"] = f"{header_title}\n\n{event_count} {event_word} ({', '.join(status_parts)})"
    else:
        meta["header"] = f"{header_title}\n\n{event_count} {event_word}"

    # Format each event as a chunk
    for event in entries:
        # Skip entries without title
        title = event.get("title")
        if not title:
            skipped_count += 1
            continue

        event_type = event.get("type", "event").capitalize()
        occurred = event.get("occurred", True)

        # Calculate timestamp from day + start time
        ts = base_ts
        start_time = event.get("start", "")
        if start_time and base_ts:
            try:
                # Parse HH:MM:SS or HH:MM
                time_parts = start_time.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                ts = base_ts + (hours * 3600 + minutes * 60 + seconds) * 1000
            except (ValueError, IndexError):
                pass

        # Build markdown
        type_prefix = "Planned " if not occurred else ""
        lines = [f"### {type_prefix}{event_type}: {title}\n", ""]

        # Time range (24h format, strip seconds for display)
        end_time = event.get("end", "")
        time_label = "Occurred" if occurred else "Scheduled"
        if start_time:
            start_display = start_time[:5] if len(start_time) >= 5 else start_time
            if end_time:
                end_display = end_time[:5] if len(end_time) >= 5 else end_time
                lines.append(f"**Time {time_label}:** {start_display} - {end_display}")
            else:
                lines.append(f"**Time {time_label}:** {start_display}")

        # Participants
        participants = event.get("participants", [])
        if participants and isinstance(participants, list):
            participants_label = "Expected Participants" if not occurred else "Participants"
            lines.append(f"**{participants_label}:** {', '.join(participants)}")

        # For anticipations, show when it was created (from source path)
        if not occurred:
            source = event.get("source", "")
            # Extract YYYYMMDD from source path like "20240101/insights/schedule.md"
            source_match = re.match(r"(\d{8})/", source)
            if source_match:
                created_day = source_match.group(1)
                created_formatted = (
                    f"{created_day[:4]}-{created_day[4:6]}-{created_day[6:8]}"
                )
                lines.append(f"**Created on:** {created_formatted}")

        lines.append("")

        # Summary
        summary = event.get("summary", "")
        if summary:
            lines.append(summary)
            lines.append("")

        # Details
        details = event.get("details", "")
        if details:
            lines.append(details)
            lines.append("")

        chunks.append({"timestamp": ts, "markdown": "\n".join(lines)})

    # Report skipped entries
    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing 'title' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logging.info(error_msg)

    return chunks, meta
