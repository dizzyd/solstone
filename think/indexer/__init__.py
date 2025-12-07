"""Indexer package for journal content.

This module provides the unified journal index and backward-compatible
re-exports for the legacy separate indexes.
"""

# Import from cli
from .cli import main

# Legacy imports from core
from .core import (
    DATE_RE,
    DB_NAMES,
    INDEX_DIR,
    SCHEMAS,
    get_index,
    reset_index,
    sanitize_fts_query,
)

# Legacy imports from events
from .events import (
    scan_events,
    search_events,
)

# Legacy imports from insights
from .insights import (
    find_event_files,
    find_insight_files,
    scan_insights,
    search_insights,
)

# Import from journal (new unified index)
from .journal import (
    get_events,
    get_journal_index,
    reset_journal_index,
    scan_journal,
    search_journal,
)

# Legacy imports from transcripts
from .transcripts import (
    AUDIO_RE,
    SCREEN_RE,
    find_transcript_files,
    scan_transcripts,
    search_transcripts,
)

# All public functions and constants
__all__ = [
    # Journal (new unified index)
    "get_events",
    "get_journal_index",
    "reset_journal_index",
    "scan_journal",
    "search_journal",
    # Legacy: Core
    "DATE_RE",
    "DB_NAMES",
    "INDEX_DIR",
    "SCHEMAS",
    "get_index",
    "reset_index",
    "sanitize_fts_query",
    # Legacy: Insights
    "find_event_files",
    "find_insight_files",
    "scan_insights",
    "search_insights",
    # Legacy: Events
    "scan_events",
    "search_events",
    # Legacy: Transcripts
    "AUDIO_RE",
    "SCREEN_RE",
    "find_transcript_files",
    "scan_transcripts",
    "search_transcripts",
    # CLI
    "main",
]
