"""Integration tests for the events indexer."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from think.indexer import (
    reset_index,
    scan_events,
    search_events,
)


@pytest.mark.integration
def test_events_indexer_scan_and_search():
    """Test scanning and searching event files from fixtures."""
    # Use fixtures journal path
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"

    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")

    # Create a temporary index directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set environment to use fixtures journal
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)

        try:
            # Set up test index directory
            test_index_dir = Path(tmpdir) / "indexer"

            # Monkey-patch the get_index function
            import think.indexer.core

            old_get_index = think.indexer.core.get_index

            def test_get_index(*, index, journal=None, day=None):
                db_name = think.indexer.core.DB_NAMES[index]
                db_path = test_index_dir / db_name
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path))
                # Ensure schema is created
                for statement in think.indexer.core.SCHEMAS[index]:
                    conn.execute(statement)
                return conn, str(db_path)

            think.indexer.core.get_index = test_get_index

            # Reset index to ensure clean state
            reset_index(str(journal_path), "events")

            # Scan the events
            scan_count = scan_events(str(journal_path))

            # We should have scanned event files (occurrences.json, ponder_meetings.json)
            # scan_count is True if files were scanned, False if all were up to date
            # Just continue - the search will verify if it worked

            # Search for meetings content (from fixtures/journal/facets/work/events/20240101.jsonl)
            total, results = search_events("Team standup")
            assert total > 0, "Should find 'Team standup' meeting"

            # Check for specific meeting times we added
            total, results = search_events("09:00")
            assert total > 0, "Should find 09:00 meeting time"

            total, results = search_events("Code review")
            assert total > 0, "Should find 'Code review' task"

            total, results = search_events("Daily sync")
            assert total > 0, "Should find 'Daily sync' in meeting summary"

            # Verify result structure
            if results:
                result = results[0]
                assert "text" in result
                assert "metadata" in result
                assert "id" in result
                assert "path" in result["metadata"]
                assert "day" in result["metadata"]
                assert "index" in result["metadata"]

        finally:
            # Restore original environment
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]

            # Restore original function
            think.indexer.core.get_index = old_get_index


@pytest.mark.integration
def test_events_indexer_with_custom_events():
    """Test events indexer with additional custom event files."""
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"

    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")

    # Create some additional event files for testing in JSONL format
    events_data = [
        {
            "type": "deployment",
            "start": "10:30:00",
            "end": "10:45:00",
            "title": "Deploy auth service",
            "summary": "Deployed authentication service to staging",
            "facet": "work",
            "topic": "activity",
            "occurred": True,
            "source": "20240101/insights/activity.md",
            "details": "version 1.2.3, environment staging",
        },
        {
            "type": "incident",
            "start": "15:45:00",
            "end": "16:00:00",
            "title": "Database timeout resolved",
            "summary": "Database connection timeout issue resolved",
            "facet": "work",
            "topic": "activity",
            "occurred": True,
            "source": "20240101/insights/activity.md",
            "details": "severity medium, duration 15 minutes",
        },
    ]

    # Write to JSONL in facets/work/events/20240101.jsonl
    events_dir = journal_path / "facets" / "work" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    events_file = events_dir / "20240101_custom.jsonl"
    jsonl_content = "\n".join(json.dumps(e) for e in events_data)
    events_file.write_text(jsonl_content)

    with tempfile.TemporaryDirectory() as tmpdir:
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)

        try:
            # Set up test index
            test_index_dir = Path(tmpdir) / "indexer"

            import think.indexer.core

            old_get_index = think.indexer.core.get_index

            def test_get_index(*, index, journal=None, day=None):
                db_name = think.indexer.core.DB_NAMES[index]
                db_path = test_index_dir / db_name
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path))
                # Ensure schema is created
                for statement in think.indexer.core.SCHEMAS[index]:
                    conn.execute(statement)
                return conn, str(db_path)

            think.indexer.core.get_index = test_get_index

            # Reset and scan
            reset_index(str(journal_path), "events")
            scan_events(str(journal_path))

            # Search for deployment event
            total, results = search_events("authentication service staging")
            assert total > 0, "Should find deployment event"

            # Search for incident
            total, results = search_events("connection timeout")
            assert total > 0, "Should find timeout incident"

        finally:
            # Clean up the test file
            if events_file.exists():
                events_file.unlink()

            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]

            think.indexer.core.get_index = old_get_index


@pytest.mark.integration
def test_events_indexer_rescan():
    """Test that rescanning events handles updates properly."""
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"

    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)

        try:
            # Set up test index
            test_index_dir = Path(tmpdir) / "indexer"

            import think.indexer.core

            old_get_index = think.indexer.core.get_index

            def test_get_index(*, index, journal=None, day=None):
                db_name = think.indexer.core.DB_NAMES[index]
                db_path = test_index_dir / db_name
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path))
                # Ensure schema is created
                for statement in think.indexer.core.SCHEMAS[index]:
                    conn.execute(statement)
                return conn, str(db_path)

            think.indexer.core.get_index = test_get_index

            # Initial scan
            reset_index(str(journal_path), "events")
            first_scan = scan_events(str(journal_path))
            # first_scan is boolean - just continue

            # Search for content
            total1, results1 = search_events("Team sync")
            initial_count = total1
            assert initial_count > 0, "Should find Team sync initially"

            # Rescan (should handle existing content gracefully)
            second_scan = scan_events(str(journal_path))

            # Results should be consistent
            total2, results2 = search_events("Team sync")
            assert total2 == initial_count, "Rescan should not duplicate entries"

        finally:
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]

            think.indexer.core.get_index = old_get_index
