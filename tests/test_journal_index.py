"""Tests for the unified journal index."""

import json
import os

import pytest


@pytest.fixture
def journal_fixture(tmp_path):
    """Create a temporary journal with test data."""
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create daily insight
    day = journal / "20240101"
    day.mkdir()
    insights_dir = day / "insights"
    insights_dir.mkdir()
    (insights_dir / "flow.md").write_text(
        "# Flow Summary\n\nWorked on project alpha.\n"
    )

    # Create segment with audio transcript
    segment = day / "100000"
    segment.mkdir()
    (segment / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello world"}
        )
        + "\n"
    )

    # Create segment markdown
    (segment / "screen.md").write_text("# Screen Summary\n\nViewed documentation.\n")

    # Create facet events
    events_dir = journal / "facets" / "work" / "events"
    events_dir.mkdir(parents=True)
    event = {
        "type": "meeting",
        "start": "09:00:00",
        "end": "09:30:00",
        "title": "Standup",
        "summary": "Daily sync meeting",
        "facet": "work",
        "topic": "meetings",
        "occurred": True,
    }
    (events_dir / "20240101.jsonl").write_text(json.dumps(event))

    # Create facet entities
    entities_dir = journal / "facets" / "work" / "entities"
    entities_dir.mkdir(parents=True)
    entity = {
        "name": "Project Alpha",
        "type": "project",
        "description": "Main project",
    }
    (entities_dir / "20240101.jsonl").write_text(json.dumps(entity))

    # Create facet news
    news_dir = journal / "facets" / "work" / "news"
    news_dir.mkdir(parents=True)
    (news_dir / "20240101.md").write_text(
        "# News\n\nImportant update about the project.\n"
    )

    return journal


def test_scan_journal(journal_fixture):
    """Test scanning journal creates index."""
    from think.indexer.journal import scan_journal

    changed = scan_journal(str(journal_fixture), verbose=True)
    assert changed is True

    # Index file should exist
    index_path = journal_fixture / "indexer" / "journal.sqlite"
    assert index_path.exists()


def test_search_journal_insights(journal_fixture):
    """Test searching returns insight chunks."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    total, results = search_journal("project alpha")
    assert total >= 1
    # Should find the flow insight mentioning "project alpha"
    found = any("alpha" in r["text"].lower() for r in results)
    assert found


def test_search_journal_events(journal_fixture):
    """Test searching returns event chunks."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    total, results = search_journal("Standup", topic="event")
    assert total >= 1
    assert any("Standup" in r["text"] for r in results)


def test_search_journal_filter_by_day(journal_fixture):
    """Test filtering search by day."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with day filter
    total, results = search_journal("", day="20240101")
    assert total >= 1
    for r in results:
        assert r["metadata"]["day"] == "20240101"


def test_search_journal_filter_by_facet(journal_fixture):
    """Test filtering search by facet."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with facet filter
    total, results = search_journal("", facet="work")
    assert total >= 1
    for r in results:
        assert r["metadata"]["facet"] == "work"


def test_search_journal_filter_by_topic(journal_fixture):
    """Test filtering search by topic."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search events by topic
    total, results = search_journal("", topic="event")
    assert total >= 1
    for r in results:
        assert r["metadata"]["topic"] == "event"


def test_get_events(journal_fixture):
    """Test get_events returns structured event data."""
    from think.indexer.journal import get_events

    events = get_events("20240101")
    assert len(events) == 1
    assert events[0]["title"] == "Standup"
    assert events[0]["start"] == "09:00:00"
    assert events[0]["facet"] == "work"


def test_get_events_filter_by_facet(journal_fixture):
    """Test get_events with facet filter."""
    from think.indexer.journal import get_events

    # Should find work facet events
    events = get_events("20240101", facet="work")
    assert len(events) == 1

    # Should not find nonexistent facet
    events = get_events("20240101", facet="personal")
    assert len(events) == 0


def test_parse_path_metadata():
    """Test path metadata extraction."""
    from think.indexer.journal import parse_path_metadata

    # Daily insight
    meta = parse_path_metadata("20240101/insights/flow.md")
    assert meta["day"] == "20240101"
    assert meta["facet"] == ""
    assert meta["topic"] == "flow"

    # Segment markdown
    meta = parse_path_metadata("20240101/100000/screen.md")
    assert meta["day"] == "20240101"
    assert meta["topic"] == "screen"

    # Segment audio
    meta = parse_path_metadata("20240101/100000/audio.jsonl")
    assert meta["day"] == "20240101"
    assert meta["topic"] == "audio"

    # Facet event
    meta = parse_path_metadata("facets/work/events/20240101.jsonl")
    assert meta["day"] == "20240101"
    assert meta["facet"] == "work"
    assert meta["topic"] == "event"

    # Facet entities detected
    meta = parse_path_metadata("facets/work/entities/20240101.jsonl")
    assert meta["day"] == "20240101"
    assert meta["facet"] == "work"
    assert meta["topic"] == "entity:detected"

    # Facet news
    meta = parse_path_metadata("facets/work/news/20240101.md")
    assert meta["day"] == "20240101"
    assert meta["facet"] == "work"
    assert meta["topic"] == "news"

    # Import summary
    meta = parse_path_metadata("imports/20240101_093000/summary.md")
    assert meta["day"] == "20240101"
    assert meta["topic"] == "import"

    # App insight
    meta = parse_path_metadata("apps/myapp/insights/custom.md")
    assert meta["day"] == ""
    assert meta["topic"] == "myapp:custom"


def test_reset_journal_index(journal_fixture):
    """Test resetting the journal index."""
    from think.indexer.journal import reset_journal_index, scan_journal

    scan_journal(str(journal_fixture))
    index_path = journal_fixture / "indexer" / "journal.sqlite"
    assert index_path.exists()

    reset_journal_index(str(journal_fixture))
    assert not index_path.exists()


def test_index_caching(journal_fixture):
    """Test that unchanged files are not re-indexed."""
    from think.indexer.journal import scan_journal

    # First scan indexes files
    changed = scan_journal(str(journal_fixture))
    assert changed is True

    # Second scan should be a no-op (all cached)
    changed = scan_journal(str(journal_fixture))
    assert changed is False


def test_find_formattable_files(journal_fixture):
    """Test file discovery function."""
    from think.formatters import find_formattable_files

    files = find_formattable_files(str(journal_fixture))

    # Should find various file types
    paths = set(files.keys())

    # Daily insights
    assert "20240101/insights/flow.md" in paths

    # Segment content
    assert "20240101/100000/screen.md" in paths
    assert "20240101/100000/audio.jsonl" in paths

    # Facet content
    assert "facets/work/events/20240101.jsonl" in paths
    assert "facets/work/entities/20240101.jsonl" in paths
    assert "facets/work/news/20240101.md" in paths


def test_search_journal_empty_query(journal_fixture):
    """Test search with empty query returns all results."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Empty query should return all chunks
    total, results = search_journal("")
    assert total > 0


def test_search_journal_pagination(journal_fixture):
    """Test search pagination."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Get first page
    total, results1 = search_journal("", limit=2, offset=0)

    # Get second page
    _, results2 = search_journal("", limit=2, offset=2)

    # Results should be different (if enough data)
    if total > 2:
        ids1 = {r["id"] for r in results1}
        ids2 = {r["id"] for r in results2}
        assert ids1 != ids2
