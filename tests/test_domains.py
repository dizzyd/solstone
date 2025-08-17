"""Tests for think.domains module."""

from pathlib import Path

import pytest

from think.domains import domain_summary

# Use the permanent fixtures in fixtures/journal/domains/
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "journal"


def test_domain_summary_full(monkeypatch):
    """Test domain_summary with full metadata."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("full-featured")

    # Check title with emoji
    assert "# 🚀 Full Featured Domain" in summary

    # Check description
    assert "**Description:** A domain for testing all features" in summary

    # Check color badge
    assert "![Color](#28a745)" in summary

    # Check entities section
    assert "## Entities" in summary
    assert "**Entity 1**: First test entity" in summary
    assert "**Entity 2**: Second test entity" in summary
    assert "**Entity 3**: Third test entity with description" in summary

    # Check matters section
    assert "## Matters" in summary
    assert "**Total:** 2 matter(s)" in summary

    # Check active matters
    assert "### Active (1)" in summary
    assert "🔴 **matter_1**: High Priority Task" in summary

    # Check completed matters
    assert "### Completed (1)" in summary
    assert "**matter_2**: Completed Task" in summary


def test_domain_summary_minimal(monkeypatch):
    """Test domain_summary with minimal metadata."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("minimal-domain")

    # Check title without emoji
    assert "# Minimal Domain" in summary

    # Should not have description, color, entities, or matters
    assert "**Description:**" not in summary
    assert "![Color]" not in summary
    assert "## Entities" not in summary
    assert "## Matters" not in summary


def test_domain_summary_test_domain(monkeypatch):
    """Test domain_summary with the existing test-domain fixture."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("test-domain")

    # Check title with emoji
    assert "# 🧪 Test Domain" in summary

    # Check description
    assert (
        "**Description:** A test domain for validating matter functionality" in summary
    )

    # Check color badge
    assert "![Color](#007bff)" in summary

    # Check matters section
    assert "## Matters" in summary
    assert "**matter_1**: Test Matter" in summary


def test_domain_summary_nonexistent(monkeypatch):
    """Test domain_summary with nonexistent domain."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    with pytest.raises(FileNotFoundError, match="Domain 'nonexistent' not found"):
        domain_summary("nonexistent")


def test_domain_summary_no_journal_path(monkeypatch):
    """Test domain_summary without JOURNAL_PATH set."""
    # Set to empty string to override any .env file
    monkeypatch.setenv("JOURNAL_PATH", "")

    with pytest.raises(RuntimeError, match="JOURNAL_PATH not set"):
        domain_summary("any-domain")


def test_domain_summary_missing_domain_json(monkeypatch):
    """Test domain_summary with missing domain.json."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    with pytest.raises(FileNotFoundError, match="domain.json not found"):
        domain_summary("broken-domain")


def test_domain_summary_empty_entities(monkeypatch):
    """Test domain_summary with empty entities file."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("empty-entities")

    # Should not include entities section if file is empty
    assert "## Entities" not in summary


def test_domain_summary_matter_priorities(monkeypatch):
    """Test domain_summary with different matter priorities."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("priority-test")

    # Check priority markers
    assert "🔴 **matter_1**: High Priority" in summary
    assert "🟡 **matter_2**: Medium Priority" in summary
    # Normal priority has no marker
    assert "**matter_3**: Normal Priority" in summary
    assert "🔴 **matter_3**" not in summary
    assert "🟡 **matter_3**" not in summary
