# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for facet-scoped entity utilities."""

import os

import pytest

from think.entities import (
    ensure_entity_folder,
    entity_file_path,
    entity_folder_path,
    load_all_attached_entities,
    load_detected_entities_recent,
    load_entities,
    normalize_entity_name,
    rename_entity_folder,
    save_entities,
)


@pytest.fixture
def fixture_journal():
    """Set JOURNAL_PATH to fixtures/journal for testing."""
    os.environ["JOURNAL_PATH"] = "fixtures/journal"
    yield
    # No cleanup needed - just testing reads


def test_entity_file_path_attached(fixture_journal):
    """Test path generation for attached entities."""
    path = entity_file_path("personal")
    assert str(path).endswith("fixtures/journal/facets/personal/entities.jsonl")
    assert path.name == "entities.jsonl"


def test_entity_file_path_detected(fixture_journal):
    """Test path generation for detected entities."""
    path = entity_file_path("personal", "20250101")
    assert str(path).endswith(
        "fixtures/journal/facets/personal/entities/20250101.jsonl"
    )
    assert path.name == "20250101.jsonl"


def test_load_entities_attached(fixture_journal):
    """Test loading attached entities from fixtures."""
    entities = load_entities("personal")
    assert len(entities) == 3

    # Check entities are dicts with expected fields
    alice = next(e for e in entities if e.get("name") == "Alice Johnson")
    assert alice["type"] == "Person"
    assert alice["description"] == "Close friend from college"
    # Check extended fields are preserved
    assert alice.get("tags") == ["friend"]
    assert alice.get("contact") == "alice@example.com"

    bob = next(e for e in entities if e.get("name") == "Bob Smith")
    assert bob["type"] == "Person"
    assert bob["description"] == "Neighbor"

    acme = next(e for e in entities if e.get("name") == "Acme Corp")
    assert acme["type"] == "Company"
    assert acme["description"] == "Local tech startup"


def test_load_entities_detected(fixture_journal):
    """Test loading detected entities from fixtures."""
    entities = load_entities("personal", "20250101")
    assert len(entities) == 2

    charlie = next(e for e in entities if e.get("name") == "Charlie Brown")
    assert charlie["type"] == "Person"
    assert charlie["description"] == "Met at coffee shop"

    project = next(e for e in entities if e.get("name") == "Home Renovation")
    assert project["type"] == "Project"
    assert project["description"] == "Kitchen remodel project"


def test_load_entities_missing_file(fixture_journal):
    """Test loading from non-existent file returns empty list."""
    entities = load_entities("personal", "20991231")
    assert entities == []


def test_load_entities_missing_facet(fixture_journal):
    """Test loading from non-existent facet returns empty list."""
    entities = load_entities("nonexistent")
    assert entities == []


def test_save_and_load_entities(fixture_journal, tmp_path):
    """Test saving and loading entities with real files."""
    # Create a temporary facet structure
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)

    # Update JOURNAL_PATH to temp directory
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save some entities (dicts with extended fields)
    test_entities = [
        {
            "type": "Person",
            "name": "Test Person",
            "description": "Test description",
            "role": "tester",
        },
        {"type": "Company", "name": "Test Co", "description": "Test company"},
    ]
    save_entities("test_facet", test_entities, "20250101")

    # Load them back
    loaded = load_entities("test_facet", "20250101")
    assert len(loaded) == 2

    person = next(e for e in loaded if e.get("name") == "Test Person")
    assert person["type"] == "Person"
    assert person["description"] == "Test description"
    assert person.get("role") == "tester"  # Extended field preserved

    company = next(e for e in loaded if e.get("name") == "Test Co")
    assert company["type"] == "Company"
    assert company["description"] == "Test company"

    # Verify file exists and has correct JSONL format
    entity_file = entities_dir / "20250101.jsonl"
    assert entity_file.exists()
    content = entity_file.read_text()
    # Should be valid JSONL
    lines = [line for line in content.strip().split("\n") if line]
    assert len(lines) == 2
    import json

    for line in lines:
        assert json.loads(line)  # Should not raise


def test_save_entities_sorting(fixture_journal, tmp_path):
    """Test that saved entities are sorted by type then name."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save unsorted entities
    import json

    unsorted = [
        {
            "type": "Project",
            "name": "Zebra Project",
            "description": "Last alphabetically",
        },
        {"type": "Company", "name": "Acme", "description": "Company name"},
        {"type": "Person", "name": "Alice", "description": "Person name"},
        {"type": "Company", "name": "Beta Corp", "description": "Another company"},
    ]
    save_entities("test_facet", unsorted)

    # Verify sorting in file (JSONL format)
    entity_file = facet_path / "entities.jsonl"
    lines = entity_file.read_text().strip().split("\n")
    entities = [json.loads(line) for line in lines if line]

    assert entities[0]["type"] == "Company" and entities[0]["name"] == "Acme"
    assert entities[1]["type"] == "Company" and entities[1]["name"] == "Beta Corp"
    assert entities[2]["type"] == "Person" and entities[2]["name"] == "Alice"
    assert entities[3]["type"] == "Project" and entities[3]["name"] == "Zebra Project"


def test_load_all_attached_entities(fixture_journal):
    """Test loading all attached entities from all facets."""
    all_entities = load_all_attached_entities()

    # Should have entities from both personal and full-featured facets
    assert len(all_entities) >= 3  # At least the personal facet entities

    # Check personal facet entities are present
    entity_names = [e.get("name") for e in all_entities]
    assert "Alice Johnson" in entity_names
    assert "Bob Smith" in entity_names
    assert "Acme Corp" in entity_names


def test_load_all_attached_entities_deduplication(fixture_journal, tmp_path):
    """Test that load_all_attached_entities deduplicates by name."""
    # Create two facets with overlapping entity names
    facet1_path = tmp_path / "facets" / "facet1"
    facet2_path = tmp_path / "facets" / "facet2"
    facet1_path.mkdir(parents=True)
    facet2_path.mkdir(parents=True)

    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save same entity name in both facets with different descriptions
    entities1 = [
        {
            "type": "Person",
            "name": "John Smith",
            "description": "Description from facet1",
        }
    ]
    entities2 = [
        {
            "type": "Person",
            "name": "John Smith",
            "description": "Description from facet2",
        }
    ]

    save_entities("facet1", entities1)
    save_entities("facet2", entities2)

    # Load all entities
    all_entities = load_all_attached_entities()

    # Should only have one "John Smith" (from first facet alphabetically)
    john_smiths = [e for e in all_entities if e.get("name") == "John Smith"]
    assert len(john_smiths) == 1
    # Should be from facet1 (alphabetically first)
    assert john_smiths[0]["description"] == "Description from facet1"


def test_aka_field_preservation(fixture_journal, tmp_path):
    """Test that aka field is preserved during save/load operations."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with aka fields
    test_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Lead engineer",
            "aka": ["Ali", "AJ"],
        },
        {
            "type": "Company",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres", "PG"],
        },
    ]
    save_entities("test_facet", test_entities)

    # Load them back
    loaded = load_entities("test_facet")
    assert len(loaded) == 2

    alice = next(e for e in loaded if e.get("name") == "Alice Johnson")
    assert alice.get("aka") == ["Ali", "AJ"]

    postgres = next(e for e in loaded if e.get("name") == "PostgreSQL")
    assert postgres.get("aka") == ["Postgres", "PG"]


# Tests for load_detected_entities_recent


def test_load_detected_entities_recent_basic(fixture_journal):
    """Test loading detected entities with count and last_seen."""
    # Fixture has detected entities in 20250101 and 20250102
    # But these dates are old (> 30 days from now), so we need to use a large days value
    detected = load_detected_entities_recent("personal", days=36500)  # ~100 years

    # Should have 4 detected entities (Charlie Brown, Home Renovation, City Fitness, Diana Prince)
    # Note: excludes Alice Johnson, Bob Smith, Acme Corp which are attached
    assert len(detected) == 4

    # Check structure includes count and last_seen
    for entity in detected:
        assert "type" in entity
        assert "name" in entity
        assert "description" in entity
        assert "count" in entity
        assert "last_seen" in entity


def test_load_detected_entities_recent_excludes_attached(fixture_journal, tmp_path):
    """Test that attached entities and their akas are excluded from detected results."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity with aka
    attached = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Attached person",
            "aka": ["Ali", "AJ"],
        }
    ]
    save_entities("test_facet", attached)

    # Create detected entities including some that match attached/aka
    detected_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Should be excluded",
        },
        {"type": "Person", "name": "Ali", "description": "Should be excluded (aka)"},
        {
            "type": "Person",
            "name": "Charlie Brown",
            "description": "Should be included",
        },
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected - should only get Charlie Brown
    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 1
    assert detected[0]["name"] == "Charlie Brown"


def test_load_detected_entities_recent_count_tracking(fixture_journal, tmp_path):
    """Test that count tracks occurrences across multiple days."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create same entity across multiple days
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Day 1 desc"}],
        "20250101",
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Day 2 desc"}],
        "20250102",
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Day 3 desc"}],
        "20250103",
    )

    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 1

    charlie = detected[0]
    assert charlie["name"] == "Charlie"
    assert charlie["count"] == 3


def test_load_detected_entities_recent_last_seen(fixture_journal, tmp_path):
    """Test that last_seen is the most recent day and description is from that day."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create entity across multiple days with different descriptions
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Oldest description"}],
        "20250101",
    )
    save_entities(
        "test_facet",
        [
            {
                "type": "Person",
                "name": "Charlie",
                "description": "Most recent description",
            }
        ],
        "20250103",
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Middle description"}],
        "20250102",
    )

    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 1

    charlie = detected[0]
    assert charlie["last_seen"] == "20250103"
    assert charlie["description"] == "Most recent description"


def test_load_detected_entities_recent_days_filter(fixture_journal, tmp_path):
    """Test that days parameter limits results to recent days."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    from datetime import datetime, timedelta

    # Create entities at various dates relative to today
    today = datetime.now()
    recent_day = (today - timedelta(days=5)).strftime("%Y%m%d")
    old_day = (today - timedelta(days=60)).strftime("%Y%m%d")

    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Recent Person", "description": "Recent"}],
        recent_day,
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Old Person", "description": "Old"}],
        old_day,
    )

    # With default 30 days, should only get recent person
    detected = load_detected_entities_recent("test_facet", days=30)
    assert len(detected) == 1
    assert detected[0]["name"] == "Recent Person"

    # With 90 days, should get both
    detected = load_detected_entities_recent("test_facet", days=90)
    assert len(detected) == 2


def test_load_detected_entities_recent_empty_facet(fixture_journal, tmp_path):
    """Test that empty or non-existent facet returns empty list."""
    facet_path = tmp_path / "facets" / "empty_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # No entities directory
    detected = load_detected_entities_recent("empty_facet")
    assert detected == []


def test_load_detected_entities_recent_type_name_key(fixture_journal, tmp_path):
    """Test that deduplication is by (type, name) tuple, not just name."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Same name, different types - should be treated as separate entities
    save_entities(
        "test_facet",
        [
            {"type": "Person", "name": "Mercury", "description": "Roman god"},
            {"type": "Project", "name": "Mercury", "description": "Space program"},
        ],
        "20250101",
    )

    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 2

    names_and_types = {(e["type"], e["name"]) for e in detected}
    assert ("Person", "Mercury") in names_and_types
    assert ("Project", "Mercury") in names_and_types


def test_timestamp_preservation(fixture_journal, tmp_path):
    """Test that attached_at and updated_at timestamps are preserved through save/load."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with timestamps
    test_entities = [
        {
            "type": "Person",
            "name": "Alice",
            "description": "Test person",
            "attached_at": 1700000000000,
            "updated_at": 1700000001000,
        },
        {
            "type": "Company",
            "name": "Acme",
            "description": "Test company",
            "attached_at": 1700000002000,
            "updated_at": 1700000002000,
        },
    ]
    save_entities("test_facet", test_entities)

    # Load them back
    loaded = load_entities("test_facet")
    assert len(loaded) == 2

    alice = next(e for e in loaded if e.get("name") == "Alice")
    assert alice["attached_at"] == 1700000000000
    assert alice["updated_at"] == 1700000001000

    acme = next(e for e in loaded if e.get("name") == "Acme")
    assert acme["attached_at"] == 1700000002000
    assert acme["updated_at"] == 1700000002000


# Tests for detached entity functionality


def test_load_entities_excludes_detached_by_default(fixture_journal, tmp_path):
    """Test that load_entities excludes detached entities by default."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with one detached
    test_entities = [
        {"type": "Person", "name": "Alice", "description": "Active person"},
        {
            "type": "Person",
            "name": "Bob",
            "description": "Detached person",
            "detached": True,
        },
        {"type": "Company", "name": "Acme", "description": "Active company"},
    ]
    save_entities("test_facet", test_entities)

    # Load without include_detached (default)
    loaded = load_entities("test_facet")
    assert len(loaded) == 2
    names = [e["name"] for e in loaded]
    assert "Alice" in names
    assert "Acme" in names
    assert "Bob" not in names


def test_load_entities_includes_detached_when_requested(fixture_journal, tmp_path):
    """Test that load_entities includes detached entities when include_detached=True."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with one detached
    test_entities = [
        {"type": "Person", "name": "Alice", "description": "Active person"},
        {
            "type": "Person",
            "name": "Bob",
            "description": "Detached person",
            "detached": True,
        },
    ]
    save_entities("test_facet", test_entities)

    # Load with include_detached=True
    loaded = load_entities("test_facet", include_detached=True)
    assert len(loaded) == 2
    names = [e["name"] for e in loaded]
    assert "Alice" in names
    assert "Bob" in names

    # Verify detached flag is preserved
    bob = next(e for e in loaded if e["name"] == "Bob")
    assert bob.get("detached") is True


def test_load_all_attached_entities_excludes_detached(fixture_journal, tmp_path):
    """Test that load_all_attached_entities excludes detached entities."""
    facet1_path = tmp_path / "facets" / "facet1"
    facet2_path = tmp_path / "facets" / "facet2"
    facet1_path.mkdir(parents=True)
    facet2_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities - one active, one detached per facet
    save_entities(
        "facet1",
        [
            {"type": "Person", "name": "Alice", "description": "Active in facet1"},
            {
                "type": "Person",
                "name": "Bob",
                "description": "Detached in facet1",
                "detached": True,
            },
        ],
    )
    save_entities(
        "facet2",
        [
            {"type": "Person", "name": "Charlie", "description": "Active in facet2"},
        ],
    )

    all_entities = load_all_attached_entities()

    # Should only have active entities
    names = [e["name"] for e in all_entities]
    assert "Alice" in names
    assert "Charlie" in names
    assert "Bob" not in names


def test_load_detected_entities_recent_shows_detached_entity_names(
    fixture_journal, tmp_path
):
    """Test that detached entities appear in detected list again (not excluded)."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity with detached=True
    attached = [
        {"type": "Person", "name": "Alice", "description": "Active person"},
        {
            "type": "Person",
            "name": "Bob",
            "description": "Detached person",
            "detached": True,
        },
    ]
    save_entities("test_facet", attached)

    # Create detected entities including the detached name
    detected_entities = [
        {
            "type": "Person",
            "name": "Alice",
            "description": "Should be excluded (active)",
        },
        {
            "type": "Person",
            "name": "Bob",
            "description": "Should be INCLUDED (detached)",
        },
        {
            "type": "Person",
            "name": "Charlie",
            "description": "Should be included (new)",
        },
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected - Alice excluded (active), Bob included (detached), Charlie included (new)
    detected = load_detected_entities_recent("test_facet", days=36500)
    names = [e["name"] for e in detected]

    assert "Alice" not in names  # Excluded - still active
    assert "Bob" in names  # Included - detached, so shows up in detected
    assert "Charlie" in names  # Included - new entity


def test_detached_entity_preserves_all_fields(fixture_journal, tmp_path):
    """Test that detached entities preserve all fields including custom ones."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entity with custom fields and detached flag
    test_entities = [
        {
            "type": "Person",
            "name": "Alice",
            "description": "Test person",
            "attached_at": 1700000000000,
            "updated_at": 1700000001000,
            "aka": ["Ali", "AJ"],
            "tags": ["friend", "colleague"],
            "custom_field": "custom_value",
            "detached": True,
        },
    ]
    save_entities("test_facet", test_entities)

    # Load with include_detached to verify all fields preserved
    loaded = load_entities("test_facet", include_detached=True)
    assert len(loaded) == 1

    alice = loaded[0]
    assert alice["name"] == "Alice"
    assert alice["description"] == "Test person"
    assert alice["attached_at"] == 1700000000000
    assert alice["updated_at"] == 1700000001000
    assert alice["aka"] == ["Ali", "AJ"]
    assert alice["tags"] == ["friend", "colleague"]
    assert alice["custom_field"] == "custom_value"
    assert alice["detached"] is True


def test_detached_flag_for_detected_entities_not_filtered(fixture_journal, tmp_path):
    """Test that include_detached only affects attached entities, not detected."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create detected entity for a specific day
    detected_entities = [
        {"type": "Person", "name": "Alice", "description": "Detected person"},
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected entities - should always return all (no detached filtering for detected)
    loaded = load_entities("test_facet", "20250101")
    assert len(loaded) == 1

    # include_detached should have no effect on detected entities
    loaded_with_flag = load_entities("test_facet", "20250101", include_detached=True)
    assert len(loaded_with_flag) == 1


# Tests for entity folder utilities


def test_normalize_entity_name_basic():
    """Test basic name normalization."""
    assert normalize_entity_name("Alice Johnson") == "alice_johnson"
    assert normalize_entity_name("Acme Corp") == "acme_corp"
    assert normalize_entity_name("PostgreSQL") == "postgresql"


def test_normalize_entity_name_special_chars():
    """Test normalization of names with special characters."""
    assert normalize_entity_name("O'Brien") == "o_brien"
    assert normalize_entity_name("AT&T") == "at_t"
    assert normalize_entity_name("C++") == "c"


def test_normalize_entity_name_unicode():
    """Test normalization of unicode names."""
    assert normalize_entity_name("José García") == "jose_garcia"
    assert normalize_entity_name("Müller") == "muller"
    # Chinese characters are transliterated to pinyin by python-slugify
    assert normalize_entity_name("北京") == "bei_jing"


def test_normalize_entity_name_whitespace():
    """Test normalization handles various whitespace."""
    assert normalize_entity_name("  Spaced  Out  ") == "spaced_out"
    assert normalize_entity_name("Tab\tSeparated") == "tab_separated"
    assert normalize_entity_name("New\nLine") == "new_line"


def test_normalize_entity_name_empty():
    """Test normalization of empty/blank names."""
    assert normalize_entity_name("") == ""
    assert normalize_entity_name("   ") == ""
    assert normalize_entity_name(None) == ""  # type: ignore


def test_normalize_entity_name_long():
    """Test normalization of very long names."""
    long_name = "A" * 300
    normalized = normalize_entity_name(long_name)
    # Should be truncated with hash suffix
    assert len(normalized) <= 200
    assert "_" in normalized[-9:]  # Hash suffix pattern


def test_entity_folder_path(fixture_journal, tmp_path):
    """Test entity folder path generation."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    path = entity_folder_path("personal", "Alice Johnson")
    expected = tmp_path / "facets" / "personal" / "entities" / "alice_johnson"
    assert path == expected


def test_entity_folder_path_empty_name(fixture_journal, tmp_path):
    """Test entity folder path with empty name raises ValueError."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    with pytest.raises(ValueError, match="normalizes to empty string"):
        entity_folder_path("personal", "")


def test_ensure_entity_folder(fixture_journal, tmp_path):
    """Test entity folder creation."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    folder = ensure_entity_folder("personal", "Bob Smith")
    assert folder.exists()
    assert folder.is_dir()
    assert folder == tmp_path / "facets" / "personal" / "entities" / "bob_smith"


def test_ensure_entity_folder_idempotent(fixture_journal, tmp_path):
    """Test that ensure_entity_folder is idempotent."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    folder1 = ensure_entity_folder("personal", "Charlie Brown")
    folder2 = ensure_entity_folder("personal", "Charlie Brown")
    assert folder1 == folder2
    assert folder1.exists()


def test_rename_entity_folder(fixture_journal, tmp_path):
    """Test renaming entity folder."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create original folder
    old_folder = ensure_entity_folder("work", "Alice Johnson")
    assert old_folder.exists()

    # Create a file inside to verify contents are moved
    (old_folder / "notes.md").write_text("Test notes")

    # Rename
    result = rename_entity_folder("work", "Alice Johnson", "Alice Smith")
    assert result is True

    # Old folder should not exist
    assert not old_folder.exists()

    # New folder should exist with contents
    new_folder = tmp_path / "facets" / "work" / "entities" / "alice_smith"
    assert new_folder.exists()
    assert (new_folder / "notes.md").read_text() == "Test notes"


def test_rename_entity_folder_not_exists(fixture_journal, tmp_path):
    """Test renaming non-existent folder returns False."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    result = rename_entity_folder("work", "NonExistent", "NewName")
    assert result is False


def test_rename_entity_folder_same_normalized(fixture_journal, tmp_path):
    """Test renaming when normalized names are the same."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create folder
    ensure_entity_folder("work", "Alice Johnson")

    # Rename with different casing (normalizes to same)
    result = rename_entity_folder("work", "Alice Johnson", "alice johnson")
    assert result is False  # No rename needed


def test_rename_entity_folder_target_exists(fixture_journal, tmp_path):
    """Test renaming when target folder already exists raises OSError."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create both folders
    ensure_entity_folder("work", "Alice")
    ensure_entity_folder("work", "Bob")

    # Try to rename Alice to Bob
    with pytest.raises(OSError, match="already exists"):
        rename_entity_folder("work", "Alice", "Bob")
