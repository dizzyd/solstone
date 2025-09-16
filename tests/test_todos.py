"""Tests for the simplified todos/today.md checklist parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from think.utils import get_todos


@pytest.fixture
def journal_root(tmp_path):
    path = tmp_path / "journal"
    path.mkdir()
    return path


def _write_todos(root: Path, day: str, lines: list[str]) -> Path:
    day_dir = root / day / "todos"
    day_dir.mkdir(parents=True)
    todo_path = day_dir / "today.md"
    todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return todo_path


def test_get_todos_returns_none_when_missing(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    assert get_todos("20240101") is None


def test_get_todos_parses_basic_fields(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "20240102",
        [
            "- [ ] **Review**: Merge analytics PR #think (10:30)",
            "- [x] **Meeting**: Project sync",  # no domain/time
            "- [ ] Write retrospective notes",
        ],
    )

    todos = get_todos("20240102")
    assert todos is not None
    assert len(todos) == 3

    first = todos[0]
    assert first["index"] == 1
    assert first["type"] == "Review"
    assert first["domain"] == "think"
    assert first["time"] == "10:30"
    assert first["completed"] is False
    assert first["text"] == "Merge analytics PR"

    second = todos[1]
    assert second["completed"] is True
    assert second["domain"] is None
    assert second["time"] is None
    assert second["text"] == "Project sync"

    third = todos[2]
    assert third["text"] == "Write retrospective notes"
    assert third["index"] == 3


def test_get_todos_handles_strikethrough_and_spacing(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "20240103",
        [
            "- [ ] ~~**Task**: Optional experiment if time allows~~",
            "  - [ ]  **Fix**: Address bug #backend  (14:45)  ",
            "not a todo line",
            "- [x] **Research**: Draft report #think",
        ],
    )

    todos = get_todos("20240103")
    assert todos is not None
    assert len(todos) == 3

    cancelled = todos[0]
    assert cancelled["cancelled"] is True
    assert cancelled["type"] == "Task"
    assert cancelled["text"] == "Optional experiment if time allows"

    second = todos[1]
    assert second["type"] == "Fix"
    assert second["domain"] == "backend"
    assert second["time"] == "14:45"
    assert second["index"] == 2

    third = todos[2]
    assert third["completed"] is True
    assert third["text"] == "Draft report"


def test_get_todos_ignores_blank_lines(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "20240104",
        [
            "",
            "- [ ] First",
            "# comment line",
            "- [ ] Second",
        ],
    )

    todos = get_todos("20240104")
    assert [item["index"] for item in todos] == [1, 2]
