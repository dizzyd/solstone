"""Tests for navigation badge helpers in dream package."""

from __future__ import annotations

from typing import Any

import pytest

import dream.__init__ as dream_app


def test_count_pending_todos_today_counts_incomplete(monkeypatch):
    """Unfinished todos should increment the nav badge count."""

    calls: list[dict[str, Any]] = []

    def fake_get_todos(day: str, *, ensure_day: bool = False):
        calls.append({"day": day, "ensure_day": ensure_day})
        return [
            {"completed": False, "cancelled": False},
            {"completed": True, "cancelled": False},
            {"completed": False, "cancelled": True},
        ]

    monkeypatch.setattr(dream_app.todo_store, "get_todos", fake_get_todos)

    count = dream_app._count_pending_todos_today()

    assert count == 1
    assert calls and calls[0]["ensure_day"] is False


def test_resolve_nav_badges_filters_zero(monkeypatch):
    """Providers returning zero should be omitted."""

    def first_provider():
        return 5

    def second_provider():
        return 0

    monkeypatch.setattr(
        dream_app,
        "NAV_BADGE_PROVIDERS",
        {"one": first_provider, "two": second_provider},
    )

    badges = dream_app._resolve_nav_badges()

    assert badges == {"one": 5}
