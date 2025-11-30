"""Insights app - browse daily insight markdown files."""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

import markdown
from flask import Blueprint, redirect, render_template, url_for

from convey import state
from convey.utils import DATE_RE, adjacent_days, format_date, format_date_short
from think.utils import day_path, get_insight_topic, get_insights

insights_bp = Blueprint(
    "app:insights",
    __name__,
    url_prefix="/app/insights",
)


def _build_topic_map() -> dict[str, dict]:
    """Build a mapping from filesystem topic name to insight key and metadata.

    Returns dict mapping topic filename (e.g., "activity", "_chat_sentiment")
    to {"key": insight_key, "meta": insight_metadata}.
    """
    insights = get_insights()
    topic_map = {}
    for key, meta in insights.items():
        topic = get_insight_topic(key)
        topic_map[topic] = {"key": key, "meta": meta}
    return topic_map


def _format_label(key: str) -> str:
    """Format insight key as display label.

    "activity" -> "Activity"
    "chat:sentiment" -> "Chat: Sentiment"
    """
    if ":" in key:
        app, topic = key.split(":", 1)
        return f"{app.replace('_', ' ').title()}: {topic.replace('_', ' ').title()}"
    return key.replace("_", " ").title()


@insights_bp.route("/")
def index() -> Any:
    """Redirect to today's insights."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:insights.insights_day", day=today))


@insights_bp.route("/<day>")
def insights_day(day: str) -> str:
    """Render insights viewer for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    topic_map = _build_topic_map()
    files = []
    insights_dir = os.path.join(str(day_path(day)), "insights")

    if os.path.isdir(insights_dir):
        for name in sorted(os.listdir(insights_dir)):
            base, ext = os.path.splitext(name)
            if ext != ".md" or base not in topic_map:
                continue
            path = os.path.join(insights_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                html = markdown.markdown(text, extensions=["extra"])
            except Exception:
                continue

            info = topic_map[base]
            key = info["key"]
            meta = info["meta"]
            files.append(
                {
                    "label": _format_label(key),
                    "html": html,
                    "topic": base,
                    "key": key,
                    "source": meta.get("source", "system"),
                    "color": meta.get("color", "#6c757d"),
                }
            )

    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "app.html",
        app="insights",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
        day=day,
        day_formatted=format_date_short(day),
    )
