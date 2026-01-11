# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Insights app - browse daily insight markdown files."""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

import markdown
from flask import Blueprint, jsonify, redirect, render_template, url_for

from convey.utils import DATE_RE, format_date
from think.utils import day_dirs, day_path, get_insight_topic, get_insights

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


def _parse_insight_filename(filename: str) -> tuple[str, str | None]:
    """Parse insight filename to extract topic and optional variant.

    Examples:
        "activity.md" -> ("activity", None)
        "activity@digitalocean.md" -> ("activity", "digitalocean")
        "_chat_sentiment.md" -> ("_chat_sentiment", None)
        "_chat_sentiment@openai.md" -> ("_chat_sentiment", "openai")

    Returns:
        (topic, variant) tuple where variant is None for base files.
    """
    base, ext = os.path.splitext(filename)
    if ext != ".md":
        return ("", None)

    if "@" in base:
        topic, variant = base.rsplit("@", 1)
        return (topic, variant)
    return (base, None)


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


def _read_insight_html(filepath: str) -> str | None:
    """Read insight file and convert to HTML.

    Returns None if file cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return markdown.markdown(text, extensions=["extra"])
    except Exception:
        return None


@insights_bp.route("/<day>")
def insights_day(day: str) -> str:
    """Render insights viewer for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    topic_map = _build_topic_map()
    insights_dir = os.path.join(str(day_path(day)), "insights")

    # Group insights by topic, collecting base file and variants
    topic_groups: dict[str, dict] = {}

    if os.path.isdir(insights_dir):
        for name in sorted(os.listdir(insights_dir)):
            topic, variant = _parse_insight_filename(name)
            if not topic or topic not in topic_map:
                continue

            path = os.path.join(insights_dir, name)
            html = _read_insight_html(path)
            if html is None:
                continue

            # Initialize topic group if not exists
            if topic not in topic_groups:
                info = topic_map[topic]
                topic_groups[topic] = {
                    "label": _format_label(info["key"]),
                    "topic": topic,
                    "key": info["key"],
                    "source": info["meta"].get("source", "system"),
                    "color": info["meta"].get("color", "#6c757d"),
                    "base": None,
                    "variants": [],
                }

            # Add as base or variant
            if variant is None:
                topic_groups[topic]["base"] = {
                    "name": "default",
                    "html": html,
                }
            else:
                topic_groups[topic]["variants"].append(
                    {
                        "name": variant,
                        "html": html,
                    }
                )

    # Build files list, including only topics with at least a base
    files = []
    for topic in sorted(topic_groups.keys()):
        group = topic_groups[topic]
        # Include if has base (variants alone don't show without base)
        if group["base"]:
            # For backward compat, set html to base content
            group["html"] = group["base"]["html"]
            files.append(group)

    title = format_date(day)

    return render_template(
        "app.html",
        title=title,
        files=files,
    )


@insights_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return insight counts for each day in a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to insight file count.
        Insights app is not facet-aware, so returns simple {day: count} mapping.
    """
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    stats: dict[str, int] = {}

    for day_name, day_dir in day_dirs().items():
        # Filter to only days in requested month
        if not day_name.startswith(month):
            continue

        insights_dir = os.path.join(day_dir, "insights")
        if os.path.isdir(insights_dir):
            # Count .md files
            md_files = [f for f in os.listdir(insights_dir) if f.endswith(".md")]
            if md_files:
                stats[day_name] = len(md_files)

    return jsonify(stats)
