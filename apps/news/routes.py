"""News app routes - facet newsletters."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from convey import state
from convey.utils import DATE_RE, adjacent_days, format_date

news_bp = Blueprint(
    "app:news",
    __name__,
    url_prefix="/app/news",
)


@news_bp.route("/")
def news_index() -> Any:
    """Redirect to today's news."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:news.news_day", day=today))


@news_bp.route("/<day>")
def news_day(day: str) -> str:
    """News view for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    prev_day, next_day = adjacent_days(state.journal_root, day)
    title = format_date(day)

    return render_template(
        "app.html",
        app="news",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
    )


@news_bp.route("/api/<facet_name>")
def get_news(facet_name: str) -> Any:
    """Get paginated news for a facet."""
    from think.facets import get_facet_news

    cursor = request.args.get("cursor")
    limit = request.args.get("days", default=5, type=int) or 5
    day = request.args.get("day")

    try:
        news_payload = get_facet_news(facet_name, cursor=cursor, limit=limit, day=day)
        return jsonify(news_payload)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
