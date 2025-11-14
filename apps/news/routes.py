"""News app routes - facet newsletters."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, render_template, request

news_bp = Blueprint(
    "app:news",
    __name__,
    url_prefix="/app/news",
)


@news_bp.route("/")
def index() -> str:
    """News page with facet integration."""
    return render_template("app.html", app="news")


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
