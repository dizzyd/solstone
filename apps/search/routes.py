from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from flask import Blueprint, jsonify, request

from convey.utils import format_date
from think.facets import get_facets
from think.indexer.journal import search_journal

search_bp = Blueprint(
    "app:search",
    __name__,
    url_prefix="/app/search",
)

# Topic icons for display
TOPIC_ICONS = {
    "flow": "📝",
    "knowledge_graph": "🗺️",
    "meetings": "📅",
    "event": "📅",
    "audio": "🎤",
    "screen": "🖥️",
    "todo": "✅",
    "entity": "👤",
    "entity:attached": "👤",
    "entity:detected": "👤",
    "news": "📰",
    "import": "📥",
}

# Topic display names
TOPIC_LABELS = {
    "flow": "Flow",
    "knowledge_graph": "Knowledge Graph",
    "meetings": "Meetings",
    "event": "Event",
    "audio": "Transcript",
    "screen": "Screen",
    "todo": "Todo",
    "entity": "Entity",
    "entity:attached": "Entity",
    "entity:detected": "Entity",
    "news": "News",
    "import": "Import",
}


def _parse_facet_filter() -> str | None:
    """Parse facet filter from request args."""
    return request.args.get("facet", "").strip() or None


def _parse_topic_filters() -> list[str] | None:
    """Parse topic filters from request args (comma-separated)."""
    topics_param = request.args.get("topics", "").strip()
    filters = [t.strip() for t in topics_param.split(",") if t.strip()]
    return filters or None


def _highlight_query_terms(text: str, query: str) -> str:
    """Add bold highlighting around query terms in text."""
    if not query:
        return text
    # Extract individual words from query (ignore FTS operators)
    terms = [t for t in query.split() if t.upper() not in ("AND", "OR", "NOT")]
    for term in terms:
        # Remove quotes and wildcards for matching
        clean_term = term.strip('"*')
        if len(clean_term) >= 2:
            # Case-insensitive replacement with bold
            pattern = re.compile(re.escape(clean_term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<strong>{m.group()}</strong>", text)
    return text


def _format_result(result: dict, query: str, facets_map: dict) -> dict:
    """Format a search result for API response."""
    meta = result.get("metadata", {})
    topic = meta.get("topic", "")
    text = result.get("text", "")
    facet_name = meta.get("facet", "")

    # Get facet metadata
    facet_info = facets_map.get(facet_name, {})

    # Clean and truncate text
    if topic in ("audio", "screen"):
        # Transcript-style: clean special chars for preview
        preview = re.sub(r"[^A-Za-z0-9\s.,!?'-]", " ", text)
        preview = re.sub(r"\s+", " ", preview).strip()
        display_text = preview
    else:
        # Insight/event-style: keep markdown structure
        words = text.split()
        if len(words) > 50:
            display_text = " ".join(words[:50]) + "..."
        else:
            display_text = text

    # Apply highlighting
    display_text = _highlight_query_terms(display_text, query)

    return {
        "id": result.get("id", ""),
        "day": meta.get("day", ""),
        "topic": topic,
        "topic_icon": TOPIC_ICONS.get(topic, "📄"),
        "topic_label": TOPIC_LABELS.get(topic, topic.title()),
        "facet": facet_name,
        "facet_title": facet_info.get("title", facet_name),
        "facet_color": facet_info.get("color", ""),
        "facet_emoji": facet_info.get("emoji", ""),
        "text": display_text,
        "path": meta.get("path", ""),
        "idx": meta.get("idx", 0),
        "score": result.get("score", 0.0),
    }


@search_bp.route("/api/search")
def search_journal_api() -> Any:
    """Unified journal search endpoint with day grouping.

    Query parameters:
        q: Search query (required)
        limit: Max results per day (default 5)
        offset: Day offset for pagination (default 0)
        facet: Filter by facet name (optional)
        topics: Comma-separated topic filter (optional)

    Returns:
        JSON with:
        - total: Total match count
        - days: List of day groups, each with date info and results
        - facets: List of facets with counts for filter sidebar
        - topics: List of topics with counts for filter sidebar
    """
    query = request.args.get("q", "").strip()

    # Parse parameters
    results_per_day = int(request.args.get("limit", 5))
    day_offset = int(request.args.get("offset", 0))
    facet_filter = _parse_facet_filter()
    topic_filters = _parse_topic_filters()

    # Load facet metadata for enriching results
    facets_map = get_facets()

    # First, get all results to compute aggregations
    # We fetch more to enable grouping and counts
    total, all_rows = search_journal(
        query,
        limit=500,  # Reasonable cap for aggregation
        offset=0,
        facet=facet_filter,
        topic=None,  # We filter topics client-side for multi-select
    )

    # Compute facet/topic counts BEFORE filtering (for sidebar)
    facet_counts: dict[str, int] = defaultdict(int)
    topic_counts: dict[str, int] = defaultdict(int)
    for r in all_rows:
        meta = r.get("metadata", {})
        f = meta.get("facet", "")
        t = meta.get("topic", "")
        if f:
            facet_counts[f] += 1
        if t:
            topic_counts[t] += 1

    # Filter by topics if specified (after computing counts)
    if topic_filters:
        all_rows = [
            r for r in all_rows if r.get("metadata", {}).get("topic") in topic_filters
        ]

    # Group results by day
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        day = r.get("metadata", {}).get("day", "")
        if day:
            by_day[day].append(r)

    # Sort days descending (most recent first)
    sorted_days = sorted(by_day.keys(), reverse=True)

    # Apply day pagination
    paginated_days = sorted_days[day_offset : day_offset + 20]

    # Build day groups with capped results
    days_response = []
    for day in paginated_days:
        day_results = by_day[day]
        total_in_day = len(day_results)

        # Cap results per day
        capped_results = day_results[:results_per_day]

        formatted_results = [
            _format_result(r, query, facets_map) for r in capped_results
        ]

        days_response.append(
            {
                "day": day,
                "date": format_date(day),
                "total": total_in_day,
                "showing": len(formatted_results),
                "has_more": total_in_day > results_per_day,
                "results": formatted_results,
            }
        )

    # Build facet list for sidebar with counts
    facets_list = []
    for name, data in facets_map.items():
        if data.get("muted"):
            continue
        facets_list.append(
            {
                "name": name,
                "title": data.get("title", name),
                "color": data.get("color", ""),
                "emoji": data.get("emoji", ""),
                "count": facet_counts.get(name, 0),
            }
        )
    # Sort by count descending
    facets_list.sort(key=lambda x: x["count"], reverse=True)

    # Build topic list for sidebar
    topics_list = []
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        topics_list.append(
            {
                "name": topic,
                "label": TOPIC_LABELS.get(topic, topic.title()),
                "icon": TOPIC_ICONS.get(topic, "📄"),
                "count": count,
            }
        )

    return jsonify(
        {
            "total": len(all_rows),
            "total_days": len(sorted_days),
            "showing_days": len(days_response),
            "days": days_response,
            "facets": facets_list,
            "topics": topics_list,
        }
    )


@search_bp.route("/api/day_results")
def day_results_api() -> Any:
    """Get more results for a specific day.

    Query parameters:
        q: Search query
        day: Day to get results for (YYYYMMDD)
        offset: Result offset within the day (default 0)
        limit: Max results (default 20)
        facet: Facet filter (optional)
        topics: Comma-separated topic filter (optional)
    """
    query = request.args.get("q", "").strip()
    day = request.args.get("day", "").strip()
    if not day:
        return jsonify({"results": [], "total": 0})

    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 20))
    facet_filter = _parse_facet_filter()
    topic_filters = _parse_topic_filters()

    facets_map = get_facets()

    # Search within the specific day
    total, rows = search_journal(
        query,
        limit=200,  # Get all for this day
        offset=0,
        day=day,
        facet=facet_filter,
    )

    # Filter by topics
    if topic_filters:
        rows = [r for r in rows if r.get("metadata", {}).get("topic") in topic_filters]

    total_in_day = len(rows)
    paginated = rows[offset : offset + limit]

    formatted = [_format_result(r, query, facets_map) for r in paginated]

    return jsonify(
        {
            "day": day,
            "total": total_in_day,
            "offset": offset,
            "results": formatted,
        }
    )
