from __future__ import annotations

import json
import os
import re
from typing import Any

import markdown  # type: ignore
from flask import Blueprint, jsonify, render_template, request

from convey import state
from convey.utils import format_date
from think.indexer import (
    search_events,
    search_insights,
    search_transcripts,
)

search_bp = Blueprint(
    "app:search",
    __name__,
    url_prefix="/app/search",
)


@search_bp.route("/api/insights")
def search_insights_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    from convey.utils import parse_pagination_params
    from think.utils import get_insights

    limit, offset = parse_pagination_params(default_limit=20)

    insights = get_insights()
    day = request.args.get("day")
    topic_filter = request.args.get("topic")
    total, rows = search_insights(query, limit, offset, day=day, topic=topic_filter)
    results = []
    for r in rows:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.startswith("insights/"):
            topic = topic[len("insights/") :]  # noqa: E203
        if topic.endswith(".md"):
            topic = topic[:-3]
        text = r["text"]
        words = text.split()
        if len(words) > 100:
            text = " ".join(words[:100]) + " ..."
        results.append(
            {
                "day": meta.get("day", ""),
                "date": format_date(meta.get("day", "")),
                "topic": topic,
                "color": insights.get(topic, {}).get("color"),
                "text": markdown.markdown(text, extensions=["extra"]),
                "score": r.get("score", 0.0),
            }
        )

    return jsonify({"total": total, "results": results})


@search_bp.route("/api/events")
def search_events_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    from convey.utils import parse_pagination_params
    from think.utils import get_insights

    limit, offset = parse_pagination_params(default_limit=10)

    insights = get_insights()
    day = request.args.get("day")
    topic_filter = request.args.get("topic")
    total, rows = search_events(query, limit, offset, day=day, topic=topic_filter)
    results = []
    for r in rows:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.startswith("insights/"):
            topic = topic[len("insights/") :]  # noqa: E203
        if topic.endswith(".md"):
            topic = topic[:-3]
        text = r.get("text", "")
        words = text.split()
        if len(words) > 100:
            text = " ".join(words[:100]) + " ..."
        start = meta.get("start", "")
        end = meta.get("end")
        length = 0
        if start and end:
            try:
                import datetime as _dt

                s = _dt.datetime.strptime(start, "%H:%M:%S")
                e = _dt.datetime.strptime(end, "%H:%M:%S")
                length = int((e - s).total_seconds() / 60)
            except Exception:
                length = 0
        results.append(
            {
                "day": meta.get("day", ""),
                "date": format_date(meta.get("day", "")),
                "topic": topic,
                "color": insights.get(topic, {}).get("color"),
                "start": start,
                "length": length,
                "text": markdown.markdown(text, extensions=["extra"]),
                "score": r.get("score", 0.0),
            }
        )

    return jsonify({"total": total, "results": results})


@search_bp.route("/api/transcripts")
def search_transcripts_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    from convey.utils import parse_pagination_params

    limit, offset = parse_pagination_params(default_limit=20)

    day = request.args.get("day")
    total, rows = search_transcripts(query, limit, offset, day=day)
    results = []
    for r in rows:
        meta = r.get("metadata", {})
        text = r.get("text", "")
        preview = re.sub(r"[^A-Za-z0-9]+", " ", text)
        preview = re.sub(r"\s+", " ", preview).strip()
        results.append(
            {
                "day": meta.get("day", ""),
                "date": format_date(meta.get("day", "")),
                "time": meta.get("time", ""),
                "type": meta.get("type", ""),
                "preview": preview,
            }
        )

    return jsonify({"total": total, "results": results})


@search_bp.route("/api/topic_detail")
def topic_detail() -> Any:
    path = request.args.get("path")
    if not path:
        return jsonify({}), 400
    full = os.path.join(state.journal_root, path)
    text = ""
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            pass
    return jsonify({"text": text})


@search_bp.route("/api/occurrence_detail")
def occurrence_detail() -> Any:
    """Return event details from a JSONL file by line index."""
    path = request.args.get("path")
    idx = int(request.args.get("index", 0))
    if not path:
        return jsonify({}), 400
    full = os.path.join(state.journal_root, path)
    data = {}
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line_num == idx:
                        data = json.loads(line.strip())
                        break
        except Exception:
            pass
    return jsonify(data)
