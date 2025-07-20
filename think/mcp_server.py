#!/usr/bin/env python3
"""MCP server for Sunstone journal assistant."""

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from think.indexer import search_occurrences as search_occurrences_impl
from think.indexer import search_topics as search_topics_impl


# Create the MCP server
mcp = FastMCP("sunstone")


@mcp.tool
def search_topic(query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
    """Full-text search over topic summaries."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    total, results = search_topics_impl(journal, query, limit, offset)
    
    items = []
    for r in results:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.endswith(".md"):
            topic = topic[:-3]
        items.append({
            "day": meta.get("day", ""),
            "filename": topic,
            "text": r.get("text", "")
        })
    
    return {"total": total, "results": items}


@mcp.tool
def search_occurrence(query: str) -> str:
    """Search structured occurrences by keyword."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    results = search_occurrences_impl(journal, query, 5)
    
    lines = []
    for r in results:
        meta = r.get("metadata", {})
        lines.append(f"{meta.get('day')} {meta.get('type')}: {r['text']}")
    
    return "\n".join(lines)


@mcp.tool
def read_markdown(date: str, filename: str) -> str:
    """Return journal markdown contents."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / date / f"{filename}.md"
    
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    
    return md_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    # When run directly, use stdio transport (default)
    mcp.run()