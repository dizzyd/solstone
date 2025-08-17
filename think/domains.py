"""Domain-specific utilities and tooling for the think module."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


def domain_summary(domain: str) -> str:
    """Generate a nicely formatted markdown summary of a domain.

    Args:
        domain: The domain name to summarize

    Returns:
        Formatted markdown string with domain title, description, and entities

    Raises:
        FileNotFoundError: If the domain doesn't exist
        RuntimeError: If JOURNAL_PATH is not set
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal or journal == "":
        raise RuntimeError("JOURNAL_PATH not set")

    domain_path = Path(journal) / "domains" / domain
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain '{domain}' not found at {domain_path}")

    # Load domain metadata
    domain_json_path = domain_path / "domain.json"
    if not domain_json_path.exists():
        raise FileNotFoundError(f"domain.json not found for domain '{domain}'")

    with open(domain_json_path, "r", encoding="utf-8") as f:
        domain_data = json.load(f)

    # Extract metadata
    title = domain_data.get("title", domain)
    description = domain_data.get("description", "")
    emoji = domain_data.get("emoji", "")
    color = domain_data.get("color", "")

    # Build markdown summary
    lines = []

    # Title with emoji if available
    if emoji:
        lines.append(f"# {emoji} {title}")
    else:
        lines.append(f"# {title}")

    # Add color as a badge if available
    if color:
        lines.append(f"![Color]({color})")
        lines.append("")

    # Description
    if description:
        lines.append(f"**Description:** {description}")
        lines.append("")

    # Load entities if available
    entities_path = domain_path / "entities.md"
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities_content = f.read().strip()

        if entities_content:
            lines.append("## Entities")
            lines.append("")
            lines.append(entities_content)
            lines.append("")

    # Check for matters
    matters = []
    for item in domain_path.iterdir():
        if item.is_dir() and item.name.startswith("matter_"):
            matter_json = item / "matter.json"
            if matter_json.exists():
                with open(matter_json, "r", encoding="utf-8") as f:
                    matter_data = json.load(f)
                    matters.append(
                        {
                            "id": item.name,
                            "title": matter_data.get("title", item.name),
                            "status": matter_data.get("status", "unknown"),
                            "priority": matter_data.get("priority", "normal"),
                        }
                    )

    if matters:
        lines.append("## Matters")
        lines.append("")
        lines.append(f"**Total:** {len(matters)} matter(s)")
        lines.append("")

        # Group by status
        by_status = {}
        for matter in matters:
            status = matter["status"]
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(matter)

        for status in ["active", "pending", "completed", "archived"]:
            if status in by_status:
                lines.append(f"### {status.capitalize()} ({len(by_status[status])})")
                lines.append("")
                for matter in by_status[status]:
                    priority_marker = (
                        "🔴"
                        if matter["priority"] == "high"
                        else "🟡" if matter["priority"] == "medium" else ""
                    )
                    lines.append(
                        f"- {priority_marker} **{matter['id']}**: {matter['title']}"
                    )
                lines.append("")

    return "\n".join(lines)
