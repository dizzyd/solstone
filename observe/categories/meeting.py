"""Formatter for meeting category content.

Renders meeting analysis JSON to rich markdown with participants and screen share.
"""

from typing import Any


def format(content: Any, context: dict) -> str:
    """Format meeting analysis to markdown.

    Args:
        content: Meeting analysis dict with platform, participants, screen_share
        context: Dict with frame, file_path, timestamp_str

    Returns:
        Formatted markdown string
    """
    if not isinstance(content, dict):
        return ""

    lines = []

    # Platform header
    platform = content.get("platform", "unknown")
    lines.append(f"**Meeting** ({platform})")
    lines.append("")

    # Participants
    participants = content.get("participants", [])
    if participants:
        lines.append("**Participants:**")
        for p in participants:
            # Handle both dict format (new) and string format (legacy)
            if isinstance(p, dict):
                name = p.get("name", "Unknown")
                status = p.get("status", "unknown")
                video = "📹" if p.get("video") else "🔇"
                lines.append(f"- {video} {name} ({status})")
            else:
                # Legacy: participant is just a name string
                lines.append(f"- {p}")
        lines.append("")

    # Screen share
    screen_share = content.get("screen_share")
    if screen_share:
        presenter = screen_share.get("presenter")
        description = screen_share.get("description", "")
        formatted_text = screen_share.get("formatted_text", "")

        presenter_str = f" by {presenter}" if presenter else ""
        lines.append(f"**Screen Share{presenter_str}:**")
        if description:
            lines.append(f"*{description}*")
        lines.append("")
        if formatted_text:
            lines.append(formatted_text.strip())
            lines.append("")

    return "\n".join(lines)
