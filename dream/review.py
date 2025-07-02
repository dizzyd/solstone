"""Combined web service for dream review apps."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from google import genai
from google.genai import types

from think.cluster_glob import FLASH_MODEL
from think.indexer import get_entities, parse_entity_line

DATE_RE = re.compile(r"\d{8}")


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to 'Wednesday April 2nd' format."""
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        day = date_obj.day
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return date_obj.strftime(f"%A %B {day}{suffix}")
    except ValueError:
        return date_str


def log_entity_operation(
    log_dir: str, operation: str, day: str, etype: str, name: str, new_name: Optional[str] = None
) -> None:
    """Log entity operations to entity_review.log."""
    log_path = os.path.join(log_dir, "entity_review.log")
    timestamp = datetime.now().isoformat()
    if new_name:
        log_entry = f"{timestamp} {operation} {day} {etype}: {name} -> {new_name}\n"
    else:
        log_entry = f"{timestamp} {operation} {day} {etype}: {name}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)


def modify_entity_in_file(
    file_path: str,
    etype: str,
    name: str,
    new_name: Optional[str] = None,
    operation: str = "remove",
    require_match: bool = True,
) -> bool:
    """Remove or rename an entity entry in an entities.md file."""
    if not os.path.isfile(file_path):
        if require_match:
            raise ValueError(f"entities.md not found at {file_path}")
        return False
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    matches: List[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        parsed = parse_entity_line(line)
        if not parsed:
            continue
        t, n, desc = parsed
        if t == etype and n == name:
            matches.append((idx, desc))
    if len(matches) == 0:
        if require_match:
            raise ValueError(f"No match found for '{etype}: {name}' in {file_path}")
        return False
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for '{etype}: {name}' in {file_path}")
    idx, desc = matches[0]
    newline = "\n" if lines[idx].endswith("\n") else ""
    if operation == "remove":
        del lines[idx]
    elif operation == "rename" and new_name:
        new_line = f"* {etype}: {new_name}"
        if desc:
            new_line += f" - {desc}"
        lines[idx] = new_line + newline
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


def modify_entity_file(
    journal: str,
    day: str,
    etype: str,
    name: str,
    new_name: Optional[str] = None,
    operation: str = "remove",
) -> None:
    """Remove or rename an entity entry in a day's ``entities.md`` file."""
    file_path = os.path.join(journal, day, "entities.md")
    modify_entity_in_file(file_path, etype, name, new_name, operation, require_match=True)
    log_entity_operation(journal, operation, day, etype, name, new_name)


def update_top_entry(journal: str, etype: str, name: str, desc: str) -> None:
    """Add or update an entry in the top entities.md file."""
    desc = desc.replace("\n", " ").replace("\r", " ").strip()
    file_path = os.path.join(journal, "entities.md")
    lines: List[str] = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    found = False
    for idx, line in enumerate(lines):
        parsed = parse_entity_line(line)
        if not parsed:
            continue
        t, n, _ = parsed
        if t == etype and n == name:
            newline = "\n" if line.endswith("\n") else ""
            lines[idx] = f"* {etype}: {name} - {desc}" + newline
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"* {etype}: {name} - {desc}\n")
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_top_summary(info: Dict[str, Any], api_key: str) -> str:
    """Merge entity descriptions into a single summary via Gemini."""
    descs = list(info.get("descriptions", {}).values())
    if not descs and info.get("primary"):
        descs.append(info["primary"])
    joined = "\n".join(f"- {d}" for d in descs if d)
    prompt = (
        "Merge the following entity descriptions into one concise summary about"
        "the same length as any individual line. Only return the final merged summary text."
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[joined],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192 * 2,
            system_instruction=prompt,
        ),
    )
    return response.text


def build_index(journal: str) -> Dict[str, List[Dict[str, Any]]]:
    """Create a mapping of YYYYMMDD folders to meeting lists."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if not os.path.isdir(path):
                continue
            file_path = os.path.join(path, "ponder_meetings.json")
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    index[name] = data
            except Exception:
                continue
    return index


app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "review", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "review", "static"),
)
app.secret_key = os.getenv("DREAM_SECRET", "sunstone-secret")
app.config["PASSWORD"] = ""

journal_root = ""
entities_index: Dict[str, Dict[str, dict]] = {}
meetings_index: Dict[str, List[Dict[str, Any]]] = {}


def reload_entities() -> None:
    global entities_index
    entities_index = get_entities(journal_root)


@app.before_request
def require_login() -> Any:
    if request.endpoint in {"login", "static"}:
        return None
    if not session.get("logged_in"):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    error = None
    if request.method == "POST":
        if request.form.get("password") == app.config.get("PASSWORD"):
            session["logged_in"] = True
            return redirect(url_for("home"))
        error = "Invalid password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout() -> Any:
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/")
def home() -> str:
    summary_path = os.path.join(journal_root, "summary.md")
    summary_html = ""
    if os.path.isfile(summary_path):
        try:
            import markdown  # type: ignore

            with open(summary_path, "r", encoding="utf-8") as f:
                summary_html = markdown.markdown(f.read())
        except Exception:
            summary_html = "<p>Error loading summary.</p>"
    return render_template("home.html", active="home", summary_html=summary_html)


@app.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@app.route("/calendar")
def calendar() -> str:
    return render_template("calendar.html", active="calendar")


@app.route("/calendar/<day>")
def calendar_day(day: str) -> str:
    if not re.fullmatch(r"\d{8}", day):
        return "", 404
    day_dir = os.path.join(journal_root, day)
    if not os.path.isdir(day_dir):
        return "", 404
    files = []
    for name in sorted(os.listdir(day_dir)):
        if name.startswith("ponder_") and name.endswith(".md"):
            path = os.path.join(day_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            try:
                import markdown  # type: ignore

                html = markdown.markdown(text)
            except Exception:
                html = "<p>Error loading file.</p>"
            label = name[7:-3].replace("_", " ").title()
            files.append({"label": label, "html": html})
    title = format_date(day)
    days = sorted(d for d in os.listdir(journal_root) if re.fullmatch(r"\d{8}", d))
    prev_day = next_day = None
    if day in days:
        idx = days.index(day)
        if idx > 0:
            prev_day = days[idx - 1]
        if idx < len(days) - 1:
            next_day = days[idx + 1]
    return render_template(
        "day.html",
        active="calendar",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
    )


@app.route("/entities/api/data")
def entities_data() -> Any:
    data: Dict[str, List[Dict[str, object]]] = {}
    for etype, names in entities_index.items():
        data[etype] = []
        for name, info in names.items():
            formatted_descriptions = {
                format_date(date): text for date, text in info.get("descriptions", {}).items()
            }
            data[etype].append(
                {
                    "name": name,
                    "dates": [format_date(d) for d in sorted(info.get("dates", []))],
                    "raw_dates": sorted(info.get("dates", [])),
                    "desc": info.get("primary", ""),
                    "top": info.get("top", False),
                    "descriptions": formatted_descriptions,
                }
            )
    return jsonify(data)


@app.route("/entities/api/top_generate", methods=["POST"])
def api_top_generate() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    info = entities_index.get(etype, {}).get(name)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or info is None:
        return ("", 400)
    try:
        desc = generate_top_summary(info, api_key)
        return jsonify({"desc": desc})
    except Exception as e:  # pragma: no cover - network errors
        return jsonify({"error": str(e)}), 500


@app.route("/entities/api/top_update", methods=["POST"])
def api_top_update() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    desc = (payload.get("desc") or "").replace("\n", " ").replace("\r", " ").strip()
    update_top_entry(journal_root, etype, name, desc)
    reload_entities()
    return jsonify({"status": "ok"})


@app.route("/entities/api/remove", methods=["POST"])
@app.route("/entities/api/rename", methods=["POST"])
def api_modify_entity() -> Any:
    payload = request.get_json(force=True)
    action = "remove" if request.path.endswith("remove") else "rename"
    days = payload.get("days", [])
    etype = payload.get("type")
    name = payload.get("name")
    new_name = payload.get("new_name") if action == "rename" else None
    for day in days:
        modify_entity_file(journal_root, day, etype, name, new_name, action)
    if action == "rename" and new_name:
        top_file = os.path.join(journal_root, "entities.md")
        try:
            modify_entity_in_file(top_file, etype, name, new_name, "rename", require_match=False)
        except Exception:
            pass
    reload_entities()
    return jsonify({"status": "ok"})


@app.route("/calendar/api/meetings")
def calendar_meetings() -> Any:
    return jsonify(meetings_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined review web service")
    parser.add_argument("journal", help="Journal directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--password",
        help="Password required for login (can also set DREAM_PASSWORD)",
        default=os.getenv("DREAM_PASSWORD"),
    )
    args = parser.parse_args()

    global journal_root, meetings_index
    journal_root = args.journal
    app.config["PASSWORD"] = args.password
    reload_entities()
    meetings_index = build_index(journal_root)

    if not app.config["PASSWORD"]:
        raise ValueError("Password must be provided via --password or DREAM_PASSWORD")

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
