from __future__ import annotations

import json
import os

from flask import Blueprint, jsonify, render_template

from .. import state
from ..utils import time_since

bp = Blueprint("agents", __name__, template_folder="../templates")


@bp.route("/agents")
def agents_page() -> str:
    """Render the Agents view."""
    return render_template("agents.html", active="agents")


def _agents_dir() -> str:
    if not state.journal_root:
        return ""
    path = os.path.join(state.journal_root, "agents")
    os.makedirs(path, exist_ok=True)
    return path


@bp.route("/agents/api/available")
def available_agents() -> object:
    """Return list of available agent definitions from think/agents/."""
    agents_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "think", "agents")
    items: list[dict[str, object]] = []
    
    if os.path.isdir(agents_path):
        # Find all .json files and match with corresponding .txt files
        json_files = [f for f in os.listdir(agents_path) if f.endswith('.json')]
        
        for json_file in json_files:
            base_name = json_file[:-5]  # Remove .json extension
            txt_file = base_name + '.txt'
            
            json_path = os.path.join(agents_path, json_file)
            txt_path = os.path.join(agents_path, txt_file)
            
            # Skip if no corresponding .txt file
            if not os.path.isfile(txt_path):
                continue
                
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    agent_config = json.load(f)
                
                items.append({
                    "id": base_name,
                    "title": agent_config.get("title", base_name),
                    "description": agent_config.get("description", ""),
                })
            except Exception:
                continue
    
    items.sort(key=lambda x: x.get("title", ""))
    return jsonify(items)


@bp.route("/agents/api/content/<agent_id>")
def agent_content(agent_id: str) -> object:
    """Return the .txt content for an agent."""
    agents_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "think", "agents")
    txt_path = os.path.join(agents_path, f"{agent_id}.txt")
    
    if not os.path.isfile(txt_path):
        return jsonify({"error": "Agent not found"}), 404
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/agents/api/list")
def agents_list() -> object:
    path = _agents_dir()
    items: list[dict[str, object]] = []
    if path and os.path.isdir(path):
        for name in os.listdir(path):
            if not name.endswith(".jsonl"):
                continue
            full = os.path.join(path, name)
            start_ms = 0
            try:
                start_ms = int(os.path.splitext(name)[0])
            except ValueError:
                try:
                    start_ms = int(os.stat(full).st_mtime * 1000)
                except Exception:
                    start_ms = 0
            start = start_ms / 1000
            prompt = ""
            persona = ""
            model = ""
            try:
                with open(full, "r", encoding="utf-8") as f:
                    for line in f:
                        j = json.loads(line)
                        if j.get("event") == "start":
                            prompt = j.get("prompt", "")
                            persona = j.get("persona", "")
                            model = j.get("model", "")
                            break
            except Exception:
                continue
            items.append(
                {
                    "id": os.path.splitext(name)[0],
                    "start": start,
                    "since": time_since(start),
                    "model": model,
                    "persona": persona,
                    "prompt": prompt,
                }
            )
    items.sort(key=lambda x: float(x.get("start", 0)), reverse=True)
    return jsonify(items)
