from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from think.domains import get_domains
from think.todo import get_todos

from .. import state
from ..utils import DATE_RE, adjacent_days, format_date

bp = Blueprint("todos", __name__, template_folder="../templates")


def _todo_path(day: str) -> Path:
    return Path(state.journal_root) / day / "todos" / "today.md"


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _modify_entry(
    path: Path,
    index: int,
    guard: str,
    transform: Callable[[str], str | None],
) -> bool:
    lines = _read_lines(path)
    tasks: list[tuple[int, int, str]] = []  # (entry_index, line_index, stripped)
    entry_index = 0

    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not stripped.startswith("- ["):
            continue
        entry_index += 1
        tasks.append((entry_index, line_index, stripped))

    target = next((item for item in tasks if item[0] == index), None)
    if target is None:
        return False

    _, line_index, stripped = target
    if guard and stripped != guard:
        return False

    new_value = transform(lines[line_index])
    if new_value is None:
        # Remove the line entirely
        del lines[line_index]
    else:
        lines[line_index] = new_value

    _write_lines(path, lines)
    return True


@bp.route("/todos")
def todos_page() -> str:
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("todos.todos_day", day=today))


@bp.route("/todos/<day>", methods=["GET", "POST"])
def todos_day(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    todo_path = _todo_path(day)

    if request.method == "POST":
        action = request.form.get("action")

        if action == "add":
            text = request.form.get("text", "").strip()
            if not text:
                flash("Cannot add an empty todo", "error")
            else:
                lines = _read_lines(todo_path)
                lines.append(f"- [ ] {text}")
                _write_lines(todo_path, lines)
            return redirect(url_for("todos.todos_day", day=day))

        index_str = request.form.get("index")
        guard = request.form.get("guard", "").strip()

        try:
            index = int(index_str) if index_str else None
        except ValueError:
            index = None

        if not index:
            flash("Missing todo index", "error")
            return redirect(url_for("todos.todos_day", day=day))

        if action == "complete":
            success = _modify_entry(
                todo_path,
                index,
                guard,
                lambda line: line.replace("- [ ]", "- [x]", 1),
            )
        elif action == "uncomplete":
            success = _modify_entry(
                todo_path,
                index,
                guard,
                lambda line: line.replace("- [x]", "- [ ]", 1),
            )
        elif action == "remove":
            success = _modify_entry(todo_path, index, guard, lambda line: None)
        else:
            flash("Unknown action", "error")
            return redirect(url_for("todos.todos_day", day=day))

        if not success:
            flash("Todo list changed, please refresh and try again", "error")

        return redirect(url_for("todos.todos_day", day=day))

    todos = get_todos(day) or []
    try:
        domain_map = get_domains()
    except Exception as exc:  # pragma: no cover - metadata is optional
        bp.logger.debug("Failed to load domain metadata: %s", exc)
        domain_map = {}
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "calendar_todos.html",
        active="todos",
        title=format_date(day),
        day=day,
        prev_day=prev_day,
        next_day=next_day,
        todos=todos,
        domain_map=domain_map,
    )


@bp.route("/todos/<day>/generate", methods=["POST"])
def generate_todos(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    from think.cortex_client import cortex_request

    day_date = datetime.strptime(day, "%Y%m%d")
    yesterday = (day_date - timedelta(days=1)).strftime("%Y%m%d")
    yesterday_path = _todo_path(yesterday)

    yesterday_content = ""
    if yesterday_path.exists():
        try:
            yesterday_content = yesterday_path.read_text(encoding="utf-8")
        except OSError:
            yesterday_content = ""

    prompt = f"""Generate a TODO checklist for {day_date.strftime('%Y-%m-%d')}.

Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Target day: {day_date.strftime('%Y-%m-%d')}
Target day folder: {day}

Yesterday's todos/today.md content:
{yesterday_content if yesterday_content else "(No todos recorded yesterday)"}

Write the generated checklist to {day}/todos/today.md"""

    try:
        active_file = cortex_request(
            prompt=prompt,
            persona="todo",
            backend="openai",
            config={},
        )
        agent_id = Path(active_file).stem.replace("_active", "")
    except Exception as exc:  # pragma: no cover - network/agent failure
        return jsonify({"error": f"Failed to spawn agent: {exc}"}), 500

    if not hasattr(state, "todo_generation_agents"):
        state.todo_generation_agents = {}
    state.todo_generation_agents[day] = agent_id

    return jsonify({"agent_id": agent_id, "status": "started"})


@bp.route("/todos/<day>/generation-status")
def todo_generation_status(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    agent_id = request.args.get("agent_id")
    if not agent_id and hasattr(state, "todo_generation_agents"):
        agent_id = state.todo_generation_agents.get(day)

    if not agent_id:
        return jsonify({"status": "none", "agent_id": None})

    from think.cortex_client import cortex_agents

    todo_path = _todo_path(day)

    agents_dir = Path(state.journal_root) / "agents"
    agent_file = agents_dir / f"{agent_id}.jsonl"

    if agent_file.exists():
        if todo_path.exists():
            if (
                hasattr(state, "todo_generation_agents")
                and day in state.todo_generation_agents
            ):
                del state.todo_generation_agents[day]
            return jsonify(
                {"status": "finished", "agent_id": agent_id, "todo_created": True}
            )
        return jsonify(
            {"status": "finished", "agent_id": agent_id, "todo_created": False}
        )

    try:
        response = cortex_agents(limit=100, offset=0)
        if response:
            agents = response.get("agents", [])
            for agent in agents:
                if agent.get("id") == agent_id:
                    return jsonify({"status": "running", "agent_id": agent_id})
            return jsonify({"status": "unknown", "agent_id": agent_id})
    except Exception:  # pragma: no cover - external call failure
        pass

    return jsonify({"status": "unknown", "agent_id": agent_id})
