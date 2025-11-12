from __future__ import annotations

from importlib import import_module

from flask import Flask

from . import (
    agents,
    calendar,
    facets,
    home,
    inbox,
    search,
    todos,
)

chat_view = import_module(".chat", __name__)


def register_views(app: Flask) -> None:
    for bp in [
        home.bp,
        facets.bp,
        inbox.bp,
        search.bp,
        calendar.bp,
        todos.bp,
        chat_view.bp,
        agents.bp,
    ]:
        app.register_blueprint(bp)
