from __future__ import annotations

from flask import Flask

from . import (
    agents,
    calendar,
    facets,
    home,
    search,
)


def register_views(app: Flask) -> None:
    for bp in [
        home.bp,
        facets.bp,
        search.bp,
        calendar.bp,
        agents.bp,
    ]:
        app.register_blueprint(bp)
