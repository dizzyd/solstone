"""Utility functions for Convey app storage in journal."""

import re
from pathlib import Path
from typing import Any

from flask import g

# Compiled pattern for app name validation
APP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def get_app_storage_path(
    app_name: str,
    *sub_dirs: str,
    ensure_exists: bool = True,
) -> Path:
    """
    Get path to app storage directory in journal.

    Args:
        app_name: App name (must match [a-z][a-z0-9_]*)
        *sub_dirs: Optional subdirectory components
        ensure_exists: Create directory if it doesn't exist (default: True)

    Returns:
        Path to <journal>/apps/<app_name>/<sub_dirs>/

    Raises:
        ValueError: If app_name contains invalid characters

    Examples:
        get_app_storage_path("search")  # → Path("<journal>/apps/search")
        get_app_storage_path("search", "cache")  # → Path("<journal>/apps/search/cache")
    """
    # Validate app_name to prevent path traversal
    if not APP_NAME_PATTERN.match(app_name):
        raise ValueError(f"Invalid app name: {app_name}")

    # Build path
    path = Path(g.state.journal_root) / "apps" / app_name
    for sub_dir in sub_dirs:
        path = path / sub_dir

    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)

    return path


def load_app_config(
    app_name: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Load app configuration from <journal>/apps/<app_name>/config.json.

    Args:
        app_name: App name
        default: Default value if config doesn't exist (default: None)

    Returns:
        Loaded JSON dict or default value if file doesn't exist

    Examples:
        config = load_app_config("my_app")  # Returns None if missing
        config = load_app_config("my_app", {})  # Returns {} if missing
    """
    from convey.utils import load_json

    storage_path = get_app_storage_path(app_name, ensure_exists=False)
    config_path = storage_path / "config.json"
    return load_json(config_path) or default


def save_app_config(
    app_name: str,
    config: dict[str, Any],
) -> bool:
    """
    Save app configuration to <journal>/apps/<app_name>/config.json.

    Args:
        app_name: App name
        config: Configuration dict to save

    Returns:
        True if successful, False otherwise
    """
    from convey.utils import save_json

    storage_path = get_app_storage_path(app_name, ensure_exists=True)
    config_path = storage_path / "config.json"
    return save_json(config_path, config)
