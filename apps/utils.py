"""Utility functions for Convey app development."""

import os
import re

from flask import g


def get_app_storage_path(app_name, *sub_paths, ensure_exists=True):
    """
    Get path to app storage directory in journal.

    Args:
        app_name: App name (must match [a-z][a-z0-9_]*)
        *sub_paths: Optional subdirectory components
        ensure_exists: Create directory if it doesn't exist (default: True)

    Returns:
        Absolute path to <journal>/apps/<app_name>/<sub_paths>/

    Raises:
        ValueError: If app_name contains invalid characters

    Examples:
        get_app_storage_path("search")  # → "<journal>/apps/search/"
        get_app_storage_path("search", "cache")  # → "<journal>/apps/search/cache/"
    """
    # Validate app_name to prevent path traversal
    if not re.match(r"^[a-z][a-z0-9_]*$", app_name):
        raise ValueError(f"Invalid app name: {app_name}")

    # Get journal root from Flask state
    journal_root = g.state.journal_root

    # Build path
    path = os.path.join(journal_root, "apps", app_name, *sub_paths)

    if ensure_exists:
        os.makedirs(path, exist_ok=True)

    return path


def load_app_config(app_name, default=None):
    """
    Load app configuration from <journal>/apps/<app>/config.json.

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

    config_path = get_app_storage_path(app_name, "config.json", ensure_exists=False)
    return load_json(config_path) or default


def save_app_config(app_name, config):
    """
    Save app configuration to <journal>/apps/<app>/config.json.

    Args:
        app_name: App name
        config: Configuration dict to save

    Returns:
        True if successful, False otherwise
    """
    from convey.utils import save_json

    config_path = get_app_storage_path(app_name, "config.json")
    return save_json(config_path, config)
