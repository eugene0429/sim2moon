"""Asset path resolution utilities.

Provides functions to resolve relative asset paths (e.g., 'assets/Ephemeris/de421.bsp')
to absolute paths regardless of the current working directory.

This is critical because Hydra changes CWD to outputs/{date}/{time}/ at runtime,
breaking any relative paths in the config.
"""

import os

# Project root = parent of the assets/ directory
_ASSETS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_ASSETS_DIR)


def get_assets_path() -> str:
    """Return the absolute path to the assets/ directory."""
    return _ASSETS_DIR


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> str:
    """Resolve a relative path against the project root.

    If the path is already absolute, it is returned as-is.
    Otherwise, it is joined with the project root.

    Args:
        relative_path: A path like 'assets/Ephemeris/de421.bsp'.

    Returns:
        Absolute path.
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(_PROJECT_ROOT, relative_path)


def resolve_asset_paths(cfg: dict) -> dict:
    """Walk a config dict and resolve known asset path fields to absolute paths.

    Recognized field names:
        ephemeris_path, profiles_path, texture_path, assets_root,
        dems_path, usd_path (only if starts with 'assets/')

    Args:
        cfg: Configuration dict (potentially nested).

    Returns:
        The same dict with asset paths resolved in-place.
    """
    _PATH_FIELDS = {
        "ephemeris_path", "profiles_path", "texture_path",
        "assets_root", "dems_path",
    }
    _CONDITIONAL_FIELDS = {"usd_path"}  # Only resolve if starts with "assets"

    def _walk(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, str) and key in _PATH_FIELDS:
                    d[key] = resolve_path(value)
                elif isinstance(value, str) and key in _CONDITIONAL_FIELDS:
                    if value.startswith("assets"):
                        d[key] = resolve_path(value)
                elif isinstance(value, (dict, list)) or hasattr(value, "__dataclass_fields__"):
                    _walk(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            _walk(item)
        # Also handle dataclass objects with __dict__
        elif hasattr(d, "__dict__") and hasattr(d, "__dataclass_fields__"):
            for field_name in d.__dataclass_fields__:
                value = getattr(d, field_name)
                if isinstance(value, str) and field_name in _PATH_FIELDS:
                    setattr(d, field_name, resolve_path(value))
                elif isinstance(value, str) and field_name in _CONDITIONAL_FIELDS:
                    if value.startswith("assets"):
                        setattr(d, field_name, resolve_path(value))
                elif isinstance(value, (dict, list)) or hasattr(value, "__dataclass_fields__"):
                    _walk(value)

    _walk(cfg)
    return cfg
