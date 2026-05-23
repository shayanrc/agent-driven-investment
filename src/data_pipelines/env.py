"""Eager .env loading for the data_pipelines module.

Called once at package import (from data_pipelines/__init__.py). Searches
upward from CWD for a .env file and loads it into os.environ. Existing env
vars are NOT overridden — values set in the real shell / CI take precedence
over the file, which is what you want for production overrides.

D6 — API keys never leak: load_dotenv() doesn't log values, and the
fail-soft behavior (no .env file → silently continue) means we never echo
the search path or contents anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_LOADED: bool = False


def load_env(override: bool = False) -> Path | None:
    """Load .env once per process. Returns the path loaded, or None if no
    .env was found. Subsequent calls are no-ops unless override=True.
    """
    global _LOADED
    if _LOADED and not override:
        return None

    path = find_dotenv(usecwd=True)
    if not path:
        # Also try the repo root relative to this file (resilient to CWD).
        candidate = Path(__file__).resolve().parents[2] / ".env"
        if candidate.is_file():
            path = str(candidate)

    if not path:
        _LOADED = True
        return None

    load_dotenv(dotenv_path=path, override=override)
    _LOADED = True
    return Path(path)


def get_required_env(name: str) -> str:
    """Read a required env var. Raises KeyError if unset/empty (consumers
    that want the typed MissingAPIKey path should call MissingAPIKey
    directly rather than wrapping this).
    """
    val = os.environ.get(name)
    if not val:
        raise KeyError(name)
    return val
