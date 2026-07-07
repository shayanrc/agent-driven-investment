"""Helpers shared across the analog_mc dashboard views."""

from __future__ import annotations

from pathlib import Path


def list_configs(configs_root: Path) -> list[Path]:
    """YAML configs under ``configs_root``, sorted; empty if the dir is absent."""
    if not configs_root.exists():
        return []
    return sorted(configs_root.glob("*.yaml"))
