"""us_equities universe loader.

Reads YAML universe files from configs/data_pipelines/domains/us_equities/.
v1 ships a seed S&P 500 list as `universe_sp500.yaml`. Point-in-time
constituent history is explicitly out of scope (open question 1).

Membership is a soft check: dispatch warns but does not reject out-of-universe
identifiers. The universe matters for bulk-seed and the agent-tool surface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4]  # → repo root
    / "configs" / "data_pipelines" / "domains" / "us_equities"
)


def _universe_path(name: str, config_root: Path | None = None) -> Path:
    root = config_root or DEFAULT_CONFIG_ROOT
    return root / f"universe_{name}.yaml"


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> dict:
    return yaml.safe_load(Path(path_str).read_text()) or {}


def load_universe(
    name: str = "sp500",
    config_root: Path | None = None,
) -> list[str]:
    """Return the list of identifiers (e.g., 'NYSE:AAPL') in the named universe.

    Combines `tickers:` (constituents) and `indices:` lists from the YAML.
    """
    p = _universe_path(name, config_root)
    if not p.is_file():
        raise FileNotFoundError(f"universe file not found: {p}")
    data = _load_yaml(str(p))
    tickers = list(data.get("tickers", []))
    indices = list(data.get("indices", []))
    return tickers + indices


def is_in_universe(
    identifier: str,
    name: str = "sp500",
    config_root: Path | None = None,
) -> bool:
    """True iff `identifier` is in the named universe."""
    return identifier in set(load_universe(name, config_root))
