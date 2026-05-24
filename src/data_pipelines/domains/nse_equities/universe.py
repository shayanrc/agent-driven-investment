"""nse_equities universe loader.

Reads YAML universe files from configs/data_pipelines/domains/nse_equities/.
v1.7 ships Nifty 50 as `universe_nifty50.yaml`. Other Nifty universes
(Next 50, Midcap 150, Smallcap 250) are deferred to v1.7.1 per open question
8 of the V1 plan.

Membership is a soft check: dispatch warns but does not reject out-of-universe
identifiers. The universe matters for bulk-seed and the agent-tool surface.

The seed list can be refreshed at any time by calling
`nselib.capital_market.nifty50_equity_list()` and re-emitting the YAML; that
function is what was used to generate the committed `universe_nifty50.yaml`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4]  # → repo root
    / "configs" / "data_pipelines" / "domains" / "nse_equities"
)


def _universe_path(name: str, config_root: Path | None = None) -> Path:
    root = config_root or DEFAULT_CONFIG_ROOT
    return root / f"universe_{name}.yaml"


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> dict:
    return yaml.safe_load(Path(path_str).read_text()) or {}


def load_universe(
    name: str = "nifty50",
    config_root: Path | None = None,
) -> list[str]:
    """Return the list of identifiers in the named universe.

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
    name: str = "nifty50",
    config_root: Path | None = None,
) -> bool:
    return identifier in set(load_universe(name, config_root))
