"""fred_macro series-universe loader.

Reads ``configs/data_pipelines/domains/fred_macro/series_<name>.yaml`` — a list
of FRED series with cadence metadata. Deliberately NOT named
``universe_*.yaml``: the equity universe shape
(``universe``/``listed_at``/``indices``/``tickers``, validated by
``tests/data_pipelines/test_universe_yaml_lint.py``) is OHLCV-specific and has
no slot for the per-series ``frequency`` this domain needs for calendar
selection. Keeping a distinct filename keeps that lint (and its spec doc)
untouched while giving macro series a shape that fits.

File shape (``series_macro.yaml``)::

    universe: macro
    series:
      - {id: DGS10,    frequency: daily,     category: rates,     description: ...}
      - {id: CPIAUCSL, frequency: monthly,   category: inflation, description: ...}
      - {id: GDPC1,    frequency: quarterly, category: growth,    description: ...}

- ``load_universe(name)``       → ``["FRED:DGS10", ...]``     (seed / agent surface)
- ``load_frequency_map(name)``  → ``{"DGS10": "daily", ...}`` (calendar selection)

Membership is a soft guardrail — ``fetch("FRED:<anything>")`` works for
out-of-universe series; the universe matters for bulk-seed and (via the
frequency map) for picking the right cadence calendar.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4]  # → repo root
    / "configs" / "data_pipelines" / "domains" / "fred_macro"
)

VALID_FREQUENCIES: frozenset[str] = frozenset({"daily", "monthly", "quarterly"})


def _series_path(name: str, config_root: Path | None = None) -> Path:
    root = config_root or DEFAULT_CONFIG_ROOT
    return root / f"series_{name}.yaml"


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> dict:
    return yaml.safe_load(Path(path_str).read_text()) or {}


def _load_series(name: str, config_root: Path | None = None) -> list[dict]:
    p = _series_path(name, config_root)
    if not p.is_file():
        raise FileNotFoundError(f"fred series file not found: {p}")
    data = _load_yaml(str(p))
    series = list(data.get("series", []))
    if not series:
        raise ValueError(f"{p}: 'series' is empty or missing")
    return series


def load_universe(name: str = "macro", config_root: Path | None = None) -> list[str]:
    """Return identifiers (e.g., ``'FRED:DGS10'``) for the named series set."""
    return [f"FRED:{str(s['id']).upper()}" for s in _load_series(name, config_root)]


def load_frequency_map(
    name: str = "macro", config_root: Path | None = None,
) -> dict[str, str]:
    """Map ``series_id`` (UPPER) → frequency (``daily``/``monthly``/``quarterly``)."""
    out: dict[str, str] = {}
    for s in _load_series(name, config_root):
        sid = str(s["id"]).upper()
        freq = s.get("frequency", "daily")
        if freq not in VALID_FREQUENCIES:
            raise ValueError(
                f"series {sid!r}: invalid frequency {freq!r}; "
                f"valid: {sorted(VALID_FREQUENCIES)}"
            )
        out[sid] = freq
    return out
