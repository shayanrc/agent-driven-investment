"""in_fundamentals universe — derived from the nse_equities universe YAMLs.

There is deliberately NO ``universe_*.yaml`` for this domain (the V3
us_fundamentals rule): the companies whose fundamentals matter are exactly
the equities the nse_equities domain tracks, so the universe is computed at
load time from those YAMLs, with the exchange prefix swapped:
``NSE:RELIANCE`` → ``INFUND:RELIANCE``. Index entries (``NIFTY:*`` /
``INDEX:*``) are excluded — indices have no financial statements.

``load_universe("nifty500")`` is the seed default (the modeling universe for
the gbdt nifty500 cells, ~500 tickers and a superset of nifty50/nifty100);
any single nse_equities universe name is accepted for partial seeds
(``seed --domain in_fundamentals --universe nifty50``).
"""

from __future__ import annotations

from data_pipelines.domains.nse_equities.universe import (
    load_universe as load_nse_equities_universe,
)

DEFAULT_UNIVERSE = "nifty500"

# Prefixes that mark index identifiers (no financial statements to fetch).
_INDEX_PREFIXES = ("INDEX", "NIFTY")


def load_universe(name: str = DEFAULT_UNIVERSE) -> list[str]:
    """Return ``INFUND:<TICKER>`` identifiers for the named nse_equities
    universe. Sorted, de-duplicated, indices excluded."""
    symbols: set[str] = set()
    for ident in load_nse_equities_universe(name):
        prefix, symbol = ident.split(":", 1)
        if prefix.upper() in _INDEX_PREFIXES:
            continue
        symbols.add(symbol.upper())
    return [f"INFUND:{s}" for s in sorted(symbols)]
