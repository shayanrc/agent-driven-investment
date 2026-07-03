"""us_fundamentals universe — derived from the us_equities universe YAMLs.

There is deliberately NO ``universe_*.yaml`` for this domain: the set of
companies whose fundamentals matter is exactly the set of equities the price
domains track, so the universe is computed at load time from the us_equities
configs (union of the requested equity universes, minus ``INDEX:*`` — indices
have no financial statements), with the exchange prefix dropped:
``NYSE:AAPL`` / ``NASDAQ:AAPL`` → ``FUND:AAPL``. One source of truth; adding a
ticker to an equity universe automatically adds its fundamentals coverage.

``load_universe("all")`` is the seed default (sp500 ∪ russell1000 ∪
nasdaq100, ~1,010 tickers); the individual equity universe names are accepted
for partial seeds (``seed --domain us_fundamentals --universe sp500``).
"""

from __future__ import annotations

from data_pipelines.domains.us_equities.universe import (
    load_universe as load_us_equities_universe,
)

# Equity universes that make up the "all" fundamentals universe.
EQUITY_UNIVERSES: tuple[str, ...] = ("sp500", "russell1000", "nasdaq100")


def load_universe(name: str = "all") -> list[str]:
    """Return ``FUND:<TICKER>`` identifiers for the named universe.

    ``name`` is ``"all"`` (union of ``EQUITY_UNIVERSES``) or any single
    us_equities universe name. Sorted, de-duplicated, indices excluded.
    """
    equity_names = EQUITY_UNIVERSES if name == "all" else (name,)
    symbols: set[str] = set()
    for eq_name in equity_names:
        for ident in load_us_equities_universe(eq_name):
            prefix, symbol = ident.split(":", 1)
            if prefix == "INDEX":
                continue
            symbols.add(symbol)
    return [f"FUND:{s}" for s in sorted(symbols)]
