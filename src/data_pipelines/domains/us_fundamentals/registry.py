"""us_fundamentals identifier parser.

Accepted form: ``FUND:<TICKER>`` (e.g., ``FUND:AAPL``, ``FUND:BRK-B``).

Tickers use the same dash-normalized symbols as the us_equities universes
(``BRK-B``, not ``BRK.B``) — the universe is derived from those YAMLs, and
per-provider spelling quirks (macrotrends slugs, EDGAR CIK lookup) are the
adapters' concern. Fundamentals have no exchange dimension (a company's 10-Q
is the same wherever the stock lists), so the cache path segment is the
placeholder ``'-'`` like fred_macro:
``data/raw/<provider>/us_fundamentals/-/<TICKER>/...``.
"""

from __future__ import annotations

VALID_PREFIXES: tuple[str, ...] = ("FUND",)

# Placeholder path segment — fundamentals have no exchange/namespace dimension.
NAMESPACE_SEGMENT = "-"


def parse_identifier(identifier: str) -> tuple[str, str]:
    """Split ``'FUND:AAPL'`` → ``('-', 'AAPL')``."""
    if ":" not in identifier:
        raise ValueError(
            f"missing domain prefix in {identifier!r}; expected 'FUND:TICKER'"
        )
    prefix, symbol = identifier.split(":", 1)
    if prefix.upper() not in VALID_PREFIXES:
        raise ValueError(
            f"unknown us_fundamentals prefix {prefix!r}; valid: {VALID_PREFIXES}"
        )
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError(f"empty ticker in {identifier!r}")
    return NAMESPACE_SEGMENT, symbol
