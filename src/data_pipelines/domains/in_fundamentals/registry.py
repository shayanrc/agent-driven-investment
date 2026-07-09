"""in_fundamentals identifier parser.

Accepted form: ``INFUND:<TICKER>`` (e.g., ``INFUND:RELIANCE``,
``INFUND:HDFCBANK``).

Tickers are the NSE trading symbols exactly as they appear in the
nse_equities universe YAMLs (``NSE:RELIANCE`` → ``INFUND:RELIANCE``) — the
universe is derived from those YAMLs, so there is no second spelling to
drift. Fundamentals have no exchange dimension (dual-listed NSE/BSE companies
file identical results with both exchanges), so the cache path segment is the
placeholder ``'-'`` like us_fundamentals:
``data/raw/<provider>/in_fundamentals/-/<TICKER>/...``.
"""

from __future__ import annotations

VALID_PREFIXES: tuple[str, ...] = ("INFUND",)

# Placeholder path segment — fundamentals have no exchange/namespace dimension.
NAMESPACE_SEGMENT = "-"


def parse_identifier(identifier: str) -> tuple[str, str]:
    """Split ``'INFUND:RELIANCE'`` → ``('-', 'RELIANCE')``."""
    if ":" not in identifier:
        raise ValueError(
            f"missing domain prefix in {identifier!r}; expected 'INFUND:TICKER'"
        )
    prefix, symbol = identifier.split(":", 1)
    if prefix.upper() not in VALID_PREFIXES:
        raise ValueError(
            f"unknown in_fundamentals prefix {prefix!r}; valid: {VALID_PREFIXES}"
        )
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError(f"empty ticker in {identifier!r}")
    return NAMESPACE_SEGMENT, symbol
