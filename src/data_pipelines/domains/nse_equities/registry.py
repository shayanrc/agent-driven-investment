"""nse_equities identifier parser.

Accepted forms:
  NSE:RELIANCE, NSE:HDFCBANK, NSE:M&M, NSE:BAJAJ-AUTO
  BSE:RELIANCE          (parser-accepted, but jugaad/nselib don't cover BSE;
                         adapters fall through and the chain terminates with
                         AllProvidersFailed unless yfinance .BO succeeds)
  NIFTY:50              → "NIFTY 50" upstream (nselib)
  NIFTY:BANK            → "NIFTY BANK"
  NIFTY:NEXT50          → "NIFTY NEXT 50"

`INDEX:` is intentionally NOT a valid NSE prefix because us_equities already
owns it (DomainRegistry's invariant: one prefix → one domain). NSE indices
get their own `NIFTY:` prefix.

Symbol case is preserved (NSE symbols like "M&M" and "BAJAJ-AUTO" use mixed
chars), but uppercased for the prefix-stripped form to match upstream NSE
conventions where everything is uppercase.
"""

from __future__ import annotations

VALID_PREFIXES: tuple[str, ...] = ("NSE", "BSE", "NIFTY")

# NIFTY: aliases → upstream string used by nselib.capital_market.index_data().
# These are the index symbols that have public OHLCV history available.
# Keep this list small — the user-facing identifier is short and discoverable;
# add new entries only when a downstream consumer asks.
NIFTY_INDEX_SLUGS: dict[str, str] = {
    "50": "NIFTY 50",
    "NEXT50": "NIFTY NEXT 50",
    "BANK": "NIFTY BANK",
    "IT": "NIFTY IT",
    "MIDCAP100": "NIFTY MIDCAP 100",
    "SMALLCAP100": "NIFTY SMALLCAP 100",
}


def parse_identifier(identifier: str) -> tuple[str, str]:
    """Split '<PREFIX>:<SYMBOL>' → (exchange, symbol).

    For NIFTY: the symbol is the short alias (e.g., "50") — adapters look up
    NIFTY_INDEX_SLUGS to translate to the upstream name. Unknown aliases are
    accepted (parser is lax); the adapter raises EmptyPayload if the upstream
    doesn't recognize the slug. This mirrors us_equities' INDEX: behavior.
    """
    if ":" not in identifier:
        raise ValueError(
            f"missing domain prefix in {identifier!r}; "
            f"expected one of: {', '.join(p + ':SYMBOL' for p in VALID_PREFIXES)}"
        )
    prefix, symbol = identifier.split(":", 1)
    prefix_u = prefix.upper()
    if prefix_u not in VALID_PREFIXES:
        raise ValueError(
            f"unknown nse_equities prefix {prefix!r}; valid: {VALID_PREFIXES}"
        )
    symbol = symbol.strip()
    if not symbol:
        raise ValueError(f"empty symbol in {identifier!r}")
    # Upstream NSE conventions: all uppercase. Symbols may contain `&`, `-`.
    symbol = symbol.upper()
    return prefix_u, symbol


def resolve_nifty_slug(symbol: str) -> str | None:
    """Return the upstream NSE index name for a NIFTY: short alias.

    Returns None if the alias is unrecognized — callers (adapters) decide
    whether to raise EmptyPayload or attempt a passthrough.
    """
    return NIFTY_INDEX_SLUGS.get(symbol.upper())
