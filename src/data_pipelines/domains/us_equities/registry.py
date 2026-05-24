"""us_equities identifier parser.

Accepted forms:
  NYSE:AAPL, NASDAQ:MSFT, INDEX:^SPX, INDEX:^NDX, INDEX:^DJI, INDEX:^RUT

A bare symbol (e.g., 'AAPL' with no prefix) raises ValueError. The framework's
DomainRegistry never reaches the parser without a prefix (UnknownDomain fires
first), but the parser still enforces the contract for direct callers.

Open question 7 in V1_IMPLEMENTATION_PLAN.md: prefix maps 1:1 to the cache path
segment under data/{raw,processed}/us_equities/<prefix>/<symbol>/.
"""

from __future__ import annotations

VALID_PREFIXES: tuple[str, ...] = ("NYSE", "NASDAQ", "INDEX")
SUPPORTED_INDICES: frozenset[str] = frozenset({"^SPX", "^NDX", "^DJI", "^RUT"})


def parse_identifier(identifier: str) -> tuple[str, str]:
    """Split '<PREFIX>:<SYMBOL>' → (exchange, symbol).

    Symbols are uppercased; the leading caret (`^`) for indices is preserved.
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
            f"unknown us_equities prefix {prefix!r}; valid: {VALID_PREFIXES}"
        )
    symbol = symbol.strip()
    if not symbol:
        raise ValueError(f"empty symbol in {identifier!r}")
    if prefix_u == "INDEX":
        # Preserve case-sensitive ^ but uppercase the alpha part.
        if not symbol.startswith("^"):
            raise ValueError(
                f"INDEX symbols must start with '^', got {symbol!r}"
            )
        symbol = "^" + symbol[1:].upper()
        if symbol not in SUPPORTED_INDICES:
            # Not a hard reject — out-of-universe handling is dispatch's job
            # (warn, don't fail). Parser accepts any well-formed INDEX symbol.
            pass
    else:
        symbol = symbol.upper()
    return prefix_u, symbol
