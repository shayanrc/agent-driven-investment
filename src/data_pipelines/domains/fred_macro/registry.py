"""fred_macro identifier parser.

Accepted form: ``FRED:<SERIES_ID>`` (e.g., ``FRED:DGS10``, ``FRED:CPIAUCSL``).

FRED has no exchange/namespace concept, so the cache path segment is a
placeholder ``'-'``:
``data/raw/fred/fred_macro/-/<SERIES_ID>/...`` (per ``raw_store`` docstring).
Series IDs are uppercased — FRED series IDs are conventionally uppercase.
"""

from __future__ import annotations

VALID_PREFIXES: tuple[str, ...] = ("FRED",)

# Placeholder path segment — FRED has no exchange/namespace dimension.
NAMESPACE_SEGMENT = "-"


def parse_identifier(identifier: str) -> tuple[str, str]:
    """Split ``'FRED:DGS10'`` → ``('-', 'DGS10')``.

    The first element is the cache path segment (the exchange/namespace slot);
    FRED has none, so it's the ``'-'`` placeholder per
    ``docs/data_pipelines/adding_a_domain.md``. The second is the uppercased
    FRED series id.
    """
    if ":" not in identifier:
        raise ValueError(
            f"missing domain prefix in {identifier!r}; expected 'FRED:SERIES_ID'"
        )
    prefix, symbol = identifier.split(":", 1)
    if prefix.upper() not in VALID_PREFIXES:
        raise ValueError(
            f"unknown fred_macro prefix {prefix!r}; valid: {VALID_PREFIXES}"
        )
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError(f"empty series id in {identifier!r}")
    return NAMESPACE_SEGMENT, symbol
