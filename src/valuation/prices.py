"""Price + split-basis plumbing for the valuation panel.

Three jobs:

1. **FUND: → us_equities identifier.** The us_fundamentals universe was derived
   by stripping the exchange prefix off the us_equities universe YAMLs; this
   reverses it (``FUND:AAPL`` → ``NASDAQ:AAPL``) so we can read the stock's
   prices.
2. **Daily ``adj_close``** from the us_equities SQLite cache (the split- AND
   dividend-adjusted column — the one with consistent cross-provider semantics).
3. **Split-basis alignment.** As-reported quarterly shares are in the split
   basis *at the time of each filing*; ``adj_close`` is in the *latest* basis.
   To make ``market_cap = adj_close × shares`` split-consistent, historical
   shares are multiplied by the cumulative product of split ratios that
   occurred *after* the share count's fiscal date (yfinance ``.splits``,
   cached). Most tickers never split → factor 1.0, a no-op.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from data_pipelines.cache import half_open_day_bounds
from data_pipelines.domains.us_equities.universe import (
    load_universe as load_us_equities_universe,
)

# Equity universes that make up the fundamentals universe (mirror of
# valuation's sibling us_fundamentals.universe.EQUITY_UNIVERSES).
_EQUITY_UNIVERSES = ("sp500", "russell1000", "nasdaq100")


def _data_root(repo_root: Path | None) -> Path:
    return (Path(repo_root) if repo_root is not None else Path.cwd()) / "data"


@lru_cache(maxsize=1)
def _symbol_to_identifier() -> dict[str, str]:
    """symbol (``AAPL``) → us_equities identifier (``NASDAQ:AAPL``).

    Built from the union of the equity-universe YAMLs, indices excluded. If a
    symbol appears under two exchanges (it shouldn't), the first wins
    deterministically (universes are loaded in a fixed order).
    """
    out: dict[str, str] = {}
    for uni in _EQUITY_UNIVERSES:
        for ident in load_us_equities_universe(uni):
            prefix, symbol = ident.split(":", 1)
            if prefix == "INDEX":
                continue
            out.setdefault(symbol, ident)
    return out


def us_equities_identifier(fund_identifier: str) -> str | None:
    """``FUND:AAPL`` → ``NASDAQ:AAPL`` (or None if the symbol isn't in any
    equity universe)."""
    symbol = fund_identifier.split(":", 1)[1] if ":" in fund_identifier else fund_identifier
    return _symbol_to_identifier().get(symbol.upper())


def read_adj_close(
    identifier: str,
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Daily ``(date, adj_close)`` for a us_equities identifier from the cache.

    Uses the half-open ``[start, end+1)`` interval so the stored
    ``'YYYY-MM-DD 00:00:00'`` time component doesn't drop the end day (the #182
    off-by-one). Returns an empty frame if the ticker isn't cached.
    """
    db = _data_root(repo_root) / "processed.db"
    if not db.is_file():
        raise FileNotFoundError(f"cache db missing at {db}")
    start_s, end_excl = half_open_day_bounds(start, end)
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(
            "SELECT date, adj_close FROM us_equities_data "
            "WHERE ticker = ? AND date >= ? AND date < ? ORDER BY date",
            con, params=(identifier, start_s, end_excl),
        )
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    df["adj_close"] = df["adj_close"].astype("float64")
    return df.dropna(subset=["adj_close"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split-basis alignment
# ---------------------------------------------------------------------------

def fetch_splits(symbol: str) -> pd.Series:
    """Split events for ``symbol`` from yfinance: a Series indexed by (naive)
    date with the split ratio (4.0 for a 4:1). Empty if none / on error."""
    try:
        import yfinance as yf
        s = yf.Ticker(symbol.replace(".", "-")).splits
    except Exception:
        return pd.Series(dtype="float64")
    if s is None or len(s) == 0:
        return pd.Series(dtype="float64")
    s = s[s > 0].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def cumulative_split_factor(splits: pd.Series, asof: pd.Timestamp) -> float:
    """Product of split ratios strictly AFTER ``asof`` — the factor that lifts a
    share count reported at ``asof`` into the latest (post-all-splits) basis.

    A 4:1 split after ``asof`` means the reported count is 1/4 of today's share
    basis, so it must be multiplied by 4.
    """
    if splits is None or len(splits) == 0:
        return 1.0
    asof = pd.Timestamp(asof).normalize()
    after = splits[splits.index > asof]
    return float(after.prod()) if len(after) else 1.0


def adjust_shares_to_latest_basis(
    fiscal_ends: pd.Series, shares: pd.Series, splits: pd.Series,
) -> pd.Series:
    """Vectorized ``shares × cumulative_split_factor(fiscal_end)`` so every
    quarter's share count is expressed in the latest split basis (matching
    ``adj_close``). ``fiscal_ends`` and ``shares`` are aligned by index."""
    factors = pd.Series(
        [cumulative_split_factor(splits, fe) for fe in fiscal_ends],
        index=shares.index, dtype="float64",
    )
    return shares.astype("float64") * factors
