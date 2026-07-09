"""Panel orchestration — assemble the daily point-in-time valuation panel.

``build_ticker_panel`` is the pure core (quarterly fundamentals + daily prices +
splits → daily ratio rows); ``build_panel`` is the I/O driver that pulls
fundamentals from the ``us_fundamentals`` cache, prices from the ``us_equities``
cache, and split factors from yfinance, then stitches every ticker together.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from valuation.prices import (
    adjust_shares_to_latest_basis,
    fetch_splits,
    nse_equities_identifier,
    read_adj_close,
    us_equities_identifier,
)
from valuation.ratios import RATIO_COLUMNS, compute_ratios
from valuation.ttm import asof_daily, build_ttm_timeline

# canonical panel columns (order)
PANEL_COLUMNS = (
    "ticker", "date",
    "pe", "ps", "p_fcf",
    "earnings_yield", "sales_yield", "fcf_yield",
    "eps_ttm", "rev_ps_ttm", "fcf_ps_ttm",
    "market_cap", "adj_close", "shares",
    "revenue_ttm", "net_income_ttm", "fcf_ttm", "revenue_q",
    "asof_fiscal_period_end", "asof_filed_date",
)

# fundamentals columns pulled from the us_fundamentals cache
_FUND_COLS = ("fiscal_period_end", "filed_date",
              "revenue", "net_income", "fcf", "shares_diluted")


def build_ticker_panel(
    fund_identifier: str,
    quarterly: pd.DataFrame,
    prices: pd.DataFrame,
    splits: pd.Series,
) -> pd.DataFrame:
    """Pure: one ticker's daily point-in-time ratio rows.

    ``quarterly`` needs ``fiscal_period_end, filed_date, revenue, net_income,
    fcf, shares_diluted``; ``prices`` needs ``date, adj_close``; ``splits`` is a
    date→ratio Series (may be empty).
    """
    q = quarterly.copy()
    q["shares"] = adjust_shares_to_latest_basis(
        q["fiscal_period_end"], q["shares_diluted"], splits
    )
    timeline = build_ttm_timeline(q)
    daily = asof_daily(timeline, prices["date"])
    daily = daily.merge(prices[["date", "adj_close"]], on="date", how="left")
    daily = compute_ratios(daily)
    daily.insert(0, "ticker", fund_identifier)
    return daily.reindex(columns=list(PANEL_COLUMNS)).reset_index(drop=True)


def build_panel(
    tickers: list[str],
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    repo_root: Path | None = None,
    splits_provider: Callable[[str], pd.Series] = fetch_splits,
    on_progress: Callable[[int, int, str], None] | None = None,
    domain=None,
    id_mapper: Callable[[str], str | None] = us_equities_identifier,
    price_table: str = "us_equities_data",
) -> pd.DataFrame:
    """Daily point-in-time valuation panel for ``tickers`` (``FUND:*``).

    Skips a ticker with no cached fundamentals, no equity-universe mapping, or
    no cached prices. ``splits_provider`` is injectable (tests / a cached
    provider avoid the per-ticker yfinance call).

    US byte-identical by default: ``domain`` defaults to the ``us_fundamentals``
    domain, ``id_mapper`` maps ``FUND:*`` → us_equities identifiers, and
    ``price_table`` reads ``us_equities_data``. The NSE path overrides all three
    (``in_fundamentals`` domain, ``nse_equities_identifier``,
    ``nse_equities_data``) — see ``build_nse_panel``.
    """
    from data_pipelines.cache import read_processed

    if domain is None:
        from data_pipelines.domains.us_fundamentals import get_domain
        domain = get_domain()
    root = repo_root if repo_root is not None else Path.cwd()
    frames: list[pd.DataFrame] = []
    for i, fund_id in enumerate(tickers, 1):
        if on_progress:
            on_progress(i, len(tickers), fund_id)
        q, _ = read_processed(root / "data", domain, fund_id)
        if q is None or q.empty:
            continue
        eq_id = id_mapper(fund_id)
        if eq_id is None:
            continue
        prices = read_adj_close(
            eq_id, start, end, repo_root=repo_root, table=price_table
        )
        if prices.empty:
            continue
        symbol = fund_id.split(":", 1)[1]
        panel = build_ticker_panel(
            fund_id, q[list(_FUND_COLS)], prices, splits_provider(symbol)
        )
        # keep only rows with a valid TTM (before the first filing → all NaN)
        panel = panel[panel["asof_filed_date"].notna()].reset_index(drop=True)
        if not panel.empty:
            frames.append(panel)
    if not frames:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def build_nse_panel(
    tickers: list[str],
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    repo_root: Path | None = None,
    splits_provider: Callable[[str], pd.Series] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """NSE convenience wrapper over ``build_panel``.

    Wires the ``in_fundamentals`` domain, ``nse_equities_identifier``
    (``INFUND:RELIANCE`` → ``NSE:RELIANCE``), and the ``nse_equities_data`` price
    table. ``tickers`` are ``INFUND:*``. The default split provider appends the
    yfinance ``.NS`` suffix. India-specific data shape (fcf all-NaN, insurers
    with no diluted shares) flows through as honest NaN ratios — no special
    handling needed here.
    """
    from data_pipelines.domains.in_fundamentals import get_domain

    if splits_provider is None:
        def splits_provider(symbol: str) -> pd.Series:  # noqa: E306
            return fetch_splits(symbol, suffix=".NS")
    return build_panel(
        tickers, start, end,
        repo_root=repo_root,
        splits_provider=splits_provider,
        on_progress=on_progress,
        domain=get_domain(),
        id_mapper=nse_equities_identifier,
        price_table="nse_equities_data",
    )


def latest_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    """The most recent row per ticker — a compact, checked-in-able summary."""
    if panel.empty:
        return panel
    return (
        panel.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
