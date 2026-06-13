"""Benchmarks + metric helpers for the cell-5 back-test (plan §6.7).

Three benchmarks, same $100K start, all measured at ``comparison_end``
(= test_end + horizon_days, the latest exit date for a position opened on
test_end) so equity is apples-to-apples with the strategy:

1. NDX cap-weighted buy-and-hold (real ``INDEX:^NDX``; price-return basis).
2. 92-ticker equal-weight basket (buy-and-hold from test_start).
3. Event-driven top-K (no Kelly, no rebalance) — the V1-pre-Path-A
   counterfactual: top-K by p each signal day, uniform 1/K sizing,
   DD/target/horizon exits only.

All on the cache's split-adjusted ``close`` (D24 price-return basis), the
same column the gbdt label used — so the comparison shares one price basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def compute_metrics(equity: pd.Series) -> dict[str, float]:
    """total_return, cagr, max_dd from a daily equity series."""
    equity = equity.dropna()
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0
    n_days = len(equity)
    years = max(n_days / TRADING_DAYS, 1e-9)
    cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 else float("nan")
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_dd = float(drawdown.min())
    return {
        "start": start,
        "end": end,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_dd": max_dd,
    }


def _price_panel(
    closes: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Align per-ticker close series into a [start, end] date×ticker frame."""
    frame = pd.DataFrame(closes)
    frame = frame[(frame.index >= start) & (frame.index <= end)].sort_index()
    return frame


def buy_and_hold(
    close: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_cash: float = 100_000.0,
) -> tuple[pd.Series, dict[str, float], int]:
    """Single-asset buy-and-hold equity from start through end (1 trade)."""
    s = close[(close.index >= start) & (close.index <= end)].dropna().sort_index()
    shares = initial_cash / float(s.iloc[0])
    equity = s * shares
    m = compute_metrics(equity)
    m["n_trades"] = 1
    m["gross_exposure_avg"] = 1.0
    return equity, m, 1


def equal_weight_basket(
    closes: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_cash: float = 100_000.0,
) -> tuple[pd.Series, dict[str, float], int]:
    """Equal-weight buy-and-hold across all tickers with data at start.

    $cash/N to each ticker priced at start, held to end. Tickers without a
    close at start are skipped (reported via n_trades). Forward-fills gaps so
    a mid-window delisting holds its last price (mirrors the engine's
    ffill_zero_volume liquidation basis)."""
    frame = _price_panel(closes, start, end).ffill()
    first = frame.iloc[0]
    tradable = first.dropna().index.tolist()
    n = len(tradable)
    per = initial_cash / n
    shares = {t: per / float(first[t]) for t in tradable}
    equity = sum(frame[t] * shares[t] for t in tradable)
    m = compute_metrics(equity)
    m["n_trades"] = n
    m["gross_exposure_avg"] = 1.0
    return equity, m, n


def event_driven_topk(
    predictions: dict[pd.Timestamp, list[tuple[str, float, float, float]]],
    closes: dict[str, pd.Series],
    *,
    K: int,
    target_return: float,
    stop_drawdown: float,
    horizon_days: int,
    breakeven_p: float,
    timeline: pd.DatetimeIndex,
    initial_cash: float = 100_000.0,
) -> tuple[pd.Series, dict[str, float], int]:
    """Top-K, uniform 1/K sizing, DD/target/horizon exits, NO rebalance.

    The V1-pre-Path-A counterfactual (no breakeven exit, no daily trim). A
    simple cash-accounting simulation on signal-day closes: each entry buys
    1/K of *current equity* at the signal-day close; exits realize at the
    trigger-day close. Tie-break matches D21 (p desc, ticker asc)."""
    pos: dict[str, dict] = {}  # ticker -> {anchor, entry_idx, value0, shares}
    cash = initial_cash
    n_trades = 0
    equity_path = []
    idx_of = {d: i for i, d in enumerate(timeline)}

    def close_on(t: str, d: pd.Timestamp) -> float | None:
        s = closes.get(t)
        if s is None or d not in s.index:
            return None
        v = float(s.loc[d])
        return v if v == v and v > 0 else None

    for d in timeline:
        i = idx_of[d]
        # exits first (frees cash)
        for t in list(pos):
            c = close_on(t, d)
            if c is None:
                continue
            p = pos[t]
            held = i - p["entry_idx"]
            hit = (
                c <= (1 - stop_drawdown) * p["anchor"]
                or c >= (1 + target_return) * p["anchor"]
                or held >= horizon_days
            )
            if hit:
                cash += p["shares"] * c
                del pos[t]
        # entries (uniform 1/K of current equity)
        rows = predictions.get(d, [])
        cands = sorted(
            [(t, pm) for (t, pm, _l, _h) in rows if pm > breakeven_p and t not in pos],
            key=lambda x: (-x[1], x[0]),
        )
        equity_now = cash + sum(
            p["shares"] * (close_on(t, d) or p["anchor"]) for t, p in pos.items()
        )
        for t, _pm in cands[:K]:
            c = close_on(t, d)
            if c is None:
                continue
            alloc = equity_now / K
            if alloc > cash:
                alloc = cash
            if alloc <= 0:
                break
            shares = alloc / c
            pos[t] = {"anchor": c, "entry_idx": i, "shares": shares}
            cash -= alloc
            n_trades += 1
        # mark equity
        mark = cash + sum(
            p["shares"] * (close_on(t, d) or p["anchor"]) for t, p in pos.items()
        )
        equity_path.append((d, mark))

    equity = pd.Series(dict(equity_path)).sort_index()
    m = compute_metrics(equity)
    m["n_trades"] = n_trades
    m["gross_exposure_avg"] = float(
        np.mean([1.0 - 0.0 for _ in equity])
    )  # placeholder; detailed exposure tracked in strategy only
    return equity, m, n_trades
