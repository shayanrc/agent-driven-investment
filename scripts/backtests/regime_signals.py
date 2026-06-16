"""Causal regime signals for the _017/_018 regime gate.

Each function returns a per-date boolean Series: True = risk-ON (allow new entries),
False = risk-OFF (mask predictions → strategy makes no new entries and decays to cash).
NaN = insufficient history (warmup); callers treat NaN as risk-ON (never gate on
unknown). EVERY signal is strictly causal — the value at date t uses only data ≤ t
(rolling / expanding / shift are all trailing) — so a regime gate built on them cannot
peek at the future.

`_017` showed a TREND signal (price > N-day SMA) structurally lags at a market top: at
the 2022 top the 200d SMA was still rising off the 2021 bull, so the gate kept entering
into the decline. `_018` adds FORWARD-LOOKING / fast-reacting signals that can flip
risk-off at or before the turn:

- ``sma``      : trend filter (price > SMA_ma, optional MA-rising slope) — the _017 baseline.
- ``vol``      : realized-vol gate — risk-OFF when trailing annualized vol > threshold
                 (vol RISES as a market turns, so this reacts at the top, not months later).
- ``drawdown`` : risk-OFF when the index is > thresh below its trailing-N-day high
                 (reacts within days of a top, independent of any moving average).
- ``breadth``  : risk-OFF when the fraction of universe names above their own M-day MA
                 falls below thresh — breadth DETERIORATES before the index top, so this is
                 genuinely leading (needs the cross-sectional panel, not just the index).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def risk_on_sma(idx_close: pd.Series, ma: int, slope: int = 0) -> pd.Series:
    """Trend gate: close > SMA_ma [and, if slope>0, SMA rising over `slope` days]."""
    sma = idx_close.rolling(ma).mean()
    on = idx_close > sma
    valid = sma.notna()
    if slope > 0:
        sma_prev = sma.shift(slope)
        on = on & (sma > sma_prev)
        valid = valid & sma_prev.notna()
    return on.where(valid)


def risk_on_vol(idx_close: pd.Series, window: int = 20,
                thresh: float = 0.20) -> pd.Series:
    """Realized-vol gate: risk-ON when trailing `window`-day annualized vol ≤ thresh.

    Vol expands as a market turns down, so this flips risk-OFF near the top — the
    behaviour the trend gate cannot get. thresh is annualized stdev of daily log
    returns (0.20 ≈ the calm-bull/selloff boundary for broad US indices).
    """
    ret = np.log(idx_close).diff()
    rv = ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (rv <= thresh).where(rv.notna())


def risk_on_drawdown(idx_close: pd.Series, window: int = 60,
                     thresh: float = 0.05) -> pd.Series:
    """Drawdown gate: risk-ON when the index is within `thresh` of its trailing
    `window`-day high (risk-OFF once it has fallen more than thresh from that high)."""
    roll_high = idx_close.rolling(window, min_periods=2).max()
    dd = idx_close / roll_high - 1.0
    return (dd > -thresh).where(roll_high.notna())


def risk_on_breadth(roster_closes: pd.DataFrame, ma: int = 50,
                    thresh: float = 0.5) -> pd.Series:
    """Breadth gate: risk-ON when the fraction of names trading above their own
    `ma`-day SMA exceeds `thresh`.

    roster_closes: wide DataFrame (index=date, columns=ticker) of closes. Breadth
    thins out before the index tops (fewer names participating), so this leads.
    Each name's above/below-MA is causal; the cross-sectional mean each day uses only
    that day's values. Days with too few valid names (< ma history) → NaN.
    """
    sma = roster_closes.rolling(ma).mean()
    above = roster_closes > sma
    # Only count names that HAVE an ma-day history that day (else NaN, excluded).
    valid = sma.notna()
    n_valid = valid.sum(axis=1)
    frac = above.where(valid).sum(axis=1) / n_valid.replace(0, np.nan)
    return (frac > thresh).where(n_valid > 0)


def compute_risk_on(signal: str, idx_close: pd.Series,
                    roster_closes: pd.DataFrame | None = None, **kw) -> pd.Series:
    """Dispatch to the named regime signal. Unknown signal → ValueError."""
    if signal == "sma":
        return risk_on_sma(idx_close, ma=kw["ma"], slope=kw.get("slope", 0))
    if signal == "vol":
        return risk_on_vol(idx_close, window=kw.get("window", 20),
                           thresh=kw.get("thresh", 0.20))
    if signal == "drawdown":
        return risk_on_drawdown(idx_close, window=kw.get("window", 60),
                                thresh=kw.get("thresh", 0.05))
    if signal == "breadth":
        if roster_closes is None or roster_closes.empty:
            raise ValueError("breadth signal requires roster_closes")
        return risk_on_breadth(roster_closes, ma=kw.get("ma", 50),
                               thresh=kw.get("thresh", 0.5))
    raise ValueError(f"unknown regime signal {signal!r}")
