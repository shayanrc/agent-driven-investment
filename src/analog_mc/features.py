"""Causal rolling features for the analog_mc pipeline.

All functions here implement constraint **C1** from the IMPLEMENTATION_PLAN:
the value at index t may only depend on data with index <= t. The unit tests
in tests/analog_mc/test_features.py verify this property directly.

Convention choices documented here (the plan flagged both as "pick one and
document"):

  * ``causal_ewma_vol`` at t uses returns[: t + 1] (inclusive of t).
    This matches the test described in the plan ("value at t equals value
    computed from series.iloc[:t+1] alone"). pandas' ``.ewm(...).std()`` is
    causal by construction, so no shift is needed.

  * ``causal_zscore`` at t for horizon h is
        mean(returns[t - h + 1 : t + 1]) / std(returns[t - h + 1 : t + 1])
    i.e., the standardized recent-trend signal over the h returns ending at
    and including t. We use the [t - h + 1 : t + 1] window (h returns ending
    at t inclusive) rather than the plan's literal [t - h : t] sketch so the
    z-score and the EWMA vol use the same "data through t" semantics. Both
    are equally causal; the symmetry simplifies reasoning at forecast origin
    dates where we want both features computed over the same observation set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_ewma_vol(returns: pd.Series, halflife: float) -> pd.Series:
    """Trailing EWMA standard deviation of returns.

    Output is a Series indexed identically to ``returns``. The value at any
    index t depends only on returns[: t + 1] (no look-ahead).

    Pandas' ``ewm(halflife=..., adjust=False).std()`` is causal — at index t
    it uses the recursion seeded from the start of the series and weighted by
    the halflife decay. The earliest value is NaN (needs at least two points).
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0; got {halflife}")
    return returns.ewm(halflife=halflife, adjust=False).std()


def causal_zscore(returns: pd.Series, horizon: int) -> pd.Series:
    """Trailing z-score over the past ``horizon`` returns ending at t inclusive.

    z_t = mean(returns[t - horizon + 1 : t + 1]) / std(returns[t - horizon + 1 : t + 1])

    This is a standardized recent-trend signal: how large is the recent mean
    return relative to its recent dispersion. Early indices (t < horizon - 1)
    are NaN.

    The value at index t depends only on returns[: t + 1] (no look-ahead),
    making it usable as a feature both at forecast-origin dates (where t is the
    most recent observation) and at historical analog-candidate dates.
    """
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2 to compute a std; got {horizon}")
    window = returns.rolling(window=horizon, min_periods=horizon)
    mean = window.mean()
    std = window.std(ddof=1)
    # Guard against zero std (constant window). Returns NaN rather than inf.
    z = mean / std.replace(0.0, np.nan)
    z.name = f"zscore_{horizon}"
    return z


def causal_trailing_mean(returns: pd.Series, horizon: int) -> pd.Series:
    """Trailing arithmetic mean of the past ``horizon`` returns ending at t inclusive.

    μ_t = mean(returns[t - horizon + 1 : t + 1])

    Uses the SAME window convention as ``causal_zscore`` so the trailing mean
    and the z-score numerator stay consistent. Required as the source of
    ``mu_origin`` in C3.

    The value at index t depends only on returns[: t + 1] (no look-ahead).
    Early indices (t < horizon - 1) are NaN.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    m = returns.rolling(window=horizon, min_periods=horizon).mean()
    m.name = f"trailing_mean_{horizon}"
    return m


def compute_features(
    returns: pd.Series,
    halflife: float,
    horizons: tuple[int, ...],
    momentum_lookback: int | None = None,
) -> pd.DataFrame:
    """Compute the standard feature bundle.

    Columns:
      * ``ewma_vol`` — trailing EWMA std (causal).
      * ``zscore_<h>`` — trailing z-score at each horizon h in ``horizons``.
      * ``trailing_mean_<max_h>`` — trailing arithmetic mean at the longest
        horizon. Used as the source of ``mu_origin`` in C3.
      * ``trailing_mean_<momentum_lookback>`` — added only when
        ``momentum_lookback`` is given AND differs from ``max(horizons)``.
        Used as the source of v2.1 trailing-momentum drift. Kept distinct
        from the mu_origin column because the two serve different roles:
        the long-horizon mean defines current regime baseline, the short
        lookback estimates recent directional pressure.

    The DataFrame is indexed identically to ``returns``.
    """
    cols: dict[str, pd.Series] = {"ewma_vol": causal_ewma_vol(returns, halflife)}
    for h in horizons:
        cols[f"zscore_{h}"] = causal_zscore(returns, h)
    max_h = max(horizons)
    cols[f"trailing_mean_{max_h}"] = causal_trailing_mean(returns, max_h)
    if momentum_lookback is not None and momentum_lookback != max_h:
        cols[f"trailing_mean_{momentum_lookback}"] = causal_trailing_mean(returns, momentum_lookback)
    return pd.DataFrame(cols, index=returns.index)
