"""Causal feature engineering for gbdt v1.

Implements the 279-column candidate pool described in
``docs/gbdt/V1_PLAN.md`` Stage 2. Two pipeline modes are exposed:

- Per-stock rolling (``F1..F13, F15, F16``): each function operates on a
  single ticker's time series in isolation. Causal at row ``t`` — values
  depend only on data at indices ``<= t``.
- Cross-sectional point-in-time (``F14`` + ``F7`` xs cols): computed per
  date across the panel.

The leakage harness in ``gbdt.leakage_harness`` exercises each per-family
function on a synthetic panel; the unit tests in ``tests/gbdt/test_features``
assert every family stays causal.

Multi-window families are parameterized on a ``lookbacks`` tuple; default
is ``(5, 10, 20, 50, 100, 200)`` per the spec.

Annualization is universe-conditional and supplied by the caller (``250``
for NIFTY; ``252`` for US universes once v1.1 adds them).
"""

from __future__ import annotations

import fnmatch
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Progress-logging throttle
# ---------------------------------------------------------------------------

# Minimum seconds between successive ``[features] family=...`` progress
# lines emitted from inside ``build_feature_matrix``. Time-based throttle
# (NOT per-step) so small panels emit ~1 line while big panels (sp500
# H=25 hits ~155 min on the features stage) emit ~50 lines. Tight scope
# by design — kept as a module-level constant, not a CLI/spec knob, per
# the runner-logging PR scope.
_FEATURES_PROGRESS_THROTTLE_SEC: float = 30.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)

# Total column count per V1_PLAN Stage 2:
#   F1 6 + F2 6 + F3 6 + F4 6 + F5 6 + F6 6 + F6b 6 + F9 6 + F9b 6 = 54
#   F7 32 = 86; F8 12 = 98; F10 6 = 104; F11 12 = 116; F12 6 = 122;
#   F13 18 = 140; F14 24 = 164; F15 10 = 174; F16 105 = 279
EXPECTED_TOTAL_COLS = 279


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _per_ticker(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` to each ticker's slice and recombine, preserving the
    ``(date, ticker)`` MultiIndex order.
    """
    return series.groupby(level="ticker", group_keys=False).apply(fn)


def _roll(series: pd.Series, n: int, op: str) -> pd.Series:
    """Native per-ticker trailing rolling agg (``min_periods=n``), realigned to
    ``series.index``. Bit-identical to
    ``_per_ticker(series, lambda s: getattr(s.rolling(n, min_periods=n), op)())``
    but runs one native Cython ``groupby.rolling`` pass instead of a Python
    ``groupby.apply`` (V1.6 Phase 1b — ~2.4× per rolling call, no split/combine
    overhead)."""
    rolled = getattr(series.groupby(level="ticker").rolling(n, min_periods=n), op)()
    return rolled.droplevel(0).reindex(series.index)


def _shift(series: pd.Series, n: int) -> pd.Series:
    """Native per-ticker shift — bit-identical to
    ``_per_ticker(series, lambda s: s.shift(n))`` (V1.6 Phase 1b)."""
    return series.groupby(level="ticker").shift(n)


def _rolling_pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Trailing percentile rank of the *current* value within its N-window:
    the fraction of the window ``<=`` the last value.

    Bit-identical to the prior
    ``rolling(n, min_periods=n).apply(lambda w: (w[-1] >= w).mean(), raw=True)``
    — ``method="max"`` gives a tied value the count of elements ``<=`` it, which
    is exactly ``(w[-1] >= w).sum()``, and ``pct=True`` divides by the window's
    valid count ``n`` — but native Cython instead of a per-window Python
    callback. The ``apply`` form was the single largest F7/F13 wall-clock
    contributor (GBDTPERF profile: ~87s across the two call sites); the
    equivalence is verified exact (incl. ties + NaN) by the feature-parity gate.
    """
    return s.rolling(n, min_periods=n).rank(method="max", pct=True)


def _safe_log_returns(close: pd.Series) -> pd.Series:
    # Guard: close of 0 (halted/sparse ticker) makes np.log(0) → -inf,
    # which then propagates through .diff() and pollutes downstream
    # rolling-std / realized-vol / beta. Replace 0 → NaN so log → NaN and
    # the NaN flows through cleanly (zero-denom #182).
    return np.log(close.replace(0, np.nan)).groupby(level="ticker").diff()


def _close(panel: pd.DataFrame) -> pd.Series:
    return panel["close"]


def _broadcast_index_to_panel(idx_feat: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Expand a date-indexed DataFrame to the panel's (date, ticker) MultiIndex.

    Per-(date, ticker) row gets the same index value (broadcast across tickers).
    """
    dates = panel.index.get_level_values("date")
    aligned = idx_feat.reindex(dates).values
    out = pd.DataFrame(aligned, index=panel.index, columns=idx_feat.columns)
    return out


# ---------------------------------------------------------------------------
# F1 / F2 / F3 — returns and relative strength (per family: 6 cols each)
# ---------------------------------------------------------------------------


def stock_return_N(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    close = _close(panel)
    out = {}
    for N in lookbacks:
        # Guard: prior close of 0 (sparse/halted ticker) would emit ±inf; route
        # to NaN so the backend's missing-value path handles it (zero-denom #182).
        out[f"stock_return_{N}"] = close / _shift(close, N).replace(0, np.nan) - 1.0
    return pd.DataFrame(out).reindex(panel.index)


def index_return_N(index_df: pd.DataFrame, panel: pd.DataFrame,
                    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    iclose = index_df["close"]
    out = {}
    for N in lookbacks:
        # Guard: prior index close of 0 → NaN, not ±inf (zero-denom #182).
        out[f"index_return_{N}"] = iclose / iclose.shift(N).replace(0, np.nan) - 1.0
    return _broadcast_index_to_panel(pd.DataFrame(out), panel)


def rel_strength_N(panel: pd.DataFrame, index_df: pd.DataFrame,
                    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
                    *,
                    sr: pd.DataFrame | None = None,
                    ir: pd.DataFrame | None = None) -> pd.DataFrame:
    """``sr`` / ``ir`` may be supplied by callers that have already computed
    F2 (``stock_return_N``) / F1 (``index_return_N``). Default-None preserves
    the standalone-call contract.
    """
    if sr is None:
        sr = stock_return_N(panel, lookbacks)
    if ir is None:
        ir = index_return_N(index_df, panel, lookbacks)
    out = {}
    for N in lookbacks:
        out[f"rel_strength_{N}"] = sr[f"stock_return_{N}"] - ir[f"index_return_{N}"]
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# F4 / F5 — realized vol (per stock + index)
# ---------------------------------------------------------------------------


def realized_vol_N(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
                    annualization: int = 250) -> pd.DataFrame:
    rets = _safe_log_returns(_close(panel))
    sqrt_ann = float(np.sqrt(annualization))
    out = {}
    for N in lookbacks:
        out[f"realized_vol_{N}"] = _roll(rets, N, "std") * sqrt_ann
    return pd.DataFrame(out).reindex(panel.index)


def index_vol_N(index_df: pd.DataFrame, panel: pd.DataFrame,
                 lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
                 annualization: int = 250) -> pd.DataFrame:
    # Guard: index close of 0 → NaN log-return, not -inf (zero-denom #182).
    irets = np.log(index_df["close"].replace(0, np.nan)).diff()
    sqrt_ann = float(np.sqrt(annualization))
    out = {}
    for N in lookbacks:
        out[f"index_vol_{N}"] = irets.rolling(N, min_periods=N).std() * sqrt_ann
    return _broadcast_index_to_panel(pd.DataFrame(out), panel)


# ---------------------------------------------------------------------------
# F6 / F6b / F9 / F9b — drawdown + runup
# ---------------------------------------------------------------------------


def drawdown_N(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    """``close[t] / max(high[t-N+1:t+1]) - 1`` per stock — uses HIGH (causal:
    today's high is observed)."""
    close = panel["close"]
    high = panel["high"]
    out = {}
    for N in lookbacks:
        rolling_max = _roll(high, N, "max")
        # Guard: rolling high max of 0 (all-zero highs on a halted ticker) → NaN
        # not ±inf (zero-denom #182).
        out[f"drawdown_{N}"] = close / rolling_max.replace(0, np.nan) - 1.0
    return pd.DataFrame(out, index=panel.index)


def runup_N(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    close = panel["close"]
    low = panel["low"]
    out = {}
    for N in lookbacks:
        rolling_min = _roll(low, N, "min")
        # Guard: rolling low min of 0 (zero-low bar on a sparse / halted
        # ticker) → NaN, not ±inf (zero-denom #182).
        out[f"runup_{N}"] = close / rolling_min.replace(0, np.nan) - 1.0
    return pd.DataFrame(out, index=panel.index)


def index_drawdown_N(index_df: pd.DataFrame, panel: pd.DataFrame,
                      lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    iclose = index_df["close"]
    ihigh = index_df["high"]
    out = {}
    for N in lookbacks:
        # Guard: index rolling high max of 0 → NaN (zero-denom #182).
        denom = ihigh.rolling(N, min_periods=N).max().replace(0, np.nan)
        out[f"index_drawdown_{N}"] = iclose / denom - 1.0
    return _broadcast_index_to_panel(pd.DataFrame(out), panel)


def index_runup_N(index_df: pd.DataFrame, panel: pd.DataFrame,
                   lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    iclose = index_df["close"]
    ilow = index_df["low"]
    out = {}
    for N in lookbacks:
        # Guard: index rolling low min of 0 → NaN (zero-denom #182).
        denom = ilow.rolling(N, min_periods=N).min().replace(0, np.nan)
        out[f"index_runup_{N}"] = iclose / denom - 1.0
    return _broadcast_index_to_panel(pd.DataFrame(out), panel)


# ---------------------------------------------------------------------------
# F7 — volume family (32 cols)
# ---------------------------------------------------------------------------


def _volume_ratio_N(panel: pd.DataFrame, lookbacks):
    vol = panel["volume"].astype(float)
    out = {}
    for N in lookbacks:
        avg = _roll(vol, N, "mean")
        out[f"volume_ratio_{N}"] = vol / avg.replace(0, np.nan)
    return out


def _obv_N(panel: pd.DataFrame, lookbacks):
    """Per-stock OBV; the N-period change scaled by trailing mean volume."""
    close = panel["close"]
    vol = panel["volume"].astype(float)

    def _per_obv(c: pd.Series) -> pd.Series:
        # c here is the per-ticker close slice with MultiIndex preserved.
        # We need volume's matching slice for the same dates.
        ticker = c.index.get_level_values("ticker")[0]
        v = vol.xs(ticker, level="ticker", drop_level=False)
        # Re-align v to c's index in case the slices differ in order
        v = v.reindex(c.index)
        sign = np.sign(c.diff()).fillna(0.0)
        return (sign * v).cumsum()

    obv = _per_ticker(close, _per_obv)

    out = {}
    for N in lookbacks:
        diff = obv - _shift(obv, N)
        avg_v = _roll(vol, N, "mean")
        out[f"obv_{N}"] = diff / (avg_v.replace(0, np.nan) * N)
    return out


def _vol_ret_corr_N(panel: pd.DataFrame, lookbacks):
    rets = _safe_log_returns(panel["close"])
    vol_change = np.log(
        panel["volume"].astype(float).replace(0, np.nan)
    ).groupby(level="ticker").diff()
    tickers = panel.index.get_level_values("ticker").unique()
    out = {}
    for N in lookbacks:
        chunks = []
        for t in tickers:
            r = rets.xs(t, level="ticker")
            vc = vol_change.xs(t, level="ticker")
            c = r.rolling(N, min_periods=N).corr(vc)
            c.index = pd.MultiIndex.from_product([c.index, [t]], names=["date", "ticker"])
            chunks.append(c)
        stitched = pd.concat(chunks).sort_index()
        out[f"vol_ret_corr_{N}"] = stitched
    return out


def _dollar_move_zscore_N(panel: pd.DataFrame, lookbacks):
    close = panel["close"]
    vol = panel["volume"].astype(float)
    dm = close.diff().abs() * vol
    out = {}
    for N in lookbacks:
        mean = _roll(dm, N, "mean")
        std = _roll(dm, N, "std")
        out[f"dollar_move_zscore_{N}"] = (dm - mean) / std.replace(0, np.nan)
    return out


def _dollar_move_rank_N(panel: pd.DataFrame, lookbacks):
    close = panel["close"]
    vol = panel["volume"].astype(float)
    dm = close.diff().abs() * vol
    out = {}
    for N in lookbacks:
        # Vectorized trailing percentile rank (native Cython) — bit-identical to
        # the prior rolling().apply((w[-1] >= w).mean()) but ~10x cheaper; this
        # was the largest contributor to F7's wall-clock. See _rolling_pct_rank.
        out[f"dollar_move_rank_{N}"] = _per_ticker(
            dm, lambda s, n=N: _rolling_pct_rank(s, n),
        )
    return out


def _dollar_move_xs(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Cross-sectional z-score and rank of today's dollar move across the panel."""
    close = panel["close"]
    vol = panel["volume"].astype(float)
    dm = close.diff().abs() * vol
    g = dm.groupby(level="date")
    z = (dm - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    r = g.rank(pct=True)
    return {"dollar_move_xs_zscore": z, "dollar_move_xs_rank": r}


def volume_family(panel: pd.DataFrame,
                   lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    out = {}
    out.update(_volume_ratio_N(panel, lookbacks))
    out.update(_obv_N(panel, lookbacks))
    out.update(_vol_ret_corr_N(panel, lookbacks))
    out.update(_dollar_move_zscore_N(panel, lookbacks))
    out.update(_dollar_move_rank_N(panel, lookbacks))
    out.update(_dollar_move_xs(panel))
    return pd.DataFrame(out).reindex(panel.index)


# ---------------------------------------------------------------------------
# F8 — higher moments
# ---------------------------------------------------------------------------


def higher_moments(panel: pd.DataFrame,
                    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    rets = _safe_log_returns(_close(panel))
    out = {}
    for N in lookbacks:
        out[f"returns_skew_{N}"] = _roll(rets, N, "skew")
        out[f"returns_kurt_{N}"] = _roll(rets, N, "kurt")
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# F10 — rolling beta of stock returns on index returns
# ---------------------------------------------------------------------------


def beta_N(panel: pd.DataFrame, index_df: pd.DataFrame,
            lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    rets = _safe_log_returns(_close(panel))
    # Guard: index close of 0 → NaN log-return (zero-denom #182).
    irets = np.log(index_df["close"].replace(0, np.nan)).diff()
    tickers = panel.index.get_level_values("ticker").unique()

    out: dict[str, pd.Series] = {}
    for N in lookbacks:
        chunks = []
        for t in tickers:
            r = rets.xs(t, level="ticker")
            ir = irets.reindex(r.index)
            cov = r.rolling(N, min_periods=N).cov(ir)
            var = ir.rolling(N, min_periods=N).var()
            beta = cov / var.replace(0, np.nan)
            beta.index = pd.MultiIndex.from_product([beta.index, [t]], names=["date", "ticker"])
            chunks.append(beta)
        out[f"beta_{N}"] = pd.concat(chunks).sort_index()
    return pd.DataFrame(out).reindex(panel.index)


# ---------------------------------------------------------------------------
# F11 — Parkinson + Garman-Klass range vol
# ---------------------------------------------------------------------------


def range_vol(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
               annualization: int = 250) -> pd.DataFrame:
    h, l, o, c = panel["high"], panel["low"], panel["open"], panel["close"]
    sqrt_ann = float(np.sqrt(annualization))

    # Guard: low of 0 (halted/sparse ticker) makes np.log(h/l) → ±inf; route
    # to NaN so the backend's missing path handles it (zero-denom #182). The
    # h/l denominator is `l`; `c/o` already had its own guard.
    l_safe = l.replace(0, np.nan)
    ln_hl_sq = (np.log(h / l_safe)) ** 2
    out = {}
    for N in lookbacks:
        park_var = _roll(ln_hl_sq, N, "mean")
        out[f"parkinson_{N}"] = np.sqrt(park_var.clip(lower=0.0) / (4.0 * np.log(2.0))) * sqrt_ann

    ln_hl = np.log(h / l_safe)
    ln_co = np.log(c / o.replace(0, np.nan))
    gk_term = 0.5 * (ln_hl ** 2) - (2 * np.log(2.0) - 1.0) * (ln_co ** 2)
    for N in lookbacks:
        gk_var = _roll(gk_term, N, "mean")
        out[f"garman_klass_{N}"] = np.sqrt(gk_var.clip(lower=0.0)) * sqrt_ann
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# F12 — distance to SMA
# ---------------------------------------------------------------------------


def sma_distance_N(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    close = _close(panel)
    out = {}
    for N in lookbacks:
        sma = _roll(close, N, "mean")
        out[f"sma_distance_{N}"] = close / sma.replace(0, np.nan) - 1.0
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# F13 — vol regime
# ---------------------------------------------------------------------------


def vol_regime(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
                annualization: int = 250,
                *,
                rvol: pd.DataFrame | None = None) -> pd.DataFrame:
    """``rvol`` may be supplied by callers that have already computed F4
    (``realized_vol_N``). Default-None preserves the standalone-call contract.
    """
    if rvol is None:
        rvol = realized_vol_N(panel, lookbacks, annualization=annualization)
    out = {}
    for N in lookbacks:
        v = rvol[f"realized_vol_{N}"]
        # Guard: prior realized_vol of 0 (e.g. a flat-price window with zero
        # log-return variance) → NaN, not ±inf (zero-denom #182).
        out[f"vol_change_{N}"] = v / _shift(v, N).replace(0, np.nan) - 1.0
        out[f"vol_of_vol_{N}"] = _roll(v, N, "std")
        # Vectorized trailing percentile rank — see _rolling_pct_rank
        # (bit-identical to the prior rolling().apply, native Cython).
        out[f"vol_pct_{N}"] = _per_ticker(
            v, lambda s, n=N: _rolling_pct_rank(s, n),
        )
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# F14 — cross-sectional rank + z (returns + vol)
# ---------------------------------------------------------------------------


def cross_sectional_rank_z(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
                             annualization: int = 250,
                             *,
                             sr: pd.DataFrame | None = None,
                             rv: pd.DataFrame | None = None) -> pd.DataFrame:
    """``sr`` / ``rv`` may be supplied by callers that have already computed
    F2 (``stock_return_N``) / F4 (``realized_vol_N``). Default-None preserves
    the standalone-call contract.
    """
    if sr is None:
        sr = stock_return_N(panel, lookbacks)
    if rv is None:
        rv = realized_vol_N(panel, lookbacks, annualization=annualization)
    out = {}
    for N in lookbacks:
        for src_col, prefix in (
            (sr[f"stock_return_{N}"], "return"),
            (rv[f"realized_vol_{N}"], "vol"),
        ):
            g = src_col.groupby(level="date")
            out[f"{prefix}_xs_rank_{N}"] = g.rank(pct=True)
            mean = g.transform("mean")
            std = g.transform("std").replace(0, np.nan)
            out[f"{prefix}_xs_zscore_{N}"] = (src_col - mean) / std
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# F15 — calendar
# ---------------------------------------------------------------------------


def calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    dates = panel.index.get_level_values("date")
    dow = dates.weekday.values.astype(float)
    dom = dates.day.values.astype(float)
    moy = dates.month.values.astype(float)

    out = pd.DataFrame(index=panel.index)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    out["dom_sin"] = np.sin(2 * np.pi * dom / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * dom / 31.0)
    out["moy_sin"] = np.sin(2 * np.pi * moy / 12.0)
    out["moy_cos"] = np.cos(2 * np.pi * moy / 12.0)

    # India binary flags — coarse first-pass windows; refinable in v1.1.
    month = np.asarray(dates.month)
    day = np.asarray(dates.day)
    out["fiscal_year_end_week"] = ((month == 3) & (day >= 24)).astype(float)
    out["budget_week"] = ((month == 2) & (day <= 7)).astype(float)
    out["diwali_week"] = (
        ((month == 10) & (day >= 25)) | ((month == 11) & (day <= 7))
    ).astype(float)
    fomc_months = np.array([1, 3, 5, 6, 7, 9, 11, 12])
    out["fomc_week"] = (
        np.isin(month, fomc_months) & np.isin(day, np.arange(15, 22))
    ).astype(float)
    return out


# ---------------------------------------------------------------------------
# F16 — signed-days-outside-band
# ---------------------------------------------------------------------------


def f16_underlying(panel: pd.DataFrame, lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
                    annualization: int = 250,
                    *,
                    rvol: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 12 native F16 z-scores (6 stock_return_zscore + 6 realized_vol_zscore).

    ``rvol`` may be supplied by callers that have already computed F4
    (``realized_vol_N``). Default-None preserves the standalone-call contract.
    """
    close = _close(panel)
    if rvol is None:
        rvol = realized_vol_N(panel, lookbacks, annualization=annualization)
    out = {}
    for N in lookbacks:
        # Guard: prior close of 0 → NaN, mirroring stock_return_N (#182).
        sr = close / _shift(close, N).replace(0, np.nan) - 1.0
        mean = _roll(sr, N, "mean")
        std = _roll(sr, N, "std")
        out[f"stock_return_zscore_{N}"] = (sr - mean) / std.replace(0, np.nan)

        rv = rvol[f"realized_vol_{N}"]
        rvmean = _roll(rv, N, "mean")
        rvstd = _roll(rv, N, "std")
        out[f"realized_vol_zscore_{N}"] = (rv - rvmean) / rvstd.replace(0, np.nan)
    return pd.DataFrame(out, index=panel.index)


def _signed_days_outside_band_one(z_values: np.ndarray, sigma: float) -> np.ndarray:
    """Compute the signed-days-outside-band statistic for one z-series.

    Value at t:
      - ``0`` if ``z[t]`` is inside ``(-sigma, +sigma)``
      - ``+k`` if z has been ``>= +sigma`` for k consecutive rows ending at t
      - ``-k`` if z has been ``<= -sigma`` for k consecutive rows ending at t
      Resets to 0 the moment z re-enters the band.
    Sign convention is current-side (option A).

    NaN policy: any NaN in ``z_values`` propagates as NaN in the output and
    resets the internal streak counter and direction to 0 — a missing
    observation breaks the run. The next valid row therefore starts a fresh
    streak of length 1 (not a continuation). This is deliberate: a missing
    bar means we don't know whether the band was breached, so claiming
    continuity across the gap would overstate the signal (PR #8 review,
    Low 6).

    Vectorized (numpy run-length) — bit-identical to the prior element-wise
    Python loop (verified exact over 433 random + edge cases: long runs, sign
    flips, band re-entry, NaN resets, ties at the band edge), but native instead
    of a Python loop per (column, sigma, ticker). This loop was the largest
    single self-time hotspot of the build (~89s; GBDTPERF profile). The reset
    semantics fall out of treating each distinct state as a run boundary: a
    state change (+1↔−1, in/out of band) OR a NaN (sentinel state) starts a
    fresh run, so the post-gap value restarts at streak 1.
    """
    z = np.asarray(z_values, dtype=float)
    n = z.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    isnan = np.isnan(z)
    state = np.zeros(n, dtype=np.int64)
    state[z >= sigma] = 1
    state[z <= -sigma] = -1
    # NaN → sentinel state 2 so it always breaks the run (and outputs NaN below).
    key = state.copy()
    key[isnan] = 2
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = key[1:] != key[:-1]
    idx = np.arange(n)
    # run start = most recent change index; run length = idx − run_start + 1.
    run_start = np.maximum.accumulate(np.where(change, idx, 0))
    out = (state * (idx - run_start + 1)).astype(float)  # 0 where state == 0
    out[isnan] = np.nan
    return out


def signed_days_outside_band_meta(
    z_columns: pd.DataFrame,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Apply :func:`_signed_days_outside_band_one` to each (column, sigma)
    pair, per ticker.

    Input: a ``(date, ticker)``-indexed DataFrame of z-scored underlyings.
    Output column naming: ``<base>_outside_band_<sigma>``.

    ONE per-ticker ``groupby.apply`` over the whole z-frame computes all
    ``len(columns) × len(sigmas)`` outputs in a single split/apply/combine, rather
    than that many separate ``_per_ticker`` passes (93 for the standard 31-col ×
    3-σ layer). The per-ticker streak function is unchanged and runs on the same
    per-ticker chronological z-series, so the output is bit-identical; this only
    collapses the ~93 split/combine passes into one. V1.6 Phase 1a — measured 3.6×
    on the sp500 daily slice (the split/combine overhead, not the streak math, is
    ~54% of the build; see ``docs/gbdt/V1.6_incremental_feature_cache_plan.md``).
    Column order (col-major: for col, for sigma) is preserved.
    """
    labels = [
        (sg, str(int(sg)) if sg == int(sg) else str(sg).replace(".", "p"))
        for sg in sigmas
    ]
    cols = list(z_columns.columns)

    def _one_ticker(sub: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for col in cols:
            v = sub[col].to_numpy()
            for sg, label in labels:
                out[f"{col}_outside_band_{label}"] = _signed_days_outside_band_one(v, sg)
        return pd.DataFrame(out, index=sub.index)

    return z_columns.groupby(level="ticker", group_keys=False).apply(_one_ticker)


def f16_meta_underlying_columns(
    panel: pd.DataFrame,
    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
    annualization: int = 250,
    *,
    f16_nat: pd.DataFrame | None = None,
    f7: pd.DataFrame | None = None,
    xs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the 31 z-score columns F16's meta layer operates on.

    Breakdown (31 underlyings × 3 sigmas = 93 meta cols; total F16 = 12 + 93 = 105):
      6 F7 dollar_move_zscore_N (rolling per-stock)
      1 F7 dollar_move_xs_zscore (cross-sectional)
      6 F14 return_xs_zscore_N
      6 F14 vol_xs_zscore_N
      6 F16 native stock_return_zscore_N
      6 F16 native realized_vol_zscore_N

    ``f16_nat`` / ``f7`` / ``xs`` may be supplied by callers that have already
    computed those families (e.g. ``build_feature_matrix`` orchestrating F7,
    F14, and F16 in one walk). Default-None preserves the standalone-call
    contract — the function is self-contained when invoked directly.
    """
    if f16_nat is None:
        f16_nat = f16_underlying(panel, lookbacks, annualization=annualization)
    if f7 is None:
        f7 = volume_family(panel, lookbacks)
    if xs is None:
        xs = cross_sectional_rank_z(panel, lookbacks, annualization=annualization)

    keep_f7 = [f"dollar_move_zscore_{N}" for N in lookbacks] + ["dollar_move_xs_zscore"]
    keep_xs = [f"return_xs_zscore_{N}" for N in lookbacks] + [f"vol_xs_zscore_{N}" for N in lookbacks]

    out = pd.concat(
        [f7[keep_f7], xs[keep_xs], f16_nat],
        axis=1,
    )
    return out


# ---------------------------------------------------------------------------
# F17 — FRED macro-regime context (date-broadcast; opt-in, NOT in "all")
# ---------------------------------------------------------------------------

# Daily market-observed FRED series only. Each is an end-of-day value used with
# a 1-trading-day lag (strictly causal, C1). Monthly/quarterly FRED series
# (CPI/GDP/payrolls) are deliberately EXCLUDED — their publication lag (release
# date >> observation date) is a look-ahead trap the (date,value) cache can't
# yet express. Date-broadcast values are constant across tickers on a date:
# allowed cross-sectional regime context, NOT a forbidden per-asset constant.
MACRO_SERIES: tuple[str, ...] = (
    "DGS10",        # 10Y Treasury yield
    "T10Y2Y",       # 10Y-2Y curve slope
    "DGS3MO",       # 3M Treasury yield
    "DFF",          # effective fed funds
    "BAMLH0A0HYM2", # HY credit OAS
    "T10YIE",       # 10Y breakeven inflation
    "DFII10",       # 10Y TIPS real yield
    "VIXCLS",       # VIX
    "DTWEXBGS",     # broad trade-weighted USD
)
_MACRO_CHG_LOOKBACKS: tuple[int, ...] = (20, 60)
_MACRO_Z_LOOKBACKS: tuple[int, ...] = (60, 120)


def fred_macro_features(
    macro_df: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    series: Iterable[str] = MACRO_SERIES,
    chg_lookbacks: Iterable[int] = _MACRO_CHG_LOOKBACKS,
    z_lookbacks: Iterable[int] = _MACRO_Z_LOOKBACKS,
) -> pd.DataFrame:
    """F17 — macro-regime context from daily FRED series, broadcast to panel.

    ``macro_df`` is date-indexed with one column per FRED series id. For each
    requested series we align it to the panel's trading-day calendar
    (forward-filled — causal, carries the last known value onto trading days the
    macro series lacks, e.g. a bond-market holiday), lag it **1 trading day** so
    origin ``t`` sees only the value as of ``t-1``, then emit per series:
      - ``macro_<id>_level``    the lagged level
      - ``macro_<id>_chg_<N>``  absolute change over N trading days
      - ``macro_<id>_z_<N>``    trailing rolling z-score (regime position)

    Every transform is strictly trailing (zero look-ahead, C1) — verified by the
    macro-perturbation leakage test. Series absent from ``macro_df`` are skipped
    (a missing/unseeded series degrades gracefully rather than crashing).
    """
    panel_dates = panel.index.get_level_values("date").unique().sort_values()
    cols: dict[str, pd.Series] = {}
    for sid in series:
        if sid not in macro_df.columns:
            continue
        # Align to the trading-day calendar, ffill macro gaps (causal), lag 1d.
        s = macro_df[sid].reindex(panel_dates).ffill().shift(1)
        cols[f"macro_{sid}_level"] = s
        for N in chg_lookbacks:
            cols[f"macro_{sid}_chg_{N}"] = s - s.shift(N)
        for N in z_lookbacks:
            mean = s.rolling(N, min_periods=N).mean()
            # Zero-denom guard (#182): a flat window (std==0, e.g. DFF held
            # constant) would emit ±inf; route to NaN like every other family.
            std = s.rolling(N, min_periods=N).std().replace(0.0, np.nan)
            cols[f"macro_{sid}_z_{N}"] = (s - mean) / std
    feat = pd.DataFrame(cols, index=panel_dates)
    return _broadcast_index_to_panel(feat, panel)


# ---------------------------------------------------------------------------
# F18 — fundamentals valuation ratios (per-(date,ticker); opt-in, NOT in "all")
# ---------------------------------------------------------------------------

# Inverse valuation yields (finite + signed, unlike raw PE) from the point-in-
# time valuation panel. NOT date-broadcast — each varies per ticker, so F18
# joins on the panel's (date, ticker) index (by symbol) rather than broadcasting.
FUND_YIELDS: tuple[str, ...] = ("earnings_yield", "sales_yield", "fcf_yield")
_FUND_GROWTH_LOOKBACK = 252   # ~1y TTM YoY
_FUND_CHG_LOOKBACK = 63       # ~1 quarter re-rating


def fundamentals_features(
    fund_df: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    yields: Iterable[str] = FUND_YIELDS,
    growth_lookback: int = _FUND_GROWTH_LOOKBACK,
    chg_lookback: int = _FUND_CHG_LOOKBACK,
) -> pd.DataFrame:
    """F18 — point-in-time valuation ratios joined per (date, ticker).

    ``fund_df`` is a ``(date, symbol)``-indexed frame from the ``valuation``
    panel with the yield columns + ``revenue_ttm``. Panel tickers
    (``NASDAQ:AAPL``) are mapped to their symbol (``AAPL``) to align. Emits:
      - ``fund_<yield>``            the point-in-time yield level
      - ``fund_<yield>_xs_rank``    its cross-sectional percentile across the
                                    panel on that date (the F14 idiom)
      - ``fund_rev_ttm_yoy``        log TTM-revenue growth vs ~1y ago (+ xs_rank)
      - ``fund_earnings_yield_chg_63``  63-td change (cheapening / re-rating)

    Already causal (C1): the valuation value at ``t`` uses only filings with
    ``filed_date ≤ t`` and the close at ``t`` (same convention as the F2 price
    features — no extra lag); the growth/change transforms are strictly
    trailing, and the cross-sectional ranks are same-date. Verified by the
    fundamentals-perturbation leakage test. Missing columns degrade gracefully.
    """
    dates = panel.index.get_level_values("date")
    symbols = panel.index.get_level_values("ticker").str.split(":").str[-1]
    join_idx = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    aligned = fund_df.sort_index().reindex(join_idx)
    aligned.index = panel.index  # relabel (date, symbol) → (date, ticker)

    out: dict[str, pd.Series] = {}
    for y in yields:
        if y not in aligned.columns:
            continue
        s = aligned[y]
        out[f"fund_{y}"] = s
        out[f"fund_{y}_xs_rank"] = s.groupby(level="date").rank(pct=True)

    if "revenue_ttm" in aligned.columns:
        rev = aligned["revenue_ttm"]
        rev_prev = rev.groupby(level="ticker").shift(growth_lookback)
        # log YoY growth; guard non-positive so no -inf/NaN-of-log surprises
        ratio = (rev / rev_prev).where((rev > 0) & (rev_prev > 0))
        g = np.log(ratio)
        out["fund_rev_ttm_yoy"] = g
        out["fund_rev_ttm_yoy_xs_rank"] = g.groupby(level="date").rank(pct=True)

    if "earnings_yield" in aligned.columns:
        ey = aligned["earnings_yield"]
        out["fund_earnings_yield_chg_63"] = (
            ey - ey.groupby(level="ticker").shift(chg_lookback)
        )

    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# F1–F16 only. F17 (macro) is intentionally OMITTED so the "all" token — and
# every existing spec / committed model that relies on it — stays byte-identical.
# Macro is opt-in via the "all_macro" token or an explicit families list.
_ALL_FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
                  "F10", "F11", "F12", "F13", "F14", "F15", "F16")


def build_feature_matrix(
    panel: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
    annualization: int = 250,
    families: str | list[str] = "all",
    exclude: list[str] | None = None,
    macro_df: pd.DataFrame | None = None,
    fund_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the candidate feature matrix.

    With defaults the result has ``EXPECTED_TOTAL_COLS == 279`` columns;
    pruning via ``families`` or ``exclude`` reduces that. F6/F6b and F9/F9b
    are siblings folded under their parent ``F6`` / ``F9`` family token —
    selecting ``F6`` brings both drawdown and runup; selecting ``F9`` brings
    both index drawdown and index runup.

    ``families`` accepts ``"all"`` (default) or any iterable of family tokens
    drawn from ``F1, F2, F3, F4, F5, F6, F7, F8, F10, F11, F12, F13, F14,
    F15, F16``. ``F9`` is bundled into ``F6`` for selection purposes (since
    F6 already implies index drawdown/runup via the spec table).

    ``exclude`` is a list of glob patterns matched against final column
    names (e.g. ``"volume_ratio_*"``).
    """
    if families == "all" or families is None:
        sel = set(_ALL_FAMILIES)
    elif families == "all_macro":
        # All baseline families PLUS the opt-in macro family. "all" is left
        # untouched above so existing specs/models are unaffected.
        sel = set(_ALL_FAMILIES) | {"F17"}
    elif families == "all_fundamentals":
        # Baseline families PLUS the opt-in fundamentals family (F18). "all"
        # untouched — existing specs/models unaffected.
        sel = set(_ALL_FAMILIES) | {"F18"}
    else:
        sel = set(families)

    pieces: list[pd.DataFrame] = []

    # Holds completed family results so later families can reuse upstream
    # computes (F3 ← F1+F2; F13 ← F4; F14 ← F2+F4; F16 ← F4+F7+F14) instead
    # of recomputing them. F7 + F4 + cross_sectional_rank_z + realized_vol
    # are the heavy hitters on sp500 (3.6M rows × 6 lookbacks). When a
    # selected lambda fires, ``_ALL_FAMILIES`` ordering guarantees its
    # dependencies are already in ``computed``; if a dependency family is
    # de-selected, the ``.get()`` returns ``None`` and the helper falls
    # back to its standalone codepath.
    computed: dict[str, pd.DataFrame] = {}

    # Build the (family, callable) plan so progress logging walks a
    # known total. ``_ALL_FAMILIES`` is the canonical order; preserving it
    # keeps the log lines deterministic across runs.
    plan: list[tuple[str, callable]] = []
    if "F1" in sel:
        plan.append(("F1", lambda: index_return_N(index_df, panel, lookbacks)))
    if "F2" in sel:
        plan.append(("F2", lambda: stock_return_N(panel, lookbacks)))
    if "F3" in sel:
        plan.append(("F3", lambda: rel_strength_N(
            panel, index_df, lookbacks,
            sr=computed.get("F2"), ir=computed.get("F1"),
        )))
    if "F4" in sel:
        plan.append(("F4", lambda: realized_vol_N(panel, lookbacks, annualization)))
    if "F5" in sel:
        plan.append(("F5", lambda: index_vol_N(index_df, panel, lookbacks, annualization)))
    if "F6" in sel:
        plan.append(("F6", lambda: pd.concat([
            drawdown_N(panel, lookbacks),
            runup_N(panel, lookbacks),
            index_drawdown_N(index_df, panel, lookbacks),
            index_runup_N(index_df, panel, lookbacks),
        ], axis=1)))
    if "F7" in sel:
        plan.append(("F7", lambda: volume_family(panel, lookbacks)))
    if "F8" in sel:
        plan.append(("F8", lambda: higher_moments(panel, lookbacks)))
    if "F10" in sel:
        plan.append(("F10", lambda: beta_N(panel, index_df, lookbacks)))
    if "F11" in sel:
        plan.append(("F11", lambda: range_vol(panel, lookbacks, annualization)))
    if "F12" in sel:
        plan.append(("F12", lambda: sma_distance_N(panel, lookbacks)))
    if "F13" in sel:
        plan.append(("F13", lambda: vol_regime(
            panel, lookbacks, annualization, rvol=computed.get("F4"),
        )))
    if "F14" in sel:
        plan.append(("F14", lambda: cross_sectional_rank_z(
            panel, lookbacks, annualization,
            sr=computed.get("F2"), rv=computed.get("F4"),
        )))
    if "F15" in sel:
        plan.append(("F15", lambda: calendar_features(panel)))
    if "F16" in sel:
        def _build_f16() -> pd.DataFrame:
            f16_nat = f16_underlying(
                panel, lookbacks, annualization, rvol=computed.get("F4"),
            )
            underlyings = f16_meta_underlying_columns(
                panel, lookbacks, annualization,
                f16_nat=f16_nat,
                f7=computed.get("F7"),
                xs=computed.get("F14"),
            )
            meta = signed_days_outside_band_meta(
                underlyings, sigmas=(1.0, 2.0, 3.0),
            )
            return pd.concat([f16_nat, meta], axis=1)
        plan.append(("F16", _build_f16))
    if "F17" in sel:
        if macro_df is None:
            raise ValueError(
                "F17 (macro) selected but macro_df is None — the runner must "
                "fetch the FRED macro panel and pass macro_df when "
                "families='all_macro' or 'F17' is in the families list."
            )
        plan.append(("F17", lambda: fred_macro_features(macro_df, panel)))
    if "F18" in sel:
        if fund_df is None:
            raise ValueError(
                "F18 (fundamentals) selected but fund_df is None — the runner "
                "must load the valuation panel and pass fund_df when "
                "families='all_fundamentals' or 'F18' is in the families list."
            )
        plan.append(("F18", lambda: fundamentals_features(fund_df, panel)))

    total = len(plan)
    t_start = time.time()
    last_log_time = t_start
    any_emit = False
    for i, (fam, fn) in enumerate(plan, start=1):
        result = fn()
        computed[fam] = result
        pieces.append(result)
        now = time.time()
        # Time-based throttle: emit at most once per
        # _FEATURES_PROGRESS_THROTTLE_SEC seconds. To guarantee at least
        # one progress line per run (so smoketests / fast panels still
        # produce evidence the stage advanced), force the final-boundary
        # emit ONLY when no prior line was emitted — otherwise the
        # throttle takes precedence (a 35s F1 followed by instantaneous
        # F2..F15 emits exactly one line, not two).
        throttle_ok = (now - last_log_time) >= _FEATURES_PROGRESS_THROTTLE_SEC
        force_final = (i == total and not any_emit)
        if throttle_ok or force_final:
            print(
                f"[features] family={fam} step={i}/{total} "
                f"elapsed={now - t_start:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            last_log_time = now
            any_emit = True

    out = pd.concat(pieces, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    if exclude:
        keep = [c for c in out.columns
                if not any(fnmatch.fnmatch(c, pat) for pat in exclude)]
        out = out[keep]

    # Hard guard (#182): features.py is contractually inf-free at source
    # (every ratio/division family routes zero-denom → NaN). If a future
    # family regresses, fail loudly here rather than letting model._sanitize_nonfinite
    # silently mask it downstream.
    inf_mask = np.isinf(out.values)
    if inf_mask.any():
        per_col = inf_mask.sum(axis=0)
        offenders = sorted(
            ((out.columns[j], int(per_col[j])) for j in range(out.shape[1]) if per_col[j] > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        raise AssertionError(
            "build_feature_matrix emitted ±inf — zero-denominator regression in "
            "features.py; band-aid in model._sanitize_nonfinite would mask it; "
            f"fix the family at source. Offending columns: {offenders[:10]}"
        )

    return out


__all__ = [
    "DEFAULT_LOOKBACKS",
    "EXPECTED_TOTAL_COLS",
    "stock_return_N",
    "index_return_N",
    "rel_strength_N",
    "realized_vol_N",
    "index_vol_N",
    "drawdown_N",
    "runup_N",
    "index_drawdown_N",
    "index_runup_N",
    "volume_family",
    "higher_moments",
    "beta_N",
    "range_vol",
    "sma_distance_N",
    "vol_regime",
    "cross_sectional_rank_z",
    "calendar_features",
    "f16_underlying",
    "f16_meta_underlying_columns",
    "signed_days_outside_band_meta",
    "build_feature_matrix",
]
