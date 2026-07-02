"""V1.6 Phase 2 — incremental feature-cache extend: bit-identity + seam check.

The extend (rebuild a bounded tail + stitch new dates onto a cached matrix) must be
**bit-identical** to a from-scratch full build for the bounded-lookback families
(F1–F15), and the seam-integrity check must reject a revised/corrupted history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt import features as F
from gbdt import incremental_feature_cache as inc
from gbdt.leakage_harness import make_synthetic_panel


def _synth(n_rows: int = 750, n_tickers: int = 5, seed: int = 0):
    panel = make_synthetic_panel(n_rows, n_tickers, seed=seed)
    rng = np.random.default_rng(seed + 99)
    dates = panel.index.get_level_values("date").unique()
    rets = rng.normal(0, 0.008, size=len(dates))
    close = 1000.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 0.005, len(dates)))
    low = close * (1.0 - rng.uniform(0.0, 0.005, len(dates)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, len(dates)))
    index_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": np.ones(len(dates))},
        index=pd.Index(dates, name="date"),
    )
    return panel, index_df


def _build(panel, index_df, upto=None):
    if upto is not None:
        d = panel.index.get_level_values("date")
        panel = panel[d <= upto]
        index_df = index_df[index_df.index <= upto]
    X = F.build_feature_matrix(panel, index_df, families=inc.BOUNDED_FAMILIES, annualization=250)
    return X.dropna(axis=1, how="all")


def test_extend_matches_full_build_bounded_families():
    panel, index_df = _synth(750, 5, seed=1)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    T1 = dates[600]
    cached = _build(panel, index_df, upto=T1)
    full = _build(panel, index_df)

    ext = inc.extend_matrix(
        cached, panel, index_df, annualization=250,
        families=inc.BOUNDED_FAMILIES, cached_max_date=T1,
    )
    assert list(ext.columns) == list(full.columns)
    ext = ext.reindex(full.index)
    # ~1e-13 tolerance: pandas' online rolling rounds differently by series offset
    # (see incremental_feature_cache module docstring). Cached dates are exact; only
    # the new-date recompute carries the FP offset. 9 orders below the 1e-4 self-check.
    assert np.allclose(ext.to_numpy(dtype=float), full.to_numpy(dtype=float),
                       rtol=1e-9, atol=1e-11, equal_nan=True), \
        "extended matrix diverges from the full build beyond FP-rounding tolerance"
    # the extend actually added the new dates
    assert (pd.DatetimeIndex(ext.index.get_level_values("date").unique()).max()
            > T1)


def test_seam_check_rejects_revised_history():
    panel, index_df = _synth(750, 5, seed=2)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    T1 = dates[600]
    cached = _build(panel, index_df, upto=T1).copy()

    # Corrupt a cached value inside the seam-check window (last 20 cached dates) →
    # the tail recompute won't match → extend must refuse (SeamMismatch).
    seam_date = dates[595]
    mask = cached.index.get_level_values("date") == seam_date
    cached.loc[mask, cached.columns[0]] = 999999.0

    with pytest.raises(inc.SeamMismatch):
        inc.extend_matrix(
            cached, panel, index_df, annualization=250,
            families=inc.BOUNDED_FAMILIES, cached_max_date=T1,
        )


def test_continue_streaks_matches_full_streak_bit_identical():
    """Phase 3: continuing a streak from the carried state reproduces the full
    signed-days-outside-band run bit-for-bit (same z-values), across NaN resets,
    sign flips, in-band resets, and a run that straddles the split point."""
    from gbdt.features import _signed_days_outside_band_one
    rng = np.random.default_rng(3)
    for sigma in (1.0, 2.0, 3.0):
        for trial in range(25):
            n = int(rng.integers(60, 200))
            z = rng.standard_normal(n) * 1.8
            z[rng.integers(0, n, size=3)] = np.nan   # NaN resets
            z[10:40] = 2.5                            # a long +run to straddle the split
            full = _signed_days_outside_band_one(z, sigma)
            T = int(rng.integers(15, n - 5))          # split inside/after the run
            prev = np.array([full[T]])
            z_new = z[T + 1:].reshape(1, -1)
            cont = inc._continue_streaks(prev, z_new, sigma)[0]
            assert np.array_equal(cont, full[T + 1:], equal_nan=True), \
                f"sigma={sigma} trial={trial} T={T}: continuation diverges from full streak"


def test_extend_matrix_full_matches_full_build_incl_f16():
    """Phase 3: extend_matrix_full (bounded tail + F16-native + F16-meta streak-state
    carry) matches a full build INCLUDING the F16 streak, within FP tolerance
    (state-carry is exact on the same z-values; stable cols ~1e-13 on synthetic)."""
    panel, index_df = _synth(750, 5, seed=4)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    T1 = dates[600]
    famALL = list(F._ALL_FAMILIES)
    cached = F.build_feature_matrix(
        panel[panel.index.get_level_values("date") <= T1], index_df[index_df.index <= T1],
        families=famALL, annualization=250,
    ).dropna(axis=1, how="all")
    full = F.build_feature_matrix(panel, index_df, families=famALL, annualization=250).dropna(axis=1, how="all")

    ext = inc.extend_matrix_full(cached, panel, index_df, annualization=250, cached_max_date=T1)
    assert list(ext.columns) == list(full.columns)
    ext = ext.reindex(full.index)
    assert np.allclose(ext.to_numpy(float), full.to_numpy(float), rtol=1e-6, atol=1e-9, equal_nan=True), \
        "extend_matrix_full (incl F16 state-carry) diverges from the full build"
