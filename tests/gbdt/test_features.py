"""Stage 2 — feature pipeline tests.

Per-family causality (via the leakage harness from Stage 1), hand-computed
spot checks, and the headline 279-column-count assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt import features as F
from gbdt.leakage_harness import make_synthetic_panel, LeakageHarness


def _synth_panel_with_index(n_rows: int = 300, n_tickers: int = 3, seed: int = 0):
    panel = make_synthetic_panel(n_rows, n_tickers, seed=seed)
    # The index has its own random walk so it's not perfectly correlated.
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


# ---------------------------------------------------------------------------
# Hand-computed spot checks
# ---------------------------------------------------------------------------


def test_stock_return_5_matches_hand_formula():
    panel, _ = _synth_panel_with_index(50, 1, seed=1)
    sr = F.stock_return_N(panel, lookbacks=(5,))
    close = panel["close"]
    # At row 10, expect close[10]/close[5] - 1
    t = panel.index[10]
    t_minus_5 = panel.index[5]
    expected = close.loc[t] / close.loc[t_minus_5] - 1.0
    assert sr.loc[t, "stock_return_5"] == pytest.approx(expected, rel=1e-9)


def test_realized_vol_annualization():
    panel, _ = _synth_panel_with_index(60, 1, seed=2)
    rv = F.realized_vol_N(panel, lookbacks=(10,), annualization=250)
    # Last row's value should equal np.std of last 10 log returns * sqrt(250)
    close = panel["close"].droplevel("ticker")
    rets = np.log(close).diff()
    expected = rets.iloc[-10:].std() * np.sqrt(250)
    actual = rv["realized_vol_10"].iloc[-1]
    assert actual == pytest.approx(expected, rel=1e-9)


def test_drawdown_uses_high_not_close():
    panel, _ = _synth_panel_with_index(40, 1, seed=3)
    dd = F.drawdown_N(panel, lookbacks=(5,))
    close = panel["close"]
    high = panel["high"]
    t = panel.index[20]
    # Find the rolling max over rows [16..20] of HIGH
    window_idx = panel.index[16:21]
    expected = close.loc[t] / high.loc[window_idx].max() - 1.0
    assert dd.loc[t, "drawdown_5"] == pytest.approx(expected, rel=1e-9)


def test_sma_distance_matches_formula():
    panel, _ = _synth_panel_with_index(30, 1, seed=4)
    sd = F.sma_distance_N(panel, lookbacks=(5,))
    close = panel["close"]
    t = panel.index[15]
    window_idx = panel.index[11:16]
    expected = close.loc[t] / close.loc[window_idx].mean() - 1.0
    assert sd.loc[t, "sma_distance_5"] == pytest.approx(expected, rel=1e-9)


def test_signed_days_outside_band_streaks():
    # Construct a z series with known streak pattern
    import numpy as np
    z = pd.Series(
        [0.0, 0.5, 2.1, 2.2, 2.3, 0.4, -2.1, -2.2, 0.0, 2.5],
        index=pd.RangeIndex(10),
    )
    out = F._signed_days_outside_band_one(z.values, sigma=2.0)
    assert list(out) == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, -1.0, -2.0, 0.0, 1.0]


def test_signed_days_outside_band_meta_batched_matches_per_column_reference():
    """V1.6 Phase 1a: the batched one-pass ``signed_days_outside_band_meta``
    (single per-ticker ``groupby.apply`` over the frame) must be bit-identical to
    the per-(column, sigma) ``_per_ticker`` reference it replaced — multi-ticker,
    with NaN rows exercising the streak reset."""
    rng = np.random.default_rng(7)
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    Z = pd.DataFrame(
        rng.standard_normal((len(idx), 4)) * 1.5,
        index=idx, columns=["za", "zb", "zc", "zd"],
    ).sort_index()
    Z.iloc[::37] = np.nan  # whole-row NaNs exercise the streak reset

    sigmas = (1.0, 2.0, 3.0)
    got = F.signed_days_outside_band_meta(Z, sigmas=sigmas)

    # Reference = the pre-Phase-1a per-(column, sigma) implementation.
    ref = {}
    for col in Z.columns:
        for sg in sigmas:
            label = str(int(sg)) if sg == int(sg) else str(sg).replace(".", "p")
            ref[f"{col}_outside_band_{label}"] = Z[col].groupby(
                level="ticker", group_keys=False
            ).apply(lambda s, g=sg: pd.Series(
                F._signed_days_outside_band_one(s.to_numpy(), g), index=s.index))
    ref = pd.DataFrame(ref)

    assert list(got.columns) == list(ref.columns)
    got2, ref2 = got.align(ref, axis=0)
    assert np.array_equal(got2.to_numpy(), ref2.to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------
# Causality (leakage harness)
# ---------------------------------------------------------------------------


def _make_index_for(panel: pd.DataFrame) -> pd.DataFrame:
    """Stable index DataFrame for the harness panel — index_df can't depend on
    the panel's leaked row (it's not in the panel's ticker set)."""
    dates = panel.index.get_level_values("date").unique()
    close = np.linspace(100.0, 150.0, len(dates))
    high = close * 1.001
    low = close * 0.999
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": np.ones(len(dates))},
        index=pd.Index(dates, name="date"),
    )


def _check_family(family_fn, *args, name: str):
    h = LeakageHarness(n_rows=120, n_tickers=2, leak_row=80)
    def wrapper(panel):
        return family_fn(panel, *args)
    report = h.check(wrapper)
    assert report.causal, f"{name} leaks: {report}"


def test_causality_stock_return():
    _check_family(F.stock_return_N, (5, 20), name="stock_return_N")


def test_causality_realized_vol():
    _check_family(F.realized_vol_N, (5, 20), 250, name="realized_vol_N")


def test_causality_drawdown_runup():
    _check_family(F.drawdown_N, (5, 20), name="drawdown_N")
    _check_family(F.runup_N, (5, 20), name="runup_N")


def test_causality_higher_moments():
    _check_family(F.higher_moments, (10,), name="higher_moments")


def test_causality_sma_distance():
    _check_family(F.sma_distance_N, (5, 20), name="sma_distance_N")


def test_causality_vol_regime():
    _check_family(F.vol_regime, (5, 20), 250, name="vol_regime")


def test_causality_volume_family():
    _check_family(F.volume_family, (5, 20), name="volume_family")


def test_causality_range_vol():
    _check_family(F.range_vol, (5, 20), 250, name="range_vol")


def test_causality_cross_sectional_rank_z():
    _check_family(F.cross_sectional_rank_z, (5, 20), 250, name="cross_sectional_rank_z")


def test_causality_f16_underlying():
    _check_family(F.f16_underlying, (5, 20), 250, name="f16_underlying")


def test_causality_calendar():
    # Calendar is purely date-driven and trivially causal, but verify anyway.
    h = LeakageHarness(n_rows=120, n_tickers=2, leak_row=80)
    report = h.check(F.calendar_features)
    assert report.causal


def test_causality_signed_days_band_meta():
    """The band-meta layer composes z-scored underlyings; harness end-to-end."""
    h = LeakageHarness(n_rows=120, n_tickers=2, leak_row=80)
    def pipe(panel):
        underlyings = F.f16_meta_underlying_columns(panel, lookbacks=(5, 20), annualization=250)
        return F.signed_days_outside_band_meta(underlyings, sigmas=(1.0, 2.0))
    report = h.check(pipe)
    assert report.causal, f"band-meta leak: {report}"


# ---------------------------------------------------------------------------
# Cross-sectional shape sanity
# ---------------------------------------------------------------------------


def test_cross_sectional_rank_z_per_date_stats():
    panel, _ = _synth_panel_with_index(60, 4, seed=5)
    xs = F.cross_sectional_rank_z(panel, lookbacks=(5,), annualization=250)
    # Pick a date well past the lookback so ranks are populated
    a_date = panel.index.get_level_values("date").unique()[20]
    col = "return_xs_rank_5"
    vals = xs.xs(a_date, level="date")[col].dropna()
    assert len(vals) > 0
    # Ranks (pct=True) should be in (0, 1]
    assert vals.min() > 0.0
    assert vals.max() <= 1.0


# ---------------------------------------------------------------------------
# Total column count
# ---------------------------------------------------------------------------


def test_build_feature_matrix_total_col_count():
    panel, idx = _synth_panel_with_index(220, 3, seed=6)
    mat = F.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                  annualization=250, families="all")
    assert mat.shape[1] == F.EXPECTED_TOTAL_COLS, (
        f"expected {F.EXPECTED_TOTAL_COLS} cols; got {mat.shape[1]}: "
        f"{sorted(mat.columns.tolist())[:20]}..."
    )


def test_f16_meta_underlying_columns_threaded_kwargs_equal_standalone():
    """Threading pre-computed f16_nat/f7/xs into f16_meta_underlying_columns
    yields a result byte-identical to the standalone call. Guards the #198
    no-redundant-recompute patch — if the kwargs path diverged from the
    self-contained path the byte-identical feature cache contract breaks.
    """
    panel, idx = _synth_panel_with_index(220, 3, seed=11)
    standalone = F.f16_meta_underlying_columns(panel, lookbacks=F.DEFAULT_LOOKBACKS,
                                                 annualization=250)
    pre_nat = F.f16_underlying(panel, F.DEFAULT_LOOKBACKS, annualization=250)
    pre_f7 = F.volume_family(panel, F.DEFAULT_LOOKBACKS)
    pre_xs = F.cross_sectional_rank_z(panel, F.DEFAULT_LOOKBACKS, annualization=250)
    threaded = F.f16_meta_underlying_columns(panel, lookbacks=F.DEFAULT_LOOKBACKS,
                                              annualization=250,
                                              f16_nat=pre_nat, f7=pre_f7, xs=pre_xs)
    pd.testing.assert_frame_equal(standalone, threaded)


def test_build_feature_matrix_does_not_recompute_f7_f14_for_f16(monkeypatch):
    """When F7, F14, and F16 are all selected, build_feature_matrix must reuse
    F7's volume_family and F14's cross_sectional_rank_z results inside F16
    instead of recomputing them. Each underlying call must fire exactly once.
    Prevents regression of the #198 redundancy that cost 137 min/sp500-cell.
    """
    from gbdt import features as F_mod
    panel, idx = _synth_panel_with_index(220, 3, seed=12)

    n_vol = [0]
    n_xs = [0]
    real_vol = F_mod.volume_family
    real_xs = F_mod.cross_sectional_rank_z

    def spy_vol(*a, **kw):
        n_vol[0] += 1
        return real_vol(*a, **kw)

    def spy_xs(*a, **kw):
        n_xs[0] += 1
        return real_xs(*a, **kw)

    monkeypatch.setattr(F_mod, "volume_family", spy_vol)
    monkeypatch.setattr(F_mod, "cross_sectional_rank_z", spy_xs)

    F_mod.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                annualization=250,
                                families=["F7", "F14", "F16"])
    assert n_vol[0] == 1, f"volume_family fired {n_vol[0]} times; expected 1 (F16 must reuse F7)"
    assert n_xs[0] == 1, f"cross_sectional_rank_z fired {n_xs[0]} times; expected 1 (F16 must reuse F14)"


# ---------------------------------------------------------------------------
# Inter-family reuse (patches B-E): byte-equivalence of standalone vs threaded
# helpers, plus an aggregate spy-count test for F1/F2/F4 dependencies.
# ---------------------------------------------------------------------------


def test_rel_strength_threaded_kwargs_equal_standalone():
    """F3 rel_strength_N — threading pre-computed sr (F2) + ir (F1) must yield
    a result byte-identical to the standalone call. Guards patch E."""
    panel, idx = _synth_panel_with_index(220, 3, seed=21)
    standalone = F.rel_strength_N(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS)
    pre_sr = F.stock_return_N(panel, F.DEFAULT_LOOKBACKS)
    pre_ir = F.index_return_N(idx, panel, F.DEFAULT_LOOKBACKS)
    threaded = F.rel_strength_N(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                 sr=pre_sr, ir=pre_ir)
    pd.testing.assert_frame_equal(standalone, threaded)


def test_vol_regime_threaded_kwargs_equal_standalone():
    """F13 vol_regime — threading pre-computed rvol (F4) must yield a result
    byte-identical to the standalone call. Guards patch B."""
    panel, _ = _synth_panel_with_index(220, 3, seed=22)
    standalone = F.vol_regime(panel, lookbacks=F.DEFAULT_LOOKBACKS, annualization=250)
    pre_rvol = F.realized_vol_N(panel, F.DEFAULT_LOOKBACKS, annualization=250)
    threaded = F.vol_regime(panel, lookbacks=F.DEFAULT_LOOKBACKS, annualization=250,
                             rvol=pre_rvol)
    pd.testing.assert_frame_equal(standalone, threaded)


def test_cross_sectional_rank_z_threaded_kwargs_equal_standalone():
    """F14 cross_sectional_rank_z — threading pre-computed sr (F2) + rv (F4)
    must yield a result byte-identical to the standalone call. Guards patch C."""
    panel, _ = _synth_panel_with_index(220, 3, seed=23)
    standalone = F.cross_sectional_rank_z(panel, lookbacks=F.DEFAULT_LOOKBACKS,
                                           annualization=250)
    pre_sr = F.stock_return_N(panel, F.DEFAULT_LOOKBACKS)
    pre_rv = F.realized_vol_N(panel, F.DEFAULT_LOOKBACKS, annualization=250)
    threaded = F.cross_sectional_rank_z(panel, lookbacks=F.DEFAULT_LOOKBACKS,
                                         annualization=250, sr=pre_sr, rv=pre_rv)
    pd.testing.assert_frame_equal(standalone, threaded)


def test_f16_underlying_threaded_kwargs_equal_standalone():
    """F16 f16_underlying — threading pre-computed rvol (F4) must yield a
    result byte-identical to the standalone call. Guards patch D."""
    panel, _ = _synth_panel_with_index(220, 3, seed=24)
    standalone = F.f16_underlying(panel, lookbacks=F.DEFAULT_LOOKBACKS,
                                    annualization=250)
    pre_rvol = F.realized_vol_N(panel, F.DEFAULT_LOOKBACKS, annualization=250)
    threaded = F.f16_underlying(panel, lookbacks=F.DEFAULT_LOOKBACKS,
                                 annualization=250, rvol=pre_rvol)
    pd.testing.assert_frame_equal(standalone, threaded)


def test_build_feature_matrix_does_not_recompute_F1_F2_F4_dependencies(monkeypatch):
    """When all families are selected, build_feature_matrix must reuse F1's
    index_return_N, F2's stock_return_N, and F4's realized_vol_N inside
    downstream families (F3 ← F1+F2; F13 ← F4; F14 ← F2+F4; F16 ← F4 via
    f16_underlying). Each underlying call must fire exactly once. Prevents
    regression of the patch-B/C/D/E redundancies that cost ~15-25 min/sp500-cell.
    """
    from gbdt import features as F_mod
    panel, idx = _synth_panel_with_index(220, 3, seed=25)

    counters = {"stock_return_N": 0, "realized_vol_N": 0, "index_return_N": 0}
    real_sr = F_mod.stock_return_N
    real_rv = F_mod.realized_vol_N
    real_ir = F_mod.index_return_N

    def spy_sr(*a, **kw):
        counters["stock_return_N"] += 1
        return real_sr(*a, **kw)

    def spy_rv(*a, **kw):
        counters["realized_vol_N"] += 1
        return real_rv(*a, **kw)

    def spy_ir(*a, **kw):
        counters["index_return_N"] += 1
        return real_ir(*a, **kw)

    monkeypatch.setattr(F_mod, "stock_return_N", spy_sr)
    monkeypatch.setattr(F_mod, "realized_vol_N", spy_rv)
    monkeypatch.setattr(F_mod, "index_return_N", spy_ir)

    F_mod.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                annualization=250, families="all")

    # stock_return_N: F2 + (F3 reuses) + (F14 reuses) = 1
    assert counters["stock_return_N"] == 1, (
        f"stock_return_N fired {counters['stock_return_N']} times; "
        "expected 1 (F3 + F14 must reuse F2)"
    )
    # realized_vol_N: F4 + (F13 reuses) + (F14 reuses) + (F16 reuses via f16_underlying) = 1
    assert counters["realized_vol_N"] == 1, (
        f"realized_vol_N fired {counters['realized_vol_N']} times; "
        "expected 1 (F13 + F14 + F16/f16_underlying must reuse F4)"
    )
    # index_return_N: F1 + (F3 reuses) = 1
    assert counters["index_return_N"] == 1, (
        f"index_return_N fired {counters['index_return_N']} times; "
        "expected 1 (F3 must reuse F1)"
    )


def test_dollar_move_rank_raw_true_matches_raw_false():
    """The raw=True rewrite of _dollar_move_rank_N must produce numerically
    identical output to a raw=False reference. raw=True hands a numpy array
    to the callback instead of building a Series per window; per-window math
    is a scalar comparison + mean, so the two paths are mathematically
    equivalent. Guards against silent regression of the win-A perf patch.
    """
    panel, _ = _synth_panel_with_index(180, 3, seed=21)
    lookbacks = (5, 10, 20)
    patched = F._dollar_move_rank_N(panel, lookbacks)

    # Reference: replicate the pre-patch raw=False codepath inline.
    close = panel["close"]
    vol = panel["volume"].astype(float)
    dm = close.diff().abs() * vol
    reference = {}
    for N in lookbacks:
        reference[f"dollar_move_rank_{N}"] = F._per_ticker(
            dm, lambda s, n=N: s.rolling(n, min_periods=n)
                                .apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False),
        )

    for N in lookbacks:
        col = f"dollar_move_rank_{N}"
        pd.testing.assert_series_equal(patched[col].rename(None),
                                        reference[col].rename(None),
                                        check_names=False)


def test_vol_pct_raw_true_matches_raw_false():
    """Same byte-equivalence check for vol_regime's vol_pct_{N} output —
    raw=True numpy callback vs raw=False Series callback must agree.
    """
    panel, idx = _synth_panel_with_index(180, 3, seed=22)
    lookbacks = (5, 20)
    patched = F.vol_regime(panel, lookbacks=lookbacks, annualization=250)

    # Reference: build realized_vol then walk vol_pct via raw=False.
    rvol = F.realized_vol_N(panel, lookbacks, annualization=250)
    reference_cols = {}
    for N in lookbacks:
        v = rvol[f"realized_vol_{N}"]
        reference_cols[f"vol_pct_{N}"] = F._per_ticker(
            v, lambda s, n=N: s.rolling(n, min_periods=n)
                                .apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False),
        )

    for N in lookbacks:
        col = f"vol_pct_{N}"
        pd.testing.assert_series_equal(patched[col].rename(None),
                                        reference_cols[col].rename(None),
                                        check_names=False)


def test_build_feature_matrix_exclude_glob():
    panel, idx = _synth_panel_with_index(220, 3, seed=7)
    mat = F.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                  annualization=250, exclude=["volume_ratio_*"])
    assert F.EXPECTED_TOTAL_COLS - mat.shape[1] == 6, (
        "expected 6 fewer cols after excluding volume_ratio_*"
    )
    assert not any(c.startswith("volume_ratio_") for c in mat.columns)


def test_build_feature_matrix_subset_family():
    panel, idx = _synth_panel_with_index(120, 3, seed=8)
    mat = F.build_feature_matrix(panel, idx, lookbacks=(5, 10), families=["F2"])
    # F2 alone = 2 lookbacks → 2 cols
    assert mat.shape[1] == 2
    assert "stock_return_5" in mat.columns
    assert "stock_return_10" in mat.columns


# ---------------------------------------------------------------------------
# Zero-denominator guards (#182)
# ---------------------------------------------------------------------------
#
# Goal: every ratio/division family in features.py must map a zero denominator
# to NaN, NOT ±inf. NaN is the correct "undefined / missing" sentinel both
# CatBoost (NaN bucket) and XGBoost (sparsity-aware split) understand;
# ±inf crashes XGBoost's DMatrix construction (V1.2 sp500 +50%/100d, #180).
# The downstream band-aid `_sanitize_nonfinite` is kept as defense-in-depth
# but features.py is now contractually inf-free at source.


def _zero_denom_panel(n_rows: int = 60, zero_at: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a minimal (date, ticker) panel with a deliberate zero OHLCV row.

    One ticker ("BAD") has every OHLCV column set to 0.0 at row ``zero_at``
    (simulating a halted-stock bar / corrupted feed); the other ("GOOD") is
    a clean monotone series. The index_df has a parallel zero row at the
    same date so index-side guards are exercised too.
    """
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    frames = []
    for t in ("BAD", "GOOD"):
        close = np.linspace(100.0, 160.0, n_rows)
        high = close * 1.01
        low = close * 0.99
        open_ = close * 1.001
        volume = np.full(n_rows, 1_000_000.0)
        if t == "BAD":
            # All OHLC + volume = 0 at zero_at. Triggers zero-denom in
            # everything that divides by a prior close/high/low/volume.
            for arr in (close, high, low, open_, volume):
                arr[zero_at] = 0.0
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": open_, "high": high, "low": low, "close": close,
            "adj_close": close, "volume": volume,
        }))
    panel = pd.concat(frames, ignore_index=True).set_index(["date", "ticker"]).sort_index()

    iclose = np.linspace(1000.0, 1200.0, n_rows)
    iclose[zero_at] = 0.0
    ihigh = iclose * 1.005
    ilow = iclose * 0.995
    iopen = iclose * 1.001
    # ihigh/ilow share the 0 at zero_at so rolling max/min hits 0 too
    ihigh[zero_at] = 0.0
    ilow[zero_at] = 0.0
    iopen[zero_at] = 0.0
    index_df = pd.DataFrame(
        {"open": iopen, "high": ihigh, "low": ilow, "close": iclose,
         "adj_close": iclose, "volume": np.ones(n_rows)},
        index=pd.Index(dates, name="date"),
    )
    return panel, index_df


def _assert_no_inf(col: pd.Series, name: str) -> None:
    inf_count = int(np.isinf(col.values).sum())
    assert inf_count == 0, (
        f"{name}: expected zero ±inf after zero-denom guard, found {inf_count}"
    )


def test_zerodenom_f1_stock_return():
    """F1 stock_return_N: prior close == 0 must yield NaN, not ±inf."""
    panel, _ = _zero_denom_panel(n_rows=60, zero_at=30)
    sr = F.stock_return_N(panel, lookbacks=(5,))
    # At BAD row 35 the denominator is BAD's close at row 30 (== 0).
    bad_date_35 = panel.index.get_level_values("date").unique()[35]
    val = sr.loc[(bad_date_35, "BAD"), "stock_return_5"]
    assert np.isnan(val), f"expected NaN at zero-denom row, got {val}"
    _assert_no_inf(sr["stock_return_5"], "stock_return_5")


def test_zerodenom_f1_index_return():
    """F1 index_return_N: prior index close == 0 must yield NaN."""
    panel, idx = _zero_denom_panel(n_rows=60, zero_at=30)
    ir = F.index_return_N(idx, panel, lookbacks=(5,))
    bad_date_35 = panel.index.get_level_values("date").unique()[35]
    val = ir.loc[(bad_date_35, "BAD"), "index_return_5"]
    assert np.isnan(val), f"expected NaN, got {val}"
    _assert_no_inf(ir["index_return_5"], "index_return_5")


def test_zerodenom_f6_drawdown():
    """F6 drawdown_N: rolling high max == 0 over the window must yield NaN."""
    panel, _ = _zero_denom_panel(n_rows=60, zero_at=30)
    # Use a small window that lands entirely on the BAD zero row so the
    # rolling-max-of-highs equals 0. Window of length 1 at zero_at exposes this.
    dd = F.drawdown_N(panel, lookbacks=(1,))
    bad_date_30 = panel.index.get_level_values("date").unique()[30]
    val = dd.loc[(bad_date_30, "BAD"), "drawdown_1"]
    assert np.isnan(val), f"expected NaN at zero-high row, got {val}"
    _assert_no_inf(dd["drawdown_1"], "drawdown_1")


def test_zerodenom_f6_runup():
    """F6 runup_N: rolling low min == 0 must yield NaN."""
    panel, _ = _zero_denom_panel(n_rows=60, zero_at=30)
    ru = F.runup_N(panel, lookbacks=(5,))
    # Window [26..30] for BAD contains the zero-low row → rolling min = 0.
    bad_date_30 = panel.index.get_level_values("date").unique()[30]
    val = ru.loc[(bad_date_30, "BAD"), "runup_5"]
    assert np.isnan(val), f"expected NaN, got {val}"
    _assert_no_inf(ru["runup_5"], "runup_5")


def test_zerodenom_f6_index_drawdown():
    """F6 index_drawdown_N: index rolling high max == 0 must yield NaN."""
    panel, idx = _zero_denom_panel(n_rows=60, zero_at=30)
    idd = F.index_drawdown_N(idx, panel, lookbacks=(1,))
    bad_date_30 = panel.index.get_level_values("date").unique()[30]
    val = idd.loc[(bad_date_30, "BAD"), "index_drawdown_1"]
    assert np.isnan(val), f"expected NaN, got {val}"
    _assert_no_inf(idd["index_drawdown_1"], "index_drawdown_1")


def test_zerodenom_f6_index_runup():
    """F6 index_runup_N: index rolling low min == 0 must yield NaN."""
    panel, idx = _zero_denom_panel(n_rows=60, zero_at=30)
    iru = F.index_runup_N(idx, panel, lookbacks=(5,))
    bad_date_30 = panel.index.get_level_values("date").unique()[30]
    val = iru.loc[(bad_date_30, "BAD"), "index_runup_5"]
    assert np.isnan(val), f"expected NaN, got {val}"
    _assert_no_inf(iru["index_runup_5"], "index_runup_5")


def test_zerodenom_f11_range_vol_zero_low():
    """F11 range_vol: low == 0 must yield NaN in parkinson/garman_klass."""
    panel, _ = _zero_denom_panel(n_rows=60, zero_at=30)
    rv = F.range_vol(panel, lookbacks=(5,), annualization=250)
    # parkinson_5 + garman_klass_5 are rolling means containing the zero row.
    _assert_no_inf(rv["parkinson_5"], "parkinson_5")
    _assert_no_inf(rv["garman_klass_5"], "garman_klass_5")


def test_zerodenom_f13_vol_change_zero_prior_vol():
    """F13 vol_change_N: prior realized_vol == 0 must yield NaN.

    Construct a panel whose log-returns are all identical (e.g. constant
    geometric growth) for a stretch — realized_vol over that window = 0,
    so vol_change at t+N divides by 0.
    """
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    # Strictly geometric: log-returns are constant → rolling std == 0.
    close = 100.0 * np.exp(np.arange(80) * 0.01)
    high = close * 1.001
    low = close * 0.999
    open_ = close * 1.0005
    volume = np.full(80, 1_000_000.0)
    df = pd.DataFrame({
        "date": dates, "ticker": "FLAT",
        "open": open_, "high": high, "low": low, "close": close,
        "adj_close": close, "volume": volume,
    })
    panel = df.set_index(["date", "ticker"]).sort_index()
    vr = F.vol_regime(panel, lookbacks=(5,), annualization=250)
    _assert_no_inf(vr["vol_change_5"], "vol_change_5")


def test_zerodenom_f16_native_stock_return_zscore():
    """F16 stock_return_zscore_N: prior close == 0 must NOT propagate ±inf."""
    panel, _ = _zero_denom_panel(n_rows=60, zero_at=30)
    f16 = F.f16_underlying(panel, lookbacks=(5,), annualization=250)
    _assert_no_inf(f16["stock_return_zscore_5"], "stock_return_zscore_5")
    _assert_no_inf(f16["realized_vol_zscore_5"], "realized_vol_zscore_5")


def test_build_feature_matrix_no_inf_with_zero_denom_panel():
    """End-to-end: full feature build on a zero-denom panel emits zero ±inf,
    AND the build-boundary assert does not trip."""
    panel, idx = _zero_denom_panel(n_rows=240, zero_at=120)
    mat = F.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                  annualization=250, families="all")
    inf_count = int(np.isinf(mat.values).sum())
    assert inf_count == 0, (
        f"build_feature_matrix emitted {inf_count} ±inf on zero-denom fixture; "
        "every ratio family must guard its denominator at source (#182)."
    )


def test_build_feature_matrix_assert_trips_on_synthetic_inf():
    """The build-boundary assert MUST trip if a future family regresses.

    We can't easily inject an inf via the public pipeline (every family is now
    guarded), so this monkeypatches one family to deliberately return ±inf and
    confirms build_feature_matrix raises AssertionError with the contract message.
    Defends the assert against future drift.
    """
    panel, idx = _synth_panel_with_index(120, 2, seed=11)

    original = F.stock_return_N
    def _evil(panel, lookbacks=F.DEFAULT_LOOKBACKS):
        out = original(panel, lookbacks)
        # Plant an inf in the first usable cell
        out.iloc[10, 0] = np.inf
        return out

    import gbdt.features as Fmod
    Fmod.stock_return_N = _evil
    try:
        with pytest.raises(AssertionError, match="zero-denominator regression"):
            F.build_feature_matrix(panel, idx, lookbacks=(5,), families=["F2"])
    finally:
        Fmod.stock_return_N = original


# ---------------------------------------------------------------------------
# Defense-in-depth: _sanitize_nonfinite still works on a synthetic inf array.
# Features no longer emit inf at source (#182), so the band-aid path goes
# uncovered by feature-build tests. This direct test keeps it exercised.
# ---------------------------------------------------------------------------


def test_sanitize_nonfinite_still_handles_synthetic_inf():
    from gbdt.model import _sanitize_nonfinite
    arr = np.array([[1.0, np.inf, 2.0], [np.nan, -np.inf, 3.0]])
    out = _sanitize_nonfinite(arr)
    # ±inf → NaN; existing NaN preserved; finite untouched
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 1])
    assert np.isnan(out[1, 0])
    assert out[0, 0] == 1.0
    assert out[1, 2] == 3.0
    assert not np.isinf(out).any()


# ---------------------------------------------------------------------------
# GBDTPERF vectorization equivalence (lock the bit-identical optimizations).
# Two feature-build hot spots were replaced with native/vectorized forms (the
# rolling percentile-rank and the F16 signed-days streak); these regression
# tests embed the exact reference implementations they replaced and require the
# current code to match them value-for-value (incl. ties + NaN), so a future
# refactor cannot silently change feature values. See docs/gbdt + the
# feature-parity gate (scripts/gbdt/check_feature_parity.py).
# ---------------------------------------------------------------------------


def test_rolling_pct_rank_matches_apply_reference():
    """``_rolling_pct_rank`` must stay bit-identical to the prior
    ``rolling(n).apply(lambda w: (w[-1] >= w).mean())`` — incl. ties and NaN."""
    rng = np.random.default_rng(0)

    def ref(s, n):
        return s.rolling(n, min_periods=n).apply(lambda w: (w[-1] >= w).mean(), raw=True)

    cases = {
        "random": pd.Series(rng.normal(size=300)),
        "heavy_ties": pd.Series(rng.integers(0, 5, size=300).astype(float)),
        "leading_nan": pd.Series([np.nan] * 7 + list(rng.normal(size=293))),
        "mid_nan": pd.Series(np.where(rng.random(300) < 0.05, np.nan, rng.normal(size=300))),
        "monotone": pd.Series(np.arange(300.0)),
    }
    for name, s in cases.items():
        for n in (5, 10, 20, 50, 100, 200):
            a = ref(s, n).to_numpy()
            b = F._rolling_pct_rank(s, n).to_numpy()
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name} N={n} NaN-structure"
            np.testing.assert_array_equal(
                a[~np.isnan(a)], b[~np.isnan(b)], err_msg=f"{name} N={n}")


def test_signed_days_streak_matches_loop_reference():
    """Vectorized ``_signed_days_outside_band_one`` must stay bit-identical to the
    element-wise Python-loop reference it replaced — long runs, sign flips, band
    re-entry, NaN resets in every position, ties exactly at the band edge."""

    def ref(z_values, sigma):
        out = np.zeros(len(z_values), dtype=float)
        streak = 0
        direction = 0
        for i, z in enumerate(z_values):
            if np.isnan(z):
                streak = 0
                direction = 0
                out[i] = np.nan
                continue
            if z >= sigma:
                streak = streak + 1 if direction == 1 else 1
                direction = 1
                out[i] = streak
            elif z <= -sigma:
                streak = streak + 1 if direction == -1 else 1
                direction = -1
                out[i] = -streak
            else:
                direction = 0
                streak = 0
                out[i] = 0.0
        return out

    rng = np.random.default_rng(1)
    edge = [[], [np.nan] * 5, [5, 5, 5, 5], [-5, -5, -5], [5, -5, 5, -5],
            [5, 5, 0, 5], [5, 5, np.nan, 5, 5], [np.nan, np.nan, 5, 5],
            [5, 5, np.nan], [1.0, 2.0, 3.0, 1.0], [-1, -2, -1.5, 1, 2]]
    series = [np.asarray(c, dtype=float) for c in edge]
    for k in range(60):
        z = rng.normal(0, 1.5, size=int(rng.integers(1, 220)))
        if k % 3 == 0:
            z[rng.random(len(z)) < 0.1] = np.nan
        series.append(z)
    for z in series:
        for sigma in (1.0, 2.0, 3.0):
            a = ref(z, sigma)
            b = F._signed_days_outside_band_one(z, sigma)
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"NaN-structure sigma={sigma}"
            np.testing.assert_array_equal(
                a[~np.isnan(a)], b[~np.isnan(b)], err_msg=f"sigma={sigma}")
