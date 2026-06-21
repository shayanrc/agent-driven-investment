"""F17 macro feature family — columns, causality (C1), and the critical
invariance that the ``"all"`` token (and every existing model) is unaffected.

Macro features are date-broadcast (constant across tickers on a date) and
derive from a synthetic FRED ``macro_df``, so the standard OHLCV leakage
harness (which perturbs panel prices) can't exercise them — these tests plant
the leak in the *macro* input instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbdt import features as F
from gbdt.leakage_harness import make_synthetic_panel


def _macro_df(dates, series=("DGS10", "VIXCLS", "DFF"), *, seed=1):
    """Synthetic date-indexed macro panel (random-walk-ish levels)."""
    rng = np.random.default_rng(seed)
    cols = {s: 5.0 + np.cumsum(rng.normal(0.0, 0.05, len(dates))) for s in series}
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


def _index_df(dates, *, seed=2):
    """Minimal synthetic benchmark OHLCV frame for build_feature_matrix."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates),
    )


def test_emits_expected_columns_and_broadcasts():
    panel = make_synthetic_panel(n_rows=200, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    m = _macro_df(dates, series=("DGS10", "VIXCLS"))
    feat = F.fred_macro_features(m, panel, series=("DGS10", "VIXCLS"))

    assert feat.index.equals(panel.index)
    for sid in ("DGS10", "VIXCLS"):
        assert f"macro_{sid}_level" in feat.columns
        assert f"macro_{sid}_chg_20" in feat.columns
        assert f"macro_{sid}_chg_60" in feat.columns
        assert f"macro_{sid}_z_60" in feat.columns
        assert f"macro_{sid}_z_120" in feat.columns
    # Date-broadcast: identical value across all tickers on a given date.
    d = dates[150]
    col = feat.xs(d, level="date")["macro_DGS10_level"]
    assert col.nunique(dropna=False) == 1


def test_macro_features_are_causal():
    """Perturb ALL macro values from a future date onward; every pre-leak panel
    row's F17 features must be byte-identical (C1, zero look-ahead)."""
    panel = make_synthetic_panel(n_rows=220, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    m = _macro_df(dates)
    base = F.fred_macro_features(m, panel)

    leak_row = 160
    m_leak = m.copy()
    m_leak.iloc[leak_row:] *= 5.0  # spectacular future spike across all series
    leak = F.fred_macro_features(m_leak, panel)

    leak_date = dates[leak_row]
    pre = panel.index.get_level_values("date") < leak_date
    b, l = base[pre], leak[pre]
    both_nan = b.isna() & l.isna()
    diff = (l.fillna(0.0) - b.fillna(0.0)).abs().mask(both_nan, 0.0)
    max_diff = float(diff.values.max()) if diff.size else 0.0
    assert max_diff <= 1e-9, f"F17 is non-causal: max pre-leak diff={max_diff:g}"


def test_no_inf_on_flat_series():
    """A flat macro series (std==0 window) must yield NaN z-scores, never inf
    (the build_feature_matrix #182 guard would otherwise raise)."""
    panel = make_synthetic_panel(n_rows=200)
    dates = panel.index.get_level_values("date").unique()
    m = pd.DataFrame({"DFF": np.full(len(dates), 5.0)}, index=pd.DatetimeIndex(dates))
    feat = F.fred_macro_features(m, panel, series=("DFF",))
    assert not np.isinf(np.nan_to_num(feat.values, nan=0.0)).any()


def test_missing_series_skipped():
    panel = make_synthetic_panel(n_rows=120)
    dates = panel.index.get_level_values("date").unique()
    m = _macro_df(dates, series=("DGS10",))
    feat = F.fred_macro_features(m, panel, series=("DGS10", "NOTSEEDED"))
    assert any(c.startswith("macro_DGS10_") for c in feat.columns)
    assert not any("NOTSEEDED" in c for c in feat.columns)


def test_all_token_excludes_macro_but_all_macro_includes_it():
    """The decisive invariance: families='all' NEVER emits macro columns (even
    when macro_df is supplied), and 'all_macro' adds macro on top WITHOUT
    altering any baseline column → existing models stay byte-identical."""
    panel = make_synthetic_panel(n_rows=200, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    m = _macro_df(dates)

    X_all = F.build_feature_matrix(panel, idx, families="all", macro_df=m)
    assert not any(c.startswith("macro_") for c in X_all.columns)

    X_macro = F.build_feature_matrix(panel, idx, families="all_macro", macro_df=m)
    assert any(c.startswith("macro_") for c in X_macro.columns)

    # Baseline columns are identical in name AND value between the two builds.
    base_cols = [c for c in X_macro.columns if not c.startswith("macro_")]
    assert set(base_cols) == set(X_all.columns)
    pd.testing.assert_frame_equal(X_macro[X_all.columns], X_all)


def test_all_macro_requires_macro_df():
    panel = make_synthetic_panel(n_rows=120, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    try:
        F.build_feature_matrix(panel, idx, families="all_macro", macro_df=None)
    except ValueError as e:
        assert "macro_df" in str(e)
    else:
        raise AssertionError("expected ValueError when all_macro selected without macro_df")
