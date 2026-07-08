"""F21 calendar2 feature family — columns, month-of-quarter / quarter-of-year
math, causality (C1 via the leakage harness), and the invariance locks: the
``"all"`` and ``"all_fundamentals"`` tokens (and every existing model) stay
byte-identical, and the new ``all_calendar2`` / ``all_fundamentals_calendar2``
tokens add exactly the F21 columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbdt import features as F
from gbdt.leakage_harness import LeakageHarness, make_synthetic_panel


def _index_df(dates, *, seed=2):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates),
    )


def _tickers(panel):
    return list(panel.index.get_level_values("ticker").unique())


def _fund_df(dates, tickers, *, seed=3):
    """Minimal synthetic (date, symbol)-indexed valuation panel for F18."""
    rows = []
    for i, tk in enumerate(tickers):
        ey = 0.03 + 0.001 * i
        rows.append(pd.DataFrame({
            "date": dates, "symbol": tk,
            "earnings_yield": ey, "sales_yield": ey * 4.0, "fcf_yield": ey * 0.8,
            "revenue_ttm": 1000.0 * (i + 1),
        }))
    return pd.concat(rows).set_index(["date", "symbol"]).sort_index()


# Explicit month -> (month-of-quarter, quarter-of-year) lookup — deliberately
# NOT the ((month-1)%3)+1 / ((month-1)//3)+1 formulas, so the value test is a
# genuine oracle rather than a circular restatement of the implementation.
_MOQ = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3,
        7: 1, 8: 2, 9: 3, 10: 1, 11: 2, 12: 3}
_QOY = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}


def test_emits_four_columns():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    feat = F.calendar2_features(panel)

    assert feat.index.equals(panel.index)
    assert sorted(feat.columns) == ["moq_cos", "moq_sin", "qoy_cos", "qoy_sin"]
    assert feat.shape[1] == 4


def test_moq_qoy_math_for_known_months():
    # Span >1 year so all 12 months appear, then check every row against the
    # explicit month->(moq,qoy) oracle above.
    panel = make_synthetic_panel(n_rows=400, n_tickers=2)
    feat = F.calendar2_features(panel)
    months = panel.index.get_level_values("date").month

    exp_moq = np.array([_MOQ[m] for m in months], dtype=float)
    exp_qoy = np.array([_QOY[m] for m in months], dtype=float)

    assert np.allclose(feat["moq_sin"].values, np.sin(2 * np.pi * exp_moq / 3.0))
    assert np.allclose(feat["moq_cos"].values, np.cos(2 * np.pi * exp_moq / 3.0))
    assert np.allclose(feat["qoy_sin"].values, np.sin(2 * np.pi * exp_qoy / 4.0))
    assert np.allclose(feat["qoy_cos"].values, np.cos(2 * np.pi * exp_qoy / 4.0))

    # sanity spot-check: every value lies on the unit circle
    assert np.allclose(feat["moq_sin"].values ** 2 + feat["moq_cos"].values ** 2, 1.0)
    assert np.allclose(feat["qoy_sin"].values ** 2 + feat["qoy_cos"].values ** 2, 1.0)


def test_calendar2_features_are_causal():
    # Date-only family — perturbing future OHLCV must not move any pre-leak
    # row. Exercised through the shared leakage harness (C1).
    LeakageHarness(n_rows=360, n_tickers=3).assert_causal(F.calendar2_features)


def test_all_calendar2_includes_f21_and_locks_baseline():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    idx = _index_df(panel.index.get_level_values("date").unique())

    X_all = F.build_feature_matrix(panel, idx, families="all")
    X_cal2 = F.build_feature_matrix(panel, idx, families="all_calendar2")

    cal2_cols = ["moq_sin", "moq_cos", "qoy_sin", "qoy_cos"]
    assert all(c in X_cal2.columns for c in cal2_cols)
    # "all" is byte-identical and carries NONE of the F21 columns (opt-in only).
    assert not any(c in X_all.columns for c in cal2_cols)
    assert X_all.shape[1] == F.EXPECTED_TOTAL_COLS == 279
    assert X_cal2.shape[1] == F.EXPECTED_TOTAL_COLS + 4 == 283
    # The all-subset inside all_calendar2 is byte-identical to all alone.
    assert X_cal2[list(X_all.columns)].equals(X_all)


def test_all_fundamentals_calendar2_includes_f18_and_f21():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    fdf = _fund_df(dates, _tickers(panel))

    X = F.build_feature_matrix(
        panel, idx, families="all_fundamentals_calendar2", fund_df=fdf)
    assert all(c in X.columns for c in ("moq_sin", "moq_cos", "qoy_sin", "qoy_cos"))
    assert any(c.startswith("fund_") for c in X.columns)
    # and it locks the plain "all" baseline (no cal2/fund leakage into all)
    X_all = F.build_feature_matrix(panel, idx, families="all")
    assert X_all.shape[1] == F.EXPECTED_TOTAL_COLS == 279
    assert X.shape[1] > X_all.shape[1]
    assert X[list(X_all.columns)].equals(X_all)
