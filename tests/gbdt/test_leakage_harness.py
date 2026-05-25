"""Stage 1 — leakage harness self-tests.

The harness must:
- Fire (``causal=False``) on a known-leaky function that uses future data.
- Stay silent (``causal=True``) on a known-causal function that uses only
  past data.
"""

from __future__ import annotations

import pandas as pd

from gbdt.leakage_harness import (
    LeakageHarness,
    make_synthetic_panel,
    plant_leak,
    synthetic_leak_test,
)


def _causal_rolling_mean_5(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-stock rolling mean of close over the last 5 rows (inclusive of t).

    Causal: shifts back by 0 but uses ``rolling(...).mean()`` which at row t
    looks at rows [t-4, t]. No look-ahead.
    """
    out = panel.groupby(level="ticker", group_keys=False)["close"].apply(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    return out.to_frame(name="causal_mean_5")


def _leaky_lead_close(panel: pd.DataFrame) -> pd.DataFrame:
    """Pull tomorrow's close into today's feature row. Explicitly leaky."""
    out = panel.groupby(level="ticker", group_keys=False)["close"].apply(
        lambda s: s.shift(-1)
    )
    return out.to_frame(name="leaky_lead_close")


def test_harness_passes_causal_function():
    report = synthetic_leak_test(_causal_rolling_mean_5)
    assert report.causal, (
        f"causal function flagged as non-causal: {report}"
    )
    assert report.max_abs_diff_pre_leak == 0.0


def test_harness_detects_leak():
    report = synthetic_leak_test(_leaky_lead_close)
    assert not report.causal, (
        f"leaky function passed harness: {report}"
    )
    assert report.max_abs_diff_pre_leak > 0
    assert "leaky_lead_close" in report.columns_with_diff


def test_plant_leak_actually_perturbs_only_leak_row():
    base = make_synthetic_panel(60, 1, seed=0)
    leaky = plant_leak(base, leak_row=40)
    leak_date = base.index.get_level_values("date").unique()[40]
    # Pre-leak rows unchanged.
    pre = base.index.get_level_values("date") < leak_date
    assert (base.loc[pre, "close"].values == leaky.loc[pre, "close"].values).all()
    # Leak row changed.
    assert leaky.loc[(leak_date, "TKR0"), "close"] > base.loc[(leak_date, "TKR0"), "close"]


def test_harness_with_custom_panel_size():
    h = LeakageHarness(n_rows=80, n_tickers=2, leak_row=50)
    report = h.check(_causal_rolling_mean_5)
    assert report.causal
    assert report.n_pre_leak > 0
