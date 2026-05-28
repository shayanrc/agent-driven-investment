"""Stage 1 — leakage harness self-tests.

The harness must:
- Fire (``causal=False``) on a known-leaky function that uses future data.
- Stay silent (``causal=True``) on a known-causal function that uses only
  past data.

The second block (V1.2 Phase 3, ``docs/gbdt/V1.2_xgboost_feature_interactions_
plan.md`` § 5.3 / § 8) is the **model-level C6 guard, parametrized over backend**.
CatBoost gets split-discipline + ordered boosting (``has_time=True``); XGBoost
gets **split-discipline only** (no ``has_time`` analogue — plan § 5.3). So the
synthetic-leakage test must run for BOTH backends: a planted *future-target*
signal must NOT yield spectacular AUC when the panel is properly masked
(causal features only), and MUST yield spectacular AUC when a future quantity
is leaked straight into the feature matrix (the positive control proving the
test can see leakage). This is what guards XGBoost's weaker internal C6
guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from gbdt.leakage_harness import (
    LeakageHarness,
    make_synthetic_panel,
    plant_leak,
    synthetic_leak_test,
)
from gbdt.model import make_model


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


# ---------------------------------------------------------------------------
# Model-level C6 guard — parametrized over backend (V1.2 Phase 3, plan § 5.3)
#
# Trains a real model on a synthetic panel whose target is a *forward-looking*
# event (it depends on FUTURE data by construction). With only causal features
# the model cannot recover the future → AUC near base-rate (no spectacular
# skill). Leak the future quantity into the feature matrix and AUC jumps to ~1
# (positive control). Run for both backends — XGBoost's C6 rests on the split
# discipline alone (no ``has_time``), so this is the guard that catches a
# masking failure on the new backend.
# ---------------------------------------------------------------------------


_HORIZON = 5
_THRESHOLD = 0.02          # +2% forward move defines the event


def _forward_event_panel(n_rows: int = 600, *, seed: int = 0):
    """Build a single-series panel with a forward-looking binary target plus
    causal-only features and one deliberately-leaked future feature.

    Returns ``(X_causal, X_leaky, y)`` as aligned numpy/Series objects over the
    rows where the forward window is fully observed. ``X_leaky`` is ``X_causal``
    with the future-window max-return column appended (the planted leak).
    """
    rng = np.random.default_rng(seed)
    # Near-random-walk log-returns → the forward event is genuinely
    # unpredictable from the past (AUC ~ 0.5 when properly masked).
    rets = rng.normal(0.0, 0.012, size=n_rows)
    close = 100.0 * np.exp(np.cumsum(rets))
    s = pd.Series(close)

    # Causal features: trailing returns / momentum (strictly past data).
    ret_1 = s.pct_change(1)
    ret_5 = s.pct_change(5)
    mom_10 = s.pct_change(10)
    roll_std_10 = ret_1.rolling(10, min_periods=10).std()

    # Forward target: does close rise >= +2% within the next H bars?
    fwd_max = (
        pd.Series(close).shift(-1).rolling(_HORIZON, min_periods=_HORIZON).max()
    )
    # Align fwd_max so index t sees max(close[t+1 : t+H]).
    fwd_max = fwd_max.shift(-(_HORIZON - 1))
    fwd_ret = fwd_max / s - 1.0
    y = (fwd_ret >= _THRESHOLD).astype(float)

    feat = pd.DataFrame({
        "ret_1": ret_1,
        "ret_5": ret_5,
        "mom_10": mom_10,
        "roll_std_10": roll_std_10,
    })
    # The planted leak: the very quantity the target is defined on, as a
    # feature available at prediction time (what a look-ahead bug would do).
    leaky = feat.copy()
    leaky["LEAK_fwd_ret"] = fwd_ret

    valid = feat.notna().all(axis=1) & fwd_ret.notna() & y.notna()
    return (
        feat.loc[valid].reset_index(drop=True),
        leaky.loc[valid].reset_index(drop=True),
        y.loc[valid].reset_index(drop=True).astype(int),
    )


def _walk_forward_auc(backend: str, X: pd.DataFrame, y: pd.Series) -> float:
    """Train ``backend`` on the early (time-ordered) split, score the held-out
    tail, return ROC-AUC. No shuffling — train precedes test in time (C6)."""
    n = len(y)
    cut = int(n * 0.7)
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
    hp = (
        {"iterations": 200, "depth": 4, "learning_rate": 0.1}
        if backend == "catboost"
        else {"n_estimators": 200, "max_depth": 4, "eta": 0.1}
    )
    m = make_model(backend, hp, feature_names=list(X.columns))
    m.fit(X_tr, y_tr.values)
    p = m.predict_proba(X_te)
    return float(roc_auc_score(y_te.values, p))


@pytest.mark.parametrize("backend", ["catboost", "xgboost"])
def test_masked_panel_no_spectacular_auc(backend):
    """C6 guard: with causal-only features, neither backend extracts
    spectacular AUC from a forward-looking target — masking held."""
    X_causal, _X_leaky, y = _forward_event_panel(seed=1)
    # Guard against a degenerate all-one / all-zero target.
    assert 0.05 < y.mean() < 0.95, f"degenerate base rate {y.mean()}"
    auc = _walk_forward_auc(backend, X_causal, y)
    assert auc < 0.75, (
        f"{backend}: properly-masked panel produced AUC={auc:.3f} — a value "
        f"this high on an unpredictable forward target signals look-ahead "
        f"leakage, not skill (C6 violation)."
    )


@pytest.mark.parametrize("backend", ["catboost", "xgboost"])
def test_leaked_future_feature_yields_spectacular_auc(backend):
    """Positive control: leaking the future quantity into the feature matrix
    DOES produce spectacular AUC on both backends — so the test above is a
    real guard, not a vacuous pass on a model that can't fit anything."""
    _X_causal, X_leaky, y = _forward_event_panel(seed=1)
    auc = _walk_forward_auc(backend, X_leaky, y)
    assert auc > 0.95, (
        f"{backend}: a planted future-target leak only reached AUC={auc:.3f}; "
        f"the harness can't detect leakage, so the masked-panel test is vacuous."
    )
