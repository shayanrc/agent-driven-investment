"""V1.2 Phase 6 — calibration + persistence verification for the XGBoost backend.

Locks in the Phase-6 acceptance row (``docs/gbdt/V1.2_xgboost_feature_
interactions_plan.md`` § 8 Phase 6 + § 4.3 + R7):

> *isotonic + Platt applied to a synthetic miscalibrated XGBoost output recovers
> calibration (Spiegelhalter |Z| drops below threshold); ``save``→``load``→
> ``predict_proba`` reproduces predictions for XGBoost.*

These tests differ from ``test_calibration.py`` (which exercises the calibration
maths on synthetic ``(p_raw, y)`` arrays) by sourcing ``p_raw`` from an **actual
fitted ``XGBoostModel``** — proving the array-based, model-untouching calibration
flow (§ 4.3 / R7) is genuinely backend-agnostic on a real XGBoost output, and that
the ``.ubj`` save/load round-trip reproduces predictions bit-identically and
composes with a fitted calibrator.

No data-cache panel build, no walk-forward production run — tiny synthetic fits
only (Phase 6 is unit/integration on synthetic data).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from gbdt.calibration import (
    apply_calibrator,
    conditional_isotonic,
    fit_isotonic,
    fit_platt,
    platt_calibration,
    spiegelhalter_z,
)
from gbdt.model import XGBoostModel, make_model
from gbdt.train import SplitSpec, walk_forward_train


# ---------------------------------------------------------------------------
# Fixtures — a tiny fitted XGBoost model that produces MISCALIBRATED raw output
# ---------------------------------------------------------------------------


def _miscalibrated_xgb_output(n: int = 6000, seed: int = 0):
    """Fit a deliberately over-confident ``XGBoostModel`` and return its raw
    ``predict_proba`` on a held-out slice alongside the true labels.

    The recipe: train a deep, lightly-regularised XGBoost on a noisy
    near-random target so that, on held-out data, the trees over-fit and push
    probabilities away from the base rate (classic tree over-confidence). The
    raw ``predict_proba`` on the eval slice is then miscalibrated vs the realised
    ``y`` — the Spiegelhalter Z fires — which is exactly the input the calibration
    layer must repair. Returns ``(y_eval, p_raw_eval)``.
    """
    rng = np.random.default_rng(seed)
    n_feat = 6
    X = rng.normal(0, 1, (n, n_feat))
    # A weak signal swamped by noise → the model can't truly separate, so when
    # it over-fits it becomes over-confident out-of-sample.
    logit = 0.6 * X[:, 0] - 0.4 * X[:, 1]
    p_true = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p_true).astype(int)

    cols = [f"f{j}" for j in range(n_feat)]
    Xdf = pd.DataFrame(X, columns=cols)
    n_tr = n // 2
    n_val = n // 4
    X_tr, y_tr = Xdf.iloc[:n_tr], y[:n_tr]
    X_val, y_val = Xdf.iloc[n_tr:n_tr + n_val], y[n_tr:n_tr + n_val]
    X_ev, y_ev = Xdf.iloc[n_tr + n_val:], y[n_tr + n_val:]

    # Deep, low-regularisation, no early stopping → over-confident trees.
    hp = {"n_estimators": 400, "max_depth": 8, "eta": 0.3, "lambda": 0.0,
          "min_child_weight": 1.0}
    model = make_model("xgboost", dict(hp), feature_names=cols, random_seed=7)
    model.fit(X_tr, y_tr)  # no eval_set → no early stopping → full over-fit
    p_raw_ev = model.predict_proba(X_ev)
    return y_ev, p_raw_ev, model, X_ev


# ---------------------------------------------------------------------------
# (a) Isotonic recovers calibration on real XGBoost output
# ---------------------------------------------------------------------------


def test_isotonic_recovers_calibration_on_xgboost_output():
    """The Spiegelhalter-gated isotonic path repairs a miscalibrated XGBoost
    output: the fitted isotonic map drops |Z| below the gate threshold.

    Confirms the calibration flow is backend-agnostic — it consumes only the
    ``(y, p_raw)`` arrays ``XGBoostModel.predict_proba`` produces; nothing in
    the path touches the XGBoost model object (§ 4.3)."""
    y, p_raw, _model, _X = _miscalibrated_xgb_output(seed=1)

    z_before, _ = spiegelhalter_z(y, p_raw)
    assert abs(z_before) >= 2.0, (
        f"fixture must be miscalibrated for a recovery test; |Z|={abs(z_before):.2f}"
    )

    dec = conditional_isotonic(y, p_raw, z_threshold=2.0)
    # The gate fired (|Z| >= threshold) → isotonic was fit, not native pass-through.
    assert dec.method == "isotonic"
    assert dec.calibrator is not None

    p_cal = apply_calibrator(p_raw, dec.calibrator)
    z_after, _ = spiegelhalter_z(y, p_cal)
    assert abs(z_after) < 2.0
    assert abs(z_after) < abs(z_before)


def test_calibration_flow_never_touches_xgboost_model_object():
    """Backend-agnostic by construction: fitting + applying the calibrator works
    from the raw arrays alone, with no model handle in scope."""
    y, p_raw, _model, _X = _miscalibrated_xgb_output(seed=2)
    # Only arrays cross the boundary — fit_isotonic / fit_platt take (y, p).
    iso = fit_isotonic(y, p_raw)
    platt = fit_platt(y, p_raw)
    # Both produce 1-D calibrated probabilities in [0,1] from p_raw alone.
    for cal in (iso, platt):
        out = apply_calibrator(p_raw, cal)
        assert out.ndim == 1 and out.shape == p_raw.shape
        assert ((out >= 0) & (out <= 1)).all()


# ---------------------------------------------------------------------------
# (b) Platt (R7) recovers calibration on real XGBoost output, backend-neutral
# ---------------------------------------------------------------------------


def test_platt_recovers_calibration_on_xgboost_output():
    """Backend-neutral Platt scaling (R7 — fit on ``(p_raw, y)``, NOT via
    XGBClassifier) repairs miscalibrated XGBoost output: |Z| drops below the
    gate threshold."""
    y, p_raw, _model, _X = _miscalibrated_xgb_output(seed=3)

    z_before, _ = spiegelhalter_z(y, p_raw)
    assert abs(z_before) >= 2.0

    dec = platt_calibration(y, p_raw, z_threshold=2.0)
    assert dec.method == "platt"
    assert dec.calibrator is not None
    # R7: the calibrator is a manual 1-D logistic on the raw probability —
    # NOT an XGBClassifier/CatBoostClassifier-derived sklearn surface.
    from gbdt.calibration import PlattCalibrator
    assert isinstance(dec.calibrator, PlattCalibrator)
    from sklearn.linear_model import LogisticRegression
    assert isinstance(dec.calibrator._lr, LogisticRegression)
    assert dec.calibrator._lr.n_features_in_ == 1

    p_cal = apply_calibrator(p_raw, dec.calibrator)
    z_after, _ = spiegelhalter_z(y, p_cal)
    assert abs(z_after) < 2.0
    assert abs(z_after) < abs(z_before)


# ---------------------------------------------------------------------------
# (c) .ubj save/load round-trip reproduces predictions + composes w/ calibration
# ---------------------------------------------------------------------------


def test_ubj_save_load_reproduces_predict_proba(tmp_path):
    """An ``XGBoostModel`` save→load→predict_proba reproduces raw probabilities
    bit-identically for the native ``.ubj`` format."""
    y, p_raw, model, X_ev = _miscalibrated_xgb_output(seed=4)

    path = tmp_path / "model.ubj"
    model.save(path)
    assert path.exists() and path.suffix == ".ubj"

    loaded = XGBoostModel.load(path, feature_names=model.feature_names)
    assert loaded.fitted
    p_reloaded = loaded.predict_proba(X_ev)
    np.testing.assert_array_equal(p_raw, p_reloaded)


def test_ubj_roundtrip_composes_with_isotonic(tmp_path):
    """The .ubj round-trip composes with calibration: raw proba reproduced →
    the SAME isotonic calibrator applied gives the SAME calibrated output."""
    y, p_raw, model, X_ev = _miscalibrated_xgb_output(seed=5)
    dec = conditional_isotonic(y, p_raw, z_threshold=2.0)
    assert dec.method == "isotonic"
    p_cal_before = apply_calibrator(p_raw, dec.calibrator)

    # Persist BOTH the model (.ubj) and the calibrator (pickle), reload, recompose.
    model.save(tmp_path / "model.ubj")
    with open(tmp_path / "calibration.pkl", "wb") as f:
        pickle.dump(dec.calibrator, f)

    loaded = XGBoostModel.load(tmp_path / "model.ubj",
                               feature_names=model.feature_names)
    with open(tmp_path / "calibration.pkl", "rb") as f:
        cal_loaded = pickle.load(f)

    p_raw_reloaded = loaded.predict_proba(X_ev)
    p_cal_after = apply_calibrator(p_raw_reloaded, cal_loaded)
    np.testing.assert_array_equal(p_cal_before, p_cal_after)


# ---------------------------------------------------------------------------
# End-to-end: calibration_method="platt" flows through walk_forward_train
# ---------------------------------------------------------------------------


_SPLIT = SplitSpec(train_rows=180, val_rows=80, eval_rows=40, test_rows=20)


def _toy_panel(seed: int = 0, n_per_ticker: int = 320, n_tickers: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_per_ticker, freq="B")
    frames = []
    for i in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_per_ticker)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i}",
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c, "volume": np.ones(n_per_ticker, dtype=int),
        }))
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    n = len(panel)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "b": b, "n1": rng.normal(0, 1, n)}, index=panel.index)
    y = pd.Series(((a > 0) ^ (b > 0)).astype(int), index=panel.index)
    return panel, X, y


@pytest.mark.parametrize("backend,hp", [
    ("xgboost", {"n_estimators": 40, "max_depth": 3, "eta": 0.3, "lambda": 1.0}),
    ("catboost", {"iterations": 30, "depth": 3, "boosting_type": "Plain",
                  "learning_rate": 0.1}),
])
def test_walk_forward_platt_method_runs(backend, hp):
    """``calibration_method="platt"`` (previously a NotImplementedError) now runs
    end-to-end through ``walk_forward_train`` for BOTH backends and records a
    Platt decision with a fitted calibrator."""
    panel, X, y = _toy_panel(seed=7)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns), hp=dict(hp),
        split=_SPLIT, max_iterations=2, backend=backend, random_seed=42,
        calibration_method="platt",
    )
    assert result.calibration.method == "platt"
    assert result.calibration.calibrator is not None
    # Predictions carry both raw + calibrated columns; calibrated lands in [0,1].
    val = result.predictions["val"]
    assert ((val["p_calibrated"] >= 0) & (val["p_calibrated"] <= 1)).all()


def test_ubj_roundtrip_composes_with_platt(tmp_path):
    """Same composition guarantee for the Platt path (R7): reloaded raw proba +
    reloaded PlattCalibrator reproduce the calibrated output exactly."""
    y, p_raw, model, X_ev = _miscalibrated_xgb_output(seed=6)
    dec = platt_calibration(y, p_raw, z_threshold=2.0)
    p_cal_before = apply_calibrator(p_raw, dec.calibrator)

    model.save(tmp_path / "model.ubj")
    with open(tmp_path / "calibration.pkl", "wb") as f:
        pickle.dump(dec.calibrator, f)

    loaded = XGBoostModel.load(tmp_path / "model.ubj",
                               feature_names=model.feature_names)
    with open(tmp_path / "calibration.pkl", "rb") as f:
        cal_loaded = pickle.load(f)

    p_raw_reloaded = loaded.predict_proba(X_ev)
    p_cal_after = apply_calibrator(p_raw_reloaded, cal_loaded)
    np.testing.assert_array_equal(p_cal_before, p_cal_after)
