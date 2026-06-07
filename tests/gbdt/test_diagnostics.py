"""Stage 6 — diagnostic bundle + FS+HP loop helpers tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gbdt.diagnostics import (
    DiagnosticBundle,
    build_diagnostic_bundle,
    reliability_curve,
    top_k_correlation,
)
from gbdt.fs_hp_loop import best_checkpoint, inner_stop_check
from gbdt.model import GBDTModel


def _toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })
    y = ((X["f1"] + rng.normal(0, 0.1, n)) > 0).astype(int).values
    return X, y


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------


def test_reliability_curve_bins():
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95])
    rel = reliability_curve(y, p, n_bins=5)
    assert rel["n_bins"] == 5
    assert len(rel["points"]) == 5
    # Each bin has the documented schema keys
    for pt in rel["points"]:
        for k in ("bin_low", "bin_high", "mean_pred", "frac_positive", "count"):
            assert k in pt


def test_top_k_correlation_returns_dict():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (100, 4)),
                      columns=["a", "b", "c", "d"])
    imp = pd.Series([1.0, 0.5, 0.1, 0.05], index=["a", "b", "c", "d"])
    out = top_k_correlation(X, imp, k=3)
    assert set(out.keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# build_diagnostic_bundle end-to-end
# ---------------------------------------------------------------------------


def test_build_diagnostic_bundle_contains_expected_keys():
    X, y = _toy_data(400)
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    bundle = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:250], y_train=y[:250],
        X_val=X.iloc[250:], y_val=y[250:],
        include_permutation=True,
    )
    d = bundle.to_dict()
    expected_keys = {
        "iter", "hp", "features", "n_features",
        "train_brier", "val_brier", "train_val_gap",
        "eval_brier_provisional", "calibration",
        "early_stop_iteration", "iteration_cap_hit",
        "importance_native", "importance_permutation",
        "top_feature_correlation", "learning_curve",
        "hp_history", "rationale", "delta_attribution", "wall_time_sec",
    }
    assert expected_keys.issubset(set(d.keys()))
    # Calibration sub-keys
    for k in ("spiegelhalter_z", "spiegelhalter_p", "reliability",
              "positive_prevalence_val", "positive_recall_val"):
        assert k in d["calibration"]


def test_build_diagnostic_bundle_serializes_to_json():
    X, y = _toy_data(300)
    m = GBDTModel({"iterations": 20, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    bundle = build_diagnostic_bundle(
        model=m, iter_idx=1, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:200], y_train=y[:200],
        X_val=X.iloc[200:], y_val=y[200:],
        include_permutation=False,
    )
    payload = bundle.to_dict()
    # Round-trip
    s = json.dumps(payload)
    re = json.loads(s)
    assert re["iter"] == 1


def test_bundle_permutation_importance_non_negative():
    X, y = _toy_data(300)
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    b = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:200], y_train=y[:200],
        X_val=X.iloc[200:], y_val=y[200:],
        include_permutation=True,
    )
    assert all(v >= 0 for v in b.importance_permutation.values())


# ---------------------------------------------------------------------------
# inner_stop_check
# ---------------------------------------------------------------------------


def test_inner_stop_cap_fires():
    history = [0.25, 0.24, 0.235, 0.232, 0.231, 0.230, 0.229, 0.228]
    stop, signal = inner_stop_check(history, max_iterations=8, plateau_threshold=0.0001)
    assert stop and signal == "cap"


def test_inner_stop_plateau_fires():
    history = [0.30, 0.28, 0.279, 0.278]
    stop, signal = inner_stop_check(history, plateau_threshold=0.005)
    assert stop and signal == "plateau"


def test_inner_stop_degradation_fires():
    history = [0.30, 0.25, 0.30]  # latest is 0.30 > 1.01*0.25
    stop, signal = inner_stop_check(history, degradation_gate=0.01)
    assert stop and signal == "degradation"


def test_inner_stop_no_signal_when_improving():
    history = [0.30, 0.28, 0.26, 0.23]
    stop, signal = inner_stop_check(history, plateau_threshold=0.005,
                                    degradation_gate=0.01, max_iterations=8)
    assert not stop
    assert signal is None


# Task #204 — disable_plateau (agent mode) gates the plateau signal off.
# degradation + cap MUST still fire so the loop is still bounded + still
# regresses-out-of cleanly.

def test_inner_stop_plateau_suppressed_when_disabled():
    # The same history that triggers plateau in default mode (above) must NOT
    # stop when disable_plateau=True.
    history = [0.30, 0.28, 0.279, 0.278]
    stop, signal = inner_stop_check(history, plateau_threshold=0.005,
                                    disable_plateau=True)
    assert not stop
    assert signal is None


def test_inner_stop_degradation_fires_when_plateau_disabled():
    # Regression IS a real stop signal — agent can't recover from val
    # blowing up without backtracking.
    history = [0.30, 0.25, 0.30]
    stop, signal = inner_stop_check(history, degradation_gate=0.01,
                                    disable_plateau=True)
    assert stop and signal == "degradation"


def test_inner_stop_cap_fires_when_plateau_disabled():
    # Cap bounds the loop in both modes — even with the agent driving,
    # max_iterations is a hard ceiling.
    history = [0.25, 0.24, 0.235, 0.232, 0.231, 0.230, 0.229, 0.228]
    stop, signal = inner_stop_check(history, max_iterations=8,
                                    plateau_threshold=0.0001,
                                    disable_plateau=True)
    assert stop and signal == "cap"


def test_best_checkpoint_picks_min():
    history = [0.30, 0.27, 0.24, 0.26, 0.25]
    # V1.4 P2: best_checkpoint returns (best_idx, tiebreak_path_label).
    best_idx, path = best_checkpoint(history)
    assert best_idx == 2
    assert path == "strict_val_brier"
