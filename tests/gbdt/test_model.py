"""Stage 4 — CatBoost wrapper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.model import GBDTModel, PINNED_HPS


def _toy_dataset(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "good_feat": rng.normal(0, 1, size=n),
        "noise1": rng.normal(0, 1, size=n),
        "noise2": rng.normal(0, 1, size=n),
    })
    # Target: 1 iff good_feat > 0, with a touch of noise
    y = ((X["good_feat"] + rng.normal(0, 0.1, n)) > 0).astype(int).values
    return X, y


def test_fit_predict_proba_shape():
    X, y = _toy_dataset(400, seed=1)
    m = GBDTModel(
        {"iterations": 50, "depth": 4, "learning_rate": 0.1, "boosting_type": "Plain"},
    )
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    p = m.predict_proba(X.iloc[300:])
    assert p.shape == (100,)
    assert ((p >= 0) & (p <= 1)).all()


def test_dominant_feature_lights_up_in_native_importance():
    X, y = _toy_dataset(800, seed=2)
    m = GBDTModel({"iterations": 100, "depth": 4, "learning_rate": 0.1,
                   "boosting_type": "Plain"})
    m.fit(X.iloc[:600], y[:600], X.iloc[600:], y[600:])
    imp = m.feature_importance("native")
    assert imp.idxmax() == "good_feat"


def test_has_time_is_pinned_true():
    m = GBDTModel({"iterations": 10, "boosting_type": "Plain"})
    assert m.hp["has_time"] is True


def test_has_time_override_rejected():
    with pytest.raises(ValueError, match="has_time"):
        GBDTModel({"iterations": 10, "has_time": False})


def test_loss_function_override_rejected():
    with pytest.raises(ValueError, match="loss_function"):
        GBDTModel({"iterations": 10, "loss_function": "CrossEntropy"})


def test_hp_out_of_range_rejected_numeric():
    with pytest.raises(ValueError, match="depth"):
        GBDTModel({"iterations": 10, "depth": 50})


def test_hp_out_of_range_rejected_enum():
    with pytest.raises(ValueError, match="boosting_type"):
        GBDTModel({"iterations": 10, "boosting_type": "Magical"})


def test_hp_out_of_range_rejected_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        GBDTModel({"iterations": 10, "learning_rate": 5.0})


def test_save_and_load_round_trip(tmp_path):
    X, y = _toy_dataset(400, seed=3)
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    p_before = m.predict_proba(X.iloc[300:])
    out_path = tmp_path / "m.cbm"
    m.save(out_path)
    loaded = GBDTModel.load(out_path)
    p_after = loaded.predict_proba(X.iloc[300:])
    assert np.allclose(p_before, p_after)


def test_class_weights_passthrough():
    """auto_class_weights kwarg should round-trip via the wrapper without raising."""
    X, y = _toy_dataset(300, seed=4)
    m = GBDTModel({"iterations": 20, "depth": 4, "boosting_type": "Plain",
                   "auto_class_weights": "SqrtBalanced"})
    m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    assert m.hp["auto_class_weights"] == "SqrtBalanced"


def test_permutation_importance_non_negative():
    X, y = _toy_dataset(400, seed=5)
    m = GBDTModel({"iterations": 50, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    imp = m.feature_importance("permutation", X.iloc[300:], y[300:])
    assert (imp >= 0).all()
