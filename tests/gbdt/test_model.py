"""Stage 4 — CatBoost wrapper tests.

V1.2 Phase 2 (``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`` § 8 / D3)
adds the backend-conditional HP-name tables + the :func:`hp_tables_for` resolver;
the CatBoost-path tests above must stay byte-for-byte green (the default
``backend="catboost"`` path is unchanged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.model import (
    ENUM_HP_VALUES,
    ENUM_HP_VALUES_XGB,
    PINNED_HPS,
    PINNED_HPS_XGB,
    TUNABLE_HP_RANGES,
    TUNABLE_HP_RANGES_XGB,
    GBDTModel,
    hp_tables_for,
)
from gbdt.model import _validate_hp


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


# ---------------------------------------------------------------------------
# V1.2 Phase 2 — backend-conditional HP-name tables + hp_tables_for resolver
# ---------------------------------------------------------------------------


def test_hp_tables_for_catboost_returns_existing_tables():
    """``hp_tables_for("catboost")`` returns the existing CatBoost triple,
    unchanged — the default path is byte-for-byte identical."""
    tunable, enum_values, pinned = hp_tables_for("catboost")
    assert tunable is TUNABLE_HP_RANGES
    assert enum_values is ENUM_HP_VALUES
    assert pinned is PINNED_HPS
    # CatBoost-only names live here; XGBoost names do not.
    assert "depth" in tunable and "max_depth" not in tunable
    assert "has_time" in pinned


def test_hp_tables_for_xgboost_exposes_xgb_names():
    """``hp_tables_for("xgboost")`` returns the *_XGB siblings: XGBoost names
    present, CatBoost-only names absent, no ``has_time`` analogue."""
    tunable, enum_values, pinned = hp_tables_for("xgboost")
    assert tunable is TUNABLE_HP_RANGES_XGB
    assert enum_values is ENUM_HP_VALUES_XGB
    assert pinned is PINNED_HPS_XGB
    # XGBoost names present.
    for name in ("max_depth", "eta", "lambda", "alpha", "colsample_bytree",
                 "min_child_weight", "gamma", "max_bin", "n_estimators"):
        assert name in tunable, f"missing XGBoost HP {name!r}"
    assert "grow_policy" in enum_values  # lowercase XGBoost values
    # CatBoost-only names rejected (not in the XGBoost vocab).
    for cb_only in ("depth", "learning_rate", "l2_leaf_reg", "iterations",
                    "rsm", "bootstrap_type", "border_count"):
        assert cb_only not in tunable and cb_only not in enum_values, (
            f"CatBoost-only HP {cb_only!r} leaked into the XGBoost table"
        )
    # No has_time analogue on the XGBoost side; determinism pins replace it.
    assert "has_time" not in pinned
    for det in ("tree_method", "n_jobs", "device", "objective"):
        assert det in pinned, f"missing XGBoost pin {det!r}"


def test_hp_tables_for_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        hp_tables_for("lightgbm")


def test_validate_hp_default_backend_is_catboost():
    """``_validate_hp`` with no backend arg behaves exactly as before the seam:
    CatBoost pins applied, CatBoost ranges enforced."""
    out = _validate_hp({"iterations": 100, "depth": 6})
    assert out["has_time"] is True               # CatBoost pin applied
    assert out["loss_function"] == "Logloss"
    # A CatBoost-only override still raises.
    with pytest.raises(ValueError, match="has_time"):
        _validate_hp({"iterations": 10, "has_time": False})


def test_validate_hp_xgboost_applies_pins_and_ranges():
    """``_validate_hp(..., backend="xgboost")`` applies XGBoost pins + ranges."""
    out = _validate_hp({"n_estimators": 1000, "max_depth": 6, "eta": 0.05},
                       backend="xgboost")
    assert out["objective"] == "binary:logistic"  # XGBoost pin applied
    assert out["tree_method"] == "exact"
    assert out["n_jobs"] == 1 and out["device"] == "cpu"


def test_validate_hp_xgboost_rejects_out_of_range():
    with pytest.raises(ValueError, match="max_depth"):
        _validate_hp({"max_depth": 50}, backend="xgboost")


def test_validate_hp_xgboost_rejects_pinned_override():
    with pytest.raises(ValueError, match="tree_method"):
        _validate_hp({"tree_method": "hist"}, backend="xgboost")


def test_validate_hp_xgboost_rejects_enum():
    with pytest.raises(ValueError, match="grow_policy"):
        _validate_hp({"grow_policy": "Magical"}, backend="xgboost")
