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
    BaseGBDTModel,
    CatBoostModel,
    GBDTModel,
    XGBoostModel,
    count_nonfinite,
    hp_tables_for,
    make_model,
)
from gbdt.model import _validate_hp, _sanitize_nonfinite


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
    # V1.2 Phase 3: determinism knobs hard-fail with a determinism-specific msg.
    with pytest.raises(ValueError, match="determinism"):
        _validate_hp({"tree_method": "hist"}, backend="xgboost")
    # Policy pins (objective / eval_metric) reject too, generic message.
    with pytest.raises(ValueError, match="objective"):
        _validate_hp({"objective": "reg:squarederror"}, backend="xgboost")


def test_validate_hp_xgboost_rejects_enum():
    with pytest.raises(ValueError, match="grow_policy"):
        _validate_hp({"grow_policy": "Magical"}, backend="xgboost")


# ---------------------------------------------------------------------------
# V1.2 Phase 1 — XGBoostModel adapter (deterministic-by-construction)
#
# The XGBoost backend lands behind the V1.2 seam: same 7-member protocol as
# CatBoost, ``predict_proba`` contracted to 1-D P(positive), ``.ubj`` persistence,
# and bit-identical determinism (the R1 guard, pulled forward as a smoke test).
# Not wired into the runner/loop yet — reachable only via ``make_model`` here.
# ---------------------------------------------------------------------------


# The full abstract surface that ``train.py`` / ``diagnostics.py`` / the loop call.
_PROTOCOL_MEMBERS = (
    "fit",
    "predict_proba",
    "feature_importance",
    "best_iteration",
    "evals_result",
    "save",
    "load",
    "hp",
    "feature_names",
    "fitted",
)


@pytest.mark.parametrize("model_cls", [CatBoostModel, XGBoostModel])
def test_protocol_conformance_both_backends(model_cls):
    """Both concrete backends expose every ``BaseGBDTModel`` abstract member,
    and have no remaining un-implemented abstract methods (so they instantiate)."""
    assert issubclass(model_cls, BaseGBDTModel)
    # No abstract methods left → the class is concrete and constructible.
    assert not getattr(model_cls, "__abstractmethods__", set())
    for member in _PROTOCOL_MEMBERS:
        assert hasattr(model_cls, member), (
            f"{model_cls.__name__} is missing protocol member {member!r}"
        )


def test_xgboost_make_model_fits_and_predicts_1d_in_range():
    """``make_model("xgboost", hp)`` fits a tiny synthetic panel; ``predict_proba``
    returns a **1-D** array of P(positive) in [0, 1]."""
    X, y = _toy_dataset(400, seed=11)
    m = make_model(
        "xgboost",
        {"n_estimators": 50, "max_depth": 4, "eta": 0.1,
         "early_stopping_rounds": 10},
    )
    assert isinstance(m, XGBoostModel)
    assert isinstance(m, BaseGBDTModel)
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    p = m.predict_proba(X.iloc[300:])
    assert p.ndim == 1                       # 1-D contract
    assert p.shape == (100,)
    assert ((p >= 0) & (p <= 1)).all()


def test_xgboost_deterministic_pins_applied_on_construction():
    """The ``PINNED_HPS_XGB`` determinism knobs + a fixed seed are applied at
    construction (deterministic-by-construction; the Phase-3 hard-fail on a
    *different* value is covered by ``test_xgboost_nondeterministic_override_raises``)."""
    m = make_model("xgboost", {"n_estimators": 10, "max_depth": 3})
    assert m.hp["objective"] == "binary:logistic"
    assert m.hp["eval_metric"] == "logloss"
    assert m.hp["tree_method"] == "exact"
    assert m.hp["n_jobs"] == 1
    assert m.hp["device"] == "cpu"
    assert m.hp["seed"] == 42                # default random_seed mirrored to seed
    # A custom seed threads through to both spellings.
    m2 = XGBoostModel({"n_estimators": 10}, random_seed=7)
    assert m2.hp["seed"] == 7 and m2.hp["random_state"] == 7


def test_xgboost_dominant_feature_lights_up_in_native_importance():
    X, y = _toy_dataset(800, seed=12)
    m = make_model("xgboost", {"n_estimators": 100, "max_depth": 4, "eta": 0.1})
    m.fit(X.iloc[:600], y[:600], X.iloc[600:], y[600:])
    imp = m.feature_importance("native")
    assert imp.idxmax() == "good_feat"


def test_xgboost_permutation_importance_non_negative():
    X, y = _toy_dataset(400, seed=13)
    m = make_model("xgboost", {"n_estimators": 50, "max_depth": 4, "eta": 0.1})
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    imp = m.feature_importance("permutation", X.iloc[300:], y[300:])
    assert (imp >= 0).all()


def test_xgboost_save_load_round_trip_ubj(tmp_path):
    """save → load → predict_proba reproduces predictions exactly via ``.ubj``."""
    X, y = _toy_dataset(400, seed=14)
    m = make_model("xgboost", {"n_estimators": 40, "max_depth": 4, "eta": 0.1,
                               "early_stopping_rounds": 10})
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    p_before = m.predict_proba(X.iloc[300:])
    out_path = tmp_path / "m.ubj"
    m.save(out_path)
    assert out_path.exists()
    loaded = XGBoostModel.load(out_path)
    p_after = loaded.predict_proba(X.iloc[300:])
    assert np.array_equal(p_before, p_after)


def test_xgboost_bit_identity_determinism():
    """A2 guard (V1.2 Phase 3, plan § 8 / § 5.1 R1): two fits with the same
    ``(features, hp, seed)`` + identical row order produce **bit-identical**
    ``predict_proba`` outputs — the load-bearing finalization-retrain contract.

    Bit-identity (``np.array_equal``, NOT ``allclose``) is the assertion: the
    loop's finalization step retrains the selected ``(features, hp)`` config and
    must reproduce the in-loop fit exactly, or ``predictions/*.csv`` silently
    disagree with the val-Brier the checkpoint was selected on."""
    X, y = _toy_dataset(500, seed=15)
    hp = {"n_estimators": 60, "max_depth": 5, "eta": 0.1,
          "early_stopping_rounds": 10}
    m_a = make_model("xgboost", dict(hp))
    m_a.fit(X.iloc[:350], y[:350], X.iloc[350:], y[350:])
    m_b = make_model("xgboost", dict(hp))
    m_b.fit(X.iloc[:350], y[:350], X.iloc[350:], y[350:])
    p_a = m_a.predict_proba(X.iloc[350:])
    p_b = m_b.predict_proba(X.iloc[350:])
    assert np.array_equal(p_a, p_b)        # bit-identical, not merely close


def test_xgboost_nondeterministic_override_raises():
    """A2 guard, negative half (plan § 8): a ``tree_method="hist", n_jobs=4``
    override — the canonical non-deterministic combo — hard-fails at
    construction with a determinism-specific message, so it can never reach
    predict-time and silently corrupt the finalization retrain."""
    with pytest.raises(ValueError, match="determinism"):
        make_model("xgboost",
                   {"n_estimators": 10, "tree_method": "hist", "n_jobs": 4})
    # Each individual determinism knob also hard-fails on its own.
    for knob, bad in (("tree_method", "hist"), ("n_jobs", 4), ("device", "cuda")):
        with pytest.raises(ValueError, match=knob):
            make_model("xgboost", {"n_estimators": 10, knob: bad})


def test_xgboost_same_pinned_value_is_noop():
    """Passing the SAME pinned determinism value is a no-op, not an error
    (the hard-fail triggers only on a DIFFERENT value)."""
    m = make_model("xgboost", {"n_estimators": 10, "tree_method": "exact",
                               "n_jobs": 1, "device": "cpu"})
    assert m.hp["tree_method"] == "exact"
    assert m.hp["n_jobs"] == 1 and m.hp["device"] == "cpu"


def test_xgboost_evals_result_normalized_shape():
    """``evals_result`` is the nested ``{split: {metric: [...]}}`` shape
    ``diagnostics.py`` consumes, with the split relabeled to ``validation``."""
    X, y = _toy_dataset(300, seed=16)
    m = make_model("xgboost", {"n_estimators": 20, "max_depth": 3, "eta": 0.1,
                               "early_stopping_rounds": 5})
    m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    er = m.evals_result
    assert "validation" in er
    assert "logloss" in er["validation"]
    assert isinstance(m.best_iteration, int)


def test_xgboost_sample_weights_round_trip():
    """Per-row uniqueness weights (LdP §4.4) flow through fit without raising."""
    X, y = _toy_dataset(300, seed=17)
    rng = np.random.default_rng(0)
    w_tr = rng.uniform(0.5, 1.0, 200)
    w_val = rng.uniform(0.5, 1.0, 100)
    m = make_model("xgboost", {"n_estimators": 20, "max_depth": 3, "eta": 0.1,
                               "early_stopping_rounds": 5})
    m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:],
          train_weight=w_tr, val_weight=w_val)
    assert m.fitted
    p = m.predict_proba(X.iloc[200:])
    assert p.ndim == 1


def test_xgboost_predict_before_fit_raises():
    X, _ = _toy_dataset(50, seed=18)
    m = make_model("xgboost", {"n_estimators": 10})
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict_proba(X)


def test_xgboost_out_of_range_hp_rejected_on_construction():
    """The factory/adapter validates against the XGBoost ranges on construction."""
    with pytest.raises(ValueError, match="max_depth"):
        make_model("xgboost", {"n_estimators": 10, "max_depth": 50})


def test_xgboost_pinned_override_rejected_on_construction():
    with pytest.raises(ValueError, match="tree_method"):
        make_model("xgboost", {"n_estimators": 10, "tree_method": "hist"})


# ---------------------------------------------------------------------------
# Non-finite (±inf / overflow) robustness — the V1.2 sp500 DMatrix crash.
#
# XGBoost's ``XGDMatrixCreateFromDense`` rejects ``±inf`` ("Input data contains
# 'inf' or a value too large, while 'missing' is not set to 'inf'"), whereas
# CatBoost trains on the identical matrix fine. The gbdt feature pipeline can
# emit ``±inf`` from ratio/division families (e.g. ``stock_return_N`` /
# ``vol_change_N`` = ``s / s.shift(n) - 1`` on a zero prior-period value for a
# sparse/halted ticker). The backend must sanitize ``±inf`` → ``NaN`` (missing)
# on BOTH fit and predict so it tolerates the same matrices CatBoost does. The
# fix is general (not feature-specific). See ``model._sanitize_nonfinite`` +
# ``model.count_nonfinite`` + the fail-fast audit in ``gbdt.__main__``.
# ---------------------------------------------------------------------------


def _toy_dataset_with_nonfinite(n: int = 500, seed: int = 0):
    """Toy dataset salted with ``+inf``, ``-inf``, and an overflow value.

    The overflow (``1e308 * 10`` → ``inf`` under IEEE-754 double) stands in for
    the "value too large" half of the XGBoost error message.
    """
    X, y = _toy_dataset(n, seed=seed)
    X = X.copy()
    X.iloc[3, 0] = np.inf
    X.iloc[7, 1] = -np.inf
    X.iloc[11, 2] = 1e308 * 10        # → +inf via overflow
    X.iloc[13, 0] = -1e308 * 10       # → -inf via overflow
    return X, y


def test_sanitize_nonfinite_maps_inf_to_nan_preserves_finite():
    """``±inf`` (incl. overflow) → ``NaN``; finite values + existing NaN kept."""
    arr = np.array(
        [[1.0, np.inf], [-np.inf, 2.0], [np.nan, 1e308 * 10], [3.0, 4.0]],
        dtype=float,
    )
    out = _sanitize_nonfinite(arr)
    assert not np.isinf(out).any()                 # no ±inf remains
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])  # ±inf → NaN
    assert np.isnan(out[2, 1])                     # overflow → NaN
    assert np.isnan(out[2, 0])                     # pre-existing NaN preserved
    # Finite values are untouched.
    assert out[0, 0] == 1.0 and out[1, 1] == 2.0
    assert out[3, 0] == 3.0 and out[3, 1] == 4.0
    # Caller's array is not mutated in place.
    assert np.isinf(arr[0, 1])


def test_count_nonfinite_flags_inf_columns_only():
    """``count_nonfinite`` counts ``±inf`` per column (NOT ``NaN``), desc-sorted."""
    X = pd.DataFrame({
        "clean": [1.0, 2.0, 3.0, 4.0],
        "two_inf": [np.inf, 1.0, -np.inf, 2.0],
        "one_inf": [1.0, np.inf, 2.0, 3.0],
        "only_nan": [np.nan, 1.0, 2.0, np.nan],   # NaN must NOT be flagged
    })
    offenders = count_nonfinite(X)
    assert offenders == {"two_inf": 2, "one_inf": 1}   # clean + only_nan excluded
    # Descending-by-count order (two_inf before one_inf).
    assert list(offenders.keys()) == ["two_inf", "one_inf"]
    # A clean matrix returns an empty dict.
    assert count_nonfinite(X[["clean"]]) == {}


def test_xgboost_fit_predict_tolerates_nonfinite_like_catboost():
    """Regression for the sp500 crash: the XGBoost backend must fit + predict on
    a matrix carrying ``±inf`` / overflow values WITHOUT raising, producing
    finite, in-range probabilities — matching CatBoost's tolerance of the
    identical matrix."""
    X, y = _toy_dataset_with_nonfinite(500, seed=21)

    # Sanity check that this matrix would crash a raw, unsanitized XGBoost fit
    # (i.e. the test is exercising a real failure mode, not a no-op).
    import xgboost as xgb
    with pytest.raises(xgb.core.XGBoostError):
        xgb.XGBClassifier(n_estimators=10, tree_method="exact").fit(
            X.values, y
        )

    m = make_model("xgboost", {"n_estimators": 30, "max_depth": 4, "eta": 0.1})
    m.fit(X.iloc[:350], y[:350], X.iloc[350:], y[350:])   # must not raise
    p = m.predict_proba(X.iloc[350:])                     # incl. inf rows
    assert p.shape == (150,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_catboost_also_handles_same_nonfinite_matrix():
    """Cross-backend parity: CatBoost trains on the identical non-finite matrix
    (it always did) — confirms the XGBoost fix brings it to parity, not that we
    regressed CatBoost."""
    X, y = _toy_dataset_with_nonfinite(400, seed=22)
    m = make_model("catboost", {"iterations": 30, "depth": 4,
                                "learning_rate": 0.1, "boosting_type": "Plain"})
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])   # must not raise
    p = m.predict_proba(X.iloc[300:])
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_xgboost_nonfinite_predict_path_independent_of_fit():
    """Even if fit data is clean, ``predict_proba`` on a matrix with ``±inf``
    must not crash (the predict-side DMatrix is sanitized too)."""
    X_clean, y = _toy_dataset(400, seed=23)
    m = make_model("xgboost", {"n_estimators": 20, "max_depth": 3, "eta": 0.1})
    m.fit(X_clean.iloc[:300], y[:300], X_clean.iloc[300:], y[300:])
    X_dirty = X_clean.iloc[300:].copy()
    X_dirty.iloc[0, 0] = np.inf
    X_dirty.iloc[1, 1] = -np.inf
    p = m.predict_proba(X_dirty)                          # must not raise
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
