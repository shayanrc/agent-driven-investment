"""V1.2 Phase 1 — backend seam: BaseGBDTModel + make_model factory.

Locks in the Phase-1 contract (``docs/gbdt/V1.2_xgboost_feature_interactions_
plan.md`` § 4 / § 8 + the per-phase test strategy):

- ``make_model("catboost", ...)`` returns a ``CatBoostModel`` (== the public
  ``GBDTModel`` alias) and is an instance of the backend-agnostic
  ``BaseGBDTModel``.
- unknown backends raise (``NotImplementedError``) — no half-built XGBoost path.
- a model built via the factory produces **bit-identical** predictions to one
  constructed the old way (``GBDTModel(...)`` directly) on a fixed synthetic
  fixture — the determinism / behaviour-identity contract the finalization
  retrain rests on.
- both the protocol surface and the spec-like dispatch resolve correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.model import (
    BaseGBDTModel,
    CatBoostModel,
    GBDTModel,
    make_model,
)


def _toy_dataset(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "good_feat": rng.normal(0, 1, size=n),
        "noise1": rng.normal(0, 1, size=n),
        "noise2": rng.normal(0, 1, size=n),
    })
    y = ((X["good_feat"] + rng.normal(0, 0.1, n)) > 0).astype(int).values
    return X, y


_HP = {"iterations": 50, "depth": 4, "learning_rate": 0.1, "boosting_type": "Plain"}


def test_factory_returns_catboost_model():
    m = make_model("catboost", dict(_HP))
    assert isinstance(m, CatBoostModel)
    assert isinstance(m, BaseGBDTModel)
    # Back-compat: the public alias is the same concrete class.
    assert isinstance(m, GBDTModel)
    assert GBDTModel is CatBoostModel


def test_factory_rejects_unknown_backend():
    with pytest.raises(NotImplementedError, match="xgboost"):
        make_model("xgboost", dict(_HP))
    with pytest.raises(NotImplementedError):
        make_model("lightgbm", dict(_HP))


def test_factory_resolves_spec_like_backend():
    """make_model accepts a spec-like object exposing backend.library, not just
    a bare string — so the runner can thread the parsed spec through."""

    class _Spec:
        class backend:  # noqa: N801 — mimics spec.backend.library
            library = "catboost"

    m = make_model(_Spec(), dict(_HP))
    assert isinstance(m, CatBoostModel)

    # dict-shaped spec too
    m2 = make_model({"backend": {"library": "catboost"}}, dict(_HP))
    assert isinstance(m2, CatBoostModel)


def test_catboost_pins_and_validation_unchanged_through_factory():
    """The factory routes through the same construction/validation as the
    direct class — has_time stays pinned, overrides + out-of-range still raise."""
    m = make_model("catboost", {"iterations": 10, "boosting_type": "Plain"})
    assert m.hp["has_time"] is True
    with pytest.raises(ValueError, match="has_time"):
        make_model("catboost", {"iterations": 10, "has_time": False})
    with pytest.raises(ValueError, match="depth"):
        make_model("catboost", {"iterations": 10, "depth": 50})


def test_factory_predictions_bit_identical_to_direct_construction():
    """A CatBoost model built via make_model produces BIT-IDENTICAL predictions
    to one built the old way (GBDTModel(...) directly), given the same
    (hp, feature_names, random_seed) + row order.

    This is the behaviour-identity guarantee the finalization retrain rests on:
    introducing the factory indirection must not perturb determinism in any way.
    """
    X, y = _toy_dataset(400, seed=11)
    X_tr, y_tr = X.iloc[:300], y[:300]
    X_val, y_val = X.iloc[300:], y[300:]

    direct = GBDTModel(dict(_HP), feature_names=list(X.columns), random_seed=42)
    direct.fit(X_tr, y_tr, X_val, y_val)
    p_direct = direct.predict_proba(X_val)

    via_factory = make_model(
        "catboost", dict(_HP), feature_names=list(X.columns), random_seed=42
    )
    via_factory.fit(X_tr, y_tr, X_val, y_val)
    p_factory = via_factory.predict_proba(X_val)

    # Bit-identical, not merely close.
    assert np.array_equal(p_direct, p_factory)


def test_base_interface_member_surface():
    """Both the abstract base and the concrete CatBoost impl expose the 7-member
    backend-agnostic surface train.py / diagnostics.py call on a model."""
    expected = {
        "fit",
        "predict_proba",
        "feature_importance",
        "best_iteration",
        "evals_result",
        "hp",
        "feature_names",
        "fitted",
        "save",
        "load",
    }
    for member in expected:
        assert hasattr(BaseGBDTModel, member), f"base missing {member}"
        assert hasattr(CatBoostModel, member), f"catboost missing {member}"


def test_base_is_abstract():
    """BaseGBDTModel cannot be instantiated directly (it's the seam, not a
    backend)."""
    with pytest.raises(TypeError):
        BaseGBDTModel()  # type: ignore[abstract]
