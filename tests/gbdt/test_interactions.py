"""V1.2 Phase 4 — XGBoost feature-interaction methodology.

Locks in the Phase-4 acceptance criteria (``docs/gbdt/V1.2_xgboost_feature_
interactions_plan.md`` § 8 Phase-4 row + the per-phase test strategy):

> *SHAP-interaction aggregator on a small hand-crafted XGBoost fixture: a known
> interaction (e.g. ``y = xor(a>0, b>0)``) shows ``(a,b)`` as the top pair with
> high sign-consistency; the streamed aggregate matches a dense reference on a
> tiny matrix; the co-occurrence cross-check returns a ranking; the
> ``interaction_constraints`` ablation reduces the measured interaction to ~0 and
> degrades Brier on the XOR fixture.*

The headline measurement is XGBoost's **native** TreeSHAP
(``booster.predict(pred_interactions=True)``) — **not** the external ``shap``
package (which stays optional / viz-only, D1 / R4). These tests assert the native
path works and that ``shap`` is never imported on the core path.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from gbdt.interactions import (
    InteractionResult,
    ablate_interactions,
    interaction_strength,
    shap_interaction_dense_reference,
)
from gbdt.model import XGBoostModel, make_model

# Deterministic XGBoost HP for the hand-crafted fixtures — small + reproducible.
_HP_XGB = {
    "n_estimators": 120,
    "max_depth": 4,
    "eta": 0.3,
    "lambda": 1.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _xor_dataset(n: int = 1200, seed: int = 7):
    """``y = xor(a>0, b>0)`` with two pure-noise features ``c``, ``d``.

    The signal lives *entirely* in the (a, b) interaction: neither ``a`` nor
    ``b`` alone is predictive (each is marginally ~50/50), but their joint
    quadrant fully determines ``y``. This is the canonical interaction the
    Phase-4 acceptance row calls for.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    d = rng.normal(size=n)
    y = ((a > 0) ^ (b > 0)).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    return X, y


def _fit_xor_model(n: int = 1200, seed: int = 7) -> tuple[XGBoostModel, pd.DataFrame, np.ndarray]:
    X, y = _xor_dataset(n=n, seed=seed)
    model = make_model("xgboost", dict(_HP_XGB), feature_names=list(X.columns))
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# kind="shap" — the headline measurement
# ---------------------------------------------------------------------------


def test_xor_surfaces_ab_as_top_pair_with_high_sign_consistency():
    """The XOR fixture must surface ``(a, b)`` as the #1 interaction pair, with
    high sign-consistency (a stable, not averaging-to-zero, interaction)."""
    model, X, _y = _fit_xor_model()
    result = interaction_strength(model, X, kind="shap", top_n=10)

    assert isinstance(result, InteractionResult)
    assert result.method == "shap"
    assert result.n_features == 4
    assert result.n_rows_used == len(X)

    top_a, top_b, top_strength, top_sign = result.top_pairs[0]
    assert {top_a, top_b} == {"a", "b"}, (
        f"expected (a,b) as the top interaction; got {result.top_pairs[:3]}"
    )
    # The (a,b) interaction must dominate the noise pairs by a wide margin.
    second_strength = result.top_pairs[1][2]
    assert top_strength > 3 * second_strength, (
        f"(a,b) strength {top_strength} should dominate the runner-up "
        f"{second_strength}"
    )
    # High sign-consistency: the interaction points the same way on most rows.
    assert top_sign >= 0.8, f"sign_consistency {top_sign} too low for a stable XOR"


def test_per_feature_involvement_and_main_effect_carried():
    """The result carries both interaction load (Σ off-diagonal) and main-effect
    (mean |diagonal SHAP|) per feature — the D7 pruning-rule inputs."""
    model, X, _y = _fit_xor_model()
    result = interaction_strength(model, X, kind="shap")

    # a and b carry far more interaction load than the noise features c, d.
    inv = result.per_feature_involvement
    assert set(inv) == {"a", "b", "c", "d"}
    assert inv["a"] > 3 * inv["c"]
    assert inv["b"] > 3 * inv["d"]

    # main-effect map present and keyed on all features.
    me = result.per_feature_main_effect
    assert set(me) == {"a", "b", "c", "d"}
    # In pure XOR the marginal main effects are weak relative to the interaction
    # load — exactly the "low main effect, high interaction load → keep" case.
    assert inv["a"] > me["a"]


def test_streamed_aggregate_matches_dense_reference():
    """The streamed pair aggregation (never materialising the full
    rows×F×F tensor) must match the dense one-shot reference on a tiny matrix."""
    model, X, _y = _fit_xor_model(n=300, seed=3)
    X_small = X.iloc[:64]  # tiny → dense reference is cheap + safe

    # streamed, with a deliberately small batch so several blocks are stitched
    streamed = interaction_strength(
        model, X_small, kind="shap", top_n=99, batch_size=7, max_rows=10_000
    )
    dense_M, names = shap_interaction_dense_reference(model, X_small)

    assert names == ["a", "b", "c", "d"]
    # Compare every pair's streamed strength against the dense matrix entry.
    for a, b, strength, _sign in streamed.top_pairs:
        i, j = names.index(a), names.index(b)
        assert strength == pytest.approx(dense_M[i, j], rel=1e-6, abs=1e-9), (
            f"streamed pair ({a},{b})={strength} != dense {dense_M[i, j]}"
        )


def test_max_rows_subsample_is_reproducible():
    """The row sub-sample (when len(X) > max_rows) is seeded → reproducible
    ranking; n_rows_used reflects the cap."""
    model, X, _y = _fit_xor_model(n=1200, seed=9)
    r1 = interaction_strength(model, X, kind="shap", max_rows=200, random_seed=123)
    r2 = interaction_strength(model, X, kind="shap", max_rows=200, random_seed=123)
    assert r1.n_rows_used == 200
    assert r1.top_pairs == r2.top_pairs


# ---------------------------------------------------------------------------
# kind="cooccurrence" — the cheap native cross-check
# ---------------------------------------------------------------------------


def test_cooccurrence_returns_a_ranking():
    """The co-occurrence cross-check returns a pair ranking (no per-row signed
    values) and surfaces (a,b) prominently on the XOR fixture."""
    model, X, _y = _fit_xor_model()
    result = interaction_strength(model, X, kind="cooccurrence", top_n=10)

    assert result.method == "cooccurrence"
    assert result.n_rows_used == 0  # no rows are scored
    assert result.per_feature_main_effect == {}  # no signed per-row notion
    assert len(result.top_pairs) >= 1
    # ranking is descending by strength
    strengths = [s for _a, _b, s, _sign in result.top_pairs]
    assert strengths == sorted(strengths, reverse=True)
    # sign_consistency is nan for the cheap cross-check
    assert all(np.isnan(sign) for _a, _b, _s, sign in result.top_pairs)
    # (a,b) co-occur structurally in the XOR trees → among the top pairs
    top_keys = [{a, b} for a, b, _s, _sign in result.top_pairs[:3]]
    assert {"a", "b"} in top_keys


def test_cooccurrence_and_shap_agree_on_xor_top_pair():
    """Both methods should rank (a,b) #1 on the XOR fixture — large agreement is
    the calibration cross-check (large disagreement would itself be diagnostic)."""
    model, X, _y = _fit_xor_model()
    shap_r = interaction_strength(model, X, kind="shap")
    cooc_r = interaction_strength(model, X, kind="cooccurrence")
    assert {shap_r.top_pairs[0][0], shap_r.top_pairs[0][1]} == {"a", "b"}
    assert {cooc_r.top_pairs[0][0], cooc_r.top_pairs[0][1]} == {"a", "b"}


# ---------------------------------------------------------------------------
# interaction_constraints causal ablation
# ---------------------------------------------------------------------------


def test_ablation_zeroes_interaction_and_degrades_brier():
    """Forbidding the top SHAP pair via interaction_constraints must (a) drop the
    measured (a,b) interaction to ~0 and (b) degrade Brier on the XOR fixture —
    closing the loop between measured interaction strength and causal
    contribution to predictability."""
    model, X, y = _fit_xor_model()

    base_result = interaction_strength(model, X, kind="shap")
    base_ab = base_result.pair_strength("a", "b")
    base_brier = brier_score_loss(y, model.predict_proba(X))

    ablated = ablate_interactions(model, X, y, [("a", "b")])
    assert isinstance(ablated, XGBoostModel)

    ablated_result = interaction_strength(ablated, X, kind="shap")
    ablated_ab = ablated_result.pair_strength("a", "b")
    ablated_brier = brier_score_loss(y, ablated.predict_proba(X))

    # (a) measured interaction collapses to ~0 (forbidden from co-splitting).
    assert ablated_ab < 0.05 * base_ab, (
        f"ablated (a,b) interaction {ablated_ab} not ~0 vs baseline {base_ab}"
    )
    # (b) Brier degrades — the interaction WAS load-bearing for predictability.
    assert ablated_brier > base_brier, (
        f"ablated Brier {ablated_brier} did not degrade vs baseline {base_brier}"
    )


def test_ablation_carries_constraints_and_validates_pairs():
    """The ablated model carries interaction_constraints, and a forbidden pair
    naming an unknown feature raises."""
    model, X, y = _fit_xor_model(n=400, seed=1)
    ablated = ablate_interactions(model, X, y, [("a", "b")])
    assert "interaction_constraints" in ablated.hp

    with pytest.raises(ValueError, match="not in the model's feature set"):
        ablate_interactions(model, X, y, [("a", "zzz_missing")])


# ---------------------------------------------------------------------------
# `shap` is NOT a hard dependency
# ---------------------------------------------------------------------------


def test_shap_package_not_imported_on_core_path():
    """The headline measurement is native TreeSHAP via the booster; the external
    `shap` package must never be imported by the core interaction path."""
    sys.modules.pop("shap", None)
    model, X, y = _fit_xor_model(n=300, seed=2)
    interaction_strength(model, X, kind="shap")
    interaction_strength(model, X, kind="cooccurrence")
    shap_interaction_dense_reference(model, X.iloc[:32])
    ablate_interactions(model, X, y, [("a", "b")])
    assert "shap" not in sys.modules, (
        "the external `shap` package was imported on the core path — it must "
        "stay optional/viz-only (D1/R4)."
    )


# ---------------------------------------------------------------------------
# misc API guards
# ---------------------------------------------------------------------------


def test_unknown_kind_raises():
    model, X, _y = _fit_xor_model(n=200, seed=4)
    with pytest.raises(ValueError, match="unknown interaction kind"):
        interaction_strength(model, X, kind="bogus")


def test_interaction_result_to_dict_is_json_shaped():
    model, X, _y = _fit_xor_model(n=200, seed=5)
    d = interaction_strength(model, X, kind="shap", top_n=3).to_dict()
    assert isinstance(d["top_pairs"], list)
    assert all(isinstance(p, list) and len(p) == 4 for p in d["top_pairs"])
    assert isinstance(d["per_feature_involvement"], dict)
    assert d["method"] == "shap"


def test_requires_fitted_xgboost_model():
    """interaction_strength rejects an unfitted model and a non-XGBoost model."""
    unfit = make_model("xgboost", dict(_HP_XGB))
    X, _y = _xor_dataset(n=50)
    with pytest.raises(RuntimeError, match="not fitted"):
        interaction_strength(unfit, X, kind="shap")
