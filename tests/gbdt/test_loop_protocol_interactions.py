"""Tests for the agent-loop ``validate_decision`` extensions in #184 + L2 (_187):

- ``interaction_constraints`` whitelisted for XGBoost (structured HP); rejected
  for CatBoost (symmetric to how ``monotone_constraints`` is rejected for XGB).
- ``min_child_weight`` already in ``TUNABLE_HP_RANGES_XGB`` — verify it stays
  accepted under ``backend="xgboost"`` and rejected as unknown under
  ``"catboost"`` (XGBoost-only knob).
- ``KNOB_CANDIDATES_XGB`` registry exposes the curated mcw grid from ``_187``.
"""
from __future__ import annotations

import pytest

from gbdt.loop_protocol import DecisionError, validate_decision
from gbdt.model import KNOB_CANDIDATES_XGB, STRUCTURED_HP_KEYS_XGB


_KNOWN = [
    "sig", "n1", "n2",
    "realized_vol_200", "realized_vol_50",
    "parkinson_20", "garman_klass_50",
    "index_vol_50", "index_vol_200",
    "beta_120", "moy_sin", "moy_cos",
]


# ---------------------------------------------------------------------------
# interaction_constraints — XGBoost only (#184)
# ---------------------------------------------------------------------------

def test_validate_accepts_interaction_constraints_xgb():
    decision = {"hp_changes": {"interaction_constraints":
                                [["index_vol_50", "index_vol_200"]]}}
    validate_decision(decision, spec=None, known_features=_KNOWN,
                      backend="xgboost")  # no raise


def test_validate_accepts_multi_group_interaction_constraints_xgb():
    decision = {"hp_changes": {"interaction_constraints": [
        ["index_vol_50", "index_vol_200"],
        ["realized_vol_50", "realized_vol_200"],
        ["parkinson_20"],
    ]}}
    validate_decision(decision, spec=None, known_features=_KNOWN,
                      backend="xgboost")  # no raise


def test_validate_accepts_none_interaction_constraints_xgb():
    decision = {"hp_changes": {"interaction_constraints": None}}
    validate_decision(decision, spec=None, known_features=_KNOWN,
                      backend="xgboost")  # no raise


def test_validate_accepts_empty_interaction_constraints_xgb():
    decision = {"hp_changes": {"interaction_constraints": []}}
    validate_decision(decision, spec=None, known_features=_KNOWN,
                      backend="xgboost")  # no raise


def test_validate_rejects_interaction_constraints_unknown_feature_xgb():
    decision = {"hp_changes": {"interaction_constraints":
                                [["index_vol_50", "not_a_real_feature"]]}}
    with pytest.raises(DecisionError, match="unknown feature"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="xgboost")


def test_validate_rejects_interaction_constraints_unknown_feature_message_carries_name():
    decision = {"hp_changes": {"interaction_constraints":
                                [["index_vol_50", "made_up_feature"]]}}
    with pytest.raises(DecisionError, match="made_up_feature"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="xgboost")


def test_validate_rejects_interaction_constraints_not_list_of_lists():
    # outer is not a list
    decision = {"hp_changes": {"interaction_constraints": "index_vol_50"}}
    with pytest.raises(DecisionError, match="list of groups"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="xgboost")


def test_validate_rejects_interaction_constraints_inner_not_list():
    # outer list, inner is a scalar
    decision = {"hp_changes": {"interaction_constraints": ["index_vol_50"]}}
    with pytest.raises(DecisionError, match="group must be a list"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="xgboost")


def test_validate_rejects_interaction_constraints_bool_element():
    # bool is a subclass of int — must NOT slip through.
    decision = {"hp_changes": {"interaction_constraints":
                                [[True, "index_vol_200"]]}}
    with pytest.raises(DecisionError, match="bool"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="xgboost")


def test_validate_rejects_interaction_constraints_wrong_element_type():
    decision = {"hp_changes": {"interaction_constraints":
                                [[1.5, "index_vol_200"]]}}
    with pytest.raises(DecisionError, match="strings or ints"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="xgboost")


def test_validate_rejects_interaction_constraints_on_catboost():
    # CatBoost has no parallel knob — falls through to "unknown HP" path.
    decision = {"hp_changes": {"interaction_constraints":
                                [["realized_vol_50", "parkinson_20"]]}}
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="catboost")


# ---------------------------------------------------------------------------
# min_child_weight — XGBoost only (L2 from _187)
# ---------------------------------------------------------------------------

def test_validate_accepts_min_child_weight_xgb():
    decision = {"hp_changes": {"min_child_weight": 10}}
    validate_decision(decision, spec=None, known_features=_KNOWN,
                      backend="xgboost")  # no raise


def test_validate_accepts_min_child_weight_xgb_candidate_grid_values():
    # The curated KNOB_CANDIDATES_XGB grid: (1, 5, 10). All must validate.
    for v in KNOB_CANDIDATES_XGB["min_child_weight"]:
        validate_decision(
            {"hp_changes": {"min_child_weight": v}},
            spec=None, known_features=_KNOWN, backend="xgboost",
        )  # no raise


def test_validate_rejects_min_child_weight_on_catboost():
    decision = {"hp_changes": {"min_child_weight": 10}}
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision(decision, spec=None, known_features=_KNOWN,
                          backend="catboost")


# ---------------------------------------------------------------------------
# Registry shape (KNOB_CANDIDATES_XGB / STRUCTURED_HP_KEYS_XGB)
# ---------------------------------------------------------------------------

def test_knob_candidates_xgb_includes_mcw_lesson():
    """L2 from _187: mcw=10 was the marginal val-best. Registry must surface
    a curated grid covering {default=1, middle=5, _187-best=10}."""
    grid = KNOB_CANDIDATES_XGB["min_child_weight"]
    assert 1 in grid and 5 in grid and 10 in grid


def test_structured_hp_keys_xgb_includes_interaction_constraints():
    assert "interaction_constraints" in STRUCTURED_HP_KEYS_XGB


def test_structured_hp_keys_xgb_does_not_leak_scalar_knobs():
    # Defensive: STRUCTURED_HP_KEYS_XGB is for non-scalar HPs only.
    # Scalar HPs (min_child_weight, max_depth, etc.) must stay out of it.
    assert "min_child_weight" not in STRUCTURED_HP_KEYS_XGB
    assert "max_depth" not in STRUCTURED_HP_KEYS_XGB
