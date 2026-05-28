"""Unit tests for the pure helpers in scripts.gbdt.diagnose.

These exercise the diagnostic math on synthetic data (no model fit, no I/O),
so they're fast and deterministic.
"""
from __future__ import annotations

import numpy as np

from scripts.gbdt.diagnose import (
    assess_overfit,
    constraint_advice,
    pdp_monotonicity,
    prevalence_drift,
    spearman_monotonicity,
)


def test_assess_overfit_negative_gap_is_no_overfit():
    # The nifty50 regression: gap -0.0048 (val better than train) => NO overfit,
    # regardless of whether early-stopping fired. Early-stop is not a factor here.
    assert assess_overfit(-0.0048) is True


def test_assess_overfit_positive_gap_is_overfit():
    assert assess_overfit(0.05) is False


def test_assess_overfit_below_threshold_is_no_overfit():
    assert assess_overfit(0.01) is True  # mild, below the 0.02 trigger


def test_assess_overfit_none_gap():
    assert assess_overfit(None) is None


def test_spearman_monotonicity_increasing():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 5000)
    # higher x -> higher P(y=1)
    y = (rng.uniform(0, 1, 5000) < x).astype(int)
    r = spearman_monotonicity(x, y)
    assert r["rho"] > 0.3
    assert r["consistency"] >= 0.75
    assert r["pr_hi"] > r["pr_lo"]


def test_spearman_monotonicity_non_monotone_u_shape():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 8000)
    # U-shape: positives at both extremes -> low decile consistency
    y = (rng.uniform(0, 1, 8000) < (x ** 2)).astype(int)
    r = spearman_monotonicity(x, y)
    assert r["consistency"] < 0.75  # not cleanly monotone


def test_spearman_monotonicity_degenerate():
    x = np.ones(100)
    y = np.zeros(100, dtype=int)
    r = spearman_monotonicity(x, y)
    assert np.isnan(r["rho"])


def test_pdp_monotonicity_increasing():
    curve = np.array([0.1, 0.2, 0.25, 0.3, 0.4])
    m = pdp_monotonicity(curve)
    assert m["monotone"] and m["increasing"] and not m["decreasing"]
    assert m["worst_dip_frac"] == 0.0


def test_pdp_monotonicity_inverted_u():
    curve = np.array([0.1, 0.3, 0.4, 0.3, 0.15])  # rises then falls
    m = pdp_monotonicity(curve)
    assert not m["monotone"]
    assert m["worst_dip_frac"] < 0  # there is a downward step


def test_prevalence_drift_monotone_decline():
    d = prevalence_drift({"train": 0.28, "val": 0.20, "eval": 0.14, "test": 0.20})
    assert d["spread"] > 0.13
    assert d["drift_flag"]  # spread is a large fraction of mean
    # not a strict monotone decline because test rebounds above eval
    assert d["monotone_decline"] is False


def test_prevalence_drift_strict_decline():
    d = prevalence_drift({"train": 0.30, "val": 0.22, "eval": 0.15, "test": 0.10})
    assert d["monotone_decline"] is True


def test_prevalence_drift_stable():
    d = prevalence_drift({"train": 0.20, "val": 0.205, "eval": 0.198, "test": 0.202})
    assert not d["drift_flag"]


def test_constraint_advice_high_interaction_avoids():
    marg = {"rho": 0.18, "consistency": 1.0}
    pdp = {"monotone": True}
    advice = constraint_advice(marg, pdp, involvement=9.0, involvement_high_thr=2.0)
    assert "AVOID" in advice and "interaction" in advice


def test_constraint_advice_inverted_u_avoids():
    marg = {"rho": 0.16, "consistency": 0.8}
    pdp = {"monotone": False}
    advice = constraint_advice(marg, pdp, involvement=0.0, involvement_high_thr=2.0)
    assert "AVOID" in advice


def test_constraint_advice_low_interaction_neutral():
    marg = {"rho": 0.12, "consistency": 1.0}
    pdp = {"monotone": True}
    advice = constraint_advice(marg, pdp, involvement=0.0, involvement_high_thr=2.0)
    assert "NEUTRAL" in advice
