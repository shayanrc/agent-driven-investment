"""Unit tests for the L1 (val-Brier tie-break) extension to ``best_checkpoint``.

Motivation: ``_187`` — on the nasdaq100 +10%/25d/dd5% cell, 9 XGBoost configs
landed val Brier in a 0.6% band (0.16859–0.16967) while train-val gap varied
~9× and Spiegelhalter Z varied ~2×. The strict val-Brier ``argmin`` cannot
distinguish a clean low-gap config from an overfit one; the tie-break biases
toward the better-generalizing config without changing the strict-winner case.

These tests verify:
(a) the strict val-Brier winner is returned when no tie-break inputs are
    supplied (backwards-compatible);
(b) the strict val-Brier winner is returned when only one config falls inside
    the tie band (singleton tie set);
(c) among configs inside the band, lower ``train_val_gap`` wins;
(d) when gaps are equal, smaller ``|spiegelhalter_z|`` wins;
(e) when both are equal, earlier iteration index wins (deterministic);
(f) explicit ``tie_band`` overrides the default fraction-of-plateau fallback;
(g) ``tie_band=0.0`` disables tie-breaking entirely;
(h) None gap/z entries are treated as worst-case (never out-prefer present ones);
(i) the default tie band recovers the ``_187`` finding: d4/λ6/mcw10 wins over
    d8/λ3 at ~equal val Brier.
"""

from __future__ import annotations

import pytest

from gbdt.fs_hp_loop import (
    DEFAULT_TIE_BAND_FRACTION,
    _resolve_tie_band,
    best_checkpoint,
)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_best_checkpoint_no_tiebreak_inputs_is_strict_argmin():
    # Same fixture as test_diagnostics.test_best_checkpoint_picks_min.
    val_briers = [0.30, 0.27, 0.24, 0.26, 0.25]
    assert best_checkpoint(val_briers) == 2


def test_best_checkpoint_strict_winner_returned_when_singleton_tie_set():
    # Best is iter 2 at 0.20; no other config within 0.01 → strict winner.
    val_briers = [0.30, 0.28, 0.20, 0.25, 0.24]
    gaps = [0.001, 0.001, 0.05, 0.001, 0.001]  # iter 2 has worst gap
    zs = [1.0, 1.0, 10.0, 1.0, 1.0]
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.01,
        )
        == 2
    )


def test_best_checkpoint_tie_band_zero_disables_tiebreak():
    # iter 2 has strictly-lowest val Brier; even though iter 4 has a much
    # lower gap, tie_band=0 means we never expand the tie set.
    val_briers = [0.30, 0.28, 0.20, 0.25, 0.201]
    gaps = [0.05, 0.05, 0.05, 0.05, 0.001]
    zs = [5.0, 5.0, 5.0, 5.0, 0.5]
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.0,
        )
        == 2
    )


# ---------------------------------------------------------------------------
# Lower train-val gap wins
# ---------------------------------------------------------------------------


def test_best_checkpoint_lower_gap_wins_within_band():
    # iter 1 and iter 3 are both inside [0.200, 0.205]; iter 3 has lower gap.
    val_briers = [0.30, 0.200, 0.28, 0.203]
    gaps = [0.05, 0.04, 0.05, 0.005]
    zs = [1.0, 1.0, 1.0, 1.0]
    out = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert out == 3


# ---------------------------------------------------------------------------
# |spiegelhalter_z| closer to 0 breaks gap ties
# ---------------------------------------------------------------------------


def test_best_checkpoint_lower_abs_z_breaks_gap_ties():
    # Both inside band, equal gaps; iter 2 has |z|=0.5 vs iter 0 |z|=10.
    val_briers = [0.200, 0.30, 0.202]
    gaps = [0.005, 0.05, 0.005]
    zs = [10.0, 1.0, 0.5]
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.005,
        )
        == 2
    )


def test_best_checkpoint_abs_z_treats_negative_as_distance():
    # |z|=2 should rank between |z|=1 and |z|=5.
    val_briers = [0.200, 0.201, 0.202]
    gaps = [0.01, 0.01, 0.01]
    zs = [-5.0, 1.0, -2.0]  # iter 1 has |z|=1 (closest to 0)
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.005,
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Earlier iteration index breaks (gap, |z|) ties deterministically
# ---------------------------------------------------------------------------


def test_best_checkpoint_earlier_index_breaks_full_tie():
    val_briers = [0.200, 0.201, 0.202]
    gaps = [0.01, 0.01, 0.01]
    zs = [1.0, 1.0, 1.0]
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.005,
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Tie-band sourcing: plateau_threshold fallback vs explicit override
# ---------------------------------------------------------------------------


def test_resolve_tie_band_explicit_wins():
    assert _resolve_tie_band(tie_band=0.01, plateau_threshold=0.005) == 0.01


def test_resolve_tie_band_zero_explicit_disables():
    # Pass-through of 0.0 (disable) even when plateau_threshold is set.
    assert _resolve_tie_band(tie_band=0.0, plateau_threshold=0.005) == 0.0


def test_resolve_tie_band_falls_back_to_plateau_fraction():
    out = _resolve_tie_band(tie_band=None, plateau_threshold=0.005)
    assert out == pytest.approx(0.005 * DEFAULT_TIE_BAND_FRACTION)


def test_resolve_tie_band_returns_zero_when_neither_set():
    assert _resolve_tie_band(tie_band=None, plateau_threshold=None) == 0.0


def test_best_checkpoint_explicit_tie_band_overrides_plateau_default():
    # plateau_threshold-derived default would be 0.0025; pass tie_band=0.05
    # to explicitly widen the band and bring iter 2 into the tie set.
    val_briers = [0.200, 0.30, 0.240]
    gaps = [0.05, 0.05, 0.001]
    zs = [5.0, 5.0, 0.5]

    # With default (fraction-of-plateau) tie_band, iter 2 is outside the band
    # → strict winner iter 0.
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            plateau_threshold=0.005,  # default → tie_band=0.0025
        )
        == 0
    )
    # With an explicit widening, iter 2 enters the band and wins on gap.
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.05,
            plateau_threshold=0.005,
        )
        == 2
    )


# ---------------------------------------------------------------------------
# None gap / z entries don't out-prefer present-metric configs
# ---------------------------------------------------------------------------


def test_best_checkpoint_none_gap_treated_as_worst_case():
    # iter 1 and iter 2 inside band; iter 1's gap is None → iter 2 wins.
    val_briers = [0.30, 0.200, 0.203]
    gaps = [0.05, None, 0.001]
    zs = [1.0, 0.5, 0.5]
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.005,
        )
        == 2
    )


def test_best_checkpoint_none_z_treated_as_worst_case():
    # Equal gaps; iter 1's z is None → iter 2 wins on its present |z|.
    val_briers = [0.30, 0.200, 0.203]
    gaps = [0.05, 0.001, 0.001]
    zs = [1.0, None, 0.5]
    assert (
        best_checkpoint(
            val_briers,
            train_val_gaps=gaps,
            spiegelhalter_zs=zs,
            tie_band=0.005,
        )
        == 2
    )


# ---------------------------------------------------------------------------
# Empty / single-entry edge cases
# ---------------------------------------------------------------------------


def test_best_checkpoint_empty_raises():
    with pytest.raises(ValueError):
        best_checkpoint([])


def test_best_checkpoint_single_entry_returns_zero():
    assert best_checkpoint([0.25]) == 0
    assert (
        best_checkpoint(
            [0.25],
            train_val_gaps=[0.01],
            spiegelhalter_zs=[1.0],
            tie_band=0.005,
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Recovery of the _187 motivating finding
# ---------------------------------------------------------------------------


def test_best_checkpoint_recovers_187_finding():
    """The L1 motivating case from ``docs/gbdt/_187_*.md``.

    The Phase A+B grid spans val Brier 0.16859–0.16967 with train-val gap
    0.0016–0.0146 and Spiegelhalter Z 6.26–13.0. With default tie_band
    (0.5 * plateau_threshold = 0.0025), the entire 0.6% band collapses to
    one tie set; the hygiene config (d4/λ6/mcw10, gap 0.0017, |z|=9.4)
    should win over the strict val-Brier winner that has worse generalization.

    Indices (per the iter log in ``_187``):
        0 — depth-6 baseline:      vb=0.16888, gap=0.01304, z=10.3
        1 — depth-4 / λ3:          vb=0.16881, gap=0.00402, z=7.47
        2 — depth-8 / λ3:          vb=0.16926, gap=0.01462, z=11.0
        3 — d4 / λ1.5:             vb=0.16967, gap=0.00664, z=13.0
        4 — d4 / λ6:               vb=0.16922, gap=0.00163, z=6.26
        5 — d4/λ6 + colsample 0.8: vb=0.16931, gap=0.00540, z=11.45
        6 — d4/λ6 + subsample 1.0: vb=0.16907, gap=0.00680, z=11.63
        7 — d4/λ6 + mcw10:         vb=0.16859, gap=0.00170, z=9.41  <-- val-best
        8 — d4/λ6 + gamma 1.0:     vb=0.16932, gap=0.00210, z=7.17

    Strict argmin: iter 7 (vb=0.16859) — happens to also have a clean gap.
    But the user-relevant point is: at default tie_band the loop should
    converge on a low-gap, well-calibrated config. We assert the winner
    is one of the *hygiene* configs (iter 4 = d4/λ6, gap 0.00163; or iter
    7 = d4/λ6/mcw10, gap 0.00170 — both are clean), NOT the high-gap
    overfit configs (iter 0 = d6, gap 0.01304; iter 2 = d8, gap 0.01462).
    """
    val_briers = [
        0.16888,  # 0 d6 baseline
        0.16881,  # 1 d4
        0.16926,  # 2 d8
        0.16967,  # 3 d4 λ1.5
        0.16922,  # 4 d4 λ6  <-- lowest gap
        0.16931,  # 5 d4/λ6 + colsample 0.8
        0.16907,  # 6 d4/λ6 + subsample 1.0
        0.16859,  # 7 d4/λ6 + mcw10  <-- strict val-Brier winner
        0.16932,  # 8 d4/λ6 + gamma 1.0
    ]
    gaps = [
        0.01304, 0.00402, 0.01462, 0.00664, 0.00163,
        0.00540, 0.00680, 0.00170, 0.00210,
    ]
    zs = [10.3, 7.47, 11.0, 13.0, 6.26, 11.45, 11.63, 9.41, 7.17]

    # All 9 configs span 0.16859–0.16967 = 0.00108 absolute, well within the
    # default tie_band of 0.0025 → entire grid is one tie set. The lowest-gap
    # config (iter 4, gap 0.00163) should win.
    out = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        plateau_threshold=0.005,  # default → tie_band=0.0025
    )
    assert out == 4, (
        f"expected iter 4 (d4/λ6, gap 0.00163, |z|=6.26) to win at default "
        f"tie_band; got iter {out} (gap={gaps[out]}, |z|={abs(zs[out])})"
    )
    # And critically: the winner is NOT one of the high-gap overfit configs.
    assert gaps[out] <= 0.005, "winner should have low train-val gap"
