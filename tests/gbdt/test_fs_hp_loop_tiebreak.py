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
    DEFAULT_TIE_BAND_ABSOLUTE,
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
    best_idx, path = best_checkpoint(val_briers)
    assert best_idx == 2
    assert path == "strict_val_brier"


def test_best_checkpoint_strict_winner_returned_when_singleton_tie_set():
    # Best is iter 2 at 0.20; no other config within 0.01 → strict winner.
    val_briers = [0.30, 0.28, 0.20, 0.25, 0.24]
    gaps = [0.001, 0.001, 0.05, 0.001, 0.001]  # iter 2 has worst gap
    zs = [1.0, 1.0, 10.0, 1.0, 1.0]
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.01,
    )
    assert best_idx == 2
    assert path == "strict_val_brier"


def test_best_checkpoint_tie_band_zero_disables_tiebreak():
    # iter 2 has strictly-lowest val Brier; even though iter 4 has a much
    # lower gap, tie_band=0 means we never expand the tie set.
    val_briers = [0.30, 0.28, 0.20, 0.25, 0.201]
    gaps = [0.05, 0.05, 0.05, 0.05, 0.001]
    zs = [5.0, 5.0, 5.0, 5.0, 0.5]
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.0,
    )
    assert best_idx == 2
    assert path == "strict_val_brier"


# ---------------------------------------------------------------------------
# Lower train-val gap wins
# ---------------------------------------------------------------------------


def test_best_checkpoint_lower_gap_wins_within_band():
    # iter 1 and iter 3 are both inside [0.200, 0.205]; iter 3 has lower gap.
    val_briers = [0.30, 0.200, 0.28, 0.203]
    gaps = [0.05, 0.04, 0.05, 0.005]
    zs = [1.0, 1.0, 1.0, 1.0]
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert out == 3
    assert path == "classic_l1"


# ---------------------------------------------------------------------------
# |spiegelhalter_z| closer to 0 breaks gap ties
# ---------------------------------------------------------------------------


def test_best_checkpoint_lower_abs_z_breaks_gap_ties():
    # Both inside band, equal gaps; iter 2 has |z|=0.5 vs iter 0 |z|=10.
    val_briers = [0.200, 0.30, 0.202]
    gaps = [0.005, 0.05, 0.005]
    zs = [10.0, 1.0, 0.5]
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert best_idx == 2
    assert path == "classic_l1"


def test_best_checkpoint_abs_z_treats_negative_as_distance():
    # |z|=2 should rank between |z|=1 and |z|=5.
    val_briers = [0.200, 0.201, 0.202]
    gaps = [0.01, 0.01, 0.01]
    zs = [-5.0, 1.0, -2.0]  # iter 1 has |z|=1 (closest to 0)
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert best_idx == 1
    assert path == "classic_l1"


# ---------------------------------------------------------------------------
# Earlier iteration index breaks (gap, |z|) ties deterministically
# ---------------------------------------------------------------------------


def test_best_checkpoint_earlier_index_breaks_full_tie():
    val_briers = [0.200, 0.201, 0.202]
    gaps = [0.01, 0.01, 0.01]
    zs = [1.0, 1.0, 1.0]
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert best_idx == 0
    assert path == "classic_l1"


# ---------------------------------------------------------------------------
# Tie-band sourcing: plateau_threshold fallback vs explicit override
# ---------------------------------------------------------------------------


def test_resolve_tie_band_explicit_wins():
    assert _resolve_tie_band(tie_band=0.01, plateau_threshold=0.005) == 0.01


def test_resolve_tie_band_zero_explicit_disables():
    # Pass-through of 0.0 (disable) even when plateau_threshold is set.
    assert _resolve_tie_band(tie_band=0.0, plateau_threshold=0.005) == 0.0


def test_resolve_tie_band_default_is_fixed_absolute_when_only_plateau_set():
    """Bug #223 fix: ``plateau_threshold`` is NO LONGER consulted by the
    resolver's default path. Even with a tiny plateau_threshold (the #204
    workaround), the tie band stays at the fixed absolute default — it
    cannot collapse to noise level."""
    out = _resolve_tie_band(tie_band=None, plateau_threshold=0.0001)
    assert out == DEFAULT_TIE_BAND_ABSOLUTE  # = 0.005, NOT 0.00005


def test_resolve_tie_band_default_when_only_plateau_005():
    # The historic plateau_threshold=0.005 used to derive tie_band=0.0025.
    # Now the resolver ignores plateau_threshold by default and returns the
    # fixed 0.005 absolute — slightly wider than the old derived 0.0025,
    # in the same ballpark, and matches the historic-v1 plateau value
    # itself (the level the L1 motivating evidence was calibrated against).
    assert _resolve_tie_band(tie_band=None, plateau_threshold=0.005) == 0.005


def test_resolve_tie_band_default_when_neither_set_is_absolute():
    """Decoupling: with neither knob set, return the fixed absolute default."""
    assert _resolve_tie_band(tie_band=None, plateau_threshold=None) == DEFAULT_TIE_BAND_ABSOLUTE
    # Calling without the kwarg at all hits the same default.
    assert _resolve_tie_band(tie_band=None) == DEFAULT_TIE_BAND_ABSOLUTE


def test_resolve_tie_band_default_constant_value():
    """Pin the absolute default value so accidental edits surface as
    test failures (matches the historic plateau_threshold default — the
    level the ``_187`` motivating evidence was calibrated against)."""
    assert DEFAULT_TIE_BAND_ABSOLUTE == 0.005


def test_best_checkpoint_explicit_tie_band_overrides_plateau_default():
    # Default tie_band (now fixed 0.005, decoupled from plateau_threshold):
    # iter 2 at 0.240 is OUTSIDE the band [0.200, 0.205] → strict winner iter 0.
    val_briers = [0.200, 0.30, 0.240]
    gaps = [0.05, 0.05, 0.001]
    zs = [5.0, 5.0, 0.5]

    # With default tie_band (0.005), iter 2 is outside the band [0.200, 0.205]
    # → strict winner iter 0 (singleton tie set short-circuit).
    best_idx_a, path_a = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        plateau_threshold=0.005,  # ignored by default-resolver post-#223
    )
    assert best_idx_a == 0
    assert path_a == "strict_val_brier"
    # With an explicit widening, iter 2 enters the band and wins on gap.
    best_idx_b, path_b = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.05,
        plateau_threshold=0.005,
    )
    assert best_idx_b == 2
    assert path_b == "classic_l1"


def test_best_checkpoint_anti_auc_workaround_does_not_collapse_tie_band():
    """Bug #223 regression test — the SKILL.md-recommended #204 workaround
    sets plateau_threshold=0.0001. Pre-fix, this collapsed the derived
    tie_band to 0.00005, narrower than the typical val_brier cluster span
    on anti-AUC cells. Post-fix, the tie_band stays at the fixed absolute
    default (0.005) and the val_brier cluster is correctly tied.

    Fixture mirrors the V1.3 A4 cell-5 trajectory (memo ``_222``): val_brier
    range 0.2243–0.2251 (0.0008 absolute spread). All 9 iters land within
    0.005 of min → all tied; among ties the higher eval R-p@1 wins per V1.3.
    Pre-fix, derived tie_band=0.00005 → only the strict argmin (iter 8,
    R-p@1=0.643) was returned — the user's manual mid-loop spec patch was
    the only thing that saved the cell-5 run. This test prevents regression
    of that surprise."""
    val_briers = [0.2628, 0.2251, 0.2246, 0.2247, 0.2246, 0.2248,
                  0.2243, 0.2246, 0.2243, 0.2245]
    eval_rp1 = [0.484, 0.508, 0.602, 0.627, 0.717, 0.586, 0.602, 0.717,
                0.643, 0.713]
    # Strict val-Brier argmin: tied between iter 6 + iter 8 (both 0.2243);
    # min picks iter 6. The V1.3 cell-5 finding: iter 4 (eval R-p@1=0.717)
    # is the right pick because it's tied within tie_band of the val-Brier
    # min AND has the highest eval R-p@1.
    # Apply the SKILL.md-recommended workaround (plateau_threshold=0.0001).
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=None, spiegelhalter_zs=None,
        plateau_threshold=0.0001,  # #204 workaround — tie_band MUST NOT collapse to 5e-5
        anti_auc_flag="true",
        eval_r_precision_at_1s=eval_rp1,
    )
    assert path == "anti_auc_eval_rp1"
    # Post-fix: iter 4 (or any other R-p@1=0.717 iter within the band) wins.
    # iter 4 is structurally first among ties; eval R-p@1 ties broken by
    # val_brier (lower wins) → iter 4 (0.2246) beats iter 7 (0.2246) only
    # by index after the val_brier tie itself. Implementation detail: among
    # multiple max-rp1 ties, val_brier asc then index asc; iter 4 and iter 7
    # both at 0.2246, iter 4 comes first → iter 4.
    assert out == 4, (
        f"expected iter 4 (R-p@1=0.717, val_brier=0.2246, within tie_band of "
        f"min 0.2243); got iter {out} with R-p@1={eval_rp1[out]}, "
        f"val_brier={val_briers[out]}. The pre-#223-fix derived tie_band "
        f"(0.00005) would have collapsed to iter 6 / 8 only."
    )


# ---------------------------------------------------------------------------
# None gap / z entries don't out-prefer present-metric configs
# ---------------------------------------------------------------------------


def test_best_checkpoint_none_gap_treated_as_worst_case():
    # iter 1 and iter 2 inside band; iter 1's gap is None → iter 2 wins.
    val_briers = [0.30, 0.200, 0.203]
    gaps = [0.05, None, 0.001]
    zs = [1.0, 0.5, 0.5]
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert best_idx == 2
    assert path == "classic_l1"


def test_best_checkpoint_none_z_treated_as_worst_case():
    # Equal gaps; iter 1's z is None → iter 2 wins on its present |z|.
    val_briers = [0.30, 0.200, 0.203]
    gaps = [0.05, 0.001, 0.001]
    zs = [1.0, None, 0.5]
    best_idx, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert best_idx == 2
    assert path == "classic_l1"


# ---------------------------------------------------------------------------
# Bug #216 — when NO tied iter presents L1 metrics, fall back to strict argmin
# ---------------------------------------------------------------------------


def test_best_checkpoint_216_all_tied_l1_metrics_none_returns_strict_argmin():
    """Bug #216 regression — the cell-5 agentloop reproduction.

    Pre-fix: when the resume checkpoint predates V1.3 Option A it doesn't
    carry train_val_gaps / spiegelhalter_zs, AND the final-finalize resume
    (force_stop=True) skips the loop body so no in-process bundle exists.
    Every tied iter's L1 sort key collapses to (inf, inf, i) and the
    "earliest iter index" deterministic fallback fires — picking the
    strictly-dominated worse-val_brier iter. The fix: when no tied iter
    presents ANY L1 metric, fall back to the strict val-Brier argmin
    (symmetric to the V1.3 anti-AUC fallback).

    Fixture mirrors the cell-5 trajectory (artifact dir
    ``wt-cell5-agentloop/results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop``):
    val_briers [0.2628, 0.2628, 0.2544, 0.2524, 0.2376, 0.2354, 0.2337];
    default tie_band 0.005 collapses {4, 5, 6} into a tie; iter_6 strictly
    dominates iter_5 (and iter_4) on val_brier and (when L1 were known) on
    gap + |z| too. Pre-fix returned iter 4 via earliest-index fallback;
    post-fix returns iter 6.
    """
    val_briers = [0.2628, 0.2628, 0.2544, 0.2524, 0.2376, 0.2354, 0.2337]
    # Pre-V1.3 resume case — all L1 metrics absent.
    gaps_all_none: list[float | None] = [None] * 7
    zs_all_none: list[float | None] = [None] * 7
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps_all_none,
        spiegelhalter_zs=zs_all_none,
        plateau_threshold=0.005,  # cell-5's spec value; tie_band defaults
                                  # to 0.005 absolute (post-#223).
    )
    assert path == "l1_fallthrough"
    assert out == 6, (
        f"expected iter 6 (strict val-Brier argmin) when no tied iter has L1 "
        f"metrics; got iter {out} (val_brier={val_briers[out]}). The bug "
        f"#216 pre-fix would have returned the earliest-index iter via the "
        f"deterministic fallback even though iter 6 strictly dominates on "
        f"val_brier."
    )


def test_best_checkpoint_216_pre_fix_behavior_demonstration():
    """Demonstrates what the pre-#216-fix would have picked, to make the
    regression concrete. We simulate the pre-fix code path explicitly by
    running the sort_key over the tied set ourselves and asserting it
    matches the buggy outcome — then run the real (post-fix) function and
    assert it returns the dominator instead."""
    val_briers = [0.2628, 0.2628, 0.2544, 0.2524, 0.2376, 0.2354, 0.2337]
    # Default tie_band = 0.005 → threshold = 0.2387; tied = {4, 5, 6}.
    tied = [i for i, v in enumerate(val_briers) if v <= 0.2337 + 0.005]
    assert tied == [4, 5, 6]
    # Pre-fix sort_key with all-None metrics → (inf, inf, i). Earliest
    # index = iter 4 wins — the buggy outcome.
    pre_fix_winner = min(tied, key=lambda i: (float("inf"), float("inf"), i))
    assert pre_fix_winner == 4, (
        "sanity: pre-fix earliest-index fallback would have picked iter 4 "
        "(or any other iter with worse val_brier than iter 6 inside the "
        "tied set)"
    )
    # Post-fix real call: strict argmin (iter 6) wins.
    post_fix_winner, post_fix_path = best_checkpoint(
        val_briers,
        train_val_gaps=[None] * 7,
        spiegelhalter_zs=[None] * 7,
        tie_band=0.005,
    )
    assert post_fix_path == "l1_fallthrough"
    assert post_fix_winner == 6, (
        f"post-fix should pick the strict val-Brier argmin (iter 6) when "
        f"no tied iter has L1 metrics present; got iter {post_fix_winner}"
    )


def test_best_checkpoint_216_some_tied_present_some_none_uses_l1():
    """Counter-test for the #216 fix: when at least ONE tied iter presents an
    L1 metric, the L1 path stays active (None entries still treated as
    worst-case — i.e. NEVER out-prefer present-metric configs). The new
    "all-None → strict argmin" branch must NOT swallow the mixed case.
    """
    # iter 5 + iter 6 both inside the band; iter 6 has present gap, iter 5
    # has None. Iter 6 should win (lower val_brier + present finite L1).
    val_briers = [0.30, 0.30, 0.30, 0.30, 0.30, 0.235, 0.234]
    gaps = [None, None, None, None, None, None, 0.039]
    zs = [None, None, None, None, None, None, 3.37]
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert path == "classic_l1"
    assert out == 6, (
        f"expected iter 6 (present L1 with finite gap/z) to beat iter 5 "
        f"(None L1, worst-case sentinel); got iter {out}"
    )


def test_best_checkpoint_216_no_regression_on_v13_anti_auc():
    """Cross-check that V1.3 Option A's anti-AUC auto-disable still works
    after the #216 fix — the anti-AUC branch returns BEFORE the new
    all-None check, so its behavior is unchanged."""
    val_briers = [0.250, 0.249, 0.248]
    # All L1 metrics None — same shape as the cell-5 trajectory.
    out_anti_auc, path_anti_auc = best_checkpoint(
        val_briers,
        train_val_gaps=[None, None, None],
        spiegelhalter_zs=[None, None, None],
        tie_band=0.005,
        anti_auc_flag="true",  # V1.3 auto-disable engaged
        eval_r_precision_at_1s=None,  # no R-p@1 series → strict_best fallback
    )
    # When the anti-AUC branch falls back to strict argmin (no R-p@1 series
    # for the tied set), the path label reports the behavioural outcome
    # (``strict_val_brier``) — the chosen iter is the strict-argmin winner,
    # not an L1 or eval-R-p@1 tie-break result.
    assert path_anti_auc == "strict_val_brier"
    assert out_anti_auc == 2, (
        f"anti-AUC branch should fall back to strict argmin (iter 2) when "
        f"R-p@1 unavailable; got iter {out_anti_auc}"
    )
    # And with R-p@1 supplied: iter 0 has highest R-p@1 → wins.
    out_anti_auc_rp1, path_anti_auc_rp1 = best_checkpoint(
        val_briers,
        train_val_gaps=[None, None, None],
        spiegelhalter_zs=[None, None, None],
        tie_band=0.005,
        anti_auc_flag="true",
        eval_r_precision_at_1s=[0.8, 0.5, 0.6],
    )
    assert path_anti_auc_rp1 == "anti_auc_eval_rp1"
    assert out_anti_auc_rp1 == 0, (
        f"anti-AUC branch with R-p@1 supplied should pick the highest "
        f"R-p@1 winner (iter 0); got iter {out_anti_auc_rp1}"
    )


# ---------------------------------------------------------------------------
# Empty / single-entry edge cases
# ---------------------------------------------------------------------------


def test_best_checkpoint_empty_raises():
    with pytest.raises(ValueError):
        best_checkpoint([])


def test_best_checkpoint_single_entry_returns_zero():
    best_idx_a, path_a = best_checkpoint([0.25])
    assert best_idx_a == 0
    assert path_a == "strict_val_brier"
    best_idx_b, path_b = best_checkpoint(
        [0.25],
        train_val_gaps=[0.01],
        spiegelhalter_zs=[1.0],
        tie_band=0.005,
    )
    assert best_idx_b == 0
    # Singleton tie set → short-circuit to strict val-Brier argmin.
    assert path_b == "strict_val_brier"


# ---------------------------------------------------------------------------
# Recovery of the _187 motivating finding
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# V1.4 P1 — non-anti-AUC val_brier-flat fallback to eval R-p@1-best
# ---------------------------------------------------------------------------


def test_non_anti_auc_val_brier_flat_falls_back_to_eval_r_p_1_best():
    """V1.4 P1 — when val_brier is flat across the tie set AND we are on the
    non-anti-AUC branch AND every tied iter has an eval R-p@1 value, fall back
    to eval R-p@1-best (matches the V1.3 Option A anti-AUC rule).

    Fixture mirrors the #239 / #241 failure mode (memos ``_239``, ``_241``):
    val_briers ``[0.100, 0.101, 0.102]`` — range 0.002 < default tie_band
    0.005 → all 3 iters are tied AND the new gate fires. L1 (gap, |z|) would
    pick iter 0 (gap 0.001, |z| 1.0 → smallest L1 sort key), but iter 2 has
    the highest eval R-p@1 (0.700) and should win post-patch.
    """
    val_briers = [0.100, 0.101, 0.102]
    # L1 metrics arranged so the pre-V1.4 L1 path would prefer iter 0:
    #   iter 0 sort key = (0.001, 1.0, 0)
    #   iter 1 sort key = (0.020, 5.0, 1)
    #   iter 2 sort key = (0.050, 10.0, 2)
    gaps = [0.001, 0.020, 0.050]
    zs = [1.0, 5.0, 10.0]
    # eval R-p@1 anti-correlated with L1 — iter 2 is the lex oracle winner.
    eval_rp1 = [0.500, 0.600, 0.700]
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
        anti_auc_flag="false",
        eval_r_precision_at_1s=eval_rp1,
    )
    assert path == "v14_val_flat_eval_rp1"
    assert out == 2, (
        f"expected iter 2 (highest eval R-p@1={eval_rp1[2]}) on the V1.4 P1 "
        f"non-anti-AUC val_brier-flat fallback; got iter {out} "
        f"(R-p@1={eval_rp1[out]}). The pre-V1.4 L1 path would have picked "
        f"iter 0 via (gap, |z|) — the strictly-lowest-R-p@1 iter."
    )


def test_non_anti_auc_val_brier_sharp_L1_stays():
    """V1.4 P1 — when val_brier has real spread (not flat) on the non-anti-AUC
    branch, the new fallback gate (val_brier_range < tie_band) does NOT fire
    and the tie-break path is not entered at all — the strict val-Brier
    argmin wins.

    Fixture: val_briers ``[0.100, 0.120, 0.150]`` — only iter 0 falls inside
    the band ``[0.100, 0.105]`` (range = 0.05 ≫ tie_band 0.005), so the tie
    set is a singleton and the function short-circuits to ``strict_best=0``
    without entering either the V1.4 P1 fallback OR the L1 (gap, |z|) path.
    This preserves the documented behaviour on cells where val_brier IS
    informative (rare-event cells per ``_185`` / ``_187``).
    """
    val_briers = [0.100, 0.120, 0.150]
    # Arrange L1 + eval R-p@1 so that IF the tie set were expanded, both
    # alternatives would prefer iter 2 — proving that the function never
    # entered tie-break mode.
    gaps = [0.050, 0.020, 0.001]
    zs = [10.0, 5.0, 0.5]
    eval_rp1 = [0.300, 0.500, 0.900]
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
        anti_auc_flag="false",
        eval_r_precision_at_1s=eval_rp1,
    )
    # Singleton tie set short-circuits to ``strict_val_brier`` — the V1.4 P1
    # gate is never entered.
    assert path == "strict_val_brier"
    assert out == 0, (
        f"expected iter 0 (strict val-Brier argmin) when val_brier range "
        f"({max(val_briers) - min(val_briers):.3f}) exceeds tie_band "
        f"(0.005) so tie-break is not entered; got iter {out}. The V1.4 P1 "
        f"fallback must NOT fire on cells where val_brier has real spread."
    )


def test_anti_auc_path_unchanged_by_v14_p1_patch():
    """V1.4 P1 — regression test: the V1.3 Option A anti-AUC code path is
    unmodified. When ``anti_auc_flag='true'`` and val_brier is flat, the
    function returns the eval R-p@1-best iter via the existing anti-AUC
    block (NOT via the new V1.4 P1 block).

    Same fixture shape as the V1.4 P1 test: val_briers ``[0.100, 0.101, 0.102]``,
    range 0.002 < tie_band 0.005, all 3 tied. With ``anti_auc_flag='true'``
    the function enters the V1.3 Option A branch first and picks iter 2
    (highest eval R-p@1) — same outcome as the V1.4 P1 patch would produce,
    proving the patch is additive (a NEW branch for the non-anti-AUC case)
    and not a rewrite of the anti-AUC branch.
    """
    val_briers = [0.100, 0.101, 0.102]
    gaps = [0.001, 0.020, 0.050]  # L1 would prefer iter 0 if it ran
    zs = [1.0, 5.0, 10.0]
    eval_rp1 = [0.500, 0.600, 0.700]
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
        anti_auc_flag="true",  # V1.3 Option A engaged — exercises the
                               # existing anti-AUC branch, NOT the V1.4 P1
                               # block.
        eval_r_precision_at_1s=eval_rp1,
    )
    assert path == "anti_auc_eval_rp1"
    assert out == 2, (
        f"V1.3 Option A regression: expected iter 2 (highest eval R-p@1) on "
        f"the anti-AUC branch; got iter {out}. The V1.4 P1 patch must leave "
        f"this path unmodified."
    )


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
    out, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        plateau_threshold=0.005,  # default → tie_band=0.0025
    )
    assert path == "classic_l1"
    assert out == 4, (
        f"expected iter 4 (d4/λ6, gap 0.00163, |z|=6.26) to win at default "
        f"tie_band; got iter {out} (gap={gaps[out]}, |z|={abs(zs[out])})"
    )
    # And critically: the winner is NOT one of the high-gap overfit configs.
    assert gaps[out] <= 0.005, "winner should have low train-val gap"


# ---------------------------------------------------------------------------
# V1.4 P2 — tie-break path labels
# ---------------------------------------------------------------------------


def test_tiebreak_path_label_anti_auc():
    """V1.4 P2 — when the V1.3 Option A anti-AUC fallback fires (anti_auc_flag
    'true' + non-singleton tie set + eval R-p@1 present for all tied), the
    returned label is ``anti_auc_eval_rp1``.

    Same fixture shape as ``test_anti_auc_path_unchanged_by_v14_p1_patch`` —
    here we focus on the LABEL contract, not the chosen iter (which is
    already covered there). The label is the load-bearing piece for
    ``report.md`` to render "Anti-AUC fallback: tie set picked by eval
    R-Precision@1 (V1.3 Option A)".
    """
    val_briers = [0.100, 0.101, 0.102]
    gaps = [0.001, 0.020, 0.050]
    zs = [1.0, 5.0, 10.0]
    eval_rp1 = [0.500, 0.600, 0.700]
    _, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
        anti_auc_flag="true",
        eval_r_precision_at_1s=eval_rp1,
    )
    assert path == "anti_auc_eval_rp1", (
        f"expected anti_auc_eval_rp1 label when V1.3 Option A fallback fires; "
        f"got {path!r}"
    )


def test_tiebreak_path_label_v14_val_flat():
    """V1.4 P2 — when the V1.4 P1 non-anti-AUC val-Brier-flat fallback fires
    (anti_auc_flag 'false' + val_brier_range < tie_band + eval R-p@1 present
    for all tied), the returned label is ``v14_val_flat_eval_rp1``.

    The label distinguishes this branch from ``anti_auc_eval_rp1`` (same
    output mechanism — eval R-p@1-best iter — but a different gate condition)
    so ``report.md`` can render "Val_brier flat: tie set picked by eval
    R-Precision@1 (V1.4 P1)".
    """
    val_briers = [0.100, 0.101, 0.102]
    gaps = [0.001, 0.020, 0.050]
    zs = [1.0, 5.0, 10.0]
    eval_rp1 = [0.500, 0.600, 0.700]
    _, path = best_checkpoint(
        val_briers,
        train_val_gaps=gaps,
        spiegelhalter_zs=zs,
        tie_band=0.005,
        anti_auc_flag="false",
        eval_r_precision_at_1s=eval_rp1,
    )
    assert path == "v14_val_flat_eval_rp1", (
        f"expected v14_val_flat_eval_rp1 label when V1.4 P1 fallback fires; "
        f"got {path!r}"
    )


def test_tiebreak_path_label_set_is_exhaustive():
    """V1.4 P2 — pin the label set so accidental new branches are forced to
    register a new code in :data:`fs_hp_loop.TiebreakPath` instead of being
    silently swallowed by the report renderer. If you add a new branch to
    ``best_checkpoint`` AND it returns a label outside this set, this test
    fails and you must update :func:`report._tiebreak_path_description` too.
    """
    from typing import get_args

    from gbdt.fs_hp_loop import TiebreakPath

    expected = {
        "strict_val_brier",
        "anti_auc_eval_rp1",
        "v14_val_flat_eval_rp1",
        "classic_l1",
        "l1_fallthrough",
    }
    actual = set(get_args(TiebreakPath))
    assert actual == expected, (
        f"TiebreakPath label set drifted; expected {expected}, got {actual}. "
        f"Update both the Literal in fs_hp_loop.py AND the description map "
        f"in report._tiebreak_path_description if you add/remove a label."
    )
