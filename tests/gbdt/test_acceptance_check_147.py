"""V1.1 Phase 6 — unit tests for the `_147` acceptance comparison logic.

The comparison harness (`scripts/gbdt/acceptance_check_147.py`) decides whether a
completed automated agent-loop run reproduces the hand-driven `_147` findings.
The *full* acceptance run is multi-hour + human-in-the-loop, so we do NOT run it
here. Instead these tests feed :func:`evaluate_acceptance` synthetic artifacts —
the parsed `iterations.jsonl` / `metrics.json` / `features.yaml` objects — so the
comparison *logic itself* (the thing that produces the PASS/FAIL verdict) is
tested deterministically and fast.

Two synthetic runs anchor the tests:

* ``_147_like_run()`` — an iteration history + metrics that MATCH the documented
  `_147` end-state (HP ceiling ~0.164, depth {6,8} plateau within noise with
  depth-4 the underfit arm, every monotone config worse than the best
  unconstrained one, no overfit, declining prevalence, no feature collapse).
  This must pass every check.
* A handful of mutated copies that each break exactly one finding (a config that
  beats baseline, depth-4 no longer the underfit arm, a "beneficial" monotone
  config, an overfit gap, flat prevalence, a collapsed feature set) — each must
  flip exactly that one check to FAIL while the rest still pass.

Plus the `load_run` round-trip on real-shaped on-disk artifacts (so the parsing
half is covered too) and the SKIP behaviour when optional data is absent.
"""

from __future__ import annotations

import json

import yaml

from scripts.gbdt.acceptance_check_147 import (
    evaluate_acceptance,
    format_table,
    load_run,
)


# ---------------------------------------------------------------------------
# A synthetic run that reproduces the `_147` end-state (mirrors the memo tables)
# ---------------------------------------------------------------------------


def _147_like_iterations() -> list[dict]:
    """Iteration history matching `_147`'s Unified-conclusion table (deconfounded).

    iter 0 baseline (all-279, depth 6, lr 0.05) 0.1642; the depth/lr sweep stays
    in [0.1633, 0.1661] with depth 6/8 a plateau within noise (deconfounded
    depth-8 = 0.1633, l2 held) and depth-4 the underfit arm; iters 5-9 apply
    monotone constraints, all worse than the best unconstrained config
    (0.1654-0.1664). No overfit (small gap throughout).
    """
    return [
        # HP sweep (non-monotone) — the depth/lr ceiling map.
        {"iter": 0, "n_features": 279, "val_brier": 0.1642, "train_val_gap": -0.0048,
         "hp": {"depth": 6, "learning_rate": 0.05}, "rationale": "baseline all-279"},
        {"iter": 1, "n_features": 279, "val_brier": 0.1633, "train_val_gap": 0.0044,
         "hp": {"depth": 8, "learning_rate": 0.05},
         "rationale": "depth 8 (l2 held) — plateau with depth-6, within noise"},
        {"iter": 2, "n_features": 279, "val_brier": 0.1661, "train_val_gap": -0.0090,
         "hp": {"depth": 4, "learning_rate": 0.05}, "rationale": "depth 4 — underfits"},
        {"iter": 3, "n_features": 279, "val_brier": 0.1641, "train_val_gap": -0.0050,
         "hp": {"depth": 6, "learning_rate": 0.02}, "rationale": "lr 0.02 — tie (noise)"},
        {"iter": 4, "n_features": 88, "val_brier": 0.1650, "train_val_gap": -0.0045,
         "hp": {"depth": 6, "learning_rate": 0.05}, "rationale": "targeted FS 88 feat"},
        # Monotone-constraint ablation — all worse than baseline.
        {"iter": 5, "n_features": 279, "val_brier": 0.1664, "train_val_gap": -0.0040,
         "hp": {"depth": 6, "learning_rate": 0.05,
                "monotone_constraints": {"garman_klass_200": 1}},
         "rationale": "monotone +1 (17 vol est) — worst"},
        {"iter": 6, "n_features": 279, "val_brier": 0.1657, "train_val_gap": -0.0040,
         "hp": {"depth": 6, "learning_rate": 0.05,
                "monotone_constraints": {"garman_klass_50": 1}},
         "rationale": "monotone +1 (safe 13)"},
        {"iter": 7, "n_features": 279, "val_brier": 0.1656, "train_val_gap": -0.0040,
         "hp": {"depth": 6, "learning_rate": 0.05,
                "monotone_constraints": {"runup_50": 1}},
         "rationale": "monotone +1 (18: +5 pruned)"},
        {"iter": 8, "n_features": 279, "val_brier": 0.1654, "train_val_gap": -0.0040,
         "hp": {"depth": 6, "learning_rate": 0.05,
                "monotone_constraints": {"parkinson_10": 1}},
         "rationale": "monotone +1 (9 low-interaction) — best monotone"},
        {"iter": 9, "n_features": 279, "val_brier": 0.1664, "train_val_gap": -0.0040,
         "hp": {"depth": 6, "learning_rate": 0.05,
                "monotone_constraints": {"garman_klass_200": 1}},
         "rationale": "monotone +1 (8 high-interaction)"},
    ]


def _147_like_metrics() -> dict:
    return {
        "experiment_name": "nifty50_up_10pct_25d_dd5pct_acceptance",
        "data": {
            "n_tickers_used": 46,
            "positive_prevalence_train": 0.280,
            "positive_prevalence_eval": 0.138,
        },
        "loop": {
            "n_iterations_run": 10,
            "best_iteration": 0,
            "inner_stop_signal": "agent_should_stop",
        },
        # R-precision carried in the headline block (lift ~2.1x); the harness
        # also accepts a weighted/base_rate pair.
        "headline_test": {
            "brier": 0.1383,
            "roc_auc": 0.733,
            "r_precision": {"weighted": 0.416, "base_rate_weighted": 0.196},
        },
    }


def _147_like_features() -> list[str]:
    # All-279 optimal on val; the 88-feat deployment artifact is the smaller end.
    # Either way the final set is substantial — synth with 90 to model "kept".
    return [f"feat_{i}" for i in range(90)]


# ---------------------------------------------------------------------------
# The happy path: a `_147`-faithful run passes every (evaluable) check.
# ---------------------------------------------------------------------------


def test_147_like_run_passes_all_checks():
    res = evaluate_acceptance(
        _147_like_iterations(), _147_like_metrics(), _147_like_features()
    )
    assert res.overall_pass, format_table(res)
    assert res.n_fail == 0
    # Every check should be evaluable on this complete synthetic run (no SKIPs).
    skipped = [c.name for c in res.checks if c.passed is None]
    assert skipped == [], f"unexpected SKIPs: {skipped}\n{format_table(res)}"
    # All nine documented findings present.
    names = {c.name for c in res.checks}
    assert names == {
        "hp_ceiling_band", "hp_ceiling_spread", "depth_optimal",
        "no_meaningful_improvement", "monotone_contraindicated",
        "no_overfit_baseline", "prevalence_drift_ceiling", "ranking_robust",
        "final_features_not_collapsed",
    }


# ---------------------------------------------------------------------------
# Each mutation breaks exactly ONE finding.
# ---------------------------------------------------------------------------


def _verdict(res, name):
    return next(c.passed for c in res.checks if c.name == name)


def test_config_beating_baseline_fails_ceiling_checks():
    """A config that meaningfully beats baseline contradicts the HP ceiling."""
    iters = _147_like_iterations()
    iters[3]["val_brier"] = 0.1620  # a real 0.0022 win over baseline 0.1642
    res = evaluate_acceptance(iters, _147_like_metrics(), _147_like_features())
    assert _verdict(res, "no_meaningful_improvement") is False
    # 0.1620 is still inside the wide ceiling band, but the HP-only SPREAD widens
    # past the tiny-band threshold => the "declare the ceiling" check also fails.
    assert _verdict(res, "hp_ceiling_spread") is False


def test_depth4_not_underfit_fails_depth_check():
    """Depth-4 NOT being the worst arm breaks the plateau finding (depth is
    supposed to be a {6,8} plateau with depth-4 the underfit arm). Set depth-4
    mid-band (better than depth-6) so it's no longer the worst — within the
    ceiling band, so ONLY the depth check flips."""
    iters = _147_like_iterations()
    iters[2]["val_brier"] = 0.1638  # depth-4 no longer the worst arm
    res = evaluate_acceptance(iters, _147_like_metrics(), _147_like_features())
    assert _verdict(res, "depth_optimal") is False
    # The mutation stays inside the ceiling band, so it does not trip the other
    # ceiling checks — the depth finding fails in isolation.
    assert _verdict(res, "hp_ceiling_spread") is True
    assert _verdict(res, "no_meaningful_improvement") is True


def test_beneficial_monotone_fails_contraindication():
    """A monotone config that beats baseline breaks 'monotone contraindicated'."""
    iters = _147_like_iterations()
    iters[8]["val_brier"] = 0.1630  # best monotone now BELOW baseline 0.1642
    res = evaluate_acceptance(iters, _147_like_metrics(), _147_like_features())
    assert _verdict(res, "monotone_contraindicated") is False


def test_overfit_gap_fails_no_overfit_check():
    """A large positive iter-0 gap (val worse than train) = overfit signal."""
    iters = _147_like_iterations()
    iters[0]["train_val_gap"] = 0.05  # well past the 0.02 no-overfit threshold
    res = evaluate_acceptance(iters, _147_like_metrics(), _147_like_features())
    assert _verdict(res, "no_overfit_baseline") is False


def test_flat_prevalence_fails_drift_ceiling():
    """No train->eval prevalence decline breaks the calibration-ceiling finding."""
    m = _147_like_metrics()
    m["data"]["positive_prevalence_eval"] = 0.275  # ~ no decline from 0.280
    res = evaluate_acceptance(_147_like_iterations(), m, _147_like_features())
    assert _verdict(res, "prevalence_drift_ceiling") is False


def test_collapsed_feature_set_fails():
    """An aggressively-pruned final set contradicts the FS-is-neutral finding."""
    res = evaluate_acceptance(
        _147_like_iterations(), _147_like_metrics(),
        [f"feat_{i}" for i in range(12)],  # collapsed to 12 features
    )
    assert _verdict(res, "final_features_not_collapsed") is False


def test_weak_ranking_fails_robustness():
    """A weak R-precision lift breaks the 'ranking robust ~2.1x' finding."""
    m = _147_like_metrics()
    m["headline_test"]["r_precision"] = {"weighted": 0.20, "base_rate_weighted": 0.196}
    res = evaluate_acceptance(_147_like_iterations(), m, _147_like_features())
    assert _verdict(res, "ranking_robust") is False


# ---------------------------------------------------------------------------
# SKIP behaviour: optional data absent => SKIP (not FAIL), with a pointer note.
# ---------------------------------------------------------------------------


def test_no_monotone_explored_skips_not_fails():
    """If the loop never tried a monotone constraint, that finding can't be
    verified — it SKIPs (loud note) rather than FAILs."""
    iters = [r for r in _147_like_iterations() if "monotone_constraints" not in r["hp"]]
    res = evaluate_acceptance(iters, _147_like_metrics(), _147_like_features())
    assert _verdict(res, "monotone_contraindicated") is None
    # A run with no monotone iteration shouldn't FAIL the overall verdict purely
    # on the missing monotone check — the other findings still hold.
    assert res.n_fail == 0


def test_missing_rprecision_skips_with_pointer():
    """No R-precision in metrics.json => SKIP with a pointer to compute_r_precision."""
    m = _147_like_metrics()
    del m["headline_test"]["r_precision"]
    res = evaluate_acceptance(_147_like_iterations(), m, _147_like_features())
    rank = next(c for c in res.checks if c.name == "ranking_robust")
    assert rank.passed is None
    assert "compute_r_precision" in rank.note


# ---------------------------------------------------------------------------
# load_run round-trip on real-shaped on-disk artifacts.
# ---------------------------------------------------------------------------


def test_load_run_roundtrip(tmp_path):
    run_dir = tmp_path / "nifty50_up_10pct_25d_dd5pct_acceptance"
    run_dir.mkdir()
    (run_dir / "iterations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _147_like_iterations()) + "\n"
    )
    (run_dir / "metrics.json").write_text(json.dumps(_147_like_metrics()))
    (run_dir / "features.yaml").write_text(yaml.safe_dump(_147_like_features()))

    iterations, metrics, features = load_run(run_dir)
    assert len(iterations) == 10
    assert metrics["data"]["positive_prevalence_train"] == 0.280
    assert len(features) == 90

    # The loaded artifacts pass the full acceptance (end-to-end through disk).
    res = evaluate_acceptance(iterations, metrics, features)
    assert res.overall_pass, format_table(res)


def test_empty_run_dir_all_skips(tmp_path):
    """A run dir with no artifacts yields all-SKIP, no FAIL (nothing to compare)."""
    iterations, metrics, features = load_run(tmp_path)
    res = evaluate_acceptance(iterations, metrics, features)
    assert res.n_pass == 0
    assert res.n_fail == 0
    assert res.n_skip == len(res.checks)
    assert res.overall_pass  # no FAIL => overall "pass" but the table is all SKIP
