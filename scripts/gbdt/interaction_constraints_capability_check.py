"""V1.2 Phase 8 — XGBoost ``interaction_constraints`` capability check (D6 cell).

This script is the *evidence* behind the Phase 8 memo's headline question:

    Can we actually RUN feature ``interaction_constraints`` in an XGBoost model
    through our gbdt pipeline, and does the constraint TAKE EFFECT?

It runs the three-part check on the fitted D6 XGBoost artifact (nifty50
up/+10%/25d/dd5%, depth-8 / 279-feature, the invB end-state config):

  1. PLUMBING — confirm ``interaction_constraints`` flows through the
     ``XGBoostModel`` construction path (``_validate_hp_xgb`` passes the
     structured value through untouched) and reaches ``xgb.XGBClassifier``.
     We exercise it via the package's :func:`gbdt.interactions.ablate_interactions`
     helper (the sanctioned out-of-band intervention surface).

  2. EFFECT — fit unconstrained, identify the top TreeSHAP ``pred_interactions``
     pair(s), refit with those pairs forbidden, and VERIFY the constraint is
     honored: the forbidden pair's SHAP-interaction magnitude collapses to ~0
     AND the pair never co-occurs on a tree path (native co-occurrence drops to
     exactly 0).

  3. METRICS — Brier + weighted R-precision (per-day variable-K = R(d)) on the
     held-out eval + test segments, unconstrained vs constrained.

Determinism is also checked: refitting the unconstrained config twice with the
pinned knobs must reproduce bit-identical predictions (the § 5.1 contract that
makes the artifact's checkpoint trustworthy).

Data is rebuilt EXACTLY as ``gbdt.__main__`` does (same panel, features, target,
uniqueness weights, split carving) so the refit sees the same train+val the
artifact's model saw.

Usage:
    uv run python -m scripts.gbdt.interaction_constraints_capability_check \
        --artifact results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_xgb_phase8 \
        --out results/gbdt/data/_phase8_xgboost_interaction_constraints_capability.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss

import gbdt.data as gbdt_data
import gbdt.features as gbdt_features
from gbdt.calibration import apply_calibrator
from gbdt.diagnose_core import per_day_r_precision
from gbdt.interactions import (
    _interaction_constraints_from_forbidden,
    ablate_interactions,
    interaction_strength,
)
from gbdt.model import XGBoostModel, _validate_hp_xgb, hp_tables_for, make_model
from gbdt.targets import build_target
from gbdt.train import SplitSpec, _gather_segment, carve_single_fold
from gbdt.uniqueness import compute_uniqueness_weights


def _build_dataset(spec: dict, repo_root: Path):
    """Rebuild (panel, X, y, sample_weights, split) exactly as the runner does."""
    target = spec["target"]
    split_d = spec.get("split", {}) or {}
    split = SplitSpec(
        train_rows=split_d.get("train_rows", 800),
        val_rows=split_d.get("val_rows", 400),
        eval_rows=split_d.get("eval_rows", 200),
        test_rows=split_d.get("test_rows", 100),
    )
    min_rows = split_d.get("min_rows_per_ticker", split.total)
    dr = spec.get("date_range", {}) or {}
    data_cfg = spec.get("data", {}) or {}
    staleness_days = int(
        data_cfg.get("staleness_days", gbdt_data.DEFAULT_STALENESS_DAYS)
    )
    panel_obj = gbdt_data.load_panel(
        target["universe"],
        start=dr.get("start"),
        end=dr.get("end"),
        min_rows=min_rows,
        repo_root=repo_root,
        staleness_days=staleness_days,
    )
    fcfg = spec.get("features", {}) or {}
    lookbacks = tuple(fcfg.get("lookback_windows", gbdt_features.DEFAULT_LOOKBACKS))
    families = fcfg.get("candidates", "all")
    exclude = fcfg.get("exclude") or []
    X = gbdt_features.build_feature_matrix(
        panel_obj.panel,
        panel_obj.index_series,
        lookbacks=lookbacks,
        annualization=panel_obj.annualization_factor,
        families=families,
        exclude=exclude,
    )
    X = X.dropna(axis=1, how="all")
    y = build_target(
        panel_obj.panel,
        direction=target["direction"],
        threshold_pct=target["threshold_pct"],
        horizon_days=target["horizon_days"],
        max_drawdown=target.get("max_drawdown"),
    )
    sample_weights = None
    if bool(target.get("uniqueness_weighting", True)):
        sample_weights = compute_uniqueness_weights(
            panel_obj.panel, horizon=int(target["horizon_days"])
        )
    return panel_obj.panel, X, y, sample_weights, split


def _segments(panel, X, y, weights, split, features):
    fold = carve_single_fold(panel, split)
    out = {}
    for name, idx in (
        ("train", fold.train_idx),
        ("val", fold.val_idx),
        ("eval", fold.eval_idx),
        ("test", fold.test_idx),
    ):
        Xs, ys, mi, ws = _gather_segment(panel, X, y, idx, weights)
        out[name] = (Xs[features], ys, mi, ws)
    return out


def _preds_df(model, calibrator, Xs, ys, mi, ws) -> pd.DataFrame:
    p_raw = model.predict_proba(Xs)
    p_cal = apply_calibrator(p_raw, calibrator)
    return pd.DataFrame(
        {
            "date": mi.get_level_values("date"),
            "ticker": mi.get_level_values("ticker"),
            "p_raw": p_raw,
            "p_calibrated": p_cal,
            "y_true": ys,
            "sample_weight": ws if ws is not None else 1.0,
        }
    )


def _weighted_brier(df: pd.DataFrame) -> float:
    w = df["sample_weight"].values.astype(float)
    return float(np.average((df["p_calibrated"] - df["y_true"]) ** 2, weights=w))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k-pairs", type=int, default=1,
                    help="number of top SHAP-interaction pairs to forbid")
    ap.add_argument("--shap-rows", type=int, default=5000)
    args = ap.parse_args()

    repo_root = Path.cwd()
    art = args.artifact
    spec = yaml.safe_load((art / "spec.yaml").read_text())
    seed = int(spec.get("random_seed", 42))
    results: dict = {"artifact": str(art), "cell": spec["target"]}

    # --------------------------------------------------------------------- #
    # PART 1: PLUMBING — is interaction_constraints a passable HP?
    # --------------------------------------------------------------------- #
    tunable, enum_values, pinned = hp_tables_for("xgboost")
    plumbing = {
        "in_tunable_ranges_xgb": "interaction_constraints" in tunable,
        "in_enum_values_xgb": "interaction_constraints" in enum_values,
        "in_pinned_hps_xgb": "interaction_constraints" in pinned,
    }
    # The construction-path validator must NOT reject a spec-level
    # interaction_constraints (unknown structured key passes through untouched).
    probe_hp = dict(yaml.safe_load((art / "hp.yaml").read_text())["hp"])
    probe_hp["interaction_constraints"] = "[[0,1],[2,3]]"
    try:
        validated = _validate_hp_xgb(probe_hp)
        plumbing["construction_validator_passes_through"] = (
            validated.get("interaction_constraints") == "[[0,1],[2,3]]"
        )
        plumbing["construction_validator_error"] = None
    except Exception as exc:  # pragma: no cover - capability assertion
        plumbing["construction_validator_passes_through"] = False
        plumbing["construction_validator_error"] = repr(exc)
    # Confirm the resume/agent decision path WOULD reject it (the documented
    # block — symmetric to monotone_constraints on CatBoost, per _174 invA).
    try:
        from gbdt.loop_protocol import DecisionError, validate_decision

        validate_decision(
            {"hp_changes": {"interaction_constraints": "[[0,1]]"}},
            spec,
            known_features=["f0", "f1"],
            backend="xgboost",
        )
        plumbing["decision_path_rejects"] = False
        plumbing["decision_path_error"] = None
    except DecisionError as exc:
        plumbing["decision_path_rejects"] = True
        plumbing["decision_path_error"] = str(exc)
    results["part1_plumbing"] = plumbing

    # --------------------------------------------------------------------- #
    # Rebuild the dataset + load the fitted reference model.
    # --------------------------------------------------------------------- #
    t0 = time.time()
    panel, X, y, weights, split = _build_dataset(spec, repo_root)
    features = list(yaml.safe_load((art / "features.yaml").read_text())["features"])
    segs = _segments(panel, X, y, weights, split, features)
    results["data_build_secs"] = round(time.time() - t0, 1)
    results["n_features"] = len(features)
    results["segment_rows"] = {k: int(len(v[1])) for k, v in segs.items()}

    X_tr, y_tr, _, w_tr = segs["train"]
    X_val, y_val, _, w_val = segs["val"]

    hp = dict(yaml.safe_load((art / "hp.yaml").read_text())["hp"])
    es = hp.pop("early_stopping_rounds", None)

    # Reference (unconstrained) refit — same (features, hp, seed, row order) as
    # the artifact. This is also the determinism check.
    ref = make_model("xgboost", dict(hp), feature_names=features, random_seed=seed)
    ref.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=es,
            train_weight=w_tr, val_weight=w_val)
    ref2 = make_model("xgboost", dict(hp), feature_names=features, random_seed=seed)
    ref2.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=es,
             train_weight=w_tr, val_weight=w_val)
    p1 = ref.predict_proba(segs["eval"][0])
    p2 = ref2.predict_proba(segs["eval"][0])
    results["determinism_bit_identical_refit"] = bool(np.array_equal(p1, p2))

    # Sanity: refit matches the on-disk artifact model's eval predictions.
    disk_model = XGBoostModel.load(art / "model.ubj", feature_names=features)
    p_disk = disk_model.predict_proba(segs["eval"][0])
    results["refit_matches_artifact"] = bool(np.allclose(p1, p_disk, atol=1e-6))

    # --------------------------------------------------------------------- #
    # PART 2: EFFECT — top SHAP pair, ablate, verify the constraint is honored
    # --------------------------------------------------------------------- #
    # SHAP interaction ranking on the in-sample train+val matrix (sub-sampled).
    X_insample = pd.concat([X_tr, X_val], axis=0)
    base_ix = interaction_strength(
        ref, X_insample, kind="shap", top_n=15,
        max_rows=args.shap_rows, random_seed=seed,
    )
    top_pairs = [(a, b) for a, b, _s, _sc in base_ix.top_pairs[: args.top_k_pairs]]
    results["top_shap_pairs"] = [
        {"a": a, "b": b, "strength": float(s), "sign_consistency": float(sc)}
        for a, b, s, sc in base_ix.top_pairs[:15]
    ]
    results["forbidden_pairs"] = [{"a": a, "b": b} for a, b in top_pairs]

    base_pair_shap = {f"{a}|{b}": float(base_ix.pair_strength(a, b))
                      for a, b in top_pairs}

    # Co-occurrence on the reference model (the "do they share a tree path?"
    # cross-check — must be > 0 for a genuinely co-splitting pair).
    base_cooc_full = interaction_strength(ref, X_insample, kind="cooccurrence",
                                          top_n=100000)
    base_pair_cooc = {f"{a}|{b}": float(base_cooc_full.pair_strength(a, b))
                      for a, b in top_pairs}

    # The intervention: forbid the top-K pairs, refit.
    constrained = ablate_interactions(
        ref, X_tr, y_tr, top_pairs,
        X_val=X_val, y_val=y_val,
        train_weight=w_tr, val_weight=w_val,
        early_stopping_rounds=es, feature_names=features, random_seed=seed,
    )
    results["constraint_injected_into_hp"] = (
        "interaction_constraints" in constrained.hp
    )
    # Verify the emitted constraint string forbids exactly the requested pairs.
    feat_index = {n: i for i, n in enumerate(features)}
    forbidden_set = {frozenset((a, b)) for a, b in top_pairs}
    expected = _interaction_constraints_from_forbidden(features, forbidden_set)
    results["constraint_string_matches_expected"] = (
        constrained.hp["interaction_constraints"] == expected
    )

    # Did it take effect? (a) SHAP interaction on the forbidden pairs collapses.
    con_ix = interaction_strength(
        constrained, X_insample, kind="shap", top_n=15,
        max_rows=args.shap_rows, random_seed=seed,
    )
    con_pair_shap = {f"{a}|{b}": float(con_ix.pair_strength(a, b))
                     for a, b in top_pairs}
    # (b) co-occurrence on the forbidden pairs drops to exactly 0 (never on a path).
    con_cooc_full = interaction_strength(constrained, X_insample,
                                         kind="cooccurrence", top_n=100000)
    con_pair_cooc = {f"{a}|{b}": float(con_cooc_full.pair_strength(a, b))
                     for a, b in top_pairs}

    pair_effects = []
    for a, b in top_pairs:
        key = f"{a}|{b}"
        base_s = base_pair_shap[key]
        con_s = con_pair_shap[key]
        pair_effects.append({
            "pair": key,
            "shap_interaction_base": base_s,
            "shap_interaction_constrained": con_s,
            "shap_collapse_ratio": (con_s / base_s) if base_s > 0 else None,
            "cooccurrence_base": base_pair_cooc[key],
            "cooccurrence_constrained": con_pair_cooc[key],
            "honored_shap_collapsed": bool(base_s > 0 and con_s < 0.05 * base_s),
            "honored_cooccurrence_zero": bool(con_pair_cooc[key] == 0.0),
        })
    results["part2_effect"] = {
        "pairs": pair_effects,
        "all_honored": bool(all(
            p["honored_shap_collapsed"] and p["honored_cooccurrence_zero"]
            for p in pair_effects
        )),
    }

    # --------------------------------------------------------------------- #
    # PART 3: METRICS — Brier + weighted R-precision, base vs constrained.
    # --------------------------------------------------------------------- #
    # Fit a fresh isotonic calibrator on val for each model so the comparison is
    # apples-to-apples (both calibrated the same way). Use isotonic-always to
    # match the artifact's conditional-isotonic outcome (Z-test rejected native
    # on this cell per _149) and keep the two models on identical post-processing.
    from gbdt.calibration import fit_isotonic

    def _calibrator(model):
        p_val_raw = model.predict_proba(X_val)
        return fit_isotonic(y_val, p_val_raw)

    cal_base = _calibrator(ref)
    cal_con = _calibrator(constrained)

    metrics = {}
    for seg in ("eval", "test"):
        Xs, ys, mi, ws = segs[seg]
        if len(ys) == 0:
            metrics[seg] = {"note": "empty segment"}
            continue
        df_base = _preds_df(ref, cal_base, Xs, ys, mi, ws)
        df_con = _preds_df(constrained, cal_con, Xs, ys, mi, ws)
        rp_base = per_day_r_precision(df_base)
        rp_con = per_day_r_precision(df_con)
        metrics[seg] = {
            "base_rate_weighted": rp_base["base_rate_weighted"],
            "brier_weighted_base": _weighted_brier(df_base),
            "brier_weighted_constrained": _weighted_brier(df_con),
            "brier_unweighted_base": float(
                brier_score_loss(df_base["y_true"], df_base["p_calibrated"])),
            "brier_unweighted_constrained": float(
                brier_score_loss(df_con["y_true"], df_con["p_calibrated"])),
            "r_precision_weighted_base": rp_base["r_precision_weighted"],
            "r_precision_weighted_constrained": rp_con["r_precision_weighted"],
            "n_days_with_positives": rp_base["n_days_with_positives"],
        }
    results["part3_metrics"] = metrics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
