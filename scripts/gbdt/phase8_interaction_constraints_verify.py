"""Phase 8 D6 verification — interaction_constraints capability + effect + metrics.

Backs the `_175` memo. Answers, in order:
  1. PLUMBING — is `interaction_constraints` passable through the XGBoost backend?
     (construction/spec path passes it through untouched; the agent-loop decision
     path rejects it — symmetric to monotone_constraints on CatBoost.)
  2. EFFECT — fit unconstrained, identify the top TreeSHAP pred_interactions pair,
     refit with it forbidden, VERIFY the constraint is honored (the pair's SHAP
     interaction collapses to 0 AND it never co-splits / co-occurrence → 0).
  3. METRICS — Brier + weighted R-precision (per-day variable-K), base vs constrained.
Also checks determinism (bit-identical refit) + that the refit matches the artifact.

SHAP budget: native TreeSHAP pred_interactions at the full F=279 active set is
~0.8 s/row on this CPU (plan § 3.3 R2 — the documented cost driver: pred_interactions
over a 280-dim feature space is impractical for thousands of rows on a single-thread
`exact` booster). The dense reference (`shap_interaction_dense_reference`) is run on
a 200-row seeded sub-sample (plan § 3.3 mitigation 2 — the ranking is stable: its top
pairs match the near-free co-occurrence ranking). Co-occurrence (the plan's § 3.2
cheap cross-check / mitigation 5 fallback) supplies the "never co-split" check. This
uses the existing public interaction tooling; it is NOT a hand-rolled SHAP impl.

Run from the repo root:
    uv run python -m scripts.gbdt.phase8_interaction_constraints_verify
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.metrics import brier_score_loss

from gbdt.calibration import apply_calibrator, fit_isotonic
from gbdt.diagnose_core import per_day_r_precision
from gbdt.interactions import (
    _interaction_constraints_from_forbidden,
    ablate_interactions,
    interaction_strength,
    shap_interaction_dense_reference,
)
from gbdt.model import XGBoostModel, _validate_hp_xgb, hp_tables_for, make_model
from gbdt.train import SplitSpec, _gather_segment, carve_single_fold
import gbdt.data as gbdt_data
import gbdt.features as gbdt_features
from gbdt.targets import build_target
from gbdt.uniqueness import compute_uniqueness_weights

ART = Path("results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_xgb_phase8")
OUT = Path("results/gbdt/data/_175_xgboost_interaction_constraints_capability.json")
SHAP_ROWS = 200
SEED = 42


def dense_top_pairs(model, X, n=15):
    M, fn = shap_interaction_dense_reference(model, X)
    iu = np.triu_indices(len(fn), k=1)
    s = M[iu]
    order = np.argsort(s)[::-1]
    pairs = [(fn[iu[0][r]], fn[iu[1][r]], float(M[iu[0][r], iu[1][r]])) for r in order[:n]]
    return pairs, M, fn


def pair_strength_from_M(M, fn, a, b):
    i, j = fn.index(a), fn.index(b)
    return float(M[i, j])


def build_segments(spec):
    target = spec["target"]
    split = SplitSpec(800, 400, 200, 100)
    panel_obj = gbdt_data.load_panel(
        target["universe"], min_rows=split.total, repo_root=Path.cwd()
    )
    X = gbdt_features.build_feature_matrix(
        panel_obj.panel, panel_obj.index_series,
        lookbacks=gbdt_features.DEFAULT_LOOKBACKS,
        annualization=panel_obj.annualization_factor,
        families="all", exclude=[],
    ).dropna(axis=1, how="all")
    y = build_target(panel_obj.panel, direction=target["direction"],
                     threshold_pct=target["threshold_pct"],
                     horizon_days=target["horizon_days"],
                     max_drawdown=target.get("max_drawdown"))
    w = compute_uniqueness_weights(panel_obj.panel, horizon=int(target["horizon_days"]))
    fold = carve_single_fold(panel_obj.panel, split)
    feats = list(yaml.safe_load((ART / "features.yaml").read_text())["features"])
    segs = {}
    for name, idx in (("train", fold.train_idx), ("val", fold.val_idx),
                      ("eval", fold.eval_idx), ("test", fold.test_idx)):
        Xs, ys, mi, ws = _gather_segment(panel_obj.panel, X, y, idx, w)
        segs[name] = (Xs[feats], ys, mi, ws)
    return segs, feats


def preds_df(model, cal, Xs, ys, mi, ws):
    p_raw = model.predict_proba(Xs)
    return pd.DataFrame({
        "date": mi.get_level_values("date"),
        "ticker": mi.get_level_values("ticker"),
        "p_calibrated": apply_calibrator(p_raw, cal),
        "y_true": ys,
        "sample_weight": ws if ws is not None else 1.0,
    })


def wbrier(df):
    w = df["sample_weight"].values.astype(float)
    return float(np.average((df["p_calibrated"] - df["y_true"]) ** 2, weights=w))


def main():
    spec = yaml.safe_load((ART / "spec.yaml").read_text())
    res = {"id": "_175", "artifact": str(ART), "cell": spec["target"],
           "shap_rows": SHAP_ROWS, "seed": SEED}

    # ---- PART 1: PLUMBING -------------------------------------------------
    tunable, enum_v, pinned = hp_tables_for("xgboost")
    plumb = {
        "in_tunable_ranges_xgb": "interaction_constraints" in tunable,
        "in_enum_values_xgb": "interaction_constraints" in enum_v,
        "in_pinned_hps_xgb": "interaction_constraints" in pinned,
    }
    probe = dict(yaml.safe_load((ART / "hp.yaml").read_text())["hp"])
    probe["interaction_constraints"] = "[[0,1],[2,3]]"
    try:
        v = _validate_hp_xgb(probe)
        plumb["construction_validator_passes_through"] = (
            v.get("interaction_constraints") == "[[0,1],[2,3]]")
        plumb["construction_validator_error"] = None
    except Exception as exc:
        plumb["construction_validator_passes_through"] = False
        plumb["construction_validator_error"] = repr(exc)
    from gbdt.loop_protocol import DecisionError, validate_decision
    try:
        validate_decision({"hp_changes": {"interaction_constraints": "[[0,1]]"}},
                          spec, known_features=["f0", "f1"], backend="xgboost")
        plumb["decision_path_rejects"] = False
        plumb["decision_path_error"] = None
    except DecisionError as exc:
        plumb["decision_path_rejects"] = True
        plumb["decision_path_error"] = str(exc)
    res["part1_plumbing"] = plumb
    print("[1] plumbing:", json.dumps(plumb), flush=True)

    # ---- data + reference fit --------------------------------------------
    t0 = time.time()
    segs, feats = build_segments(spec)
    res["data_build_secs"] = round(time.time() - t0, 1)
    res["n_features"] = len(feats)
    res["segment_rows"] = {k: int(len(v[1])) for k, v in segs.items()}
    X_tr, y_tr, _, w_tr = segs["train"]
    X_val, y_val, _, w_val = segs["val"]
    hp = dict(yaml.safe_load((ART / "hp.yaml").read_text())["hp"])
    es = hp.pop("early_stopping_rounds", None)

    ref = make_model("xgboost", dict(hp), feature_names=feats, random_seed=SEED)
    ref.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=es,
            train_weight=w_tr, val_weight=w_val)
    ref2 = make_model("xgboost", dict(hp), feature_names=feats, random_seed=SEED)
    ref2.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=es,
             train_weight=w_tr, val_weight=w_val)
    p1 = ref.predict_proba(segs["eval"][0]); p2 = ref2.predict_proba(segs["eval"][0])
    res["determinism_bit_identical_refit"] = bool(np.array_equal(p1, p2))
    disk = XGBoostModel.load(ART / "model.ubj", feature_names=feats)
    res["refit_matches_artifact"] = bool(
        np.allclose(p1, disk.predict_proba(segs["eval"][0]), atol=1e-6))
    res["n_trees"] = int(ref.best_iteration + 1)
    print(f"[fit] determinism={res['determinism_bit_identical_refit']} "
          f"matches_artifact={res['refit_matches_artifact']} ntrees={res['n_trees']}",
          flush=True)

    # ---- PART 2: EFFECT — top SHAP pair, ablate, verify -------------------
    X_ins = pd.concat([X_tr, X_val], axis=0)
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(X_ins), size=SHAP_ROWS, replace=False))
    X_shap = X_ins.iloc[sel]

    ts = time.time()
    base_pairs, M_base, fn = dense_top_pairs(ref, X_shap, n=15)
    res["shap_secs_base"] = round(time.time() - ts, 1)
    top = base_pairs[0]
    forbidden = [(top[0], top[1])]
    res["top_shap_pairs"] = [{"a": a, "b": b, "strength": s} for a, b, s in base_pairs]
    res["forbidden_pairs"] = [{"a": top[0], "b": top[1]}]
    print(f"[2] top SHAP pair forbidden: {top[0]} x {top[1]} str={top[2]:.6f} "
          f"({res['shap_secs_base']}s)", flush=True)

    base_cooc = interaction_strength(ref, X_ins, kind="cooccurrence", top_n=100000)
    base_pair_cooc = base_cooc.pair_strength(top[0], top[1])

    constrained = ablate_interactions(
        ref, X_tr, y_tr, forbidden, X_val=X_val, y_val=y_val,
        train_weight=w_tr, val_weight=w_val, early_stopping_rounds=es,
        feature_names=feats, random_seed=SEED)
    res["constraint_injected_into_hp"] = "interaction_constraints" in constrained.hp
    expected = _interaction_constraints_from_forbidden(
        feats, {frozenset((top[0], top[1]))})
    res["constraint_string_matches_expected"] = (
        constrained.hp["interaction_constraints"] == expected)

    ts = time.time()
    _, M_con, _ = dense_top_pairs(constrained, X_shap, n=15)
    res["shap_secs_constrained"] = round(time.time() - ts, 1)
    con_pair_shap = pair_strength_from_M(M_con, fn, top[0], top[1])
    con_cooc = interaction_strength(constrained, X_ins, kind="cooccurrence", top_n=100000)
    con_pair_cooc = con_cooc.pair_strength(top[0], top[1])

    res["part2_effect"] = {
        "pair": f"{top[0]}|{top[1]}",
        "shap_interaction_base": float(top[2]),
        "shap_interaction_constrained": con_pair_shap,
        "shap_collapse_ratio": (con_pair_shap / top[2]) if top[2] > 0 else None,
        "cooccurrence_base": base_pair_cooc,
        "cooccurrence_constrained": con_pair_cooc,
        "honored_shap_collapsed": bool(top[2] > 0 and con_pair_shap < 0.05 * top[2]),
        "honored_cooccurrence_zero": bool(con_pair_cooc == 0.0),
    }
    res["part2_effect"]["all_honored"] = bool(
        res["part2_effect"]["honored_shap_collapsed"]
        and res["part2_effect"]["honored_cooccurrence_zero"])
    print(f"[2] SHAP base={top[2]:.6f} -> constrained={con_pair_shap:.6f}; "
          f"cooc {base_pair_cooc:.2f} -> {con_pair_cooc:.2f}; "
          f"all_honored={res['part2_effect']['all_honored']}", flush=True)

    # ---- PART 3: METRICS --------------------------------------------------
    cal_base = fit_isotonic(y_val, ref.predict_proba(X_val))
    cal_con = fit_isotonic(y_val, constrained.predict_proba(X_val))
    metrics = {}
    for seg in ("eval", "test"):
        Xs, ys, mi, ws = segs[seg]
        db = preds_df(ref, cal_base, Xs, ys, mi, ws)
        dc = preds_df(constrained, cal_con, Xs, ys, mi, ws)
        rb = per_day_r_precision(db); rc = per_day_r_precision(dc)
        metrics[seg] = {
            "base_rate_weighted": rb["base_rate_weighted"],
            "brier_weighted_base": wbrier(db),
            "brier_weighted_constrained": wbrier(dc),
            "brier_unweighted_base": float(brier_score_loss(db["y_true"], db["p_calibrated"])),
            "brier_unweighted_constrained": float(brier_score_loss(dc["y_true"], dc["p_calibrated"])),
            "r_precision_weighted_base": rb["r_precision_weighted"],
            "r_precision_weighted_constrained": rc["r_precision_weighted"],
            "n_days_with_positives": rb["n_days_with_positives"],
        }
    res["part3_metrics"] = metrics
    print("[3] metrics:", json.dumps(metrics), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
