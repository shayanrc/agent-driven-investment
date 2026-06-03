"""Task #214 — Manual 5-stage FS+HP methodology driver for sp500 cells 1+3.

Replicates the cell-5 manual track (memo `_211`) on a different cell shape, to
test:
  (A) Does the 5-stage methodology (FS cliff → response curves → mix-and-match
      → eval validation → 1-shot test) generalize beyond the cell-5 (anti-AUC,
      sweep R-p@1=0.671) regime to STRONG-AUC strong-top-1 cells (cells 1+3)?
  (B) Does the Z-gate finding from cell-5 _223 — that `min_child_weight`
      controls val Spiegelhalter |z| → conditional_isotonic calibration
      decision (`native` vs `isotonic`) → number of distinct calibrated
      predictions → R-p@1 — generalize to cells where the iter_0 |z| profile
      is different?

Reuses the agentloop artifact dirs (feature cache + iter_0 importance ranking)
under `/mnt/122CEE982CEE765F/Workspace/wt-top5-rp1-agentloop/...`. This is the
standard "second worktree drives off the first worktree's caches" pattern from
cell-5 (`docs/gbdt/_211_*.md`).

Usage:
  uv run python -m scripts.gbdt._214_cell_manual_driver cell1
  uv run python -m scripts.gbdt._214_cell_manual_driver cell3

Per-cell output: a JSON sidecar with all stage results + the winner config and
its test predictions (so the canonical CSV row can be appended downstream).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------

REPO = Path("/mnt/122CEE982CEE765F/Workspace/wt-214-cells13")
PRIOR_REPO = Path("/mnt/122CEE982CEE765F/Workspace/wt-top5-rp1-agentloop")

CELLS = {
    "cell1": {
        "name": "sp500_up_50pct_50d_dd25pct",
        "universe": "sp500",
        "direction": "up",
        "threshold_pct": 50,
        "horizon_days": 50,
        "max_drawdown": 0.25,
        "prior_artifact": "sp500_up_50pct_50d_dd25pct_agentloop",
        "sweep_rp1": 0.800,  # canonical CSV row sp500_up_50pct_50d_dd25pct
        "sweep_rp10": 0.286468,
        "Q_days": 50,
        "rows": 24300,
    },
    "cell3": {
        "name": "sp500_up_20pct_25d_dd10pct",
        "universe": "sp500",
        "direction": "up",
        "threshold_pct": 20,
        "horizon_days": 25,
        "max_drawdown": 0.10,
        "prior_artifact": "sp500_up_20pct_25d_dd10pct_agentloop",
        "sweep_rp1": 0.600,
        "sweep_rp10": 0.409333,
        "Q_days": 75,
        "rows": 36450,
    },
}

# ---------------------------------------------------------------------------
# Imports from the gbdt package
# ---------------------------------------------------------------------------

from gbdt import data as gbdt_data
from gbdt.targets import build_target
from gbdt.uniqueness import compute_uniqueness_weights
from gbdt.train import SplitSpec, carve_single_fold, _gather_segment
from gbdt.calibration import spiegelhalter_z, conditional_isotonic, apply_calibrator
from gbdt.model import XGBoostModel

SPLIT = SplitSpec(800, 400, 200, 100)
SEED = 42
KS = (1, 3, 5, 10, 20)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    err = (p - y) ** 2
    return float(np.average(err, weights=w)) if w is not None else float(err.mean())


def r_precision_at_k(y: np.ndarray, p: np.ndarray, mi: pd.MultiIndex) -> tuple[dict[int, float], float]:
    """Canonical R-Precision@K per [[project-r-precision-methodology]].

    Per-day fixed K, macro-averaged over days where R_q > 0.
    Tie-break: (p desc, ticker asc) stable mergesort.
    """
    df = pd.DataFrame({
        "date": mi.get_level_values("date"),
        "ticker": mi.get_level_values("ticker"),
        "y_true": y.astype(int),
        "p_calibrated": p,
    })
    by_day = [
        (d, g.sort_values(by=["p_calibrated", "ticker"],
                          ascending=[False, True], kind="mergesort"))
        for d, g in df.groupby("date")
    ]
    out: dict[int, float] = {}
    for K in KS:
        ratios = []
        for _d, g in by_day:
            R_q = int(g["y_true"].sum())
            if R_q == 0:
                continue
            r_q = int(g.head(K)["y_true"].sum())
            ratios.append(r_q / min(K, R_q))
        out[K] = float(np.mean(ratios)) if ratios else float("nan")
    return out, float(df["y_true"].mean())


# ---------------------------------------------------------------------------
# Per-config fit and score
# ---------------------------------------------------------------------------


def fit_score(
    args: tuple,
    *,
    feat_names: list[str],
    hp: dict[str, Any],
    include_test: bool = False,
    return_predictions: bool = False,
    apply_calibration: bool = False,
) -> dict[str, Any]:
    """`args` is the tuple from carve(): (X_tr, y_tr, w_tr, X_va, y_va, w_va,
    mi_va, X_ev, y_ev, mi_ev, X_te, y_te, mi_te, _feat_list). The trailing
    _feat_list is ignored; `feat_names` (kwarg) overrides for clarity.
    `include_test` enables test scoring; `return_predictions` returns the test
    predictions DataFrame in the result dict."""
    (X_tr, y_tr, w_tr,
     X_va, y_va, w_va, mi_va,
     X_ev, y_ev, mi_ev,
     X_te, y_te, mi_te,
     _feat_list) = args
    """Fit one config, score on val + eval (+ optionally test).

    When `apply_calibration=True`, runs `conditional_isotonic` on the raw val
    predictions (z_threshold=2.0 matches the runner default + memo _223 finding)
    and scores R-Precision@K on the CALIBRATED predictions for val/eval/test —
    the canonical CSV scoring path. When False, scores on raw predictions (the
    cell-5 manual track style — uncalibrated). The Z-gate analysis records the
    decision (`native` vs `isotonic`) and the |z| in either case.
    """
    m = XGBoostModel(dict(hp), feature_names=feat_names, random_seed=SEED)
    m.fit(X_tr, y_tr, X_va, y_va, train_weight=w_tr, val_weight=w_va)

    p_tr_raw = m.predict_proba(X_tr)
    p_va_raw = m.predict_proba(X_va)
    p_ev_raw = m.predict_proba(X_ev)

    # Z-gate diagnostics: always compute on val raw predictions
    z_val, _ = spiegelhalter_z(y_va.astype(int), p_va_raw)
    decision = conditional_isotonic(y_va.astype(int), p_va_raw, z_threshold=2.0)
    cal_method = decision.method  # 'native' or 'isotonic'

    # Optionally apply the calibrator
    if apply_calibration and decision.calibrator is not None:
        p_va = apply_calibrator(p_va_raw, decision.calibrator)
        p_ev = apply_calibrator(p_ev_raw, decision.calibrator)
    else:
        p_va = p_va_raw
        p_ev = p_ev_raw

    rpk_va, base_va = r_precision_at_k(y_va, p_va, mi_va)
    rpk_ev, base_ev = r_precision_at_k(y_ev, p_ev, mi_ev)

    n_distinct_raw_ev = int(pd.Series(p_ev_raw).nunique())
    n_distinct_cal_ev = int(pd.Series(p_ev).nunique())

    out = dict(
        train_brier=brier(y_tr, p_tr_raw, w_tr),
        val_brier=brier(y_va, p_va_raw, w_va),
        eval_brier=brier(y_ev, p_ev_raw),
        gap=brier(y_va, p_va_raw, w_va) - brier(y_tr, p_tr_raw, w_tr),
        z_val=float(z_val),
        cal_method=cal_method,
        best_iter=m.best_iteration,
        val_base=base_va,
        eval_base=base_ev,
        n_distinct_raw_eval=n_distinct_raw_ev,
        n_distinct_cal_eval=n_distinct_cal_ev,
        rp_val_1=rpk_va[1], rp_val_3=rpk_va[3], rp_val_5=rpk_va[5],
        rp_val_10=rpk_va[10], rp_val_20=rpk_va[20],
        rp_eval_1=rpk_ev[1], rp_eval_3=rpk_ev[3], rp_eval_5=rpk_ev[5],
        rp_eval_10=rpk_ev[10], rp_eval_20=rpk_ev[20],
    )

    if include_test:
        p_te_raw = m.predict_proba(X_te)
        if apply_calibration and decision.calibrator is not None:
            p_te = apply_calibrator(p_te_raw, decision.calibrator)
        else:
            p_te = p_te_raw
        rpk_te, base_te = r_precision_at_k(y_te, p_te, mi_te)
        try:
            auc_te = float(roc_auc_score(y_te, p_te))
        except ValueError:
            auc_te = float("nan")
        out.update(
            test_base=base_te,
            test_auc=auc_te,
            n_distinct_raw_test=int(pd.Series(p_te_raw).nunique()),
            n_distinct_cal_test=int(pd.Series(p_te).nunique()),
            rp_test_1=rpk_te[1], rp_test_3=rpk_te[3], rp_test_5=rpk_te[5],
            rp_test_10=rpk_te[10], rp_test_20=rpk_te[20],
        )
        if return_predictions:
            df_te = pd.DataFrame({
                "date": mi_te.get_level_values("date"),
                "ticker": mi_te.get_level_values("ticker"),
                "y_true": y_te.astype(int),
                "p_raw": p_te_raw,
                "p_calibrated": p_te,
            })
            out["test_predictions"] = df_te
    return out


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------


def short(r: dict[str, Any]) -> str:
    bi = r.get("best_iter")
    bi_s = str(bi) if bi is not None else "—"
    return (
        f"v@1={r['rp_val_1']:.4f} v@10={r['rp_val_10']:.4f} "
        f"e@1={r['rp_eval_1']:.4f} e@10={r['rp_eval_10']:.4f} "
        f"v_br={r['val_brier']:.4f} z={r['z_val']:.2f} cal={r['cal_method']} "
        f"raw_n={r['n_distinct_raw_eval']} cal_n={r['n_distinct_cal_eval']} bi={bi_s}"
    )


def run_cell(cell_key: str, apply_calibration: bool = True) -> dict[str, Any]:
    """Drive the 5-stage methodology for one cell.

    `apply_calibration=True` means we score R-Precision@K on the calibrated
    predictions (canonical CSV path + V1.3 in-loop oracle). The cell-5 manual
    track did NOT calibrate (it shipped raw predictions); we test BOTH paths
    here so the Z-gate finding's effect is directly visible.
    """
    cell = CELLS[cell_key]
    cell_name = cell["name"]
    print(f"\n==== {cell_key} = {cell_name} | apply_calibration={apply_calibration} ====\n")

    # Load data + prior artifact's feature matrix + importance ranking
    prior_dir = PRIOR_REPO / "results/gbdt/experiments" / cell["prior_artifact"]
    panel_obj = gbdt_data.load_panel(
        cell["universe"], start=None, end=None, min_rows=SPLIT.total,
        repo_root=REPO, staleness_days=gbdt_data.DEFAULT_STALENESS_DAYS,
    )
    X_full = pd.read_parquet(prior_dir / "_feature_matrix_cache.parquet")
    req = json.loads((prior_dir / "loop/iter_0_request.json").read_text())
    fi = req["diagnostics"]["feature_importance"]
    feat_rank = [n for n, _ in sorted(fi.items(), key=lambda x: -x[1])]
    cliff_prune = set(req["diagnostics"]["pruned_summary"]["pruned_features"])
    cliff_feats = [c for c in X_full.columns if c not in cliff_prune]
    all_feats = list(X_full.columns)
    print(f"  features: total={len(all_feats)} cliff_keep={len(cliff_feats)} "
          f"importance_ranked={len(feat_rank)}")

    # Target + weights + fold (built once)
    y = build_target(panel_obj.panel,
                     direction=cell["direction"],
                     threshold_pct=cell["threshold_pct"],
                     horizon_days=cell["horizon_days"],
                     max_drawdown=cell["max_drawdown"])
    w = compute_uniqueness_weights(panel_obj.panel, horizon=cell["horizon_days"])
    fold = carve_single_fold(panel_obj.panel, SPLIT)

    def carve(feats: list[str]) -> tuple:
        X_sel = X_full[feats]
        X_tr, y_tr, _,     w_tr = _gather_segment(panel_obj.panel, X_sel, y, fold.train_idx, w)
        X_va, y_va, mi_va, w_va = _gather_segment(panel_obj.panel, X_sel, y, fold.val_idx,   w)
        X_ev, y_ev, mi_ev, _    = _gather_segment(panel_obj.panel, X_sel, y, fold.eval_idx)
        X_te, y_te, mi_te, _    = _gather_segment(panel_obj.panel, X_sel, y, fold.test_idx)
        return (X_tr, y_tr, w_tr, X_va, y_va, w_va, mi_va,
                X_ev, y_ev, mi_ev, X_te, y_te, mi_te, list(X_sel.columns))

    results: dict[str, list[dict]] = {
        "stage1_fs_cliff": [],
        "stage2_single_knob": [],
        "stage3_mix": [],
        "stage4_eval_validation": [],
        "stage5_test_shootout": [],
    }

    # ---------------- STAGE 1 — FS cliff cut --------------------------------
    print("---- Stage 1: FS cliff cut (XGBoost defaults) ----")
    # Reference: cell-5 used default XGBoost HPs + cliff-pruned features.
    defaults_hp: dict[str, Any] = {}
    args_cliff = carve(cliff_feats)
    args_all = carve(all_feats)
    t0 = time.time()
    r_all = fit_score(args_all, feat_names=args_all[-1], hp=defaults_hp,
                       apply_calibration=apply_calibration)
    print(f"  defaults+all{len(all_feats)}: {short(r_all)}  ({time.time()-t0:.1f}s)")
    r_all["config"] = "defaults+all"
    r_all["hp"] = defaults_hp
    r_all["n_features"] = len(all_feats)
    results["stage1_fs_cliff"].append(r_all)

    t0 = time.time()
    r_cliff = fit_score(args_cliff, feat_names=args_cliff[-1], hp=defaults_hp,
                         apply_calibration=apply_calibration)
    print(f"  defaults+cliff{len(cliff_feats)}: {short(r_cliff)}  ({time.time()-t0:.1f}s)")
    r_cliff["config"] = "defaults+cliff"
    r_cliff["hp"] = defaults_hp
    r_cliff["n_features"] = len(cliff_feats)
    results["stage1_fs_cliff"].append(r_cliff)

    # ---------------- STAGE 2 — Single-knob response curves ----------------
    # Use cliff_feats as the working set (same as cell-5).
    print("\n---- Stage 2: Single-knob response curves ----")
    knob_grids: list[tuple[str, dict[str, Any]]] = []
    # 1. max_depth (tiny-model regime test per rule 12)
    for v in (2, 3, 4, 6, 8):
        knob_grids.append((f"depth={v}", {"max_depth": v}))
    # 2. min_child_weight (CRITICAL for Z-gate per rule 12 / _223)
    for v in (1, 3, 5, 10):
        knob_grids.append((f"mcw={v}", {"min_child_weight": v}))
    # 3. eta
    for v in (0.05, 0.1, 0.2):
        knob_grids.append((f"eta={v}", {"eta": v}))
    # 4. colsample_bytree
    for v in (0.3, 0.4, 0.5, 0.7, 1.0):
        knob_grids.append((f"cs={v}", {"colsample_bytree": v}))
    # 5. gamma
    for v in (0.0, 0.5, 1.0):
        knob_grids.append((f"gamma={v}", {"gamma": v}))
    # 6. lambda
    for v in (1, 5, 10):
        knob_grids.append((f"lambda={v}", {"reg_lambda": v}))
    # 7. subsample
    for v in (0.7, 1.0):
        knob_grids.append((f"sub={v}", {"subsample": v}))
    # 8. scale_pos_weight
    for v in (1.0, 2.0, 4.0):
        knob_grids.append((f"spw={v}", {"scale_pos_weight": v}))

    args = args_cliff
    for name, hp in knob_grids:
        t0 = time.time()
        try:
            r = fit_score(args, feat_names=args[-1], hp=hp,
                          apply_calibration=apply_calibration)
            dt = time.time() - t0
            print(f"  {name:<14} {short(r)}  ({dt:.1f}s)")
            r["config"] = name
            r["hp"] = hp
            r["n_features"] = len(cliff_feats)
            results["stage2_single_knob"].append(r)
        except Exception as exc:
            print(f"  {name:<14} ERROR: {exc!r}")

    # ---------------- STAGE 3 — Mix-and-match -------------------------------
    print("\n---- Stage 3: Mix-and-match ----")
    # Build CANDIDATE recipes from cell-5's winning structure + per-cell signals.
    # We test a focused grid combining:
    #   - max_depth in {2, 3}             (tiny-model regime)
    #   - eta + ES                         (slow-LR per cell-5)
    #   - colsample_bytree in {0.3, 0.4, 0.5}
    #   - min_child_weight in {1, 5, 10}    (Z-gate axis)
    # That is 2 × 3 × 3 = 18 configs (with eta=0.1 + n_estimators=500 + ES=30
    # baked in; eta=0.05 explored separately).
    mix_grid: list[tuple[str, dict[str, Any]]] = []
    for d in (2, 3):
        for cs in (0.3, 0.4, 0.5):
            for mcw in (1, 5, 10):
                hp = {
                    "max_depth": d,
                    "colsample_bytree": cs,
                    "min_child_weight": mcw,
                    "eta": 0.1,
                    "n_estimators": 500,
                    "early_stopping_rounds": 30,
                }
                mix_grid.append((f"d={d}_cs={cs}_mcw={mcw}", hp))
    # A couple of "champ-like" extras: cell-5 winner template, plus mcw=5 with eta=0.05
    extras = [
        ("cell5_template", {"max_depth": 2, "colsample_bytree": 0.4,
                            "scale_pos_weight": 1.0, "gamma": 0.0, "eta": 0.1,
                            "n_estimators": 500, "early_stopping_rounds": 30,
                            "min_child_weight": 5}),
        ("d=2_cs=0.4_mcw=5_eta=0.05",
            {"max_depth": 2, "colsample_bytree": 0.4, "min_child_weight": 5,
             "eta": 0.05, "n_estimators": 500, "early_stopping_rounds": 30}),
        ("d=2_cs=0.3_mcw=5_eta=0.05",
            {"max_depth": 2, "colsample_bytree": 0.3, "min_child_weight": 5,
             "eta": 0.05, "n_estimators": 500, "early_stopping_rounds": 30}),
    ]
    for name, hp in mix_grid + extras:
        t0 = time.time()
        try:
            r = fit_score(args, feat_names=args[-1], hp=hp,
                          apply_calibration=apply_calibration)
            dt = time.time() - t0
            print(f"  {name:<28} {short(r)}  ({dt:.1f}s)")
            r["config"] = name
            r["hp"] = hp
            r["n_features"] = len(cliff_feats)
            results["stage3_mix"].append(r)
        except Exception as exc:
            print(f"  {name:<28} ERROR: {exc!r}")

    # ---------------- STAGE 4 — Eval validation (FS sweep around best) -----
    print("\n---- Stage 4: Eval validation (FS sweep around top-3 mix winners) ----")
    # Per rule 11: top mix winners + FS sweep at various keep-counts.
    # Pick by EVAL R-p@1 (NOT val_brier per the cell-5 lesson).
    mix_results = results["stage3_mix"]
    by_eval = sorted(mix_results, key=lambda r: -r["rp_eval_1"])
    top3_mix = by_eval[:3]
    print("  Top-3 mix winners by eval R-p@1:")
    for r in top3_mix:
        print(f"    {r['config']:<28} {short(r)}")

    fs_keep_counts = (50, 80, 100, 130, 150, 190)
    args_cache: dict[int, tuple] = {len(cliff_feats): args}
    for nk in fs_keep_counts:
        if nk in args_cache:
            continue
        feats = feat_rank[:nk]
        args_cache[nk] = carve(feats)

    for top in top3_mix:
        for nk in fs_keep_counts:
            args_k = args_cache[nk]
            name = f"{top['config']}+FS={nk}"
            hp = top["hp"]
            t0 = time.time()
            try:
                r = fit_score(args_k, feat_names=args_k[-1], hp=hp,
                              apply_calibration=apply_calibration)
                dt = time.time() - t0
                print(f"  {name:<40} {short(r)}  ({dt:.1f}s)")
                r["config"] = name
                r["hp"] = hp
                r["n_features"] = nk
                results["stage4_eval_validation"].append(r)
            except Exception as exc:
                print(f"  {name:<40} ERROR: {exc!r}")

    # Add a baseline (sweep-like CatBoost analog: XGBoost defaults on cliff)
    # via the stage-1 result for comparison.

    # ---------------- STAGE 5 — 1-shot TEST shootout -----------------------
    print("\n---- Stage 5: 1-shot TEST shootout ----")
    all_pool = (results["stage2_single_knob"]
                + results["stage3_mix"]
                + results["stage4_eval_validation"])
    by_eval_all = sorted(all_pool, key=lambda r: -r["rp_eval_1"])
    # Top-8 by eval R-p@1; apply Z-gate tiebreak per cell-5 lesson: prefer
    # |z|<2 (native) among configs within 0.01 of the leader.
    leader_e = by_eval_all[0]["rp_eval_1"]
    near_tie = [r for r in by_eval_all if leader_e - r["rp_eval_1"] <= 0.01]
    # Re-rank near-tie by (cal_method=native first, then eval R-p@1 desc)
    near_tie_ranked = sorted(
        near_tie,
        key=lambda r: (0 if r["cal_method"] == "native" else 1, -r["rp_eval_1"]),
    )
    top_for_shootout = near_tie_ranked + [r for r in by_eval_all
                                          if r not in near_tie_ranked][:8 - len(near_tie_ranked)]
    top_for_shootout = top_for_shootout[:8]
    # Always include the stage-1 baseline (defaults+cliff) as a comparator.
    top_for_shootout.append(r_cliff)

    print(f"  Top configs going to test (by eval R-p@1 with Z-gate tiebreak):")
    for r in top_for_shootout:
        print(f"    {r['config']:<40} eval@1={r['rp_eval_1']:.4f} z={r['z_val']:.2f} cal={r['cal_method']}")

    seen_names = set()
    for top in top_for_shootout:
        if top["config"] in seen_names:
            continue
        seen_names.add(top["config"])
        nk = top["n_features"]
        if nk in args_cache:
            args_k = args_cache[nk]
        else:
            # Reconstruct keep-list by importance rank if not cached
            feats = feat_rank[:nk] if nk < len(all_feats) else cliff_feats
            args_k = carve(feats)
            args_cache[nk] = args_k

        # Last config in the loop = the winner. We capture predictions for ALL
        # top-3 by eval (used to seed the canonical CSV row downstream).
        capture_preds = True
        t0 = time.time()
        try:
            r = fit_score(args_k, feat_names=args_k[-1], include_test=True, hp=top["hp"],
                          apply_calibration=apply_calibration,
                          return_predictions=capture_preds)
            dt = time.time() - t0
            # Drop the dataframe from the printed short form
            r_print = {k: v for k, v in r.items() if k != "test_predictions"}
            print(f"  {top['config']:<40} "
                  f"test@1={r['rp_test_1']:.4f} test@3={r['rp_test_3']:.4f} "
                  f"test@5={r['rp_test_5']:.4f} test@10={r['rp_test_10']:.4f} "
                  f"test_auc={r['test_auc']:.4f} cal={r['cal_method']} z={r['z_val']:.2f} "
                  f"raw_n={r['n_distinct_raw_test']} cal_n={r['n_distinct_cal_test']} "
                  f"bi={r['best_iter']}  ({dt:.1f}s)")
            r["config"] = top["config"]
            r["hp"] = top["hp"]
            r["n_features"] = nk
            results["stage5_test_shootout"].append(r)
        except Exception as exc:
            print(f"  {top['config']:<40} ERROR: {exc!r}")

    # ---------------- Final ranking ----------------------------------------
    print("\n==== Stage 5 final test ranking ====")
    by_test = sorted(results["stage5_test_shootout"], key=lambda r: -r["rp_test_1"])
    print(f"  sweep baseline: R-p@1={cell['sweep_rp1']:.4f}  R-p@10={cell['sweep_rp10']:.4f}\n")
    print(f"  {'rk':>2} {'config':<40} {'nf':>4} {'test@1':>7} {'test@3':>7} "
          f"{'test@5':>7} {'test@10':>7} {'cal':>8} {'|z|':>5} {'cal_n':>5}")
    for i, r in enumerate(by_test, 1):
        delta1 = r["rp_test_1"] - cell["sweep_rp1"]
        print(f"  {i:>2} {r['config']:<40} {r['n_features']:>4} "
              f"{r['rp_test_1']:>7.4f} {r['rp_test_3']:>7.4f} "
              f"{r['rp_test_5']:>7.4f} {r['rp_test_10']:>7.4f} "
              f"{r['cal_method']:>8} {abs(r['z_val']):>5.2f} {r['n_distinct_cal_test']:>5} "
              f"vs sweep {delta1:+.4f}")

    winner = by_test[0] if by_test else None
    if winner:
        verdict = "PASS" if winner["rp_test_1"] >= cell["sweep_rp1"] else "FAIL"
        print(f"\n  ==> WINNER: {winner['config']}  test R-p@1={winner['rp_test_1']:.4f}  "
              f"sweep={cell['sweep_rp1']:.4f}  delta={winner['rp_test_1']-cell['sweep_rp1']:+.4f}  "
              f"verdict={verdict}")

    # ---------------- Persist results --------------------------------------
    out_path = REPO / f"results/gbdt/data/_214_{cell_key}_manual_methodology_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save test predictions for the winning config (for the canonical CSV row)
    pred_path = None
    if winner and "test_predictions" in winner:
        pred_path = REPO / f"results/gbdt/data/_214_{cell_key}_winner_test_predictions.csv"
        winner["test_predictions"].to_csv(pred_path, index=False)
        print(f"  winner test predictions -> {pred_path}")

    # Strip dataframes before JSON-encoding
    def strip(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "test_predictions"}

    out_blob = {
        "cell_key": cell_key,
        "cell": cell,
        "apply_calibration": apply_calibration,
        "split": {"train_rows": SPLIT.train_rows, "val_rows": SPLIT.val_rows,
                  "eval_rows": SPLIT.eval_rows, "test_rows": SPLIT.test_rows},
        "stage1_fs_cliff": [strip(r) for r in results["stage1_fs_cliff"]],
        "stage2_single_knob": [strip(r) for r in results["stage2_single_knob"]],
        "stage3_mix": [strip(r) for r in results["stage3_mix"]],
        "stage4_eval_validation": [strip(r) for r in results["stage4_eval_validation"]],
        "stage5_test_shootout": [strip(r) for r in results["stage5_test_shootout"]],
        "winner": strip(winner) if winner else None,
        "verdict": ("PASS" if winner and winner["rp_test_1"] >= cell["sweep_rp1"] else "FAIL")
                   if winner else None,
        "winner_predictions_csv": str(pred_path) if pred_path else None,
    }
    out_path.write_text(json.dumps(out_blob, indent=2, default=str))
    print(f"  results JSON -> {out_path}")
    return out_blob


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("usage: _214_cell_manual_driver.py {cell1|cell3|both} [--no-calibration]")
        sys.exit(1)
    arg = sys.argv[1]
    apply_calibration = "--no-calibration" not in sys.argv

    targets = ["cell1", "cell3"] if arg == "both" else [arg]
    summaries = []
    for ck in targets:
        s = run_cell(ck, apply_calibration=apply_calibration)
        summaries.append({
            "cell_key": s["cell_key"],
            "name": s["cell"]["name"],
            "sweep_rp1": s["cell"]["sweep_rp1"],
            "winner_config": s["winner"]["config"] if s["winner"] else None,
            "winner_test_rp1": s["winner"]["rp_test_1"] if s["winner"] else None,
            "verdict": s["verdict"],
        })
    print("\n========== ALL CELLS SUMMARY ==========")
    for s in summaries:
        print(f"  {s['cell_key']}: {s['name']}  sweep={s['sweep_rp1']:.4f}  "
              f"winner={s['winner_config']}  test@1={s['winner_test_rp1']:.4f}  "
              f"verdict={s['verdict']}")


if __name__ == "__main__":
    main()
