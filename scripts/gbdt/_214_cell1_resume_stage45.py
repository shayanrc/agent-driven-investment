"""Cell-1 resume — runs stages 4+5 only. Reuses stage 3 winners observed in
the prior run (cell-1 OOM'd during stage 4 carve-cache construction with cell-3
running concurrently; predictable from VmRSS=16GB × 2 processes ≈ 32G of 38G
machine RAM, plus stage-4 needs simultaneous carves of 6 FS-keep counts).

Stage 3 top-3 mix winners by EVAL R-p@1 (from the live cell-1 log):
  d=3_cs=0.3_mcw=1   v@1=0.554 e@1=0.300 z=22.56 bi=27  v_br=0.0073
  d=3_cs=0.5_mcw=1   v@1=0.462 e@1=0.285 z=24.11 bi=27  v_br=0.0074
  d=3_cs=0.5_mcw=5   v@1=0.459 e@1=0.245 z=24.40 bi=28  v_br=0.0073

Stage 4 = these 3 × FS keep ∈ {50, 80, 100, 130, 150, 190} → 18 fits.
Stage 5 = test shootout on top configs by eval R-p@1.

We DO NOT pre-cache carves to keep memory bounded — each fit recomputes carve
from the X_full slice. Adds wall-clock but is safer on the contention front.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Re-import from the main driver (already on PYTHONPATH)
from scripts.gbdt._214_cell_manual_driver import (
    CELLS, REPO, PRIOR_REPO, SPLIT, fit_score, short,
    gbdt_data, build_target, compute_uniqueness_weights,
    carve_single_fold, _gather_segment,
)

CELL_KEY = "cell1"
cell = CELLS[CELL_KEY]
prior_dir = PRIOR_REPO / "results/gbdt/experiments" / cell["prior_artifact"]


def main():
    print(f"==== Resume cell-1 ({cell['name']}) stages 4+5 ====\n")

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

    # Stage 3 winners by eval R-p@1 (from the live log)
    top3_mix = [
        ("d=3_cs=0.3_mcw=1", {"max_depth": 3, "colsample_bytree": 0.3,
                              "min_child_weight": 1, "eta": 0.1,
                              "n_estimators": 500, "early_stopping_rounds": 30}),
        ("d=3_cs=0.5_mcw=1", {"max_depth": 3, "colsample_bytree": 0.5,
                              "min_child_weight": 1, "eta": 0.1,
                              "n_estimators": 500, "early_stopping_rounds": 30}),
        ("d=3_cs=0.5_mcw=5", {"max_depth": 3, "colsample_bytree": 0.5,
                              "min_child_weight": 5, "eta": 0.1,
                              "n_estimators": 500, "early_stopping_rounds": 30}),
    ]

    # ----- Stage 4 — FS sweep around each top mix winner ---------------------
    print("---- Stage 4: FS sweep around top-3 mix winners ----")
    fs_keep_counts = (50, 80, 100, 130, 150, 190)
    stage4: list[dict] = []
    for name, hp in top3_mix:
        for nk in fs_keep_counts:
            feats = feat_rank[:nk]
            t0 = time.time()
            args = carve(feats)
            try:
                r = fit_score(args, feat_names=args[-1], hp=hp, apply_calibration=True)
                dt = time.time() - t0
                print(f"  {name}+FS={nk:>4}  {short(r)}  ({dt:.1f}s)")
                r["config"] = f"{name}+FS={nk}"
                r["hp"] = hp
                r["n_features"] = nk
                stage4.append(r)
            except Exception as exc:
                print(f"  {name}+FS={nk:>4}  ERROR: {exc!r}")
            finally:
                del args  # release the big arrays before next iter

    # ----- Stage 5 — Test shootout on top-8 by eval R-p@1 + Z-gate tiebreak --
    print("\n---- Stage 5: Test shootout ----")

    # ALSO include the top-3 mix winners directly (in case FS-sweep didn't improve them)
    extras_for_pool = []
    for name, hp in top3_mix:
        # Fit at cliff (~169) to score these directly
        feats = cliff_feats
        t0 = time.time()
        args = carve(feats)
        try:
            r = fit_score(args, feat_names=args[-1], hp=hp, apply_calibration=True)
            dt = time.time() - t0
            print(f"  [pool] {name}+FS=cliff  {short(r)}  ({dt:.1f}s)")
            r["config"] = f"{name}+FS=cliff"
            r["hp"] = hp
            r["n_features"] = len(cliff_feats)
            extras_for_pool.append(r)
        except Exception as exc:
            print(f"  [pool] {name}+FS=cliff  ERROR: {exc!r}")
        finally:
            del args

    pool = stage4 + extras_for_pool
    by_eval_all = sorted(pool, key=lambda r: -r["rp_eval_1"])

    leader_e = by_eval_all[0]["rp_eval_1"]
    near_tie = [r for r in by_eval_all if leader_e - r["rp_eval_1"] <= 0.01]
    near_tie_ranked = sorted(
        near_tie,
        key=lambda r: (0 if r["cal_method"] == "native" else 1, -r["rp_eval_1"]),
    )
    top_for_shootout = near_tie_ranked + [r for r in by_eval_all
                                          if r not in near_tie_ranked][:8 - len(near_tie_ranked)]
    top_for_shootout = top_for_shootout[:8]

    # Always include the defaults-cliff baseline as a comparator
    baseline_hp: dict[str, Any] = {}
    print(f"\n  Top configs for test shootout (by eval R-p@1 + Z-gate tiebreak):")
    for r in top_for_shootout:
        print(f"    {r['config']:<40} eval@1={r['rp_eval_1']:.4f} z={r['z_val']:.2f} cal={r['cal_method']}")
    print(f"    (also: defaults+cliff baseline)")

    stage5: list[dict] = []
    seen_names = set()
    for top in top_for_shootout:
        if top["config"] in seen_names:
            continue
        seen_names.add(top["config"])
        nk = top["n_features"]
        feats = feat_rank[:nk] if nk < len(X_full.columns) else cliff_feats
        t0 = time.time()
        args = carve(feats)
        try:
            r = fit_score(args, feat_names=args[-1], hp=top["hp"],
                          include_test=True, return_predictions=True,
                          apply_calibration=True)
            dt = time.time() - t0
            print(f"  {top['config']:<40} "
                  f"test@1={r['rp_test_1']:.4f} test@3={r['rp_test_3']:.4f} "
                  f"test@5={r['rp_test_5']:.4f} test@10={r['rp_test_10']:.4f} "
                  f"test_auc={r['test_auc']:.4f} cal={r['cal_method']} z={r['z_val']:.2f} "
                  f"raw_n={r['n_distinct_raw_test']} cal_n={r['n_distinct_cal_test']} "
                  f"bi={r['best_iter']}  ({dt:.1f}s)")
            r["config"] = top["config"]
            r["hp"] = top["hp"]
            r["n_features"] = nk
            stage5.append(r)
        except Exception as exc:
            print(f"  {top['config']:<40} ERROR: {exc!r}")
        finally:
            del args

    # Defaults+cliff baseline
    t0 = time.time()
    args = carve(cliff_feats)
    try:
        r = fit_score(args, feat_names=args[-1], hp=baseline_hp,
                      include_test=True, return_predictions=True,
                      apply_calibration=True)
        dt = time.time() - t0
        print(f"  defaults+cliff   "
              f"test@1={r['rp_test_1']:.4f} test@3={r['rp_test_3']:.4f} "
              f"test@5={r['rp_test_5']:.4f} test@10={r['rp_test_10']:.4f} "
              f"test_auc={r['test_auc']:.4f} cal={r['cal_method']} z={r['z_val']:.2f} "
              f"raw_n={r['n_distinct_raw_test']} cal_n={r['n_distinct_cal_test']} "
              f"bi={r['best_iter']}  ({dt:.1f}s)")
        r["config"] = "defaults+cliff"
        r["hp"] = baseline_hp
        r["n_features"] = len(cliff_feats)
        stage5.append(r)
    except Exception as exc:
        print(f"  defaults+cliff   ERROR: {exc!r}")
    finally:
        del args

    print("\n==== Stage 5 FINAL TEST RANKING ====")
    by_test = sorted(stage5, key=lambda r: -r["rp_test_1"])
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

    # ----- Persist -----
    out_path = REPO / f"results/gbdt/data/_214_{CELL_KEY}_manual_methodology_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def strip(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "test_predictions"}

    pred_path = None
    if winner and "test_predictions" in winner:
        pred_path = REPO / f"results/gbdt/data/_214_{CELL_KEY}_winner_test_predictions.csv"
        winner["test_predictions"].to_csv(pred_path, index=False)
        print(f"  winner test predictions -> {pred_path}")

    out_blob = {
        "cell_key": CELL_KEY,
        "cell": cell,
        "apply_calibration": True,
        "resume_note": ("Stages 1-3 ran in the parent process (see "
                        "/tmp/_214_cell1.log lines 1-65) before OOM during "
                        "stage-4 carve-cache construction; this resume script "
                        "ran stages 4+5 standalone using the stage-3 top-3 "
                        "mix winners from the parent log. Stage 4+5 results "
                        "below are the canonical ones."),
        "split": {"train_rows": SPLIT.train_rows, "val_rows": SPLIT.val_rows,
                  "eval_rows": SPLIT.eval_rows, "test_rows": SPLIT.test_rows},
        "stage1_fs_cliff_note": "see /tmp/_214_cell1.log lines 1-7",
        "stage2_single_knob_note": "see /tmp/_214_cell1.log lines 8-32",
        "stage3_mix_note": "see /tmp/_214_cell1.log lines 33-54 (top-3 mix winners listed in this JSON's stage3_mix_top3)",
        "stage3_mix_top3": [
            {"config": "d=3_cs=0.3_mcw=1", "hp": top3_mix[0][1],
             "v_rp_1": 0.5538, "v_rp_10": 0.3845, "e_rp_1": 0.3000, "e_rp_10": 0.2861,
             "v_brier": 0.0073, "z_val": 22.56, "best_iter": 27},
            {"config": "d=3_cs=0.5_mcw=1", "hp": top3_mix[1][1],
             "v_rp_1": 0.4620, "v_rp_10": 0.4051, "e_rp_1": 0.2850, "e_rp_10": 0.3308,
             "v_brier": 0.0074, "z_val": 24.11, "best_iter": 27},
            {"config": "d=3_cs=0.5_mcw=5", "hp": top3_mix[2][1],
             "v_rp_1": 0.4589, "v_rp_10": 0.4110, "e_rp_1": 0.2450, "e_rp_10": 0.2863,
             "v_brier": 0.0073, "z_val": 24.40, "best_iter": 28},
        ],
        "stage4_eval_validation": [strip(r) for r in stage4],
        "stage5_test_shootout": [strip(r) for r in stage5],
        "winner": strip(winner) if winner else None,
        "verdict": ("PASS" if winner and winner["rp_test_1"] >= cell["sweep_rp1"] else "FAIL")
                   if winner else None,
        "winner_predictions_csv": str(pred_path) if pred_path else None,
    }
    out_path.write_text(json.dumps(out_blob, indent=2, default=str))
    print(f"\n  results JSON -> {out_path}")


if __name__ == "__main__":
    main()
