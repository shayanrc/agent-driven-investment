"""Aggregate the nifty500 canonical scan (V1.10, task #55): 40 cells -> registry
rows + the base-vs-fund delta table, judged on the held-out TEST window.

Reads each cell's predictions/test.csv (R-p@K via the canonical min(K,R_q) tool) +
metrics.json (AUC, rows, segment dates), appends rows to r_precision_at_k.csv
(dedup by experiment), and prints the fbase-vs-ffund comparison ranked for Phase 2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.gbdt.compute_r_precision import per_day_p_at_k

EXP = Path("results/gbdt/experiments")
REG = Path("results/gbdt/data/r_precision_at_k.csv")
THRESHOLDS = [10, 20, 30, 50]
HORIZONS = [10, 25, 50, 100, 200]
KS = [1, 3, 5, 10, 20]
REG_COLS = ["experiment", "rows", "Q_days", "base_rate", "AUC",
            "R_precision_at_1", "R_precision_at_3", "R_precision_at_5",
            "R_precision_at_10", "R_precision_at_20", "mode", "n_iterations_run",
            "backend", "train_start", "train_end", "val_start", "val_end",
            "eval_start", "eval_end", "test_start", "test_end"]


def one_cell(name: str) -> dict | None:
    cell = EXP / name
    tp = cell / "predictions" / "test.csv"
    mp = cell / "metrics.json"
    if not tp.exists() or not mp.exists():
        return None
    preds = pd.read_csv(tp, parse_dates=["date"])
    rpk = per_day_p_at_k(preds)
    m = json.loads(mp.read_text())
    seg = m.get("segment_dates", {})
    ht = m.get("headline_test", {})
    row = {
        "experiment": name,
        "rows": ht.get("n_rows"),
        "Q_days": rpk["by_k"][1]["n_days_R_gt_0"],
        "base_rate": rpk["base_rate"],
        "AUC": ht.get("roc_auc"),
        "mode": "scan", "n_iterations_run": 1, "backend": "xgboost",
        "train_start": seg.get("train", {}).get("start"),
        "train_end": seg.get("train", {}).get("end"),
        "val_start": seg.get("val", {}).get("start"),
        "val_end": seg.get("val", {}).get("end"),
        "eval_start": seg.get("eval", {}).get("start"),
        "eval_end": seg.get("eval", {}).get("end"),
        "test_start": seg.get("test", {}).get("start"),
        "test_end": seg.get("test", {}).get("end"),
    }
    for k in KS:
        row[f"R_precision_at_{k}"] = rpk["by_k"][k]["r_precision_at_k"]
    return row


def main() -> None:
    rows = []
    missing = []
    for thr in THRESHOLDS:
        dd = thr // 2
        for hor in HORIZONS:
            for arm in ("fbase", "ffund"):
                name = f"nifty500_up_{thr}pct_{hor}d_dd{dd}pct_{arm}_canon"
                r = one_cell(name)
                (rows.append(r) if r else missing.append(name))
    df = pd.DataFrame(rows)
    print(f"aggregated {len(df)}/40 cells; missing {len(missing)}: {missing}")

    # append to registry (dedup by experiment, keep newest)
    if REG.exists():
        reg = pd.read_csv(REG)
        reg = reg[~reg["experiment"].isin(df["experiment"])]
        out = pd.concat([reg, df[REG_COLS]], ignore_index=True)
    else:
        out = df[REG_COLS]
    out.to_csv(REG, index=False)
    print(f"registry -> {REG} ({len(out)} rows)")

    # base-vs-fund delta table
    df["thr"] = df.experiment.str.extract(r"_up_(\d+)pct").astype(int)
    df["hor"] = df.experiment.str.extract(r"pct_(\d+)d").astype(int)
    df["arm"] = df.experiment.str.extract(r"_(fbase|ffund)_canon")
    piv_cols = ["base_rate", "AUC"] + [f"R_precision_at_{k}" for k in KS]
    b = df[df.arm == "fbase"].set_index(["thr", "hor"])[piv_cols]
    f = df[df.arm == "ffund"].set_index(["thr", "hor"])[piv_cols]
    delta = (f - b).add_prefix("d_")
    comp = b.join(f, lsuffix="_base", rsuffix="_fund").join(delta).reset_index().sort_values(["hor", "thr"])
    comp.to_csv("results/gbdt/data/_nifty500_canon_base_vs_fund.csv", index=False)
    print("\n=== fund-minus-base delta by horizon (mean across thresholds) ===")
    dcols = ["d_AUC"] + [f"d_R_precision_at_{k}" for k in KS]
    by_h = comp.groupby("hor")[dcols].mean().round(4)
    print(by_h.to_string())
    print("\nfull comparison -> results/gbdt/data/_nifty500_canon_base_vs_fund.csv")


if __name__ == "__main__":
    main()
