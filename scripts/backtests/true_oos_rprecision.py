"""True post-test-set OOS R-Precision for a set of gbdt cells.

For each cell: score its saved model on dates strictly AFTER its published test
window (via the validated ``infer_fresh_predictions`` tool — same feature build +
self-check), attach realized labels with ``build_target`` (NaN until the cell's
horizon of forward bars exists → the complete-label boundary), and compute
R-Precision@K on the labelable OOS days only. This is the strict generalization
check: how the model's top-K ranking holds up on data it never saw, after its test
set.

Denominator is the canonical ``min(K, R_q)`` per day, macro-averaged over days with
R_q > 0; tie-break (p desc, ticker asc, mergesort) matches the registry.

Usage:
    uv run python -m scripts.backtests.true_oos_rprecision \\
        --top-n 10 --end 2026-06-20 --out results/backtests/data/_266_true_oos_data.json
    uv run python -m scripts.backtests.true_oos_rprecision --cells A B C ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gbdt.data import load_panel
from gbdt.targets import build_target

KS = [1, 3, 5, 10, 20]
REG = "results/gbdt/data/r_precision_at_k.csv"


def r_precision_at_k(df: pd.DataFrame) -> dict:
    """df has columns date, p, y (y in {0,1}). Canonical per-day fixed-K, macro avg."""
    out = {}
    for K in KS:
        ratios = []
        for _, g in df.groupby("date"):
            R = int(g["y"].sum())
            if R == 0:
                continue
            top = g.sort_values(["p", "ticker"], ascending=[False, True],
                                kind="mergesort").head(K)
            ratios.append(int(top["y"].sum()) / min(K, R))
        out[K] = float(np.mean(ratios)) if ratios else None
    return out


def score_cell(cell_name: str, end: str) -> dict:
    cell = Path("results/gbdt/experiments") / cell_name
    rec = {"cell": cell_name}
    if not (cell / "spec.yaml").exists() or not (cell / "predictions" / "test.csv").exists():
        return {**rec, "status": "no artifact dir / test.csv on disk"}
    spec = yaml.safe_load((cell / "spec.yaml").read_text())
    tgt = spec["target"]
    metrics = json.load(open(cell / "metrics.json"))
    test_end = metrics["segment_dates"]["test"]["end"]
    rec.update(universe=tgt["universe"], test_end=test_end,
               horizon=tgt["horizon_days"], threshold_pct=tgt["threshold_pct"])

    # 1. fresh predictions for dates > test_end (validated infer; incremental slice)
    fresh_csv = f"/tmp/fresh_{cell_name}.csv"
    try:
        subprocess.run(
            [sys.executable, "-m", "scripts.backtests.infer_fresh_predictions",
             "--cell", str(cell), "--out", fresh_csv, "--end", end, "--since", test_end],
            check=True, capture_output=True, text=True, timeout=1200,
        )
    except subprocess.CalledProcessError as e:
        return {**rec, "status": f"infer failed: {(e.stderr or '')[-300:]}"}
    fresh = pd.read_csv(fresh_csv, parse_dates=["date"])
    if fresh.empty:
        return {**rec, "status": "no fresh predictions (test_end at/after data edge)"}

    # 2. realized labels on the OOS panel (short window + low min_rows: we only
    #    need prices, not model-eligibility). build_target -> NaN past the
    #    complete-label boundary, so only labelable OOS days survive.
    start = str((pd.Timestamp(test_end) - pd.Timedelta(days=20)).date())
    up = load_panel(tgt["universe"], start=start, end=end, min_rows=50)
    y = build_target(up.panel, direction=tgt["direction"],
                     threshold_pct=tgt["threshold_pct"], horizon_days=tgt["horizon_days"],
                     max_drawdown=tgt.get("max_drawdown"))
    y = y.rename("y").reset_index()
    y["date"] = pd.to_datetime(y["date"])

    # 3. join + keep labelable rows (date > test_end already guaranteed by infer)
    m = fresh.merge(y, on=["date", "ticker"], how="inner")
    m = m[m["y"].notna()].copy()
    if m.empty:
        return {**rec, "status": "no labelable OOS days (window shorter than horizon)",
                "n_fresh_days": int(fresh["date"].nunique())}
    m = m.rename(columns={"p_calibrated": "p"})
    res = r_precision_at_k(m[["date", "ticker", "p", "y"]])
    return {
        **rec, "status": "ok",
        "oos_start": str(m["date"].min().date()), "oos_end": str(m["date"].max().date()),
        "n_oos_days": int(m["date"].nunique()), "n_rows": int(len(m)),
        "base_rate": float(m["y"].mean()),
        **{f"R_precision_at_{K}": res[K] for K in KS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--rank-by", default="R_precision_at_3")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--end", default="2026-06-20")
    ap.add_argument("--out", default="results/backtests/data/_266_true_oos_data.json")
    args = ap.parse_args()

    if args.cells:
        cells = args.cells
    else:
        reg = pd.read_csv(REG)
        cells = reg.sort_values(args.rank_by, ascending=False).head(args.top_n)["experiment"].tolist()

    rows = []
    for i, c in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {c} ...", flush=True)
        r = score_cell(c, args.end)
        print(f"    -> {r.get('status')}  "
              + (f"oos {r.get('oos_start')}→{r.get('oos_end')} ndays={r.get('n_oos_days')} "
                 f"@1={r.get('R_precision_at_1')} @3={r.get('R_precision_at_3')}"
                 if r.get("status") == "ok" else ""), flush=True)
        rows.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rank_by": args.rank_by, "end": args.end, "cells": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
