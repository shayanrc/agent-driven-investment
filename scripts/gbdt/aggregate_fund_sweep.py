#!/usr/bin/env python
"""Aggregate the F18 fundamentals horizon×target sweep into a delta table.

For each canonical cell, reads the ``_fbase`` and ``_ffund`` experiment
artifacts, computes R-Precision@K (via the shared topk methodology) on the test
predictions, and reports the ``fund − base`` deltas for AUC, test Brier, and
R-Precision@{1,10,20} — laid out as a (threshold × horizon) grid so you can see
WHERE fundamentals help. Writes JSON to results/gbdt/data/_274_fund_sweep_data.json.

Usage:
    uv run python -m scripts.gbdt.aggregate_fund_sweep [universe]
"""
import json
import subprocess
import sys
from pathlib import Path

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "sp500"
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results" / "gbdt" / "experiments"
KS = ["1", "3", "5", "10", "20"]


def rpk(exp: str):
    csv = EXP / exp / "predictions" / "test.csv"
    out = subprocess.run(
        ["uv", "run", "python", "scripts/gbdt/compute_r_precision.py", str(csv), "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    d = json.loads(out.stdout)["r_precision_at_k"]
    return d["base_rate"], {k: d["by_k"][k]["r_precision_at_k"] for k in KS}, d["n_days_total"]


def headline(exp: str):
    m = json.loads((EXP / exp / "metrics.json").read_text())
    sd = m["segment_dates"]
    return (m["headline_test"]["roc_auc"], m["headline_test"]["brier"],
            sd["test"]["start"], sd["test"]["end"], m["headline_test"]["n_rows"])


def main() -> None:
    import glob, re, yaml
    EXCLUDE = re.compile(r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|"
                         r"fund|trail|_(da)?sw(base|macro)|_fbase|_ffund")
    cells = []
    for f in sorted(glob.glob(str(ROOT / "configs/gbdt/experiments" / f"{UNIVERSE}_up_*pct_*d_dd*pct.yaml"))):
        b = Path(f).name
        if EXCLUDE.search(b):
            continue
        t = yaml.safe_load(open(f))["target"]
        cells.append((t["threshold_pct"], t["horizon_days"], b[:-5]))
    cells.sort()

    rows = []
    for thr, h, cell in cells:
        fb, ff = f"{cell}_fbase", f"{cell}_ffund"
        if not (EXP / fb / "metrics.json").is_file() or not (EXP / ff / "metrics.json").is_file():
            print(f"  skip {cell} (missing artifacts)")
            continue
        b_rate, b_rpk, ndays = rpk(fb)
        _, f_rpk, _ = rpk(ff)
        b_auc, b_brier, ts, te, nrows = headline(fb)
        f_auc, f_brier, _, _, _ = headline(ff)
        rows.append({
            "threshold_pct": thr, "horizon_days": h, "cell": cell,
            "base_rate": round(b_rate, 5), "n_days": ndays, "test_window": [ts, te],
            "d_auc": round(f_auc - b_auc, 5),
            "d_test_brier": round(f_brier - b_brier, 6),
            "d_rpk": {k: round(f_rpk[k] - b_rpk[k], 5) for k in KS},
            "base_rpk": {k: round(b_rpk[k], 5) for k in KS},
            "fund_rpk": {k: round(f_rpk[k], 5) for k in KS},
        })

    out = {"experiment": "_274_fund_horizon_target_sweep", "universe": UNIVERSE,
           "window": "date_aligned", "snapshot_end": "2026-07-02", "cells": rows}
    dst = ROOT / "results" / "gbdt" / "data" / "_274_fund_sweep_data.json"
    dst.write_text(json.dumps(out, indent=2))

    # grid print: fund−base delta at AUC / Brier / R-p@1 / R-p@10
    print(f"\n=== F18 fund − base deltas ({len(rows)} cells, date_aligned) ===")
    print(f"{'cell':<28} {'base_rate':>9} {'dAUC':>8} {'dBrier':>9} {'dR-p@1':>8} {'dR-p@10':>8}")
    for r in rows:
        print(f"{r['cell']:<28} {r['base_rate']:>9.4f} {r['d_auc']:>+8.4f} "
              f"{r['d_test_brier']:>+9.5f} {r['d_rpk']['1']:>+8.4f} {r['d_rpk']['10']:>+8.4f}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
