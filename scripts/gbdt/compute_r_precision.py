"""
Per-day R-precision for the H=25 4-cell cross-market memo.

R-precision per day: rank items by p_calibrated, take top-R(d) where R(d) is
the number of positives that day, score = correct_picks / R(d). Scale-invariant
across markets/days; baseline (random picker) = mean per-day base rate.

Tie-breaking: (p_calibrated desc, ticker asc) with stable mergesort, matching
the runner's per-day P@k convention (src/gbdt/topk_diagnostics.py).

Memo-only — does not modify the runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def per_day_r_precision(preds: pd.DataFrame) -> dict:
    """Compute per-day R-precision.

    Args:
        preds: dataframe with columns date, ticker, p_calibrated, y_true.

    Returns:
        dict with keys:
            r_precision_mean_unweighted: mean over days (R>0) of per-day R-precision
            r_precision_weighted:        sum(correct@R) / sum(R) — global recall@R
            base_rate_mean_unweighted:   mean per-day R(d)/n(d) (random-picker baseline)
            base_rate_weighted:          total positives / total rows
            lift_mean:                   r_precision_mean / base_rate_mean
            lift_weighted:               r_precision_weighted / base_rate_weighted
            n_days_total
            n_days_with_positives
            per_day_rprec_quantiles:     {p10, p25, p50, p75, p90}
            r_distribution:              {min, mean, max} of R(d) for days with R>0
    """
    per_day = []
    for date, day_df in preds.groupby("date"):
        n = len(day_df)
        r = int(day_df["y_true"].sum())
        if r == 0 or r > n:
            # skip degenerate days for the rprec mean; still count for stats
            per_day.append({"date": date, "n": n, "R": r, "correct_at_R": None, "rprec": None})
            continue

        # canonical tie-break: (p_calibrated desc, ticker asc), stable mergesort
        ordered = day_df.sort_values(
            by=["p_calibrated", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
        top_r = ordered.head(r)
        correct = int(top_r["y_true"].sum())
        per_day.append({"date": date, "n": n, "R": r, "correct_at_R": correct, "rprec": correct / r})

    df = pd.DataFrame(per_day)
    valid = df[df["rprec"].notna()].copy()

    if len(valid) == 0:
        return {
            "r_precision_mean_unweighted": None,
            "r_precision_weighted": None,
            "base_rate_mean_unweighted": None,
            "base_rate_weighted": None,
            "lift_mean": None,
            "lift_weighted": None,
            "n_days_total": int(len(df)),
            "n_days_with_positives": 0,
            "per_day_rprec_quantiles": None,
            "r_distribution": None,
        }

    rprec_mean = float(valid["rprec"].mean())
    total_correct = int(valid["correct_at_R"].sum())
    total_r = int(valid["R"].sum())
    rprec_weighted = total_correct / total_r if total_r > 0 else None

    base_rates = (valid["R"] / valid["n"]).astype(float)
    base_rate_mean = float(base_rates.mean())
    total_rows = int(valid["n"].sum())
    base_rate_weighted = total_r / total_rows if total_rows > 0 else None

    quantiles = valid["rprec"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
    quantiles = {f"p{int(q*100)}": float(v) for q, v in quantiles.items()}

    return {
        "r_precision_mean_unweighted": rprec_mean,
        "r_precision_weighted": float(rprec_weighted) if rprec_weighted is not None else None,
        "base_rate_mean_unweighted": base_rate_mean,
        "base_rate_weighted": float(base_rate_weighted) if base_rate_weighted is not None else None,
        "lift_mean": float(rprec_mean / base_rate_mean) if base_rate_mean > 0 else None,
        "lift_weighted": float(rprec_weighted / base_rate_weighted) if rprec_weighted and base_rate_weighted else None,
        "n_days_total": int(len(df)),
        "n_days_with_positives": int(len(valid)),
        "per_day_rprec_quantiles": quantiles,
        "r_distribution": {
            "min": int(valid["R"].min()),
            "mean": float(valid["R"].mean()),
            "max": int(valid["R"].max()),
        },
    }


def per_day_p_at_k(preds: pd.DataFrame, k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Per-day P@k with the corrected denominator ``min(R(d), k)``.

    For each day:
      R(d) = positives that day
      sort by (p_calibrated desc, ticker asc) mergesort
      hits = positives in top k
      day denominator = min(R(d), k)  (achievable positives, not picks-made)

    Weighted aggregate (preferred): sum(hits) / sum(min(R(d), k))
    Mean unweighted: average of per-day values over days where R(d) > 0

    This is the corrected formula post-2026-05-28. Earlier code used
    ``denominator = min(k, n_tickers_in_day)`` (picks-made), which silently
    mis-normalizes on staggered panels where R(d) < k for many days. See
    ``.claude/memories/project-r-precision-methodology.md``.
    """
    by_k = {k: {"hits": 0, "denom": 0, "per_day_values": [], "n_days_R_lt_k": 0}
            for k in k_values}
    n_days = 0
    for _date, day_df in preds.groupby("date"):
        n_days += 1
        r = int(day_df["y_true"].sum())
        ordered = day_df.sort_values(
            by=["p_calibrated", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
        for k in k_values:
            picks = ordered.head(k)
            hits = int(picks["y_true"].sum())
            denom = min(r, k)
            by_k[k]["hits"] += hits
            by_k[k]["denom"] += denom
            if r < k:
                by_k[k]["n_days_R_lt_k"] += 1
            if denom > 0:
                by_k[k]["per_day_values"].append(hits / denom)
    base = float(preds["y_true"].mean()) if len(preds) else 0.0

    out = {"base_rate": base, "n_days_total": n_days, "by_k": {}}
    for k, d in by_k.items():
        weighted = d["hits"] / d["denom"] if d["denom"] > 0 else None
        mean_unw = (sum(d["per_day_values"]) / len(d["per_day_values"])
                    if d["per_day_values"] else None)
        out["by_k"][k] = {
            "p_at_k_weighted": weighted,
            "p_at_k_mean_unweighted": mean_unw,
            "n_hits": d["hits"],
            "n_denom": d["denom"],
            "n_days_R_lt_k": d["n_days_R_lt_k"],
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("preds_csv", type=Path, help="Path to predictions/test.csv (or eval.csv)")
    p.add_argument("--label", default=None, help="Cell label for output (e.g. 'nasdaq H=25 test')")
    p.add_argument("--json", action="store_true", help="Emit JSON only (no human summary)")
    p.add_argument("--pk", action="store_true",
                   help="Also emit per-day P@k (k=1,3,5,10) with corrected min(R(d), k) denominator")
    args = p.parse_args()

    df = pd.read_csv(args.preds_csv, parse_dates=["date"])
    result = per_day_r_precision(df)
    result["label"] = args.label or args.preds_csv.parent.parent.name + " / " + args.preds_csv.stem
    if args.pk:
        result["p_at_k"] = per_day_p_at_k(df)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"=== {result['label']} ===")
    print(f"  rows: {len(df):,}  days_total: {result['n_days_total']}  days_with_pos: {result['n_days_with_positives']}")
    bd = result["base_rate_mean_unweighted"]
    bw = result["base_rate_weighted"]
    print(f"  base rate: {bd:.4f} (mean per-day) / {bw:.4f} (weighted)")
    rd = result["r_precision_mean_unweighted"]
    rw = result["r_precision_weighted"]
    lm = result["lift_mean"]
    lw = result["lift_weighted"]
    print(f"  R-precision: {rd:.4f} (mean per-day, lift {lm:.3f}x) / {rw:.4f} (weighted, lift {lw:.3f}x)")
    q = result["per_day_rprec_quantiles"]
    print(f"  per-day rprec quantiles: p10={q['p10']:.3f} p25={q['p25']:.3f} p50={q['p50']:.3f} p75={q['p75']:.3f} p90={q['p90']:.3f}")
    r = result["r_distribution"]
    print(f"  R(d) distribution: min={r['min']} mean={r['mean']:.1f} max={r['max']}")
    if "p_at_k" in result:
        pk = result["p_at_k"]
        print(f"  P@k (denominator = min(R(d), k); base={pk['base_rate']:.4f}):")
        for k, v in pk["by_k"].items():
            wt = v["p_at_k_weighted"]
            mn = v["p_at_k_mean_unweighted"]
            wt_s = f"{wt:.4f}" if wt is not None else "n/a"
            mn_s = f"{mn:.4f}" if mn is not None else "n/a"
            print(f"    k={k:>2}: weighted={wt_s}  mean_unw={mn_s}  "
                  f"({v['n_hits']}/{v['n_denom']})  days_R<k={v['n_days_R_lt_k']}/{pk['n_days_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
