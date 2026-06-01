"""
Per-day R-Precision@K + legacy weighted R-precision for gbdt experiment cells.

R-Precision@K (current, post-2026-06-01 — the project headline cross-cell metric):
  For each day q:
    R_q = positives that day
    sort by (p_calibrated desc, ticker asc) mergesort
    r_q = positives in top K picks
    per-day ratio = r_q / min(K, R_q)  (skip if R_q == 0)
  Macro: R-Precision@K = (1/Q) * sum_q ratio  (Q = days with R_q > 0)
  Standard K = {1, 3, 5, 10, 20}.

Legacy weighted R-precision (pre-2026-06-01, retained for cross-walk):
  Per-day variable K = R(d); micro aggregation: sum(correct@R(d)) / sum(R(d)).
  Different metric — NOT a special case of R-Precision@K at K=R(d).

Tie-breaking: (p_calibrated desc, ticker asc) with stable mergesort, matching
the runner's per-day P@k convention (src/gbdt/topk_diagnostics.py).

See .claude/memories/project-r-precision-methodology.md for the full definition,
relationship between the two metrics, and reporting conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# per_day_r_precision lives in the package (src/gbdt/diagnose_core.py) so the
# V1.1 in-loop diagnose payload can reuse the SAME function object without
# importing from this ad-hoc scripts/ tree. Re-exported here so this module's
# public API + CLI are unchanged.
from gbdt.diagnose_core import per_day_r_precision  # noqa: F401  (re-exported)


def per_day_p_at_k(preds: pd.DataFrame, k_values: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict:
    """Per-day top-K metrics: emits both R-Precision@K (macro, current headline)
    and the micro-aggregated form (sum/sum, legacy).

    For each day:
      R(d) = positives that day
      sort by (p_calibrated desc, ticker asc) mergesort
      hits = positives in top k
      day denominator = min(R(d), k)  (achievable positives, not picks-made)

    Returned per K:
      ``r_precision_at_k``        — macro: (1/Q) * sum(hits / min(R(d), k))  [project headline]
      ``p_at_k_micro``            — micro: sum(hits) / sum(min(R(d), k))     [legacy form]
      ``n_hits`` / ``n_denom``    — raw counters backing the micro form
      ``n_days_R_lt_k``           — days where R(d) < k (denominator pinned at R(d))

    Denominator is ``min(R(d), k)`` per the corrected formula (post-2026-05-28).
    See ``.claude/memories/project-r-precision-methodology.md``.
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
        p_at_k_micro = d["hits"] / d["denom"] if d["denom"] > 0 else None
        r_precision_at_k = (sum(d["per_day_values"]) / len(d["per_day_values"])
                            if d["per_day_values"] else None)
        out["by_k"][k] = {
            "r_precision_at_k": r_precision_at_k,
            "p_at_k_micro": p_at_k_micro,
            "n_hits": d["hits"],
            "n_denom": d["denom"],
            "n_days_R_lt_k": d["n_days_R_lt_k"],
            "n_days_R_gt_0": len(d["per_day_values"]),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("preds_csv", type=Path, help="Path to predictions/test.csv (or eval.csv)")
    p.add_argument("--label", default=None, help="Cell label for output (e.g. 'nasdaq H=25 test')")
    p.add_argument("--json", action="store_true", help="Emit JSON only (no human summary)")
    p.add_argument("--no-rpk", action="store_true",
                   help="Skip the R-Precision@K (macro) emission. Default is to emit at K=1,3,5,10,20.")
    p.add_argument("--no-legacy", action="store_true",
                   help="Skip the legacy weighted R-precision (per-day variable K) emission.")
    args = p.parse_args()

    df = pd.read_csv(args.preds_csv, parse_dates=["date"])
    label = args.label or args.preds_csv.parent.parent.name + " / " + args.preds_csv.stem
    if args.no_legacy:
        result = {"label": label}
    else:
        result = per_day_r_precision(df)
        result["label"] = label
    if not args.no_rpk:
        result["r_precision_at_k"] = per_day_p_at_k(df)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"=== {result['label']} ===")
    if "r_precision_at_k" in result:
        rpk = result["r_precision_at_k"]
        print(f"  base rate: {rpk['base_rate']:.4f}  days_total: {rpk['n_days_total']}")
        print(f"  R-Precision@K (macro: (1/Q) * sum r_q / min(K, R_q) over Q days with R_q > 0):")
        for k, v in rpk["by_k"].items():
            rpk_v = v["r_precision_at_k"]
            mic_v = v["p_at_k_micro"]
            rpk_s = f"{rpk_v:.4f}" if rpk_v is not None else "n/a"
            mic_s = f"{mic_v:.4f}" if mic_v is not None else "n/a"
            lift = (rpk_v / rpk["base_rate"]) if (rpk_v is not None and rpk["base_rate"] > 0) else None
            lift_s = f"  lift={lift:.2f}x" if lift is not None else ""
            print(f"    K={k:>2}: R-Precision@K={rpk_s}{lift_s}  micro_form={mic_s}  "
                  f"({v['n_hits']}/{v['n_denom']})  days_R<K={v['n_days_R_lt_k']}/{rpk['n_days_total']}")
    if "r_precision_weighted" in result:
        print(f"  --- legacy weighted R-precision (per-day variable K=R(d), micro aggregation) ---")
        print(f"  rows: {len(df):,}  days_total: {result['n_days_total']}  days_with_pos: {result['n_days_with_positives']}")
        bd = result["base_rate_mean_unweighted"]
        bw = result["base_rate_weighted"]
        print(f"  base rate: {bd:.4f} (mean per-day) / {bw:.4f} (weighted)")
        rd = result["r_precision_mean_unweighted"]
        rw = result["r_precision_weighted"]
        lm = result["lift_mean"]
        lw = result["lift_weighted"]
        print(f"  weighted R-precision (legacy): {rd:.4f} (mean per-day, lift {lm:.3f}x) / {rw:.4f} (weighted, lift {lw:.3f}x)")
        q = result["per_day_rprec_quantiles"]
        print(f"  per-day rprec quantiles: p10={q['p10']:.3f} p25={q['p25']:.3f} p50={q['p50']:.3f} p75={q['p75']:.3f} p90={q['p90']:.3f}")
        r = result["r_distribution"]
        print(f"  R(d) distribution: min={r['min']} mean={r['mean']:.1f} max={r['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
