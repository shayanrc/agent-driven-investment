"""
Cross-cell per-ticker analysis for NSE H=25 cells (nifty50 + nifty100).

For the combined H=25 memo (#138): identify NSE tickers that the model
systematically over-picks but under-performs on, especially ones that
recur across both nifty50 and nifty100. Cross-cell recurrence is the
signal of a real feature pathology vs random noise.

For each (cell, segment, ticker):
  - picks = how many times the ticker appeared in the model's top-K per day
  - hits  = how many of those picks were actual positives (y_true=1)
  - hit_rate = hits / picks
  - base_rate_on_picked_days = positives among all (date, ticker) rows in the
        segment for that ticker
  - anti_score = (base_rate - hit_rate) / base_rate (positive = anti-predictive)

K is per-day variable: K = R(date) (number of positives that day) — matches
the R-precision convention.

Output: one table per segment (eval, test) for each cell, plus a
cross-cell intersect table for tickers anti-predictive in BOTH nifty50
and nifty100.

**FROZEN ONE-SHOT.** The cross-cell intersect table is committed in the
H=25 cross-market memo (#138). The script reads from sibling worktrees
under ``${WORKSPACE_ROOT}/wt-exp-nifty{50,100}-up10-25d/``. To re-run, set
``WORKSPACE_ROOT`` per per-user memory ``scratch-cache-path``; the
referenced worktrees may have been pruned, in which case the per-cell
``predictions/*.csv`` reads fail loudly with a clear FileNotFoundError.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


def per_ticker_stats(preds_csv: Path, min_picks: int = 5) -> pd.DataFrame:
    """For each ticker, compute pick count + hit rate at R-precision picks.

    Tie-breaking: (p_calibrated desc, ticker asc) with stable mergesort —
    matches the runner's convention.
    """
    df = pd.read_csv(preds_csv, parse_dates=["date"])

    # base rate per ticker (across all (date, ticker) rows in segment)
    base_rates = df.groupby("ticker")["y_true"].mean().to_dict()
    n_rows_per_ticker = df.groupby("ticker").size().to_dict()

    # per-day top-R picks
    pick_counts = {}
    hit_counts = {}
    for date, day_df in df.groupby("date"):
        r = int(day_df["y_true"].sum())
        if r == 0 or r > len(day_df):
            continue
        ordered = day_df.sort_values(
            by=["p_calibrated", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
        top_r = ordered.head(r)
        for _, row in top_r.iterrows():
            t = row["ticker"]
            pick_counts[t] = pick_counts.get(t, 0) + 1
            if row["y_true"] == 1:
                hit_counts[t] = hit_counts.get(t, 0) + 1

    rows = []
    for ticker, picks in pick_counts.items():
        hits = hit_counts.get(ticker, 0)
        hit_rate = hits / picks if picks > 0 else 0.0
        base_rate = base_rates.get(ticker, 0.0)
        anti_score = (base_rate - hit_rate) / base_rate if base_rate > 0 else 0.0
        rows.append({
            "ticker": ticker,
            "picks": picks,
            "hits": hits,
            "hit_rate": round(hit_rate, 3),
            "base_rate": round(base_rate, 3),
            "anti_score": round(anti_score, 3),
            "n_rows": n_rows_per_ticker.get(ticker, 0),
        })

    out = pd.DataFrame(rows).sort_values("picks", ascending=False)
    return out[out["picks"] >= min_picks].reset_index(drop=True)


def main() -> int:
    # WORKSPACE_ROOT = parent dir where ``wt-*/`` worktrees live; per-machine,
    # see per-user memory ``scratch-cache-path`` for the literal.
    workspace_root = os.environ.get("WORKSPACE_ROOT", "<SET-WORKSPACE_ROOT>")
    runs = {
        "nifty50":  f"{workspace_root}/wt-exp-nifty50-up10-25d/results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct",
        "nifty100": f"{workspace_root}/wt-exp-nifty100-up10-25d/results/gbdt/experiments/nifty100_up_10pct_25d_dd5pct",
    }

    by_cell_seg = {}
    for cell, run_dir in runs.items():
        for seg in ("eval", "test"):
            preds_csv = Path(run_dir) / "predictions" / f"{seg}.csv"
            if not preds_csv.exists():
                print(f"[skip] {cell}/{seg}: {preds_csv} not found")
                continue
            stats = per_ticker_stats(preds_csv, min_picks=5)
            by_cell_seg[(cell, seg)] = stats
            print(f"\n===== {cell} {seg} — top 10 picked tickers =====")
            print(stats.head(10).to_string(index=False))
            print(f"\n----- {cell} {seg} — most anti-predictive (anti_score > 0.5, picks >= 10) -----")
            anti = stats[(stats["anti_score"] > 0.5) & (stats["picks"] >= 10)].sort_values("anti_score", ascending=False)
            print(anti.to_string(index=False) if len(anti) else "(none)")

    # cross-cell intersect on TEST segment
    print("\n" + "=" * 60)
    print("CROSS-CELL: tickers anti-predictive in BOTH nifty50 + nifty100 (test segment)")
    print("=" * 60)
    n50 = by_cell_seg.get(("nifty50", "test"), pd.DataFrame())
    n100 = by_cell_seg.get(("nifty100", "test"), pd.DataFrame())
    if len(n50) and len(n100):
        n50_anti = set(n50[(n50["anti_score"] > 0.3) & (n50["picks"] >= 10)]["ticker"])
        n100_anti = set(n100[(n100["anti_score"] > 0.3) & (n100["picks"] >= 10)]["ticker"])
        common = n50_anti & n100_anti
        print(f"\nn50 anti-predictive count (anti>0.3, picks>=10): {len(n50_anti)}")
        print(f"n100 anti-predictive count (anti>0.3, picks>=10): {len(n100_anti)}")
        print(f"Common: {len(common)}")
        if common:
            n50_sub = n50[n50["ticker"].isin(common)].set_index("ticker")[["picks", "hit_rate", "base_rate", "anti_score"]]
            n100_sub = n100[n100["ticker"].isin(common)].set_index("ticker")[["picks", "hit_rate", "base_rate", "anti_score"]]
            merged = n50_sub.join(n100_sub, lsuffix="_n50", rsuffix="_n100").sort_values("picks_n50", ascending=False)
            print("\n--- Cross-cell anti-predictive table ---")
            print(merged.to_string())
    else:
        print("Missing data for one or both cells.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
