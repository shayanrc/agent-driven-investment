"""_028 sidecar: 20-variant consensus grid, CLEAN vs LEAKY, june→june, ungated.

Re-infers all 5 cells ONCE over 2025-06-01→2026-06-01 (build_scores_multi via
``consensus_june_check.infer_topk``, ~2016 warmup) then runs the alloc {20,25,33,50,100}% ×
+target/−stop {20/10,30/15,40/20,50/25} grid under both winner maps:

  * **clean** — each model OOS-masked (votes only on dates ≥ its test_end+1); breadth ramps.
  * **leaky** — all five vote every day (sp500/nasdaq score in-sample pre-OOS) — sensitivity.

Variants named V1..V20 (alloc asc, barrier asc); legacy A=V6, B=V11, C=V10, D=V7. Cross-check:
clean V6/V10 reproduce the prior A=+89.6%/C=+104.7%. ~20-min run (one shared inference pass).
Companion to ``consensus_variant_grid.py`` (the fast forward-log 122d grid + duration + box plot).

    uv run python -m scripts.backtests.consensus_grid_cleanleaky
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtests.consensus_june_check import infer_topk, winner_map, sim

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "results/backtests/_028_consensus_backtest"
START = pd.Timestamp("2025-06-01")
END = "2026-06-01"
ALLOCS = [20, 25, 33, 50, 100]
BARRIERS = [(20, 10), (30, 15), (40, 20), (50, 25)]
NAMES = {(a, t, s): f"V{1 + i*4 + j}" for i, a in enumerate(ALLOCS) for j, (t, s) in enumerate(BARRIERS)}


def main() -> None:
    preds = infer_topk(START, END)
    print(f"re-inferred {preds.date.min().date()}..{preds.date.max().date()} ({preds.date.nunique()} days)")
    rows = []
    for clean in (True, False):
        wmap, tk, _ = winner_map(preds, clean, 1)   # ungated
        for a in ALLOCS:
            for (t, s) in BARRIERS:
                r = sim(wmap, tk, a / 100, t / 100, s / 100, START, pd.Timestamp(END))
                rows.append({"version": "clean" if clean else "leaky", "name": NAMES[(a, t, s)],
                             "alloc": a, "target": t, "stop": s, "total": r["total"], "cagr": r["cagr"],
                             "sharpe": r["sharpe"], "maxdd": r["maxdd"], "win": r["win"],
                             "entries": r["entries"], "n_target": r["target"], "n_stop": r["stop"],
                             "spx": r["spx"], "n_days": r["n_days"]})
    df = pd.DataFrame(rows)
    OUTD.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTD / "consensus_variant_grid_cleanleaky.csv", index=False)
    print(f"wrote {OUTD/'consensus_variant_grid_cleanleaky.csv'} ({len(df)} rows)")
    print(df.assign(total=(df.total*100).round(1))[["version", "name", "alloc", "target", "stop", "total"]].to_string(index=False))


if __name__ == "__main__":
    main()
