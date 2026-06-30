"""_028 sidecar: alloc × band variant grid for the cross-model consensus strategy.

Sweeps allocation {20,25,33,50,100}% × +target/−stop {20/10,30/15,40/20,50/25} (20 variants,
named V1..V20 alloc-asc then barrier-asc; legacy A=V6, B=V11, C=V10, D=V7) on the **forward-log
window** via ``consensus_backtest.run`` (ungated, min_models=2 breadth ramp). Reports
total/CAGR/Sharpe/maxDD/win/entries + holding-duration stats per variant, and renders return
box plots grouped by band and by allocation.

No inference / feature build — reads the committed ``forward_predictions_log.csv`` (K is a pure
strategy knob applied after prediction). Companion to the clean-vs-leaky 251d grid produced by
``consensus_grid_cleanleaky.py`` (which re-infers via ``consensus_june_check``).

    uv run python -m scripts.backtests.consensus_variant_grid
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.backtests.consensus_backtest import run

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "results/backtests/_028_consensus_backtest"
ALLOCS = [20, 25, 33, 50, 100]
BARRIERS = [(20, 10), (30, 15), (40, 20), (50, 25)]
BARLAB = ["+20/−10", "+30/−15", "+40/−20", "+50/−25"]
NAMES = {(a, t, s): f"V{1 + i*4 + j}" for i, a in enumerate(ALLOCS) for j, (t, s) in enumerate(BARRIERS)}


def _durations(r) -> dict:
    """Match BUY→exit per ticker on the trade log; return holding-duration stats (trading days)."""
    tr = r["_tr"]; cal = list(r["_eq"].index); pos = {d: i for i, d in enumerate(cal)}
    oe, durs, dt, ds = {}, [], [], []
    for row in tr.itertuples():
        if row.action == "BUY":
            oe[row.ticker] = row.date
        elif row.action in ("target", "stop"):
            ein = oe.pop(row.ticker, None)
            if ein is not None and ein in pos and row.date in pos:
                d = pos[row.date] - pos[ein]; durs.append(d)
                (dt if row.action == "target" else ds).append(d)
    opn = [len(cal) - 1 - pos[ein] for ein in oe.values() if ein in pos]
    return {"mean_dur": np.mean(durs) if durs else np.nan, "med_dur": np.median(durs) if durs else np.nan,
            "max_dur": max(durs + opn) if (durs or opn) else np.nan, "n_open": len(oe),
            "mean_tgt_dur": np.mean(dt) if dt else np.nan, "mean_stop_dur": np.mean(ds) if ds else np.nan}


def main() -> None:
    rows = []
    for a in ALLOCS:
        for (t, s) in BARRIERS:
            r = run("consensus", a / 100, t / 100, s / 100, min_models=2)
            rows.append({"name": NAMES[(a, t, s)], "alloc": a, "target": t, "stop": s,
                         "total": r["total_return"], "cagr": r["cagr"], "sharpe": r["sharpe"],
                         "maxdd": r["max_dd"], "win": r["win_rate"], "entries": r["n_entries"],
                         "n_target": r["n_target"], "n_stop": r["n_stop"], "n_names": r["n_names"],
                         "window": r["window"], "n_days": r["n_days"], "spx": r["spx_return"],
                         **_durations(r)})
    df = pd.DataFrame(rows)
    OUTD.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTD / "consensus_variant_grid_fwdlog122d.csv", index=False)
    spx = float(df.spx.iloc[0]) * 100

    # Return box plots: by band (5 alloc points each) and by allocation (4 band points each).
    np.random.seed(0)
    bands = {f"+{t}/−{s}": [df[(df.alloc == a) & (df.target == t)].total.iloc[0] * 100 for a in ALLOCS]
             for (t, s) in BARRIERS}
    allocs = {a: [df[(df.alloc == a) & (df.target == t)].total.iloc[0] * 100 for (t, s) in BARRIERS]
              for a in ALLOCS}
    ac = {a: c for a, c in zip(ALLOCS, plt.cm.viridis(np.linspace(0, 0.85, len(ALLOCS))))}
    bc = {l: c for l, c in zip(BARLAB, plt.cm.plasma(np.linspace(0, 0.8, len(BARLAB))))}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True)
    bx = ax1.boxplot([bands[l] for l in BARLAB], positions=range(1, 5), widths=0.55, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for p in bx["boxes"]:
        p.set(facecolor="#cfe8ff", alpha=.7)
    for x, l in enumerate(BARLAB, 1):
        for a, v in zip(ALLOCS, bands[l]):
            ax1.scatter(x + np.random.uniform(-.13, .13), v, color=ac[a], s=55, zorder=3, edgecolor="k", lw=.5)
    ax1.set_xticks(range(1, 5)); ax1.set_xticklabels(BARLAB)
    ax1.axhline(spx, color="red", ls="--", lw=1.2); ax1.set_ylabel("total return %")
    ax1.set_title("Total return by BAND (target/stop)", weight="bold"); ax1.set_xlabel("band  (tighter → wider)")
    ax1.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=ac[a], markeredgecolor='k',
                               label=f"{a}%", ms=9) for a in ALLOCS], title="allocation", fontsize=8, loc="upper center")
    bx2 = ax2.boxplot([allocs[a] for a in ALLOCS], positions=range(1, 6), widths=0.55, patch_artist=True,
                      medianprops=dict(color="black", lw=2))
    for p in bx2["boxes"]:
        p.set(facecolor="#d6f5d6", alpha=.7)
    for x, a in enumerate(ALLOCS, 1):
        for l, v in zip(BARLAB, allocs[a]):
            ax2.scatter(x + np.random.uniform(-.13, .13), v, color=bc[l], s=55, zorder=3, edgecolor="k", lw=.5)
    ax2.set_xticks(range(1, 6)); ax2.set_xticklabels([f"{a}%" for a in ALLOCS])
    ax2.axhline(spx, color="red", ls="--", lw=1.2); ax2.set_title("Total return by ALLOCATION", weight="bold")
    ax2.set_xlabel("allocation  (smaller → larger)")
    ax2.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=bc[l], markeredgecolor='k',
                               label=l, ms=9) for l in BARLAB], title="band", fontsize=8, loc="upper left")
    fig.suptitle(f"Consensus variant total return — forward log {df.window.iloc[0]} "
                 f"({int(df.n_days.iloc[0])}d, ungated, SPX {spx:+.1f}%)", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    (OUTD / "figs").mkdir(exist_ok=True)
    fig.savefig(OUTD / "figs" / "_028_variant_return_boxplot.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTD / "figs" / "_028_variant_return_boxplot.svg", bbox_inches="tight")
    print(f"wrote {OUTD/'consensus_variant_grid_fwdlog122d.csv'} + box plot ({df.window.iloc[0]}, {int(df.n_days.iloc[0])}d)")
    print(df[["name", "alloc", "target", "stop", "total", "sharpe", "maxdd", "mean_dur", "entries"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
