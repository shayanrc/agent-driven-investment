"""Regime-conditioning analysis (_010).

The `_009` open question: is the rank/equal rolling edge gated by the market
regime (the index's own return over each window) rather than by the universe or
the cell's AUC? Every rolling_windows.csv already records ``idx_ret`` (the
benchmark return over that exact window) alongside ``strat_ret`` and ``excess``,
so we can label each window by regime and test the dependence *within* each cell
(horizon fixed → no horizon/regime conflation) and pooled by universe.

Hypotheses under test:
  H1 (regime-gated)   : excess rises with idx_ret — the strategy is effectively
                        long-beta and only beats a rising market.
  H2 (regime-neutral) : excess is independent of idx_ret — the edge is real
                        ranking skill that survives down tapes.

If H1 holds *within both* the US and NSE cells, the `_008`-vs-`_009` "US wins /
NSE loses" split collapses to a single regime axis (US sampled bull windows, NSE
sampled a bear one). If US down-windows still show positive excess while NSE
down-windows are negative, regime is only part of the story.

No new back-tests: pure post-hoc over the committed rolling_windows.csv files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# (memo_dir, cell_subdir, label, universe, market, AUC, horizon)
CELLS = [
    ("_006_rolling", "cell5_revalreg", "ndx_cell5",      "nasdaq100", "US", 0.52, 50),
    ("_006_rolling", "b_acceptance",   "ndx_baccept",    "nasdaq100", "US", 0.47, 50),
    ("_007_mix_fresh", "ndx40_rolling","ndx40_mix",      "nasdaq100", "US", 0.74, 50),
    ("_008_roll", "sp500_50_rolling",  "sp500_50",       "sp500",     "US", 0.90, 50),
    ("_008_roll", "sp500_20_rolling",  "sp500_20",       "sp500",     "US", 0.76, 25),
    ("_008_roll", "r1k_50_rolling",    "r1k_50",         "russell1000","US",0.73, 200),
    ("_009_nifty", "n500_50_50d_rolling","nifty_50_50d", "nifty500",  "NSE",0.89, 50),
    ("_009_nifty", "n500_50_25d_rolling","nifty_50_25d", "nifty500",  "NSE",0.83, 25),
    ("_009_nifty", "n500_30_25d_rolling","nifty_30_25d", "nifty500",  "NSE",0.81, 25),
    ("_009_nifty", "n500_20_25d_rolling","nifty_20_25d", "nifty500",  "NSE",0.72, 25),
    ("_009_nifty", "n500_10_25d_rolling","nifty_10_25d", "nifty500",  "NSE",0.60, 25),
]
ROOT = Path("results/backtests")
OUT = ROOT / "_010_regime"


def load_windows() -> pd.DataFrame:
    rows = []
    for memo, sub, label, uni, mkt, auc, H in CELLS:
        f = ROOT / memo / sub / "rolling_windows.csv"
        w = pd.read_csv(f, parse_dates=["origin", "end"])
        w["cell"] = label; w["universe"] = uni; w["market"] = mkt
        w["AUC"] = auc; w["H"] = H
        rows.append(w)
    return pd.concat(rows, ignore_index=True)


def per_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Within-cell (H fixed): Spearman(excess, idx_ret) + up/down-regime split."""
    out = []
    for label, g in df.groupby("cell", sort=False):
        meta = g.iloc[0]
        rho, p = stats.spearmanr(g["idx_ret"], g["excess"])
        up = g[g["idx_ret"] > 0]; dn = g[g["idx_ret"] <= 0]
        out.append({
            "cell": label, "market": meta["market"], "universe": meta["universe"],
            "AUC": meta["AUC"], "H": meta["H"], "n": len(g),
            "spearman_excess_vs_idx": round(rho, 3), "p": round(p, 3),
            "n_up": len(up), "med_excess_up": round(up["excess"].median(), 4) if len(up) else np.nan,
            "winrate_up": round((up["excess"] > 0).mean(), 3) if len(up) else np.nan,
            "n_dn": len(dn), "med_excess_dn": round(dn["excess"].median(), 4) if len(dn) else np.nan,
            "winrate_dn": round((dn["excess"] > 0).mean(), 3) if len(dn) else np.nan,
        })
    return pd.DataFrame(out)


def pooled_regime_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled by market × regime-sign — does US keep an edge in down windows?"""
    df = df.copy()
    df["regime"] = np.where(df["idx_ret"] > 0, "up", "down")
    out = []
    for (mkt, reg), g in df.groupby(["market", "regime"]):
        out.append({"market": mkt, "regime": reg, "n_windows": len(g),
                    "med_idx_ret": round(g["idx_ret"].median(), 4),
                    "med_strat_ret": round(g["strat_ret"].median(), 4),
                    "med_excess": round(g["excess"].median(), 4),
                    "winrate_excess>0": round((g["excess"] > 0).mean(), 3)})
    return pd.DataFrame(out).sort_values(["market", "regime"])


def make_figure(df: pd.DataFrame, path: Path) -> None:
    """Excess vs idx_ret, faceted by horizon (H fixed per panel → comparable)."""
    Hs = sorted(df["H"].unique())
    fig, axes = plt.subplots(1, len(Hs), figsize=(5 * len(Hs), 4.5), squeeze=False)
    for ax, H in zip(axes[0], Hs):
        sub = df[df["H"] == H]
        for mkt, color in [("US", "#1f77b4"), ("NSE", "#ff7f0e")]:
            s = sub[sub["market"] == mkt]
            ax.scatter(s["idx_ret"] * 100, s["excess"] * 100, s=22, alpha=0.6,
                       c=color, label=mkt, edgecolors="none")
        ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
        if len(sub) > 2:
            rho, _ = stats.spearmanr(sub["idx_ret"], sub["excess"])
            ax.set_title(f"H={H}d  (Spearman ρ={rho:+.2f})", fontsize=11)
        ax.set_xlabel("index return over window (%)")
        ax.set_ylabel("strategy excess return (%)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.suptitle("Regime-conditioning: does rank/equal excess depend on the market's own return?",
                 fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_windows()
    pc = per_cell_stats(df)
    pool = pooled_regime_table(df)
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure(df, OUT / "figs" / "regime_scatter.png")
    pc.to_csv(OUT / "per_cell_regime.csv", index=False)
    pool.to_csv(OUT / "pooled_regime.csv", index=False)

    # overall pooled Spearman within each market (standardize idx_ret within cell
    # so horizons are comparable when pooling)
    d = df.copy()
    d["idx_z"] = d.groupby("cell")["idx_ret"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    d["exc_z"] = d.groupby("cell")["excess"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    summary = {}
    for mkt, g in d.groupby("market"):
        rho, p = stats.spearmanr(g["idx_z"], g["exc_z"])
        summary[mkt] = {"n_windows": int(len(g)), "n_cells": int(g["cell"].nunique()),
                        "pooled_spearman_excessZ_vs_idxZ": round(rho, 3), "p": round(p, 4)}
    rho_all, p_all = stats.spearmanr(d["idx_z"], d["exc_z"])
    summary["ALL"] = {"n_windows": int(len(d)), "n_cells": int(d["cell"].nunique()),
                      "pooled_spearman_excessZ_vs_idxZ": round(rho_all, 3), "p": round(p_all, 6)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=== per-cell: Spearman(excess, idx_ret) + up/down split ===")
    print(pc.to_string(index=False))
    print("\n=== pooled by market × regime ===")
    print(pool.to_string(index=False))
    print("\n=== pooled within-cell-standardized Spearman ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
