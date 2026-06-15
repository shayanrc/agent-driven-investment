"""NSE signal forensics (_011).

`_010` refuted the regime explanation for the US–NSE gap and redirected to the
real question: **why doesn't NSE's high label-AUC convert to tradeable excess?**
Two non-exclusive hypotheses:

  A. NOT REAL / NON-GENERALIZING — the test-split AUC is overfit; on genuinely
     fresh OOS the model's discrimination collapses toward 0.5.
  B. REAL BUT UNCAPTURABLE — the AUC holds OOS, but the ranked names' big moves
     happen as overnight jump-gaps the next-open fill can't capture.

This script tests both, with US cells (sp500_50, r1k_50) as positive controls.

Forensic A — fresh-OOS discrimination:
  Recompute the realized triple-barrier label (gbdt.targets.build_target, the exact
  training label) on the fresh OOS, merge with the cell's fresh predictions, and
  compute AUC + base-rate + R-Precision@K on the fresh set. Compare to the published
  test-split AUC. AUC holds → hypothesis A is out, look to B. AUC collapses → A.

Forensic B — capturability:
  The strategy enters at the NEXT open after a signal. The overnight gap
  (next_open/signal_close − 1) is return the strategy CANNOT capture. For the top-K
  daily picks, compare the entry-gap distribution (and the gap among realized winners)
  US vs NSE. Large positive NSE entry gaps ⇒ the move front-runs the fill ⇒ B.

Pure post-hoc over committed specs + the fresh-prediction CSVs. No new back-tests.
R-Precision@K uses the project denominator min(K, R_q), macro-averaged over days
with R_q>0 (see [[project-r-precision-methodology]]).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from gbdt.data import load_panel
from gbdt.targets import build_target

K = 3
OUT = Path("results/backtests/_011_forensics")

# (label, cell_dir, fresh_csv, market, test_auc)
CELLS = [
    ("nifty_50_50d", "nifty500_up_50pct_50d_dd25pct_aligned", "_009_nifty/n500_50_50d_fresh.csv", "NSE", 0.889),
    ("nifty_50_25d", "nifty500_up_50pct_25d_dd25pct_aligned", "_009_nifty/n500_50_25d_fresh.csv", "NSE", 0.827),
    ("nifty_30_25d", "nifty500_up_30pct_25d_dd15pct_aligned", "_009_nifty/n500_30_25d_fresh.csv", "NSE", 0.814),
    ("nifty_20_25d", "nifty500_up_20pct_25d_dd10pct_aligned", "_009_nifty/n500_20_25d_fresh.csv", "NSE", 0.722),
    ("nifty_10_25d", "nifty500_up_10pct_25d_dd5pct_aligned", "_009_nifty/n500_10_25d_fresh.csv", "NSE", 0.601),
    ("sp500_50",     "sp500_up_50pct_50d_dd25pct_agentloop", "_008_roll/sp500_50_fresh.csv", "US", 0.899),
    ("r1k_50",       "russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1", "_008_roll/r1k_50_fresh.csv", "US", 0.726),
]
EXP = Path("results/gbdt/experiments")
RES = Path("results/backtests")


def r_precision_at_k(df: pd.DataFrame, k: int) -> float:
    """Macro-avg over days: (1/Q) Σ_q hits_in_topk_q / min(k, R_q), days with R_q>0."""
    vals = []
    for _, g in df.groupby("date"):
        R = int(g["y"].sum())
        if R == 0:
            continue
        top = g.nlargest(k, "p")
        vals.append(top["y"].sum() / min(k, R))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # group cells by universe so each panel loads once
    by_uni: dict[str, list] = {}
    specs = {}
    for label, cd, fresh, mkt, auc in CELLS:
        spec = yaml.safe_load((EXP / cd / "spec.yaml").read_text())["target"]
        specs[label] = (cd, fresh, mkt, auc, spec)
        by_uni.setdefault(spec["universe"], []).append(label)

    panels = {}
    rowsA, rowsB = [], []
    for uni, labels in by_uni.items():
        if uni not in panels:
            p = load_panel(uni, start="2019-01-01", end="2026-06-15", cache_only=True)
            panels[uni] = p.panel
        panel = panels[uni]
        # next-day open per ticker (for the entry-gap / capturability forensic)
        no = panel[["open"]].copy()
        no["next_open"] = no.groupby(level="ticker")["open"].shift(-1)
        next_open = no["next_open"]

        for label in labels:
            cd, fresh, mkt, auc, spec = specs[label]
            y = build_target(panel, direction=spec["direction"],
                             threshold_pct=spec["threshold_pct"],
                             horizon_days=spec["horizon_days"],
                             max_drawdown=spec.get("max_drawdown"))
            fp = pd.read_csv(RES / fresh, parse_dates=["date"])
            fp["key"] = list(zip(fp["date"], fp["ticker"]))
            ymap = y.dropna()
            ykeys = {(d, t): v for (d, t), v in ymap.items()}
            fp["y"] = fp["key"].map(ykeys)
            lab = fp.dropna(subset=["y"]).copy()
            lab["y"] = lab["y"].astype(int)
            lab["p"] = lab["p_raw"]  # ranking score for R-precision / top-K

            # --- Forensic A: fresh-OOS discrimination ---
            n = len(lab); base = lab["y"].mean()
            fresh_auc = (roc_auc_score(lab["y"], lab["p_raw"])
                         if lab["y"].nunique() == 2 else float("nan"))
            rp = {k: r_precision_at_k(lab, k) for k in (1, 3, 5, 10, 20)}
            rowsA.append({"cell": label, "market": mkt, "test_auc": auc,
                          "fresh_auc": round(fresh_auc, 3), "auc_drop": round(auc - fresh_auc, 3),
                          "n_labeled": n, "fresh_base": round(base, 4),
                          "rp@1": round(rp[1], 3), "rp@3": round(rp[3], 3),
                          "rp@5": round(rp[5], 3), "rp@10": round(rp[10], 3),
                          "rp1_lift": round(rp[1] / base, 2) if base else float("nan")})

            # --- Forensic B: entry-gap capturability for top-K daily picks ---
            lab["close"] = [panel.loc[(d, t), "close"] if (d, t) in panel.index else np.nan
                            for d, t in lab["key"]]
            lab["next_open"] = [next_open.get((d, t), np.nan) for d, t in lab["key"]]
            lab["entry_gap"] = lab["next_open"] / lab["close"] - 1.0
            picks = (lab.sort_values(["date", "p"], ascending=[True, False])
                        .groupby("date").head(K).dropna(subset=["entry_gap"]))
            win = picks[picks["y"] == 1]
            rowsB.append({"cell": label, "market": mkt, "n_picks": len(picks),
                          "median_entry_gap": round(picks["entry_gap"].median(), 4),
                          "p90_entry_gap": round(picks["entry_gap"].quantile(0.9), 4),
                          "frac_gap_up_gt2pct": round((picks["entry_gap"] > 0.02).mean(), 3),
                          "n_winners": len(win),
                          "winner_median_entry_gap": round(win["entry_gap"].median(), 4) if len(win) else float("nan")})

    A = pd.DataFrame(rowsA); B = pd.DataFrame(rowsB)
    A.to_csv(OUT / "forensic_a_fresh_auc.csv", index=False)
    B.to_csv(OUT / "forensic_b_capturability.csv", index=False)
    print("=== Forensic A: fresh-OOS AUC vs test AUC ===")
    print(A.to_string(index=False))
    print("\n=== Forensic B: entry-gap capturability (top-K daily picks) ===")
    print(B.to_string(index=False))
    # pooled medians by market for B
    print("\n=== B pooled by market (median across cells) ===")
    print(B.groupby("market")[["median_entry_gap", "frac_gap_up_gt2pct", "winner_median_entry_gap"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
