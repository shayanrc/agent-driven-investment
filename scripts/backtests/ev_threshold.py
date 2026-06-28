"""Derive an expected-value **breakeven** raw-`p` entry threshold per leaderboard model (_027).

For each distinct gbdt model on the back-test leaderboard (`backtest_summary.csv`), build the
threshold-vs-precision curve on the model's **held-out eval split** (leak-free) and find the
LOWEST raw-`p` threshold τ at which a trade is positive-expected-value under the simplified
payoff model:

    a true positive  (predicted-positive that hit the +threshold% target)  earns  +target
    a false positive (predicted-positive that did NOT)                      loses  -min(max_dd, 0.20)

Per-entry  E[r] = precision·target − (1−precision)·loss ≥ 0  ⇔  precision ≥ loss/(target+loss) =: p*.
So τ = min raw-`p` cutoff whose predicted-positive set (p_raw ≥ τ) has eval precision ≥ p*
(with a ≥`min_support` guard so the precision estimate is stable). If no τ clears p*, the model
cannot be traded +EV under these payoffs → recorded as "no trade".

τ is fit ONLY on eval; the back-test then gates the **test/OOS** window at p_raw ≥ τ (the
champion's top-K=3 selection runs among survivors). This keeps threshold selection out of the
window it is scored on. The +target / −loss assumption is a *threshold-derivation* device only —
the back-test itself uses the real engine (actual price paths, horizon/DD exits, Kelly/equal).

Usage:
    uv run python -m scripts.backtests.ev_threshold [--out results/backtests/_027_ev_threshold_gate/thresholds.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "results/backtests/data/backtest_summary.csv"
EXP = ROOT / "results/gbdt/experiments"
DD_CAP = 0.20            # "max DD / 20% whichever is lower"
MIN_SUPPORT = 20        # min predicted-positives on eval for a stable precision estimate
N_GRID = 201            # quantile grid resolution for the threshold scan


def derive_one(cell_dir: Path) -> dict | None:
    spec_f, eval_f = cell_dir / "spec.yaml", cell_dir / "predictions" / "eval.csv"
    if not spec_f.exists() or not eval_f.exists():
        return None
    tgt = yaml.safe_load(spec_f.read_text())["target"]
    target = float(tgt["threshold_pct"]) / 100.0
    loss = min(float(tgt["max_drawdown"]), DD_CAP)
    p_star = loss / (target + loss)                      # breakeven precision
    e = pd.read_csv(eval_f, usecols=["p_raw", "y_true"])
    base = float(e.y_true.mean())
    tau = prec = npp = None
    for cut in np.unique(np.quantile(e.p_raw, np.linspace(0, 1, N_GRID))):
        sel = e.p_raw >= cut
        n = int(sel.sum())
        if n < MIN_SUPPORT:
            continue
        pr = float(e.y_true[sel].mean())
        if pr >= p_star:                                 # lowest τ that clears breakeven
            tau, prec, npp = float(cut), pr, n
            break
    return {"target": target, "loss": loss, "base_rate": round(base, 4),
            "p_star": round(p_star, 4),
            "tau": None if tau is None else round(tau, 4),
            "prec_at_tau": None if prec is None else round(prec, 4),
            "n_pp_eval": npp,
            "tradable": tau is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/backtests/_027_ev_threshold_gate/thresholds.csv")
    args = ap.parse_args()
    lb = pd.read_csv(LB)
    cells = sorted({Path(c.rstrip("/")).name for c in lb.model_artifact_path.dropna().unique()
                    if c.strip()})
    rows = []
    for name in cells:
        d = derive_one(EXP / name)
        if d is None:
            rows.append({"model": name, "universe": name.split("_")[0], "note": "missing spec/eval",
                         "tradable": False})
        else:
            rows.append({"model": name, "universe": name.split("_")[0], **d})
    t = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 64)
    print(t.to_string(index=False))
    print(f"\n{int(t.tradable.sum())}/{len(t)} models clear breakeven; wrote {args.out}")


if __name__ == "__main__":
    main()
