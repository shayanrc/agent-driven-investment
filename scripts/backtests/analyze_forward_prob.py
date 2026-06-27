"""Predicted-probability / early-calibration analysis of the `_019` forward-OOS log.

Read-only over the committed forward log (`results/backtests/data/forward_predictions_log.csv`)
and the `us_equities` price cache. Classifies every logged top-K pick by its *realized*
outcome over the cell's horizon, replicating ``gbdt.targets.build_target``'s path-honesty
triple-barrier label exactly (CLOSE-based: target = first close >= +threshold%; stop =
close <= -max_drawdown%; first-touch wins). Six buckets:

  TARGET       hit the +threshold% target barrier first (the trained label == 1)
  STOP         closed <= the -max_drawdown% floor before the target (a path-honesty loss)
  ENDED_POS    no barrier AND the full `horizon` has elapsed; end-of-horizon return > 0
  ENDED_NEG    no barrier AND the full `horizon` has elapsed; end-of-horizon return < 0
  CURRENT_POS  no barrier AND horizon NOT yet elapsed (open) — marked to last close, > entry
  CURRENT_NEG  no barrier AND horizon NOT yet elapsed (open) — marked to last close, < entry

`max_drawdown` is parsed from the cell name (``dd25pct`` -> 0.25); `threshold_pct` and
`horizon_days` come from the log. The cache ``close`` column is the same series the label
sees, so the classification is faithful by construction.

Renders (into ``results/backtests/_019_fwd_oos/figs/``):
  _019_prob_dist_overview.png            per-model p distribution / rank-decay / monthly trend
  _019_prob_outcome6_top{10,3}_box.png   p by 6-bucket outcome, box
  _019_prob_outcome6_top{10,3}_violin.png  ... violin

and prints the per-bucket p summary + the TGT-vs-STOP rank-AUC separation test.

Caveat baked into the read: with a backfill from 2026-01 and a 2026-06-27 as-of, the long
horizons are mostly UNRESOLVED (CURRENT± dominate) and the window is a strong bull — so this
is the *shape* of the predictions, not a settled calibration verdict. Re-run as horizons mature.

Usage:
    uv run python -m scripts.backtests.analyze_forward_prob [--as-of 2026-06-27]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a dep but degrade gracefully
    _HAVE_SCIPY = False

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/backtests/data/forward_predictions_log.csv"
DB = ROOT / "data/processed.db"
FIGDIR = ROOT / "results/backtests/_019_fwd_oos/figs"

ORDER = ["sp500_50", "sp500_20", "russell_50_200", "russell_40_100", "nasdaq_40_50"]
TGT = {
    "sp500_50": "+50%/50d/dd25", "sp500_20": "+20%/25d/dd10",
    "russell_50_200": "+50%/200d/dd25", "russell_40_100": "+40%/100d/dd20",
    "nasdaq_40_50": "+40%/50d/dd20",
}
BKS = ["TARGET", "STOP", "ENDED_POS", "ENDED_NEG", "CURRENT_POS", "CURRENT_NEG"]
LAB = {"TARGET": "TGT", "STOP": "STOP", "ENDED_POS": "END+", "ENDED_NEG": "END−",
       "CURRENT_POS": "CUR+", "CURRENT_NEG": "CUR−"}
BC = {"TARGET": "#2ca02c", "STOP": "#d62728", "ENDED_POS": "#98df8a",
      "ENDED_NEG": "#ff9896", "CURRENT_POS": "#c7e9c0", "CURRENT_NEG": "#fdd0a2"}
HATCH = {"CURRENT_POS": "//", "CURRENT_NEG": "//"}


def load_and_classify() -> pd.DataFrame:
    log = pd.read_csv(LOG)
    mdd = {c: int(re.search(r"dd(\d+)pct", c).group(1)) / 100 for c in log.cell.unique()}
    tk = sorted(log.ticker.unique())
    qs = ",".join("?" * len(tk))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    px = pd.read_sql(
        f"SELECT date,ticker,close FROM us_equities_data WHERE ticker IN ({qs}) AND date>='2026-01-01'",
        con, params=tk,
    )
    con.close()
    px["date"] = pd.to_datetime(px.date).dt.normalize()
    px = px.sort_values(["ticker", "date"])
    ser = {t: (g.date.values, g.close.values) for t, g in px.groupby("ticker")}

    def classify(row) -> str:
        t = row.ticker
        d = pd.Timestamp(row.snapshot_date).normalize()
        H = int(row.horizon_days)
        thr = row.threshold_pct / 100.0
        dd = mdd[row.cell]
        if t not in ser:
            return "NOFWD"
        dates, close = ser[t]
        pos = np.searchsorted(dates, d)
        if pos >= len(dates) or dates[pos] != d:
            return "NOFWD"
        c0 = close[pos]
        fwd = close[pos + 1: pos + 1 + H]
        avail = len(fwd)
        if avail == 0:
            return "NOFWD"
        ft = np.where(fwd >= c0 * (1 + thr))[0]
        fs = np.where(fwd <= c0 * (1 - dd))[0]
        ft = ft[0] if ft.size else None
        fs = fs[0] if fs.size else None
        if ft is not None and (fs is None or ft <= fs):
            return "TARGET"
        if fs is not None and (ft is None or fs < ft):
            return "STOP"
        if avail >= H:  # full horizon elapsed, neither barrier touched
            return "ENDED_POS" if fwd[-1] > c0 else "ENDED_NEG"
        return "CURRENT_POS" if fwd[-1] > c0 else "CURRENT_NEG"  # open, marked-to-last

    log["outcome"] = log.apply(classify, axis=1)
    return log


def print_summaries(log: pd.DataFrame) -> None:
    for topk in (10, 3):
        g0 = log[log["rank"] <= topk]
        print(f"\n=== p by 6-bucket outcome (top-{topk}) ===")
        print(f"{'model':15}" + "".join(f"{LAB[b]:>7}" for b in BKS))
        for m in ORDER:
            vc = g0[g0.model == m].outcome.value_counts()
            print(f"{m:15}" + "".join(f"{vc.get(b, 0):>7}" for b in BKS))
    print("\n=== TGT-vs-STOP separation: rank-AUC = P(p_TGT > p_STOP), Mann-Whitney p ===")
    for topk in (10, 3):
        print(f"  -- top-{topk} --")
        print(f"  {'model':15}{'nTGT':>6}{'nSTOP':>7}{'pTGT':>7}{'pSTOP':>7}{'AUC':>7}{'MW_p':>8}")
        g0 = log[log["rank"] <= topk]
        for m in ORDER:
            g = g0[g0.model == m]
            T = g[g.outcome == "TARGET"].p_calibrated.values
            S = g[g.outcome == "STOP"].p_calibrated.values
            if len(T) < 2 or len(S) < 2:
                print(f"  {m:15}{len(T):>6}{len(S):>7}   (insufficient)")
                continue
            if _HAVE_SCIPY:
                mw = mannwhitneyu(T, S, alternative="two-sided")
                auc, pv = mw.statistic / (len(T) * len(S)), mw.pvalue
            else:
                auc = np.mean([1.0 * (a > b) + 0.5 * (a == b) for a in T for b in S])
                pv = float("nan")
            print(f"  {m:15}{len(T):>6}{len(S):>7}{T.mean():>7.3f}{S.mean():>7.3f}{auc:>7.2f}{pv:>8.3f}")


def fig_dist_overview(log: pd.DataFrame, as_of: str) -> Path:
    df = log.copy()
    df["dt"] = pd.to_datetime(df.snapshot_date)
    col = {"sp500_50": "#1f77b4", "sp500_20": "#17becf", "russell_50_200": "#d62728",
           "russell_40_100": "#ff7f0e", "nasdaq_40_50": "#2ca02c"}
    br = {m: df[df.model == m].base_rate.iloc[0] for m in ORDER}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Predicted-probability overview — _019 forward-OOS log (top-10/day) · as of {as_of} IST",
                 fontsize=13, fontweight="bold")
    a = ax[0, 0]
    bp = a.boxplot([df[df.model == m].p_calibrated.values for m in ORDER],
                   tick_labels=[f"{m}\n{TGT[m]}" for m in ORDER], patch_artist=True,
                   showfliers=False, widths=0.6)
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(col[m]); patch.set_alpha(0.55)
    for i, m in enumerate(ORDER):
        a.plot(i + 1, br[m], "kD", ms=8, zorder=5)
    a.plot([], [], "kD", label="base rate"); a.legend(loc="upper right")
    a.set_title("A. Top-10 predicted-prob distribution (◆ = base rate)")
    a.set_ylabel("p_calibrated"); a.tick_params(axis="x", labelsize=8); a.grid(axis="y", alpha=0.3)
    b = ax[0, 1]
    df["mo"] = df.dt.dt.to_period("M").dt.to_timestamp()
    t1 = df[df["rank"] == 1]
    for m in ORDER:
        s = t1[t1.model == m].groupby("mo").p_calibrated.mean()
        b.plot(s.index, s.values, "-o", color=col[m], label=m, lw=2, ms=5)
    b.set_title("B. Monthly mean of the rank-1 pick's p"); b.set_ylabel("mean rank-1 p")
    b.legend(fontsize=8); b.grid(alpha=0.3); b.tick_params(axis="x", rotation=30)
    c = ax[1, 0]
    pv = df.pivot_table(index="rank", columns="model", values="p_calibrated", aggfunc="mean")
    for m in ORDER:
        c.plot(pv.index, pv[m], "-o", color=col[m], label=m, lw=2, ms=4)
    c.set_title("C. Conviction gradient — mean p by rank (1->10)")
    c.set_xlabel("rank"); c.set_ylabel("mean p"); c.set_xticks(range(1, 11))
    c.legend(fontsize=8); c.grid(alpha=0.3)
    d = ax[1, 1]
    x = np.arange(len(ORDER)); w = 0.38
    t1m = [t1[t1.model == m].p_calibrated.mean() for m in ORDER]
    brl = [br[m] for m in ORDER]
    d.bar(x - w / 2, t1m, w, label="mean rank-1 p", color=[col[m] for m in ORDER], alpha=0.85)
    d.bar(x + w / 2, brl, w, label="base rate", color="lightgray", edgecolor="black")
    for i, (p, r) in enumerate(zip(t1m, brl)):
        d.text(i - w / 2, p + .01, f"{p / r:.1f}x", ha="center", fontsize=8, fontweight="bold")
    d.set_title("D. Mean rank-1 p vs base rate"); d.set_xticks(x)
    d.set_xticklabels(ORDER, fontsize=8, rotation=15); d.set_ylabel("probability")
    d.legend(fontsize=8); d.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "_019_prob_dist_overview.png"
    plt.savefig(out, dpi=140); plt.close()
    return out


def fig_by_outcome(log: pd.DataFrame, topk: int, kind: str, as_of: str) -> Path:
    g0 = log[log["rank"] <= topk]
    fig, ax = plt.subplots(1, 5, figsize=(22, 6))
    tier = f"top-{topk}"
    fig.suptitle(
        f"Predicted probability by outcome ({kind}) — {tier} picks · ENDED±=full horizon (settled), "
        f"CURRENT±=open marked-to-last (hatched) · _019 forward-OOS, {as_of} IST",
        fontsize=10.5, fontweight="bold")
    nmin = 1 if kind == "box" else 2
    for i, m in enumerate(ORDER):
        a = ax[i]
        g = g0[g0.model == m]
        base = g.base_rate.iloc[0]
        present = [b for b in BKS if (g.outcome == b).sum() >= nmin]
        data = [g[g.outcome == b].p_calibrated.values for b in present]
        if kind == "box":
            bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.7,
                           medianprops=dict(color="black"))
            for patch, b in zip(bp["boxes"], present):
                patch.set_facecolor(BC[b]); patch.set_alpha(0.75)
                if b in HATCH:
                    patch.set_hatch(HATCH[b])
        else:
            parts = a.violinplot(data, showmedians=True, showextrema=False,
                                 quantiles=[[0.25, 0.75] for _ in data], widths=0.9)
            for pc, b in zip(parts["bodies"], present):
                pc.set_facecolor(BC[b]); pc.set_alpha(0.65)
                pc.set_edgecolor("black"); pc.set_linewidth(0.6)
                if b in HATCH:
                    pc.set_hatch(HATCH[b])
            parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.4)
            if "cquantiles" in parts:
                parts["cquantiles"].set_color("gray"); parts["cquantiles"].set_linestyle(":")
            for j, dd in enumerate(data, 1):
                a.scatter(np.random.default_rng(j).normal(j, 0.05, len(dd)), dd,
                          s=5, color="black", alpha=0.22, zorder=3)
        a.set_xticks(range(1, len(present) + 1))
        a.set_xticklabels([f"{LAB[b]}\nN={len(d)}" for b, d in zip(present, data)], fontsize=7.5)
        a.axhline(base, ls="--", color="navy", lw=1)
        a.text(0.98, base, f"base {base:.3f}", color="navy", fontsize=7, va="bottom",
               ha="right", transform=a.get_yaxis_transform())
        a.set_title(f"{m}\n{TGT[m]}", fontsize=9); a.set_ylabel("p_calibrated")
        a.grid(axis="y", alpha=0.3); a.set_ylim(bottom=max(0, a.get_ylim()[0]))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGDIR / f"_019_prob_outcome6_top{topk}_{kind}.png"
    plt.savefig(out, dpi=140); plt.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-06-27", help="date label for figure titles (IST)")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    log = load_and_classify()
    print_summaries(log)
    outs = [fig_dist_overview(log, args.as_of)]
    for topk in (10, 3):
        for kind in ("box", "violin"):
            outs.append(fig_by_outcome(log, topk, kind, args.as_of))
    print("\nwrote:")
    for o in outs:
        print(" ", o.relative_to(ROOT))


if __name__ == "__main__":
    main()
