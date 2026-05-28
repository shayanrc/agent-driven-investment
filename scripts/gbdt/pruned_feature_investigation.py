"""Do the PRUNED features (importance<0.01) actually relate to the target, and
were they pruned because they're noise or because they're REDUNDANT?

Uses the cached in-sample matrix (no rebuild). For each pruned feature:
  - Spearman rho(feature, target) + decile-consistency  -> does it show up in data?
  - max |corr| with any KEPT (importance>=0.01) feature   -> is it redundant?

Heuristic expected direction per family is annotated so we can compare
"should it be monotone?" against "is it monotone in the data?".
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr

from gbdt import data as gbdt_data
from gbdt.targets import build_target

CACHE = "results/gbdt/experiments/_nifty50_insample_matrix.parquet"
BEFORE = "/mnt/122CEE982CEE765F/Workspace/wt-exp-nifty50-up10-25d/results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct/model.cbm"

# Logic/heuristic: should this family be monotone in target, and which sign?
# (+1 expect increasing, -1 expect decreasing, 0 = no clean prior / non-monotone expected)
FAMILY_PRIOR = {
    "stock_return": ("+/-", "momentum(+) vs mean-reversion(-)"),
    "rel_strength": ("+", "relative momentum"),
    "realized_vol_zscore": ("+", "vol expansion (same logic as vol estimators)"),
    "vol_xs_zscore": ("+", "cross-sectional vol rank"),
    "vol_xs_rank": ("+", "cross-sectional vol rank"),
    "dollar_move_zscore": ("+", "F16 signed recent-move magnitude"),
    "stock_return_zscore": ("+", "F16 signed recent-return"),
    "return_xs_zscore": ("+", "cross-sectional return rank (momentum)"),
    "return_xs_rank": ("+", "cross-sectional return rank (momentum)"),
    "dollar_move_xs": ("+", "cross-sectional dollar move"),
    "dollar_move_rank": ("+", "cross-sectional dollar move"),
    "volume_ratio": ("+", "volume expansion precedes moves"),
    "obv": ("+", "on-balance-volume accumulation"),
    "vol_ret_corr": ("0", "vol-return correlation; ambiguous"),
    "returns_skew": ("0", "higher moment; ambiguous"),
    "returns_kurt": ("0", "higher moment; ambiguous"),
    "beta": ("0", "market beta; ambiguous"),
    "vol_of_vol": ("0", "non-monotone (confirmed)"),
    "sma_distance": ("0", "distance from MA; mean-rev vs momentum"),
    "vol_pct": ("+", "vol percentile"),
    "vol_change": ("+", "vol acceleration"),
    "drawdown": ("0", "path drawdown; ambiguous"),
    "runup": ("+", "path runup (momentum)"),
    "index": ("0", "index-context; mixed"),
}
CALENDAR = ("moy_", "dom_", "dow_", "fomc_", "budget_", "fiscal_")


def fam(f: str) -> str:
    for k in sorted(FAMILY_PRIOR, key=len, reverse=True):
        if f.startswith(k):
            return k
    return re.sub(r"_\d+.*$", "", f)


def main() -> None:
    X = pd.read_parquet(CACHE)
    po = gbdt_data.load_panel("nifty50", min_rows=1600)
    y = build_target(po.panel, direction="up", threshold_pct=10, horizon_days=25, max_drawdown=0.05)
    y = y.reindex(X.index)
    ok = y.notna().values
    Xv = X[ok]; yv = y[ok].astype(int).values

    m = CatBoostClassifier(); m.load_model(BEFORE)
    names = list(m.feature_names_)
    imp = dict(zip(names, m.get_feature_importance()))
    pruned = [f for f in X.columns if imp.get(f, 0.0) < 0.01]
    kept = [f for f in X.columns if imp.get(f, 0.0) >= 0.01]
    print(f"pruned={len(pruned)} kept={len(kept)}  in-sample rows={len(Xv)} prevalence={yv.mean():.3f}\n")

    Xkept = Xv[kept].fillna(Xv[kept].median())
    rows = []
    for f in pruned:
        x = Xv[f].values.astype(float)
        m_ok = np.isfinite(x)
        if m_ok.sum() < 500 or np.nanstd(x[m_ok]) == 0:
            continue
        rho, _ = spearmanr(x[m_ok], yv[m_ok])
        if not np.isfinite(rho):
            continue
        try:
            q = pd.qcut(pd.Series(x[m_ok]).rank(method="first"), 10, labels=False)
            pr = pd.Series(yv[m_ok]).groupby(q.values).mean().values
            steps = np.diff(pr)
            cons = float(np.mean(np.sign(steps) == (np.sign(rho) or 1)))
        except ValueError:
            cons = np.nan
        # redundancy: max |spearman corr| with any KEPT feature
        xs = pd.Series(x).rank()
        maxc = 0.0
        kc = Xkept.corrwith(pd.Series(x, index=Xv.index), method="spearman").abs()
        maxc = float(kc.max()) if len(kc) else 0.0
        rows.append((f, fam(f), rho, cons, maxc))

    df = pd.DataFrame(rows, columns=["feat", "family", "rho", "cons", "maxcorr_kept"])

    # Per-family rollup
    print("=== PRUNED features by family: relationship-with-target + redundancy ===")
    print(f"{'family':<22}{'n':>4}{'mean|rho|':>10}{'n mono*':>8}{'mean maxcorr_kept':>18}  prior")
    print("-" * 86)
    for famname, g in sorted(df.groupby("family"), key=lambda kv: -kv[1]["rho"].abs().mean()):
        n_mono = int(((g["rho"].abs() >= 0.04) & (g["cons"] >= 0.75)).sum())
        prior = FAMILY_PRIOR.get(famname, ("?", ""))[0]
        print(f"{famname:<22}{len(g):>4}{g['rho'].abs().mean():>10.3f}{n_mono:>8}{g['maxcorr_kept'].mean():>18.2f}  {prior}")
    print("\n* n mono = pruned features with |rho|>=0.04 AND decile-consistency>=0.75 (a real monotone marginal relationship)")

    # Headline counts
    real_mono = df[(df["rho"].abs() >= 0.04) & (df["cons"] >= 0.75)]
    redundant = real_mono[real_mono["maxcorr_kept"] >= 0.7]
    print(f"\nPRUNED features with a REAL monotone marginal relationship: {len(real_mono)}/{len(df)}")
    print(f"  ...of those, REDUNDANT (max corr w/ a kept feature >= 0.7): {len(redundant)}/{len(real_mono)}")
    print(f"  ...genuinely weak/noise (|rho|<0.04 or non-monotone): {len(df)-len(real_mono)}/{len(df)}")
    # The strongest pruned-but-related features
    print("\nTop 12 pruned features by |rho| (related to target despite ~0 importance):")
    for _, r in real_mono.sort_values('rho', key=lambda s: s.abs(), ascending=False).head(12).iterrows():
        print(f"  {r['feat']:<40} rho={r['rho']:+.3f} cons={r['cons']:.2f} maxcorr_kept={r['maxcorr_kept']:.2f}")


if __name__ == "__main__":
    main()
