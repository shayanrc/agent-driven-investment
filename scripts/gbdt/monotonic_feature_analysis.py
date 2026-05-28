"""Empirical monotonicity analysis for nifty50 H=25 — derive directional priors.

For each important feature, measure on IN-SAMPLE rows (train+val era, the final
test window excluded to avoid leakage into the prior):
  - Spearman rho(feature, target): sign = direction, |rho| = monotonic strength.
  - Decile positive-rate curve: shape (monotone vs U-shaped).
  - monotonicity_consistency: fraction of adjacent-decile steps moving in the
    dominant direction. ~1.0 => cleanly monotone; ~0.5 => non-monotone/U-shaped.

A feature is a defensible monotone-constraint candidate when |rho| is non-trivial
AND consistency is high. U-shaped features (consistency near 0.5) must stay
unconstrained — forcing monotonicity there would hurt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gbdt import data as gbdt_data
from gbdt import features as gbdt_features
from gbdt.targets import build_target

UNIVERSE = "nifty50"
DIRECTION, THRESH, HORIZON, MAXDD = "up", 10, 25, 0.05
# Exclude the trailing test window (~100 rows/ticker) from the prior so we don't
# peek at the test period when deciding constraint directions.
TEST_TAIL_ROWS = 100
TOP_N = 30  # analyze the features that actually carry importance


def main() -> None:
    panel_obj = gbdt_data.load_panel(UNIVERSE, min_rows=1600)
    panel = panel_obj.panel
    X = gbdt_features.build_feature_matrix(
        panel, panel_obj.index_series,
        annualization=panel_obj.annualization_factor,
    ).dropna(axis=1, how="all")
    y = build_target(panel, direction=DIRECTION, threshold_pct=THRESH,
                     horizon_days=HORIZON, max_drawdown=MAXDD)

    # In-sample mask: drop each ticker's trailing TEST_TAIL_ROWS + NaN targets.
    df = X.copy()
    df["_y"] = y
    df = df.dropna(subset=["_y"])
    keep_idx = []
    for tkr, g in df.groupby(level="ticker"):
        keep_idx.append(g.iloc[:-TEST_TAIL_ROWS] if len(g) > TEST_TAIL_ROWS else g)
    insample = pd.concat(keep_idx)
    yv = insample["_y"].values.astype(int)
    feats = [c for c in insample.columns if c != "_y"]
    print(f"in-sample rows={len(insample)} prevalence={yv.mean():.4f} n_feats={len(feats)}")

    # Rank features by |spearman| to find the ones worth constraining.
    rows = []
    for f in feats:
        x = insample[f].values.astype(float)
        ok = np.isfinite(x)
        if ok.sum() < 500 or np.nanstd(x[ok]) == 0:
            continue
        rho, _ = spearmanr(x[ok], yv[ok])
        if not np.isfinite(rho):
            continue
        # decile positive-rate curve
        try:
            q = pd.qcut(pd.Series(x[ok]).rank(method="first"), 10, labels=False)
        except ValueError:
            continue
        pr = pd.Series(yv[ok]).groupby(q.values).mean().values
        if len(pr) < 3:
            continue
        steps = np.diff(pr)
        dom = np.sign(rho) if rho != 0 else 1
        consistency = float(np.mean(np.sign(steps) == dom))
        spread = float(pr.max() - pr.min())
        rows.append((f, rho, consistency, spread, pr[0], pr[-1]))

    rows.sort(key=lambda r: -abs(r[1]))
    print(f"\n{'feature':<42}{'rho':>7}{'consist':>9}{'spread':>8}{'pr_d0':>7}{'pr_d9':>7}  dir")
    print("-" * 90)
    for f, rho, cons, spread, pr0, pr9 in rows[:TOP_N]:
        # Constraint suggestion: strong & consistent => +1/-1; else 0.
        if abs(rho) >= 0.04 and cons >= 0.75:
            sug = "+1" if rho > 0 else "-1"
        else:
            sug = "0 (non-mono)"
        print(f"{f:<42}{rho:>7.3f}{cons:>9.2f}{spread:>8.3f}{pr0:>7.3f}{pr9:>7.3f}  {sug}")


if __name__ == "__main__":
    main()
