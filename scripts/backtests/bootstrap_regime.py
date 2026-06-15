"""Block-bootstrap + bear-sub-window validation of the equal@K=3 champion (_015).

The `_008` rolling headline ("90-92% of windows beat the index") is computed over
HEAVILY OVERLAPPING windows (stride 5 << horizon) on only 9-22 trades, so it is NOT
90+ independent trials. This script puts an honest effective-N and a confidence band
on the edge, and checks the worst index sub-window.

Method (per champion cell, from its committed `daily_equity.csv`):
  * Daily excess log-return  d_t = log(strat_t/strat_{t-1}) - log(idx_t/idx_{t-1}).
  * MOVING-BLOCK BOOTSTRAP: resample length-L blocks (L ≈ horizon, to preserve the
    autocorrelation the overlap induces) to rebuild a series of the same length;
    statistic = mean daily excess (annualized ×252). 2000 reps → 95% CI + a one-sided
    bootstrap p (fraction of resamples with mean ≤ 0). CI excluding 0 ⇒ the edge
    survives an autocorrelation-aware significance test.
  * BEAR SUB-WINDOW: the index's worst peak-to-trough drawdown stretch on the OOS
    calendar; report strat vs index total return over exactly that stretch.

Pure post-hoc over daily_equity.csv. Determinism: a fixed integer seed is passed in
via args (no Math.random equivalent issues — numpy default_rng(seed)).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COST = Path("results/backtests/_015_validation/cost")
OUT = Path("results/backtests/_015_validation")
# (label, cost=0 run dir, horizon) — the equal@K=3 champions
CHAMPS = [
    ("sp500_50", "sp500_50_eq_c0", 50),
    ("sp500_20", "sp500_20_eq_c0", 25),
    ("ndx40", "ndx40_eq_c0", 50),
    ("r1k_50", "r1k_50_eq_c0", 200),
]
N_REP = 2000


def block_bootstrap_mean(d: np.ndarray, L: int, n_rep: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap of the mean of d (block length L)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    L = max(1, min(L, n))
    n_blocks = int(np.ceil(n / L))
    starts_pool = n - L + 1
    means = np.empty(n_rep)
    for r in range(n_rep):
        starts = rng.integers(0, starts_pool, size=n_blocks)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        means[r] = d[idx].mean()
    return means


def worst_index_drawdown_window(idx: pd.Series) -> tuple:
    """Return (peak_date, trough_date) of the index's deepest peak→trough drop."""
    arr = idx.to_numpy()
    run_max = np.maximum.accumulate(arr)
    dd = arr / run_max - 1.0
    trough = int(dd.argmin())
    peak = int(arr[: trough + 1].argmax())
    return idx.index[peak], idx.index[trough], float(dd[trough])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, rundir, H in CHAMPS:
        f = COST / rundir / "daily_equity.csv"
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        s = df["strat_equity"]; x = df["idx_equity"]
        ls = np.diff(np.log(s.to_numpy())); lx = np.diff(np.log(x.to_numpy()))
        d = ls - lx  # daily excess log-return
        ann = 252.0

        means = block_bootstrap_mean(d, L=H, n_rep=N_REP, seed=12345)
        ci = np.percentile(means, [2.5, 97.5]) * ann
        p_le0 = float((means <= 0).mean())
        # crude effective-N: trades drive the edge; report both n_days and the
        # autocorrelation-deflated effective sample (Politis block count n/L).
        n_days = len(d); eff_n = n_days / H

        peak, trough, ddmin = worst_index_drawdown_window(x)
        seg = (slice(peak, trough))
        strat_seg = s.loc[seg].iloc[-1] / s.loc[seg].iloc[0] - 1.0
        idx_seg = x.loc[seg].iloc[-1] / x.loc[seg].iloc[0] - 1.0

        rows.append({
            "cell": label, "H": H, "n_days": n_days, "eff_n_blocks": round(eff_n, 1),
            "ann_excess_mean": round(float(d.mean() * ann), 4),
            "ci95_lo": round(float(ci[0]), 4), "ci95_hi": round(float(ci[1]), 4),
            "boot_p_le0": round(p_le0, 4),
            "ci_excludes_0": bool(ci[0] > 0),
            "bear_peak": str(peak.date()), "bear_trough": str(trough.date()),
            "bear_idx_dd": round(ddmin, 4),
            "bear_strat_ret": round(float(strat_seg), 4),
            "bear_idx_ret": round(float(idx_seg), 4),
            "bear_strat_beat": bool(strat_seg > idx_seg),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "bootstrap_regime.csv", index=False)
    (OUT / "bootstrap_regime.json").write_text(json.dumps(rows, indent=2))
    print("=== block-bootstrap (annualized daily excess) + bear sub-window ===")
    cols = ["cell", "H", "n_days", "eff_n_blocks", "ann_excess_mean",
            "ci95_lo", "ci95_hi", "boot_p_le0", "ci_excludes_0",
            "bear_idx_dd", "bear_strat_ret", "bear_idx_ret", "bear_strat_beat"]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
