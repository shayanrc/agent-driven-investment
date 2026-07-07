"""V4.5.9 — Drawdown-feature sanity check.

V5.B's hypothesis is that `drawdown_60d_norm` = (close[t] - min(close[t-60:t])) /
std(returns[t-60:t]) captures the "recent extreme drawdown velocity" signature
that fails Cohort-2 anchors share. Verify by computing this feature causally
and checking whether the top-K nearest historical candidates by drawdown alone
include the expected V-recovery analogs (e.g., 1998-10-08 for 2020-03-16).

If sanity passes (V-recovery analogs appear in the top-K), V5.B's feature
design is sound. If it fails, V5.B needs a different feature definition or
should be replaced by V5.C (delay coordinates).

Outputs: results/analog_mc/data/v4_5_9_drawdown_sanity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import load_close_series

REPO = Path(__file__).resolve().parents[3]
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"
ANCHORS = REPO / "results/analog_mc/data/fat_tail_eval_anchors.json"
OUT = REPO / "results/analog_mc/data/v4_5_9_drawdown_sanity.json"

# Cohort-2 anchors from V4.5.7 (anchors where both v24 and A2.1 lift < 1).
COHORT_2 = ["2012-03-14", "2001-04-04", "2001-10-02", "2020-03-16", "2022-03-01"]
TOP_K = 20


def compute_drawdown_norm(close: pd.Series, window: int = 60) -> np.ndarray:
    """Causal: log(close[t] / peak_window) / std(log_returns_window).

    Captures "how many vol-units below the trailing peak we are." Dimensionless.
    Negative when close < peak (drawdown); zero when at peak.

    Returns ndarray same length as close, with NaN for the first `window` entries.
    """
    close_arr = close.to_numpy()
    log_ret = np.log(close_arr[1:] / close_arr[:-1])
    log_ret = np.concatenate([[np.nan], log_ret])  # align length

    n = close_arr.size
    out = np.full(n, np.nan)
    for t in range(window, n):
        w_close = close_arr[t - window + 1 : t + 1]
        w_ret = log_ret[t - window + 1 : t + 1]
        peak = w_close.max()
        ret_std = np.nanstd(w_ret, ddof=0)
        if ret_std > 0 and peak > 0:
            out[t] = float(np.log(close_arr[t] / peak)) / ret_std
    return out


def main() -> None:
    cfg = Config.from_yaml(V24_RUN / "config.yaml")
    close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)
    # Re-index to ensure causality.
    close = close.sort_index()
    dd = compute_drawdown_norm(close, window=60)

    anchors_data = json.loads(ANCHORS.read_text())
    by_date = {}
    for sec in ("positive", "negative", "regime_coverage"):
        for a in anchors_data[sec]:
            by_date[a["anchor_date"]] = a

    # Compute log returns + forward 60-day return for context.
    close_arr = close.to_numpy()
    log_ret = np.concatenate([[np.nan], np.log(close_arr[1:] / close_arr[:-1])])
    n = close_arr.size
    forward_60d_pct = np.full(n, np.nan)
    for i in range(n - 60):
        forward_60d_pct[i] = np.expm1(log_ret[i + 1 : i + 61].sum()) * 100.0

    # Map anchor date → index in `close` (and in returns, same indexing post-shift).
    date_to_idx = {d.date().isoformat(): i for i, d in enumerate(close.index)}

    rows = []
    for date in COHORT_2:
        anchor = by_date[date]
        # The anchor JSON's origin_idx is in log-return space (offset by 1 from close).
        # For drawdown comparison we use close indexing; close index = returns index + 1.
        close_idx = anchor["origin_idx"] + 1
        if close_idx >= n:
            print(f"  {date}: out of range")
            continue
        dd_target = dd[close_idx]
        if np.isnan(dd_target):
            print(f"  {date}: target drawdown NaN, skip")
            continue

        # Candidate pool: all close indices with a full forward window AND valid dd,
        # AND earlier than the anchor (causal).
        candidate_pool = np.arange(60, close_idx - 60, dtype=np.int64)
        valid = (~np.isnan(dd[candidate_pool])) & (~np.isnan(forward_60d_pct[candidate_pool - 1]))
        cands = candidate_pool[valid]
        dd_cands = dd[cands]
        fwd_cands = forward_60d_pct[cands - 1]  # forward from the corresponding log-return index

        # Top-K by drawdown similarity (|dd_target - dd_cand|).
        distances = np.abs(dd_cands - dd_target)
        order = np.argsort(distances)[:TOP_K]
        top_cands = cands[order]
        top_dates = [close.index[i].date().isoformat() for i in top_cands]
        top_dd = dd_cands[order]
        top_fwd = fwd_cands[order]

        # Cluster by year.
        years = [close.index[i].year for i in top_cands]
        year_counts = pd.Series(years).value_counts().head(5).to_dict()

        rows.append({
            "anchor_date": date,
            "realized_60d_return_pct": anchor["realized_60d_return_pct"],
            "anchor_drawdown_norm": float(dd_target),
            "top_20_dates": top_dates,
            "top_20_drawdown": top_dd.tolist(),
            "top_20_forward_60d_pct": top_fwd.tolist(),
            "top_year_counts": {int(k): int(v) for k, v in year_counts.items()},
            "mean_top20_forward_pct": float(np.nanmean(top_fwd)),
            "median_top20_forward_pct": float(np.nanmedian(top_fwd)),
            "frac_top20_same_sign": float((np.sign(top_fwd) == np.sign(anchor["realized_60d_return_pct"])).mean()),
        })

        print(f"\n=== {date} (realized {anchor['realized_60d_return_pct']:+.1f}%, dd_target {dd_target:.2f}) ===")
        print(f"  Top-5 drawdown-near analogs:")
        for j in range(5):
            print(f"    {top_dates[j]}  dd={top_dd[j]:>6.2f}  fwd={top_fwd[j]:>+6.1f}%")
        print(f"  Top-20 forward stats: mean={np.nanmean(top_fwd):+.1f}%  "
              f"median={np.nanmedian(top_fwd):+.1f}%  "
              f"frac_same_sign_as_realized={(np.sign(top_fwd) == np.sign(anchor['realized_60d_return_pct'])).mean():.0%}")
        print(f"  Top years: {year_counts}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": {
            "description": "V4.5.9 drawdown_60d_norm feature sanity: rank historical "
                          "candidates by |dd_target - dd_cand| at each Cohort-2 anchor, "
                          "inspect whether expected V-recovery analogs surface.",
            "feature": "drawdown_60d_norm = (close[t] - max(close[t-59:t+1])) / std(log_returns[t-59:t+1])",
            "window": 60,
            "top_k": TOP_K,
        },
        "anchors": rows,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
