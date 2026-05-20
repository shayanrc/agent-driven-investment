"""Select the canonical fat-tail evaluation anchors for v4+ experiments.

Selection criteria:
1. Compute classical-scale z50 = (rolling 50-day mean / std) * sqrt(50).
2. Filter to anchors inside the canonical run's walk-forward test windows
   (so the cached forecast is available without re-running inference).
3. Positive side: take all |z| > 3 anchors, cluster-pick by 120-trading-day
   min-gap (keep the most-extreme per cluster).
4. Negative side: no anchors reach -3 in the data; take top 3 most-extreme
   below -2, same 120-day clustering.

Writes results/analog_mc/data/fat_tail_eval_anchors.json — the canonical
list to drive FAT_TAIL_EVAL.md panel rendering and every v4 experiment
deliverable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_close_series, load_returns
from analog_mc.features import compute_features


CANONICAL_RUN = "runs/analog_mc/20260520T045525Z"
OUT_JSON = "results/analog_mc/data/fat_tail_eval_anchors.json"


def cluster_pick(indices, scores, min_gap: int):
    """Greedy: rank by |score| desc, accept if no prior pick within min_gap."""
    picks = []
    for i in sorted(indices, key=lambda j: -abs(scores[j])):
        if all(abs(i - p) >= min_gap for p in picks):
            picks.append(i)
    return sorted(picks)


def main() -> None:
    run_dir = Path(CANONICAL_RUN)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)
    returns_s = load_returns(cfg)

    features = compute_features(
        returns_s,
        halflife=cfg.ewma_halflife,
        horizons=cfg.zscore_horizons,
        momentum_lookback=cfg.momentum_lookback,
    )
    z50 = features["zscore_50"].to_numpy() * np.sqrt(50)  # to classical scale

    all_origins = set()
    for fd in sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name)):
        s = json.loads((fd / "summary.json").read_text())
        all_origins.update(range(s["test_start"], s["test_end"] + 1))

    pos_mask = np.isfinite(z50) & (z50 > 3.0)
    pos_idx = [i for i in np.where(pos_mask)[0] if i in all_origins]
    pos_picks = cluster_pick(pos_idx, z50, min_gap=120)

    neg_mask = np.isfinite(z50) & (z50 < -2.0)
    neg_idx = [i for i in np.where(neg_mask)[0] if i in all_origins]
    neg_picks_all = cluster_pick(neg_idx, z50, min_gap=120)
    neg_picks = sorted(neg_picks_all, key=lambda i: z50[i])[:3]
    neg_picks = sorted(neg_picks)

    def row(i):
        return {
            "origin_idx": int(i),
            "anchor_date": str(close.index[i + 1].date()),
            "z50_classical": float(z50[i]),
            "anchor_close": float(close.iloc[i + 1]),
            "realized_60d_close": float(close.iloc[i + 60]) if i + 60 < len(close) else None,
            "realized_60d_return_pct": (
                float((close.iloc[i + 60] / close.iloc[i + 1] - 1) * 100)
                if i + 60 < len(close) else None
            ),
        }

    anchors = {
        "selection": {
            "z50_scale": "classical (rolling_mean / rolling_std * sqrt(50))",
            "positive_threshold": 3.0,
            "negative_threshold": -2.0,
            "negative_n_kept": 3,
            "cluster_min_gap_days": 120,
            "canonical_run": CANONICAL_RUN,
        },
        "positive": [row(i) for i in pos_picks],
        "negative": [row(i) for i in neg_picks],
    }

    out = Path(OUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(anchors, indent=2))
    print(f"wrote {out}")
    print(f"  {len(pos_picks)} positive anchors, {len(neg_picks)} negative")
    for side in ("positive", "negative"):
        for a in anchors[side]:
            print(f"  {side[:3]}  {a['anchor_date']}  z={a['z50_classical']:+5.2f}  "
                  f"close={a['anchor_close']:9,.1f}  60d={a['realized_60d_return_pct']:+6.2f}%")


if __name__ == "__main__":
    main()
