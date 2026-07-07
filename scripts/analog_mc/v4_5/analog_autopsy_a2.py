"""V4.5.2 — A2.1 analog autopsy at 10 regression + 5 win anchors.

For each fat-tail anchor, reproduce the matcher's probability layer under
A2.1 (corrwindow, n_eff=50) AND v2.4 (weighted-Euclidean). Compute:
- Top-20 analog dates + their realized 60-day forward returns.
- Temporal Herfindahl index on top-K analog years (H = Σ p²_year).
- Weighted-mean forward return (vs realized).

Tests whether temporal clustering (H > 0.4) discriminates A2.1 regressions
from wins, and would make a better gate signal than val_crps (V4.5.1).

Outputs:
- results/analog_mc/data/v4_5_2_analog_autopsy.json
- prints anchor-by-anchor summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.distances import composite_distance, distances_to_probs
from analog_mc.distances_corrwindow import corrwindow_distance
from analog_mc.features import compute_features
from analog_mc.simulate import eligible_candidates

REPO = Path(__file__).resolve().parents[3]
A2_RUN = REPO / "runs/analog_mc/20260521T061730Z"
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"
ANCHORS = REPO / "results/analog_mc/data/fat_tail_eval_anchors.json"
OUT = REPO / "results/analog_mc/data/v4_5_2_analog_autopsy.json"

TOP_K = 20


def load_fold_summaries(run_dir: Path) -> list[dict]:
    folds_dir = run_dir / "folds"
    out = []
    for d in sorted(folds_dir.iterdir(), key=lambda p: int(p.name)):
        out.append(json.loads((d / "summary.json").read_text()))
    return out


def fold_for_origin(folds: list[dict], origin_idx: int) -> dict | None:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    return None


def herfindahl(years: np.ndarray, probs: np.ndarray) -> float:
    """Σ p²_year where p_year is summed probability per calendar year."""
    df = pd.DataFrame({"year": years, "prob": probs})
    p_year = df.groupby("year")["prob"].sum().to_numpy()
    return float((p_year ** 2).sum())


def analyze_anchor(
    anchor: dict,
    section: str,
    returns_arr: np.ndarray,
    dates: pd.DatetimeIndex,
    features: pd.DataFrame,
    a2_fold: dict,
    v24_fold: dict,
    cfg_a2: Config,
    cfg_v24: Config,
    forward_logret: np.ndarray,
) -> dict:
    origin = anchor["origin_idx"]
    realized = anchor["realized_60d_return_pct"]
    candidate_idx_a2 = np.arange(0, a2_fold["train_end"] + 1, dtype=np.int64)
    candidate_idx_v24 = np.arange(0, v24_fold["train_end"] + 1, dtype=np.int64)

    # A2.1 distances + probabilities at the anchor origin.
    elig_a2 = eligible_candidates(candidate_idx_a2, features, origin, cfg_a2)
    dist_a2 = corrwindow_distance(
        returns_arr, origin, elig_a2,
        window_length=cfg_a2.corrwindow_length,
    )
    n_eff_a2 = float(a2_fold["n_eff"])
    probs_a2 = distances_to_probs(dist_a2, target_n_eff=min(n_eff_a2, elig_a2.size))

    # v2.4 distances + probabilities.
    elig_v24 = eligible_candidates(candidate_idx_v24, features, origin, cfg_v24)
    z_target = features.loc[features.index[origin], [f"zscore_{h}" for h in cfg_v24.zscore_horizons]].to_numpy()
    z_cands_v24 = features.iloc[elig_v24][[f"zscore_{h}" for h in cfg_v24.zscore_horizons]].to_numpy()
    weights_v24 = np.array(v24_fold["weights"], dtype=np.float64)
    dist_v24 = composite_distance(z_target, z_cands_v24, weights_v24)
    n_eff_v24 = float(v24_fold["n_eff"])
    probs_v24 = distances_to_probs(dist_v24, target_n_eff=min(n_eff_v24, elig_v24.size))

    # Top-K analogs.
    def topk(eligible_idx: np.ndarray, probs: np.ndarray, k: int = TOP_K) -> dict:
        order = np.argsort(probs)[::-1][:k]
        top_idx = eligible_idx[order]
        top_probs = probs[order]
        # Renormalize within top-K for the cluster metric (consistent w/ V3.5.4).
        top_probs_norm = top_probs / top_probs.sum() if top_probs.sum() > 0 else top_probs
        years = np.array([dates[i].year for i in top_idx])
        forwards = forward_logret[top_idx]  # log-return sum, may be NaN at the tail
        forward_pct = np.where(np.isnan(forwards), np.nan, np.expm1(forwards) * 100.0)
        h = herfindahl(years, top_probs_norm)
        # Weighted-mean forward return on the FULL probability layer, not just top-K.
        full_forwards = forward_logret[eligible_idx]
        valid = np.isfinite(full_forwards)
        if valid.any():
            full_p = probs.copy()
            full_p = np.where(valid, full_p, 0.0)
            if full_p.sum() > 0:
                full_p = full_p / full_p.sum()
                weighted_log_forward = float((full_p[valid] * full_forwards[valid]).sum())
                weighted_pct_forward = float(np.expm1(weighted_log_forward) * 100.0)
            else:
                weighted_pct_forward = float("nan")
        else:
            weighted_pct_forward = float("nan")
        # Years summary.
        df = pd.DataFrame({"year": years, "prob": top_probs_norm})
        year_mass = df.groupby("year")["prob"].sum().sort_values(ascending=False)
        return {
            "top_dates": [dates[i].date().isoformat() for i in top_idx],
            "top_probs": top_probs.tolist(),
            "top_forward_pct": forward_pct.tolist(),
            "herfindahl_year": h,
            "top_year_mass": {int(y): float(m) for y, m in year_mass.head(5).items()},
            "weighted_pct_forward": weighted_pct_forward,
            "n_eligible": int(eligible_idx.size),
        }

    a2_top = topk(elig_a2, probs_a2)
    v24_top = topk(elig_v24, probs_v24)

    return {
        "anchor_date": anchor["anchor_date"],
        "section": section,
        "origin_idx": origin,
        "realized_60d_return_pct": realized,
        "fold_index_a2": a2_fold["fold_index"],
        "weights_v24": list(weights_v24),
        "n_eff_a2": n_eff_a2,
        "n_eff_v24": n_eff_v24,
        "a2": a2_top,
        "v24": v24_top,
        "a2_minus_v24_herfindahl": a2_top["herfindahl_year"] - v24_top["herfindahl_year"],
    }


def main() -> None:
    a2_folds = load_fold_summaries(A2_RUN)
    v24_folds = load_fold_summaries(V24_RUN)

    cfg_a2 = Config.from_yaml(A2_RUN / "config.yaml")
    cfg_v24 = Config.from_yaml(V24_RUN / "config.yaml")

    log_ret = load_returns(cfg_v24)
    returns_arr = log_ret.to_numpy()
    dates = log_ret.index
    features = compute_features(
        log_ret, halflife=cfg_v24.ewma_halflife, horizons=cfg_v24.zscore_horizons
    )

    # Forward 60d log-return sum array, NaN where insufficient.
    n = returns_arr.size
    H = cfg_v24.forecast_horizon  # 60
    forward_logret = np.full(n, np.nan)
    for i in range(n - H):
        forward_logret[i] = returns_arr[i + 1 : i + 1 + H].sum()

    anchors_data = json.loads(ANCHORS.read_text())
    all_anchors = []
    for sec in ("positive", "negative", "regime_coverage"):
        for a in anchors_data[sec]:
            all_anchors.append((a, sec))

    rows = []
    for a, sec in all_anchors:
        origin = a["origin_idx"]
        a2_fold = fold_for_origin(a2_folds, origin)
        v24_fold = fold_for_origin(v24_folds, origin)
        if a2_fold is None or v24_fold is None:
            continue
        try:
            row = analyze_anchor(
                a, sec, returns_arr, dates, features,
                a2_fold, v24_fold, cfg_a2, cfg_v24, forward_logret,
            )
            rows.append(row)
            print(f"  {a['anchor_date']:<14} A2.H={row['a2']['herfindahl_year']:.3f}  "
                  f"v24.H={row['v24']['herfindahl_year']:.3f}  "
                  f"A2.weight_fwd={row['a2']['weighted_pct_forward']:+6.1f}%  "
                  f"realized={a['realized_60d_return_pct']:+6.1f}%")
        except Exception as e:
            print(f"  {a['anchor_date']:<14} ERROR: {e}")
            raise

    out = {
        "method": {
            "description": "V4.5.2 analog autopsy: top-K analog year-Herfindahl under A2.1 vs v2.4 at fat-tail anchors.",
            "top_k": TOP_K,
            "a2_run": str(A2_RUN.relative_to(REPO)),
            "v24_run": str(V24_RUN.relative_to(REPO)),
        },
        "anchors": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
