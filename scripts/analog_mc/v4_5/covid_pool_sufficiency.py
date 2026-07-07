"""V4.5.4 — COVID 2020-03-16 pool sufficiency.

V3.5.2 already established the pool at the 2020-03-16 fold contains 179
candidates with realized 60d > +30%, so the pool is not structurally empty.
The question now: are those candidates being selected by any v4 matcher?

For each matcher (v2.4 weighted-Euclidean, A2.1 corrwindow), compute the
total probability mass assigned to candidates whose realized 60d return >
+30%, and report top-10 high-return candidates with their assigned probs.

Outputs: results/analog_mc/data/v4_5_4_covid_pool.json
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
from analog_mc.local_linear import forward_logret_sums
from analog_mc.simulate import eligible_candidates

REPO = Path(__file__).resolve().parents[3]
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"
A2_RUN = REPO / "runs/analog_mc/20260521T061730Z"
OUT = REPO / "results/analog_mc/data/v4_5_4_covid_pool.json"

COVID_ORIGIN_IDX = 8621  # 2020-03-16
COVID_REALIZED_60D_PCT = 43.79


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


def main() -> None:
    cfg_v24 = Config.from_yaml(V24_RUN / "config.yaml")
    cfg_a2 = Config.from_yaml(A2_RUN / "config.yaml")

    log_ret = load_returns(cfg_v24)
    returns_arr = log_ret.to_numpy()
    features = compute_features(log_ret, halflife=cfg_v24.ewma_halflife,
                                 horizons=cfg_v24.zscore_horizons)
    forward_logret = forward_logret_sums(returns_arr, cfg_v24.forecast_horizon)
    forward_pct = np.where(np.isnan(forward_logret), np.nan, np.expm1(forward_logret) * 100.0)

    v24_folds = load_fold_summaries(V24_RUN)
    a2_folds = load_fold_summaries(A2_RUN)
    v24_fold = fold_for_origin(v24_folds, COVID_ORIGIN_IDX)
    a2_fold = fold_for_origin(a2_folds, COVID_ORIGIN_IDX)
    assert v24_fold is not None and a2_fold is not None

    out: dict = {
        "anchor_date": "2020-03-16",
        "origin_idx": COVID_ORIGIN_IDX,
        "realized_60d_return_pct": COVID_REALIZED_60D_PCT,
        "v24_fold_index": v24_fold["fold_index"],
        "a2_fold_index": a2_fold["fold_index"],
        "matchers": {},
    }

    # ---- v2.4 matcher ----
    candidate_idx_v24 = np.arange(0, v24_fold["train_end"] + 1, dtype=np.int64)
    elig_v24 = eligible_candidates(candidate_idx_v24, features, COVID_ORIGIN_IDX, cfg_v24)
    z_cols = [f"zscore_{h}" for h in cfg_v24.zscore_horizons]
    z_target = features.iloc[COVID_ORIGIN_IDX][z_cols].to_numpy()
    z_cands = features.iloc[elig_v24][z_cols].to_numpy()
    weights_v24 = np.array(v24_fold["weights"])
    dist_v24 = composite_distance(z_target, z_cands, weights_v24)
    probs_v24 = distances_to_probs(dist_v24, target_n_eff=min(float(v24_fold["n_eff"]), elig_v24.size))

    # ---- A2.1 matcher ----
    candidate_idx_a2 = np.arange(0, a2_fold["train_end"] + 1, dtype=np.int64)
    elig_a2 = eligible_candidates(candidate_idx_a2, features, COVID_ORIGIN_IDX, cfg_a2)
    dist_a2 = corrwindow_distance(returns_arr, COVID_ORIGIN_IDX, elig_a2,
                                  window_length=cfg_a2.corrwindow_length)
    probs_a2 = distances_to_probs(dist_a2, target_n_eff=min(float(a2_fold["n_eff"]), elig_a2.size))

    for name, elig, probs, fold in [
        ("v2.4_weighted_euclidean", elig_v24, probs_v24, v24_fold),
        ("a2.1_corrwindow_L100", elig_a2, probs_a2, a2_fold),
    ]:
        fwds = forward_pct[elig]
        # High-return mask: realized 60d > +30%, > +40%, > +50%.
        mask_30 = fwds > 30
        mask_40 = fwds > 40
        mask_50 = fwds > 50
        # Probability mass concentrated on high-return candidates.
        prob_30 = float(probs[mask_30].sum())
        prob_40 = float(probs[mask_40].sum())
        prob_50 = float(probs[mask_50].sum())
        # Uniform baseline = count / N.
        uniform_30 = float(mask_30.sum() / elig.size)
        uniform_40 = float(mask_40.sum() / elig.size)
        uniform_50 = float(mask_50.sum() / elig.size)
        # Top-10 highest-return candidates with their assigned probs.
        order = np.argsort(fwds)[::-1][:10]
        top_rows = []
        for j in order:
            top_rows.append({
                "analog_idx": int(elig[j]),
                "date": log_ret.index[elig[j]].date().isoformat(),
                "forward_60d_pct": float(fwds[j]),
                "prob": float(probs[j]),
                "rank_by_prob": int((probs >= probs[j]).sum()),
            })
        # Weighted mean forward.
        wf = float((probs * fwds).sum())
        # Plus matcher's top-10 by prob, with forwards.
        prob_order = np.argsort(probs)[::-1][:10]
        top_prob = []
        for j in prob_order:
            top_prob.append({
                "analog_idx": int(elig[j]),
                "date": log_ret.index[elig[j]].date().isoformat(),
                "prob": float(probs[j]),
                "forward_60d_pct": float(fwds[j]),
            })
        out["matchers"][name] = {
            "n_eligible": int(elig.size),
            "count_above_30pct": int(mask_30.sum()),
            "count_above_40pct": int(mask_40.sum()),
            "count_above_50pct": int(mask_50.sum()),
            "prob_mass_above_30pct": prob_30,
            "prob_mass_above_40pct": prob_40,
            "prob_mass_above_50pct": prob_50,
            "uniform_baseline_above_30pct": uniform_30,
            "uniform_baseline_above_40pct": uniform_40,
            "uniform_baseline_above_50pct": uniform_50,
            "lift_above_30pct": prob_30 / uniform_30 if uniform_30 > 0 else float("nan"),
            "lift_above_40pct": prob_40 / uniform_40 if uniform_40 > 0 else float("nan"),
            "weighted_mean_forward_pct": wf,
            "top_10_by_forward_return": top_rows,
            "top_10_by_probability": top_prob,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT.relative_to(REPO)}")
    print()
    print(f"COVID anchor (2020-03-16): realized 60d = +{COVID_REALIZED_60D_PCT:.1f}%")
    for name, info in out["matchers"].items():
        print(f"\n--- {name} ---")
        print(f"  Eligible candidates: {info['n_eligible']}")
        print(f"  Candidates with realized >+30%: {info['count_above_30pct']} "
              f"({info['count_above_30pct']/info['n_eligible']*100:.1f}% of pool)")
        print(f"  Probability mass on those: {info['prob_mass_above_30pct']*100:.2f}% "
              f"(uniform baseline: {info['uniform_baseline_above_30pct']*100:.2f}%)")
        print(f"  Lift over uniform: {info['lift_above_30pct']:.2f}×")
        print(f"  Same for >+40%: count={info['count_above_40pct']}, "
              f"mass={info['prob_mass_above_40pct']*100:.3f}%, "
              f"lift={info['lift_above_40pct']:.2f}×")
        print(f"  Weighted mean forward: {info['weighted_mean_forward_pct']:+.2f}%")
        print(f"  Top-3 highest-return candidates and their probability rank in this matcher:")
        for r in info['top_10_by_forward_return'][:3]:
            print(f"    {r['date']}  fwd={r['forward_60d_pct']:+.1f}%  prob={r['prob']:.4f}  rank_by_prob={r['rank_by_prob']}/{info['n_eligible']}")


if __name__ == "__main__":
    main()
