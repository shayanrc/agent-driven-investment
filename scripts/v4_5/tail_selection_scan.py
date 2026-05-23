"""V4.5.7 — Tail-positive selection generality.

V4.5.4 found that at 2020-03-16, both v2.4 and A2.1 assign less-than-uniform
probability mass to candidates with realized 60d > +30%. Does this hold at
the other fat-tail anchors?

Method: for each of the 15 anchors, compute matcher lift = mass(realized_60d
above |realized|) / uniform-baseline. If realized > 0, look at positive-tail
mass concentration; if realized < 0, look at negative-tail mass concentration.

Outputs: results/analog_mc/data/v4_5_7_tail_selection_scan.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.distances import composite_distance, distances_to_probs
from analog_mc.distances_corrwindow import corrwindow_distance
from analog_mc.features import compute_features
from analog_mc.local_linear import forward_logret_sums
from analog_mc.simulate import eligible_candidates

REPO = Path(__file__).resolve().parents[2]
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"
A2_RUN = REPO / "runs/analog_mc/20260521T061730Z"
ANCHORS = REPO / "results/analog_mc/data/fat_tail_eval_anchors.json"
OUT = REPO / "results/analog_mc/data/v4_5_7_tail_selection_scan.json"


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

    anchors_data = json.loads(ANCHORS.read_text())
    all_anchors = []
    for sec in ("positive", "negative", "regime_coverage"):
        for a in anchors_data[sec]:
            all_anchors.append((a, sec))

    z_cols = [f"zscore_{h}" for h in cfg_v24.zscore_horizons]

    rows = []
    print(f"{'anchor':<14} {'real%':>7} {'thr%':>6} {'v24 lift':>9} {'A2 lift':>9} {'v24 mass':>9} {'A2 mass':>9}")
    for a, sec in all_anchors:
        origin = a["origin_idx"]
        real = a["realized_60d_return_pct"]
        v24_fold = fold_for_origin(v24_folds, origin)
        a2_fold = fold_for_origin(a2_folds, origin)
        if v24_fold is None or a2_fold is None:
            continue

        # Probability layers.
        elig_v24 = eligible_candidates(np.arange(0, v24_fold["train_end"] + 1, dtype=np.int64),
                                       features, origin, cfg_v24)
        z_target = features.iloc[origin][z_cols].to_numpy()
        z_cands = features.iloc[elig_v24][z_cols].to_numpy()
        weights_v24 = np.array(v24_fold["weights"])
        dist_v24 = composite_distance(z_target, z_cands, weights_v24)
        probs_v24 = distances_to_probs(dist_v24, target_n_eff=min(float(v24_fold["n_eff"]), elig_v24.size))

        elig_a2 = eligible_candidates(np.arange(0, a2_fold["train_end"] + 1, dtype=np.int64),
                                      features, origin, cfg_a2)
        dist_a2 = corrwindow_distance(returns_arr, origin, elig_a2,
                                      window_length=cfg_a2.corrwindow_length)
        probs_a2 = distances_to_probs(dist_a2, target_n_eff=min(float(a2_fold["n_eff"]), elig_a2.size))

        # Build tail mask: candidates whose realized 60d return is on the same
        # side of zero as realized, and magnitude >= |realized|. Both matchers
        # must use the SAME elig set for a fair lift comparison — use elig_v24
        # for v24 stats and elig_a2 for A2 stats.
        thr = abs(real)
        if real > 0:
            mask_v24 = forward_pct[elig_v24] > thr
            mask_a2 = forward_pct[elig_a2] > thr
        else:
            mask_v24 = forward_pct[elig_v24] < -thr
            mask_a2 = forward_pct[elig_a2] < -thr

        n_tail_v24 = int(mask_v24.sum())
        n_tail_a2 = int(mask_a2.sum())
        n_v24 = elig_v24.size
        n_a2 = elig_a2.size

        if n_tail_v24 == 0 or n_tail_a2 == 0:
            # Pool genuinely lacks comparable analogs — different story than under-selection.
            lift_v24 = float("nan")
            lift_a2 = float("nan")
            mass_v24 = 0.0
            mass_a2 = 0.0
        else:
            mass_v24 = float(probs_v24[mask_v24].sum())
            mass_a2 = float(probs_a2[mask_a2].sum())
            uniform_v24 = n_tail_v24 / n_v24
            uniform_a2 = n_tail_a2 / n_a2
            lift_v24 = mass_v24 / uniform_v24 if uniform_v24 > 0 else float("nan")
            lift_a2 = mass_a2 / uniform_a2 if uniform_a2 > 0 else float("nan")

        rows.append({
            "anchor_date": a["anchor_date"],
            "section": sec,
            "origin_idx": origin,
            "realized_60d_return_pct": real,
            "threshold_pct": thr,
            "n_eligible_v24": n_v24,
            "n_tail_v24": n_tail_v24,
            "mass_tail_v24": mass_v24,
            "lift_v24": lift_v24,
            "n_eligible_a2": n_a2,
            "n_tail_a2": n_tail_a2,
            "mass_tail_a2": mass_a2,
            "lift_a2": lift_a2,
        })
        print(f"{a['anchor_date']:<14} {real:>+6.1f}% {thr:>5.1f}% "
              f"{lift_v24:>9.2f} {lift_a2:>9.2f} {mass_v24*100:>8.2f}% {mass_a2*100:>8.2f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": {
            "description": "V4.5.7 tail selection scan — for each anchor, compute matcher mass "
                          "concentration on same-sign candidates with |realized| ≥ |anchor realized|. "
                          "Lift = mass / uniform_baseline; <1 means matcher under-selects the tail.",
            "v24_run": str(V24_RUN.relative_to(REPO)),
            "a2_run": str(A2_RUN.relative_to(REPO)),
        },
        "anchors": rows,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
