"""V4.5.8 — V5.A.2 path-level ensemble preview.

Build a hypothetical V5.A.2 fat-tail panel by mixing v2.4 + A2.1 paths
50/50 from existing forecasts.npz files. Compares to v2.4 baseline AND to
A2.1 to score the v5 promotion bar before committing to a formal canonical.

Outputs:
- results/analog_mc/data/v4_5_8_v5a2_preview.json
- prints anchor-by-anchor + aggregate tables
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analog_mc.scoring import crps_per_step

REPO = Path(__file__).resolve().parents[2]
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"
A2_RUN = REPO / "runs/analog_mc/20260521T061730Z"

ANCHORS = REPO / "results/analog_mc/data/fat_tail_eval_anchors.json"
BASELINE_JSON = REPO / "results/analog_mc/data/fat_tail_baseline_v24.json"
A2_JSON = REPO / "results/analog_mc/data/fat_tail_a2_corrwindow_L100.json"
OUT = REPO / "results/analog_mc/data/v4_5_8_v5a2_preview.json"

FAILURE_DATES = {"2010-04-23", "2001-10-02", "2018-10-08", "2020-03-16", "2026-02-19"}
RNG = np.random.default_rng(42)


def load_fold_summaries(run_dir: Path) -> list[dict]:
    return [
        json.loads((run_dir / "folds" / d.name / "summary.json").read_text())
        for d in sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name))
    ]


def find_fold(origin_idx: int, folds: list[dict]) -> dict | None:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    return None


def get_paths_realized(run_dir: Path, fold_idx: int, origin_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    npz = np.load(run_dir / "folds" / str(fold_idx) / "forecasts.npz")
    origins = npz["origin_idx"]
    matches = np.where(origins == origin_idx)[0]
    if matches.size == 0:
        return None
    pos = int(matches[0])
    return npz["paths"][pos].astype(np.float64), npz["realized"][pos]


def eval_paths(paths: np.ndarray, realized: np.ndarray) -> dict:
    """Compute mean CRPS + 50/90-band coverage given (n_paths, H) paths."""
    cum_paths = np.cumsum(paths, axis=1)
    cum_realized = np.cumsum(realized)
    q5 = np.quantile(cum_paths, 0.05, axis=0)
    q25 = np.quantile(cum_paths, 0.25, axis=0)
    q75 = np.quantile(cum_paths, 0.75, axis=0)
    q95 = np.quantile(cum_paths, 0.95, axis=0)
    in_50 = int(((cum_realized >= q25) & (cum_realized <= q75)).sum())
    in_90 = int(((cum_realized >= q5) & (cum_realized <= q95)).sum())
    per_step = crps_per_step(paths, realized)
    return {
        "mean_crps": float(per_step.mean()),
        "in_50_band_days": in_50,
        "in_90_band_days": in_90,
    }


def main() -> None:
    anchors = json.loads(ANCHORS.read_text())
    all_anchors = []
    for sec in ("positive", "negative", "regime_coverage"):
        for a in anchors[sec]:
            all_anchors.append({**a, "section": sec})

    v24_folds = load_fold_summaries(V24_RUN)
    a2_folds = load_fold_summaries(A2_RUN)

    # Load baseline + A2 fat-tail JSONs for direct comparison.
    base = json.loads(BASELINE_JSON.read_text())
    a2_eval = json.loads(A2_JSON.read_text())
    base_by_date = {}
    for sec_rows in base["anchors_by_section"].values():
        for r in sec_rows:
            base_by_date[r["anchor_date"]] = r
    a2_by_date = {}
    for sec_rows in a2_eval["anchors_by_section"].values():
        for r in sec_rows:
            a2_by_date[r["anchor_date"]] = r

    rows = []
    print(f"{'anchor':<14} {'real%':>7} {'v24 CRPS':>9} {'A2 CRPS':>8} {'v5A2 CRPS':>10}  "
          f"{'Δ vs v24':>9} {'Δ vs A2':>9}  {'v24 90':>6} {'A2 90':>6} {'v5A2 90':>7} {'fail?':>5}")
    for a in all_anchors:
        date = a["anchor_date"]
        origin = a["origin_idx"]
        v24_fold = find_fold(origin, v24_folds)
        a2_fold = find_fold(origin, a2_folds)
        if v24_fold is None or a2_fold is None:
            continue
        v24_out = get_paths_realized(V24_RUN, v24_fold["fold_index"], origin)
        a2_out = get_paths_realized(A2_RUN, a2_fold["fold_index"], origin)
        if v24_out is None or a2_out is None:
            print(f"  {date}: missing in cache, skip")
            continue
        v24_paths, v24_real = v24_out
        a2_paths, a2_real = a2_out
        assert np.allclose(v24_real, a2_real, atol=1e-9), f"realized mismatch at {date}"

        # Try multiple mix ratios: fraction of paths from A2.1.
        mix_results = {}
        n_v24 = v24_paths.shape[0]
        n_a2 = a2_paths.shape[0]
        n_target = 1000
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            n_a2_take = int(round(alpha * n_target))
            n_v24_take = n_target - n_a2_take
            idx_v24 = RNG.choice(n_v24, size=min(n_v24_take, n_v24), replace=False) if n_v24_take > 0 else np.array([], dtype=int)
            idx_a2 = RNG.choice(n_a2, size=min(n_a2_take, n_a2), replace=False) if n_a2_take > 0 else np.array([], dtype=int)
            parts = []
            if idx_v24.size > 0:
                parts.append(v24_paths[idx_v24])
            if idx_a2.size > 0:
                parts.append(a2_paths[idx_a2])
            mixed = np.concatenate(parts, axis=0)
            mix_results[f"alpha_{alpha}"] = eval_paths(mixed, v24_real)

        # Use 50/50 as the "primary" V5.A.2 result.
        mix_metrics = mix_results["alpha_0.5"]

        # Score full v24 and A2 paths for sanity.
        v24_metrics = eval_paths(v24_paths, v24_real)
        a2_metrics = eval_paths(a2_paths, a2_real)

        # Baseline values from cached JSONs (sanity-check our eval matches).
        base_crps = base_by_date[date]["mean_crps"]
        a2_crps = a2_by_date[date]["mean_crps"]
        # Small drift expected between live eval and cached due to quantile rounding;
        # we trust the live eval for the mixed case since we generated the mix here.

        d_vs_v24 = (mix_metrics["mean_crps"] - base_crps) / max(base_crps, 1e-9)
        d_vs_a2 = (mix_metrics["mean_crps"] - a2_crps) / max(a2_crps, 1e-9)

        is_failure = date in FAILURE_DATES

        rows.append({
            "anchor_date": date,
            "section": a["section"],
            "origin_idx": origin,
            "realized_60d_return_pct": a["realized_60d_return_pct"],
            "v24_baseline_crps": base_crps,
            "a2_crps_cached": a2_crps,
            "v24_live_crps": v24_metrics["mean_crps"],
            "a2_live_crps": a2_metrics["mean_crps"],
            "v5a2_crps": mix_metrics["mean_crps"],
            "delta_crps_rel_vs_v24": d_vs_v24,
            "delta_crps_rel_vs_a2": d_vs_a2,
            "v24_in_50": base_by_date[date]["coverage"]["in_50_band_days"],
            "v24_in_90": base_by_date[date]["coverage"]["in_90_band_days"],
            "a2_in_50": a2_by_date[date]["coverage"]["in_50_band_days"],
            "a2_in_90": a2_by_date[date]["coverage"]["in_90_band_days"],
            "v5a2_in_50": mix_metrics["in_50_band_days"],
            "v5a2_in_90": mix_metrics["in_90_band_days"],
            "is_failure": is_failure,
            "alpha_sweep": mix_results,
        })
        print(f"{date:<14} {a['realized_60d_return_pct']:>+6.1f}% "
              f"{base_crps:>9.5f} {a2_crps:>8.5f} {mix_metrics['mean_crps']:>10.5f}  "
              f"{d_vs_v24*100:>+8.1f}% {d_vs_a2*100:>+8.1f}%  "
              f"{base_by_date[date]['coverage']['in_90_band_days']:>6d} "
              f"{a2_by_date[date]['coverage']['in_90_band_days']:>6d} "
              f"{mix_metrics['in_90_band_days']:>7d} "
              f"{'YES' if is_failure else '':>5}")

    # Promotion-bar scoring.
    failures_recovered = sum(1 for r in rows if r["is_failure"] and r["v5a2_in_90"] >= 45)
    regressions_vs_v24 = sum(1 for r in rows if r["delta_crps_rel_vs_v24"] > 0.05)
    a2_failures_recovered = sum(1 for r in rows if r["is_failure"] and r["a2_in_90"] >= 45)
    a2_regressions = sum(1 for r in rows
                         if (r["a2_crps_cached"] - r["v24_baseline_crps"]) / max(r["v24_baseline_crps"], 1e-9) > 0.05)

    print()
    print("=== Promotion-bar scoring ===")
    print(f"  {'':<25} {'v2.4':>8} {'A2.1':>8} {'V5.A.2':>8}")
    print(f"  {'failures recovered /5':<25} {'(ref)':>8} {a2_failures_recovered:>8d} {failures_recovered:>8d}")
    print(f"  {'regressions >5% /15':<25} {'(ref)':>8} {a2_regressions:>8d} {regressions_vs_v24:>8d}")
    passed = failures_recovered >= 3 and regressions_vs_v24 <= 2
    print(f"  Promotion bar (≥3/5 AND ≤2/15): {'PASS' if passed else 'FAIL'}")

    # Aggregates.
    avg_v24 = float(np.mean([r["v24_baseline_crps"] for r in rows]))
    avg_a2 = float(np.mean([r["a2_crps_cached"] for r in rows]))
    avg_v5a2 = float(np.mean([r["v5a2_crps"] for r in rows]))
    fail_v24 = float(np.mean([r["v24_baseline_crps"] for r in rows if r["is_failure"]]))
    fail_a2 = float(np.mean([r["a2_crps_cached"] for r in rows if r["is_failure"]]))
    fail_v5a2 = float(np.mean([r["v5a2_crps"] for r in rows if r["is_failure"]]))
    ctrl_v24 = float(np.mean([r["v24_baseline_crps"] for r in rows if not r["is_failure"]]))
    ctrl_a2 = float(np.mean([r["a2_crps_cached"] for r in rows if not r["is_failure"]]))
    ctrl_v5a2 = float(np.mean([r["v5a2_crps"] for r in rows if not r["is_failure"]]))
    print()
    print(f"  Mean CRPS (15 anchors):  v2.4={avg_v24:.5f}  A2.1={avg_a2:.5f}  V5.A.2={avg_v5a2:.5f}")
    print(f"  Failure mean CRPS (5):   v2.4={fail_v24:.5f}  A2.1={fail_a2:.5f}  V5.A.2={fail_v5a2:.5f} "
          f"(Δ vs v24: {(fail_v5a2-fail_v24)/fail_v24*100:+.1f}%)")
    print(f"  Control mean CRPS (10):  v2.4={ctrl_v24:.5f}  A2.1={ctrl_a2:.5f}  V5.A.2={ctrl_v5a2:.5f} "
          f"(Δ vs v24: {(ctrl_v5a2-ctrl_v24)/ctrl_v24*100:+.1f}%)")

    payload = {
        "method": {
            "description": "V4.5.8 path-level 50/50 ensemble preview using cached v2.4 + A2.1 forecasts.",
            "v24_run": str(V24_RUN.relative_to(REPO)),
            "a2_run": str(A2_RUN.relative_to(REPO)),
            "n_paths_per_anchor": 1000,
            "split": "500 from v2.4 + 500 from A2.1 (random sample, seed=42)",
        },
        "promotion_bar": {
            "v5_a2_failures_recovered_of_5": failures_recovered,
            "v5_a2_regressions_above_5pct_of_15": regressions_vs_v24,
            "v5_a2_promotion_bar_passed": passed,
            "a2_baseline_failures_recovered_of_5": a2_failures_recovered,
            "a2_baseline_regressions_of_15": a2_regressions,
        },
        "aggregate": {
            "all_anchors_n": len(rows),
            "v24_mean_crps": avg_v24,
            "a2_mean_crps": avg_a2,
            "v5_a2_mean_crps": avg_v5a2,
            "failure_v24_mean_crps": fail_v24,
            "failure_a2_mean_crps": fail_a2,
            "failure_v5_a2_mean_crps": fail_v5a2,
            "failure_delta_vs_v24_pct": (fail_v5a2 - fail_v24) / fail_v24 * 100.0,
            "control_v24_mean_crps": ctrl_v24,
            "control_a2_mean_crps": ctrl_a2,
            "control_v5_a2_mean_crps": ctrl_v5a2,
            "control_delta_vs_v24_pct": (ctrl_v5a2 - ctrl_v24) / ctrl_v24 * 100.0,
        },
        "anchors": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    # Alpha sweep summary.
    print()
    print("=== α-sweep promotion-bar scoring (α = A2.1 fraction) ===")
    print(f"{'alpha':>6} {'fail_rec/5':>10} {'regr_>5%/15':>11} {'fail_CRPS':>10} {'ctrl_CRPS':>10} {'pass?':>5}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        key = f"alpha_{alpha}"
        fr = sum(1 for r in rows if r["is_failure"] and r["alpha_sweep"][key]["in_90_band_days"] >= 45)
        regs = sum(1 for r in rows
                   if (r["alpha_sweep"][key]["mean_crps"] - r["v24_baseline_crps"]) / max(r["v24_baseline_crps"], 1e-9) > 0.05)
        fail_c = float(np.mean([r["alpha_sweep"][key]["mean_crps"] for r in rows if r["is_failure"]]))
        ctrl_c = float(np.mean([r["alpha_sweep"][key]["mean_crps"] for r in rows if not r["is_failure"]]))
        passed_a = fr >= 3 and regs <= 2
        print(f"{alpha:>6.2f} {fr:>10d} {regs:>11d} {fail_c:>10.5f} {ctrl_c:>10.5f} {'PASS' if passed_a else 'no':>5}")

    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
