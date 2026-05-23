"""V4.5.1 — A2.1 gate-signal validation.

Question: Does `corrwindow val_crps > k× cross-fold median` discriminate
A2.1 catastrophic folds from wins?

Method:
1. Read per-fold val_crps from A2.1v1 canonical.
2. Compute cross-fold median; threshold candidates {1.2, 1.5, 2.0, 3.0}.
3. For each fat-tail anchor, find its fold (test_start ≤ origin_idx ≤ test_end).
4. Reconstruct hypothetical V5.1 fat-tail panel per threshold: substitute
   v2.4 cells for gated folds.
5. Score against the v4 promotion bar.

Outputs:
- results/analog_mc/data/v4_5_1_gate_signal.json
- prints summary tables
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A2_RUN = REPO / "runs/analog_mc/20260521T061730Z"
B1_RUN = REPO / "runs/analog_mc/20260520T155220Z"
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"

DIFF_A2 = REPO / "results/analog_mc/data/fat_tail_a2_corrwindow_L100_diff.json"
DIFF_BASELINE = REPO / "results/analog_mc/data/fat_tail_baseline_v24.json"
ANCHORS = REPO / "results/analog_mc/data/fat_tail_eval_anchors.json"
OUT = REPO / "results/analog_mc/data/v4_5_1_gate_signal.json"

FAILURE_ANCHORS = {
    "2001-10-02",
    "2010-04-23",
    "2018-10-08",
    "2020-03-16",
    "2026-02-19",
}


def load_folds(run_dir: Path) -> list[dict]:
    out = []
    folds_dir = run_dir / "folds"
    for d in sorted(folds_dir.iterdir(), key=lambda p: int(p.name)):
        s = json.loads((d / "summary.json").read_text())
        out.append(s)
    return out


def fold_for_origin(folds: list[dict], origin_idx: int) -> int | None:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f["fold_index"]
    return None


def main() -> None:
    a2_folds = load_folds(A2_RUN)
    val_crps_a2 = [f["val_crps"] for f in a2_folds]
    n_folds = len(a2_folds)
    median = statistics.median(val_crps_a2)
    mean = statistics.fmean(val_crps_a2)

    anchors = json.loads(ANCHORS.read_text())
    # Flatten all anchor entries with section labels.
    all_anchors = []
    for sec in ("positive", "negative", "regime_coverage"):
        for a in anchors[sec]:
            all_anchors.append({**a, "section": sec})

    # Load the diff JSONs to get per-anchor CRPS / coverage.
    diff_a2 = json.loads(DIFF_A2.read_text())
    diff_index = {a["anchor_date"]: a for a in diff_a2["per_anchor"]}

    # Per-anchor enrichment with fold + val_crps.
    rows = []
    for a in all_anchors:
        date = a["anchor_date"]
        origin = a["origin_idx"]
        fold = fold_for_origin(a2_folds, origin)
        if fold is None:
            # Anchor outside any test window (e.g., very early or recent that
            # didn't get a test slice).
            continue
        fold_summary = a2_folds[fold]
        d = diff_index[date]
        rows.append({
            "anchor_date": date,
            "section": a["section"],
            "origin_idx": origin,
            "realized_60d_return_pct": a["realized_60d_return_pct"],
            "fold": fold,
            "val_crps_a2": fold_summary["val_crps"],
            "test_crps_a2_fold": fold_summary["test_crps"],
            "baseline_anchor_crps": d["baseline_mean_crps"],
            "a2_anchor_crps": d["eval_mean_crps"],
            "baseline_in_90": d["baseline_in_90"],
            "a2_in_90": d["eval_in_90"],
            "delta_crps_rel": d["delta_mean_crps_rel"],
            "delta_in_90": d["delta_in_90"],
            "is_failure": date in FAILURE_ANCHORS,
        })

    # Threshold sweep.
    thresholds = [1.2, 1.5, 2.0, 3.0]
    sweep_results = {}
    for k in thresholds:
        thr = k * median
        gated_folds = {f["fold_index"] for f in a2_folds if f["val_crps"] > thr}
        # Predict V5.1 panel: for gated rows substitute v2.4; for non-gated keep A2.1.
        predicted = []
        for r in rows:
            if r["fold"] in gated_folds:
                pred_crps = r["baseline_anchor_crps"]
                pred_in90 = r["baseline_in_90"]
                source = "v2.4_via_gate"
            else:
                pred_crps = r["a2_anchor_crps"]
                pred_in90 = r["a2_in_90"]
                source = "a2.1"
            predicted.append({
                **r,
                "v5_1_pred_crps": pred_crps,
                "v5_1_pred_in_90": pred_in90,
                "pred_source": source,
                "v5_1_delta_crps_rel": (pred_crps - r["baseline_anchor_crps"])
                / r["baseline_anchor_crps"] if r["baseline_anchor_crps"] else 0.0,
            })
        # Promotion-bar metrics.
        regressions_5pct = sum(1 for p in predicted if p["v5_1_delta_crps_rel"] > 0.05)
        # Failures recovered = 90-band >= 45 at the 5 failure anchors.
        failures_recovered = sum(1 for p in predicted if p["is_failure"] and p["v5_1_pred_in_90"] >= 45)
        # Anchors classified.
        n_gated_anchors = sum(1 for p in predicted if p["pred_source"] == "v2.4_via_gate")
        sweep_results[f"k_{k}"] = {
            "threshold_multiplier": k,
            "threshold_val_crps": thr,
            "n_gated_folds": len(gated_folds),
            "gated_folds": sorted(gated_folds),
            "n_gated_anchors": n_gated_anchors,
            "failures_recovered_of_5": failures_recovered,
            "regressions_above_5pct_of_15": regressions_5pct,
            "predicted_per_anchor": predicted,
            "promotion_bar_passed": failures_recovered >= 3 and regressions_5pct <= 2,
        }

    out = {
        "method": {
            "description": "V4.5.1 gate-signal validation: substitute v2.4 anchor cells "
                          "for folds where A2.1 val_crps > k × cross-fold median.",
            "a2_run": str(A2_RUN.relative_to(REPO)),
            "n_folds": n_folds,
        },
        "val_crps_stats_a2": {
            "median": median,
            "mean": mean,
            "min": min(val_crps_a2),
            "max": max(val_crps_a2),
            "p25": sorted(val_crps_a2)[n_folds // 4],
            "p75": sorted(val_crps_a2)[3 * n_folds // 4],
        },
        "anchors_in_test_windows": len(rows),
        "thresholds": sweep_results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT.relative_to(REPO)}")
    print(f"\nVal CRPS stats: median={median:.5f} mean={mean:.5f} min={min(val_crps_a2):.5f} max={max(val_crps_a2):.5f}")
    print(f"\n{'k':>4} {'thr':>9} {'gFolds':>7} {'gAnch':>6} {'failRec/5':>10} {'regr>5%/15':>11} {'pass?':>6}")
    for key, s in sweep_results.items():
        print(f"{s['threshold_multiplier']:>4.1f} {s['threshold_val_crps']:>9.5f} "
              f"{s['n_gated_folds']:>7d} {s['n_gated_anchors']:>6d} "
              f"{s['failures_recovered_of_5']:>10d} {s['regressions_above_5pct_of_15']:>11d} "
              f"{'YES' if s['promotion_bar_passed'] else 'no':>6}")

    # Per-anchor disposition table at k=1.5 (the V4_RESULTS recommendation).
    print("\nAt k=1.5 (V4_RESULTS recommendation):")
    print(f"{'anchor':<14} {'fold':>5} {'val_crps':>9} {'gated?':>7} {'A2 Δ':>7} {'V5.1 Δ':>7} {'A2 in90':>7} {'V5.1 in90':>9} {'fail?':>6}")
    s15 = sweep_results["k_1.5"]
    for p in s15["predicted_per_anchor"]:
        gated = "yes" if p["pred_source"] == "v2.4_via_gate" else "no"
        print(f"{p['anchor_date']:<14} {p['fold']:>5} {p['val_crps_a2']:>9.5f} {gated:>7} "
              f"{p['delta_crps_rel']*100:>+6.1f}% {p['v5_1_delta_crps_rel']*100:>+6.1f}% "
              f"{p['a2_in_90']:>7d} {p['v5_1_pred_in_90']:>9d} "
              f"{'YES' if p['is_failure'] else '':>6}")


if __name__ == "__main__":
    main()
