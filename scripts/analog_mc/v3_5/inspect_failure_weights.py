"""V3.5.1 — per-failure weight inspection.

Reads the canonical run's per-fold summary.json files. For each of the 5
failure anchors and 5 control anchors, locates the fold whose test window
contains the anchor's origin_idx and records (weights, n_eff, val_crps,
test_crps). Computes cross-fold median weights/n_eff for context.

Writes:
  - results/analog_mc/data/v3_5_1_weights.json
  - docs/analog_mc/v3.5/_v3_5_1_weights.md
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "v3_5_1_weights.json"
OUT_MD = ROOT / "docs" / "analog_mc" / "v3.5" / "_v3_5_1_weights.md"

# 5 failure anchors from FAT_TAIL_EVAL.md "Pattern 4"
FAILURE_DATES = [
    "2010-04-23",
    "2001-10-02",
    "2018-10-08",
    "2020-03-16",
    "2026-02-19",
]
# 5 control anchors (passed strongly per V3.5 plan)
CONTROL_DATES = [
    "1991-03-26",
    "2010-11-10",
    "2012-03-14",
    "2025-07-02",
    "2017-06-01",
]


def load_anchors() -> dict[str, dict]:
    payload = json.loads(ANCHORS_JSON.read_text())
    out: dict[str, dict] = {}
    for section in ("positive", "negative", "regime_coverage"):
        for entry in payload.get(section, []):
            out[entry["anchor_date"]] = entry
    return out


def load_all_folds() -> list[dict]:
    summaries: list[dict] = []
    for fold_dir in sorted(
        (RUN_DIR / "folds").iterdir(), key=lambda p: int(p.name)
    ):
        summary = json.loads((fold_dir / "summary.json").read_text())
        summaries.append(summary)
    return summaries


def find_fold(origin_idx: int, folds: list[dict]) -> dict | None:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    return None


def fmt_weights(w: list[float]) -> str:
    return f"[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]"


def main() -> None:
    anchors = load_anchors()
    folds = load_all_folds()

    rows_failure = []
    rows_control = []
    for label, dates, bucket in [
        ("failure", FAILURE_DATES, rows_failure),
        ("control", CONTROL_DATES, rows_control),
    ]:
        for d in dates:
            if d not in anchors:
                raise SystemExit(f"anchor {d} not in eval JSON")
            origin_idx = anchors[d]["origin_idx"]
            fold = find_fold(origin_idx, folds)
            if fold is None:
                raise SystemExit(f"no fold contains origin_idx={origin_idx} for {d}")
            bucket.append({
                "label": label,
                "anchor_date": d,
                "origin_idx": origin_idx,
                "fold_index": fold["fold_index"],
                "weights": fold["weights"],
                "n_eff": fold["n_eff"],
                "val_crps": fold["val_crps"],
                "test_crps": fold["test_crps"],
                "realized_60d_pct": anchors[d]["realized_60d_return_pct"],
            })

    # Cross-fold median
    weights_arr = [f["weights"] for f in folds]
    median_w = [statistics.median([w[i] for w in weights_arr]) for i in range(3)]
    mean_w = [statistics.fmean([w[i] for w in weights_arr]) for i in range(3)]
    median_n_eff = statistics.median([f["n_eff"] for f in folds])
    mean_n_eff = statistics.fmean([f["n_eff"] for f in folds])

    # Buckets for w0 (z20) — myopia detector
    w0_vals = [w[0] for w in weights_arr]
    w0_short = sum(1 for x in w0_vals if x > 0.5)
    w0_long = sum(1 for x in w0_vals if x < 0.2)

    # Failure w0 stats
    failure_w0 = [r["weights"][0] for r in rows_failure]
    failure_w2 = [r["weights"][2] for r in rows_failure]
    control_w0 = [r["weights"][0] for r in rows_control]
    control_w2 = [r["weights"][2] for r in rows_control]

    out = {
        "canonical_run": str(RUN_DIR.relative_to(ROOT)),
        "n_folds": len(folds),
        "cross_fold": {
            "median_weights": median_w,
            "mean_weights": mean_w,
            "median_n_eff": median_n_eff,
            "mean_n_eff": mean_n_eff,
            "w0_short_horizon_count_gt_0_5": w0_short,
            "w0_long_horizon_count_lt_0_2": w0_long,
        },
        "failures": rows_failure,
        "controls": rows_control,
        "summary_stats": {
            "failure_w0_mean": statistics.fmean(failure_w0),
            "failure_w0_median": statistics.median(failure_w0),
            "failure_w2_mean": statistics.fmean(failure_w2),
            "control_w0_mean": statistics.fmean(control_w0),
            "control_w0_median": statistics.median(control_w0),
            "control_w2_mean": statistics.fmean(control_w2),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Markdown report
    lines: list[str] = []
    lines.append("# V3.5.1 — per-failure weight inspection")
    lines.append("")
    lines.append(
        "Canonical run: "
        f"`{RUN_DIR.relative_to(ROOT)}` ({len(folds)} folds, "
        "66×5 weight grid)."
    )
    lines.append("")
    lines.append("Weights are `[w_z20, w_z50, w_z200]` (short, medium, long horizon).")
    lines.append("")

    lines.append("## Cross-fold reference")
    lines.append("")
    lines.append("| Stat | w_z20 | w_z50 | w_z200 | n_eff |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| median | {median_w[0]:.3f} | {median_w[1]:.3f} | {median_w[2]:.3f} | "
        f"{median_n_eff:.1f} |"
    )
    lines.append(
        f"| mean | {mean_w[0]:.3f} | {mean_w[1]:.3f} | {mean_w[2]:.3f} | "
        f"{mean_n_eff:.1f} |"
    )
    lines.append("")
    lines.append(
        f"Folds with **w_z20 > 0.5** (short-horizon dominant): "
        f"**{w0_short}/{len(folds)}**."
    )
    lines.append(
        f"Folds with **w_z20 < 0.2** (long-horizon dominant): "
        f"**{w0_long}/{len(folds)}**."
    )
    lines.append("")

    for label, rows in [("Failure", rows_failure), ("Control", rows_control)]:
        lines.append(f"## {label} anchors")
        lines.append("")
        lines.append(
            "| Anchor | Fold | Weights `[w20,w50,w200]` | n_eff | val_crps | test_crps | realized 60d |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['anchor_date']} | {r['fold_index']} | "
                f"{fmt_weights(r['weights'])} | {r['n_eff']:.0f} | "
                f"{r['val_crps']:.5f} | {r['test_crps']:.5f} | "
                f"{r['realized_60d_pct']:+.1f}% |"
            )
        lines.append("")

    lines.append("## Distribution stats")
    lines.append("")
    lines.append("| | failure | control | cross-fold |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| w_z20 mean | {statistics.fmean(failure_w0):.3f} | "
        f"{statistics.fmean(control_w0):.3f} | {mean_w[0]:.3f} |"
    )
    lines.append(
        f"| w_z20 median | {statistics.median(failure_w0):.3f} | "
        f"{statistics.median(control_w0):.3f} | {median_w[0]:.3f} |"
    )
    lines.append(
        f"| w_z200 mean | {statistics.fmean(failure_w2):.3f} | "
        f"{statistics.fmean(control_w2):.3f} | {mean_w[2]:.3f} |"
    )
    lines.append("")

    # Verdict
    all_failures_short = all(w > 0.5 for w in failure_w0) and all(
        r["weights"][2] < 0.2 for r in rows_failure
    )
    failure_at_extremes = sum(
        1 for r in rows_failure
        if max(r["weights"]) > 0.95 or min(r["weights"]) > 0.30  # near-uniform if all ~0.33
    )
    near_uniform_failures = sum(
        1 for r in rows_failure
        if max(r["weights"]) < 0.55 and min(r["weights"]) > 0.20
    )
    extreme_corner_failures = sum(
        1 for r in rows_failure
        if max(r["weights"]) > 0.95
    )

    lines.append("## Verdict")
    lines.append("")
    if all_failures_short:
        verdict = (
            "**All 5 failures favor short-horizon (w_z20 > 0.5 with w_z200 < 0.2).** "
            "Search is myopic at transitions. **Recommend promoting B3 (Dirichlet "
            "weight posterior) to P0 in V4_EXPERIMENTS_PLAN.md.**"
        )
    elif extreme_corner_failures >= 3 and near_uniform_failures >= 1:
        verdict = (
            f"**Mixed corner/uniform pattern**: {extreme_corner_failures}/5 failures "
            f"sit at extreme weight corners (max weight > 0.95) and "
            f"{near_uniform_failures}/5 sit near uniform. Val-set unreliability "
            "suspected — search lands at boundary or center depending on noise. "
            "**Motivates a v4 experiment regularizing the search prior** "
            "(e.g., Dirichlet prior, or shrinkage toward cross-fold mean)."
        )
    else:
        verdict = (
            "**Weights heterogeneous across failures.** No single tuning regime "
            "explains all 5 misses — this isn't a search-myopia problem in "
            "isolation. **Non-finding for v4 reshape — does not change priorities.** "
            "Proceed to V3.5.2."
        )
    lines.append(verdict)
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))

    # Print to stdout for quick inspection
    print("\n".join(lines))


if __name__ == "__main__":
    main()
