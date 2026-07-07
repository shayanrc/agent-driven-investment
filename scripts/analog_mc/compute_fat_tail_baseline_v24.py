"""Compute the v2.4 fat-tail baseline JSON.

For each of the 15 anchors in `fat_tail_eval_anchors.json` (positive +
negative + regime_coverage), locates the canonical-run fold containing the
anchor's origin_idx, reads the cached forecast paths from
`forecasts.npz`, and tabulates:

  - per-step CRPS over the 60-day horizon (cumulative log return)
  - mean CRPS over the 60 days
  - 50% / 90% band day-count coverage
  - cumulative-return quantile curves (5/25/50/75/95) for downstream plots
  - terminal (day-60) percent-return bands and forecast median

Writes `results/analog_mc/data/fat_tail_baseline_v24.json`. Every v4
experiment that produces a forecast diffs its per-anchor numbers against
this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analog_mc.scoring import crps_per_step

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_baseline_v24.json"


def load_fold_summaries() -> list[dict]:
    return [
        json.loads((RUN_DIR / "folds" / d.name / "summary.json").read_text())
        for d in sorted((RUN_DIR / "folds").iterdir(), key=lambda p: int(p.name))
    ]


def find_fold(origin_idx: int, folds: list[dict]) -> dict:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    raise SystemExit(f"no fold contains origin_idx={origin_idx}")


def evaluate_anchor(
    anchor: dict, section: str, fold: dict
) -> dict:
    origin_idx = anchor["origin_idx"]
    fold_idx = fold["fold_index"]
    npz = np.load(RUN_DIR / "folds" / str(fold_idx) / "forecasts.npz")
    origins = npz["origin_idx"]
    matches = np.where(origins == origin_idx)[0]
    if matches.size == 0:
        raise SystemExit(
            f"origin_idx={origin_idx} not in fold {fold_idx}'s forecasts.npz "
            f"(fold has origins {origins[0]}–{origins[-1]})"
        )
    pos = int(matches[0])
    paths = npz["paths"][pos].astype(np.float64)  # (n_paths, 60)
    realized = npz["realized"][pos]  # (60,)

    cum_paths = np.cumsum(paths, axis=1)
    cum_realized = np.cumsum(realized)
    q5 = np.quantile(cum_paths, 0.05, axis=0)
    q25 = np.quantile(cum_paths, 0.25, axis=0)
    q50 = np.quantile(cum_paths, 0.50, axis=0)
    q75 = np.quantile(cum_paths, 0.75, axis=0)
    q95 = np.quantile(cum_paths, 0.95, axis=0)

    in_50 = int(((cum_realized >= q25) & (cum_realized <= q75)).sum())
    in_90 = int(((cum_realized >= q5) & (cum_realized <= q95)).sum())

    per_step = crps_per_step(paths, realized)
    mean_crps = float(per_step.mean())

    # Terminal percent-return band edges (price-relative).
    def pct(x: float) -> float:
        return float(np.exp(x) - 1.0) * 100.0

    return {
        "anchor_date": anchor["anchor_date"],
        "section": section,
        "origin_idx": origin_idx,
        "fold_index": fold_idx,
        "realized_60d_return_pct": anchor["realized_60d_return_pct"],
        "z50_classical": anchor.get("z50_classical"),
        "regime_label": anchor.get("regime_label"),
        "anchor_close": anchor.get("anchor_close"),
        "realized_60d_close": anchor.get("realized_60d_close"),
        "mean_crps": mean_crps,
        "crps_per_step": per_step.tolist(),
        "coverage": {
            "in_50_band_days": in_50,
            "in_90_band_days": in_90,
            "horizon_days": int(realized.size),
        },
        "terminal_60d_pct": {
            "q5": pct(float(q5[-1])),
            "q25": pct(float(q25[-1])),
            "q50": pct(float(q50[-1])),
            "q75": pct(float(q75[-1])),
            "q95": pct(float(q95[-1])),
        },
        "cum_logret_quantiles": {
            "q5": q5.tolist(),
            "q25": q25.tolist(),
            "q50": q50.tolist(),
            "q75": q75.tolist(),
            "q95": q95.tolist(),
        },
    }


def main() -> None:
    folds = load_fold_summaries()
    anchors_payload = json.loads(ANCHORS_JSON.read_text())

    sections = ["positive", "negative", "regime_coverage"]
    by_section: dict[str, list[dict]] = {s: [] for s in sections}
    flat: list[dict] = []
    for section in sections:
        for entry in anchors_payload.get(section, []):
            fold = find_fold(entry["origin_idx"], folds)
            row = evaluate_anchor(entry, section, fold)
            by_section[section].append(row)
            flat.append(row)

    # Aggregate stats.
    mean_crps_all = float(np.mean([r["mean_crps"] for r in flat]))
    cov_50 = [r["coverage"]["in_50_band_days"] for r in flat]
    cov_90 = [r["coverage"]["in_90_band_days"] for r in flat]
    n_anchors = len(flat)
    h = flat[0]["coverage"]["horizon_days"]

    # Failure / control sub-aggregates per V3.5 (using anchor_date matching).
    failure_dates = {
        "2010-04-23", "2001-10-02", "2018-10-08", "2020-03-16", "2026-02-19",
    }
    control_dates = {
        "1991-03-26", "2010-11-10", "2012-03-14", "2025-07-02", "2017-06-01",
    }
    failures = [r for r in flat if r["anchor_date"] in failure_dates]
    controls = [r for r in flat if r["anchor_date"] in control_dates]

    summary = {
        "canonical_run": str(RUN_DIR.relative_to(ROOT)),
        "n_anchors": n_anchors,
        "horizon_days": h,
        "aggregate": {
            "mean_crps": mean_crps_all,
            "median_in_50_band_days": float(np.median(cov_50)),
            "median_in_90_band_days": float(np.median(cov_90)),
            "mean_in_50_band_days": float(np.mean(cov_50)),
            "mean_in_90_band_days": float(np.mean(cov_90)),
        },
        "failure_anchors_aggregate": {
            "n": len(failures),
            "mean_crps": float(np.mean([r["mean_crps"] for r in failures])),
            "mean_in_50_band_days": float(
                np.mean([r["coverage"]["in_50_band_days"] for r in failures])
            ),
            "mean_in_90_band_days": float(
                np.mean([r["coverage"]["in_90_band_days"] for r in failures])
            ),
        },
        "control_anchors_aggregate": {
            "n": len(controls),
            "mean_crps": float(np.mean([r["mean_crps"] for r in controls])),
            "mean_in_50_band_days": float(
                np.mean([r["coverage"]["in_50_band_days"] for r in controls])
            ),
            "mean_in_90_band_days": float(
                np.mean([r["coverage"]["in_90_band_days"] for r in controls])
            ),
        },
        "anchors_by_section": by_section,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Console summary table.
    print(f"v2.4 fat-tail baseline written to {OUT_JSON.relative_to(ROOT)}")
    print(f"  {n_anchors} anchors, horizon {h} days")
    print(
        f"  aggregate mean CRPS: {mean_crps_all:.5f}, "
        f"mean 50-band: {summary['aggregate']['mean_in_50_band_days']:.1f}/60, "
        f"mean 90-band: {summary['aggregate']['mean_in_90_band_days']:.1f}/60"
    )
    print()
    print(
        f"  {'anchor':<14} {'section':<18} {'realized':>10} "
        f"{'mean_crps':>10} {'50/60':>6} {'90/60':>6}"
    )
    for r in flat:
        print(
            f"  {r['anchor_date']:<14} {r['section']:<18} "
            f"{r['realized_60d_return_pct']:>+9.1f}% "
            f"{r['mean_crps']:>10.5f} "
            f"{r['coverage']['in_50_band_days']:>6d} "
            f"{r['coverage']['in_90_band_days']:>6d}"
        )


if __name__ == "__main__":
    main()
