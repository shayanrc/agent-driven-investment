"""Compute the fat-tail eval JSON for any walk-forward run.

Generic version of `compute_fat_tail_baseline_v24.py`. Takes a run dir and a
label; produces the per-anchor coverage/CRPS table and (optionally) a diff
against a baseline JSON.

Usage:
    # Compute the eval for a new B1 run, diffed against the v2.4 baseline.
    uv run python scripts/analog_mc/compute_fat_tail_eval.py \\
        --run-dir runs/analog_mc/20260520T155220Z \\
        --label b1_local_linear \\
        --baseline-json results/analog_mc/data/fat_tail_baseline_v24.json

Outputs:
    results/analog_mc/data/fat_tail_<label>.json     # eval payload
    docs/analog_mc/experiments/_<label>_fat_tail.md  # markdown summary table

The v2.4 case can be reproduced by passing --label baseline_v24 with no
baseline, then reusing this file for all v4 experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analog_mc.scoring import crps_per_step

ROOT = Path(__file__).resolve().parents[2]
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"

FAILURE_DATES = {
    "2010-04-23", "2001-10-02", "2018-10-08", "2020-03-16", "2026-02-19",
}
CONTROL_DATES = {
    "1991-03-26", "2010-11-10", "2012-03-14", "2025-07-02", "2017-06-01",
}


def load_fold_summaries(run_dir: Path) -> list[dict]:
    return [
        json.loads((run_dir / "folds" / d.name / "summary.json").read_text())
        for d in sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name))
    ]


def find_fold(origin_idx: int, folds: list[dict]) -> dict:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    raise SystemExit(f"no fold contains origin_idx={origin_idx}")


def evaluate_anchor(anchor: dict, section: str, fold: dict, run_dir: Path) -> dict:
    origin_idx = anchor["origin_idx"]
    fold_idx = fold["fold_index"]
    npz = np.load(run_dir / "folds" / str(fold_idx) / "forecasts.npz")
    origins = npz["origin_idx"]
    matches = np.where(origins == origin_idx)[0]
    if matches.size == 0:
        raise SystemExit(
            f"origin_idx={origin_idx} not in fold {fold_idx}'s forecasts.npz "
            f"(fold has origins {origins[0]}–{origins[-1]})"
        )
    pos = int(matches[0])
    paths = npz["paths"][pos].astype(np.float64)
    realized = npz["realized"][pos]

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


def aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_crps": float(np.mean([r["mean_crps"] for r in rows])),
        "mean_in_50_band_days": float(
            np.mean([r["coverage"]["in_50_band_days"] for r in rows])
        ),
        "mean_in_90_band_days": float(
            np.mean([r["coverage"]["in_90_band_days"] for r in rows])
        ),
    }


def diff_payload(eval_payload: dict, baseline_payload: dict) -> dict:
    """Per-anchor and aggregate diffs of `eval_payload` against
    `baseline_payload`. Looks up by anchor_date.
    """
    base_by_date = {}
    for sec_rows in baseline_payload["anchors_by_section"].values():
        for r in sec_rows:
            base_by_date[r["anchor_date"]] = r
    eval_by_date = {}
    for sec_rows in eval_payload["anchors_by_section"].values():
        for r in sec_rows:
            eval_by_date[r["anchor_date"]] = r

    per_anchor = []
    for d, r in eval_by_date.items():
        b = base_by_date.get(d)
        if b is None:
            continue
        per_anchor.append({
            "anchor_date": d,
            "section": r["section"],
            "realized_60d_return_pct": r["realized_60d_return_pct"],
            "baseline_mean_crps": b["mean_crps"],
            "eval_mean_crps": r["mean_crps"],
            "delta_mean_crps": r["mean_crps"] - b["mean_crps"],
            "delta_mean_crps_rel": (r["mean_crps"] - b["mean_crps"]) / max(b["mean_crps"], 1e-9),
            "baseline_in_50": b["coverage"]["in_50_band_days"],
            "eval_in_50": r["coverage"]["in_50_band_days"],
            "delta_in_50": r["coverage"]["in_50_band_days"] - b["coverage"]["in_50_band_days"],
            "baseline_in_90": b["coverage"]["in_90_band_days"],
            "eval_in_90": r["coverage"]["in_90_band_days"],
            "delta_in_90": r["coverage"]["in_90_band_days"] - b["coverage"]["in_90_band_days"],
        })

    # Aggregate diff
    base_agg = baseline_payload["aggregate"]
    eval_agg = eval_payload["aggregate"]
    base_fail = baseline_payload["failure_anchors_aggregate"]
    eval_fail = eval_payload["failure_anchors_aggregate"]
    base_ctrl = baseline_payload["control_anchors_aggregate"]
    eval_ctrl = eval_payload["control_anchors_aggregate"]

    def agg_diff(b: dict, e: dict) -> dict:
        return {
            "baseline_mean_crps": b["mean_crps"],
            "eval_mean_crps": e["mean_crps"],
            "delta_mean_crps": e["mean_crps"] - b["mean_crps"],
            "delta_mean_crps_rel": (e["mean_crps"] - b["mean_crps"]) / max(b["mean_crps"], 1e-9),
            "baseline_mean_in_90": b["mean_in_90_band_days"],
            "eval_mean_in_90": e["mean_in_90_band_days"],
            "delta_mean_in_90": e["mean_in_90_band_days"] - b["mean_in_90_band_days"],
        }

    return {
        "per_anchor": per_anchor,
        "all_anchors": agg_diff(base_agg, eval_agg),
        "failure_anchors": agg_diff(base_fail, eval_fail),
        "control_anchors": agg_diff(base_ctrl, eval_ctrl),
    }


def build_markdown(label: str, payload: dict, diff: dict | None) -> str:
    lines: list[str] = []
    lines.append(f"# {label} — fat-tail panel")
    lines.append("")
    lines.append(
        f"Run: `{payload['canonical_run']}` · "
        f"{payload['n_anchors']} anchors × {payload['horizon_days']}-day horizon."
    )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    if diff is None:
        lines.append("| Group | n | Mean CRPS | Mean 50/60 | Mean 90/60 |")
        lines.append("|---|---:|---:|---:|---:|")
        for label_, key in [("All", "aggregate"), ("Failure", "failure_anchors_aggregate"), ("Control", "control_anchors_aggregate")]:
            d = payload[key]
            n = d.get("n", payload.get("n_anchors", "—"))
            lines.append(
                f"| {label_} | {n} | {d['mean_crps']:.5f} | "
                f"{d.get('mean_in_50_band_days', '—'):.1f} | "
                f"{d.get('mean_in_90_band_days', '—'):.1f} |"
            )
    else:
        lines.append("| Group | Baseline CRPS | Eval CRPS | Δ rel | Baseline 90/60 | Eval 90/60 | Δ 90 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for grp_label, grp_key in [
            ("All", "all_anchors"),
            ("Failure", "failure_anchors"),
            ("Control", "control_anchors"),
        ]:
            d = diff[grp_key]
            lines.append(
                f"| {grp_label} | {d['baseline_mean_crps']:.5f} | "
                f"{d['eval_mean_crps']:.5f} | "
                f"{d['delta_mean_crps_rel']*100:+.2f}% | "
                f"{d['baseline_mean_in_90']:.1f} | "
                f"{d['eval_mean_in_90']:.1f} | "
                f"{d['delta_mean_in_90']:+.1f} |"
            )
    lines.append("")

    lines.append("## Per-anchor")
    lines.append("")
    if diff is None:
        lines.append("| Anchor | Stratum | Realized | Mean CRPS | 50/60 | 90/60 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for sec, rows in payload["anchors_by_section"].items():
            for r in rows:
                lines.append(
                    f"| {r['anchor_date']} | {sec} | "
                    f"{r['realized_60d_return_pct']:+.1f}% | "
                    f"{r['mean_crps']:.5f} | "
                    f"{r['coverage']['in_50_band_days']} | "
                    f"{r['coverage']['in_90_band_days']} |"
                )
    else:
        lines.append("| Anchor | Stratum | Realized | Baseline CRPS | Eval CRPS | Δ rel | Baseline 90/60 | Eval 90/60 | Δ 90 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in diff["per_anchor"]:
            lines.append(
                f"| {r['anchor_date']} | {r['section']} | "
                f"{r['realized_60d_return_pct']:+.1f}% | "
                f"{r['baseline_mean_crps']:.5f} | "
                f"{r['eval_mean_crps']:.5f} | "
                f"{r['delta_mean_crps_rel']*100:+.2f}% | "
                f"{r['baseline_in_90']} | {r['eval_in_90']} | "
                f"{r['delta_in_90']:+d} |"
            )
    lines.append("")

    if diff is not None:
        lines.append("## Headline questions")
        lines.append("")
        n_fail_recovered = sum(
            1 for r in diff["per_anchor"]
            if r["anchor_date"] in FAILURE_DATES and r["eval_in_90"] >= 45
        )
        n_regressions = sum(
            1 for r in diff["per_anchor"] if r["delta_mean_crps_rel"] > 0.05
        )
        lines.append(
            f"- **V3.5 failures recovered (90-band ≥45/60)**: "
            f"**{n_fail_recovered}/5** "
            f"(promotion bar: ≥3)."
        )
        lines.append(
            f"- **Anchors regressing (CRPS up >5%)**: {n_regressions}/15 "
            f"(promotion bar: ≤2 without justification)."
        )
        lines.append(
            f"- **Failure mean CRPS Δ**: "
            f"{diff['failure_anchors']['delta_mean_crps_rel']*100:+.2f}%."
        )
        lines.append(
            f"- **Control mean CRPS Δ**: "
            f"{diff['control_anchors']['delta_mean_crps_rel']*100:+.2f}%."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="walk-forward run dir")
    p.add_argument("--label", required=True, help="experiment label (e.g. b1_local_linear)")
    p.add_argument("--baseline-json", default=None, help="optional baseline JSON for diff")
    p.add_argument(
        "--out-json", default=None,
        help="output JSON path (default: results/analog_mc/data/fat_tail_<label>.json)",
    )
    p.add_argument(
        "--out-md", default=None,
        help="output markdown path (default: docs/analog_mc/experiments/_<label>_fat_tail.md)",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    folds = load_fold_summaries(run_dir)
    anchors_payload = json.loads(ANCHORS_JSON.read_text())

    sections = ["positive", "negative", "regime_coverage"]
    by_section: dict[str, list[dict]] = {s: [] for s in sections}
    flat: list[dict] = []
    for section in sections:
        for entry in anchors_payload.get(section, []):
            fold = find_fold(entry["origin_idx"], folds)
            row = evaluate_anchor(entry, section, fold, run_dir)
            by_section[section].append(row)
            flat.append(row)

    failures = [r for r in flat if r["anchor_date"] in FAILURE_DATES]
    controls = [r for r in flat if r["anchor_date"] in CONTROL_DATES]

    payload = {
        "label": args.label,
        "canonical_run": str(run_dir),
        "n_anchors": len(flat),
        "horizon_days": flat[0]["coverage"]["horizon_days"],
        "aggregate": aggregate_metrics(flat),
        "failure_anchors_aggregate": aggregate_metrics(failures),
        "control_anchors_aggregate": aggregate_metrics(controls),
        "anchors_by_section": by_section,
    }

    out_json = Path(args.out_json) if args.out_json else \
        ROOT / "results" / "analog_mc" / "data" / f"fat_tail_{args.label}.json"
    out_md = Path(args.out_md) if args.out_md else \
        ROOT / "docs" / "analog_mc" / "experiments" / f"_{args.label}_fat_tail.md"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    diff = None
    if args.baseline_json:
        baseline_payload = json.loads(Path(args.baseline_json).read_text())
        diff = diff_payload(payload, baseline_payload)
        # Write the diff next to the eval JSON, with `_diff` suffix.
        diff_path = out_json.with_name(out_json.stem + "_diff.json")
        diff_path.write_text(json.dumps(diff, indent=2))
        print(f"diff JSON: {diff_path}")

    md = build_markdown(args.label, payload, diff)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(f"eval JSON: {out_json}")
    print(f"markdown:  {out_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
