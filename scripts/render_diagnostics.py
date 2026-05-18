"""Render every Stage 9 diagnostic for a given run dir, save figures to <run>/figs/.

Usage:
    uv run python scripts/render_diagnostics.py <run_dir> [--skip-fixed-baseline]

If <run_dir> is omitted, defaults to the most recent run under runs/analog_mc/.
``--skip-fixed-baseline`` skips the (1/3, 1/3, 1/3) re-eval, which re-runs the
whole walk-forward and is intractable on configs with conditional_block_sampling
turned on (~12 h re-eval). Use it for ablation runs where the cell-vs-cell
deltas are the comparison and the fixed-weight baseline isn't needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analog_mc import (
    Config,
    acf_comparison,
    aggregate_crps_overall,
    aggregate_crps_per_fold,
    aggregate_crps_per_step,
    aggregate_crps_per_vol_regime,
    clip_hit_summary,
    conditional_pit_by_vol_regime,
    crps_surface_plot,
    decision_rules,
    fixed_weight_baseline_crps,
    global_pit_histogram,
    load_returns,
    load_run,
    reliability_diagram,
    weight_trajectory_plot,
)


def _pick_run_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    root = Path("runs/analog_mc")
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "summary.parquet").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        sys.exit(f"no completed run found under {root}")
    return candidates[0]


def main() -> None:
    argv = sys.argv[1:]
    skip_fixed = "--skip-fixed-baseline" in argv
    argv = [a for a in argv if a != "--skip-fixed-baseline"]
    run_dir = _pick_run_dir(argv[0] if argv else None)
    print(f"== loading run from {run_dir}")
    run = load_run(run_dir)
    config = run.config

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    print(f"== loading returns from {config.data_path}")
    returns = load_returns(config)

    overall = aggregate_crps_overall(run)
    print(f"\n== aggregate OOS CRPS")
    print(f"  mean:   {overall['mean_crps']:.5f}")
    print(f"  median: {overall['median_crps']:.5f}")
    print(f"  N pairs: {overall['n_origin_step_pairs']:,}")

    print(f"\n== per-fold summary (first/last)")
    pf = aggregate_crps_per_fold(run)
    print(pf.head(3).to_string(index=False))
    print("...")
    print(pf.tail(3).to_string(index=False))

    print(f"\n== per-step mean CRPS")
    ps = aggregate_crps_per_step(run)
    print(f"  h=1   {ps['mean_crps'].iloc[0]:.5f}")
    print(f"  h={config.forecast_horizon//4}  {ps['mean_crps'].iloc[config.forecast_horizon//4 - 1]:.5f}")
    print(f"  h={config.forecast_horizon//2}  {ps['mean_crps'].iloc[config.forecast_horizon//2 - 1]:.5f}")
    print(f"  h={config.forecast_horizon}  {ps['mean_crps'].iloc[-1]:.5f}")

    print(f"\n== per-vol-regime CRPS")
    pr = aggregate_crps_per_vol_regime(run, returns)
    print(pr.to_string(index=False))

    plots = {
        "weight_trajectory.png": lambda: weight_trajectory_plot(run),
        "crps_surface_fold0.png": lambda: crps_surface_plot(run, fold_index=0),
        "global_pit.png": lambda: global_pit_histogram(run),
        "conditional_pit.png": lambda: conditional_pit_by_vol_regime(run, returns),
        "reliability.png": lambda: reliability_diagram(run),
        "acf_comparison.png": lambda: acf_comparison(run, returns),
        "clip_hit_summary.png": lambda: clip_hit_summary(run),
    }
    print("\n== rendering plots")
    for name, fn in plots.items():
        try:
            fig = fn()
            out = figs_dir / name
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")
        except Exception as exc:
            print(f"  FAILED {name}: {exc!r}")

    if skip_fixed:
        print("\n== fixed-weight baseline: SKIPPED (--skip-fixed-baseline)")
        baseline = None
    else:
        print(f"\n== fixed-weight baseline (1/3, 1/3, 1/3, n_eff=30) — this re-runs walk-forward eval")
        baseline = fixed_weight_baseline_crps(returns, config)
        print(f"  mean test CRPS (fixed):  {baseline['mean_test_crps']:.5f}")
        print(f"  mean test CRPS (tuned):  {pf['test_crps'].mean():.5f}")
        print(f"  delta: {(baseline['mean_test_crps'] - pf['test_crps'].mean()) / pf['test_crps'].mean() * 100:+.2f}%")

    print(f"\n== v2-trigger decision rules")
    rules = decision_rules(run, returns, fixed_baseline=baseline)
    for name, body in rules.items():
        icon = "FIRED" if body["fired"] else "ok   "
        print(f"  [{icon}] {name:35s}  metric={body['metric']:+.4f}")
        if body["fired"] and body["recommendation"]:
            print(f"            → {body['recommendation']}")

    # Save a tidy JSON summary for the dashboard / future scripts.
    report_path = run_dir / "diagnostic_report.json"
    report = {
        "overall": overall,
        "fixed_baseline": baseline,
        "decision_rules": {k: {**v, "fired": bool(v["fired"]), "metric": float(v["metric"])} for k, v in rules.items()},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n== wrote {report_path}")


if __name__ == "__main__":
    main()
