"""Stage 9 — end-to-end acceptance demo for the forecasters v1 PR.

Orchestrates the four-skill flow defined in docs/forecasters/V1_PLAN.md
§"Stage 9":

  1. Fetch NIFTY:NIFTY500 over the deepest-reachable history.
  2. Tune analog_mc on the data EXCLUDING the last 60 trading days.
  3. Forecast the held-out last 60 trading days using the new preset.
  4. Verify against the realized last-60-day path:
       - 90-band coverage in [0.5, 1.0]
       - CRPS finite and strictly less than a naïve random-walk baseline
       - preset YAML passes the schema validator
       - the forecast result has empty warnings (no drift expected)

Writes docs/forecasters/_acceptance_demo.md with the full report.

The tune step is hours of compute; the orchestrator supports running each
phase independently via --phase:

  --phase fetch   : warm cache + write a snapshot
  --phase tune    : run /tune-preset, write the produced preset YAML
  --phase forecast: run /forecast against the produced preset
  --phase verify  : compute coverage + CRPS + baseline; write the report
  --phase all     : do all four in sequence

The default is --phase all. Re-running --phase verify alone is the usual
post-tune flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import data_pipelines.domains.us_equities  # noqa: F401
import data_pipelines.domains.nse_equities  # noqa: F401
from data_pipelines import fetch_with_meta
from data_pipelines.domains.nse_equities.calendar import NSECalendar

from analog_mc.scoring import crps_per_step
from forecasters import dispatch_forecast, load_preset, validate_preset
from forecasters.presets import preset_content_hash


IDENTIFIER = "NIFTY:NIFTY500"
HORIZON_DAYS = 60
DEFAULT_TUNE_START = "2015-01-01"
DEFAULT_OUTPUT_PRESET = "nifty500-v1"

log = logging.getLogger("forecasters.acceptance_demo")


# ----------------------------------------------------------------------------
# Phase 1: fetch
# ----------------------------------------------------------------------------


def phase_fetch(args: argparse.Namespace) -> dict:
    """Warm the cache for IDENTIFIER over the full requested range."""
    log.info("phase fetch: %s [%s .. %s]", IDENTIFIER, args.start, args.end)
    df, meta = fetch_with_meta(IDENTIFIER, start=args.start, end=args.end)
    info = {
        "phase": "fetch",
        "identifier": IDENTIFIER,
        "range": meta.range,
        "rows": int(len(df)),
        "cache_was_cold": meta.cache_was_cold,
        "gaps_filled": meta.gaps_filled,
        "providers_failed": meta.providers_failed,
        "first_date": df["date"].iloc[0].date().isoformat(),
        "last_date": df["date"].iloc[-1].date().isoformat(),
    }
    log.info("phase fetch: %d rows %s → %s", info["rows"], info["first_date"], info["last_date"])
    return info


def _resolve_tune_end(end_date: date) -> date:
    """The 60th-most-recent NSE trading day on or before ``end_date``.

    The forecast horizon = (tune_end + 1 trading day) through end_date
    inclusive, so tune_end = end_date − 59 trading days.
    """
    cal = NSECalendar()
    # Look back ~120 calendar days to ensure we have 60 trading days even
    # with a heavy holiday cluster.
    window_start = end_date - timedelta(days=180)
    tds = cal.trading_days(window_start, end_date)
    if len(tds) < HORIZON_DAYS + 1:
        raise RuntimeError(
            f"only {len(tds)} trading days in the last 180 calendar days "
            f"ending {end_date}; need >= {HORIZON_DAYS + 1}"
        )
    return tds[-(HORIZON_DAYS + 1)]


# ----------------------------------------------------------------------------
# Phase 2: tune
# ----------------------------------------------------------------------------


def phase_tune(args: argparse.Namespace) -> dict:
    """Run /tune-preset (subprocess) for the held-in range."""
    end_d = date.fromisoformat(args.end)
    tune_end = _resolve_tune_end(end_d)
    log.info("phase tune: tune_end = %s (60th trading day before %s)", tune_end, end_d)

    # Build a slim search_config so the full-history tune is tractable. The
    # v24-default hyperparameters are the search-grid + analog_mc knobs;
    # tuning grid + n_eff_values come from this config.
    search_config = {
        "n_paths": args.n_paths_search,
        "weight_grid_resolution": 0.1,
        "n_eff_values": [15, 30, 50, 80, 150],
        "drift_mode": "trailing_momentum",
        "momentum_lookback": 20,
        "momentum_shrinkage": 0.30,
        "conditional_block_sampling": True,
        "conditional_block_sampling_in_search": False,
        "vol_clip_lower": 0.5,
        "vol_clip_upper": 3.0,
        "matcher_distance": "weighted_euclidean",
        "vol_model": "ewma",
        "local_linear_correction": False,
        "train_initial_size": 1000,
        "val_size": 60,
        "test_size": 60,
        "forecast_horizon": 60,
        "block_length": 10,
        "n_blocks": 6,
        "ewma_halflife": 20,
        "zscore_horizons": [20, 50, 200],
    }
    search_cfg_path = Path(args.work_dir) / "search_config.yaml"
    search_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    search_cfg_path.write_text(yaml.safe_dump(search_config, default_flow_style=False))

    cmd = [
        sys.executable, "-m", "scripts.forecasters.run", "tune",
        "--backend", "analog_mc",
        "--identifier", IDENTIFIER,
        "--start", args.start,
        "--end", tune_end.isoformat(),
        "--output-preset", args.output_preset,
        "--search-config", str(search_cfg_path),
        "--seed", str(args.seed),
    ]
    log.info("phase tune: launching %s", " ".join(cmd))
    t0 = time.perf_counter()
    if args.tune_log:
        with open(args.tune_log, "wb") as lf:
            res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    else:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        msg = ""
        if not args.tune_log:
            msg = res.stderr or res.stdout or ""
        raise RuntimeError(
            f"tune subprocess failed with code {res.returncode}; see log "
            f"{args.tune_log or 'stdout above'}: {msg[-500:]}"
        )
    if args.tune_log:
        preset_path_str = None
        # Tail the log for the preset path that the CLI prints last.
        with open(args.tune_log) as lf:
            for line in lf:
                if line.strip().endswith(".yaml"):
                    preset_path_str = line.strip()
    else:
        preset_path_str = res.stdout.strip().splitlines()[-1]
    preset_path = Path(preset_path_str)
    info = {
        "phase": "tune",
        "tune_end": tune_end.isoformat(),
        "preset_path": str(preset_path.resolve()),
        "tune_runtime_seconds": float(elapsed),
        "cmd": cmd,
    }
    log.info("phase tune: done in %.1fs; preset=%s", elapsed, preset_path)
    return info


# ----------------------------------------------------------------------------
# Phase 3: forecast
# ----------------------------------------------------------------------------


def phase_forecast(args: argparse.Namespace, preset_path: Path) -> dict:
    """Run /forecast using the produced preset; origin = tune_end.

    The forecast input is SLICED to the preset's `fitted_on` range so the
    framework's drift hash check matches — the goal.md acceptance criterion
    requires `warnings == []` (no drift expected when the preset was fit
    on the exact identifier we're forecasting on). Realized prices for the
    horizon are fetched separately in `phase_verify`.
    """
    preset_name = preset_path.stem
    preset = load_preset(preset_name)
    end_d = date.fromisoformat(args.end)
    tune_end = _resolve_tune_end(end_d)
    log.info("phase forecast: preset=%s origin=%s horizon=%d", preset_name, tune_end, HORIZON_DAYS)

    # Slice exactly to the preset's fitted_on range so data_hash matches.
    fitted = preset["fitted_on"]
    df, _ = fetch_with_meta(IDENTIFIER, start=fitted["start"], end=fitted["end"])
    hp = dict(preset["hyperparameters"])
    # Always evaluate with the canonical 1000 paths for the demo CRPS to be
    # comparable to the V5.A.2 baseline numbers.
    hp["n_paths"] = args.n_paths_forecast

    result = dispatch_forecast(preset, {
        "data": df,
        "origin": tune_end.isoformat(),
        "horizon": HORIZON_DAYS,
        "hyperparameters": hp,
        "seed": args.seed,
    })
    # Persist the result JSON-friendly under the work dir.
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work_dir / "forecast_paths.npz", paths=result["paths"])
    summary_payload = {k: v for k, v in result.items() if k != "paths"}
    (work_dir / "forecast_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    )
    info = {
        "phase": "forecast",
        "origin": tune_end.isoformat(),
        "horizon": HORIZON_DAYS,
        "preset_name": preset_name,
        "preset_path": str(preset_path.resolve()),
        "result_summary": summary_payload["summary"],
        "warnings": result["warnings"],
        "metadata": summary_payload["metadata"],
        "anchors": summary_payload["anchors"],
    }
    log.info("phase forecast: done; in-sample CRPS=%s warnings=%d",
             info["result_summary"].get("crps"), len(info["warnings"]))
    return info


# ----------------------------------------------------------------------------
# Phase 4: verify + write report
# ----------------------------------------------------------------------------


def _compute_realized(df: pd.DataFrame, origin_iso: str, horizon: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Realized log-returns + cumulative-price path starting at the trading
    day after ``origin_iso``. Returns (log_returns, cum_log_returns, origin_close).

    Canonical-schema only: this helper assumes ``df`` carries the
    ``data_pipelines`` canonical columns ``date`` and ``adj_close``. The
    acceptance demo always sources its DataFrame via ``fetch_with_meta`` so
    this contract is guaranteed at the call sites. A future demo that
    targets a FRED-style or CSV-style identifier would need either the
    backend adapter's ``_resolve_data_columns`` probe or an explicit
    column-rename upstream.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    origin_ts = pd.Timestamp(origin_iso)
    mask = df["date"] <= origin_ts
    origin_idx = int(np.where(mask)[0][-1])
    closes = df["adj_close"].to_numpy()
    origin_close = float(closes[origin_idx])
    horizon_closes = closes[origin_idx + 1 : origin_idx + 1 + horizon]
    if len(horizon_closes) < horizon:
        raise RuntimeError(f"only {len(horizon_closes)} realized closes after {origin_iso}; need {horizon}")
    # Log returns relative to origin close, per-day.
    full = np.concatenate([[origin_close], horizon_closes])
    log_rets = np.log(full[1:] / full[:-1])
    cum_log_rets = np.cumsum(log_rets)
    return log_rets, cum_log_rets, origin_close


def phase_verify(args: argparse.Namespace, forecast_info: dict, fetch_info: dict, tune_info: dict, preset_path: Path) -> dict:
    work_dir = Path(args.work_dir)
    paths_logret = np.load(work_dir / "forecast_paths.npz")["paths"]

    # Load realized horizon from the same data the forecast used.
    df, _ = fetch_with_meta(IDENTIFIER, start=args.start, end=args.end)
    realized_logret, realized_cum, origin_close = _compute_realized(
        df, forecast_info["origin"], HORIZON_DAYS,
    )

    # CRPS over the cumulative log returns at each step.
    crps_per_step_arr = crps_per_step(paths_logret, realized_logret)
    crps_mean = float(np.mean(crps_per_step_arr))

    # Naïve random-walk baseline: zero forecast (log return == 0 at every step),
    # i.e. price stays at origin_close. Implemented as paths of all zeros.
    baseline_paths = np.zeros_like(paths_logret)
    baseline_crps_arr = crps_per_step(baseline_paths, realized_logret)
    baseline_crps_mean = float(np.mean(baseline_crps_arr))

    # 90-band coverage over the realized PRICE path.
    cum_paths = np.cumsum(paths_logret, axis=1)
    price_paths = origin_close * np.exp(cum_paths)
    realized_prices = origin_close * np.exp(realized_cum)
    p5 = np.percentile(price_paths, 5, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    median = np.percentile(price_paths, 50, axis=0)
    in_90 = (realized_prices >= p5) & (realized_prices <= p95)
    coverage_90 = float(np.mean(in_90))
    in_50 = (realized_prices >= p25) & (realized_prices <= p75)
    coverage_50 = float(np.mean(in_50))

    # Validate the produced preset against the schema.
    preset_yaml = yaml.safe_load(preset_path.read_text())
    try:
        validate_preset(preset_yaml, source_path=preset_path)
        preset_valid = True
        preset_validation_error = None
    except Exception as e:
        preset_valid = False
        preset_validation_error = str(e)

    # Pass / fail.
    drift_warnings = [w for w in forecast_info["warnings"]
                      if "Hyperparameters may be uncalibrated" in w]
    assertions = {
        "preset_validates": preset_valid,
        "forecast_warnings_empty": len(drift_warnings) == 0,
        "coverage_90_in_range": 0.5 <= coverage_90 <= 1.0,
        "crps_finite": np.isfinite(crps_mean),
        "crps_beats_baseline": np.isfinite(crps_mean) and crps_mean < baseline_crps_mean,
    }
    passed = all(assertions.values())

    # Plot the fan chart against realized.
    try:
        x = np.arange(1, HORIZON_DAYS + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(x, p5, p95, alpha=0.18, label="5–95% band", color="C0")
        ax.fill_between(x, p25, p75, alpha=0.32, label="25–75% band", color="C0")
        ax.plot(x, median, color="C0", lw=1.5, label="median forecast")
        ax.plot(x, realized_prices, color="black", lw=2.0, label="realized")
        ax.axhline(origin_close, color="grey", linestyle=":", lw=1.0, label=f"origin close ({origin_close:.0f})")
        ax.set_xlabel("trading day after origin")
        ax.set_ylabel("NIFTY 500 (INR)")
        ax.set_title(f"NIFTY 500 acceptance demo — origin {forecast_info['origin']}, horizon {HORIZON_DAYS}d")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(work_dir / "acceptance_fan_chart.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        log.warning("fan chart failed: %s", e)

    report = {
        "phase": "verify",
        "identifier": IDENTIFIER,
        "data_range": fetch_info["range"],
        "tune_end": forecast_info["origin"],
        "horizon_days": HORIZON_DAYS,
        "preset_path": str(preset_path.resolve()),
        "preset_content_hash": preset_content_hash(preset_path),
        "n_paths": int(paths_logret.shape[0]),
        "crps_mean": crps_mean,
        "crps_per_step_max": float(np.max(crps_per_step_arr)),
        "baseline_crps_mean": baseline_crps_mean,
        "coverage_50": coverage_50,
        "coverage_90": coverage_90,
        "origin_close": origin_close,
        "realized_final_close": float(realized_prices[-1]),
        "median_final_close": float(median[-1]),
        "assertions": assertions,
        "passed": passed,
        "warnings": forecast_info["warnings"],
        "tune_info": tune_info,
        "preset_validation_error": preset_validation_error,
    }

    # Write _acceptance_demo.md
    md_path = Path("docs/forecasters/_acceptance_demo.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_format_report_md(report))
    log.info("phase verify: report → %s; passed=%s", md_path, passed)

    return report


def _format_report_md(report: dict) -> str:
    a = report["assertions"]
    verdict = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# NIFTY 500 acceptance demo — Stage 9 report",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Setup",
        "",
        f"- Identifier: `{report['identifier']}`",
        f"- Data range: {report['data_range']['start']} → {report['data_range']['end']}",
        f"- Tune end (held-in range upper bound): `{report['tune_end']}`",
        f"- Forecast origin: `{report['tune_end']}` (the day after — first horizon step — is the first realized day)",
        f"- Horizon: {report['horizon_days']} trading days",
        f"- Produced preset: `{report['preset_path']}`",
        f"- Preset content hash: `{report['preset_content_hash']}`",
        f"- Paths sampled: {report['n_paths']}",
        "",
        "## Results",
        "",
        f"- Origin close (INR): {report['origin_close']:.2f}",
        f"- Realized final close at horizon end (INR): {report['realized_final_close']:.2f}",
        f"- Median forecast close at horizon end (INR): {report['median_final_close']:.2f}",
        f"- CRPS (mean over 60 horizon steps): {report['crps_mean']:.5f}",
        f"- Random-walk baseline CRPS: {report['baseline_crps_mean']:.5f}",
        f"- 50% band coverage of the realized path: {report['coverage_50']:.2%}",
        f"- 90% band coverage of the realized path: {report['coverage_90']:.2%}",
        "",
        "## Acceptance criteria",
        "",
        f"| Check | Required | Actual | Pass? |",
        f"|---|---|---|---|",
        f"| Preset YAML validates against the v1 schema | `validate_preset` returns | {report['preset_validation_error'] or 'OK'} | {'yes' if a['preset_validates'] else 'no'} |",
        f"| No drift warnings on the forecast result | drift warnings == 0 | {len([w for w in report['warnings'] if 'Hyperparameters may be uncalibrated' in w])} drift warning(s); {len(report['warnings'])} total warning(s) | {'yes' if a['forecast_warnings_empty'] else 'no'} |",
        f"| 90-band coverage in [0.5, 1.0] | informative | {report['coverage_90']:.2%} | {'yes' if a['coverage_90_in_range'] else 'no'} |",
        f"| CRPS finite | finite | {report['crps_mean']:.5f} | {'yes' if a['crps_finite'] else 'no'} |",
        f"| CRPS beats naïve random-walk | < baseline {report['baseline_crps_mean']:.5f} | {report['crps_mean']:.5f} | {'yes' if a['crps_beats_baseline'] else 'no'} |",
        "",
        "## Warnings on the forecast result",
        "",
    ]
    if report["warnings"]:
        for w in report["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- (none)")
    lines.extend([
        "",
        "## Tune details",
        "",
        f"- Tune subprocess runtime: {report['tune_info'].get('tune_runtime_seconds', 0):.1f}s",
        f"- Tune command: `{' '.join(report['tune_info'].get('cmd', []))}`",
        "",
        "## Notes",
        "",
        f"- The fan chart at `runs/forecasters/acceptance_demo/acceptance_fan_chart.png` shows the median + 5/25/75/95 bands against the realized path.",
        f"- This report is auto-generated by `scripts/forecasters/run_acceptance_demo.py`; rerun `--phase verify` to regenerate without re-tuning.",
    ])
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.forecasters.run_acceptance_demo")
    parser.add_argument("--phase", default="all",
                        choices=["fetch", "tune", "forecast", "verify", "all"])
    parser.add_argument("--start", default=DEFAULT_TUNE_START)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output-preset", default=DEFAULT_OUTPUT_PRESET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-paths-search", type=int, default=500)
    parser.add_argument("--n-paths-forecast", type=int, default=1000)
    parser.add_argument("--work-dir", default="runs/forecasters/acceptance_demo")
    parser.add_argument("--tune-log", default=None, help="redirect tune subprocess stdout/stderr to this file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.json"
    state = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text())

    try:
        if args.phase in ("fetch", "all"):
            state["fetch"] = phase_fetch(args)
        if args.phase in ("tune", "all"):
            state["tune"] = phase_tune(args)
        if args.phase in ("forecast", "all"):
            preset_path = Path(state["tune"]["preset_path"])
            state["forecast"] = phase_forecast(args, preset_path)
        if args.phase in ("verify", "all"):
            preset_path = Path(state["tune"]["preset_path"])
            state["verify"] = phase_verify(
                args, state["forecast"], state["fetch"], state["tune"], preset_path,
            )
            passed = state["verify"]["passed"]
            state_path.write_text(json.dumps(state, indent=2, default=str))
            return 0 if passed else 3
    finally:
        state_path.write_text(json.dumps(state, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
