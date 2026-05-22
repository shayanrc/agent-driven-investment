"""B1 — fast sanity check at the 5 V3.5 failure anchors.

Per-anchor comparison: at each failure anchor's fold/origin, re-run the v2.4
forecast both with and without B1 (using the canonical fold's selected
weights/n_eff). Reports:

  - B1 correction magnitude per anchor (raw + per-day)
  - Mean CRPS change (B1 − v2.4)
  - 50/90 day-count coverage change
  - Terminal band shift

Isolates the correction's effect (search-time effect is NOT included — that
needs a full walk-forward, deferred to the next step).

Writes:
  - results/analog_mc/data/v4_b1_sanity.json
  - docs/analog_mc/experiments/_b1_sanity_v0.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.features import compute_features
from analog_mc.local_linear import (
    fit_local_linear_correction,
    forward_logret_sums,
)
from analog_mc.scoring import crps_per_step
from analog_mc.simulate import eligible_candidates, forecast
from analog_mc.distances import composite_distance, distances_to_probs

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "v4_b1_sanity.json"
OUT_MD = ROOT / "docs" / "analog_mc" / "experiments" / "_b1_sanity_v0.md"

FAILURE_DATES = [
    "2010-04-23",
    "2001-10-02",
    "2018-10-08",
    "2020-03-16",
    "2026-02-19",
]
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


def load_fold_summaries() -> list[dict]:
    return [
        json.loads((RUN_DIR / "folds" / d.name / "summary.json").read_text())
        for d in sorted((RUN_DIR / "folds").iterdir(), key=lambda p: int(p.name))
    ]


def find_fold(origin_idx: int, folds: list[dict]) -> dict:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    raise SystemExit(f"no fold for origin_idx={origin_idx}")


def coverage_counts(
    paths: np.ndarray, realized: np.ndarray
) -> tuple[int, int, float, float]:
    cum_paths = np.cumsum(paths, axis=1)
    cum_realized = np.cumsum(realized)
    q5 = np.quantile(cum_paths, 0.05, axis=0)
    q25 = np.quantile(cum_paths, 0.25, axis=0)
    q75 = np.quantile(cum_paths, 0.75, axis=0)
    q95 = np.quantile(cum_paths, 0.95, axis=0)
    in_50 = int(((cum_realized >= q25) & (cum_realized <= q75)).sum())
    in_90 = int(((cum_realized >= q5) & (cum_realized <= q95)).sum())
    width_50 = float(np.exp(q75[-1]) - np.exp(q25[-1])) * 100.0
    width_90 = float(np.exp(q95[-1]) - np.exp(q5[-1])) * 100.0
    return in_50, in_90, width_50, width_90


def analyze(
    anchor_date: str, label: str, entry: dict, fold: dict, cfg_off: Config,
    returns_arr: np.ndarray, features, fold_realized: np.ndarray,
) -> dict:
    origin_idx = entry["origin_idx"]
    weights = np.array(fold["weights"], dtype=np.float64)
    n_eff = float(fold["n_eff"])
    train_end = fold["train_end"]
    candidate_idx = np.arange(0, train_end + 1, dtype=np.int64)

    # ---- B1 OFF (v2.4 reproduction) ----
    rng_off = np.random.default_rng(cfg_off.random_seed + origin_idx)
    paths_off = forecast(
        origin_idx=origin_idx,
        returns=returns_arr,
        candidate_idx=candidate_idx,
        features=features,
        weights=weights,
        n_eff=n_eff,
        config=cfg_off,
        rng=rng_off,
    )

    # ---- B1 ON ----
    cfg_on = replace(cfg_off, local_linear_correction=True)
    rng_on = np.random.default_rng(cfg_off.random_seed + origin_idx)
    paths_on = forecast(
        origin_idx=origin_idx,
        returns=returns_arr,
        candidate_idx=candidate_idx,
        features=features,
        weights=weights,
        n_eff=n_eff,
        config=cfg_on,
        rng=rng_on,
    )

    # ---- B1 correction in isolation (for the report) ----
    eligible = eligible_candidates(candidate_idx, features, origin_idx, cfg_off)
    z_cols = [f"zscore_{h}" for h in cfg_off.zscore_horizons]
    z_target = features[z_cols].iloc[origin_idx].to_numpy()
    z_cand = features[z_cols].iloc[eligible].to_numpy()
    distances = composite_distance(z_target, z_cand, weights)
    target_n_eff = min(n_eff, float(eligible.size))
    probs = distances_to_probs(distances, target_n_eff=target_n_eff)
    forward_all = forward_logret_sums(returns_arr, cfg_off.forecast_horizon)
    correction, diag = fit_local_linear_correction(
        z_target, z_cand, probs, forward_all[eligible]
    )

    # Coverage / CRPS comparison
    in50_off, in90_off, w50_off, w90_off = coverage_counts(paths_off, fold_realized)
    in50_on, in90_on, w50_on, w90_on = coverage_counts(paths_on, fold_realized)
    crps_off = float(crps_per_step(paths_off, fold_realized).mean())
    crps_on = float(crps_per_step(paths_on, fold_realized).mean())

    return {
        "label": label,
        "anchor_date": anchor_date,
        "origin_idx": origin_idx,
        "fold_index": fold["fold_index"],
        "weights": weights.tolist(),
        "n_eff": n_eff,
        "realized_60d_pct": entry["realized_60d_return_pct"],
        "b1_correction_raw_logret": float(correction),
        "b1_correction_per_day": float(correction / cfg_off.forecast_horizon),
        "b1_correction_pct_terminal": float(np.exp(correction) - 1.0) * 100.0,
        "b1_matcher_mean": diag.matcher_mean,
        "b1_predicted_mean": diag.predicted_mean,
        "b1_clamp_hit": diag.clamp_hit,
        "b1_beta_norm": diag.beta_norm,
        "v24": {
            "mean_crps": crps_off,
            "in_50": in50_off,
            "in_90": in90_off,
            "terminal_50_width_pct": w50_off,
            "terminal_90_width_pct": w90_off,
        },
        "b1": {
            "mean_crps": crps_on,
            "in_50": in50_on,
            "in_90": in90_on,
            "terminal_50_width_pct": w50_on,
            "terminal_90_width_pct": w90_on,
        },
        "delta": {
            "mean_crps": crps_on - crps_off,
            "mean_crps_rel": (crps_on - crps_off) / max(crps_off, 1e-9),
            "in_50_days": in50_on - in50_off,
            "in_90_days": in90_on - in90_off,
        },
    }


def main() -> None:
    cfg_off = Config.from_yaml(str(RUN_DIR / "config.yaml"))
    returns_series = load_returns(cfg_off)
    returns_arr = returns_series.to_numpy()
    features = compute_features(
        returns_series,
        halflife=cfg_off.ewma_halflife,
        horizons=tuple(cfg_off.zscore_horizons),
        momentum_lookback=cfg_off.momentum_lookback,
    )
    folds = load_fold_summaries()
    anchors = load_anchors()

    results = []
    for label, dates in [("failure", FAILURE_DATES), ("control", CONTROL_DATES)]:
        for d in dates:
            entry = anchors[d]
            fold = find_fold(entry["origin_idx"], folds)

            # Pull realized from the fold's npz (the canonical realized log-returns).
            npz = np.load(RUN_DIR / "folds" / str(fold["fold_index"]) / "forecasts.npz")
            pos = int(np.where(npz["origin_idx"] == entry["origin_idx"])[0][0])
            realized = npz["realized"][pos]

            r = analyze(
                d, label, entry, fold, cfg_off,
                returns_arr, features, realized,
            )
            results.append(r)

    # Aggregate stats
    failures = [r for r in results if r["label"] == "failure"]
    controls = [r for r in results if r["label"] == "control"]
    agg = {
        "failure_mean_crps_v24": float(np.mean([r["v24"]["mean_crps"] for r in failures])),
        "failure_mean_crps_b1": float(np.mean([r["b1"]["mean_crps"] for r in failures])),
        "failure_mean_in_90_v24": float(np.mean([r["v24"]["in_90"] for r in failures])),
        "failure_mean_in_90_b1": float(np.mean([r["b1"]["in_90"] for r in failures])),
        "control_mean_crps_v24": float(np.mean([r["v24"]["mean_crps"] for r in controls])),
        "control_mean_crps_b1": float(np.mean([r["b1"]["mean_crps"] for r in controls])),
        "control_mean_in_90_v24": float(np.mean([r["v24"]["in_90"] for r in controls])),
        "control_mean_in_90_b1": float(np.mean([r["b1"]["in_90"] for r in controls])),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"anchors": results, "aggregate": agg}, indent=2))

    # Markdown report
    lines: list[str] = []
    lines.append("# B1 sanity v0 — per-failure isolated comparison")
    lines.append("")
    lines.append(
        "Direct per-anchor comparison of B1-on vs B1-off, holding everything "
        "else fixed (same fold-selected weights/n_eff from the canonical v2.4 "
        "run). Search-time effect is NOT included — this is the isolated "
        "correction-magnitude diagnostic from step 7 of the B1 build order."
    )
    lines.append("")
    lines.append(
        f"Canonical run: `{RUN_DIR.relative_to(ROOT)}` · 5 failure + 5 control "
        "anchors from V3.5_RESULTS."
    )
    lines.append("")

    lines.append("## B1 correction magnitudes")
    lines.append("")
    lines.append(
        "| Anchor | Group | Realized | Matcher E[60d] | B1 pred | Correction | Per-day | Clamp |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in results:
        clamp = "**clamp**" if r["b1_clamp_hit"] else "—"
        lines.append(
            f"| {r['anchor_date']} | {r['label']} | "
            f"{r['realized_60d_pct']:+.1f}% | "
            f"{(np.exp(r['b1_matcher_mean'])-1)*100:+.1f}% | "
            f"{(np.exp(r['b1_predicted_mean'])-1)*100:+.1f}% | "
            f"{r['b1_correction_pct_terminal']:+.2f}% | "
            f"{r['b1_correction_per_day']*1e4:+.2f}bp | "
            f"{clamp} |"
        )
    lines.append("")

    lines.append("## Coverage and CRPS deltas")
    lines.append("")
    lines.append(
        "| Anchor | Group | v2.4 CRPS | B1 CRPS | Δ CRPS | Δ CRPS rel | v24 90/60 | B1 90/60 | Δ 90 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['anchor_date']} | {r['label']} | "
            f"{r['v24']['mean_crps']:.5f} | {r['b1']['mean_crps']:.5f} | "
            f"{r['delta']['mean_crps']:+.5f} | "
            f"{r['delta']['mean_crps_rel']*100:+.2f}% | "
            f"{r['v24']['in_90']} | {r['b1']['in_90']} | "
            f"{r['delta']['in_90_days']:+d} |"
        )
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        "| Group | v2.4 mean CRPS | B1 mean CRPS | Δ rel | v2.4 mean 90/60 | B1 mean 90/60 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    fcrps_delta = (agg["failure_mean_crps_b1"] - agg["failure_mean_crps_v24"]) / max(agg["failure_mean_crps_v24"], 1e-9)
    ccrps_delta = (agg["control_mean_crps_b1"] - agg["control_mean_crps_v24"]) / max(agg["control_mean_crps_v24"], 1e-9)
    lines.append(
        f"| Failure (5) | {agg['failure_mean_crps_v24']:.5f} | "
        f"{agg['failure_mean_crps_b1']:.5f} | "
        f"{fcrps_delta*100:+.2f}% | {agg['failure_mean_in_90_v24']:.1f} | "
        f"{agg['failure_mean_in_90_b1']:.1f} |"
    )
    lines.append(
        f"| Control (5) | {agg['control_mean_crps_v24']:.5f} | "
        f"{agg['control_mean_crps_b1']:.5f} | "
        f"{ccrps_delta*100:+.2f}% | {agg['control_mean_in_90_v24']:.1f} | "
        f"{agg['control_mean_in_90_b1']:.1f} |"
    )
    lines.append("")

    lines.append("## Sanity verdict")
    lines.append("")
    n_clamps = sum(1 for r in results if r["b1_clamp_hit"])
    # Direction-of-correction sanity: does the correction sign agree with the
    # realized-minus-matcher-mean residual? If matcher under-shoots upside, the
    # correction should be positive.
    sign_agreements = sum(
        1 for r in results
        if np.sign(r["realized_60d_pct"]) * np.sign(r["b1_correction_pct_terminal"]) >= 0
        and abs(r["b1_correction_pct_terminal"]) > 0.01
    )
    failure_sign_agreements = sum(
        1 for r in failures
        if np.sign(r["realized_60d_pct"]) * np.sign(r["b1_correction_pct_terminal"]) >= 0
        and abs(r["b1_correction_pct_terminal"]) > 0.01
    )
    lines.append(f"- **Clamps fired**: {n_clamps}/10 anchors.")
    lines.append(
        f"- **Correction sign matches realized direction**: "
        f"{sign_agreements}/10 anchors total, "
        f"{failure_sign_agreements}/5 failures."
    )
    lines.append(
        f"- **Failure CRPS change (isolated)**: {fcrps_delta*100:+.2f}%."
    )
    lines.append(
        f"- **Control CRPS change (isolated)**: {ccrps_delta*100:+.2f}%."
    )
    lines.append("")
    lines.append(
        "**Caveat.** This is the *isolated* effect — the matcher's weights/n_eff "
        "are held at v2.4-selected values. The full canonical B1 run "
        "re-searches weights with B1 active (decision D10), which may strengthen "
        "or weaken these numbers. This sanity output is a sign-of-life + "
        "magnitude check, not a promotion decision."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
