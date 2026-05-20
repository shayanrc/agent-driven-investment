"""A2.1 — fast sanity check at the 5 V3.5 failure anchors.

Per-anchor comparison: at each failure anchor, run the v2.4 forecast
(matcher_distance='weighted_euclidean', fold-selected weights/n_eff) vs
A2.1 (matcher_distance='corrwindow', same n_eff but weights ignored).

Search-time effect NOT included. This is the sign-of-life + magnitude
check that gates whether A2.1 canonical is worth running.

We sweep `corrwindow_length ∈ {10, 20, 60, 100}` to see whether window
choice matters at the failure anchors.

Writes:
  - results/analog_mc/data/v4_a2_corrwindow_sanity.json
  - docs/analog_mc/experiments/_a2_corrwindow_sanity_v0.md
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.features import compute_features
from analog_mc.scoring import crps_per_step
from analog_mc.simulate import forecast

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "v4_a2_corrwindow_sanity.json"
OUT_MD = ROOT / "docs" / "analog_mc" / "experiments" / "_a2_corrwindow_sanity_v0.md"

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

WINDOW_LENGTHS = [10, 20, 60, 100]


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


def coverage(paths: np.ndarray, realized: np.ndarray) -> tuple[int, int]:
    cum_paths = np.cumsum(paths, axis=1)
    cum_realized = np.cumsum(realized)
    q5 = np.quantile(cum_paths, 0.05, axis=0)
    q25 = np.quantile(cum_paths, 0.25, axis=0)
    q75 = np.quantile(cum_paths, 0.75, axis=0)
    q95 = np.quantile(cum_paths, 0.95, axis=0)
    in_50 = int(((cum_realized >= q25) & (cum_realized <= q75)).sum())
    in_90 = int(((cum_realized >= q5) & (cum_realized <= q95)).sum())
    return in_50, in_90


def analyze(
    anchor_date: str, label: str, entry: dict, fold: dict, cfg_baseline: Config,
    returns_arr: np.ndarray, features, realized: np.ndarray,
) -> dict:
    origin_idx = entry["origin_idx"]
    weights = np.array(fold["weights"], dtype=np.float64)
    n_eff = float(fold["n_eff"])
    train_end = fold["train_end"]
    candidate_idx = np.arange(0, train_end + 1, dtype=np.int64)

    # v2.4 reproduction (weighted_euclidean, conditional sampling, drift on)
    rng_v24 = np.random.default_rng(cfg_baseline.random_seed + origin_idx)
    paths_v24 = forecast(
        origin_idx=origin_idx, returns=returns_arr, candidate_idx=candidate_idx,
        features=features, weights=weights, n_eff=n_eff,
        config=cfg_baseline, rng=rng_v24,
    )
    crps_v24 = float(crps_per_step(paths_v24, realized).mean())
    in50_v24, in90_v24 = coverage(paths_v24, realized)

    # A2.1: corrwindow at each window length.
    per_window = {}
    for L in WINDOW_LENGTHS:
        cfg_a2 = replace(
            cfg_baseline, matcher_distance="corrwindow", corrwindow_length=L,
        )
        rng_a2 = np.random.default_rng(cfg_baseline.random_seed + origin_idx)
        paths_a2 = forecast(
            origin_idx=origin_idx, returns=returns_arr, candidate_idx=candidate_idx,
            features=features, weights=weights, n_eff=n_eff,
            config=cfg_a2, rng=rng_a2,
        )
        crps_a2 = float(crps_per_step(paths_a2, realized).mean())
        in50, in90 = coverage(paths_a2, realized)
        per_window[L] = {
            "mean_crps": crps_a2,
            "in_50": in50,
            "in_90": in90,
            "delta_crps_rel": (crps_a2 - crps_v24) / max(crps_v24, 1e-9),
            "delta_in_90": in90 - in90_v24,
        }

    return {
        "label": label,
        "anchor_date": anchor_date,
        "origin_idx": origin_idx,
        "fold_index": fold["fold_index"],
        "weights": weights.tolist(),
        "n_eff": n_eff,
        "realized_60d_pct": entry["realized_60d_return_pct"],
        "v24": {
            "mean_crps": crps_v24,
            "in_50": in50_v24,
            "in_90": in90_v24,
        },
        "a2_by_window_length": per_window,
    }


def main() -> None:
    cfg = Config.from_yaml(str(RUN_DIR / "config.yaml"))
    returns_series = load_returns(cfg)
    returns_arr = returns_series.to_numpy()
    features = compute_features(
        returns_series,
        halflife=cfg.ewma_halflife,
        horizons=tuple(cfg.zscore_horizons),
        momentum_lookback=cfg.momentum_lookback,
    )
    folds = load_fold_summaries()
    anchors = load_anchors()

    results = []
    for label, dates in [("failure", FAILURE_DATES), ("control", CONTROL_DATES)]:
        for d in dates:
            entry = anchors[d]
            fold = find_fold(entry["origin_idx"], folds)
            npz = np.load(RUN_DIR / "folds" / str(fold["fold_index"]) / "forecasts.npz")
            pos = int(np.where(npz["origin_idx"] == entry["origin_idx"])[0][0])
            realized = npz["realized"][pos]
            r = analyze(d, label, entry, fold, cfg, returns_arr, features, realized)
            results.append(r)

    # Aggregate per window length.
    agg = {}
    for L in WINDOW_LENGTHS:
        failures = [r for r in results if r["label"] == "failure"]
        controls = [r for r in results if r["label"] == "control"]
        agg[L] = {
            "failure_mean_crps_v24": float(np.mean([r["v24"]["mean_crps"] for r in failures])),
            "failure_mean_crps_a2": float(np.mean([r["a2_by_window_length"][L]["mean_crps"] for r in failures])),
            "failure_mean_in_90_v24": float(np.mean([r["v24"]["in_90"] for r in failures])),
            "failure_mean_in_90_a2": float(np.mean([r["a2_by_window_length"][L]["in_90"] for r in failures])),
            "control_mean_crps_v24": float(np.mean([r["v24"]["mean_crps"] for r in controls])),
            "control_mean_crps_a2": float(np.mean([r["a2_by_window_length"][L]["mean_crps"] for r in controls])),
            "control_mean_in_90_v24": float(np.mean([r["v24"]["in_90"] for r in controls])),
            "control_mean_in_90_a2": float(np.mean([r["a2_by_window_length"][L]["in_90"] for r in controls])),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"anchors": results, "aggregate_by_L": agg}, indent=2))

    lines: list[str] = []
    lines.append("# A2.1 corrwindow sanity v0 — per-failure isolated comparison")
    lines.append("")
    lines.append(
        "Per-anchor v2.4 (weighted_euclidean) vs A2.1 (corrwindow) sweep over "
        "window_length ∈ {10, 20, 60, 100}. Search-time effect not included; "
        "fold-selected n_eff is reused, weights are ignored under corrwindow."
    )
    lines.append("")
    lines.append("**Important caveat**: A2.1 disables conditional block sampling "
                 "(see A2.1 design §3), while v2.4 here uses it. So we're "
                 "comparing two distance functions *and* the conditional-sampling "
                 "regime simultaneously.")
    lines.append("")

    lines.append("## Aggregate by window length")
    lines.append("")
    lines.append("| L | Failure CRPS Δ | Failure 90/60 Δ | Control CRPS Δ | Control 90/60 Δ |")
    lines.append("|---:|---:|---:|---:|---:|")
    for L in WINDOW_LENGTHS:
        a = agg[L]
        fcrps_d = (a["failure_mean_crps_a2"] - a["failure_mean_crps_v24"]) / max(a["failure_mean_crps_v24"], 1e-9) * 100
        ccrps_d = (a["control_mean_crps_a2"] - a["control_mean_crps_v24"]) / max(a["control_mean_crps_v24"], 1e-9) * 100
        f90_d = a["failure_mean_in_90_a2"] - a["failure_mean_in_90_v24"]
        c90_d = a["control_mean_in_90_a2"] - a["control_mean_in_90_v24"]
        lines.append(
            f"| {L} | {fcrps_d:+.2f}% | {f90_d:+.1f} | "
            f"{ccrps_d:+.2f}% | {c90_d:+.1f} |"
        )
    lines.append("")

    lines.append("## Per-anchor 90-band coverage")
    lines.append("")
    cols = " | ".join([f"L={L}" for L in WINDOW_LENGTHS])
    lines.append(f"| Anchor | Group | Realized | v2.4 | {cols} |")
    sep = " | ".join(["---:"] * (len(WINDOW_LENGTHS) + 4))
    lines.append(f"|---|---|---:|{sep}|")
    for r in results:
        windows_str = " | ".join(
            f"{r['a2_by_window_length'][L]['in_90']}" for L in WINDOW_LENGTHS
        )
        lines.append(
            f"| {r['anchor_date']} | {r['label']} | "
            f"{r['realized_60d_pct']:+.1f}% | {r['v24']['in_90']} | "
            f"{windows_str} |"
        )
    lines.append("")

    lines.append("## Per-anchor CRPS")
    lines.append("")
    lines.append(f"| Anchor | Group | v2.4 | {cols} |")
    lines.append(f"|---|---|---:|{sep[: -len('| ---:')]}|")
    for r in results:
        windows_str = " | ".join(
            f"{r['a2_by_window_length'][L]['mean_crps']:.4f}" for L in WINDOW_LENGTHS
        )
        lines.append(
            f"| {r['anchor_date']} | {r['label']} | "
            f"{r['v24']['mean_crps']:.4f} | {windows_str} |"
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    # Pick best L by failure CRPS reduction
    best_L = min(WINDOW_LENGTHS, key=lambda L: agg[L]["failure_mean_crps_a2"])
    a = agg[best_L]
    fcrps_d = (a["failure_mean_crps_a2"] - a["failure_mean_crps_v24"]) / max(a["failure_mean_crps_v24"], 1e-9) * 100
    ccrps_d = (a["control_mean_crps_a2"] - a["control_mean_crps_v24"]) / max(a["control_mean_crps_v24"], 1e-9) * 100
    lines.append(f"- **Best window length on failures**: L={best_L}.")
    lines.append(f"- Failure CRPS Δ at L={best_L}: **{fcrps_d:+.2f}%**.")
    lines.append(f"- Control CRPS Δ at L={best_L}: **{ccrps_d:+.2f}%**.")
    lines.append("")
    if fcrps_d < -3.0:
        lines.append(
            "**Material failure improvement.** Worth launching A2.1 canonical "
            f"with L={best_L}, then comparing the joint search-time effect. "
            "Re-evaluate priority in V4_EXPERIMENTS_PLAN."
        )
    elif fcrps_d < 0:
        lines.append(
            "**Small failure improvement.** A2.1 helps but only marginally — "
            "may be that conditional sampling (disabled under corrwindow) was "
            "contributing more than expected. Consider an A2.1+conditional "
            "variant before canonical."
        )
    else:
        lines.append(
            "**No improvement.** Corrwindow distance alone doesn't help; the "
            "matcher-distance lever may be less load-bearing than V3.5.4 "
            "implied, or the disabling of conditional sampling under "
            "corrwindow is dominating. Keep A2.1 as a deprioritized check; "
            "focus on B1 + B-class variants."
        )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
