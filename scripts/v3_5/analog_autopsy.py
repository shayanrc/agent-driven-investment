"""V3.5.4 — analog autopsy.

For each failure anchor, faithfully reproduces the v2.4 matcher's *first-block*
probability distribution over the eligible candidate pool, identifies the
top-20 highest-probability analogs, and tabulates their 60-day-forward returns.

Computes:
  - expected 60d return under the matcher's probability distribution
  - share of probability mass placed on analogs whose 60d-forward sign matches
    the realized sign
  - top-20 listing with date / 60d-forward return / matcher probability

For each anchor we also report the matcher's expected vs realized 60d return
delta — i.e., how much of the miss is the matcher's mean placement vs path
sampling variance.

Writes:
  - results/analog_mc/data/v3_5_4_analog_autopsy.json
  - docs/analog_mc/v3.5/_v3_5_4_analog_autopsy.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.distances import composite_distance, distances_to_probs
from analog_mc.features import compute_features
from analog_mc.simulate import eligible_candidates

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "v3_5_4_analog_autopsy.json"
OUT_MD = ROOT / "docs" / "analog_mc" / "v3.5" / "_v3_5_4_analog_autopsy.md"

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

TOP_K = 20


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


def analyze_anchor(
    anchor_date: str,
    label: str,
    entry: dict,
    fold: dict,
    cfg: Config,
    returns_series: pd.Series,
    features: pd.DataFrame,
    forward_pct: np.ndarray,
) -> dict:
    origin_idx = entry["origin_idx"]
    weights = np.array(fold["weights"], dtype=np.float64)
    n_eff = float(fold["n_eff"])
    train_end = fold["train_end"]
    candidate_idx = np.arange(0, train_end + 1, dtype=np.int64)
    eligible = eligible_candidates(candidate_idx, features, origin_idx, cfg)

    z_cols = [f"zscore_{h}" for h in cfg.zscore_horizons]
    z_target = features[z_cols].iloc[origin_idx].to_numpy()
    z_cand = features[z_cols].iloc[eligible].to_numpy()

    distances = composite_distance(z_target, z_cand, weights)
    target_n_eff = min(n_eff, float(eligible.size))
    probs = distances_to_probs(distances, target_n_eff=target_n_eff)

    # Forward return per eligible candidate
    fwd = forward_pct[eligible]  # may include NaN at the tail
    # Mask NaN forward returns (candidate too close to end of data — rare for train).
    fwd_valid = ~np.isnan(fwd)
    # Renormalize probs over valid forward returns
    if not fwd_valid.all():
        probs_valid = probs.copy()
        probs_valid[~fwd_valid] = 0.0
        probs_valid /= probs_valid.sum()
    else:
        probs_valid = probs

    realized = entry["realized_60d_return_pct"]
    sign_realized = np.sign(realized)

    # Expected 60d forward return under matcher
    fwd_for_mean = np.where(fwd_valid, fwd, 0.0)
    expected_60d = float(np.sum(probs_valid * fwd_for_mean))

    # Share of probability mass on analogs whose forward sign matches realized
    sign_match = (np.sign(fwd_for_mean) == sign_realized) & fwd_valid
    p_sign_match = float(probs_valid[sign_match].sum())

    # Share of probability mass on |fwd| >= |realized|
    big_enough = (np.abs(fwd_for_mean) >= abs(realized)) & fwd_valid
    p_big_enough = float(probs_valid[big_enough].sum())

    # Share of mass on +30% rallies, -30% drops
    p_rally_30 = float(probs_valid[(fwd_for_mean >= 30) & fwd_valid].sum())
    p_drop_30 = float(probs_valid[(fwd_for_mean <= -30) & fwd_valid].sum())

    # Top-K by probability
    order = np.argsort(probs)[::-1][:TOP_K]
    top = []
    for rank, j in enumerate(order, start=1):
        cand_i = int(eligible[j])
        cand_date = str(returns_series.index[cand_i].date())
        top.append({
            "rank": rank,
            "candidate_idx": cand_i,
            "candidate_date": cand_date,
            "prob": float(probs[j]),
            "distance": float(distances[j]),
            "forward_60d_pct": float(fwd[j]) if fwd_valid[j] else None,
        })

    # Cumulative probability concentration
    sorted_probs = np.sort(probs)[::-1]
    cum = np.cumsum(sorted_probs)

    return {
        "label": label,
        "anchor_date": anchor_date,
        "origin_idx": origin_idx,
        "fold_index": fold["fold_index"],
        "weights": weights.tolist(),
        "n_eff_requested": n_eff,
        "n_eff_used": target_n_eff,
        "pool_size": int(eligible.size),
        "realized_60d_pct": realized,
        "expected_60d_pct_matcher": expected_60d,
        "miss_pct_matcher_mean": realized - expected_60d,
        "prob_mass_sign_match": p_sign_match,
        "prob_mass_abs_ge_realized": p_big_enough,
        "prob_mass_rally_ge_30": p_rally_30,
        "prob_mass_drop_le_neg30": p_drop_30,
        "top_5_cum_prob": float(cum[4]) if cum.size >= 5 else float(cum[-1]),
        "top_20_cum_prob": float(cum[19]) if cum.size >= 20 else float(cum[-1]),
        "top_100_cum_prob": float(cum[99]) if cum.size >= 100 else float(cum[-1]),
        "top_analogs": top,
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

    h = cfg.forecast_horizon
    n = len(returns_arr)
    cumret = np.concatenate([[0.0], np.cumsum(returns_arr)])
    forward_logret = np.full(n, np.nan)
    valid_end = n - h - 1
    idx = np.arange(0, valid_end + 1)
    forward_logret[idx] = cumret[idx + h + 1] - cumret[idx + 1]
    forward_pct = (np.exp(forward_logret) - 1.0) * 100.0

    results = []
    for label, dates in [("failure", FAILURE_DATES), ("control", CONTROL_DATES)]:
        for d in dates:
            entry = anchors[d]
            fold = find_fold(entry["origin_idx"], folds)
            r = analyze_anchor(
                d, label, entry, fold, cfg,
                returns_series, features, forward_pct,
            )
            results.append(r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown
    lines: list[str] = []
    lines.append("# V3.5.4 — analog autopsy")
    lines.append("")
    lines.append(
        "For each anchor, reproduces the v2.4 matcher's first-block probability "
        "distribution exactly (same weights, n_eff, features, eligibility) and "
        "tabulates where the probability mass lands relative to the realized "
        "60-day move."
    )
    lines.append("")

    lines.append("## Headline: matcher mean placement vs realized")
    lines.append("")
    lines.append(
        "| Anchor | Group | Realized 60d | E[60d∣matcher] | Miss (real − E) | "
        "P(sign match) | P(|fwd|≥|realized|) | P(rally≥30%) | P(drop≤−30%) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['anchor_date']} | {r['label']} | "
            f"{r['realized_60d_pct']:+.1f}% | "
            f"{r['expected_60d_pct_matcher']:+.2f}% | "
            f"{r['miss_pct_matcher_mean']:+.1f}% | "
            f"{r['prob_mass_sign_match']*100:.1f}% | "
            f"{r['prob_mass_abs_ge_realized']*100:.1f}% | "
            f"{r['prob_mass_rally_ge_30']*100:.1f}% | "
            f"{r['prob_mass_drop_le_neg30']*100:.1f}% |"
        )
    lines.append("")

    lines.append("## Probability concentration")
    lines.append("")
    lines.append(
        "| Anchor | Pool | n_eff (used) | top-5 ∑p | top-20 ∑p | top-100 ∑p |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['anchor_date']} | {r['pool_size']} | "
            f"{r['n_eff_used']:.0f} | "
            f"{r['top_5_cum_prob']*100:.1f}% | "
            f"{r['top_20_cum_prob']*100:.1f}% | "
            f"{r['top_100_cum_prob']*100:.1f}% |"
        )
    lines.append("")

    # Per-anchor top-20 listings (failures only — controls go into JSON)
    lines.append("## Top-20 analogs per failure anchor")
    lines.append("")
    for r in results:
        if r["label"] != "failure":
            continue
        lines.append(
            f"### {r['anchor_date']} "
            f"(fold {r['fold_index']}, realized {r['realized_60d_pct']:+.1f}%, "
            f"weights {r['weights']}, n_eff={r['n_eff_used']:.0f})"
        )
        lines.append("")
        lines.append("| Rank | Candidate date | Prob | Distance | 60d-fwd return |")
        lines.append("|---:|---|---:|---:|---:|")
        for t in r["top_analogs"]:
            fwd = t["forward_60d_pct"]
            fwd_s = f"{fwd:+.1f}%" if fwd is not None else "—"
            lines.append(
                f"| {t['rank']} | {t['candidate_date']} | "
                f"{t['prob']*100:.2f}% | {t['distance']:.3f} | {fwd_s} |"
            )
        lines.append("")

    # Diagnostic pattern detection
    lines.append("## Pattern detection")
    lines.append("")
    lines.append(
        "Three patterns from the plan: (a) wrong-era matches, (b) right-era "
        "but no V-recoveries / right-era no drops, (c) bimodal distribution."
    )
    lines.append("")
    failures = [r for r in results if r["label"] == "failure"]
    sign_mismatch = [r for r in failures if r["prob_mass_sign_match"] < 0.5]
    has_some_extreme = [
        r for r in failures
        if (r["prob_mass_rally_ge_30"] + r["prob_mass_drop_le_neg30"]) > 0.05
    ]
    lines.append(
        f"- Failures where matcher places <50% mass on the realized sign: "
        f"**{len(sign_mismatch)}/5** "
        f"({', '.join(r['anchor_date'] for r in sign_mismatch)})"
    )
    lines.append(
        f"- Failures where matcher places >5% mass on the right tail: "
        f"**{len(has_some_extreme)}/5** "
        f"({', '.join(r['anchor_date'] for r in has_some_extreme)})"
    )
    lines.append("")

    # Verdict
    n_sign_miss = len(sign_mismatch)
    n_with_extreme = len(has_some_extreme)
    avg_top5 = float(np.mean([r["top_5_cum_prob"] for r in failures]))
    lines.append("## Verdict")
    lines.append("")
    if n_sign_miss >= 3 and n_with_extreme <= 1:
        verdict = (
            f"**Wrong-era matches dominant.** {n_sign_miss}/5 failures place "
            "majority mass on the wrong-sign forward distribution and "
            f"{5 - n_with_extreme}/5 place essentially no mass on the realized "
            "tail. The matcher is picking historically-similar-by-z-score "
            "candidates whose dynamics differ from the regime transition the "
            "anchor sits at. **Promote A2 (max-corr distance) to P0 — state "
            "representation needs reworking.** Confirms the matching-problem "
            "diagnosis from V3.5.2."
        )
    elif n_with_extreme >= 3:
        verdict = (
            f"**Bimodal analog distribution detected.** {n_with_extreme}/5 "
            "failures place ≥5% mass on the right tail, but the conditional "
            "mean still misses badly because majority mass sits on "
            "trend-continuation analogs. **B1 (Platzer local-linear) is "
            "well-targeted** — its Jacobian correction would up-weight the "
            "high-Lyapunov-tail mass."
        )
    else:
        verdict = (
            f"**Sign placement {n_sign_miss}/5 wrong, tail mass "
            f"{n_with_extreme}/5 present, top-5 concentration "
            f"{avg_top5*100:.1f}% on average.** Mixed pattern — see per-anchor "
            "tables. Recommend keeping current v4 sequencing and re-evaluating "
            "with the empirical analog distributions in hand."
        )
    lines.append(verdict)
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
