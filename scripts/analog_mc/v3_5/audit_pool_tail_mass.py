"""V3.5.2 — candidate pool tail-mass audit.

For each of the 5 failure folds, build the eligible candidate pool exactly as
the v2.4 matcher sees it (per ``eligible_candidates()`` in
``src/analog_mc/simulate.py``: forward-block boundary + feature completeness).
For every candidate, compute the 60-day forward log-return sum and convert to
percent return. Bucket and tabulate.

Headline question: how many ≥+30% 60-day rallies exist per pool?

Writes:
  - results/analog_mc/data/v3_5_2_tail_mass.json
  - docs/analog_mc/v3.5/_v3_5_2_tail_mass.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.features import compute_features
from analog_mc.simulate import eligible_candidates

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "v3_5_2_tail_mass.json"
OUT_MD = ROOT / "docs" / "analog_mc" / "v3.5" / "_v3_5_2_tail_mass.md"

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

# Bucket edges as percent returns; inclusive lower, exclusive upper.
BUCKETS: list[tuple[float, float, str]] = [
    (-1e9, -50, "(-∞, -50%)"),
    (-50, -30, "[-50%, -30%)"),
    (-30, -20, "[-30%, -20%)"),
    (-20, -10, "[-20%, -10%)"),
    (-10, 0, "[-10%, 0%)"),
    (0, 10, "[0%, +10%)"),
    (10, 20, "[+10%, +20%)"),
    (20, 30, "[+20%, +30%)"),
    (30, 50, "[+30%, +50%)"),
    (50, 1e9, "[+50%, +∞)"),
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


def bucket_returns(pct: np.ndarray) -> list[tuple[str, int, float]]:
    out = []
    n = len(pct)
    for lo, hi, label in BUCKETS:
        mask = (pct >= lo) & (pct < hi)
        cnt = int(mask.sum())
        share = float(cnt / n) if n else 0.0
        out.append((label, cnt, share))
    return out


def main() -> None:
    cfg = Config.from_yaml(str(RUN_DIR / "config.yaml"))
    returns = load_returns(cfg)
    returns_arr = returns.to_numpy()
    features = compute_features(
        returns,
        halflife=cfg.ewma_halflife,
        horizons=tuple(cfg.zscore_horizons),
        momentum_lookback=cfg.momentum_lookback,
    )
    folds = load_fold_summaries()
    anchors = load_anchors()

    # Pre-compute 60d forward return per index, vectorized.
    h = cfg.forecast_horizon
    n = len(returns_arr)
    # forward_logret[i] = sum(returns[i+1 : i+1+h]) if i+h < n else NaN
    cumret = np.concatenate([[0.0], np.cumsum(returns_arr)])  # cumret[k] = sum r[0..k-1]
    forward_logret = np.full(n, np.nan)
    valid_end = n - h - 1
    idx = np.arange(0, valid_end + 1)
    forward_logret[idx] = cumret[idx + h + 1] - cumret[idx + 1]
    forward_pct = (np.exp(forward_logret) - 1.0) * 100.0

    results = []
    for label, dates in [("failure", FAILURE_DATES), ("control", CONTROL_DATES)]:
        for d in dates:
            entry = anchors[d]
            origin_idx = entry["origin_idx"]
            fold = find_fold(origin_idx, folds)
            train_end = fold["train_end"]
            candidate_idx = np.arange(0, train_end + 1, dtype=np.int64)
            eligible = eligible_candidates(candidate_idx, features, origin_idx, cfg)
            # Restrict to candidates with valid 60d forward return.
            mask_fwd = ~np.isnan(forward_pct[eligible])
            pool = eligible[mask_fwd]
            pct = forward_pct[pool]
            n_pool = int(pool.size)
            n_rallies_30 = int((pct >= 30).sum())
            n_rallies_20 = int((pct >= 20).sum())
            n_drops_20 = int((pct <= -20).sum())
            n_drops_30 = int((pct <= -30).sum())

            buckets = bucket_returns(pct)
            results.append({
                "label": label,
                "anchor_date": d,
                "origin_idx": origin_idx,
                "fold_index": fold["fold_index"],
                "train_end": train_end,
                "pool_size": n_pool,
                "pool_mean_60d_pct": float(np.mean(pct)),
                "pool_std_60d_pct": float(np.std(pct)),
                "pool_min_60d_pct": float(np.min(pct)),
                "pool_max_60d_pct": float(np.max(pct)),
                "n_rallies_ge_20pct": n_rallies_20,
                "n_rallies_ge_30pct": n_rallies_30,
                "n_drops_le_20pct": n_drops_20,
                "n_drops_le_30pct": n_drops_30,
                "buckets": [
                    {"range": r, "count": c, "share": s} for r, c, s in buckets
                ],
                "realized_60d_pct": entry["realized_60d_return_pct"],
            })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown
    lines: list[str] = []
    lines.append("# V3.5.2 — candidate pool tail-mass audit")
    lines.append("")
    lines.append(
        "Eligible pool per fold = `train_idx` filtered by (a) forward-block "
        "boundary `c + block_length < origin_idx`, (b) feature completeness "
        "(no NaN z₂₀/z₅₀/z₂₀₀/ewma_vol/trailing_mean_200). Same filter as "
        "`eligible_candidates()` in `src/analog_mc/simulate.py:86`."
    )
    lines.append("")
    lines.append("60-day forward return = `exp(sum(returns[c+1 : c+61])) − 1`.")
    lines.append("")

    lines.append("## Pool size and tail counts")
    lines.append("")
    lines.append(
        "| Anchor | Fold | Pool | Realized | mean | std | min | max | "
        "≥+20% | **≥+30%** | ≤−20% | ≤−30% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['anchor_date']} ({r['label']}) | {r['fold_index']} | "
            f"{r['pool_size']} | {r['realized_60d_pct']:+.1f}% | "
            f"{r['pool_mean_60d_pct']:+.2f}% | {r['pool_std_60d_pct']:.2f}% | "
            f"{r['pool_min_60d_pct']:+.1f}% | {r['pool_max_60d_pct']:+.1f}% | "
            f"{r['n_rallies_ge_20pct']} | **{r['n_rallies_ge_30pct']}** | "
            f"{r['n_drops_le_20pct']} | {r['n_drops_le_30pct']} |"
        )
    lines.append("")

    lines.append("## Per-anchor histograms")
    lines.append("")
    # One bucket table for all failures, one for controls
    for grp_label, grp in [("Failure anchors", "failure"), ("Control anchors", "control")]:
        lines.append(f"### {grp_label}")
        lines.append("")
        rs = [r for r in results if r["label"] == grp]
        bucket_labels = [b["range"] for b in rs[0]["buckets"]]
        header = "| Bucket | " + " | ".join(r["anchor_date"] for r in rs) + " |"
        sep = "|---|" + "|".join(["---:"] * len(rs)) + "|"
        lines.append(header)
        lines.append(sep)
        for i, lbl in enumerate(bucket_labels):
            cells = []
            for r in rs:
                c = r["buckets"][i]["count"]
                s = r["buckets"][i]["share"] * 100
                cells.append(f"{c} ({s:.1f}%)")
            lines.append(f"| {lbl} | " + " | ".join(cells) + " |")
        lines.append("")

    # Verdict
    failure_results = [r for r in results if r["label"] == "failure"]
    rally_counts = [r["n_rallies_ge_30pct"] for r in failure_results]
    min_rally = min(rally_counts)
    max_rally = max(rally_counts)
    all_rich = all(c >= 20 for c in rally_counts)
    any_sparse = any(c < 5 for c in rally_counts)
    all_absent = all(c == 0 for c in rally_counts)

    lines.append("## Verdict")
    lines.append("")
    if all_absent:
        verdict = (
            "**+30% rallies absent in every failure pool** "
            f"(range {min_rally}–{max_rally}). The analog primitive structurally "
            "cannot produce a +40% path regardless of weighting or matching. "
            "Confirms structural ceiling. **STOP per plan §Stop conditions** — "
            "no point running V3.5.3/4. Document tail inflation as v5+ scope."
        )
    elif all_rich:
        verdict = (
            f"**+30% rallies are abundant** ({min_rally}–{max_rally} instances "
            "per failure pool). The matcher *could* sample them but doesn't. "
            "**Matching problem** — A2 (max-corr distance) candidate strengthens. "
            "Continue to V3.5.3."
        )
    elif any_sparse:
        verdict = (
            f"**Pool is sparse** in +30% rallies "
            f"(range {min_rally}–{max_rally}; threshold <5). The analog primitive "
            "is fundamentally evidence-limited for these moves. "
            "**Document tail inflation as v5 scope.** B1 alone won't fix the "
            "structural ceiling. Continue to V3.5.3 for FHS comparison."
        )
    else:
        verdict = (
            f"**Pool is mid-thin** in +30% rallies "
            f"(range {min_rally}–{max_rally}; 5–20 instances). Matching could "
            "theoretically pick them, but evidence is thin. Continue to V3.5.3."
        )
    lines.append(verdict)
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
