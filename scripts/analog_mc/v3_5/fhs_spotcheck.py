"""V3.5.3 — GARCH-FHS spot-check at the 5 failure anchors.

For each failure anchor:
  1. Read the canonical v2.4 forecast paths from `runs/.../folds/<F>/forecasts.npz`.
  2. Fit GARCH(1,1) on returns[:origin+1]; simulate 1000 σ-paths over 60 days.
  3. Sample i.i.d. residual sequences from the causal standardized residual
     pool (`r_t / ewma_σ_t`, all t ≤ origin).
  4. Multiply: FHS path r[i, t] = σ_paths[i, t] * residual_sample[i, t].
  5. Integrate to cumulative log-returns; compute 50% and 90% bands.
  6. Coverage = number of 60 realized days inside each band.

Compares v2.4 vs FHS bands per anchor. Question: does FHS catch the rallies
v2.4 misses?

Writes:
  - results/analog_mc/data/v3_5_3_fhs_spotcheck.json
  - docs/analog_mc/v3.5/_v3_5_3_fhs_spotcheck.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.features import causal_ewma_vol
from analog_mc.vol import fit_garch, simulate_garch_sigma_paths

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs" / "analog_mc" / "20260520T045525Z"
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"
OUT_JSON = ROOT / "results" / "analog_mc" / "data" / "v3_5_3_fhs_spotcheck.json"
OUT_MD = ROOT / "docs" / "analog_mc" / "v3.5" / "_v3_5_3_fhs_spotcheck.md"

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

N_PATHS = 1000
SEED = 20260520


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


def band_coverage(
    paths: np.ndarray, realized: np.ndarray
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 50% and 90% band coverage day-by-day.

    Args:
        paths: (n_paths, horizon) log returns
        realized: (horizon,) realized log returns
    Returns:
        (n_in_50, n_in_90, q5, q25, q75, q95) — counts and quantile curves
        of CUMULATIVE log returns.
    """
    cum_paths = np.cumsum(paths, axis=1)  # (n_paths, horizon)
    cum_realized = np.cumsum(realized)
    q5 = np.quantile(cum_paths, 0.05, axis=0)
    q25 = np.quantile(cum_paths, 0.25, axis=0)
    q75 = np.quantile(cum_paths, 0.75, axis=0)
    q95 = np.quantile(cum_paths, 0.95, axis=0)
    in_50 = int(((cum_realized >= q25) & (cum_realized <= q75)).sum())
    in_90 = int(((cum_realized >= q5) & (cum_realized <= q95)).sum())
    return in_50, in_90, q5, q25, q75, q95


def fhs_simulate(
    returns: np.ndarray,
    origin_idx: int,
    horizon: int,
    halflife: float,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run GARCH(1,1) + filtered-historical-simulation residual draw.

    Returns (n_paths, horizon) log-return paths.
    """
    import pandas as pd

    # Causal returns up to and including origin.
    causal = returns[: origin_idx + 1]

    # GARCH-driven σ paths (forecast horizon).
    fit = fit_garch(causal)
    sigma_paths = simulate_garch_sigma_paths(fit, horizon, n_paths, rng)

    # Standardized residuals: r_t / ewma_σ_t, using causal EWMA σ.
    ewma_sigma = causal_ewma_vol(pd.Series(causal), halflife).to_numpy()
    # Drop the warm-up NaNs and the first few near-zero σ entries; require σ > 0.
    valid = ~np.isnan(ewma_sigma) & (ewma_sigma > 0)
    z_pool = causal[valid] / ewma_sigma[valid]
    if z_pool.size < 100:
        raise SystemExit(
            f"residual pool too small ({z_pool.size}) for origin {origin_idx}"
        )

    # Sample (n_paths, horizon) residuals i.i.d. from the pool.
    sample_idx = rng.integers(0, z_pool.size, size=(n_paths, horizon))
    z_sample = z_pool[sample_idx]

    return sigma_paths * z_sample


def main() -> None:
    cfg = Config.from_yaml(str(RUN_DIR / "config.yaml"))
    returns = load_returns(cfg).to_numpy()
    folds = load_fold_summaries()
    anchors = load_anchors()
    horizon = cfg.forecast_horizon

    results = []
    for label, dates in [("failure", FAILURE_DATES), ("control", CONTROL_DATES)]:
        for d in dates:
            entry = anchors[d]
            origin_idx = entry["origin_idx"]
            fold = find_fold(origin_idx, folds)
            fold_idx = fold["fold_index"]

            # Load v2.4 paths for this origin from the fold's forecasts.npz.
            npz = np.load(RUN_DIR / "folds" / str(fold_idx) / "forecasts.npz")
            origins = npz["origin_idx"]
            pos = int(np.where(origins == origin_idx)[0][0])
            v24_paths = npz["paths"][pos].astype(np.float64)  # (1000, 60)
            realized = npz["realized"][pos]  # (60,)

            v24_50, v24_90, v24_q5, v24_q25, v24_q75, v24_q95 = band_coverage(
                v24_paths, realized
            )

            # FHS: same seed family but offset per anchor for independence.
            rng = np.random.default_rng(SEED + origin_idx)
            fhs_paths = fhs_simulate(
                returns,
                origin_idx,
                horizon,
                cfg.ewma_halflife,
                N_PATHS,
                rng,
            )
            fhs_50, fhs_90, fhs_q5, fhs_q25, fhs_q75, fhs_q95 = band_coverage(
                fhs_paths, realized
            )

            # Width comparison at terminal step (60d) — log-return band width.
            def width(q_lo: np.ndarray, q_hi: np.ndarray) -> float:
                return float(np.exp(q_hi[-1]) - np.exp(q_lo[-1])) * 100.0

            results.append({
                "label": label,
                "anchor_date": d,
                "origin_idx": origin_idx,
                "fold_index": fold_idx,
                "realized_60d_pct": float(np.exp(realized.sum()) - 1.0) * 100.0,
                "v24": {
                    "in_50": v24_50,
                    "in_90": v24_90,
                    "terminal_50_width_pct": width(v24_q25, v24_q75),
                    "terminal_90_width_pct": width(v24_q5, v24_q95),
                },
                "fhs": {
                    "in_50": fhs_50,
                    "in_90": fhs_90,
                    "terminal_50_width_pct": width(fhs_q25, fhs_q75),
                    "terminal_90_width_pct": width(fhs_q5, fhs_q95),
                },
            })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown
    lines: list[str] = []
    lines.append("# V3.5.3 — GARCH-FHS spot-check vs v2.4")
    lines.append("")
    lines.append(
        "FHS = fit GARCH(1,1) on causal returns → simulate 1000 σ-paths → "
        "i.i.d. draw 1000 residual sequences from `r_t/σ_t` pool. Compare "
        "50/90 band coverage and terminal band widths against v2.4."
    )
    lines.append("")

    lines.append("## Coverage (days of 60 inside band)")
    lines.append("")
    lines.append(
        "| Anchor | Group | Realized 60d | v2.4 50/60 | **v2.4 90/60** | "
        "FHS 50/60 | **FHS 90/60** | Δ90 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        d90 = r["fhs"]["in_90"] - r["v24"]["in_90"]
        lines.append(
            f"| {r['anchor_date']} | {r['label']} | "
            f"{r['realized_60d_pct']:+.1f}% | {r['v24']['in_50']} | "
            f"**{r['v24']['in_90']}** | {r['fhs']['in_50']} | "
            f"**{r['fhs']['in_90']}** | {d90:+d} |"
        )
    lines.append("")

    lines.append("## Terminal (day-60) band widths (price-relative %)")
    lines.append("")
    lines.append("| Anchor | v2.4 50% | v2.4 90% | FHS 50% | FHS 90% | 90% ratio FHS/v24 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        ratio = r["fhs"]["terminal_90_width_pct"] / max(
            r["v24"]["terminal_90_width_pct"], 1e-9
        )
        lines.append(
            f"| {r['anchor_date']} | {r['v24']['terminal_50_width_pct']:.1f}% | "
            f"{r['v24']['terminal_90_width_pct']:.1f}% | "
            f"{r['fhs']['terminal_50_width_pct']:.1f}% | "
            f"{r['fhs']['terminal_90_width_pct']:.1f}% | {ratio:.2f}× |"
        )
    lines.append("")

    # Verdict
    failures = [r for r in results if r["label"] == "failure"]
    fhs_catches = sum(1 for r in failures if r["fhs"]["in_90"] >= 45)
    v24_catches = sum(1 for r in failures if r["v24"]["in_90"] >= 45)
    fhs_widening = sum(
        1
        for r in failures
        if r["fhs"]["terminal_90_width_pct"] > r["v24"]["terminal_90_width_pct"]
    )

    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"- v2.4 catches **{v24_catches}/5** failures at 90%-band ≥45/60."
    )
    lines.append(
        f"- FHS catches **{fhs_catches}/5** failures at 90%-band ≥45/60."
    )
    lines.append(
        f"- FHS produces a wider 90%-band terminal width than v2.4 in "
        f"**{fhs_widening}/5** failure anchors."
    )
    lines.append("")
    if fhs_catches >= 3:
        verdict = (
            f"**FHS catches {fhs_catches}/5 failures** — analog primitive is "
            "actively suppressing tail dispersion. **Promote A1 ahead of B1 in "
            "V4_EXPERIMENTS_PLAN.md.** A1 could ship as a partial fix or as a "
            "hybrid (matcher direction + FHS dispersion)."
        )
    elif fhs_catches <= 1 and fhs_widening <= 1:
        verdict = (
            f"**FHS also misses ({fhs_catches}/5 caught, only {fhs_widening}/5 "
            "wider).** σ-based dispersion alone doesn't recover these moves; "
            "B1's drift-correction is needed. **Tail inflation is v5+ scope.**"
        )
    else:
        verdict = (
            f"**Mixed FHS result** ({fhs_catches}/5 caught, {fhs_widening}/5 "
            "wider). FHS gives partial improvement on some anchors but isn't "
            "a clear winner. A1 worth evaluating formally but not promoting "
            "ahead of B1 on this evidence alone."
        )
    lines.append(verdict)
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
