"""Diagnostics and v2-trigger decision rules (Stage 9).

Operates on the persisted artifacts written by ``walk_forward.run_walk_forward``.
Each diagnostic is a small function returning a matplotlib Figure or a
DataFrame — the Streamlit dashboard (Stage 11) consumes these directly.

Diagnostic decision rules at the bottom of this file evaluate the plan's
"trigger v2 if X" conditions on a completed run and return a structured
verdict (which rule fired, what the magnitude was).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_mc.config import Config


# ---------------------------------------------------------------------------
# Loading run artifacts
# ---------------------------------------------------------------------------


@dataclass
class FoldArtifacts:
    """Per-fold artifacts persisted by walk_forward."""
    index: int
    summary: dict
    paths: np.ndarray         # (n_origins, n_paths, horizon)
    ratios: np.ndarray        # (n_origins, n_paths, n_blocks) — pre-clip σ ratios
    realized: np.ndarray      # (n_origins, horizon)
    origin_idx: np.ndarray    # (n_origins,)
    search_grid: pd.DataFrame # full grid evaluation table


@dataclass
class RunArtifacts:
    """Everything walk_forward.run_walk_forward writes for one run."""
    run_dir: Path
    config: Config
    meta: dict
    summary: pd.DataFrame
    folds: list[FoldArtifacts]

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def load_run(run_dir: str | Path) -> RunArtifacts:
    """Load all persisted artifacts from a run directory.

    Tolerates in-progress runs: if ``summary.parquet`` is missing (because the
    walk-forward hasn't finished), the summary table is rebuilt from per-fold
    ``summary.json`` files and only completed folds are loaded.

    Raises FileNotFoundError only if the run is so incomplete that there are
    no per-fold summaries at all.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"config.yaml missing in {run_dir} — not a run directory")

    config = Config.from_yaml(run_dir / "config.yaml")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    folds: list[FoldArtifacts] = []
    folds_root = run_dir / "folds"
    if folds_root.exists():
        for fold_dir in sorted(folds_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not (fold_dir / "summary.json").exists():
                continue
            if not (fold_dir / "forecasts.npz").exists():
                continue
            fold_summary = json.loads((fold_dir / "summary.json").read_text())
            with np.load(fold_dir / "forecasts.npz") as data:
                folds.append(FoldArtifacts(
                    index=int(fold_dir.name),
                    summary=fold_summary,
                    paths=data["paths"],
                    ratios=data["ratios"],
                    realized=data["realized"],
                    origin_idx=data["origin_idx"],
                    search_grid=pd.read_parquet(fold_dir / "search.parquet"),
                ))

    if not folds:
        raise FileNotFoundError(
            f"No completed folds found in {run_dir}; run may not have started yet."
        )

    # Prefer the flat summary.parquet if the run finished; otherwise rebuild
    # from per-fold summary.json so partial in-progress runs still load.
    parquet_path = run_dir / "summary.parquet"
    if parquet_path.exists():
        summary = pd.read_parquet(parquet_path)
    else:
        summary = pd.DataFrame([
            {
                "fold_index": f.summary["fold_index"],
                "w0": f.summary["weights"][0],
                "w1": f.summary["weights"][1],
                "w2": f.summary["weights"][2],
                "n_eff": f.summary["n_eff"],
                "val_crps": f.summary["val_crps"],
                "test_crps": f.summary["test_crps"],
                "n_test_origins": f.summary["n_test_origins"],
            }
            for f in folds
        ])

    return RunArtifacts(run_dir=run_dir, config=config, meta=meta, summary=summary, folds=folds)


# ---------------------------------------------------------------------------
# Computational primitives
# ---------------------------------------------------------------------------


def pit_ranks(paths: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Per-step PIT rank of the realized cumulative return in the forecast distribution.

    Args:
        paths:    (n_origins, n_paths, horizon) log returns.
        realized: (n_origins, horizon) log returns.

    Returns:
        (n_origins, horizon) array of PIT ranks in [0, 1].
        rank = (count of forecast samples < realized + 0.5 * count equal) / n_paths
    """
    fc_cum = paths.cumsum(axis=2)             # (O, P, H)
    rl_cum = realized.cumsum(axis=1)          # (O, H)
    less = (fc_cum < rl_cum[:, None, :]).sum(axis=1)
    equal = (fc_cum == rl_cum[:, None, :]).sum(axis=1)
    n_paths = paths.shape[1]
    return (less + 0.5 * equal) / n_paths


def per_step_crps(paths: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Per-(origin, step) CRPS on cumulative log returns."""
    from analog_mc.scoring import crps_per_step
    n_origins = paths.shape[0]
    horizon = paths.shape[2]
    out = np.empty((n_origins, horizon))
    for i in range(n_origins):
        out[i] = crps_per_step(paths[i].astype(np.float64), realized[i].astype(np.float64))
    return out


def concatenate_oos(folds: list[FoldArtifacts]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate test-block paths/realized/origin_idx across folds."""
    paths = np.concatenate([f.paths for f in folds], axis=0)
    realized = np.concatenate([f.realized for f in folds], axis=0)
    origin_idx = np.concatenate([f.origin_idx for f in folds], axis=0)
    return paths, realized, origin_idx


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def weight_trajectory_plot(run: RunArtifacts) -> plt.Figure:
    """Per-fold weight values across folds, three lines (one per horizon)."""
    summary = run.summary
    horizons = run.config.zscore_horizons
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, h in zip(["w0", "w1", "w2"], horizons):
        ax.plot(summary["fold_index"], summary[col], marker="o", lw=1.5, label=f"w (h={h})")
    ax.set_xlabel("fold index")
    ax.set_ylabel("weight")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Weight trajectory across walk-forward folds")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def crps_surface_plot(run: RunArtifacts, fold_index: int = 0) -> plt.Figure:
    """CRPS surface over the weight simplex for one fold (default: first).

    Renders three 2D scatter plots, one per fixed-vertex projection. Color = CRPS.
    """
    fold = next((f for f in run.folds if f.index == fold_index), None)
    if fold is None:
        raise ValueError(f"fold {fold_index} not found in run")
    grid = fold.search_grid
    # Pick the best n_eff for visualization (the one with min CRPS aggregated).
    best_n_eff = grid.groupby("n_eff")["val_crps"].mean().idxmin()
    sub = grid[grid["n_eff"] == best_n_eff]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(sub["w0"], sub["w1"], c=sub["val_crps"], cmap="viridis_r", s=200, edgecolor="k")
    plt.colorbar(sc, ax=ax, label="val CRPS")
    ax.set_xlabel(f"w (h={run.config.zscore_horizons[0]})")
    ax.set_ylabel(f"w (h={run.config.zscore_horizons[1]})")
    ax.set_title(f"CRPS surface — fold {fold_index}, n_eff={best_n_eff}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def global_pit_histogram(run: RunArtifacts, n_bins: int | None = None) -> plt.Figure:
    """PIT histogram over every (origin, step) pair in the OOS record.

    A well-calibrated forecast yields a flat histogram. Slope ⇒ drift mismatch;
    U-shape ⇒ under-dispersed; n-shape ⇒ over-dispersed.
    """
    paths, realized, _ = concatenate_oos(run.folds)
    n_bins = n_bins if n_bins is not None else run.config.pit_n_bins
    ranks = pit_ranks(paths, realized).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ranks, bins=n_bins, range=(0, 1), edgecolor="k", alpha=0.7)
    ax.axhline(len(ranks) / n_bins, color="red", linestyle="--", label="flat (ideal)")
    ax.set_xlabel("PIT rank")
    ax.set_ylabel("count")
    ax.set_title(f"Global PIT histogram (N = {len(ranks):,} (origin × step) pairs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def conditional_pit_by_vol_regime(
    run: RunArtifacts, returns: pd.Series, n_bins: int | None = None,
) -> plt.Figure:
    """Three PIT histograms bucketed by σ at the forecast origin (vol terciles)."""
    from analog_mc.features import causal_ewma_vol
    sigma = causal_ewma_vol(returns, halflife=run.config.ewma_halflife).to_numpy()
    n_bins = n_bins if n_bins is not None else run.config.pit_n_bins
    q_lo, q_hi = run.config.vol_regime_quantiles

    paths, realized, origin_idx = concatenate_oos(run.folds)
    sigma_at_origin = sigma[origin_idx]
    cutoffs = np.quantile(sigma_at_origin, [q_lo, q_hi])
    buckets = np.digitize(sigma_at_origin, cutoffs)  # 0/1/2

    ranks = pit_ranks(paths, realized)  # (O, H)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    labels = [
        f"low σ (<{cutoffs[0]:.4f})",
        f"mid σ ({cutoffs[0]:.4f}–{cutoffs[1]:.4f})",
        f"high σ (>{cutoffs[1]:.4f})",
    ]
    for i, ax in enumerate(axes):
        mask = buckets == i
        if mask.sum() == 0:
            ax.set_title(f"{labels[i]} (empty)")
            continue
        r = ranks[mask].flatten()
        ax.hist(r, bins=n_bins, range=(0, 1), edgecolor="k", alpha=0.7)
        ax.axhline(len(r) / n_bins, color="red", linestyle="--")
        ax.set_title(f"{labels[i]} — N = {len(r):,}")
        ax.set_xlabel("PIT rank")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Conditional PIT by vol regime at forecast origin")
    fig.tight_layout()
    return fig


def reliability_diagram(
    run: RunArtifacts, quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> plt.Figure:
    """For each predicted quantile q, plot the empirical fraction of realizations
    falling below the q-th quantile of the forecast distribution.

    Perfect calibration → diagonal.
    """
    paths, realized, _ = concatenate_oos(run.folds)
    fc_cum = paths.cumsum(axis=2)   # (O, P, H)
    rl_cum = realized.cumsum(axis=1)  # (O, H)

    empirical = []
    for q in quantiles:
        # Quantile across paths for each (origin, step):
        q_pred = np.quantile(fc_cum, q, axis=1)  # (O, H)
        below = (rl_cum < q_pred).mean()
        empirical.append(float(below))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="perfect")
    ax.plot(quantiles, empirical, marker="o", lw=2, color="darkred", label="empirical")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical fraction below")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fixed_weight_baseline_crps(
    returns: pd.Series,
    config: Config,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    n_eff: float = 30.0,
    run_dir: Path | None = None,
) -> dict:
    """Run a parallel walk-forward with fixed weights/n_eff and return aggregate test CRPS.

    Used by the decision rules to assess whether the per-fold tuned search adds value.
    """
    from analog_mc.data import generate_folds
    from analog_mc.features import compute_features
    from analog_mc.walk_forward import _evaluate_on_test

    features = compute_features(
        returns,
        halflife=config.ewma_halflife,
        horizons=config.zscore_horizons,
        momentum_lookback=config.momentum_lookback if config.drift_mode != "zero" else None,
    )
    folds = generate_folds(returns, config)
    weights_arr = np.array(weights)

    test_crpss: list[float] = []
    n_origins: list[int] = []
    for fold in folds:
        _, _, _, origin_idx, test_crps = _evaluate_on_test(
            fold, weights_arr, float(n_eff), returns.to_numpy(), features, config,
        )
        if origin_idx.size > 0:
            test_crpss.append(test_crps)
            n_origins.append(origin_idx.size)

    weighted = (
        sum(c * n for c, n in zip(test_crpss, n_origins)) / sum(n_origins)
        if n_origins else float("inf")
    )
    return {
        "weights": list(weights),
        "n_eff": float(n_eff),
        "mean_test_crps": float(np.mean(test_crpss)) if test_crpss else float("inf"),
        "weighted_test_crps": float(weighted),
        "per_fold_test_crps": test_crpss,
    }


def acf_comparison(
    run: RunArtifacts, returns: pd.Series,
) -> plt.Figure:
    """ACF of squared returns: simulated paths vs realized series.

    Per the plan: flag if the simulated ACF is >30% below the realized ACF at
    any seam lag (10, 20, 30, 40, 50). That's reported by ``decision_rules``;
    this plot just visualizes both curves.
    """
    lags = list(run.config.acf_lags)
    paths, _, _ = concatenate_oos(run.folds)

    # Realized squared-return ACF over the in-sample range covered by tests.
    realized_sq = (returns.to_numpy() ** 2)
    realized_acf = _acf(realized_sq, max(lags))
    realized_at_lags = [float(realized_acf[lag]) for lag in lags]

    # Simulated squared-return ACF averaged over (origins × paths).
    sim_sq = paths ** 2  # (O, P, H)
    O, P, H = sim_sq.shape
    sim_flat = sim_sq.reshape(O * P, H)
    sim_acf_avg = np.zeros(max(lags) + 1)
    for row in sim_flat:
        sim_acf_avg += _acf(row, max(lags))
    sim_acf_avg /= sim_flat.shape[0]
    sim_at_lags = [float(sim_acf_avg[lag]) for lag in lags]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags, realized_at_lags, marker="o", lw=2, label="realized")
    ax.plot(lags, sim_at_lags, marker="s", lw=2, label="simulated (avg)")
    for seam in (10, 20, 30, 40, 50):
        if seam in lags:
            ax.axvline(seam, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF of squared returns")
    ax.set_title("Squared-return ACF: simulated vs realized")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF up to lag max_lag (inclusive). Returns array of length max_lag + 1."""
    x = x - x.mean()
    n = x.size
    out = np.empty(max_lag + 1)
    var = (x * x).sum()
    if var == 0:
        return np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[lag] = 1.0
            continue
        out[lag] = (x[:-lag] * x[lag:]).sum() / var
    return out


def clip_hit_summary(run: RunArtifacts) -> plt.Figure:
    """Histogram of σ ratios (pre-clip) with vertical lines at the clip bounds."""
    all_ratios = np.concatenate([f.ratios.flatten() for f in run.folds])
    lo, hi = run.config.vol_clip_lower, run.config.vol_clip_upper

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_ratios, bins=80, range=(0, hi * 1.3), edgecolor="k", alpha=0.7)
    ax.axvline(lo, color="red", linestyle="--", label=f"lower clip = {lo}")
    ax.axvline(hi, color="red", linestyle="--", label=f"upper clip = {hi}")
    pct_lo = float((all_ratios < lo).mean() * 100)
    pct_hi = float((all_ratios > hi).mean() * 100)
    ax.set_title(
        f"σ ratio distribution — {pct_lo:.1f}% hit lower bound, {pct_hi:.1f}% hit upper bound"
    )
    ax.set_xlabel("σ_current / σ_historical (pre-clip)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Aggregate CRPS report (Stage 10)
# ---------------------------------------------------------------------------


def aggregate_crps_overall(run: RunArtifacts) -> dict:
    """Mean per-step CRPS across every (origin, step) pair in the OOS record."""
    paths, realized, _ = concatenate_oos(run.folds)
    crps = per_step_crps(paths, realized)
    return {
        "mean_crps": float(crps.mean()),
        "median_crps": float(np.median(crps)),
        "n_origin_step_pairs": int(crps.size),
    }


def aggregate_crps_per_fold(run: RunArtifacts) -> pd.DataFrame:
    """Per-fold val and test mean CRPS, with origin counts and locked params."""
    return run.summary.copy()


def aggregate_crps_per_step(run: RunArtifacts) -> pd.DataFrame:
    """Mean CRPS at each horizon step, averaged across all OOS origins.

    Returns a DataFrame with columns [step, mean_crps, std_crps, n_origins].
    """
    paths, realized, _ = concatenate_oos(run.folds)
    crps = per_step_crps(paths, realized)  # (O, H)
    return pd.DataFrame({
        "step": np.arange(1, crps.shape[1] + 1),
        "mean_crps": crps.mean(axis=0),
        "std_crps": crps.std(axis=0),
        "n_origins": np.full(crps.shape[1], crps.shape[0]),
    })


def aggregate_crps_per_vol_regime(run: RunArtifacts, returns: pd.Series) -> pd.DataFrame:
    """Mean CRPS bucketed by σ at the forecast origin (low/mid/high terciles)."""
    from analog_mc.features import causal_ewma_vol

    sigma = causal_ewma_vol(returns, halflife=run.config.ewma_halflife).to_numpy()
    paths, realized, origin_idx = concatenate_oos(run.folds)
    crps = per_step_crps(paths, realized).mean(axis=1)  # per-origin mean CRPS

    sigma_at_origin = sigma[origin_idx]
    q_lo, q_hi = run.config.vol_regime_quantiles
    cutoffs = np.quantile(sigma_at_origin, [q_lo, q_hi])
    buckets = np.digitize(sigma_at_origin, cutoffs)

    rows = []
    for i, label in enumerate(["low_vol", "mid_vol", "high_vol"]):
        mask = buckets == i
        rows.append({
            "regime": label,
            "n_origins": int(mask.sum()),
            "mean_crps": float(crps[mask].mean()) if mask.any() else float("nan"),
            "median_crps": float(np.median(crps[mask])) if mask.any() else float("nan"),
            "sigma_lower_bound": float(0.0 if i == 0 else cutoffs[i - 1]),
            "sigma_upper_bound": float(cutoffs[i]) if i < 2 else float("inf"),
        })
    return pd.DataFrame(rows)


def generate_report(
    run: RunArtifacts,
    returns: pd.Series,
    fixed_baseline: dict | None = None,
) -> dict:
    """One-stop summary: overall CRPS + per-fold + per-step + per-regime + decision rules.

    Returns a dict (JSON-serializable except for the DataFrame entries) with:
      * overall:      {mean_crps, median_crps, n_origin_step_pairs}
      * per_fold:     DataFrame
      * per_step:     DataFrame
      * per_regime:   DataFrame
      * fixed_baseline: dict or None
      * decision_rules: dict (output of decision_rules())
    """
    return {
        "overall": aggregate_crps_overall(run),
        "per_fold": aggregate_crps_per_fold(run),
        "per_step": aggregate_crps_per_step(run),
        "per_regime": aggregate_crps_per_vol_regime(run, returns),
        "fixed_baseline": fixed_baseline,
        "decision_rules": decision_rules(run, returns, fixed_baseline=fixed_baseline),
    }


# ---------------------------------------------------------------------------
# Decision rules — the v2-trigger summary
# ---------------------------------------------------------------------------


def decision_rules(
    run: RunArtifacts,
    returns: pd.Series,
    fixed_baseline: dict | None = None,
) -> dict:
    """Evaluate the plan's v2-trigger rules on the run.

    Returns a dict where each key is a rule name and the value contains:
      * fired: bool — whether the rule triggered
      * metric: numeric magnitude
      * recommendation: free-text v2 guidance from the plan

    If ``fixed_baseline`` is None, the "fixed weights match tuned" rule is skipped.
    """
    rules: dict[str, dict] = {}
    paths, realized, origin_idx = concatenate_oos(run.folds)
    ranks = pit_ranks(paths, realized).flatten()

    # 1. Sloped global PIT → recommend trailing_momentum drift in v2.
    # Measure slope by comparing low-rank vs high-rank fractions.
    low_frac = float((ranks < 0.25).mean())
    high_frac = float((ranks > 0.75).mean())
    slope_metric = high_frac - low_frac
    rules["sloped_global_pit"] = {
        "fired": abs(slope_metric) > 0.1,
        "metric": slope_metric,
        "recommendation": (
            "Enable drift_mode='trailing_momentum' in v2: PIT skew suggests a "
            "directional bias the zero-drift v1 cannot capture."
        ),
    }

    # 2. U-shaped high-vol-bucket PIT → raise vol_clip_upper and/or n_eff candidates.
    from analog_mc.features import causal_ewma_vol
    sigma = causal_ewma_vol(returns, halflife=run.config.ewma_halflife).to_numpy()
    cutoffs = np.quantile(sigma[origin_idx], list(run.config.vol_regime_quantiles))
    high_mask = sigma[origin_idx] > cutoffs[1]
    if high_mask.sum() > 0:
        high_ranks = pit_ranks(paths[high_mask], realized[high_mask]).flatten()
        # U-shape: tails are over-represented vs middle.
        tail = float(((high_ranks < 0.1) | (high_ranks > 0.9)).mean())
        middle = float(((high_ranks > 0.4) & (high_ranks < 0.6)).mean())
        u_metric = tail / max(middle, 1e-6)
        rules["u_shaped_high_vol_pit"] = {
            "fired": u_metric > 2.5,
            "metric": u_metric,
            "recommendation": (
                "Raise vol_clip_upper and/or expand n_eff_values upward in v2: "
                "high-vol-regime forecasts are too sharp."
            ),
        }
    else:
        rules["u_shaped_high_vol_pit"] = {"fired": False, "metric": float("nan"), "recommendation": ""}

    # 3. Squared-return ACF degradation > 30% at seam lags → v2 conditional sampling.
    lags = list(run.config.acf_lags)
    seam_lags = [l for l in lags if l in (10, 20, 30, 40, 50)]
    sim_sq_flat = (paths ** 2).reshape(paths.shape[0] * paths.shape[1], paths.shape[2])
    sim_acf_avg = np.zeros(max(lags) + 1)
    for row in sim_sq_flat:
        sim_acf_avg += _acf(row, max(lags))
    sim_acf_avg /= sim_sq_flat.shape[0]
    realized_acf = _acf(returns.to_numpy() ** 2, max(lags))
    seam_deltas = []
    for lag in seam_lags:
        r = float(realized_acf[lag])
        s = float(sim_acf_avg[lag])
        if r > 1e-6:
            seam_deltas.append((s - r) / r)
    worst = float(min(seam_deltas)) if seam_deltas else 0.0
    rules["acf_seam_degradation"] = {
        "fired": worst < -0.3,
        "metric": worst,
        "recommendation": (
            "Implement vol-aware conditional block sampling in v2: simulated "
            "squared-return ACF collapses at seam lags vs realized."
        ),
    }

    # 4. Fixed-weight baseline within 1% of tuned → drop the per-fold search.
    if fixed_baseline is not None:
        tuned = float(run.summary["test_crps"].mean())
        fixed = float(fixed_baseline["mean_test_crps"])
        rel_diff = (fixed - tuned) / max(tuned, 1e-12)
        rules["fixed_weight_close_to_tuned"] = {
            "fired": abs(rel_diff) < 0.01,
            "metric": rel_diff,
            "recommendation": (
                "Ship fixed weights — per-fold search adds <1% CRPS improvement "
                "and costs an order of magnitude more compute."
            ),
        }

    # 5. Clip-hit fraction > 15% on either bound → revisit distance metric.
    all_ratios = np.concatenate([f.ratios.flatten() for f in run.folds])
    pct_lo = float((all_ratios < run.config.vol_clip_lower).mean())
    pct_hi = float((all_ratios > run.config.vol_clip_upper).mean())
    rules["clip_hit_excessive"] = {
        "fired": max(pct_lo, pct_hi) > 0.15,
        "metric": max(pct_lo, pct_hi),
        "recommendation": (
            "Distance metric is failing to match vol regime — analogs are "
            "frequently coming from materially different vol contexts."
        ),
    }

    return rules
