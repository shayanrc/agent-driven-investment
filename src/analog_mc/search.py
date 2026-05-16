"""Hyperparameter search: grid + Nelder-Mead refinement (Stage 7).

The grid covers the 3-simplex of z-score-horizon weights (10% spacing by
default -> 66 points) crossed with a small discrete list of ``n_eff`` values.
For each (weights, n_eff) pair we compute the mean per-step CRPS across all
val origins in the fold.

Nelder-Mead then refines the weight vector from each of the top-k grid points,
with ``n_eff`` held fixed at that grid point's value. **n_eff is never
optimized continuously** — the CRPS surface in n_eff is non-smooth, and per
the plan the search treats n_eff as a discrete tuning knob via the grid.

Determinism: each (weights, n_eff, origin) gets its own seeded RNG derived
from ``config.random_seed``, so evaluating the same point twice returns the
same CRPS. This makes Nelder-Mead's objective deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analog_mc.config import Config
from analog_mc.data import Fold
from analog_mc.scoring import crps_sample
from analog_mc.simulate import forecast


# ---------------------------------------------------------------------------
# Weight grid
# ---------------------------------------------------------------------------


def generate_weight_grid(resolution: float) -> np.ndarray:
    """Return all 3-simplex grid points with the given spacing.

    Each row sums to exactly 1.0 and each component is non-negative.

    With ``resolution=0.1`` this yields 66 points (the binomial coefficient
    C(12, 2), i.e. compositions of 10 into 3 non-negative parts).
    """
    if not (0 < resolution <= 1.0):
        raise ValueError(f"resolution must be in (0, 1.0]; got {resolution}")
    n = int(round(1.0 / resolution))
    if abs(n * resolution - 1.0) > 1e-9:
        raise ValueError(f"resolution {resolution} must evenly divide 1.0")
    points = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            points.append((i, j, k))
    return np.array(points, dtype=np.float64) / n


# ---------------------------------------------------------------------------
# Per-(weights, n_eff) evaluation
# ---------------------------------------------------------------------------


def _seed_for(base_seed: int, weights: np.ndarray, n_eff: float, origin: int) -> int:
    """Derive a deterministic per-(weights, n_eff, origin) seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(base_seed).to_bytes(8, "little", signed=False))
    h.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    h.update(np.float64(n_eff).tobytes())
    h.update(int(origin).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little")


def evaluate(
    weights: np.ndarray,
    n_eff: float,
    fold: Fold,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
    eval_set: str = "val",
) -> float:
    """Mean per-step CRPS averaged over all eligible origins in val (or test).

    Origins are silently skipped if the forecast can't be generated (e.g. NaN
    features at origin, empty candidate pool) or if the realized window runs
    past the end of the returns array. If no origin produces a forecast,
    returns +inf so the search treats it as worst-possible.
    """
    if eval_set == "val":
        origins = fold.val_idx
    elif eval_set == "test":
        origins = fold.test_idx
    else:
        raise ValueError(f"eval_set must be 'val' or 'test'; got {eval_set}")

    crpss: list[float] = []
    for origin_idx in origins:
        origin_idx_i = int(origin_idx)
        if origin_idx_i + 1 + config.forecast_horizon > returns.size:
            continue
        rng = np.random.default_rng(_seed_for(config.random_seed, weights, n_eff, origin_idx_i))
        try:
            paths = forecast(
                origin_idx=origin_idx_i,
                returns=returns,
                candidate_idx=fold.train_idx,
                features=features,
                weights=weights,
                n_eff=n_eff,
                config=config,
                rng=rng,
            )
        except ValueError:
            continue
        realized = returns[origin_idx_i + 1 : origin_idx_i + 1 + config.forecast_horizon]
        crpss.append(crps_sample(paths, realized))

    if not crpss:
        return float("inf")
    return float(np.mean(crpss))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def grid_search(
    fold: Fold,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Evaluate every (weight grid point, n_eff) on the fold's val set.

    Returns a DataFrame with columns ``[w0, w1, w2, n_eff, val_crps]``,
    sorted by ``val_crps`` ascending (best first).
    """
    weights_grid = generate_weight_grid(config.weight_grid_resolution)
    n_eff_values = list(config.n_eff_values)

    rows: list[dict] = []
    for w in weights_grid:
        for n_eff in n_eff_values:
            val_crps = evaluate(w, float(n_eff), fold, returns, features, config, eval_set="val")
            rows.append({
                "w0": float(w[0]), "w1": float(w[1]), "w2": float(w[2]),
                "n_eff": int(n_eff), "val_crps": val_crps,
            })
    df = pd.DataFrame(rows)
    return df.sort_values("val_crps", ascending=True, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Nelder-Mead local refinement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one full search (grid + refine)."""
    weights: np.ndarray            # shape (3,), sums to 1
    n_eff: float
    val_crps: float
    n_forecasts: int               # forecasts evaluated during the search
    grid_df: pd.DataFrame          # full grid evaluation table


def _refine_one_start(
    w_start: np.ndarray,
    n_eff_fixed: float,
    fold: Fold,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
    eval_counter: list[int],
) -> tuple[np.ndarray, float]:
    """Refine weights from a single starting point with n_eff fixed."""
    # Parameterize as (w0, w1) with w2 = 1 - w0 - w1; penalize leaving the simplex.
    def objective(x: np.ndarray) -> float:
        w0, w1 = float(x[0]), float(x[1])
        w2 = 1.0 - w0 - w1
        if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
            # Outside the simplex — return a large finite penalty proportional to
            # the violation so Nelder-Mead has a smooth-ish gradient back inside.
            violation = max(0.0, -w0) + max(0.0, -w1) + max(0.0, -w2)
            return 1.0 + violation
        weights = np.array([w0, w1, w2])
        eval_counter[0] += 1
        return evaluate(weights, n_eff_fixed, fold, returns, features, config, eval_set="val")

    x0 = np.array([w_start[0], w_start[1]])
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "xatol": config.nelder_mead_xatol,
            "fatol": config.nelder_mead_xatol,
            "maxiter": config.nelder_mead_maxiter,
            "adaptive": True,
        },
    )
    w0_star, w1_star = float(res.x[0]), float(res.x[1])
    w2_star = 1.0 - w0_star - w1_star
    return np.array([w0_star, w1_star, w2_star]), float(res.fun)


def local_refine(
    top_k_rows: pd.DataFrame,
    fold: Fold,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, float, float, int]:
    """Nelder-Mead from each top-k grid point; pick the best refined result.

    Returns ``(weights, n_eff, val_crps, n_objective_evals)``.
    """
    best_weights: np.ndarray | None = None
    best_n_eff: float | None = None
    best_crps = float("inf")
    eval_counter = [0]

    for _, row in top_k_rows.iterrows():
        w_start = np.array([row["w0"], row["w1"], row["w2"]])
        n_eff_fixed = float(row["n_eff"])
        weights, crps = _refine_one_start(
            w_start, n_eff_fixed, fold, returns, features, config, eval_counter,
        )
        if crps < best_crps:
            best_weights = weights
            best_n_eff = n_eff_fixed
            best_crps = crps

    # Fallback: if no refinement improved on the grid (shouldn't happen with
    # the no-op start, but be defensive), return the top grid row as-is.
    if best_weights is None:
        top = top_k_rows.iloc[0]
        best_weights = np.array([top["w0"], top["w1"], top["w2"]])
        best_n_eff = float(top["n_eff"])
        best_crps = float(top["val_crps"])

    return best_weights, float(best_n_eff), best_crps, eval_counter[0]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_search(
    fold: Fold,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
) -> SearchResult:
    """Full search for one fold: grid → top-k → Nelder-Mead refine."""
    grid_df = grid_search(fold, returns, features, config)
    n_grid_evals = len(grid_df) * len(fold.val_idx)

    top_k = grid_df.head(config.local_refine_top_k)
    weights, n_eff, val_crps, n_refine_evals = local_refine(
        top_k, fold, returns, features, config,
    )
    n_refine_forecasts = n_refine_evals * len(fold.val_idx)

    return SearchResult(
        weights=weights,
        n_eff=n_eff,
        val_crps=val_crps,
        n_forecasts=n_grid_evals + n_refine_forecasts,
        grid_df=grid_df,
    )
