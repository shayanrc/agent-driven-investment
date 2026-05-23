"""V4.5.3 — B1 β autopsy at the 5 regressions.

For each B1 regression anchor:
1. Reproduce the matcher probability layer at the anchor origin using the
   B1 canonical fold's (weights, n_eff). Note: B1 uses weighted-Euclidean,
   not corrwindow.
2. Compute the local-linear regression β + diagnostics + leverage.
3. Identify the max-leverage candidate; refit β with it dropped.
4. Compare original correction vs leverage-trimmed correction.

Tests whether V5.3 "leverage-trimmed B1" is a viable v5 follow-up.

Outputs: results/analog_mc/data/v4_5_3_b1_beta_autopsy.json
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
from analog_mc.local_linear import (
    fit_local_linear_correction,
    forward_logret_sums,
)
from analog_mc.simulate import eligible_candidates

REPO = Path(__file__).resolve().parents[2]
B1_RUN = REPO / "runs/analog_mc/20260520T155220Z"
ANCHORS = REPO / "results/analog_mc/data/fat_tail_eval_anchors.json"
OUT = REPO / "results/analog_mc/data/v4_5_3_b1_beta_autopsy.json"

# B1 regressions from fat_tail_b1_local_linear_diff.json.
B1_REGRESSIONS = ["1991-03-26", "2025-07-02", "1990-09-24", "2001-04-04", "2008-10-03"]
B1_WINS = ["2010-04-23", "2012-03-14", "2001-10-02"]  # for comparison


def load_fold_summaries(run_dir: Path) -> list[dict]:
    folds_dir = run_dir / "folds"
    out = []
    for d in sorted(folds_dir.iterdir(), key=lambda p: int(p.name)):
        out.append(json.loads((d / "summary.json").read_text()))
    return out


def fold_for_origin(folds: list[dict], origin_idx: int) -> dict | None:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    return None


def compute_leverage_scores(
    z_candidates: np.ndarray,
    z_target: np.ndarray,
    probs: np.ndarray,
    forward_returns: np.ndarray,
    tikhonov_rel: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Compute per-candidate leverage in the same weighted-LS that B1 uses.

    Leverage h_ii = w_i × x_iᵀ (XᵀWX + λI)⁻¹ x_i. High h_ii means that
    candidate has outsized influence on β.
    """
    valid = ~np.isnan(forward_returns)
    X_raw = z_candidates[valid]
    y = forward_returns[valid]
    w = probs[valid]
    if w.sum() <= 0:
        raise ValueError("All-zero probs")
    w = w / w.sum()
    K_eff, D = X_raw.shape
    X = np.concatenate([np.ones((K_eff, 1)), X_raw], axis=1)
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    trace = float(np.trace(XtWX))
    lam = tikhonov_rel * trace / (D + 1) if trace > 0 else tikhonov_rel
    XtWX_reg = XtWX + lam * np.eye(D + 1)
    XtWX_inv = np.linalg.inv(XtWX_reg)
    # h_ii = w_i × x_iᵀ (XᵀWX_reg)⁻¹ x_i.
    h_diag = w * np.einsum("ij,jk,ik->i", X, XtWX_inv, X)
    beta = np.linalg.solve(XtWX_reg, Xw.T @ y)
    return h_diag, {
        "beta": beta,
        "matcher_mean": float(w @ y),
        "x_star": np.concatenate([[1.0], z_target]),
        "valid_indices_into_orig": np.flatnonzero(valid),
    }


def autopsy_anchor(
    anchor: dict,
    section: str,
    returns_arr: np.ndarray,
    features: pd.DataFrame,
    fold: dict,
    cfg: Config,
    forward_returns_arr: np.ndarray,
) -> dict:
    origin = anchor["origin_idx"]
    candidate_idx = np.arange(0, fold["train_end"] + 1, dtype=np.int64)
    elig = eligible_candidates(candidate_idx, features, origin, cfg)

    z_cols = [f"zscore_{h}" for h in cfg.zscore_horizons]
    z_target = features.iloc[origin][z_cols].to_numpy()
    z_candidates = features.iloc[elig][z_cols].to_numpy()
    weights = np.array(fold["weights"], dtype=np.float64)
    n_eff = float(fold["n_eff"])

    distances = composite_distance(z_target, z_candidates, weights)
    probs = distances_to_probs(distances, target_n_eff=min(n_eff, elig.size))

    forward_at_elig = forward_returns_arr[elig]

    # Original B1 fit.
    correction, diag = fit_local_linear_correction(
        z_target=z_target,
        z_candidates=z_candidates,
        probs=probs,
        forward_returns=forward_at_elig,
    )

    # Leverage analysis.
    h_diag, lev_info = compute_leverage_scores(
        z_candidates, z_target, probs, forward_at_elig
    )
    h_argmax = int(np.argmax(h_diag))
    h_max = float(h_diag.max())

    # Refit with max-leverage candidate dropped.
    valid_indices = lev_info["valid_indices_into_orig"]
    drop_idx_in_elig = int(valid_indices[h_argmax])  # position in z_candidates/forward_at_elig
    mask = np.ones(elig.size, dtype=bool)
    mask[drop_idx_in_elig] = False
    correction_trim, diag_trim = fit_local_linear_correction(
        z_target=z_target,
        z_candidates=z_candidates[mask],
        probs=probs[mask],
        forward_returns=forward_at_elig[mask],
    )

    # Top-5 by leverage with their fwd returns + probs (for the report).
    top_lev_order = np.argsort(h_diag)[::-1][:5]
    top_lev_rows = []
    for j in top_lev_order:
        global_idx = elig[int(valid_indices[j])]
        top_lev_rows.append({
            "analog_idx": int(global_idx),
            "analog_date": features.index[global_idx].date().isoformat(),
            "leverage": float(h_diag[j]),
            "prob": float(probs[valid_indices[j]]),
            "forward_pct": float(np.expm1(forward_at_elig[valid_indices[j]]) * 100.0),
        })

    return {
        "anchor_date": anchor["anchor_date"],
        "section": section,
        "origin_idx": origin,
        "realized_60d_return_pct": anchor["realized_60d_return_pct"],
        "fold_index": fold["fold_index"],
        "weights": list(weights),
        "n_eff": n_eff,
        "n_eligible": int(elig.size),
        "original_b1": {
            "correction": float(correction),
            "matcher_mean": diag.matcher_mean,
            "predicted_mean": diag.predicted_mean,
            "clamp_hit": diag.clamp_hit,
            "beta_norm": diag.beta_norm,
            "n_used": diag.n_candidates_used,
            "n_dropped_no_forward": diag.n_candidates_dropped_no_forward,
            "horizon_drift_pct": float(np.expm1(correction) * 100.0),
        },
        "leverage": {
            "max": h_max,
            "argmax_global_idx": int(elig[drop_idx_in_elig]),
            "argmax_date": features.index[elig[drop_idx_in_elig]].date().isoformat(),
            "top_5": top_lev_rows,
        },
        "trimmed_b1": {
            "correction": float(correction_trim),
            "matcher_mean": diag_trim.matcher_mean,
            "predicted_mean": diag_trim.predicted_mean,
            "clamp_hit": diag_trim.clamp_hit,
            "horizon_drift_pct": float(np.expm1(correction_trim) * 100.0),
        },
        "trim_change_pct": (
            (correction_trim - correction) / correction * 100.0
            if abs(correction) > 1e-12 else 0.0
        ),
    }


def main() -> None:
    folds = load_fold_summaries(B1_RUN)
    cfg = Config.from_yaml(B1_RUN / "config.yaml")
    log_ret = load_returns(cfg)
    returns_arr = log_ret.to_numpy()
    features = compute_features(
        log_ret, halflife=cfg.ewma_halflife, horizons=cfg.zscore_horizons
    )
    forward_returns_arr = forward_logret_sums(returns_arr, cfg.forecast_horizon)

    anchors_data = json.loads(ANCHORS.read_text())
    all_anchors = []
    for sec in ("positive", "negative", "regime_coverage"):
        for a in anchors_data[sec]:
            all_anchors.append((a, sec))

    targets = set(B1_REGRESSIONS + B1_WINS)
    rows = []
    for a, sec in all_anchors:
        if a["anchor_date"] not in targets:
            continue
        fold = fold_for_origin(folds, a["origin_idx"])
        if fold is None:
            continue
        row = autopsy_anchor(a, sec, returns_arr, features, fold, cfg, forward_returns_arr)
        rows.append(row)
        ob = row["original_b1"]
        tb = row["trimmed_b1"]
        lv = row["leverage"]
        cls = "REG" if a["anchor_date"] in B1_REGRESSIONS else "WIN"
        print(f"  {a['anchor_date']:<14} {cls:<4} "
              f"corr_orig={ob['correction']:+.4f} drift={ob['horizon_drift_pct']:+.2f}% "
              f"clamp={'Y' if ob['clamp_hit'] else 'n'} "
              f"h_max={lv['max']:.3f} h_arg={lv['argmax_date']} "
              f"corr_trim={tb['correction']:+.4f} Δ={row['trim_change_pct']:+.1f}%")

    out = {
        "method": {
            "description": "V4.5.3 B1 β autopsy: reproduce B1's WLS fit at each regression anchor; "
                          "compute leverage scores; refit with max-leverage candidate dropped.",
            "b1_run": str(B1_RUN.relative_to(REPO)),
        },
        "b1_regressions": B1_REGRESSIONS,
        "b1_wins_compared": B1_WINS,
        "anchors": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
