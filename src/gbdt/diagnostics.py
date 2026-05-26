"""FS+HP diagnostic bundle for gbdt v1.

Each FS+HP iteration emits one bundle the agent reads to decide the next
iteration's feature prunes + HP changes. The bundle is JSON-serializable so
it lands cleanly in ``iterations.jsonl``.

Bundle contents (per V1_PLAN.md Stage 6):
- importance (native + permutation, side-by-side)
- correlation among top-K features
- val calibration (reliability curve points, Brier, Spiegelhalter Z)
- train-vs-val Brier gap
- learning curve (per-iter train + val loss)
- HP history table
- last-iteration delta attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from gbdt.calibration import spiegelhalter_z
from gbdt.uniqueness import (
    effective_sample_size,
    weighted_brier,
    weighted_spiegelhalter_z,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reliability_curve(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10,
) -> dict:
    """Return reliability-curve points + per-bin counts.

    Bins are equal-width over [0, 1]. Each bin reports:
    ``{bin_low, bin_high, mean_pred, frac_positive, count}``.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    points = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            points.append({
                "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                "mean_pred": None, "frac_positive": None, "count": 0,
            })
            continue
        points.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "mean_pred": float(p[mask].mean()),
            "frac_positive": float(y[mask].mean()),
            "count": n,
        })
    return {"n_bins": n_bins, "points": points}


def top_k_correlation(
    X: pd.DataFrame, importance: pd.Series, k: int = 50,
) -> dict[str, dict[str, float]]:
    """Pairwise correlation among the top-K features by importance.

    Returns a nested dict for JSON serialization; the inner keys are the
    top-K feature names and the values are the column's correlations to
    every other top-K feature.
    """
    if len(importance) == 0 or X.empty:
        return {}
    k = min(k, len(importance))
    top = importance.sort_values(ascending=False).head(k).index.tolist()
    top = [c for c in top if c in X.columns]
    if not top:
        return {}
    sub = X[top].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr().round(4)
    return {c: corr[c].dropna().to_dict() for c in corr.columns}


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticBundle:
    """One FS+HP iteration's diagnostic snapshot. Serializes to JSON."""

    iter: int
    hp: dict
    features: list[str]
    n_features: int

    train_brier: float
    val_brier: float
    train_val_gap: float
    eval_brier_provisional: float | None

    spiegelhalter_z: float
    spiegelhalter_p: float
    reliability: dict
    positive_prevalence_val: float
    positive_recall_val: float

    early_stop_iteration: int | None
    iteration_cap_hit: bool

    importance_native: dict[str, float]
    importance_permutation: dict[str, float] | None
    top_feature_correlation: dict[str, dict[str, float]]

    learning_curve: dict[str, list[float]]
    hp_history: list[dict[str, Any]] = field(default_factory=list)

    # Sample-uniqueness telemetry (LdP §4.4). When weights are uniform
    # (the legacy / opt-out case) ESS == row count and the inflation
    # ratio is 1.0.
    effective_sample_size_train: float | None = None
    effective_sample_size_val: float | None = None
    effective_sample_size_eval: float | None = None
    n_rows_train: int | None = None
    n_rows_val: int | None = None
    n_rows_eval: int | None = None

    rationale: str = ""
    delta_attribution: str = ""
    wall_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable. NaN/Infs get coerced to None."""

        def clean(x):
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [clean(v) for v in x]
            if isinstance(x, float):
                if np.isnan(x) or np.isinf(x):
                    return None
                return float(x)
            if isinstance(x, (np.floating, np.integer)):
                v = float(x)
                if np.isnan(v) or np.isinf(v):
                    return None
                return v
            return x

        d = {
            "iter": int(self.iter),
            "hp": clean(self.hp),
            "features": list(self.features),
            "n_features": int(self.n_features),
            "train_brier": clean(self.train_brier),
            "val_brier": clean(self.val_brier),
            "train_val_gap": clean(self.train_val_gap),
            "eval_brier_provisional": clean(self.eval_brier_provisional),
            "calibration": {
                "spiegelhalter_z": clean(self.spiegelhalter_z),
                "spiegelhalter_p": clean(self.spiegelhalter_p),
                "reliability": clean(self.reliability),
                "positive_prevalence_val": clean(self.positive_prevalence_val),
                "positive_recall_val": clean(self.positive_recall_val),
            },
            "early_stop_iteration": (
                int(self.early_stop_iteration)
                if self.early_stop_iteration is not None else None
            ),
            "iteration_cap_hit": bool(self.iteration_cap_hit),
            "importance_native": clean(self.importance_native),
            "importance_permutation": clean(self.importance_permutation),
            "top_feature_correlation": clean(self.top_feature_correlation),
            "learning_curve": clean(self.learning_curve),
            "hp_history": clean(self.hp_history),
            "sample_uniqueness": {
                "effective_sample_size_train": clean(self.effective_sample_size_train),
                "effective_sample_size_val": clean(self.effective_sample_size_val),
                "effective_sample_size_eval": clean(self.effective_sample_size_eval),
                "n_rows_train": (
                    int(self.n_rows_train) if self.n_rows_train is not None else None
                ),
                "n_rows_val": (
                    int(self.n_rows_val) if self.n_rows_val is not None else None
                ),
                "n_rows_eval": (
                    int(self.n_rows_eval) if self.n_rows_eval is not None else None
                ),
            },
            "rationale": str(self.rationale),
            "delta_attribution": str(self.delta_attribution),
            "wall_time_sec": clean(self.wall_time_sec),
        }
        return d


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def build_diagnostic_bundle(
    *,
    model,
    iter_idx: int,
    hp: dict,
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_eval: pd.DataFrame | None = None,
    y_eval: np.ndarray | None = None,
    w_train: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    w_eval: np.ndarray | None = None,
    hp_history: list[dict] | None = None,
    rationale: str = "",
    delta_attribution: str = "",
    wall_time_sec: float = 0.0,
    include_permutation: bool = True,
    top_k_corr: int = 50,
) -> DiagnosticBundle:
    """Build one bundle from a fitted ``model``.

    ``model`` is expected to expose ``predict_proba(X)`` + ``best_iteration``
    + ``evals_result`` + ``feature_importance(kind=...)`` — i.e. the
    :class:`gbdt.model.GBDTModel` surface.
    """
    p_train = model.predict_proba(X_train)
    p_val = model.predict_proba(X_val)
    # Weighted Brier when weights supplied (LdP §4.4 uniqueness); collapses
    # to unweighted brier_score_loss for uniform / None weights.
    if w_train is not None:
        train_brier = float(weighted_brier(y_train, p_train, w_train))
    else:
        train_brier = float(brier_score_loss(y_train, p_train))
    if w_val is not None:
        val_brier = float(weighted_brier(y_val, p_val, w_val))
    else:
        val_brier = float(brier_score_loss(y_val, p_val))
    gap = val_brier - train_brier

    eval_brier = None
    if X_eval is not None and y_eval is not None and len(y_eval):
        p_eval = model.predict_proba(X_eval)
        if w_eval is not None:
            eval_brier = float(weighted_brier(y_eval, p_eval, w_eval))
        else:
            eval_brier = float(brier_score_loss(y_eval, p_eval))

    if w_val is not None:
        z, pval = weighted_spiegelhalter_z(y_val, p_val, w_val)
    else:
        z, pval = spiegelhalter_z(y_val, p_val)
    rel = reliability_curve(y_val, p_val, n_bins=10)
    # Weighted prevalence when weights supplied; this is what the model
    # actually sees under sample_weight= fits (and what an honest
    # base-rate baseline should reflect).
    if w_val is not None and len(w_val) and float(np.sum(w_val)) > 0:
        pos_prev = float(np.sum(w_val * y_val) / np.sum(w_val))
    else:
        pos_prev = float(np.mean(y_val))
    # Recall at threshold 0.5 of the positive class on val
    pred_pos = (p_val >= 0.5).astype(int)
    true_pos = int(((pred_pos == 1) & (y_val == 1)).sum())
    pos_total = int((y_val == 1).sum())
    pos_recall = (true_pos / pos_total) if pos_total > 0 else 0.0

    best_iter = getattr(model, "best_iteration", None)
    iters_cap = hp.get("iterations", None)
    cap_hit = (
        best_iter is not None and iters_cap is not None
        and best_iter >= 0.9 * iters_cap
    )

    imp_native = model.feature_importance("native")
    imp_perm = None
    if include_permutation:
        try:
            imp_perm = model.feature_importance("permutation", X_val, y_val)
        except Exception:
            imp_perm = None

    corr = top_k_correlation(X_val, imp_native, k=top_k_corr)

    # Learning curve from CatBoost evals_result; trim to scalars
    learning_curve: dict[str, list[float]] = {}
    evals = getattr(model, "evals_result", None)
    if evals:
        for split, metrics in evals.items():
            for metric, vals in metrics.items():
                key = f"{split}_{metric}"
                learning_curve[key] = [float(v) for v in vals]

    ess_train = (
        float(effective_sample_size(w_train)) if w_train is not None
        else float(len(y_train))
    )
    ess_val = (
        float(effective_sample_size(w_val)) if w_val is not None
        else float(len(y_val))
    )
    ess_eval = None
    if y_eval is not None:
        ess_eval = (
            float(effective_sample_size(w_eval)) if w_eval is not None
            else float(len(y_eval))
        )

    return DiagnosticBundle(
        iter=iter_idx,
        hp=dict(hp),
        features=list(feature_names),
        n_features=len(feature_names),
        train_brier=train_brier,
        val_brier=val_brier,
        train_val_gap=gap,
        eval_brier_provisional=eval_brier,
        spiegelhalter_z=z,
        spiegelhalter_p=pval,
        reliability=rel,
        positive_prevalence_val=pos_prev,
        positive_recall_val=pos_recall,
        early_stop_iteration=int(best_iter) if best_iter is not None else None,
        iteration_cap_hit=cap_hit,
        importance_native=imp_native.to_dict(),
        importance_permutation=imp_perm.to_dict() if imp_perm is not None else None,
        top_feature_correlation=corr,
        learning_curve=learning_curve,
        hp_history=list(hp_history or []),
        effective_sample_size_train=ess_train,
        effective_sample_size_val=ess_val,
        effective_sample_size_eval=ess_eval,
        n_rows_train=int(len(y_train)),
        n_rows_val=int(len(y_val)),
        n_rows_eval=int(len(y_eval)) if y_eval is not None else None,
        rationale=rationale,
        delta_attribution=delta_attribution,
        wall_time_sec=float(wall_time_sec),
    )


__all__ = [
    "DiagnosticBundle",
    "build_diagnostic_bundle",
    "reliability_curve",
    "top_k_correlation",
]
