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
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from gbdt.calibration import (
    apply_calibrator,
    conditional_isotonic,
    isotonic_always,
    platt_calibration,
    spiegelhalter_z,
)
from gbdt.uniqueness import (
    effective_sample_size,
    weighted_brier,
    weighted_spiegelhalter_z,
)


# ---------------------------------------------------------------------------
# V1.3 Option A — anti-AUC flag (canonical CSV lookup at iter_0)
# ---------------------------------------------------------------------------

# Tightened thresholds per V1.3 plan § 0 D4 (preempt false-positives for the
# loop-doctrine auto-disables in fs_hp_loop.best_checkpoint / inner_stop_check).
# CLAUDE.md "What not to do — gbdt" anti-AUC bullet is updated to the same
# numbers — single source of truth.
ANTI_AUC_FLAG_AUC_LOW = 0.46
ANTI_AUC_FLAG_AUC_HIGH = 0.54
ANTI_AUC_FLAG_RP10_LIFT_MIN = 1.8


def compute_anti_auc_flag(
    sweep_row: dict | None,
) -> Literal["true", "false", "unknown"]:
    """V1.3 Option A — apply the cell-shape anti-AUC rule (§ 3.3 / D4).

    A cell is flagged ``"true"`` when its canonical sweep row satisfies
    ``AUC ∈ [0.46, 0.54]`` AND ``R-Precision@10 lift > 1.8x`` (lift
    relative to the segment base rate). Returns ``"unknown"`` when the
    sweep row is None (new cell, lookup miss) so the loop's auto-disables
    safely default to NOT firing.

    Tighter than the CLAUDE.md compound rule (``[0.45, 0.55] + lift >
    1.5x``) by design: this flag drives auto-disables (skip the L1
    tie-break, skip the val_brier auto-plateau), so a false-positive is
    costlier here than in the human-facing CLAUDE.md narrative.
    """
    if sweep_row is None:
        return "unknown"
    try:
        auc = float(sweep_row["AUC"])
        rp10 = float(sweep_row["R_precision_at_10"])
        base = float(sweep_row["base_rate"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if base <= 0.0:
        return "unknown"
    lift10 = rp10 / base
    if (
        ANTI_AUC_FLAG_AUC_LOW <= auc <= ANTI_AUC_FLAG_AUC_HIGH
        and lift10 > ANTI_AUC_FLAG_RP10_LIFT_MIN
    ):
        return "true"
    return "false"


def compute_degenerate_sink_warning(
    val_brier: float | None,
    weighted_base_rate_brier: float | None,
    threshold: float,
) -> bool:
    """V1.3 Option A — degenerate-sink warning (§ 3.4 / D5).

    Returns True when ``val_brier <= threshold * weighted_base_rate_brier``
    — i.e. the model's val Brier is within ``threshold-1.0`` of the trivial
    constant-predictor's Brier. Default ``threshold=1.05`` (5% above
    trivial) catches the cell-5 γ≥5 / alpha=10 regimes without firing on
    healthy near-degenerate models (cell-5 lean+γ=1 ratio 1.079 → no
    trigger). Spec-overridable via
    ``backend.fs_hp_loop.degenerate_sink_threshold``.

    Both arguments may be None (e.g. iter_0 not yet computed, or weighted
    base-rate denominator unavailable) — in that case the function returns
    False (no warning) so the doctrine signals are conservative.
    """
    if val_brier is None or weighted_base_rate_brier is None:
        return False
    if weighted_base_rate_brier <= 0.0:
        return False
    return float(val_brier) <= float(threshold) * float(weighted_base_rate_brier)


def _r_precision_at_k_from_arrays(
    dates: np.ndarray,
    tickers: np.ndarray,
    p_calibrated: np.ndarray,
    y_true: np.ndarray,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[int, float]:
    """V1.3 Option A — canonical R-Precision@K on aligned per-row arrays.

    Per-day fixed K, macro-averaged across days where ``R_q > 0``:

        R-Precision@K = (1/Q) · Σ_q  r_q / min(K, R_q)

    Tie-break inside each day: ``(p_calibrated desc, ticker asc)`` stable
    mergesort — same convention as
    :func:`gbdt.topk_diagnostics.compute_top_k_metrics` and the canonical
    CSV ``results/gbdt/data/r_precision_at_k.csv`` (see
    ``.claude/memories/project-r-precision-methodology.md``).

    Empty arrays / no R_q > 0 day → empty dict (the caller surfaces this
    as ``eval_r_precision_at_k = None``).

    Uses a tiny per-call DataFrame for the groupby — same scaffolding as
    ``topk_diagnostics.compute_top_k_metrics`` so the result is byte-
    identical to the canonical CSV's per-cell numbers.
    """
    n = len(p_calibrated)
    if n == 0:
        return {}
    df = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "p_calibrated": p_calibrated,
        "y_true": y_true.astype(int),
    })
    sorted_df = df.sort_values(
        ["date", "p_calibrated", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    grouped = sorted_df.groupby("date", sort=False)
    per_day_r = grouped["y_true"].sum().astype(int)
    qualifying_days = per_day_r[per_day_r > 0].index
    if len(qualifying_days) == 0:
        return {}

    out: dict[int, float] = {}
    for k in k_values:
        picks = grouped.head(int(k))
        per_day_caught = picks.groupby("date", sort=False)["y_true"].sum().astype(int)
        per_day_caught = per_day_caught.reindex(qualifying_days, fill_value=0)
        per_day_R = per_day_r.reindex(qualifying_days)
        per_day_denom = per_day_R.clip(upper=int(k))
        per_day_ratio = per_day_caught.astype(float) / per_day_denom.astype(float)
        out[int(k)] = float(per_day_ratio.mean())
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_in_loop_calibrator(
    *,
    y_val: np.ndarray,
    p_val_raw: np.ndarray,
    method: str,
    z_threshold: float,
):
    """Fit a per-iter calibrator on ``(y_val, p_val_raw)`` — bug #222 fix.

    Mirrors the finalization-side dispatch in
    :func:`gbdt.train.walk_forward_train` exactly (same valid methods, same
    Spiegelhalter Z gate). Returns the calibrator object (or ``None`` for
    the native / Spiegelhalter-pass-through path) that
    :func:`gbdt.calibration.apply_calibrator` consumes.

    Unsupported methods are treated like ``"native"`` (pass-through) — never
    raise: a bundle-build crash would degrade the agent's per-iter signal
    to "no value". The calling code at ``build_diagnostic_bundle`` already
    wraps the whole eval R-p@K computation in a try/except for defense in
    depth.
    """
    if method == "native":
        return None
    if method == "conditional_isotonic":
        return conditional_isotonic(
            y_val, p_val_raw, z_threshold=z_threshold,
        ).calibrator
    if method == "isotonic_always":
        return isotonic_always(
            y_val, p_val_raw, z_threshold=z_threshold,
        ).calibrator
    if method == "platt":
        return platt_calibration(
            y_val, p_val_raw, z_threshold=z_threshold,
        ).calibrator
    # Unknown method: pass through, do not crash. The finalization path
    # raises NotImplementedError in this case — the loop should not.
    return None


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

    # V1.3 Option A additions (plan § 2.1 / D7) — all additive with sensible
    # defaults so non-anti-AUC cells and existing call sites are
    # backward-compatible. Populated by build_diagnostic_bundle when X_eval +
    # sweep_row are wired through walk_forward_train.
    eval_r_precision_at_k: dict[int, float] | None = None
    anti_auc_flag: Literal["true", "false", "unknown"] = "unknown"
    degenerate_sink_warning: bool = False
    weighted_base_rate_brier: float | None = None
    eval_segment_size: int | None = None

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
            # V1.3 Option A surface (plan § 2.1 / D7) — agent reads these in
            # loop/iter_<N>_request.json. eval_r_precision_at_k stays None
            # when the eval segment wasn't wired (older callers) or is too
            # slim to contain any R_q > 0 day; anti_auc_flag defaults
            # "unknown" (no sweep row → no auto-disable).
            "eval_r_precision_at_k": (
                {str(k): clean(v) for k, v in self.eval_r_precision_at_k.items()}
                if self.eval_r_precision_at_k is not None else None
            ),
            "anti_auc_flag": str(self.anti_auc_flag),
            "degenerate_sink_warning": bool(self.degenerate_sink_warning),
            "weighted_base_rate_brier": clean(self.weighted_base_rate_brier),
            "eval_segment_size": (
                int(self.eval_segment_size)
                if self.eval_segment_size is not None else None
            ),
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
    # V1.3 Option A additions (plan § 3.1 / D1+D2).
    # The eval-segment MultiIndex carries (date, ticker) — required for the
    # per-day R-Precision@K computation. The sweep_row is looked up once at
    # iter_0 (plan § 3.3 / D3) by the caller and threaded through every call
    # for the run so anti_auc_flag is constant. The degenerate_sink_threshold
    # comes from backend.fs_hp_loop.degenerate_sink_threshold (default 1.05,
    # plan § 0 D5).
    mi_eval: pd.MultiIndex | None = None,
    sweep_row: dict | None = None,
    degenerate_sink_threshold: float = 1.05,
    # Bug #222 fix — eval R-p@K MUST be computed on calibrated predictions to
    # match canonical CSV scoring (`compute_r_precision.py` +
    # `[[project-r-precision-methodology]]`). Pre-fix, the bundle computed
    # R-p@K on raw model output under the FALSE assumption that isotonic
    # monotonicity preserves rank order — true mathematically, but FALSE on
    # piecewise-constant isotonic + tiny models, where many distinct raw
    # probabilities map to a single calibrated value and the canonical
    # ``(p_calibrated desc, ticker asc)`` tie-break dominates (see memo
    # ``_222`` § "The calibration-collapse pathology"). The in-loop
    # calibrator is fit fresh per-iter on (y_val, p_val_raw) using the
    # same method the finalization step will use; it is NOT the
    # finalization calibrator (that fit happens once, on the best-checkpoint
    # model's val predictions).
    calibration_method: str = "conditional_isotonic",
    calibration_z_threshold: float = 2.0,
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

    # ----- V1.3 Option A signals (plan § 3) ---------------------------------
    # weighted_base_rate_brier — the trivial constant-predictor's Brier on val
    # under uniqueness weights (= p_val * (1 - p_val)). Constant for the run
    # (re-derived each iter from the same val segment / same weights, so byte-
    # identical iter-over-iter) but included in every bundle for context.
    weighted_base_rate_brier: float | None = None
    if w_val is not None and len(w_val) and float(np.sum(w_val)) > 0:
        p_val_base = float(np.sum(w_val * y_val) / np.sum(w_val))
    elif len(y_val):
        p_val_base = float(np.mean(y_val))
    else:
        p_val_base = None
    if p_val_base is not None:
        weighted_base_rate_brier = float(p_val_base * (1.0 - p_val_base))

    # degenerate_sink_warning — val_brier within threshold * trivial baseline.
    degenerate_sink_warning = compute_degenerate_sink_warning(
        val_brier, weighted_base_rate_brier, degenerate_sink_threshold,
    )

    # eval_r_precision_at_k — end-of-iter predict on the carved eval segment
    # (D2). Cost: < 1s on hist+nj=8 regardless. Bug #222 fix: the eval R-p@K
    # MUST be computed on CALIBRATED predictions to match canonical CSV
    # scoring. The earlier "isotonic is monotone, raw rank order is the
    # same" justification is FALSE on piecewise-constant isotonic + tiny
    # models (218 distinct raw values → 7 distinct calibrated values; the
    # canonical ``(p_calibrated desc, ticker asc)`` tie-break then chooses
    # alphabetically among many ties). See memo ``_222`` § "The
    # calibration-collapse pathology" for the worked example.
    #
    # We fit the in-loop calibrator fresh per-iter on ``(y_val, p_val)``
    # using the spec's ``calibration_method`` — this is NOT the finalization
    # calibrator (that's fit once at the end on the best-checkpoint model).
    # The compute cost is negligible (~ms): isotonic regression is O(n log n)
    # on val size, Platt is one LR fit. The fit is read-only with respect
    # to the model.
    eval_r_precision_at_k: dict[int, float] | None = None
    eval_segment_size: int | None = None
    if X_eval is not None and y_eval is not None and len(y_eval) and mi_eval is not None:
        eval_segment_size = int(len(y_eval))
        try:
            p_eval_raw = np.asarray(model.predict_proba(X_eval), dtype=float)
            # Fit an in-loop calibrator on val so we can apply it to eval.
            # The same method the finalization step will use is fit here
            # (separately) so the iter signal aligns with canonical scoring.
            in_loop_calibrator = _fit_in_loop_calibrator(
                y_val=np.asarray(y_val, dtype=int),
                p_val_raw=np.asarray(p_val, dtype=float),
                method=calibration_method,
                z_threshold=calibration_z_threshold,
            )
            p_eval_calibrated = apply_calibrator(p_eval_raw, in_loop_calibrator)
            # mi_eval is a (date, ticker) MultiIndex of length len(y_eval).
            dates = mi_eval.get_level_values("date").to_numpy()
            tickers = mi_eval.get_level_values("ticker").to_numpy()
            rp_at_k = _r_precision_at_k_from_arrays(
                dates=dates,
                tickers=tickers,
                p_calibrated=np.asarray(p_eval_calibrated, dtype=float),
                y_true=np.asarray(y_eval, dtype=int),
            )
            eval_r_precision_at_k = rp_at_k or None
        except Exception:
            # Never let an eval-side compute crash the bundle build; the
            # field stays None and the agent / doctrine sees "no signal".
            eval_r_precision_at_k = None

    # anti_auc_flag — constant for the run; computed from the canonical sweep
    # CSV row passed in by the caller at every iter (plan § 3.3 / D3).
    anti_auc_flag = compute_anti_auc_flag(sweep_row)

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
        # V1.3 Option A
        eval_r_precision_at_k=eval_r_precision_at_k,
        anti_auc_flag=anti_auc_flag,
        degenerate_sink_warning=degenerate_sink_warning,
        weighted_base_rate_brier=weighted_base_rate_brier,
        eval_segment_size=eval_segment_size,
    )


__all__ = [
    "DiagnosticBundle",
    "build_diagnostic_bundle",
    "reliability_curve",
    "top_k_correlation",
    # V1.3 Option A
    "compute_anti_auc_flag",
    "compute_degenerate_sink_warning",
    "ANTI_AUC_FLAG_AUC_LOW",
    "ANTI_AUC_FLAG_AUC_HIGH",
    "ANTI_AUC_FLAG_RP10_LIFT_MIN",
]
