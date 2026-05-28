"""Per-iteration ``diagnose.json``-shaped payload from in-memory loop state.

V1.1 Phase 3 (``docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md`` § 0 Q5):
the per-iteration request the agent reads should be the **richer
``diagnose.json`` payload** produced by ``/gbdt-diagnose``, not the minimal
Phase-2 ``DiagnosticBundle`` dump.

The design tension Phase 3 resolves
-----------------------------------
``/gbdt-diagnose`` (``scripts/gbdt/diagnose.py::diagnose``) was built to analyze
a **completed cell's artifacts on disk**. It needs ``model.cbm`` + ``spec.yaml``
+ ``predictions/*.csv`` + ``iterations.jsonl``, and it **rebuilds the full
in-sample feature matrix from scratch** (6-28 min, universe-size-dependent) plus
re-fits 1D-PDP curves and interaction importances. None of that is appropriate
to run *synchronously inside every loop iteration* before the run has finalized:
at iteration N there is no ``model.cbm`` / calibrated ``predictions/`` on disk
yet, and the matrix rebuild is exactly the cost the loop must not pay per-iter.

So Phase 3 does **not** call ``diagnose()`` in the loop. Instead this module is
a **pure function** that assembles a ``diagnose.json``-*shaped* dict from the
**in-memory iteration results** the loop already has — chiefly the
:class:`~gbdt.diagnostics.DiagnosticBundle` the callback receives, which already
carries native feature importance, the train/val gap, calibration (Spiegelhalter
z + reliability), val prevalence, the learning curve, eval brier, and the
sample-uniqueness telemetry. It **reuses** the existing diagnose-family pure
helpers (no metric is re-derived):

- ``scripts.gbdt.diagnose.assess_overfit``  — the train/val-gap overfit read
  (identical threshold + semantics as the on-disk diagnose).
- ``scripts.gbdt.diagnose.prevalence_drift`` — the prevalence-drift /
  calibration-ceiling flag (same function the on-disk diagnose calls).
- ``gbdt.topk_diagnostics.compute_top_k_metrics`` /
  ``compute_per_ticker_hit_rate`` / ``compute_prediction_range`` — the corrected
  ``min(R(d), k)`` per-day P@k, per-ticker hit-rate, and prediction-range
  diagnostics (the exact functions the runner's report layer uses).
- ``scripts.gbdt.compute_r_precision.per_day_r_precision`` — per-day variable-K
  (K = R(d)) weighted R-precision (the canonical cross-cell comparison metric).

The signals that genuinely require the rebuilt matrix + the live model object
(in-sample marginal monotonicity, 1D-PDP model-monotonicity, interaction pairs,
the correlation heatmap) are **not** computed here — they'd cost the per-iter
matrix rebuild. The payload exposes ``full_diagnose_available: false`` +
``artifact_dir`` so the agent can run the full ``/gbdt-diagnose`` on demand for a
deeper look (plan § 0.5: "Include the artifact dir path for ad-hoc deeper
queries"). Native importance + the bundle's top-feature correlation ARE included
(they're already in-memory and free).

Per-day P@k + R-precision need a per-segment prediction frame
``(date, ticker, p_calibrated, y_true)``. The loop does not build one per
iteration (the runner only carves + calibrates predictions over the *best*
checkpoint at finalization — ``train.py``), so :func:`build_diagnose_payload`
takes an **optional** ``val_predictions`` frame: when supplied (e.g. a future
wiring that threads val preds, or a test) those sections are populated; when
absent (the current in-loop default) they carry ``available: false`` rather than
fabricated numbers.

Everything returned is NaN/Inf-safe + JSON-serializable (the same contract as
``DiagnosticBundle.to_dict``); :func:`_json_safe` mirrors that coercion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Reused diagnose-family pure helpers — DO NOT re-derive these metrics here.
from scripts.gbdt.compute_r_precision import per_day_r_precision
from scripts.gbdt.diagnose import (
    _OVERFIT_GAP_THR,
    assess_overfit,
    prevalence_drift,
)
from gbdt.topk_diagnostics import (
    compute_per_ticker_hit_rate,
    compute_prediction_range,
    compute_top_k_metrics,
)

# Bumped alongside the request-schema version when this payload's shape changes
# breakingly (plan § 12 R6). Kept distinct so a reader can tell which payload
# generation produced a given request file even if the envelope is unchanged.
DIAGNOSE_PAYLOAD_VERSION = "v1"

# Default per-day K ladder for the in-loop P@k block (matches the canonical
# report ladder + the compute_r_precision CLI default).
_DEFAULT_K_VALUES = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# NaN/Inf-safe JSON coercion (same contract as DiagnosticBundle.to_dict)
# ---------------------------------------------------------------------------


def _json_safe(x: Any) -> Any:
    """Recursively coerce ``x`` to a JSON-serializable, NaN/Inf-free structure.

    Mirrors :meth:`gbdt.diagnostics.DiagnosticBundle.to_dict`'s ``clean``:
    NaN/Inf floats -> ``None``; numpy scalars -> Python scalars; numpy arrays /
    pandas containers -> lists/dicts; everything else passes through.
    """
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (pd.Series,)):
        return {str(k): _json_safe(v) for k, v in x.to_dict().items()}
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(x, float):
        return None if (np.isnan(x) or np.isinf(x)) else x
    return x


# ---------------------------------------------------------------------------
# Helpers (small, local, no I/O)
# ---------------------------------------------------------------------------


def _segment_prevalence(bundle: Any) -> dict[str, float | None]:
    """Per-segment positive prevalence from the in-memory bundle.

    Only ``val`` prevalence is carried on the per-iteration bundle
    (``positive_prevalence_val``). The on-disk diagnose reads train/val/eval/
    test prevalence from saved predictions, which don't exist mid-loop, so the
    other segments are ``None`` here — :func:`prevalence_drift` handles a
    short/partial mapping gracefully (it filters non-finite/None values).
    """
    val_prev = getattr(bundle, "positive_prevalence_val", None)
    return {
        "train": None,
        "val": float(val_prev) if val_prev is not None else None,
        "eval": None,
        "test": None,
    }


def _importance_summary(imp: dict[str, float], top_n: int) -> list[dict]:
    """Top-``top_n`` features by native importance, descending."""
    items = sorted(
        ((str(f), float(v)) for f, v in (imp or {}).items()),
        key=lambda kv: -kv[1],
    )
    return [{"feature": f, "importance": v} for f, v in items[:top_n]]


def _pruned_summary(imp: dict[str, float], importance_threshold: float) -> dict:
    """Kept/pruned split by the importance threshold (the diagnose convention).

    This is the cheap, model-free part of the on-disk diagnose's pruned-feature
    investigation: how many features fall below the importance floor. The
    redundancy-vs-noise verdict (which needs the rebuilt in-sample matrix's
    Spearman correlations) is deferred to the full ``/gbdt-diagnose``.
    """
    imp = imp or {}
    kept = [f for f, v in imp.items() if float(v) >= importance_threshold]
    pruned = [f for f, v in imp.items() if float(v) < importance_threshold]
    return {
        "kept_count": len(kept),
        "pruned_count": len(pruned),
        "importance_threshold": float(importance_threshold),
        "pruned_features": sorted(pruned),
    }


def _per_day_p_at_k_from_topk(topk: dict) -> dict:
    """Re-key ``compute_top_k_metrics`` per-day block into the diagnose shape.

    Drops the per-k ``lift`` field (CLAUDE.md reporting convention: no
    lift-as-a-column / -value in structured payloads; lift is computed on
    demand from ``p_at_k`` + ``base_rate``).
    """
    out: dict[str, dict] = {}
    for k, blk in (topk.get("per_day") or {}).items():
        out[str(k)] = {
            "p_at_k": blk.get("p_at_k"),
            "n_positives_in_picks": blk.get("n_positives_in_picks"),
            "n_denom": blk.get("n_denom"),
            "n_days_R_lt_k": blk.get("n_days_R_lt_k"),
            "n_days_total": blk.get("n_days_total"),
        }
    return out


def _tuning_guidance(overfit: dict, drift: dict, pruned: dict) -> list[str]:
    """The auto-flagged tuning-guidance lines, mirroring the on-disk diagnose.

    Same playbook rules the on-disk ``diagnose._render_report`` emits, applied
    to the in-memory overfit + prevalence-drift reads. Pure-text, no I/O.
    """
    lines: list[str] = []
    no_of = overfit.get("no_overfit")
    if no_of is True:
        lines.append(
            f"NO OVERFIT (train/val gap {overfit.get('train_val_gap')} "
            f"<= {_OVERFIT_GAP_THR}; early-stop is orthogonal) -> do NOT prune "
            f"for regularization (rule 1). FS will be neutral-to-harmful."
        )
    elif no_of is False:
        lines.append(
            f"OVERFIT signal (train/val gap {overfit.get('train_val_gap')} "
            f"> {_OVERFIT_GAP_THR}, val worse than train) -> pruning / "
            f"regularization (raise l2, drop depth) may help (rule 1)."
        )
    if drift.get("drift_flag") or drift.get("monotone_decline"):
        lines.append(
            f"PREVALENCE DRIFT across segments (spread {drift.get('spread')}"
            f"{', monotone decline' if drift.get('monotone_decline') else ''}) "
            f"-> calibration ceiling likely; the lever is recency / "
            f"regime-conditional calibration, OUT of the FS/HP loop (rule 5)."
        )
    if pruned.get("pruned_count"):
        lines.append(
            f"{pruned['pruned_count']} feature(s) below the importance floor "
            f"(<{pruned['importance_threshold']}). importance approx 0 usually "
            f"means redundant, not unrelated (rule 2) -> run /gbdt-diagnose for "
            f"the redundancy-vs-noise verdict (needs the in-sample matrix)."
        )
    return lines


# ---------------------------------------------------------------------------
# The pure function
# ---------------------------------------------------------------------------


def build_diagnose_payload(
    bundle: Any,
    *,
    artifact_dir: str | None = None,
    cell: dict | None = None,
    val_predictions: pd.DataFrame | None = None,
    top_n: int = 30,
    importance_threshold: float = 0.01,
    k_values: tuple[int, ...] = _DEFAULT_K_VALUES,
    per_ticker_k: int = 5,
) -> dict:
    """Assemble a ``diagnose.json``-shaped dict from in-memory iteration state.

    Parameters
    ----------
    bundle:
        The in-memory :class:`~gbdt.diagnostics.DiagnosticBundle` for the
        current iteration (the object the FS+HP callback receives). Read for
        importance, train/val gap, calibration, prevalence, learning curve,
        eval brier, and sample-uniqueness.
    artifact_dir:
        Absolute path to the cell's artifact dir, surfaced so the agent can run
        the full ``/gbdt-diagnose`` for the matrix-dependent analyses (plan
        § 0.5). ``None`` mid-run before the dir is known.
    cell:
        The ``(universe, direction, threshold_pct, horizon_days, max_drawdown)``
        identity, surfaced verbatim for the agent. ``None`` if unavailable.
    val_predictions:
        Optional per-segment prediction frame ``(date, ticker, p_calibrated,
        y_true)``. When supplied the per-day P@k / R-precision / per-ticker /
        prediction-range sections are populated; when ``None`` (the current
        in-loop default — the loop does not carve per-iter predictions) those
        sections carry ``available: false`` instead of fabricated numbers.
    top_n, importance_threshold, k_values, per_ticker_k:
        Knobs matching the on-disk diagnose / report defaults.

    Returns
    -------
    dict
        NaN/Inf-safe + JSON-serializable. Shaped like ``diagnose.json`` (with
        an explicit ``payload_version`` + ``source: in_memory_iteration`` so a
        reader can tell it apart from the on-disk artifact).
    """
    imp = dict(getattr(bundle, "importance_native", {}) or {})

    # ---- overfit read (reuses diagnose.assess_overfit) ----
    gap = getattr(bundle, "train_val_gap", None)
    overfit = {
        "train_val_gap": gap,
        "early_stop": getattr(bundle, "early_stop_iteration", None),
        "iteration_cap_hit": getattr(bundle, "iteration_cap_hit", None),
        "no_overfit": assess_overfit(gap),
    }

    # ---- prevalence drift (reuses diagnose.prevalence_drift) ----
    seg_prev = _segment_prevalence(bundle)
    drift = prevalence_drift(seg_prev)

    # ---- feature importance + cheap pruned split ----
    top_features = _importance_summary(imp, top_n)
    pruned = _pruned_summary(imp, importance_threshold)

    # ---- calibration (straight from the bundle) ----
    calibration = {
        "spiegelhalter_z": getattr(bundle, "spiegelhalter_z", None),
        "spiegelhalter_p": getattr(bundle, "spiegelhalter_p", None),
        "positive_prevalence_val": getattr(bundle, "positive_prevalence_val", None),
        "positive_recall_val": getattr(bundle, "positive_recall_val", None),
        "reliability": getattr(bundle, "reliability", {}),
    }

    # ---- per-day P@k + R-precision + per-ticker + prediction-range ----
    # These need a (date, ticker, p_calibrated, y_true) frame. The loop does
    # not build one per iteration, so default to "unavailable" markers and only
    # populate when a caller threads a frame (forward-compat / tests).
    if (
        val_predictions is not None
        and not val_predictions.empty
        and {"date", "ticker", "p_calibrated", "y_true"}.issubset(
            val_predictions.columns
        )
    ):
        topk = compute_top_k_metrics(val_predictions, k_values=k_values)
        per_day_p_at_k = {
            "available": True,
            "base_rate": topk.get("base_rate"),
            "by_k": _per_day_p_at_k_from_topk(topk),
        }
        rprec_raw = per_day_r_precision(val_predictions)
        r_precision = {
            "available": True,
            "weighted": rprec_raw.get("r_precision_weighted"),
            "mean_unweighted": rprec_raw.get("r_precision_mean_unweighted"),
            "base_rate_weighted": rprec_raw.get("base_rate_weighted"),
            "n_days_with_positives": rprec_raw.get("n_days_with_positives"),
            "per_day_quantiles": rprec_raw.get("per_day_rprec_quantiles"),
            "r_distribution": rprec_raw.get("r_distribution"),
        }
        per_ticker = {
            "available": True,
            **compute_per_ticker_hit_rate(val_predictions, k=per_ticker_k),
        }
        prediction_range = {
            "available": True,
            **compute_prediction_range(val_predictions),
        }
    else:
        _na = {
            "available": False,
            "reason": (
                "no per-segment prediction frame threaded to the in-loop "
                "payload; run /gbdt-diagnose on artifact_dir post-run for the "
                "calibrated per-day picks"
            ),
        }
        per_day_p_at_k = dict(_na)
        r_precision = dict(_na)
        per_ticker = dict(_na)
        prediction_range = dict(_na)

    payload = {
        "payload_version": DIAGNOSE_PAYLOAD_VERSION,
        "source": "in_memory_iteration",
        # The agent runs the full /gbdt-diagnose against artifact_dir for the
        # matrix-dependent analyses (monotonicity, PDP, interaction pairs,
        # correlation heatmap, redundancy verdict) that are too expensive to
        # rebuild per iteration (plan § 0.5).
        "full_diagnose_available": False,
        "artifact_dir": artifact_dir,
        "deferred_to_full_diagnose": [
            "marginal_monotonicity",
            "model_pdp_monotonicity",
            "interaction_pairs",
            "correlation_heatmap",
            "pruned_redundancy_verdict",
        ],
        "cell": cell or {},
        "iter": getattr(bundle, "iter", None),
        "n_features_in_model": getattr(bundle, "n_features", None),
        "features": list(getattr(bundle, "features", []) or []),
        "hp": dict(getattr(bundle, "hp", {}) or {}),
        "metrics": {
            "train_brier": getattr(bundle, "train_brier", None),
            "val_brier": getattr(bundle, "val_brier", None),
            "train_val_gap": gap,
            "eval_brier_provisional": getattr(
                bundle, "eval_brier_provisional", None
            ),
        },
        "overfit": overfit,
        "prevalence_by_segment": seg_prev,
        "prevalence_drift": drift,
        "calibration": calibration,
        "top_features": top_features,
        "feature_importance": imp,
        "top_feature_correlation": getattr(bundle, "top_feature_correlation", {}),
        "pruned_summary": pruned,
        "per_day_p_at_k": per_day_p_at_k,
        "r_precision": r_precision,
        "per_ticker_hit_rate": per_ticker,
        "prediction_range": prediction_range,
        "learning_curve": getattr(bundle, "learning_curve", {}),
        "sample_uniqueness": {
            "effective_sample_size_train": getattr(
                bundle, "effective_sample_size_train", None
            ),
            "effective_sample_size_val": getattr(
                bundle, "effective_sample_size_val", None
            ),
            "n_rows_train": getattr(bundle, "n_rows_train", None),
            "n_rows_val": getattr(bundle, "n_rows_val", None),
        },
        "tuning_guidance": _tuning_guidance(overfit, drift, pruned),
    }
    return _json_safe(payload)


__all__ = [
    "DIAGNOSE_PAYLOAD_VERSION",
    "build_diagnose_payload",
]
