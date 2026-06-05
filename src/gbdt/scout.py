"""V1.3 Option B — scout phase (single-knob response curves).

Phase 1.5 of ``/gbdt-experiment``'s lifecycle (between data build and iter_0).
Runs 8 single-knob curves × ~4–5 values each + a defaults zeroth config
(~41 fits/cell) and emits per-config metrics so the combine phase
(:func:`lexicographic_winner` in default mode, or the agent in
``agent_file_protocol`` mode) can pick iter_0's ``hp_starting``.

D1 — knob grid (XGBoost canonical names):
- ``max_depth``: {2, 3, 4, 6, 8}
- ``eta``: {0.01, 0.05, 0.1, 0.2, 0.3}
- ``colsample_bytree``: {0.3, 0.4, 0.5, 0.7, 1.0}
- ``min_child_weight``: {1, 5, 10, 25}
- ``gamma``: {0, 0.1, 0.5, 1.0}  (skipped for CatBoost — no direct map)
- ``alpha``: {0, 0.1, 0.5, 1.0}
- ``subsample``: {0.5, 0.7, 0.85, 1.0}
- ``scale_pos_weight``: {1, sqrt(neg/pos), neg/pos}

D2 — per-config records carry ``(knob_name, knob_value, val_brier, train_brier,
eval_R_p_at_K, train_val_gap, spiegelhalter_z, fit_seconds, status)``. No
runner-side oracle pick at scout time.

D3a — per-config fit-timeout (30 s XGBoost / 120 s CatBoost), soft wall-clock
cap (5 min XGBoost / 30 min CatBoost). Timed-out / past-cap configs get
``status = "timeout"`` and are excluded from the oracle.

D9.1.A — defaults are scout's zeroth config (always included; 1 extra fit).

D9.2.A — :func:`detect_degenerate_sink` flags the val_brier ≈ baseline +
train_val_gap ≈ 0 pathology; default-mode fallback path discards the scout
winner and uses defaults for iter_0 when this fires.

D12 — :func:`lexicographic_winner` is the default-mode auto-compose: per-knob
argmax by ``eval_R_p_at_K`` with lex priority
``R-p@1 > R-p@3 > R-p@5 > R-p@10 > R-p@20``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# D1 — knob grid (XGBoost canonical names)
# ---------------------------------------------------------------------------


# Defaults grid per V1.3 Option B plan § 4 D1. The spec can override per-knob
# values via ``backend.scout.grid``.
DEFAULT_SCOUT_GRID: dict[str, tuple] = {
    "max_depth": (2, 3, 4, 6, 8),
    "eta": (0.01, 0.05, 0.1, 0.2, 0.3),
    "colsample_bytree": (0.3, 0.4, 0.5, 0.7, 1.0),
    "min_child_weight": (1, 5, 10, 25),
    "gamma": (0.0, 0.1, 0.5, 1.0),
    "alpha": (0.0, 0.1, 0.5, 1.0),
    "subsample": (0.5, 0.7, 0.85, 1.0),
    # scale_pos_weight is built dynamically (depends on neg/pos ratio); use
    # the SENTINELS here and resolve them at build_grid() time.
    "scale_pos_weight": ("1", "sqrt(neg/pos)", "neg/pos"),
}


# D1 — backend translation table (XGBoost → CatBoost). ``gamma`` has no direct
# map → CatBoost grid omits the gamma knob.
_XGB_TO_CATBOOST: dict[str, str] = {
    "max_depth": "depth",
    "eta": "learning_rate",
    "colsample_bytree": "rsm",
    "min_child_weight": "min_data_in_leaf",
    "alpha": "l2_leaf_reg",       # semantic-similar regularization (not exact)
    "subsample": "subsample",
    "scale_pos_weight": "class_weights",   # special-case (dict form)
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoutConfig:
    """One row of the scout grid — a single-knob overlay on the defaults
    baseline. The zeroth ``defaults`` config carries ``knob_name="defaults"``
    + an empty ``hp_overlay`` so the scout's "did we beat defaults?" comparator
    is auto-emitted (D9.1.A)."""

    knob_name: str
    knob_value: Any            # int / float / str / dict (e.g. CatBoost class_weights)
    hp_overlay: dict           # the {knob: value} dict that overlays on defaults

    def to_dict(self) -> dict:
        return {
            "knob_name": self.knob_name,
            "knob_value": self.knob_value,
            "hp_overlay": dict(self.hp_overlay),
        }


@dataclass(frozen=True)
class ScoutResult:
    """One fit's outcome — per-config metrics emitted by the scout.

    ``status``:
    - ``"ok"`` — fit completed; all metrics populated.
    - ``"timeout"`` — per-config timeout exceeded OR soft wall-clock cap hit
      before fit started; excluded from the oracle.
    - ``"error"`` — fit raised; ``error_message`` carries the exception repr.
    """

    config: ScoutConfig
    val_brier: float | None
    train_brier: float | None
    eval_R_p_at_K: dict[int, float] | None     # {1, 3, 5, 10, 20} → R-p@K
    train_val_gap: float | None
    spiegelhalter_z: float | None
    fit_seconds: float
    status: Literal["ok", "timeout", "error"]
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "val_brier": _clean_float(self.val_brier),
            "train_brier": _clean_float(self.train_brier),
            "eval_R_p_at_K": (
                {str(k): _clean_float(v) for k, v in self.eval_R_p_at_K.items()}
                if self.eval_R_p_at_K is not None else None
            ),
            "train_val_gap": _clean_float(self.train_val_gap),
            "spiegelhalter_z": _clean_float(self.spiegelhalter_z),
            "fit_seconds": float(self.fit_seconds),
            "status": str(self.status),
            "error_message": self.error_message,
        }


def _clean_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


# ---------------------------------------------------------------------------
# Backend translation
# ---------------------------------------------------------------------------


def _translate_for_backend(hp_overlay: dict, backend: str) -> dict:
    """Translate an XGBoost-named ``hp_overlay`` to ``backend`` vocabulary.

    XGBoost → CatBoost mappings (V1.3 Option B plan § 4 D1):

    - ``max_depth`` → ``depth``
    - ``eta`` → ``learning_rate``
    - ``colsample_bytree`` → ``rsm``
    - ``min_child_weight`` → ``min_data_in_leaf``
    - ``gamma`` → (no direct map; **dropped silently** — the CatBoost grid
      omits this knob per ``build_grid``)
    - ``alpha`` → ``l2_leaf_reg`` (semantic-similar regularization)
    - ``subsample`` → ``subsample``
    - ``scale_pos_weight`` → ``class_weights`` (serialized ``{0: 1.0, 1: ratio}``
      dict)

    Pass-through for ``backend == "xgboost"``.
    """
    if backend == "xgboost":
        return dict(hp_overlay)
    if backend != "catboost":
        # Unknown backend → pass-through; caller's HP validator surfaces the
        # mismatch downstream.
        return dict(hp_overlay)
    out: dict = {}
    for k, v in hp_overlay.items():
        if k == "gamma":
            continue            # no direct CatBoost analog
        if k == "scale_pos_weight":
            # CatBoost expects class_weights as {0: w0, 1: w1}. Map
            # spw=1 → balanced; spw=ratio → {0:1.0, 1:ratio}.
            try:
                ratio = float(v)
            except (TypeError, ValueError):
                continue
            out["class_weights"] = {0: 1.0, 1: ratio}
            continue
        cb_name = _XGB_TO_CATBOOST.get(k)
        if cb_name is None:
            continue
        out[cb_name] = v
    return out


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------


def _resolve_scale_pos_weight_sentinels(
    values: tuple, n_positive: int | None, n_negative: int | None,
) -> list:
    """Replace ``"sqrt(neg/pos)"`` and ``"neg/pos"`` sentinels with floats.

    Sentinels resolving to None (no positives / no negatives observed) are
    dropped — there is no informative value to scout at that point.
    """
    if n_positive is None or n_negative is None or n_positive <= 0:
        # No positives → all the spw sentinels would be infinite/undefined.
        # Keep just the "1" baseline.
        return [1.0]
    ratio = float(n_negative) / float(n_positive)
    out: list = []
    for v in values:
        if isinstance(v, (int, float)):
            out.append(float(v))
            continue
        s = str(v).strip()
        if s in ("1", "1.0"):
            out.append(1.0)
        elif s == "sqrt(neg/pos)":
            out.append(float(math.sqrt(ratio)))
        elif s == "neg/pos":
            out.append(ratio)
        else:
            # Unknown sentinel — try float parse; skip silently if it fails.
            try:
                out.append(float(s))
            except (TypeError, ValueError):
                pass
    return out


def build_grid(
    backend: str,
    spec_overrides: dict | None = None,
    *,
    n_positive: int | None = None,
    n_negative: int | None = None,
) -> list[ScoutConfig]:
    """Build the 8-knob × ~4-5 values single-knob curves grid.

    Always-include defaults zeroth (D9.1.A): the first config has
    ``knob_name="defaults"`` and an empty ``hp_overlay`` so combine has a
    "did scout improve on defaults?" comparator.

    ``spec_overrides`` (optional) can override per-knob values. Shape:
    ``{"max_depth": [2, 3, 4], "eta": [0.05, 0.1]}``. Sentinel strings are
    accepted under ``scale_pos_weight`` (resolved against ``n_positive`` /
    ``n_negative``).

    ``backend`` controls translation: the XGBoost-canonical knob names get
    mapped to CatBoost equivalents per the table in module docstring (the
    ``gamma`` knob is silently dropped for CatBoost — no direct analog).
    """
    overrides = dict(spec_overrides or {})
    grid: list[ScoutConfig] = []

    # Zeroth: defaults sentinel — no overlay.
    grid.append(ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={}))

    knobs_in_order: list[str] = [
        "max_depth", "eta", "colsample_bytree", "min_child_weight",
        "gamma", "alpha", "subsample", "scale_pos_weight",
    ]
    for knob in knobs_in_order:
        # CatBoost has no gamma analog — skip the whole curve for that backend.
        if knob == "gamma" and backend == "catboost":
            continue
        values_raw = overrides.get(knob, DEFAULT_SCOUT_GRID[knob])
        if knob == "scale_pos_weight":
            values: list[Any] = _resolve_scale_pos_weight_sentinels(
                tuple(values_raw), n_positive, n_negative,
            )
        else:
            values = [v for v in values_raw]
        for v in values:
            overlay = _translate_for_backend({knob: v}, backend)
            if not overlay:
                # Translation dropped the knob (e.g. unmapped); skip the row.
                continue
            grid.append(ScoutConfig(
                knob_name=knob,
                knob_value=v,
                hp_overlay=overlay,
            ))
    return grid


# ---------------------------------------------------------------------------
# Scout runner
# ---------------------------------------------------------------------------


def _compute_train_val_gap(train_brier: float | None,
                            val_brier: float | None) -> float | None:
    if train_brier is None or val_brier is None:
        return None
    return float(val_brier) - float(train_brier)


def run_scout(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray | None,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    w_eval: np.ndarray | None,
    mi_eval: pd.MultiIndex | None,
    fit_one: Callable,
    backend: str,
    spec: dict | None = None,
    per_config_timeout_seconds: int | None = None,
    soft_wall_clock_seconds: int | None = None,
    n_positive: int | None = None,
    n_negative: int | None = None,
) -> list[ScoutResult]:
    """Run the 41-fit scout per D1 (8 single-knob curves + defaults zeroth).

    ``fit_one(hp_dict, X_train, y_train, w_train, X_val, y_val, w_val) ->
    (val_brier, train_brier, p_eval, train_val_gap, spiegelhalter_z)``.
    The callable is provided by ``walk_forward_train`` so the runner stays
    backend-agnostic at this layer.

    ``mi_eval`` is the eval segment's ``(date, ticker)`` MultiIndex — required
    for the per-day R-Precision@K computation (matches the canonical CSV
    methodology). Pass ``None`` to skip R-Precision computation (returns
    ``eval_R_p_at_K = None`` per row).

    ``per_config_timeout_seconds`` and ``soft_wall_clock_seconds`` default to
    the per-backend D3a.C caps when None (30 / 300 for XGBoost; 120 / 1800
    for CatBoost). Caps are enforced as a SOFT wall-clock budget per D3a:
    once exceeded, subsequent configs get ``status="timeout"`` with empty
    metrics — they're not actually fit.

    Note (PR #125 review Medium 3): the per-config timeout is POST-HOC —
    a runaway fit that exceeds the cap will still run to completion
    before its result is recorded as ``status="timeout"``. We do not use
    ``signal.alarm`` / ``concurrent.futures.timeout`` because CatBoost's
    signal handling is unreliable cross-platform and the risk of "kill
    mid-fit corrupts state" outweighs the predictability gain. The soft
    wall-clock cap bounds the total scout duration; the per-config cap
    bounds which configs the oracle considers (timeout configs are
    excluded from ``lexicographic_winner`` + ``per_knob_winners`` via
    the ``status == "ok"`` filter). Total scout duration is therefore
    bounded by ``soft_wall_clock_seconds + max_fit_duration`` — at most
    one runaway fit can slip past the wall-clock cap.
    """
    scout_cfg = ((spec or {}).get("backend", {}) or {}).get("scout", {}) or {}
    if per_config_timeout_seconds is None:
        if backend == "catboost":
            per_config_timeout_seconds = int(
                scout_cfg.get("per_config_timeout_seconds", 120)
            )
        else:
            per_config_timeout_seconds = int(
                scout_cfg.get("per_config_timeout_seconds", 30)
            )
    if soft_wall_clock_seconds is None:
        if backend == "catboost":
            soft_wall_clock_seconds = int(
                scout_cfg.get("wall_clock_cap_seconds", 1800)
            )
        else:
            soft_wall_clock_seconds = int(
                scout_cfg.get("wall_clock_cap_seconds", 300)
            )

    grid_overrides = scout_cfg.get("grid")
    grid = build_grid(
        backend=backend,
        spec_overrides=grid_overrides,
        n_positive=n_positive,
        n_negative=n_negative,
    )

    results: list[ScoutResult] = []
    t_start = time.time()
    for cfg in grid:
        elapsed = time.time() - t_start
        if elapsed > soft_wall_clock_seconds:
            # Past the soft wall-clock cap — emit a timeout row without fitting.
            results.append(ScoutResult(
                config=cfg,
                val_brier=None, train_brier=None,
                eval_R_p_at_K=None,
                train_val_gap=None, spiegelhalter_z=None,
                fit_seconds=0.0,
                status="timeout",
                error_message=f"soft wall-clock cap ({soft_wall_clock_seconds}s) exceeded",
            ))
            continue

        t0 = time.time()
        try:
            fit_out = fit_one(
                hp_overlay=cfg.hp_overlay,
                X_train=X_train, y_train=y_train, w_train=w_train,
                X_val=X_val, y_val=y_val, w_val=w_val,
                X_eval=X_eval, y_eval=y_eval, w_eval=w_eval,
                mi_eval=mi_eval,
            )
        except Exception as exc:    # noqa: BLE001 — collect all errors as data
            fit_seconds = time.time() - t0
            results.append(ScoutResult(
                config=cfg,
                val_brier=None, train_brier=None,
                eval_R_p_at_K=None,
                train_val_gap=None, spiegelhalter_z=None,
                fit_seconds=float(fit_seconds),
                status="error",
                error_message=f"{type(exc).__name__}: {exc}"[:512],
            ))
            continue
        fit_seconds = time.time() - t0

        # ``fit_out`` is expected to be a dict with these keys (all optional
        # except ``val_brier`` + ``train_brier``):
        val_brier = fit_out.get("val_brier")
        train_brier = fit_out.get("train_brier")
        eval_rp = fit_out.get("eval_R_p_at_K")
        gap = fit_out.get("train_val_gap")
        if gap is None:
            gap = _compute_train_val_gap(train_brier, val_brier)
        z = fit_out.get("spiegelhalter_z")

        status: Literal["ok", "timeout", "error"] = (
            "timeout" if fit_seconds > per_config_timeout_seconds else "ok"
        )
        results.append(ScoutResult(
            config=cfg,
            val_brier=_clean_float(val_brier),
            train_brier=_clean_float(train_brier),
            eval_R_p_at_K=(
                {int(k): float(v) for k, v in eval_rp.items()}
                if eval_rp else None
            ),
            train_val_gap=_clean_float(gap),
            spiegelhalter_z=_clean_float(z),
            fit_seconds=float(fit_seconds),
            status=status,
            error_message=None,
        ))
    return results


# ---------------------------------------------------------------------------
# Combine — default-mode auto-compose (D12)
# ---------------------------------------------------------------------------


# Lex priority (D12): R-p@1 > R-p@3 > R-p@5 > R-p@10 > R-p@20.
LEX_ORACLE_PRIORITY: tuple[int, ...] = (1, 3, 5, 10, 20)


def _lex_key(rp: dict[int, float] | None) -> tuple:
    """Lex-key for a per-config eval R-p@K dict. None / missing keys sort
    LAST (worst). Negate so argmax via min() works without a custom cmp."""
    if not rp:
        return tuple(0.0 for _ in LEX_ORACLE_PRIORITY)
    return tuple(float(rp.get(k, 0.0) or 0.0) for k in LEX_ORACLE_PRIORITY)


def lexicographic_winner(results: list[ScoutResult]) -> ScoutConfig:
    """Default-mode auto-compose (D12 lex auto-compose).

    Per-knob argmax by eval_R_p_at_K under lex priority
    ``R-p@1 > R-p@3 > R-p@5 > R-p@10 > R-p@20``. Composes one knob value
    per knob into a single hp_overlay (overlaid on defaults).

    The defaults zeroth config (``knob_name == "defaults"``) participates as
    a fallback: when a knob's best-scoring scout-row is no better than the
    defaults baseline for that knob, the composed config simply omits the
    knob (which is equivalent to using its default).

    ``status != "ok"`` configs are excluded from the oracle. If no ``ok``
    configs exist at all, returns a defaults-only config (the safe fallback).
    """
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={})

    # Group by knob_name; "defaults" is the comparator for each knob.
    by_knob: dict[str, list[ScoutResult]] = {}
    defaults_row: ScoutResult | None = None
    for r in ok:
        if r.config.knob_name == "defaults":
            defaults_row = r
            continue
        by_knob.setdefault(r.config.knob_name, []).append(r)

    composed_overlay: dict = {}
    for _, rows in by_knob.items():
        # argmax under lex priority
        rows_sorted = sorted(rows, key=lambda r: _lex_key(r.eval_R_p_at_K),
                              reverse=True)
        best = rows_sorted[0]
        if defaults_row is not None:
            if _lex_key(best.eval_R_p_at_K) <= _lex_key(defaults_row.eval_R_p_at_K):
                # No knob value beat defaults — skip (use default).
                continue
        composed_overlay.update(best.config.hp_overlay)

    return ScoutConfig(
        knob_name="lex_auto_compose",
        knob_value=None,
        hp_overlay=composed_overlay,
    )


def per_knob_winners(results: list[ScoutResult]) -> dict[str, dict]:
    """Per-knob argmax under lex priority, for ``metrics.json::scout``.

    Returns ``{knob_name: {knob_value, eval_R_p_at_K, val_brier}}`` per knob
    (excluding the "defaults" row, which is reported separately).
    """
    by_knob: dict[str, list[ScoutResult]] = {}
    for r in results:
        if r.status != "ok":
            continue
        if r.config.knob_name == "defaults":
            continue
        by_knob.setdefault(r.config.knob_name, []).append(r)
    out: dict[str, dict] = {}
    for knob, rows in by_knob.items():
        best = max(rows, key=lambda r: _lex_key(r.eval_R_p_at_K))
        out[knob] = {
            "knob_value": best.config.knob_value,
            "eval_R_p_at_K": (
                {str(k): float(v) for k, v in (best.eval_R_p_at_K or {}).items()}
            ),
            "val_brier": _clean_float(best.val_brier),
        }
    return out


# ---------------------------------------------------------------------------
# Degenerate-sink detector (D9.2.A)
# ---------------------------------------------------------------------------


def detect_degenerate_sink(
    winner: ScoutConfig,
    results: list[ScoutResult],
    baseline_brier: float | None,
    *,
    brier_threshold: float = 1.05,
    gap_threshold: float = 1e-3,
) -> bool:
    """D9.2.A degenerate-sink detector.

    Returns True when the winner's metrics match the cell-5 anti-AUC
    pathology: ``val_brier ≈ baseline_brier`` (within ``brier_threshold``)
    AND ``train_val_gap ≈ 0`` (within ``gap_threshold``). Mirrors the
    ``compute_degenerate_sink_warning`` semantics in
    :mod:`gbdt.diagnostics`.

    Used by the default-mode fallback path in :func:`walk_forward_train`
    to discard the scout composed config + fall back to defaults for iter_0
    (flagging ``metrics.json::scout::status = "degenerate_sink_fallback"``).
    """
    if baseline_brier is None or baseline_brier <= 0:
        return False
    # Find the winner's row in the results. The lex_auto_compose winner
    # may not have its own ScoutResult row (it's a composed overlay) — in
    # that case, find the result with matching hp_overlay; failing that,
    # fall back to the per-knob argmax of the lex priority across all
    # rows (best proxy when no single row matches).
    target = None
    for r in results:
        if r.status == "ok" and dict(r.config.hp_overlay) == dict(winner.hp_overlay):
            target = r
            break
    if target is None:
        ok = [r for r in results if r.status == "ok"]
        if not ok:
            return False
        target = max(ok, key=lambda r: _lex_key(r.eval_R_p_at_K))
    if target.val_brier is None or target.train_val_gap is None:
        return False
    brier_close = float(target.val_brier) <= float(brier_threshold) * float(baseline_brier)
    gap_close = abs(float(target.train_val_gap)) <= float(gap_threshold)
    return brier_close and gap_close


# ---------------------------------------------------------------------------
# Speed-biased agent prompt (§ 3.5)
# ---------------------------------------------------------------------------


SPEED_BIASED_COMBINE_PROMPT: str = (
    "Prefer configs that train fast — favor shallow max_depth ∈ {2, 3}, higher "
    "eta with early stopping (eta ≥ 0.1 + early_stopping_rounds: 30), smaller "
    "n_estimators, lower colsample_bytree ∈ {0.3, 0.4, 0.5}. The cell-5 winner "
    "was a 6-tree depth-2 fit in 3 seconds; aim for similar leaf budgets "
    "unless scout data argues otherwise."
)


__all__ = [
    "ScoutConfig",
    "ScoutResult",
    "DEFAULT_SCOUT_GRID",
    "LEX_ORACLE_PRIORITY",
    "SPEED_BIASED_COMBINE_PROMPT",
    "build_grid",
    "run_scout",
    "lexicographic_winner",
    "per_knob_winners",
    "detect_degenerate_sink",
]
