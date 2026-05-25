"""Walk-forward training driver for gbdt v1.

Carves the panel into ``train + val + eval + test`` segments per
``configs/gbdt/default.yaml::split`` (default 800/400/200/100 = 1,600 rows
per stock), runs the FS+HP loop, and returns the per-segment predictions
plus the loop's diagnostic history.

The inner FS+HP loop is *agent-driven* in production (the skill drives
it); for the unit tests and the CLI atom this module accepts an optional
``fs_hp_callback`` that receives the previous iteration's bundle and
returns the next iteration's ``(features, hp_dict, rationale)``. The CLI
atom's default callback is a simple algorithmic fallback (importance-based
prune + a small HP nudge) so the CLI runs end-to-end without an agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from gbdt.calibration import (
    CalibrationDecision,
    apply_calibrator,
    conditional_isotonic,
    isotonic_always,
    spiegelhalter_z,
)
from gbdt.diagnostics import DiagnosticBundle, build_diagnostic_bundle
from gbdt.fs_hp_loop import best_checkpoint, inner_stop_check
from gbdt.model import GBDTModel


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


@dataclass
class SplitSpec:
    train_rows: int = 800
    val_rows: int = 400
    eval_rows: int = 200
    test_rows: int = 100

    @property
    def total(self) -> int:
        return self.train_rows + self.val_rows + self.eval_rows + self.test_rows


@dataclass
class Fold:
    """Per-ticker positional segments. Indices are positions within the
    ticker's sorted time series, not absolute dates."""

    train_idx: dict[str, np.ndarray]
    val_idx: dict[str, np.ndarray]
    eval_idx: dict[str, np.ndarray]
    test_idx: dict[str, np.ndarray]


def carve_single_fold(
    panel: pd.DataFrame,
    split: SplitSpec,
) -> Fold:
    """Carve one trailing-anchor fold per ticker.

    For each ticker, take the latest ``split.total`` rows (sorted ascending
    by date) and split them into ``[train | val | eval | test]`` in order.
    Tickers with fewer rows than ``split.total`` are dropped (the caller
    is responsible for the row gate via ``min_rows_per_ticker``).
    """
    train, val, ev, te = {}, {}, {}, {}
    tickers = panel.index.get_level_values("ticker").unique()
    n_train = split.train_rows
    n_val = split.val_rows
    n_eval = split.eval_rows
    n_test = split.test_rows
    n_total = split.total
    for t in tickers:
        sub = panel.xs(t, level="ticker").sort_index()
        if len(sub) < n_total:
            continue
        tail = np.arange(len(sub) - n_total, len(sub))
        train[t] = tail[:n_train]
        val[t] = tail[n_train: n_train + n_val]
        ev[t] = tail[n_train + n_val: n_train + n_val + n_eval]
        te[t] = tail[n_train + n_val + n_eval: n_total]
    return Fold(train_idx=train, val_idx=val, eval_idx=ev, test_idx=te)


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


def _gather_segment(
    panel: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    idx: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, pd.MultiIndex]:
    """Return (X_seg, y_seg, multi_idx) for a segment defined by per-ticker
    positional indices, with NaN-target rows dropped."""
    keys = []
    for ticker, positions in idx.items():
        sub_dates = panel.xs(ticker, level="ticker").sort_index().index[positions]
        keys.extend((d, ticker) for d in sub_dates)
    mi = pd.MultiIndex.from_tuples(keys, names=["date", "ticker"]).sort_values()
    X_seg = X.reindex(mi)
    y_seg = y.reindex(mi)
    # Drop rows where target is NaN
    mask = ~y_seg.isna()
    return X_seg.loc[mask], y_seg.loc[mask].values.astype(int), mi[mask]


# ---------------------------------------------------------------------------
# Algorithmic fallback callback (CLI atom path)
# ---------------------------------------------------------------------------


def default_fs_hp_callback(
    bundle: DiagnosticBundle,
    available_features: list[str],
) -> tuple[list[str], dict, str]:
    """Cheap algorithmic prune + a small HP nudge for the CLI atom.

    Replaces the agent loop when ``/gbdt-experiment`` is not in the loop.
    Rules:
      - Drop features whose native importance is < 1% of the top feature.
      - Always keep at least 10 features.
      - If train_brier << val_brier (gap > 0.02), nudge l2_leaf_reg up 1.5x.
      - If learning curve hit cap, double iterations + halve learning_rate.
    """
    imp = bundle.importance_native or {}
    if imp:
        top = max(imp.values())
        keep = [f for f, v in imp.items() if v >= 0.01 * top]
        if len(keep) < 10:
            # Keep top-10 by importance
            keep = sorted(imp, key=imp.get, reverse=True)[:10]
        # Restrict to currently-available
        keep = [f for f in keep if f in available_features]
    else:
        keep = list(available_features)

    hp = dict(bundle.hp)
    rationale_parts = [f"algorithmic fallback: kept {len(keep)}/{len(imp)} features"]
    if bundle.train_val_gap > 0.02:
        old = float(hp.get("l2_leaf_reg", 3.0))
        new = min(old * 1.5, 30.0)
        if new != old:
            hp["l2_leaf_reg"] = new
            rationale_parts.append(f"l2_leaf_reg {old}->{new} (overfit gap)")
    if bundle.iteration_cap_hit:
        old_it = int(hp.get("iterations", 1000))
        old_lr = float(hp.get("learning_rate", 0.05))
        hp["iterations"] = min(old_it * 2, 10_000)
        hp["learning_rate"] = max(old_lr / 2, 0.005)
        rationale_parts.append(
            f"iterations {old_it}->{hp['iterations']} + lr {old_lr}->{hp['learning_rate']} (cap hit)"
        )
    return keep, hp, "; ".join(rationale_parts)


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    best_iteration: int
    best_model: GBDTModel
    best_features: list[str]
    best_hp: dict
    best_val_brier: float
    iterations: list[DiagnosticBundle]
    calibration: CalibrationDecision
    inner_stop_signal: str
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)


def _carve_X_y(
    X_full: pd.DataFrame, y_full: pd.Series, panel: pd.DataFrame,
    split: SplitSpec, features: list[str],
):
    fold = carve_single_fold(panel, split)
    X_use = X_full[features]
    parts = {}
    for name, idx in (
        ("train", fold.train_idx),
        ("val", fold.val_idx),
        ("eval", fold.eval_idx),
        ("test", fold.test_idx),
    ):
        Xs, ys, mi = _gather_segment(panel, X_use, y_full, idx)
        parts[name] = (Xs, ys, mi)
    return parts


def walk_forward_train(
    *,
    panel: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    hp: dict,
    split: SplitSpec | None = None,
    calibration_method: str = "conditional_isotonic",
    calibration_z_threshold: float = 2.0,
    max_iterations: int = 8,
    plateau_threshold: float = 0.005,
    degradation_gate: float = 0.01,
    fs_hp_callback: Optional[Callable] = None,
    random_seed: int = 42,
) -> WalkForwardResult:
    """Run one walk-forward fold with the FS+HP iteration loop on top.

    Boundary discipline: each segment is a strictly-forward slice of every
    ticker's tail (no shuffling), per CLAUDE.md C6.
    """
    split = split or SplitSpec()
    if fs_hp_callback is None:
        fs_hp_callback = default_fs_hp_callback

    current_features = list(features)
    current_hp = dict(hp)
    history: list[DiagnosticBundle] = []
    hp_history: list[dict] = []
    val_briers: list[float] = []
    models: list[GBDTModel] = []
    feature_lists: list[list[str]] = []
    hp_lists: list[dict] = []
    inner_signal: str | None = None

    iter_idx = 0
    while True:
        parts = _carve_X_y(X, y, panel, split, current_features)
        X_tr, y_tr, _ = parts["train"]
        X_val, y_val, _ = parts["val"]
        X_ev, y_ev, _ = parts["eval"]

        # Sanity: enough training rows
        if len(y_tr) == 0:
            raise RuntimeError("training segment is empty; check split + min_rows")

        t0 = time.time()
        model = GBDTModel(current_hp, feature_names=current_features,
                          random_seed=random_seed)
        model.fit(X_tr, y_tr, X_val, y_val)
        wall = time.time() - t0

        hp_history.append({"iter": iter_idx, "hp": dict(current_hp)})
        rationale = (
            "iteration 0 — full feature pool, default HPs"
            if iter_idx == 0 else f"iteration {iter_idx} from FS+HP callback"
        )
        bundle = build_diagnostic_bundle(
            model=model, iter_idx=iter_idx, hp=current_hp,
            feature_names=current_features,
            X_train=X_tr, y_train=y_tr,
            X_val=X_val, y_val=y_val,
            X_eval=X_ev, y_eval=y_ev,
            hp_history=hp_history,
            rationale=rationale,
            wall_time_sec=wall,
            include_permutation=False,        # too expensive on hot loop
        )
        history.append(bundle)
        models.append(model)
        feature_lists.append(list(current_features))
        hp_lists.append(dict(current_hp))
        val_briers.append(bundle.val_brier)

        # Inner-stop check (does the loop continue?)
        stop, signal = inner_stop_check(
            val_briers,
            plateau_threshold=plateau_threshold,
            degradation_gate=degradation_gate,
            max_iterations=max_iterations,
        )
        if stop:
            inner_signal = signal
            history[-1].delta_attribution = f"inner_stop={signal}"
            break

        # Ask the callback for next iteration
        keep, next_hp, agent_rationale = fs_hp_callback(bundle, current_features)
        current_features = keep
        current_hp = next_hp
        iter_idx += 1
        # Record what the agent did into the previous iteration's record
        history[-1].delta_attribution = agent_rationale

    # Best checkpoint
    best_i = best_checkpoint(val_briers)
    best_model = models[best_i]
    best_features = feature_lists[best_i]
    best_hp = hp_lists[best_i]

    # Score the best checkpoint and apply calibration
    best_parts = _carve_X_y(X, y, panel, split, best_features)
    X_tr, y_tr, mi_tr = best_parts["train"]
    X_val, y_val, mi_val = best_parts["val"]
    X_ev, y_ev, mi_ev = best_parts["eval"]
    X_te, y_te, mi_te = best_parts["test"]

    p_val_raw = best_model.predict_proba(X_val)
    if calibration_method == "native":
        cal = CalibrationDecision(
            method="native",
            spiegelhalter_z=spiegelhalter_z(y_val, p_val_raw)[0],
            spiegelhalter_p=spiegelhalter_z(y_val, p_val_raw)[1],
            z_threshold=calibration_z_threshold,
            calibrator=None,
            rationale="spec override: native (no post-calibration)",
        )
    elif calibration_method == "conditional_isotonic":
        cal = conditional_isotonic(y_val, p_val_raw, z_threshold=calibration_z_threshold)
    elif calibration_method == "isotonic_always":
        cal = isotonic_always(y_val, p_val_raw, z_threshold=calibration_z_threshold)
    else:
        raise NotImplementedError(f"calibration_method={calibration_method!r}")

    predictions = {}
    for name, (Xs, ys, mi) in (
        ("train", (X_tr, y_tr, mi_tr)),
        ("val", (X_val, y_val, mi_val)),
        ("eval", (X_ev, y_ev, mi_ev)),
        ("test", (X_te, y_te, mi_te)),
    ):
        if len(Xs) == 0:
            predictions[name] = pd.DataFrame(
                columns=["date", "ticker", "p_raw", "p_calibrated", "y_true"]
            )
            continue
        p_raw = best_model.predict_proba(Xs)
        p_cal = apply_calibrator(p_raw, cal.calibrator)
        df = pd.DataFrame({
            "date": mi.get_level_values("date"),
            "ticker": mi.get_level_values("ticker"),
            "p_raw": p_raw,
            "p_calibrated": p_cal,
            "y_true": ys,
        })
        predictions[name] = df

    return WalkForwardResult(
        best_iteration=best_i,
        best_model=best_model,
        best_features=best_features,
        best_hp=best_hp,
        best_val_brier=val_briers[best_i],
        iterations=history,
        calibration=cal,
        inner_stop_signal=inner_signal or "cap",
        predictions=predictions,
    )


__all__ = [
    "SplitSpec",
    "Fold",
    "carve_single_fold",
    "WalkForwardResult",
    "walk_forward_train",
    "default_fs_hp_callback",
]
