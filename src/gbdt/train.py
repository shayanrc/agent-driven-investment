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
    platt_calibration,
    spiegelhalter_z,
)
from gbdt.diagnostics import DiagnosticBundle, build_diagnostic_bundle
from gbdt.fs_hp_loop import best_checkpoint, inner_stop_check
from gbdt.model import BaseGBDTModel, make_model


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
    weights: pd.Series | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.MultiIndex, np.ndarray | None]:
    """Return (X_seg, y_seg, multi_idx, w_seg) for a segment defined by
    per-ticker positional indices, with NaN-target rows dropped.

    ``weights`` (optional) is a per-(date, ticker) sample-weight Series
    (the LdP §4.4 uniqueness weights). When provided, the returned
    ``w_seg`` is the aligned, mask-filtered weight vector. ``None``
    propagates through as ``None`` so unweighted call sites keep working.
    """
    keys = []
    for ticker, positions in idx.items():
        sub_dates = panel.xs(ticker, level="ticker").sort_index().index[positions]
        keys.extend((d, ticker) for d in sub_dates)
    mi = pd.MultiIndex.from_tuples(keys, names=["date", "ticker"]).sort_values()
    X_seg = X.reindex(mi)
    y_seg = y.reindex(mi)
    # Drop rows where target is NaN
    mask = ~y_seg.isna()
    w_seg = None
    if weights is not None:
        w_seg = weights.reindex(mi).loc[mask].values.astype(float)
    return X_seg.loc[mask], y_seg.loc[mask].values.astype(int), mi[mask], w_seg


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
    best_model: BaseGBDTModel
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
    weights: pd.Series | None = None,
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
        Xs, ys, mi, ws = _gather_segment(panel, X_use, y_full, idx, weights)
        parts[name] = (Xs, ys, mi, ws)
    return parts


def _fit_one(
    X: pd.DataFrame, y: pd.Series, panel: pd.DataFrame, split: SplitSpec,
    features: list[str], hp: dict, random_seed: int,
    sample_weights: pd.Series | None,
    backend: str = "catboost",
) -> BaseGBDTModel:
    """Fit a single model for a (features, hp) configuration.

    Used both inside the loop and at finalization, when the best checkpoint
    selected from the full val-Brier history corresponds to a prior
    (non-retrained) iteration on the exit-and-resume path. Retraining one
    config is cheap and avoids serializing model blobs into the checkpoint
    (plan § 0.2). The retrain assumes bit-identical reproduction of the
    in-loop fit, which the chosen ``backend`` must guarantee given the same
    ``(features, hp, random_seed)`` + row order.
    """
    parts = _carve_X_y(X, y, panel, split, features, sample_weights)
    X_tr, y_tr, _, w_tr = parts["train"]
    X_val, y_val, _, w_val = parts["val"]
    if len(y_tr) == 0:
        raise RuntimeError("training segment is empty; check split + min_rows")
    model = make_model(backend, hp, feature_names=features, random_seed=random_seed)
    model.fit(X_tr, y_tr, X_val, y_val, train_weight=w_tr, val_weight=w_val)
    return model


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
    tie_band: float | None = None,
    fs_hp_callback: Optional[Callable] = None,
    random_seed: int = 42,
    sample_weights: pd.Series | None = None,
    resume_state: dict | None = None,
    loop_state_sink: dict | None = None,
    backend: str = "catboost",
) -> WalkForwardResult:
    """Run one walk-forward fold with the FS+HP iteration loop on top.

    Boundary discipline: each segment is a strictly-forward slice of every
    ticker's tail (no shuffling), per CLAUDE.md C6.

    ``resume_state`` (V1.1 exit-and-resume, plan § 0): when provided, the loop
    SEEDS at iteration ``resume_state["iter_idx"]`` with the decision already
    applied (``current_features`` / ``current_hp``) and the prior-iteration
    history threaded back in (``val_briers`` for the inner-stop check, plus the
    per-iter ``(features, hp)`` so the best checkpoint can be retrained if it
    lands on a non-retrained prior iter). Iterations 0..N are NOT re-trained —
    only iteration N+1 runs. ``resume_state["force_stop"]`` finalizes the loop
    after this single iteration (the agent's ``should_stop=true``). When
    ``resume_state is None`` the loop behaves exactly as v1 (byte-for-byte).

    ``loop_state_sink`` (V1.1 agent_file_protocol): an optional mutable dict the
    loop populates with the live accumulated history each iteration *before*
    invoking ``fs_hp_callback``. The agent-file-protocol callback reads it to
    write the resume checkpoint, then raises ``PauseForAgentDecision`` (caught
    by ``run_experiment``). ``default`` mode passes ``None`` here, so this is
    inert on the v1 path.

    ``backend`` (V1.2 backend seam, plan § 6.2): the ``backend.library`` value
    from the spec. Threaded to :func:`gbdt.model.make_model` for both the
    in-loop fit and the finalization retrain. Defaults to ``"catboost"`` so
    every existing spec + test stays byte-for-byte unchanged.
    """
    split = split or SplitSpec()
    if fs_hp_callback is None:
        fs_hp_callback = default_fs_hp_callback

    history: list[DiagnosticBundle] = []
    models: list[BaseGBDTModel | None] = []
    inner_signal: str | None = None

    if resume_state is not None:
        # --- exit-and-resume seed (plan § 0) ---------------------------------
        # Prior iters' (features, hp, val_brier) are threaded back so the
        # inner-stop check sees the full history and the best checkpoint can be
        # retrained if it lands on a prior iter. Prior models are NOT carried
        # (no blob in the checkpoint) — placeholder None entries.
        current_features = list(resume_state["current_features"])
        current_hp = dict(resume_state["current_hp"])
        iter_idx = int(resume_state["iter_idx"])
        val_briers = list(resume_state.get("val_briers", []))
        hp_history = list(resume_state.get("hp_history", []))
        feature_lists = [list(f) for f in resume_state.get("feature_history", [])]
        hp_lists = [dict(h) for h in resume_state.get("hp_lists", [])]
        prior_deltas = list(resume_state.get("delta_attributions", []))
        force_stop = bool(resume_state.get("force_stop", False))
        # Prior iters occupy slots 0..N in feature_lists/hp_lists/val_briers;
        # models[] for those slots is None (retrained lazily at finalization).
        models = [None] * len(feature_lists)
    else:
        current_features = list(features)
        current_hp = dict(hp)
        iter_idx = 0
        val_briers = []
        hp_history = []
        feature_lists = []
        hp_lists = []
        prior_deltas = []
        force_stop = False

    # ``force_stop`` (agent should_stop=true on resume, plan § 8): finalize the
    # loop at the iters already done — do NOT train a new exploration iteration.
    # The decision's prune/hp_changes are not used to seed a new fit (there is
    # none). The best checkpoint is retrained from the prior history below.
    while not force_stop:
        parts = _carve_X_y(X, y, panel, split, current_features, sample_weights)
        X_tr, y_tr, _, w_tr = parts["train"]
        X_val, y_val, _, w_val = parts["val"]
        X_ev, y_ev, _, w_ev = parts["eval"]

        # Sanity: enough training rows
        if len(y_tr) == 0:
            raise RuntimeError("training segment is empty; check split + min_rows")

        t0 = time.time()
        model = make_model(backend, current_hp, feature_names=current_features,
                           random_seed=random_seed)
        model.fit(
            X_tr, y_tr, X_val, y_val,
            train_weight=w_tr, val_weight=w_val,
        )
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
            w_train=w_tr, w_val=w_val, w_eval=w_ev,
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

        # Inner-stop check (does the loop continue?).
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

        # Hand the live accumulated history to the agent-file-protocol callback
        # (via loop_state_sink) so it can write a complete resume checkpoint
        # before pausing. Inert in default mode (sink is None).
        if loop_state_sink is not None:
            # L1 from _187: persist per-iter (gap, z) alongside val_briers so
            # the tie-break logic in best_checkpoint() can see the full
            # history across an exit-and-resume boundary. Older checkpoints
            # written before this field default to empty lists on the resume
            # side, which the resolver treats as None (worst-case).
            sink_gaps = [
                (float(b.train_val_gap) if b.train_val_gap is not None else None)
                for b in history
            ]
            sink_zs = [
                (float(b.spiegelhalter_z) if b.spiegelhalter_z is not None else None)
                for b in history
            ]
            # Prepend any prior-iter (gap, z) carried in from the resume seed
            # — those slots are not represented in ``history`` (which only
            # holds bundles built in this process).
            prior_gap_seed = (
                list(resume_state.get("train_val_gaps", []))
                if resume_state is not None else []
            )
            prior_z_seed = (
                list(resume_state.get("spiegelhalter_zs", []))
                if resume_state is not None else []
            )
            n_prior = max(0, len(val_briers) - len(history))
            sink_gaps = list(prior_gap_seed[:n_prior]) + sink_gaps
            sink_zs = list(prior_z_seed[:n_prior]) + sink_zs

            loop_state_sink.clear()
            loop_state_sink.update({
                "iter_idx": iter_idx,
                "current_features": list(current_features),
                "current_hp": dict(current_hp),
                "val_briers": list(val_briers),
                "train_val_gaps": sink_gaps,
                "spiegelhalter_zs": sink_zs,
                "hp_history": list(hp_history),
                "feature_history": [list(f) for f in feature_lists],
                "hp_lists": [dict(h) for h in hp_lists],
                "delta_attributions": list(prior_deltas),
                "max_iterations": int(max_iterations),
            })

        # Ask the callback for next iteration. The agent-file-protocol callback
        # raises PauseForAgentDecision here (caught by run_experiment); the
        # default + scripted callbacks return (keep, next_hp, rationale).
        keep, next_hp, agent_rationale = fs_hp_callback(bundle, current_features)
        current_features = keep
        current_hp = next_hp
        iter_idx += 1
        # Record what the agent did into the previous iteration's record
        history[-1].delta_attribution = agent_rationale
        prior_deltas.append(agent_rationale)

    if force_stop:
        # Loop body skipped entirely — finalize at the prior history. The
        # best checkpoint is retrained from (feature_history, hp_lists) below.
        inner_signal = "agent_should_stop"

    if not val_briers:
        raise RuntimeError(
            "no iterations to finalize: empty val-Brier history. On resume this "
            "means the checkpoint carried no prior iterations."
        )

    # Best checkpoint across the FULL val-Brier history (prior + this-process).
    # L1 from _187: among configs whose val Brier lands inside the tie band
    # (default 0.5 * plateau_threshold), prefer lower train-val gap, then
    # |Spiegelhalter Z| closer to 0. ``history`` is the in-process bundles;
    # resume-seeded prior iterations carry their (gap, z) via the checkpoint
    # so the tie-break sees the full history when available, otherwise the
    # prior-iter entry is None and falls back to worst-case (i.e. never wins
    # a tie over a present-metric config).
    bundle_by_idx: dict[int, DiagnosticBundle] = {}
    if resume_state is not None:
        # Prior iters occupy slots 0..N from the resume seed; in-process
        # bundles fill the remaining slots starting at len(resume_history).
        first_in_process = len(val_briers) - len(history)
    else:
        first_in_process = 0
    for offset, b in enumerate(history):
        bundle_by_idx[first_in_process + offset] = b

    # Resume-seeded prior (gap, z) if the checkpoint carried them; older
    # checkpoints predating this field default to None and the corresponding
    # slot is treated as worst-case in tie-breaking.
    prior_gaps = (
        list(resume_state.get("train_val_gaps", []))
        if resume_state is not None else []
    )
    prior_zs = (
        list(resume_state.get("spiegelhalter_zs", []))
        if resume_state is not None else []
    )

    def _gap_for(i: int) -> float | None:
        b = bundle_by_idx.get(i)
        if b is not None:
            return float(b.train_val_gap) if b.train_val_gap is not None else None
        if i < len(prior_gaps):
            v = prior_gaps[i]
            return float(v) if v is not None else None
        return None

    def _z_for(i: int) -> float | None:
        b = bundle_by_idx.get(i)
        if b is not None:
            return float(b.spiegelhalter_z) if b.spiegelhalter_z is not None else None
        if i < len(prior_zs):
            v = prior_zs[i]
            return float(v) if v is not None else None
        return None

    gaps_seq = [_gap_for(i) for i in range(len(val_briers))]
    zs_seq = [_z_for(i) for i in range(len(val_briers))]
    best_i = best_checkpoint(
        val_briers,
        train_val_gaps=gaps_seq,
        spiegelhalter_zs=zs_seq,
        tie_band=tie_band,
        plateau_threshold=plateau_threshold,
    )
    best_features = feature_lists[best_i]
    best_hp = hp_lists[best_i]
    best_model = models[best_i]
    if best_model is None:
        # Exit-and-resume: the best checkpoint landed on a prior iteration whose
        # model was not carried in the checkpoint (no blob — plan § 0.2). Retrain
        # that single (features, hp) config now for calibration + prediction.
        best_model = _fit_one(
            X, y, panel, split, best_features, best_hp, random_seed,
            sample_weights, backend=backend,
        )

    # Score the best checkpoint and apply calibration
    best_parts = _carve_X_y(X, y, panel, split, best_features, sample_weights)
    X_tr, y_tr, mi_tr, w_tr = best_parts["train"]
    X_val, y_val, mi_val, w_val = best_parts["val"]
    X_ev, y_ev, mi_ev, w_ev = best_parts["eval"]
    X_te, y_te, mi_te, w_te = best_parts["test"]

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
    elif calibration_method == "platt":
        # Backend-neutral Platt scaling fit on (p_val_raw, y_val) — V1.2 plan R7.
        cal = platt_calibration(y_val, p_val_raw, z_threshold=calibration_z_threshold)
    else:
        raise NotImplementedError(f"calibration_method={calibration_method!r}")

    predictions = {}
    for name, (Xs, ys, mi, ws) in (
        ("train", (X_tr, y_tr, mi_tr, w_tr)),
        ("val", (X_val, y_val, mi_val, w_val)),
        ("eval", (X_ev, y_ev, mi_ev, w_ev)),
        ("test", (X_te, y_te, mi_te, w_te)),
    ):
        if len(Xs) == 0:
            predictions[name] = pd.DataFrame(
                columns=["date", "ticker", "p_raw", "p_calibrated", "y_true",
                         "sample_weight"]
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
            # Always-present column. When weights weren't supplied the
            # column is all-1.0 so downstream weighted-metric code is
            # uniform; with uniform weights weighted metrics collapse to
            # unweighted ones.
            "sample_weight": (
                ws if ws is not None
                else np.ones(len(ys), dtype=float)
            ),
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
