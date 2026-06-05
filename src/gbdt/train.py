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
from datetime import date
from typing import Callable, Literal, Optional

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
from gbdt.diagnostics import (
    DiagnosticBundle,
    _r_precision_at_k_from_arrays,
    _fit_in_loop_calibrator,
    build_diagnostic_bundle,
)
from gbdt.fs_hp_loop import best_checkpoint, inner_stop_check
from gbdt.model import BaseGBDTModel, make_model
from gbdt import fs_prefit as fs_prefit_mod
from gbdt import scout as scout_mod
from gbdt.uniqueness import weighted_brier, weighted_spiegelhalter_z
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


@dataclass
class SplitSpec:
    """Walk-forward split spec.

    Two modes (V1.4):

    - ``mode == "trailing"`` (default, back-compat): each ticker's last
      ``total`` rows are carved into ``[train | val | eval | test]`` in
      time order. Used by every pre-V1.4 spec. Silently re-defines the
      cell across cache growth (the eval/test windows slide forward as
      new bars arrive) — the V1.4 plan's motivating bug (§1).
    - ``mode == "date_aligned"`` (V1.4 opt-in): segment windows are
      anchored to **universe-level calendar dates** computed from
      ``train_start`` + the per-segment durations on the universe's
      canonical trading calendar (NYSE for US universes, NSE for NSE
      universes). Per-ticker membership respects the gate
      ``min_train_rows_per_ticker`` on the train segment (≥ max
      ``lookback_windows`` = 200 valid feature rows) and ``≥ 1`` valid
      feature row on val/eval/test. Late-IPO tickers contribute only to
      whichever segments they have valid features for. Reproducible
      across cache growth — adding new bars past ``test_end`` leaves
      segments bit-identical.

    When ``mode == "date_aligned"``, ``train_rows`` / ``val_rows`` /
    ``eval_rows`` / ``test_rows`` are interpreted as **trading-day
    durations** measured on the universe calendar (not row counts).
    ``train_start`` is required (default in the runner is
    ``2019-01-01`` per V1.4 D2).
    """

    train_rows: int = 800
    val_rows: int = 400
    eval_rows: int = 200
    test_rows: int = 100
    # V1.4 fields kept AFTER the row-count fields so existing positional
    # call sites (e.g. ``SplitSpec(800, 400, 200, 100)`` in tests) stay
    # byte-compatible.
    mode: Literal["trailing", "date_aligned"] = "trailing"
    train_start: date | None = None
    min_train_rows_per_ticker: int = 200

    @property
    def total(self) -> int:
        return self.train_rows + self.val_rows + self.eval_rows + self.test_rows


@dataclass
class Fold:
    """Per-ticker positional segments. Indices are positions within the
    ticker's sorted time series, not absolute dates.

    ``segment_dates`` (V1.4): optional 4-segment date envelope (universe-
    calendar windows) populated by ``carve_universe_aligned``. None for
    trailing carves — the caller computes calendar-union dates from the
    predictions DataFrame in that path.
    """

    train_idx: dict[str, np.ndarray]
    val_idx: dict[str, np.ndarray]
    eval_idx: dict[str, np.ndarray]
    test_idx: dict[str, np.ndarray]
    segment_dates: dict[str, dict[str, str]] | None = None


def carve_single_fold(
    panel: pd.DataFrame,
    split: SplitSpec,
    universe_calendar: pd.DatetimeIndex | None = None,
) -> Fold:
    """Carve one fold per ticker.

    Dispatches by ``split.mode``:

    - ``"trailing"`` (default): each ticker's latest ``split.total`` rows
      become ``[train | val | eval | test]`` in time order. Tickers with
      fewer rows than ``split.total`` are dropped (the caller is
      responsible for the row gate via ``min_rows_per_ticker``).
    - ``"date_aligned"`` (V1.4): universe-level calendar windows; see
      :func:`carve_universe_aligned`. Requires ``universe_calendar``.
    """
    if split.mode == "date_aligned":
        if universe_calendar is None:
            raise ValueError(
                "carve_single_fold: split.mode='date_aligned' requires the "
                "universe_calendar argument (a pd.DatetimeIndex of the "
                "universe's trading days). Caller (run_experiment) must "
                "resolve this from configs/gbdt/default.yaml::universes::"
                "<name>::calendar via pandas_market_calendars."
            )
        return carve_universe_aligned(panel, split, universe_calendar)
    return _carve_trailing(panel, split)


def _carve_trailing(panel: pd.DataFrame, split: SplitSpec) -> Fold:
    """Trailing-anchor carve — pre-V1.4 behaviour preserved byte-for-byte."""
    train, val, ev, te = {}, {}, {}, {}
    tickers = panel.index.get_level_values("ticker").unique()
    n_train = split.train_rows
    n_val = split.val_rows
    n_eval = split.eval_rows
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


def carve_universe_aligned(
    panel: pd.DataFrame,
    split: SplitSpec,
    universe_calendar: pd.DatetimeIndex,
) -> Fold:
    """Date-aligned carve on universe-level calendar windows (V1.4, plan §4.2).

    Compute calendar segment boundaries on the universe's canonical trading
    calendar from ``split.train_start`` and the per-segment durations. Then
    derive per-ticker membership: a ticker contributes to a segment iff its
    panel slice in the segment window has ≥ ``min_train_rows_per_ticker``
    rows for the train segment, ≥ 1 row for val/eval/test.

    The panel is feature-NaN-propagated upstream
    (``gbdt.features.build_feature_matrix``), so the row counts here are
    "valid feature rows" by construction.

    Returns a ``Fold`` whose positional indices are computed against each
    ticker's sorted panel (same convention as the trailing carve, so
    ``_gather_segment`` downstream works unchanged), plus a
    ``segment_dates`` ISO-string envelope for ``metrics.json``.
    """
    if split.train_start is None:
        raise ValueError(
            "carve_universe_aligned: split.train_start must be set "
            "(date_aligned mode anchors all segments to this date)."
        )
    cal = pd.DatetimeIndex(universe_calendar).sort_values()
    if len(cal) == 0:
        raise ValueError("carve_universe_aligned: universe_calendar is empty.")

    # Normalize train_start to a Timestamp aligned to the calendar. `side="left"`
    # advances to the next trading day if train_start falls on a non-trading day.
    ts_train_start = pd.Timestamp(split.train_start)
    days_train_start = int(cal.searchsorted(ts_train_start, side="left"))
    days_val_start = days_train_start + split.train_rows
    days_eval_start = days_val_start + split.val_rows
    days_test_start = days_eval_start + split.eval_rows
    days_test_end = days_test_start + split.test_rows - 1  # inclusive index

    if days_test_end >= len(cal):
        raise ValueError(
            "carve_universe_aligned: requested window "
            f"[{ts_train_start.date()} + {split.train_rows + split.val_rows + split.eval_rows + split.test_rows} "
            f"trading days] runs past the end of the supplied universe_calendar "
            f"({cal[-1].date()}). Either extend the calendar or shrink the "
            f"per-segment durations."
        )

    seg_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        "train": (cal[days_train_start], cal[days_val_start - 1]),
        "val":   (cal[days_val_start],   cal[days_eval_start - 1]),
        "eval":  (cal[days_eval_start],  cal[days_test_start - 1]),
        "test":  (cal[days_test_start],  cal[days_test_end]),
    }
    segment_dates: dict[str, dict[str, str]] = {
        seg: {
            "start": s.date().isoformat(),
            "end":   e.date().isoformat(),
        }
        for seg, (s, e) in seg_bounds.items()
    }

    train_idx: dict[str, np.ndarray] = {}
    val_idx: dict[str, np.ndarray] = {}
    eval_idx: dict[str, np.ndarray] = {}
    test_idx: dict[str, np.ndarray] = {}
    seg_idx_map = {
        "train": train_idx,
        "val": val_idx,
        "eval": eval_idx,
        "test": test_idx,
    }

    tickers = panel.index.get_level_values("ticker").unique()
    for t in tickers:
        sub = panel.xs(t, level="ticker").sort_index()
        sub_dates = sub.index
        for seg, (s, e) in seg_bounds.items():
            mask = (sub_dates >= s) & (sub_dates <= e)
            n_valid = int(mask.sum())
            gate = split.min_train_rows_per_ticker if seg == "train" else 1
            if n_valid >= gate:
                seg_idx_map[seg][t] = np.flatnonzero(mask)
            # else: ticker excluded from this segment.

    return Fold(
        train_idx=train_idx,
        val_idx=val_idx,
        eval_idx=eval_idx,
        test_idx=test_idx,
        segment_dates=segment_dates,
    )


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
# V1.3 Option B — scout + FS-prefit helper (Phase 1.4 + 1.5 + 1.6)
# ---------------------------------------------------------------------------


def _build_scout_fit_one(
    *, backend: str, current_hp: dict, current_features: list[str],
    random_seed: int, calibration_method: str, calibration_z_threshold: float,
) -> Callable:
    """Build the per-scout-config fit closure (V1.3 Option B Phase 1.5).

    Each scout config carries a small ``hp_overlay`` dict. The closure
    overlays it onto ``current_hp``, fits the backend, scores
    ``(val_brier, train_brier, eval_R_p_at_K, train_val_gap,
    spiegelhalter_z)`` and returns them. Errors propagate to ``run_scout``
    which catches them as ``status="error"`` rows.
    """
    def _fit_one(
        *, hp_overlay, X_train, y_train, w_train,
        X_val, y_val, w_val, X_eval, y_eval, w_eval, mi_eval,
    ):
        hp = dict(current_hp)
        hp.update(hp_overlay or {})
        model = make_model(
            backend, hp, feature_names=current_features,
            random_seed=random_seed,
        )
        model.fit(
            X_train, y_train, X_val, y_val,
            train_weight=w_train, val_weight=w_val,
        )
        p_train = model.predict_proba(X_train)
        p_val = model.predict_proba(X_val)
        if w_train is not None:
            tr_b = float(weighted_brier(y_train, p_train, w_train))
        else:
            tr_b = float(brier_score_loss(y_train, p_train))
        if w_val is not None:
            val_b = float(weighted_brier(y_val, p_val, w_val))
            z, _p = weighted_spiegelhalter_z(y_val, p_val, w_val)
        else:
            val_b = float(brier_score_loss(y_val, p_val))
            z, _p = spiegelhalter_z(y_val, p_val)
        # eval R-p@K on CALIBRATED predictions per bug #222 doctrine.
        eval_rp: dict[int, float] | None = None
        if X_eval is not None and len(X_eval) and mi_eval is not None:
            try:
                p_eval_raw = np.asarray(model.predict_proba(X_eval), dtype=float)
                calibrator = _fit_in_loop_calibrator(
                    y_val=np.asarray(y_val, dtype=int),
                    p_val_raw=np.asarray(p_val, dtype=float),
                    method=calibration_method,
                    z_threshold=calibration_z_threshold,
                )
                p_eval_cal = apply_calibrator(p_eval_raw, calibrator)
                dates = mi_eval.get_level_values("date").to_numpy()
                tickers = mi_eval.get_level_values("ticker").to_numpy()
                rp = _r_precision_at_k_from_arrays(
                    dates=dates, tickers=tickers,
                    p_calibrated=np.asarray(p_eval_cal, dtype=float),
                    y_true=np.asarray(y_eval, dtype=int),
                )
                eval_rp = rp or None
            except Exception:
                eval_rp = None
        return {
            "val_brier": val_b,
            "train_brier": tr_b,
            "train_val_gap": val_b - tr_b,
            "spiegelhalter_z": float(z),
            "eval_R_p_at_K": eval_rp,
        }
    return _fit_one


def _build_fs_prefit_fit_one(
    *, backend: str, random_seed: int, feature_names: list[str],
) -> Callable:
    """Build the FS-prefit fit closure (V1.3 Option B Phase 1.4).

    Trains one default-HP fit on the full feature matrix and returns the
    feature → importance pd.Series. The runner's scout/prefit specs control
    whether early-stopping is used here; we honor whatever is in the HP
    dict.
    """
    def _fit_one(*, hp, X_train, y_train, w_train,
                  X_val, y_val, w_val):
        model = make_model(
            backend, hp, feature_names=feature_names,
            random_seed=random_seed,
        )
        model.fit(
            X_train, y_train, X_val, y_val,
            train_weight=w_train, val_weight=w_val,
        )
        return model.feature_importance("native")
    return _fit_one


def _maybe_run_scout_and_prefit(
    *,
    panel: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    split: SplitSpec,
    sample_weights: pd.Series | None,
    current_features: list[str],
    current_hp: dict,
    random_seed: int,
    backend: str,
    scout_spec: dict | None,
    fs_prefit_spec: dict | None,
    callback_mode: str,
    calibration_method: str,
    calibration_z_threshold: float,
    degenerate_sink_threshold: float,
    universe_calendar: pd.DatetimeIndex | None,
    # V1.3 Option B D6.2.A — FS-prefit kept-feature cache (cross-cell reuse).
    # All four are optional; when any is missing we run the prefit without
    # caching (back-compat for old callers / tests). The runner threads all
    # of these in default mode.
    universe: str | None = None,
    cache_root: str | None = None,
    features_source_sha256: str | None = None,
    snapshot_end_iso: str | None = None,
) -> dict | None:
    """Run Phase 1.4 (FS-prefit) + Phase 1.5 (scout) + Phase 1.6 (combine).

    Returns ``None`` when scout is disabled (the byte-for-byte back-compat
    path). Otherwise returns a dict ``{current_features, current_hp,
    report}`` reflecting the cliff-cut feature pool + scout-composed HP +
    the metrics.json::scout payload.

    Mode behaviour (plan § 3.4):
    - ``sweep``: hard-OFF (caller is responsible for not setting
      ``scout.enabled: true`` in sweep specs; defense-in-depth here too).
    - ``default``: scout + lex auto-compose; no exit-resume.
    - ``agent_file_protocol``: scout runs to produce data; the
      ``combine_request.json`` + exit-resume is handled in ``__main__.py``
      around this call (P4). Here we only run the scout itself and return
      the report; the agent's combine HP overlay flows in via
      ``scout_spec['combine_decision_hp']`` on the resume side (set by P4).
    """
    scout_cfg = dict(scout_spec or {})
    prefit_cfg = dict(fs_prefit_spec or {})
    scout_enabled = bool(scout_cfg.get("enabled", False))
    prefit_enabled = bool(prefit_cfg.get("enabled", scout_enabled))   # default = scout's
    if not scout_enabled and not prefit_enabled:
        return None
    if callback_mode == "sweep":
        # Defense in depth — sweep mode is supposed to skip scout per D4.
        return None
    if callback_mode == "agent_file_protocol":
        # The runner (__main__.py) handles scout + combine for
        # agent_file_protocol mode via the exit-resume cycles (P4). By the
        # time walk_forward_train is called in that mode, the agent's
        # combine winner (if any) is already applied to ``current_hp`` via
        # the spec's hp_starting + the feature pool is the cliff-cut subset.
        return None

    # Carve once. The scout + prefit reuse the same train/val/eval segments
    # iter_0 uses (D10.A); features are the FULL current_features at this
    # point.
    parts = _carve_X_y(
        X, y, panel, split, current_features, sample_weights,
        universe_calendar=universe_calendar,
    )
    X_tr_full, y_tr, _, w_tr = parts["train"]
    X_val_full, y_val, _, w_val = parts["val"]
    X_ev_full, y_ev, mi_ev, w_ev = parts["eval"]

    if len(y_tr) == 0:
        # No training rows → skip the whole phase. The caller will fall
        # through to the normal loop which will raise the empty-train error.
        return None

    # ---- Phase 1.4 — FS-prefit ------------------------------------------
    kept_features: list[str] = list(current_features)
    fs_prefit_report: dict = {"enabled": False}
    if prefit_enabled:
        cliff_pct = float(prefit_cfg.get("cliff_pct", 0.01))
        # D6.2.A cache key — only computable when ALL key components are
        # provided. Missing any component disables caching (back-compat /
        # test paths). Cliff_pct is folded into the default_hp_sha256 input
        # so two cells with different cliffs don't share a cache entry.
        cache_key: str | None = None
        if (
            universe is not None
            and features_source_sha256 is not None
            and snapshot_end_iso is not None
            and cache_root is not None
        ):
            cache_key = fs_prefit_mod.fs_prefit_cache_key(
                universe=str(universe),
                features_source_sha256=str(features_source_sha256),
                snapshot_end=str(snapshot_end_iso),
                default_hp_sha256=fs_prefit_mod.hp_sha256(
                    {"hp": dict(current_hp), "cliff_pct": float(cliff_pct),
                     "backend": str(backend)},
                ),
            )

        prefit_result = None
        cache_hit = False
        if cache_key is not None:
            cached = fs_prefit_mod.load_fs_prefit_cache(cache_root, cache_key)
            if cached is not None:
                prefit_result = cached
                cache_hit = True

        if prefit_result is None:
            prefit_fit_one = _build_fs_prefit_fit_one(
                backend=backend, random_seed=random_seed,
                feature_names=current_features,
            )
            try:
                prefit_result = fs_prefit_mod.run_fs_prefit(
                    X_train=X_tr_full, y_train=y_tr, w_train=w_tr,
                    X_val=X_val_full, y_val=y_val, w_val=w_val,
                    fit_one=prefit_fit_one,
                    backend=backend,
                    default_hp=dict(current_hp),
                    cliff_pct=cliff_pct,
                )
                if cache_key is not None:
                    try:
                        fs_prefit_mod.save_fs_prefit_cache(
                            cache_root, cache_key, prefit_result,
                        )
                    except OSError:
                        # Cache write failures are non-fatal — the run
                        # continues with the freshly-fit result.
                        pass
            except Exception as exc:    # noqa: BLE001
                fs_prefit_report = {
                    "enabled": True,
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}"[:512],
                }
                # Fall through with the original feature set.
                kept_features = list(current_features)
                prefit_result = None

        if prefit_result is not None:
            kept_features = list(prefit_result.kept_features)
            fs_prefit_report = {
                "enabled": True,
                "cliff_pct": cliff_pct,
                "n_kept": len(prefit_result.kept_features),
                "n_dropped": len(prefit_result.dropped_features),
                "top_importance": prefit_result.top_importance,
                "cliff_threshold": prefit_result.cliff_threshold,
                "fit_seconds": prefit_result.fit_seconds,
                "backend": prefit_result.backend,
                "cache_hit": cache_hit,
                "cache_key": cache_key,
            }

    # Restrict X to kept features for the scout fits.
    X_tr = X_tr_full[kept_features]
    X_val = X_val_full[kept_features]
    X_ev = X_ev_full[kept_features]

    # ---- Phase 1.5 — Scout ----------------------------------------------
    scout_results: list[scout_mod.ScoutResult] = []
    if scout_enabled:
        n_pos = int(np.sum(np.asarray(y_tr) == 1))
        n_neg = int(np.sum(np.asarray(y_tr) == 0))
        scout_fit_one = _build_scout_fit_one(
            backend=backend, current_hp=dict(current_hp),
            current_features=kept_features,
            random_seed=random_seed,
            calibration_method=calibration_method,
            calibration_z_threshold=calibration_z_threshold,
        )
        # Spec-shape the scout config the way ``run_scout`` expects (it
        # reads from ``spec.backend.scout``).
        spec_shim = {"backend": {"scout": scout_cfg}}
        scout_results = scout_mod.run_scout(
            X_train=X_tr, y_train=y_tr, w_train=w_tr,
            X_val=X_val, y_val=y_val, w_val=w_val,
            X_eval=X_ev, y_eval=y_ev, w_eval=w_ev,
            mi_eval=mi_ev,
            fit_one=scout_fit_one,
            backend=backend,
            spec=spec_shim,
            per_config_timeout_seconds=scout_cfg.get("per_config_timeout_seconds"),
            soft_wall_clock_seconds=scout_cfg.get("wall_clock_cap_seconds"),
            n_positive=n_pos, n_negative=n_neg,
        )

    # ---- Phase 1.6 — Combine --------------------------------------------
    composed_overlay: dict = {}
    combine_status = "skipped"
    degenerate_sink_fallback = False
    if scout_enabled and scout_results:
        # The defaults-zeroth row gives us the per-cell baseline brier.
        defaults_row = next(
            (r for r in scout_results
              if r.config.knob_name == "defaults" and r.status == "ok"),
            None,
        )
        baseline_brier = defaults_row.val_brier if defaults_row else None

        if callback_mode == "agent_file_protocol":
            # In agent mode the runner exits AFTER scout (Cycle 1 of D12).
            # That exit-resume is handled in __main__.py. Here we leave
            # composed_overlay empty so iter_0 uses ``current_hp`` until the
            # agent's combine_decision lands.
            #
            # The agent may have already written its decision on a prior
            # resume cycle; if so the runner stashes it in
            # ``scout_cfg['_combine_winner_overlay']`` (P4).
            combine_winner = scout_cfg.get("_combine_winner_overlay")
            if combine_winner is not None:
                composed_overlay = dict(combine_winner)
                combine_status = "agent_combine_winner"
            else:
                combine_status = "awaiting_agent_combine_decision"
        else:
            # Default mode: lex auto-compose.
            winner = scout_mod.lexicographic_winner(scout_results)
            composed_overlay = dict(winner.hp_overlay)
            combine_status = "lex_auto_compose"
            # D9.2.A: degenerate-sink fallback in default mode.
            if scout_mod.detect_degenerate_sink(
                winner, scout_results, baseline_brier=baseline_brier,
                brier_threshold=float(degenerate_sink_threshold),
            ):
                composed_overlay = {}
                combine_status = "degenerate_sink_fallback"
                degenerate_sink_fallback = True

    # Apply the composed overlay to current_hp.
    next_hp = dict(current_hp)
    next_hp.update(composed_overlay)

    # ---- Build the metrics.json::scout block -----------------------------
    per_knob = scout_mod.per_knob_winners(scout_results) if scout_results else {}
    defaults_metrics = None
    for r in scout_results:
        if r.config.knob_name == "defaults":
            defaults_metrics = r.to_dict()
            break
    n_completed = sum(1 for r in scout_results if r.status == "ok")
    runtime_seconds = float(sum(r.fit_seconds for r in scout_results))
    scout_metrics_block = {
        "enabled": scout_enabled,
        "backend": backend,
        "n_configs_total": len(scout_results),
        "n_configs_completed": n_completed,
        "runtime_seconds": runtime_seconds,
        "defaults_metrics": defaults_metrics,
        "per_knob_winner": per_knob,
        "lexicographic_auto_compose": {
            "hp_overlay": (
                dict(scout_mod.lexicographic_winner(scout_results).hp_overlay)
                if scout_results else {}
            ),
        },
        "status": combine_status,
        "degenerate_sink_fallback": degenerate_sink_fallback,
        "grid_spec": dict(scout_cfg.get("grid", {})),
    }

    report = {
        "scout": scout_metrics_block,
        "fs_prefit": fs_prefit_report,
        "combine": {
            "status": combine_status,
            "composed_overlay": composed_overlay,
            "n_mix_configs_completed": 0,    # P4 fills this in agent mode
        },
        # Raw rows for the scout/ subdir dump (P5 writes them to disk).
        "_scout_results_raw": [r.to_dict() for r in scout_results],
    }

    return {
        "current_features": kept_features,
        "current_hp": next_hp,
        "report": report,
    }


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
    # V1.4: ISO date envelope for date-aligned carves; None for trailing
    # carves (the runner computes calendar-union dates from
    # ``predictions`` directly in that path).
    segment_dates: dict[str, dict[str, str]] | None = None
    # V1.3 Option B (plan § 4 D7.1) — scout + combine + FS-prefit summary
    # bundle for metrics.json emission. None when scout is disabled (the
    # default; existing specs unchanged). Populated by walk_forward_train
    # when ``backend.scout.enabled: true``.
    scout_report: dict | None = None


def _carve_X_y(
    X_full: pd.DataFrame, y_full: pd.Series, panel: pd.DataFrame,
    split: SplitSpec, features: list[str],
    weights: pd.Series | None = None,
    universe_calendar: pd.DatetimeIndex | None = None,
):
    fold = carve_single_fold(panel, split, universe_calendar=universe_calendar)
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
    # Stash the fold's segment_dates so walk_forward_train can surface them
    # without re-running the carve.
    parts["__segment_dates__"] = fold.segment_dates
    return parts


def _fit_one(
    X: pd.DataFrame, y: pd.Series, panel: pd.DataFrame, split: SplitSpec,
    features: list[str], hp: dict, random_seed: int,
    sample_weights: pd.Series | None,
    backend: str = "catboost",
    universe_calendar: pd.DatetimeIndex | None = None,
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
    parts = _carve_X_y(
        X, y, panel, split, features, sample_weights,
        universe_calendar=universe_calendar,
    )
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
    disable_plateau: bool = False,
    # V1.3 Option A (plan § 3.2): sweep_row is the canonical CSV row for the
    # cell (looked up once at iter_0 by run_experiment from
    # results/gbdt/data/r_precision_at_k.csv) — constant for the run.
    # build_diagnostic_bundle computes anti_auc_flag from it. None when no
    # matching row exists (new cell) → flag stays "unknown". The threshold
    # is spec-overridable via backend.fs_hp_loop.degenerate_sink_threshold.
    sweep_row: dict | None = None,
    degenerate_sink_threshold: float = 1.05,
    # V1.4: universe trading calendar (NYSE for US universes, NSE for NSE
    # universes). Required when ``split.mode == "date_aligned"``; ignored
    # for the trailing-anchor carve. Caller resolves via
    # ``pandas_market_calendars.get_calendar(...).schedule(...)`` per
    # configs/gbdt/default.yaml::universes::<name>::calendar.
    universe_calendar: pd.DatetimeIndex | None = None,
    # V1.3 Option B (plan § 3 D8 / D11) — scout + FS-prefit pre-iter_0
    # additions. ALL DEFAULT-OFF for byte-for-byte back-compat with
    # pre-V1.3 Option B specs. The runner reads ``backend.scout.enabled``
    # + ``backend.fs_prefit.enabled`` from the spec and threads the
    # sub-dicts here.
    scout_spec: dict | None = None,
    fs_prefit_spec: dict | None = None,
    callback_mode: str = "default",
    # V1.3 Option B D6.2.A — FS-prefit kept-feature cache (cross-cell reuse).
    # Threaded into ``_maybe_run_scout_and_prefit``; when any is None the
    # cache is bypassed (back-compat for callers that don't supply the
    # universe/snapshot/source-sha context).
    fs_prefit_universe: str | None = None,
    fs_prefit_cache_root: str | None = None,
    fs_prefit_features_source_sha256: str | None = None,
    fs_prefit_snapshot_end_iso: str | None = None,
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

    ``disable_plateau`` (task #204): when True, the inner-stop plateau gate
    is suppressed; only ``degradation`` + ``cap`` can terminate the loop. The
    runner sets this when ``callback_mode == "agent_file_protocol"`` so the
    agent stays in charge of when to stop (and is free to pivot to a
    structurally-different knob after one knob plateaus). Default ``False``
    preserves the v1 sweep-mode behaviour byte-for-byte.
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

    # ---- V1.3 Option B Phase 1.4 + 1.5 + 1.6 (plan § 3) -------------------
    # Only run on the FRESH path (not --resume); the scout writes the iter_0
    # ``hp_starting`` + cliff-cut feature pool, which the resume path inherits
    # from the prior run's checkpoint. Sweep mode is hard-OFF (D4); default
    # mode auto-composes lexicographically (D12); agent_file_protocol mode
    # exits to write combine_request.json and resumes for combine fits + iter_0
    # (handled in __main__.py per the exit-resume cycles).
    scout_report: dict | None = None
    if resume_state is None:
        scout_outcome = _maybe_run_scout_and_prefit(
            panel=panel, X=X, y=y,
            split=split, sample_weights=sample_weights,
            current_features=current_features,
            current_hp=current_hp,
            random_seed=random_seed,
            backend=backend,
            scout_spec=scout_spec,
            fs_prefit_spec=fs_prefit_spec,
            callback_mode=callback_mode,
            calibration_method=calibration_method,
            calibration_z_threshold=calibration_z_threshold,
            degenerate_sink_threshold=degenerate_sink_threshold,
            universe_calendar=universe_calendar,
            universe=fs_prefit_universe,
            cache_root=fs_prefit_cache_root,
            features_source_sha256=fs_prefit_features_source_sha256,
            snapshot_end_iso=fs_prefit_snapshot_end_iso,
        )
        if scout_outcome is not None:
            current_features = scout_outcome["current_features"]
            current_hp = scout_outcome["current_hp"]
            scout_report = scout_outcome["report"]

    # ``force_stop`` (agent should_stop=true on resume, plan § 8): finalize the
    # loop at the iters already done — do NOT train a new exploration iteration.
    # The decision's prune/hp_changes are not used to seed a new fit (there is
    # none). The best checkpoint is retrained from the prior history below.
    while not force_stop:
        parts = _carve_X_y(
            X, y, panel, split, current_features, sample_weights,
            universe_calendar=universe_calendar,
        )
        X_tr, y_tr, _, w_tr = parts["train"]
        X_val, y_val, _, w_val = parts["val"]
        X_ev, y_ev, mi_ev, w_ev = parts["eval"]

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
            # V1.3 Option A (plan § 3.2): mi_ev carries (date, ticker) for
            # per-day R-Precision@K; sweep_row + degenerate_sink_threshold
            # are constant for the run (passed through from run_experiment).
            mi_eval=mi_ev,
            sweep_row=sweep_row,
            degenerate_sink_threshold=degenerate_sink_threshold,
            # Bug #222 fix — thread the spec's calibration method + Z
            # threshold so the bundle's eval R-p@K is computed on calibrated
            # predictions matching canonical CSV scoring. Per-iter calibrator
            # is fit fresh inside the bundle build; not the finalization fit.
            calibration_method=calibration_method,
            calibration_z_threshold=calibration_z_threshold,
        )
        history.append(bundle)
        models.append(model)
        feature_lists.append(list(current_features))
        hp_lists.append(dict(current_hp))
        val_briers.append(bundle.val_brier)

        # Inner-stop check (does the loop continue?).
        # V1.3 Option A: pass anti_auc_flag so the plateau gate is auto-
        # disabled on anti-AUC cells (plan § 3.5 / D6). The flag is constant
        # for the run; bundle.anti_auc_flag carries it from
        # build_diagnostic_bundle (computed from sweep_row at iter_0).
        stop, signal = inner_stop_check(
            val_briers,
            plateau_threshold=plateau_threshold,
            degradation_gate=degradation_gate,
            max_iterations=max_iterations,
            disable_plateau=disable_plateau,
            anti_auc_flag=str(getattr(bundle, "anti_auc_flag", "unknown")),
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

            # V1.3 Option A: per-iter eval R-p@1 series so the resume-side
            # best_checkpoint can tie-break across the full history (mirrors
            # the gap/z plumbing). Anti_auc_flag is constant per run.
            sink_eval_rp1 = [
                (
                    float(b.eval_r_precision_at_k.get(1))
                    if (
                        b.eval_r_precision_at_k is not None
                        and b.eval_r_precision_at_k.get(1) is not None
                    )
                    else None
                )
                for b in history
            ]
            prior_eval_rp1_seed = (
                list(resume_state.get("eval_r_precision_at_1s", []))
                if resume_state is not None else []
            )
            sink_eval_rp1 = list(prior_eval_rp1_seed[:n_prior]) + sink_eval_rp1
            run_anti_auc_flag = (
                str(history[-1].anti_auc_flag) if history else
                (
                    str(resume_state.get("anti_auc_flag", "unknown"))
                    if resume_state is not None else "unknown"
                )
            )
            # auto_disabled: audit trail per § 3.5 — records which V1.3
            # auto-disable mechanisms are active for this run. The L1
            # tie-break + val_brier plateau gate are both auto-disabled when
            # anti_auc_flag=="true"; non-anti-AUC cells get an empty dict
            # (advisory only, behavior unchanged from pre-V1.3).
            v13_auto_disabled = {}
            if run_anti_auc_flag == "true":
                v13_auto_disabled = {
                    "l1_tie_break": "anti_auc_flag=true",
                    "val_brier_plateau": "anti_auc_flag=true",
                }

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
                # V1.3 Option A — carried across resume so the resume-side
                # finalization can see the full history when picking the
                # best checkpoint. Constant run-level fields:
                "anti_auc_flag": run_anti_auc_flag,
                "auto_disabled": v13_auto_disabled,
                # Per-iter:
                "eval_r_precision_at_1s": sink_eval_rp1,
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

    # V1.3 Option A: resume-seeded prior eval R-p@1 series if the checkpoint
    # carried it; older checkpoints predating this field default to None
    # (best_checkpoint then falls back to strict val-Brier argmin among ties).
    prior_eval_rp1 = (
        list(resume_state.get("eval_r_precision_at_1s", []))
        if resume_state is not None else []
    )

    def _eval_rp1_for(i: int) -> float | None:
        b = bundle_by_idx.get(i)
        if b is not None:
            rp = b.eval_r_precision_at_k
            if rp is not None:
                return float(rp.get(1)) if rp.get(1) is not None else None
            return None
        if i < len(prior_eval_rp1):
            v = prior_eval_rp1[i]
            return float(v) if v is not None else None
        return None

    gaps_seq = [_gap_for(i) for i in range(len(val_briers))]
    zs_seq = [_z_for(i) for i in range(len(val_briers))]
    eval_rp1_seq = [_eval_rp1_for(i) for i in range(len(val_briers))]
    # V1.3 Option A: resolve the run's constant anti_auc_flag — prefer an
    # in-process bundle's value (every bundle carries the same flag); fall
    # back to whatever the resume checkpoint recorded; default "unknown".
    anti_auc_flag = "unknown"
    for b in history:
        if getattr(b, "anti_auc_flag", None) is not None:
            anti_auc_flag = str(b.anti_auc_flag)
            break
    if anti_auc_flag == "unknown" and resume_state is not None:
        anti_auc_flag = str(resume_state.get("anti_auc_flag", "unknown"))
    best_i = best_checkpoint(
        val_briers,
        train_val_gaps=gaps_seq,
        spiegelhalter_zs=zs_seq,
        tie_band=tie_band,
        plateau_threshold=plateau_threshold,
        anti_auc_flag=anti_auc_flag,
        eval_r_precision_at_1s=eval_rp1_seq,
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
            universe_calendar=universe_calendar,
        )

    # Score the best checkpoint and apply calibration
    best_parts = _carve_X_y(
        X, y, panel, split, best_features, sample_weights,
        universe_calendar=universe_calendar,
    )
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
        segment_dates=best_parts.get("__segment_dates__"),
        scout_report=scout_report,
    )


__all__ = [
    "SplitSpec",
    "Fold",
    "carve_single_fold",
    "carve_universe_aligned",
    "WalkForwardResult",
    "walk_forward_train",
    "default_fs_hp_callback",
]
