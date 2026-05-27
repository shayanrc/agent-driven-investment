"""CLI orchestrator: ``python -m gbdt experiment <spec.yaml>``.

Loads a spec, builds the universe panel + 279-col feature matrix + binary
target, runs the walk-forward driver with the default algorithmic FS+HP
fallback (the ``/gbdt-experiment`` skill overrides this with agent loops),
applies the calibration policy, and emits the full per-experiment artifact
directory at ``results/gbdt/experiments/<experiment_name>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from gbdt import data as gbdt_data
from gbdt import features as gbdt_features
from gbdt.report import compute_segment_diagnostics, emit_figures, render_report
from gbdt.targets import build_target
from gbdt.train import SplitSpec, walk_forward_train
from gbdt.uniqueness import (
    compute_uniqueness_weights,
    effective_sample_size,
    weighted_auc,
    weighted_brier,
)


# ---------------------------------------------------------------------------
# Spec loading + validation
# ---------------------------------------------------------------------------


_VALID_DIRECTIONS = {"up", "down"}
_VALID_CAL_METHODS = {"native", "conditional_isotonic", "isotonic_always", "platt"}

# Issue #32 — sweep-mode "FS+HP loop" is feature-selection-only when the
# loop budget is too small to surface HP variation. We flag the loop as
# ``hp_search_active`` only when ``max_iterations >= _HP_SEARCH_ITER_THRESHOLD``;
# below the threshold the default fallback callback reuses the prior HP
# unchanged on every iteration (FS-only screening). Picked 5 because the
# canonical 8-iter budget comfortably exceeds it while the 3-iter sweep
# budget falls below — see the issue for the why.
_HP_SEARCH_ITER_THRESHOLD = 5

# Issue #31 — minimum number of usable test rows the runner expects to
# emit before it warns the user that the test segment was structurally
# eaten by the horizon. Picked 100 to match the default ``split.test_rows``
# budget so a fully-eaten test (the H=100 evidence in the issue) is well
# under the bar but the normal case (test_rows ≥ 100 - horizon per
# ticker × tickers) is comfortably above.
_TEST_ROWS_WARNING_THRESHOLD = 100


def load_spec(spec_path: Path, default_path: Path | None = None) -> dict:
    """Load + validate a spec, merging on top of ``default.yaml``.

    Returns the fully-merged spec ready for the runner. The on-disk
    per-experiment spec contents (pre-merge) are also stashed under
    ``__per_experiment_spec__`` for downstream artifact snapshotting —
    see issue #30: the merged dict contains the entire ``universes::``
    registry from defaults, which the snapshot at
    ``results/.../spec.yaml`` MUST NOT echo verbatim.
    """
    raw = yaml.safe_load(spec_path.read_text()) or {}

    default_path = default_path or Path("configs/gbdt/default.yaml")
    defaults = yaml.safe_load(default_path.read_text()) if default_path.exists() else {}

    merged = _deep_merge(defaults, raw)
    _validate_spec(merged)
    # Stash the per-experiment-only contents so the runner can snapshot
    # exactly what the user authored (not the merge with defaults).
    merged["__per_experiment_spec__"] = raw
    return merged


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate_spec(spec: dict) -> None:
    target = spec.get("target")
    if not target:
        raise ValueError("spec.target is required")
    for k in ("universe", "direction", "threshold_pct", "horizon_days"):
        if k not in target:
            raise ValueError(f"spec.target.{k} is required")
    if target["direction"] not in _VALID_DIRECTIONS:
        raise ValueError(
            f"spec.target.direction must be in {_VALID_DIRECTIONS}, got {target['direction']!r}"
        )
    if target["threshold_pct"] <= 0:
        raise ValueError("spec.target.threshold_pct must be > 0")
    if target["horizon_days"] <= 0:
        raise ValueError("spec.target.horizon_days must be > 0")
    md = target.get("max_drawdown")
    if md is not None and not (0 < md < 1):
        raise ValueError(f"spec.target.max_drawdown must be in (0, 1), got {md}")
    uw = target.get("uniqueness_weighting", True)
    if not isinstance(uw, bool):
        raise ValueError(
            f"spec.target.uniqueness_weighting must be bool, got {uw!r}"
        )

    backend = spec.get("backend", {}) or {}
    if backend.get("library", "catboost") != "catboost":
        raise ValueError("v1 supports backend.library='catboost' only")
    cal = backend.get("calibration_method", "conditional_isotonic")
    if cal not in _VALID_CAL_METHODS:
        raise ValueError(f"backend.calibration_method must be in {_VALID_CAL_METHODS}")
    loop = backend.get("fs_hp_loop", {}) or {}
    if "max_iterations" in loop and not (1 <= loop["max_iterations"] <= 16):
        raise ValueError("backend.fs_hp_loop.max_iterations must be in [1, 16]")
    sp = spec.get("split", {}) or {}
    if sp:
        total = (sp.get("train_rows", 0) + sp.get("val_rows", 0)
                  + sp.get("eval_rows", 0) + sp.get("test_rows", 0))
        if total > sp.get("min_rows_per_ticker", total):
            raise ValueError("split sum exceeds min_rows_per_ticker")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_hash(spec: dict) -> str:
    # Hash on the merged-but-public spec; strip internal keys so the
    # presence of ``__per_experiment_spec__`` (added by ``load_spec`` to
    # back the snapshot path for issue #30) doesn't change the hash.
    return "sha256:" + hashlib.sha256(
        json.dumps(_strip_internal_keys(spec), sort_keys=True, default=str).encode()
    ).hexdigest()


def _strip_internal_keys(spec: dict) -> dict:
    """Return a copy of ``spec`` with internal ``__*__`` keys removed.

    The merged spec carries ``__per_experiment_spec__`` (see ``load_spec``)
    which must not be hashed or persisted into downstream artifacts.
    """
    return {k: v for k, v in spec.items() if not (
        isinstance(k, str) and k.startswith("__") and k.endswith("__")
    )}


def _project_test_rows(
    panel: pd.DataFrame,
    *,
    test_rows_per_ticker: int,
    horizon_days: int,
) -> dict:
    """Estimate the number of usable rows the test segment will yield.

    The walk-forward driver carves the trailing ``test_rows`` positional
    rows per ticker as the test segment. The target builder produces
    ``NaN`` for the last ``horizon_days`` rows per ticker (forward window
    incomplete), and the driver drops NaN-target rows before scoring. So
    the expected usable test row count per ticker is
    ``max(0, test_rows - horizon_days)``, summed across kept tickers.

    Returns a dict with ``expected_test_rows``, ``per_ticker_usable``,
    and ``n_tickers`` so the caller can compose a human-readable warning.
    """
    n_tickers = int(panel.index.get_level_values("ticker").nunique())
    per_ticker_usable = max(0, int(test_rows_per_ticker) - int(horizon_days))
    expected = per_ticker_usable * n_tickers
    return {
        "expected_test_rows": int(expected),
        "per_ticker_usable": int(per_ticker_usable),
        "n_tickers": int(n_tickers),
        "test_rows_per_ticker": int(test_rows_per_ticker),
        "horizon_days": int(horizon_days),
    }


def _format_test_split_warning(projection: dict, threshold: int) -> str:
    """One-line, human-readable explanation of the test_rows projection."""
    per = projection["per_ticker_usable"]
    h = projection["horizon_days"]
    tpt = projection["test_rows_per_ticker"]
    n = projection["n_tickers"]
    exp = projection["expected_test_rows"]
    if per == 0:
        return (
            f"Test segment expected to be EMPTY: horizon_days={h} >= "
            f"split.test_rows={tpt}, so every ticker's trailing {tpt} rows "
            f"have NaN targets (forward window incomplete). "
            f"headline_test will be {{}} and predictions/test.csv will be "
            f"header-only. Eval segment is still measured. "
            f"(threshold={threshold})"
        )
    return (
        f"Test segment will be SMALL: per-ticker usable = "
        f"max(0, test_rows={tpt} - horizon_days={h}) = {per}; "
        f"{n} kept ticker(s) -> expected ~{exp} rows < threshold={threshold}. "
        f"Headline_test will be computed but may be unreliable."
    )


def _data_hash(panel: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(panel.shape).encode())
    h.update(str(panel.index[:5].tolist()).encode())
    h.update(str(panel.index[-5:].tolist()).encode())
    return "sha256:" + h.hexdigest()


def _compute_headline(pred_df: pd.DataFrame | None) -> dict:
    """Headline metrics on a prediction segment.

    Uses LdP §4.4 sample weights from the ``sample_weight`` column when
    present (uniform-1.0 fallback collapses to unweighted metrics so
    legacy callers and the opt-out path produce numerically-identical
    outputs).
    """
    if pred_df is None or pred_df.empty:
        return {}
    y = pred_df["y_true"].values.astype(int)
    p = pred_df["p_calibrated"].values
    if "sample_weight" in pred_df.columns:
        w = pred_df["sample_weight"].values.astype(float)
    else:
        w = np.ones_like(y, dtype=float)

    total_w = float(w.sum())
    base = float(np.sum(w * y) / total_w) if total_w > 0 else float(np.mean(y))
    brier = weighted_brier(y, p, w)
    brier_base = weighted_brier(y, np.full_like(y, base, dtype=float), w)
    # Unweighted variants kept for backward-compat / cross-check.
    brier_unw = float(brier_score_loss(y, p))
    brier_base_unw = float(brier_score_loss(
        y, np.full_like(y, float(np.mean(y)), dtype=float),
    ))
    # log_loss has a sample_weight kwarg.
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w))
    out = {
        "brier": float(brier),
        "brier_baseline_baserate": float(brier_base),
        "brier_improvement_vs_baseline": float(brier_base - brier),
        "log_loss": ll,
        "brier_unweighted": brier_unw,
        "brier_baseline_baserate_unweighted": brier_base_unw,
        "brier_improvement_vs_baseline_unweighted": brier_base_unw - brier_unw,
        "effective_sample_size_kish": float(effective_sample_size(w)),
        "sum_weights": float(w.sum()),
        "n_rows": int(len(y)),
        "weighted_prevalence": base,
    }
    out["roc_auc"] = weighted_auc(y, p, w)
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_experiment(spec_path: Path, *, overwrite: bool = False,
                    repo_root: Path | None = None) -> Path:
    """Run the experiment end-to-end. Returns the artifact dir path."""
    spec_path = Path(spec_path).resolve()
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    spec = load_spec(spec_path, default_path=repo_root / "configs/gbdt/default.yaml")
    name = spec_path.stem
    out_root = repo_root / spec.get("artifacts", {}).get(
        "experiment_dir", "results/gbdt/experiments"
    )
    out_dir = Path(out_root) / name
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        print(f"[experiment] artifact dir already exists at {out_dir}", file=sys.stderr)
        print("[experiment] pass --overwrite to replace", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[experiment] start spec={spec_path.name} -> {out_dir}", flush=True)
    t0 = time.time()

    # -------- Phase 1: data --------
    target = spec["target"]
    dr = spec.get("date_range", {}) or {}
    split_d = spec.get("split", {}) or {}
    split = SplitSpec(
        train_rows=split_d.get("train_rows", 800),
        val_rows=split_d.get("val_rows", 400),
        eval_rows=split_d.get("eval_rows", 200),
        test_rows=split_d.get("test_rows", 100),
    )
    min_rows = split_d.get("min_rows_per_ticker", split.total)

    print(f"[data] start universe={target['universe']}", flush=True)
    t1 = time.time()
    data_cfg = spec.get("data", {}) or {}
    staleness_days = int(data_cfg.get(
        "staleness_days", gbdt_data.DEFAULT_STALENESS_DAYS,
    ))
    panel_obj = gbdt_data.load_panel(
        target["universe"],
        start=dr.get("start"),
        end=dr.get("end"),
        min_rows=min_rows,
        repo_root=repo_root,
        staleness_days=staleness_days,
    )
    if panel_obj.stale_tickers:
        print(
            f"[data] warning: {len(panel_obj.stale_tickers)} stale ticker(s) "
            f"(cache > {staleness_days}d old): "
            f"{panel_obj.stale_tickers[:5]}{'...' if len(panel_obj.stale_tickers) > 5 else ''}",
            flush=True,
        )
    print(f"[data] complete in {time.time()-t1:.1f}s rows={len(panel_obj.panel)} "
           f"tickers_kept={len(panel_obj.tickers_kept)}", flush=True)

    # -------- Phase 1b: project test segment size + warn if structurally slim --------
    # Issue #31 — the walk-forward driver silently emits an empty test
    # segment when ``horizon_days >= split.test_rows``: every ticker's
    # trailing ``test_rows`` rows have NaN targets (forward window
    # incomplete) and get dropped before scoring. The user saw
    # ``headline_test={}`` + ``predictions/test.csv`` with only a header
    # and no warning. Project the row count here and emit a clear warning
    # to the run log; the same string is persisted into
    # ``metrics.json::data.test_split_warning`` and surfaced in
    # ``report.md`` so downstream readers can't miss it. We do NOT
    # auto-shift the split (deferred to V2 per the issue).
    test_split_projection = _project_test_rows(
        panel_obj.panel,
        test_rows_per_ticker=split.test_rows,
        horizon_days=int(target["horizon_days"]),
    )
    test_split_warning: str | None = None
    if test_split_projection["expected_test_rows"] < _TEST_ROWS_WARNING_THRESHOLD:
        test_split_warning = _format_test_split_warning(
            test_split_projection, _TEST_ROWS_WARNING_THRESHOLD,
        )
        print(f"[data] WARNING: {test_split_warning}", flush=True)

    # -------- Phase 2: features --------
    print("[features] start", flush=True)
    t1 = time.time()
    fcfg = spec.get("features", {}) or {}
    lookbacks = tuple(fcfg.get("lookback_windows", gbdt_features.DEFAULT_LOOKBACKS))
    families = fcfg.get("candidates", "all")
    exclude = fcfg.get("exclude") or []
    X = gbdt_features.build_feature_matrix(
        panel_obj.panel, panel_obj.index_series,
        lookbacks=lookbacks,
        annualization=panel_obj.annualization_factor,
        families=families, exclude=exclude,
    )
    # Drop all-NaN columns (some features may produce no values on a short-history ticker).
    X = X.dropna(axis=1, how="all")
    print(f"[features] complete in {time.time()-t1:.1f}s shape={X.shape}", flush=True)

    # -------- Phase 3: target --------
    print("[target] start", flush=True)
    t1 = time.time()
    y = build_target(
        panel_obj.panel,
        direction=target["direction"],
        threshold_pct=target["threshold_pct"],
        horizon_days=target["horizon_days"],
        max_drawdown=target.get("max_drawdown"),
    )
    print(f"[target] complete in {time.time()-t1:.1f}s "
           f"positive_prevalence={float(y.dropna().mean()):.3f}", flush=True)

    # -------- Phase 3b: sample-uniqueness weights (LdP §4.4) --------
    # ON by default. Opt-out reproduces the legacy (biased) behavior
    # where every row enters the loss with weight 1.0 — useful only for
    # reproducing pre-PR results / measuring the overlap-bias delta.
    uniqueness_on = bool(target.get("uniqueness_weighting", True))
    if uniqueness_on:
        print("[uniqueness] start", flush=True)
        t1 = time.time()
        sample_weights = compute_uniqueness_weights(
            panel_obj.panel, horizon=int(target["horizon_days"]),
        )
        # Effective-sample-size summary across the full panel (pre-segment)
        ess_full = float(effective_sample_size(sample_weights.values))
        print(
            f"[uniqueness] complete in {time.time()-t1:.1f}s "
            f"horizon={target['horizon_days']} rows={len(sample_weights)} "
            f"ESS={ess_full:.0f} inflation={len(sample_weights)/max(ess_full,1):.2f}x",
            flush=True,
        )
    else:
        sample_weights = None
        print("[uniqueness] disabled by spec (target.uniqueness_weighting=false)",
              flush=True)

    # -------- Phase 4: walk-forward + FS+HP loop --------
    backend = spec.get("backend", {}) or {}
    hp_starting = backend.get("hp_starting", {}) or {}
    loop_cfg = backend.get("fs_hp_loop", {}) or {}
    cal_method = backend.get("calibration_method", "conditional_isotonic")
    cal_z_thr = backend.get("calibration_z_threshold", 2.0)
    seed = spec.get("random_seed", 42)

    print(f"[loop] start max_iter={loop_cfg.get('max_iterations', 8)}", flush=True)
    t1 = time.time()
    result = walk_forward_train(
        panel=panel_obj.panel, X=X, y=y,
        features=list(X.columns), hp=dict(hp_starting), split=split,
        calibration_method=cal_method,
        calibration_z_threshold=cal_z_thr,
        max_iterations=loop_cfg.get("max_iterations", 8),
        plateau_threshold=loop_cfg.get("plateau_threshold", 0.005),
        degradation_gate=loop_cfg.get("degradation_gate", 0.01),
        random_seed=seed,
        sample_weights=sample_weights,
    )
    print(f"[loop] complete in {time.time()-t1:.1f}s best_iter={result.best_iteration} "
           f"val_brier={result.best_val_brier:.4f} signal={result.inner_stop_signal}",
           flush=True)

    # -------- Phase 5: artifact emit --------
    print("[artifact] start", flush=True)
    t1 = time.time()

    # Issue #30 — the snapshot at ``spec.yaml`` MUST be the per-experiment
    # spec as authored on disk, NOT the merge of defaults+spec. Dumping
    # the merged dict (the pre-fix behaviour) buried the actual target
    # under hundreds of lines of universe-registry content from defaults,
    # which made archived artifacts look corrupted (see the
    # nasdaq100_up_10pct_100d_dd5pct_pre_uniqueness_fix archive: the
    # snapshot's first 30 lines listed nifty50/nifty100/... while the
    # actual target was buried at the bottom).
    per_exp_spec = spec.get("__per_experiment_spec__") or {
        k: v for k, v in spec.items()
        if k in ("target", "date_range", "split", "features", "backend",
                  "random_seed", "artifacts", "data")
    }
    # Defence-in-depth: refuse to write a snapshot whose target.universe
    # disagrees with the run we just executed. This is the fail-loud
    # regression assertion called for in issue #30 — even if a future
    # refactor regresses the snapshot path, the user will not silently
    # get an artifact pointing at the wrong universe.
    snap_universe = (per_exp_spec.get("target") or {}).get("universe")
    run_universe = target["universe"]
    if snap_universe is not None and snap_universe != run_universe:
        raise RuntimeError(
            f"spec.yaml snapshot universe mismatch: snapshot says "
            f"{snap_universe!r} but the run actually executed against "
            f"{run_universe!r}. This is a runner bug — refusing to "
            f"persist a misleading artifact (see issue #30)."
        )
    (out_dir / "spec.yaml").write_text(
        yaml.safe_dump(per_exp_spec, sort_keys=False)
    )

    result.best_model.save(out_dir / "model.cbm")
    # Always write a pickle. When no calibrator is needed (native pass) we
    # still pickle ``None`` so downstream ``pickle.load`` is uniform — see
    # PR #8 review (Minor 2): a plaintext-vs-pickle mix produced
    # ``UnpicklingError: invalid load key, '#'.``
    with open(out_dir / "calibration.pkl", "wb") as f:
        pickle.dump(result.calibration.calibrator, f)

    # YAML artifacts are written as explicit top-level-keyed dicts (not
    # bare collections) so they are self-describing and merge/diff cleanly
    # in the cross-experiment table — see PR #8 review (Minor 3).
    (out_dir / "features.yaml").write_text(
        yaml.safe_dump({"features": list(result.best_features)}, sort_keys=False)
    )
    (out_dir / "hp.yaml").write_text(
        yaml.safe_dump({"hp": dict(result.best_hp)}, sort_keys=False)
    )

    with open(out_dir / "iterations.jsonl", "w") as f:
        last_idx = len(result.iterations) - 1
        for i, b in enumerate(result.iterations):
            d = b.to_dict()
            d["inner_stop_signal"] = (
                result.inner_stop_signal if i == last_idx else None
            )
            f.write(json.dumps(d, default=str) + "\n")

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for seg, df in result.predictions.items():
        df.to_csv(pred_dir / f"{seg}.csv", index=False)

    headline_eval = _compute_headline(result.predictions.get("eval"))
    headline_test = _compute_headline(result.predictions.get("test"))
    train_pred = result.predictions.get("train")
    val_pred = result.predictions.get("val")

    # Per-fold ESS — single-fold for v1 (carve_single_fold), so this is a
    # one-entry dict keyed by ``"fold_0"``. Multi-fold mode (V1.1_TBD)
    # will populate one entry per fold.
    #
    # We report two distinct quantities and they answer different questions:
    #   ess_kish (Kish):           (Σw)² / Σw² — variance-effective sample
    #                              size; reduces to ``n`` for uniform
    #                              weights regardless of scale. Use for
    #                              confidence intervals on weighted means.
    #   sum_weights (independent): Σw — approximate count of *independent*
    #                              forward events the panel encodes. For
    #                              uniqueness weights this is ≈ n/(2H-1)
    #                              when n >> H. Use for "how much
    #                              information is actually here".
    def _seg_ess(seg: str) -> dict[str, float | int | None]:
        df = result.predictions.get(seg)
        if df is None or df.empty:
            return {
                "ess_kish": None,
                "sum_weights": None,
                "n_rows": 0,
                "overlap_inflation_ratio": None,
            }
        w = df["sample_weight"].values.astype(float) if "sample_weight" in df.columns \
            else np.ones(len(df), dtype=float)
        ess_kish = float(effective_sample_size(w))
        s = float(w.sum())
        n = int(len(df))
        ratio = float(n / max(s, 1.0))
        return {
            "ess_kish": ess_kish,
            "sum_weights": s,
            "n_rows": n,
            "overlap_inflation_ratio": ratio,
        }

    ess_summary = {
        "uniqueness_weighting": uniqueness_on,
        "horizon_days": int(target["horizon_days"]),
        "effective_sample_size_per_fold": {
            "fold_0": {
                "train": _seg_ess("train"),
                "val": _seg_ess("val"),
                "eval": _seg_ess("eval"),
                "test": _seg_ess("test"),
            },
        },
    }
    metrics = {
        "experiment_name": name,
        "spec_hash": _spec_hash(spec),
        "data_hash": _data_hash(panel_obj.panel),
        "data": {
            "n_tickers_in_universe": len(panel_obj.statuses),
            "n_tickers_used": len(panel_obj.tickers_kept),
            "tickers_excluded": panel_obj.tickers_excluded,
            # Cache freshness / NaN-row drop telemetry (PR #8 review, Minor 1+4).
            "staleness_days_threshold": panel_obj.staleness_days_threshold,
            "stale_tickers": panel_obj.stale_tickers,
            "n_tickers_stale": len(panel_obj.stale_tickers),
            "cache_age_days_by_ticker": {
                s.ticker: s.cache_age_days
                for s in panel_obj.statuses
                if s.kept and s.cache_age_days is not None
            },
            "nan_rows_dropped_by_ticker": {
                s.ticker: s.nan_rows_dropped
                for s in panel_obj.statuses
                if s.nan_rows_dropped > 0
            },
            "n_rows_train": int(len(train_pred)) if train_pred is not None else 0,
            "n_rows_val": int(len(val_pred)) if val_pred is not None else 0,
            "n_rows_eval": int(len(result.predictions.get("eval", pd.DataFrame()))),
            "n_rows_test": int(len(result.predictions.get("test", pd.DataFrame()))),
            "positive_prevalence_train": (
                float(train_pred["y_true"].mean())
                if train_pred is not None and len(train_pred) else None
            ),
            "positive_prevalence_eval": (
                float(result.predictions.get("eval", pd.DataFrame({"y_true": []}))["y_true"].mean())
                if len(result.predictions.get("eval", pd.DataFrame())) else None
            ),
            # Issue #31 — surfaces structurally-thin/empty test segments
            # (horizon eats the test window). Absent/None when the test
            # segment is expected to be normally sized; a human-readable
            # string explaining the calculation when below threshold.
            "test_split_warning": test_split_warning,
            "test_split_projection": test_split_projection,
        },
        "loop": {
            "n_iterations_run": len(result.iterations),
            "best_iteration": int(result.best_iteration),
            "inner_stop_signal": result.inner_stop_signal,
            # Issue #32 — the default FS+HP callback only nudges HPs in
            # response to overfit/cap signals; when ``max_iterations`` is
            # small (sweep mode = 3) the loop typically reuses the
            # starting HP unchanged across every iteration, so the
            # ``hp_history`` field is honest about FS-only behaviour.
            # We flag this here so artifact readers don't misinterpret
            # the "FS+HP loop" name as evidence of real HP search.
            "hp_search_active": bool(
                int(loop_cfg.get("max_iterations", 8))
                >= _HP_SEARCH_ITER_THRESHOLD
            ),
            "hp_search_iter_threshold": int(_HP_SEARCH_ITER_THRESHOLD),
            "max_iterations": int(loop_cfg.get("max_iterations", 8)),
        },
        "calibration": {
            "method": cal_method,
            "decision": result.calibration.method,
            "spiegelhalter_z": result.calibration.spiegelhalter_z,
            "spiegelhalter_p": result.calibration.spiegelhalter_p,
        },
        "sample_uniqueness": ess_summary,
        "headline_eval": headline_eval,
        "headline_test": headline_test,
        # Per-segment top-K + per-ticker + per-quarter + pred-range
        # diagnostics. Operate on the same (date, ticker, p_calibrated,
        # y_true) row schema the headline metrics consume; covers eval
        # and test segments (empty-shaped block for missing/empty).
        "segment_diagnostics": compute_segment_diagnostics(result.predictions),
        "wall_time_total_sec": time.time() - t0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    emit_figures(out_dir, result.iterations, result.predictions)
    render_report(out_dir)

    print(f"[artifact] complete in {time.time()-t1:.1f}s -> {out_dir}", flush=True)
    print(f"[experiment] complete in {time.time()-t0:.1f}s", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gbdt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("experiment", help="Run one gbdt experiment end-to-end")
    p_exp.add_argument("spec", type=Path, help="Path to spec YAML")
    p_exp.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing non-empty artifact dir")

    args = parser.parse_args(argv)
    if args.cmd == "experiment":
        run_experiment(args.spec, overwrite=args.overwrite)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
