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
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from gbdt import checkpoint as gbdt_checkpoint
from gbdt import data as gbdt_data
from gbdt import features as gbdt_features
from gbdt import loop_protocol
from gbdt.heartbeat import Heartbeat
from gbdt.model import _validate_hp, model_filename
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
# V1.2 Phase 5 (plan § 6.1): the runner accepts both GBDT backends behind the
# same experiment-loop contract. ``backend.library`` selects the model adapter
# (``gbdt.model.make_model``), the HP-validation tables, and the persisted-model
# filename (``gbdt.model.model_filename``).
_VALID_BACKENDS = {"catboost", "xgboost"}

# V1.1 — selects which FS+HP iteration callback drives the walk-forward loop.
# ``default`` (or absent) preserves v1 behaviour: ``walk_forward_train`` falls
# back to ``default_fs_hp_callback`` (the algorithmic prune + HP nudge).
# ``agent_file_protocol`` is the exit-and-resume agent loop (callback body
# lands in V1.1 Phase 2). See ``docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md``.
_VALID_CALLBACK_MODES = {"default", "agent_file_protocol"}
_DEFAULT_CALLBACK_MODE = "default"

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

    # V1.2 Phase 5 (plan § 6.3): ``default.yaml``'s ``backend.hp_starting`` /
    # ``backend.hp_pinned`` carry CatBoost-named knobs (``iterations``, ``depth``,
    # ``l2_leaf_reg``, ``has_time`` …). A deep-merge would leak those into an
    # XGBoost spec's ``hp_starting``, producing an HP dict with both vocabularies
    # — which XGBoost can't consume. So when the per-experiment spec selects a
    # non-catboost backend, drop the default's CatBoost HP blocks before merging;
    # the spec carries its own (XGBoost-named, backend-validated) ``hp_starting``.
    raw_library = ((raw.get("backend") or {}).get("library")) or "catboost"
    if raw_library != "catboost":
        defaults = dict(defaults)
        if isinstance(defaults.get("backend"), dict):
            defaults["backend"] = {
                k: v for k, v in defaults["backend"].items()
                if k not in ("hp_starting", "hp_pinned")
            }

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
    # V1.2 Phase 5 (plan § 6.1/§ 6.2): the runner now accepts both backends.
    # ``backend.library`` is the load-bearing switch; the iter-0 ``hp_starting``
    # is validated against the backend's HP tables here (fail-fast on an
    # out-of-vocab / out-of-range / pinned-override HP before any data is loaded)
    # so a malformed XGBoost spec is rejected at parse time, not deep in the loop.
    library = backend.get("library", "catboost")
    if library not in _VALID_BACKENDS:
        raise ValueError(
            f"backend.library must be in {sorted(_VALID_BACKENDS)}, got {library!r}"
        )
    hp_starting = backend.get("hp_starting", {}) or {}
    if hp_starting:
        try:
            _validate_hp(dict(hp_starting), backend=library)
        except ValueError as exc:
            raise ValueError(
                f"backend.hp_starting is invalid for backend.library={library!r}: "
                f"{exc}"
            ) from exc
    cal = backend.get("calibration_method", "conditional_isotonic")
    if cal not in _VALID_CAL_METHODS:
        raise ValueError(f"backend.calibration_method must be in {_VALID_CAL_METHODS}")
    loop = backend.get("fs_hp_loop", {}) or {}
    if "max_iterations" in loop and not (1 <= loop["max_iterations"] <= 16):
        raise ValueError("backend.fs_hp_loop.max_iterations must be in [1, 16]")
    if "callback_mode" in loop and loop["callback_mode"] not in _VALID_CALLBACK_MODES:
        raise ValueError(
            f"backend.fs_hp_loop.callback_mode must be in {sorted(_VALID_CALLBACK_MODES)}, "
            f"got {loop['callback_mode']!r}"
        )
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


def _collect_preflight(repo_root: Path) -> dict:
    """Capture a fingerprint of the cache + code state at run start.

    Six fields, captured BEFORE any data is loaded so the fingerprint
    survives even when the data stage fails:

    - ``cache_db``: resolved absolute path to ``<repo_root>/data/processed.db``.
      Symlinks are followed (via ``os.path.realpath``) so a run reading
      from a tmpfs-backed cache mounted at ``data/`` records the real
      target path. Empty string if the file does not exist.
    - ``cache_db_size``: size in bytes; ``0`` if missing.
    - ``cache_db_mtime``: UTC ISO-8601 mtime; empty string if missing.
    - ``data_root``: resolved absolute path to ``<repo_root>/data``
      (the directory data_pipelines was rooted at). Same realpath rule.
    - ``code_commit``: ``git rev-parse HEAD`` output, or ``"unknown"``
      when git is unavailable / repo_root is not a git checkout.
    - ``code_dirty``: ``True`` when ``git status --porcelain`` is
      non-empty, else ``False``. Defaults to ``False`` when git is
      unavailable (matches the ``code_commit="unknown"`` fail-safe).

    Motivated by the ``/tmp/exp_data`` tmpfs wipe (May 27): two runs on
    consecutive dates reading from a transient cache looked identical
    in their artifacts but in fact consumed different cache snapshots.
    Persisting these six fields into the artifact (``metrics.json``)
    makes archived runs self-describing post-hoc.
    """
    data_root = Path(repo_root) / "data"
    cache_db = data_root / "processed.db"

    data_root_resolved = os.path.realpath(data_root)
    if cache_db.exists():
        cache_db_resolved = os.path.realpath(cache_db)
        try:
            st = os.stat(cache_db_resolved)
            cache_db_size = int(st.st_size)
            cache_db_mtime = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc,
            ).isoformat()
        except OSError:
            cache_db_size = 0
            cache_db_mtime = ""
    else:
        cache_db_resolved = ""
        cache_db_size = 0
        cache_db_mtime = ""

    code_commit = "unknown"
    code_dirty = False
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if rev.returncode == 0:
            commit = rev.stdout.strip()
            if commit:
                code_commit = commit
                status = subprocess.run(
                    ["git", "-C", str(repo_root), "status", "--porcelain"],
                    capture_output=True, text=True, check=False, timeout=5,
                )
                if status.returncode == 0:
                    code_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # ``git`` binary missing, perms denied, hung, etc. — preserve the
        # ``unknown``/``False`` fail-safe and continue. Never crash a run.
        pass

    return {
        "cache_db": cache_db_resolved,
        "cache_db_size": cache_db_size,
        "cache_db_mtime": cache_db_mtime,
        "data_root": data_root_resolved,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
    }


def _format_preflight_line(pf: dict) -> str:
    return (
        f"[preflight] cache_db={pf['cache_db']} "
        f"cache_db_size={pf['cache_db_size']} "
        f"cache_db_mtime={pf['cache_db_mtime']} "
        f"data_root={pf['data_root']} "
        f"code_commit={pf['code_commit']} "
        f"code_dirty={pf['code_dirty']}"
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
# FS+HP loop callback resolution (V1.1)
# ---------------------------------------------------------------------------


def _resolve_callback(
    loop_cfg: dict,
    run_id: str,
    *,
    artifact_dir: Path | None = None,
    loop_state_sink: dict | None = None,
    max_iterations: int = 8,
    cell: dict | None = None,
):
    """Select the FS+HP iteration callback from ``backend.fs_hp_loop`` config.

    Returns either ``None`` (so ``walk_forward_train`` keeps using its built-in
    ``default_fs_hp_callback`` — v1 behaviour, byte-for-byte preserved) or a
    callable matching the ``(bundle, current_features) -> (keep, next_hp,
    rationale)`` signature.

    - ``callback_mode == "default"`` (or absent) → ``None``.
    - ``callback_mode == "agent_file_protocol"`` → the exit-and-resume callback
      (plan § 0): at the end of iteration N it writes
      ``loop/iter_<N>_request.json`` (the minimal request bundle) + a resume
      checkpoint, then raises :class:`~gbdt.loop_protocol.PauseForAgentDecision`
      to hand control back to the agent. ``run_experiment`` catches the pause,
      logs the ``--resume`` hint, and exits cleanly.

    The agent-file-protocol callback needs the artifact dir (where loop files
    land) and the live loop history (``loop_state_sink``, populated by
    ``walk_forward_train`` before each callback invocation). When those are not
    wired (e.g. a Phase 1-style resolution-only call) it raises a clear error
    instead of silently mis-writing.
    """
    mode = (loop_cfg or {}).get("callback_mode", _DEFAULT_CALLBACK_MODE)
    if mode == "default":
        return None
    if mode == "agent_file_protocol":
        return _make_agent_file_protocol_callback(
            run_id=run_id,
            artifact_dir=artifact_dir,
            loop_state_sink=loop_state_sink,
            max_iterations=max_iterations,
            cell=cell,
        )
    # Unreachable when the spec passed _validate_spec, but defend anyway.
    raise ValueError(
        f"unknown callback_mode: {mode!r} (expected one of {sorted(_VALID_CALLBACK_MODES)})"
    )


def _make_agent_file_protocol_callback(
    *,
    run_id: str,
    artifact_dir: Path | None,
    loop_state_sink: dict | None,
    max_iterations: int,
    cell: dict | None = None,
):
    """Build the exit-and-resume callback (plan § 0).

    On invocation (end of iteration N, loop continuing) it: (1) builds + writes
    the minimal request bundle to ``loop/iter_<N>_request.json``; (2) writes a
    resume checkpoint capturing exactly what's needed to seed iter N+1 without
    re-training 0..N (iteration index, accumulated history, current features +
    HP, run/spec identity — NO model blobs, plan § 0.2); (3) raises
    :class:`PauseForAgentDecision`. ``run_experiment`` catches it and exits 0.
    """
    def _cb(bundle, current_features):
        if artifact_dir is None or loop_state_sink is None:
            raise RuntimeError(
                "agent_file_protocol callback invoked without artifact_dir / "
                "loop_state_sink wired — resolve it via run_experiment, not the "
                "Phase-1 resolution-only path."
            )
        state = dict(loop_state_sink)
        iter_n = int(state["iter_idx"])
        # (1) request bundle (Phase 3: the diagnose.json-shaped payload built
        # in-memory from this iteration's DiagnosticBundle — reuses the
        # /gbdt-diagnose pure helpers, no matrix rebuild). ``artifact_dir`` +
        # ``cell`` are surfaced so the agent can run the full on-disk diagnose
        # for the matrix-dependent analyses (plan § 0.5).
        req_payload = loop_protocol.build_request_bundle(
            bundle,
            iter_n=iter_n,
            run_id=run_id,
            max_iterations=int(state.get("max_iterations", max_iterations)),
            available_features=list(current_features),
            artifact_dir=artifact_dir,
            cell=cell,
        )
        req_path = loop_protocol.write_request(artifact_dir, iter_n, req_payload)
        # (2) resume checkpoint — full loop state, no model blobs.
        ckpt_state = {
            "run_id": run_id,
            "iter_idx": iter_n,
            "max_iterations": int(state.get("max_iterations", max_iterations)),
            "current_features": list(state["current_features"]),
            "current_hp": dict(state["current_hp"]),
            "val_briers": list(state["val_briers"]),
            "hp_history": list(state["hp_history"]),
            "feature_history": list(state["feature_history"]),
            "hp_lists": list(state["hp_lists"]),
            "delta_attributions": list(state["delta_attributions"]),
        }
        ckpt_path = gbdt_checkpoint.write_checkpoint(artifact_dir, ckpt_state)
        # (3) hand control back to the agent.
        raise loop_protocol.PauseForAgentDecision(
            iter_n=iter_n,
            request_path=req_path,
            checkpoint_path=ckpt_path,
            run_id=run_id,
        )
    return _cb


def _load_and_apply_resume(out_dir: Path, spec: dict, *, run_id: str) -> dict:
    """Load the checkpoint + the agent's decision, validate + apply it.

    Returns the ``resume_state`` dict ``walk_forward_train`` seeds the loop with
    at iteration N+1 (plan § 0). The checkpoint (written by the
    agent-file-protocol callback when it paused at iter N) carries the iteration
    index, the accumulated history, and the iter-N features/HP. The decision at
    ``loop/iter_<N>_decision.json`` is validated against the spec
    (bounds / pinned / known features — :func:`loop_protocol.validate_decision`)
    and applied: ``prune_features`` removed + ``hp_changes`` merged → the
    features/HP that seed iter N+1.

    Raises a clear error (caller surfaces it) when the checkpoint is missing,
    the decision is missing/malformed, or the decision violates a constraint —
    the user fixes the decision file + relaunches ``--resume``.
    """
    ckpt = gbdt_checkpoint.read_checkpoint(out_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"[resume] no checkpoint at "
            f"{gbdt_checkpoint.checkpoint_path(out_dir)} — cannot --resume a run "
            f"that never paused. Launch the experiment first (without --resume)."
        )
    iter_n = int(ckpt["iter_idx"])
    print(f"[resume] loaded checkpoint at iter {iter_n} (run_id={run_id})", flush=True)

    decision = loop_protocol.read_decision(out_dir, iter_n)

    # Validate against the spec: HP bounds, no pinned-HP changes, prune_features
    # ⊆ the active feature set the checkpoint paused on. The HP names are
    # validated against the spec's backend table (V1.2 Phase 2 / plan § 6.2):
    # an xgboost spec validates against the *_XGB tables, a catboost spec (the
    # default when unset) against the CatBoost tables.
    known_features = list(ckpt["current_features"])
    backend = ((spec or {}).get("backend", {}) or {}).get("library", "catboost")
    loop_protocol.validate_decision(
        decision, spec, known_features, backend=backend,
    )

    next_features, next_hp, should_stop = loop_protocol.apply_decision(
        decision, known_features, dict(ckpt["current_hp"]),
    )
    n_pruned = len(known_features) - len(next_features)
    hp_changed = sorted((decision.get("hp_changes") or {}).keys())
    print(
        f"[resume] decision applied: pruned {n_pruned} feature(s), "
        f"hp_changes={hp_changed}, should_stop={should_stop}",
        flush=True,
    )

    # The applied decision becomes iter N's recorded delta_attribution.
    prior_deltas = list(ckpt.get("delta_attributions", []))
    prior_deltas.append(decision.get("rationale", "agent decision (no rationale)"))

    return {
        "iter_idx": iter_n + 1,
        "current_features": next_features,
        "current_hp": next_hp,
        "val_briers": list(ckpt.get("val_briers", [])),
        "hp_history": list(ckpt.get("hp_history", [])),
        "feature_history": list(ckpt.get("feature_history", [])),
        "hp_lists": list(ckpt.get("hp_lists", [])),
        "delta_attributions": prior_deltas,
        "force_stop": should_stop,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_experiment(spec_path: Path, *, overwrite: bool = False,
                    callback_mode_override: str | None = None,
                    resume: str | None = None,
                    repo_root: Path | None = None) -> Path:
    """Run the experiment end-to-end. Returns the artifact dir path.

    ``callback_mode_override`` (CLI ``--callback-mode``): when set, overrides
    ``backend.fs_hp_loop.callback_mode`` from the spec. Validated against
    :data:`_VALID_CALLBACK_MODES`.

    ``resume`` (CLI ``--resume <run_id>``): V1.1 Phase 1 scaffolding only.
    Accepted + logged; the exit-and-resume control flow lands in Phase 2.
    """
    spec_path = Path(spec_path).resolve()
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    # Pre-flight fingerprint — captured BEFORE spec load / artifact dir
    # checks / data load so even an aborted run leaves a trail in stdout.
    # Persisted into ``metrics.json::preflight`` below for post-hoc audit.
    preflight = _collect_preflight(repo_root)
    print(_format_preflight_line(preflight), flush=True)

    spec = load_spec(spec_path, default_path=repo_root / "configs/gbdt/default.yaml")
    name = spec_path.stem

    # V1.1 — CLI ``--callback-mode`` overrides the spec's
    # ``backend.fs_hp_loop.callback_mode`` (and the snapshotted value, since we
    # mutate the merged spec in place before the snapshot below). Validated the
    # same way as the spec-level field.
    if callback_mode_override is not None:
        if callback_mode_override not in _VALID_CALLBACK_MODES:
            raise ValueError(
                f"--callback-mode must be in {sorted(_VALID_CALLBACK_MODES)}, "
                f"got {callback_mode_override!r}"
            )
        spec.setdefault("backend", {}).setdefault("fs_hp_loop", {})[
            "callback_mode"
        ] = callback_mode_override
        # Mirror the override into the per-experiment snapshot source (issue
        # #30) so the persisted spec.yaml reflects the *effective* callback
        # mode this run actually used, not the on-disk default.
        per_exp = spec.get("__per_experiment_spec__")
        if isinstance(per_exp, dict):
            per_exp.setdefault("backend", {}).setdefault("fs_hp_loop", {})[
                "callback_mode"
            ] = callback_mode_override

    out_root = repo_root / spec.get("artifacts", {}).get(
        "experiment_dir", "results/gbdt/experiments"
    )
    out_dir = Path(out_root) / name
    # On --resume the artifact dir is EXPECTED to exist (it holds the prior
    # iteration's loop/ request + checkpoint), so the non-empty-dir guard is
    # bypassed. A fresh run still refuses to clobber a non-empty dir.
    if (
        resume is None
        and out_dir.exists()
        and any(out_dir.iterdir())
        and not overwrite
    ):
        print(f"[experiment] artifact dir already exists at {out_dir}", file=sys.stderr)
        print("[experiment] pass --overwrite to replace", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[experiment] start spec={spec_path.name} -> {out_dir}", flush=True)

    # V1.1 exit-and-resume (plan § 0): load the prior checkpoint + the agent's
    # decision, validate + apply it, and build the ``resume_state`` that seeds
    # ``walk_forward_train`` at iter N+1 (without re-training 0..N). ``None``
    # when this is a fresh run.
    resume_state: dict | None = None
    if resume is not None:
        resume_state = _load_and_apply_resume(out_dir, spec, run_id=resume)
    t0 = time.time()

    # Liveness heartbeat: emits [heartbeat] lines on a fixed cadence so a
    # stalled run is detectable (a stale heartbeat = wedged process) without a
    # tight timeout. Daemon thread — dies with the process; stopped explicitly
    # on the normal path below. Disable via GBDT_HEARTBEAT_INTERVAL=0.
    heartbeat = Heartbeat.from_env().start()

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

    heartbeat.set_phase("data")
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
    heartbeat.set_phase("features")
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
    heartbeat.set_phase("target")
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
        heartbeat.set_phase("uniqueness")
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
    backend_library = backend.get("library", "catboost")
    hp_starting = backend.get("hp_starting", {}) or {}
    loop_cfg = backend.get("fs_hp_loop", {}) or {}
    cal_method = backend.get("calibration_method", "conditional_isotonic")
    cal_z_thr = backend.get("calibration_z_threshold", 2.0)
    seed = spec.get("random_seed", 42)

    callback_mode = loop_cfg.get("callback_mode", _DEFAULT_CALLBACK_MODE)
    max_iter = loop_cfg.get("max_iterations", 8)
    # V1.1 — resolve the FS+HP callback from the (possibly CLI-overridden) spec.
    # ``default`` resolves to None so walk_forward_train keeps using its built-in
    # default_fs_hp_callback (v1 behaviour preserved byte-for-byte). The
    # agent_file_protocol callback gets the artifact dir + a live loop-state
    # sink so it can write a complete resume checkpoint before pausing.
    loop_state_sink: dict | None = (
        {} if callback_mode == "agent_file_protocol" else None
    )
    fs_hp_callback = _resolve_callback(
        loop_cfg, run_id=name,
        artifact_dir=out_dir,
        loop_state_sink=loop_state_sink,
        max_iterations=int(max_iter),
        cell={
            k: target.get(k)
            for k in ("universe", "direction", "threshold_pct",
                      "horizon_days", "max_drawdown")
        },
    )

    heartbeat.set_phase("loop")
    resume_note = (
        f" (resume from iter {resume_state['iter_idx']})" if resume_state else ""
    )
    print(
        f"[loop] start max_iter={max_iter} callback_mode={callback_mode}{resume_note}",
        flush=True,
    )
    t1 = time.time()
    try:
        result = walk_forward_train(
            panel=panel_obj.panel, X=X, y=y,
            features=list(X.columns), hp=dict(hp_starting), split=split,
            calibration_method=cal_method,
            calibration_z_threshold=cal_z_thr,
            max_iterations=max_iter,
            plateau_threshold=loop_cfg.get("plateau_threshold", 0.005),
            degradation_gate=loop_cfg.get("degradation_gate", 0.01),
            fs_hp_callback=fs_hp_callback,
            random_seed=seed,
            sample_weights=sample_weights,
            resume_state=resume_state,
            loop_state_sink=loop_state_sink,
            backend=backend_library,
        )
    except loop_protocol.PauseForAgentDecision as pause:
        # Exit half of exit-and-resume (plan § 0): the callback wrote the
        # request bundle + checkpoint and handed control back. Log a
        # copy-pasteable resume hint and return cleanly (NOT an error). The
        # agent reads the request, writes loop/iter_<N>_decision.json, then
        # relaunches `--resume <run_id>` to continue at iter N+1.
        heartbeat.stop()
        print(
            f"[loop] paused at iter {pause.iter_n} — request written: "
            f"{pause.request_path}",
            flush=True,
        )
        print(
            f"[loop] checkpoint written: {pause.checkpoint_path}",
            flush=True,
        )
        print(
            f"[loop] paused at iter {pause.iter_n} — resume with: "
            f"uv run python -m gbdt experiment {spec_path.name} "
            f"--resume {pause.run_id}",
            flush=True,
        )
        return out_dir
    print(f"[loop] complete in {time.time()-t1:.1f}s best_iter={result.best_iteration} "
           f"val_brier={result.best_val_brier:.4f} signal={result.inner_stop_signal}",
           flush=True)

    # -------- Phase 5: artifact emit --------
    heartbeat.set_phase("artifact")
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

    # Backend-determined model filename (V1.2 plan § 4.4): catboost → model.cbm,
    # xgboost → model.ubj. The single source of truth is gbdt.model.model_filename,
    # which the /gbdt-diagnose loader also consults so the two always agree.
    result.best_model.save(out_dir / model_filename(backend_library))
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
        # Which GBDT backend produced this artifact (V1.2 plan § 4.4) — so a
        # post-hoc reader knows whether model.cbm or model.ubj sits beside it,
        # and which model class /gbdt-diagnose must load.
        "backend": {
            "library": backend_library,
            "model_filename": model_filename(backend_library),
        },
        "spec_hash": _spec_hash(spec),
        "data_hash": _data_hash(panel_obj.panel),
        # Pre-flight cache + code fingerprint (see ``_collect_preflight``).
        # Six fields populated even when git is unavailable.
        "preflight": preflight,
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
    heartbeat.stop()
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
    p_exp.add_argument(
        "--callback-mode",
        choices=sorted(_VALID_CALLBACK_MODES),
        default=None,
        help="Override backend.fs_hp_loop.callback_mode from the spec "
             "(default: use the spec's value, or 'default' if absent).",
    )
    p_exp.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume a paused agent-driven FS+HP run (callback_mode="
             "agent_file_protocol). Loads the run's checkpoint + the agent's "
             "loop/iter_<N>_decision.json, validates + applies it, and "
             "continues at iteration N+1. RUN_ID is the value printed in the "
             "pause hint.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "experiment":
        run_experiment(
            args.spec,
            overwrite=args.overwrite,
            callback_mode_override=args.callback_mode,
            resume=args.resume,
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
