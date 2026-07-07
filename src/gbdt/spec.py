"""Spec loading, validation, and hashing for the gbdt experiment runner.

Extracted verbatim from ``gbdt.__main__`` (the v1 Stage 8 orchestrator) by the
runner split — behavior unchanged. ``gbdt.__main__`` re-exports every name
here, so both import paths stay valid.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from gbdt.model import _validate_hp


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

# V1.3 Option A — canonical R-Precision@K + AUC + base_rate CSV the
# anti-AUC cell-flag lookup hits at iter_0 (plan § 3.3 / D3). Lives under
# the repo's results/ tree, regenerated via
# ``scripts/gbdt/regenerate_r_precision_at_k_csv``. The default below is
# repo-relative; ``run_experiment`` resolves it against ``repo_root``.
_DEFAULT_SWEEP_CSV_RELPATH = "results/gbdt/data/r_precision_at_k.csv"

# V1.3 Option A — degenerate-sink warning threshold default (plan § 0 D5).
# Spec-overridable via ``backend.fs_hp_loop.degenerate_sink_threshold``.
_DEFAULT_DEGENERATE_SINK_THRESHOLD = 1.05


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
    # V1.3 Option B (plan § 6): minimal schema validation for the new
    # ``backend.scout`` + ``backend.fs_prefit`` blocks. Both are optional;
    # ``enabled`` must be bool when present; numeric caps must be positive.
    scout_cfg = backend.get("scout", {}) or {}
    if scout_cfg:
        if "enabled" in scout_cfg and not isinstance(scout_cfg["enabled"], bool):
            raise ValueError(
                f"backend.scout.enabled must be bool, got "
                f"{scout_cfg['enabled']!r}"
            )
        for key in ("n_configs_cap", "per_config_timeout_seconds",
                     "wall_clock_cap_seconds"):
            if key in scout_cfg and not (
                isinstance(scout_cfg[key], (int, float))
                and scout_cfg[key] > 0
            ):
                raise ValueError(
                    f"backend.scout.{key} must be a positive number, got "
                    f"{scout_cfg[key]!r}"
                )
    fs_prefit_cfg = backend.get("fs_prefit", {}) or {}
    if fs_prefit_cfg:
        if "enabled" in fs_prefit_cfg and not isinstance(
            fs_prefit_cfg["enabled"], bool,
        ):
            raise ValueError(
                f"backend.fs_prefit.enabled must be bool, got "
                f"{fs_prefit_cfg['enabled']!r}"
            )
        if "cliff_pct" in fs_prefit_cfg:
            cp = fs_prefit_cfg["cliff_pct"]
            if not (isinstance(cp, (int, float)) and 0 <= cp <= 1):
                raise ValueError(
                    f"backend.fs_prefit.cliff_pct must be in [0, 1], got "
                    f"{cp!r}"
                )
    sp = spec.get("split", {}) or {}
    if sp:
        total = (sp.get("train_rows", 0) + sp.get("val_rows", 0)
                  + sp.get("eval_rows", 0) + sp.get("test_rows", 0))
        if total > sp.get("min_rows_per_ticker", total):
            raise ValueError("split sum exceeds min_rows_per_ticker")


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

