"""Exit-and-resume control flow for the V1.1 agent-driven FS+HP loop.

Architecture (``docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md`` § 0,
AUTHORITATIVE): the runner does NOT block-and-poll. Each iteration N the
runner trains, writes a request bundle + resume checkpoint, and **exits
cleanly**. The agent (the user's session) reads the bundle, writes
``loop/iter_<N>_decision.json``, then relaunches ``--resume <run_id>``; the
resumed run validates + applies the decision, runs iteration N+1, and pauses
again. No long-lived blocked process; all state lives on disk.

This module is the Phase 2 control-flow primitives, kept pure (no imports from
``gbdt.__main__``) so they unit-test in isolation:

- :class:`PauseForAgentDecision` — the sentinel exception the agent-file-
  protocol callback raises to hand control back to the agent.
- request / decision file I/O helpers (co-located under ``<artifact_dir>/loop/``
  per plan § 0.6).
- :func:`validate_decision` — mirrors plan § 5 + § 9: HP changes within bounds,
  no pinned-HP changes, ``prune_features`` ⊆ known feature names, types correct.
- :func:`apply_decision` — turns a validated decision into the
  ``(features, hp)`` pair that seeds the next iteration.

Phase 2 wrote a **minimal** request bundle (the in-memory ``DiagnosticBundle``
dumped under ``diagnostics``). **Phase 3** (``docs/gbdt/V1.1_...plan.md`` § 0 Q5)
swaps that payload for the richer ``diagnose.json``-*shaped* dict — assembled
in-memory by :func:`gbdt.diagnose_payload.build_diagnose_payload` (which reuses
the ``/gbdt-diagnose`` pure helpers; it does NOT rebuild the in-sample matrix or
re-fit PDPs). The loop-control envelope (``schema_version``, ``run_id``,
``iter``, ``max_iterations``, ``available_features``) is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gbdt.checkpoint import _LOOP_SUBDIR
from gbdt.diagnose_payload import build_diagnose_payload
from gbdt.model import hp_tables_for

# Bumped on any breaking change to the request/decision file shapes
# (plan § 12 R6). Tracks the checkpoint schema independently.
REQUEST_SCHEMA_VERSION = "v1"
DECISION_SCHEMA_VERSION = "v1"


# ---------------------------------------------------------------------------
# Sentinel exception — the exit half of exit-and-resume
# ---------------------------------------------------------------------------


class PauseForAgentDecision(Exception):
    """Raised by the agent-file-protocol callback to pause the FS+HP loop.

    The callback has already written ``iter_<N>_request.json`` + the resume
    checkpoint before raising. ``run_experiment`` catches this, logs a
    copy-pasteable ``--resume`` hint, and exits **0** (a clean pause, not an
    error). The agent reads the request, writes the decision file, and
    relaunches with ``--resume <run_id>`` to continue at iteration N+1.
    """

    def __init__(
        self,
        iter_n: int,
        request_path: str | Path,
        checkpoint_path: str | Path,
        run_id: str,
    ) -> None:
        self.iter_n = int(iter_n)
        self.request_path = Path(request_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.run_id = str(run_id)
        super().__init__(
            f"paused at iter {self.iter_n} awaiting agent decision "
            f"(request={self.request_path}, run_id={self.run_id!r})"
        )


# ---------------------------------------------------------------------------
# Co-located loop file paths (plan § 0.6)
# ---------------------------------------------------------------------------


def request_path(artifact_dir: str | Path, iter_n: int) -> Path:
    """``<artifact_dir>/loop/iter_<N>_request.json``."""
    return Path(artifact_dir) / _LOOP_SUBDIR / f"iter_{int(iter_n)}_request.json"


def decision_path(artifact_dir: str | Path, iter_n: int) -> Path:
    """``<artifact_dir>/loop/iter_<N>_decision.json``."""
    return Path(artifact_dir) / _LOOP_SUBDIR / f"iter_{int(iter_n)}_decision.json"


# ---------------------------------------------------------------------------
# Request bundle (Phase 3: diagnose.json-shaped — reuses /gbdt-diagnose logic)
# ---------------------------------------------------------------------------


def build_request_bundle(
    bundle: Any,
    *,
    iter_n: int,
    run_id: str,
    max_iterations: int,
    available_features: list[str],
    artifact_dir: str | Path | None = None,
    cell: dict | None = None,
    val_predictions: Any = None,
) -> dict:
    """Assemble the per-iteration request the agent reads.

    The loop-control **envelope** the agent needs to write a decision is
    unchanged from Phase 2: the iteration index, the budget, and the currently
    available feature names (so a ``prune_features`` decision can be validated
    as a subset). **Phase 3** replaces the ``diagnostics`` payload: instead of a
    raw :meth:`DiagnosticBundle.to_dict` dump it now carries the richer
    ``diagnose.json``-*shaped* dict from
    :func:`gbdt.diagnose_payload.build_diagnose_payload`.

    That payload reuses the ``/gbdt-diagnose`` pure helpers (overfit read,
    prevalence-drift flag, per-day P@k with the ``min(R(d), k)`` denominator,
    per-day variable-K R-precision, prediction-range, tuning-guidance lines) and
    is assembled purely from the in-memory iteration state — it does NOT rebuild
    the in-sample matrix or re-fit PDPs (those matrix-dependent analyses stay in
    the on-disk ``/gbdt-diagnose`` reachable via ``artifact_dir``; plan § 0.5).

    ``artifact_dir`` / ``cell`` / ``val_predictions`` are optional extra context
    threaded through to the payload (the in-loop callback passes ``artifact_dir``
    + ``cell``; ``val_predictions`` stays ``None`` in-loop because the runner
    only carves calibrated predictions over the best checkpoint at
    finalization). When ``bundle`` is not a real ``DiagnosticBundle`` (a raw dict
    passed through), it is embedded verbatim under ``diagnostics``.
    """
    if hasattr(bundle, "importance_native") or hasattr(bundle, "val_brier"):
        diagnostics = build_diagnose_payload(
            bundle,
            artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
            cell=cell,
            val_predictions=val_predictions,
        )
    else:
        # Defensive: a plain dict / already-serialized payload passes through.
        diagnostics = bundle.to_dict() if hasattr(bundle, "to_dict") else bundle
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "iter": int(iter_n),
        "max_iterations": int(max_iterations),
        "available_features": list(available_features),
        # Phase 3: diagnose.json-shaped payload (NaN/Inf-safe, JSON-serializable
        # via build_diagnose_payload's _json_safe coercion).
        "diagnostics": diagnostics,
    }


def write_request(artifact_dir: str | Path, iter_n: int, payload: dict) -> Path:
    """Write the request bundle JSON; create ``loop/`` as needed; return path."""
    path = request_path(artifact_dir, iter_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_decision(artifact_dir: str | Path, iter_n: int) -> dict:
    """Read + parse ``iter_<N>_decision.json``.

    Raises :class:`DecisionError` with a clear message when the file is
    missing or is not parseable JSON (plan § 9).
    """
    path = decision_path(artifact_dir, iter_n)
    if not path.exists():
        raise DecisionError(
            f"decision file not found: {path}. Write the agent's decision "
            f"for iter {iter_n} there, then relaunch with --resume."
        )
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise DecisionError(
            f"decision file {path} is not valid JSON: {e}. "
            f"Fix the JSON and relaunch with --resume."
        ) from e


# ---------------------------------------------------------------------------
# Decision validation (plan § 5 + § 9)
# ---------------------------------------------------------------------------


class DecisionError(ValueError):
    """Raised when a decision file is malformed or violates a constraint.

    The message names the specific problem so the user can fix the decision
    file and relaunch ``--resume`` (plan § 9 recovery: log → user rewrites).
    """


def _search_space_bounds(spec: dict | None) -> dict[str, dict]:
    """Optional per-spec ``search_space`` narrowing of the canonical bounds.

    Plan § 5 lets a spec carry ``backend.fs_hp_loop.search_space`` to narrow
    the canonical model.py ranges (e.g. ``learning_rate: {min, max}``). When
    absent (the common case — no shipped spec carries one) the canonical
    per-backend tables from :func:`~gbdt.model.hp_tables_for` are authoritative.
    """
    if not spec:
        return {}
    loop = ((spec.get("backend") or {}).get("fs_hp_loop") or {})
    ss = loop.get("search_space") or {}
    return ss if isinstance(ss, dict) else {}


def validate_decision(
    decision: dict,
    spec: dict | None,
    known_features: list[str],
    *,
    backend: str = "catboost",
) -> None:
    """Validate an agent decision dict in place (raises on first violation).

    Mirrors plan § 5 (the decision schema) + § 9 (failure modes):

    - ``decision`` is a dict (JSON object), not a list/scalar.
    - ``should_stop`` (if present) is a bool.
    - ``prune_features`` (if present) is a list of strings, each ⊆
      ``known_features``.
    - ``hp_changes`` (if present) is a dict; each key is a real, tunable HP
      **for the requested backend**; no key is a pinned HP (CatBoost's
      ``has_time``/``loss_function``/…, or XGBoost's ``objective``/``tree_method``/
      …); each value is within the canonical bounds (and any per-spec
      ``search_space`` narrowing). ``calibration_method`` is pinned-by-policy and
      rejected for both backends.

    ``backend`` (V1.2 Phase 2 / plan D3 + § 5.2) selects which backend's HP-name
    tables the ``hp_changes`` are validated against — an XGBoost decision
    (``backend="xgboost"``) validates against the ``*_XGB`` tables (``max_depth``,
    ``eta``, ``lambda``, …) and rejects CatBoost-only names (``depth``,
    ``has_time``); vice versa for ``"catboost"``. It defaults to ``"catboost"``
    so every existing call site stays byte-for-byte identical.

    Raises :class:`DecisionError` with a distinct, specific message per
    violation class so the user can fix the file and relaunch ``--resume``.
    """
    tunable_ranges, enum_values, pinned_hps = hp_tables_for(backend)
    if not isinstance(decision, dict):
        raise DecisionError(
            f"decision must be a JSON object, got {type(decision).__name__}"
        )

    # should_stop type
    if "should_stop" in decision and not isinstance(decision["should_stop"], bool):
        raise DecisionError(
            f"decision.should_stop must be a bool, got "
            f"{decision['should_stop']!r}"
        )

    # prune_features type + subset
    prune = decision.get("prune_features", [])
    if prune is None:
        prune = []
    if not isinstance(prune, list):
        raise DecisionError(
            f"decision.prune_features must be a list of feature names, got "
            f"{type(prune).__name__}"
        )
    known = set(known_features)
    for f in prune:
        if not isinstance(f, str):
            raise DecisionError(
                f"decision.prune_features entries must be strings, got "
                f"{f!r} ({type(f).__name__})"
            )
        if f not in known:
            raise DecisionError(
                f"decision.prune_features references unknown feature {f!r}; "
                f"it is not in the current active feature set "
                f"({len(known)} features)."
            )

    # hp_changes type + pinned + bounds
    hp_changes = decision.get("hp_changes", {})
    if hp_changes is None:
        hp_changes = {}
    if not isinstance(hp_changes, dict):
        raise DecisionError(
            f"decision.hp_changes must be a JSON object (HP name -> value), "
            f"got {type(hp_changes).__name__}"
        )

    ss = _search_space_bounds(spec)
    # ``calibration_method`` is pinned-by-policy for the loop (plan § 5):
    # it is a backend-level choice, not a per-iteration FS+HP knob. The rest of
    # the pinned set is the requested backend's never-override list (CatBoost:
    # ``has_time``/loss/eval; XGBoost: ``objective``/``tree_method``/…).
    pinned_names = set(pinned_hps) | {"calibration_method"}
    doc = (
        "docs/gbdt/XGBOOST_HP_REFERENCE.md" if backend == "xgboost"
        else "docs/gbdt/CATBOOST_HP_REFERENCE.md"
    )

    for name, value in hp_changes.items():
        if name in pinned_names:
            raise DecisionError(
                f"decision.hp_changes attempts to change pinned HP {name!r}; "
                f"pinned HPs ({sorted(pinned_names)}) are never overridable "
                f"(see CLAUDE.md gbdt §)."
            )
        is_numeric = name in tunable_ranges
        is_enum = name in enum_values
        if not (is_numeric or is_enum):
            raise DecisionError(
                f"decision.hp_changes references unknown HP {name!r}; not a "
                f"tunable {backend} HP (see {doc})."
            )
        if is_enum:
            allowed = enum_values[name]
            # Per-spec search_space may narrow the allowed enum values.
            ss_vals = (ss.get(name) or {}).get("values")
            if ss_vals is not None:
                allowed = tuple(ss_vals)
            if value not in allowed:
                raise DecisionError(
                    f"decision.hp_changes[{name!r}]={value!r} is not one of "
                    f"the allowed values {list(allowed)}."
                )
        else:  # numeric
            lo, hi = tunable_ranges[name]
            # Per-spec search_space may narrow the numeric bounds.
            ss_b = ss.get(name) or {}
            if "min" in ss_b:
                lo = max(lo, ss_b["min"]) if lo is not None else ss_b["min"]
            if "max" in ss_b and ss_b["max"] is not None:
                hi = min(hi, ss_b["max"]) if hi is not None else ss_b["max"]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise DecisionError(
                    f"decision.hp_changes[{name!r}]={value!r} must be numeric."
                )
            if value < lo or (hi is not None and value > hi):
                bound = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
                raise DecisionError(
                    f"decision.hp_changes[{name!r}]={value} is outside the "
                    f"allowed range {bound}."
                )


def apply_decision(
    decision: dict,
    current_features: list[str],
    current_hp: dict,
) -> tuple[list[str], dict, bool]:
    """Apply a (validated) decision to the current loop state.

    Returns ``(next_features, next_hp, should_stop)``:

    - ``next_features``: ``current_features`` with ``prune_features`` removed
      (order preserved).
    - ``next_hp``: ``current_hp`` merged with ``hp_changes`` (changed knobs
      overwritten; everything else carried forward).
    - ``should_stop``: the decision's flag (default ``False``).

    Validation is the caller's responsibility — call :func:`validate_decision`
    first.
    """
    prune = set(decision.get("prune_features") or [])
    next_features = [f for f in current_features if f not in prune]
    next_hp = dict(current_hp)
    next_hp.update(decision.get("hp_changes") or {})
    should_stop = bool(decision.get("should_stop", False))
    return next_features, next_hp, should_stop


__all__ = [
    "PauseForAgentDecision",
    "DecisionError",
    "REQUEST_SCHEMA_VERSION",
    "DECISION_SCHEMA_VERSION",
    "request_path",
    "decision_path",
    "build_request_bundle",
    "write_request",
    "read_decision",
    "validate_decision",
    "apply_decision",
]
