"""GBDT classifier backends for gbdt.

This module defines the **backend seam** for the gbdt module (V1.2 Phase 1,
``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`` § 4 / § 8):

- :class:`BaseGBDTModel` — a thin abstract base capturing the backend-agnostic
  public surface that ``train.py`` / ``diagnostics.py`` / the loop call on a
  model (``fit``, ``predict_proba``, ``feature_importance``, ``best_iteration``,
  ``evals_result``, the ``hp`` / ``feature_names`` / ``fitted`` properties, and
  ``save`` / ``load``). It deliberately bakes in **no** CatBoost-only
  assumptions (no ``has_time``, no ``.cbm``) so a future ``XGBoostModel`` can
  slot in without redesign.
- :class:`CatBoostModel` — the concrete CatBoost backend. This is the current
  v1 wrapper, behaviour-identical to before the seam was introduced. It:
    - Pins the non-tunable HPs (``has_time``, ``loss_function``, ``eval_metric``,
      ``custom_metric``, ``random_seed``) per V1_PLAN.md Stage 4. ``has_time=True``
      is mandatory for walk-forward correctness; the wrapper raises if a caller
      tries to override it.
    - Validates tunable HPs against ``CATBOOST_HP_REFERENCE.md`` per-parameter
      ranges. Anything out of range is rejected on construction with a clear
      error.
    - Exposes ``fit / predict_proba / feature_importance / save / load``.
- :class:`XGBoostModel` — the concrete XGBoost backend (V1.2 Phase 1). Mirrors
  :class:`CatBoostModel`'s public surface; deterministic-by-construction
  (``tree_method=hist`` + ``n_jobs=8`` + ``device=cpu`` + fixed ``seed``;
  empirically byte-identical across runs at a fixed ``n_jobs`` on a fixed
  machine, 15–90× faster than the prior ``exact`` + ``n_jobs=1`` pin),
  ``binary:logistic`` objective, ``.ubj`` persistence. Reachable via
  :func:`make_model` / unit tests only — not yet wired into the runner/loop
  (the spec validation still rejects ``backend.library != "catboost"``; that
  wiring is V1.2 Phase 5).
- :func:`make_model` — the factory that dispatches on ``backend.library`` and
  returns the right concrete model. ``"catboost"`` routes to
  :class:`CatBoostModel`, ``"xgboost"`` to :class:`XGBoostModel`; other
  backends raise ``NotImplementedError``.

``GBDTModel`` is retained as a public alias of :class:`CatBoostModel` so every
existing import / spec / test keeps working byte-for-byte.

This is the v1 *model* surface — calibration lives separately in
``gbdt.calibration`` (and is already backend-neutral — it operates only on
``(y_val, p_raw)`` arrays), and the FS+HP iteration loop lives in
``gbdt.train`` / ``gbdt.fs_hp_loop``.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
import xgboost as xgb


# ---------------------------------------------------------------------------
# HP validation tables (backend-conditional — V1.2 Phase 2 / plan D3)
# ---------------------------------------------------------------------------
#
# The HP vocabularies for the two backends are genuinely different
# (``depth``↔``max_depth``, ``learning_rate``↔``eta``, ``l2_leaf_reg``↔``lambda``,
# ``rsm``↔``colsample_bytree``, …) and XGBoost has **no ``has_time`` analogue**.
# Per V1.2 plan decision D3 we deliberately reject a lossy canonical
# cross-backend HP vocabulary in favour of **sibling ``*_XGB`` tables** + a
# :func:`hp_tables_for` resolver, so each backend reads its own HP reference doc
# (``CATBOOST_HP_REFERENCE.md`` / ``XGBOOST_HP_REFERENCE.md``) and the agent
# always requests names from the backend it is actually tuning.
#
# Phase 2 shipped the **name-mapping data + the resolver + backend-aware
# validation**; Phase 1 shipped the XGBoost *model adapter* + ``xgboost``
# dependency. Phase 3 (this change) adds the **determinism hard-fail**:
# :func:`_validate_hp_xgb` raises at construction if a caller overrides any of
# the ``PINNED_HPS_XGB`` determinism knobs (``tree_method``/``n_jobs``/``device``)
# with a value that would break the bit-identical finalization retrain (plan D5 /
# § 5.1), mirroring how :func:`_validate_hp` raises on a ``has_time`` override
# for CatBoost.


# ---------------------------------------------------------------------------
# HP validation tables — CatBoost
# ---------------------------------------------------------------------------


# Source: docs/gbdt/CATBOOST_HP_REFERENCE.md per-parameter "Range/values".
# Tuples (min, max) are inclusive bounds; None means "no enforced upper bound."
# These are deliberately tight — the agent can request a value the reference
# documents as practical; outside-of-practical raises.
TUNABLE_HP_RANGES: dict[str, tuple[float, float | None]] = {
    "iterations": (1, 20_000),                    # practical: 100..10000
    "learning_rate": (1e-4, 1.0),                 # practical: 0.01..0.3
    "depth": (1, 16),                              # CPU cap
    "l2_leaf_reg": (0.0, 100.0),                   # practical: 1..30
    "min_data_in_leaf": (1, 10_000),
    "rsm": (0.05, 1.0),
    "bagging_temperature": (0.0, 100.0),
    "subsample": (0.05, 1.0),
    "random_strength": (0.0, 100.0),
    "early_stopping_rounds": (1, 5_000),
    "scale_pos_weight": (1e-4, 1_000.0),
    "border_count": (1, 65_535),
    "leaf_estimation_iterations": (1, 100),
    "max_leaves": (2, 64),
    "od_pval": (0.0, 1.0),
    "od_wait": (1, 5_000),
}

ENUM_HP_VALUES: dict[str, tuple[str, ...]] = {
    "bootstrap_type": ("Bayesian", "Bernoulli", "MVS", "Poisson", "No"),
    "boosting_type": ("Ordered", "Plain"),
    "auto_class_weights": ("None", "Balanced", "SqrtBalanced"),
    "grow_policy": ("SymmetricTree", "Depthwise", "Lossguide"),
    "sampling_frequency": ("PerTree", "PerTreeLevel"),
    "leaf_estimation_method": ("Newton", "Gradient", "Exact"),
    "od_type": ("Iter", "IncToDec"),
}

# Pinned: never overridable from a spec. has_time is the C6 correctness gate.
PINNED_HPS: dict[str, Any] = {
    "has_time": True,
    "loss_function": "Logloss",
    "eval_metric": "BrierScore",
    "custom_metric": ["Logloss", "BrierScore", "AUC"],
}


# ---------------------------------------------------------------------------
# HP validation tables — XGBoost (sibling tables; V1.2 plan D3 / § 5.2)
# ---------------------------------------------------------------------------


# Source: docs/gbdt/XGBOOST_HP_REFERENCE.md per-parameter "Range/values".
# CatBoost↔XGBoost name mapping (project-xgboost-training-essentials § 2):
#   depth↔max_depth, learning_rate↔eta, l2_leaf_reg↔lambda/reg_lambda,
#   (L1) reg_alpha/alpha, min_data_in_leaf↔min_child_weight (loosely),
#   gamma/min_split_loss, subsample, rsm↔colsample_bytree, border_count↔max_bin.
# There is deliberately **no ``has_time`` analogue** — XGBoost has no
# ordered-boosting concept, so the walk-forward C6 guarantee rests on the split
# discipline alone (plan § 5.3). Tuples (min, max) are inclusive bounds; None
# means "no enforced upper bound."
TUNABLE_HP_RANGES_XGB: dict[str, tuple[float, float | None]] = {
    "n_estimators": (1, 20_000),               # ↔ catboost iterations
    "eta": (1e-4, 1.0),                         # ↔ learning_rate
    "max_depth": (1, 16),                       # ↔ depth
    "min_child_weight": (0.0, 1_000.0),         # ↔ min_data_in_leaf (loosely)
    "lambda": (0.0, 100.0),                     # L2 ↔ l2_leaf_reg
    "alpha": (0.0, 100.0),                      # L1 (no clean CatBoost analog)
    "gamma": (0.0, 100.0),                      # min split loss
    "subsample": (0.05, 1.0),
    "colsample_bytree": (0.05, 1.0),            # ↔ rsm
    "colsample_bylevel": (0.05, 1.0),
    "colsample_bynode": (0.05, 1.0),
    "max_leaves": (0, 64),                      # 0 = no limit (lossguide)
    "early_stopping_rounds": (1, 5_000),
    "scale_pos_weight": (1e-4, 1_000.0),
    "max_bin": (2, 65_535),                     # ↔ border_count
}

ENUM_HP_VALUES_XGB: dict[str, tuple[str, ...]] = {
    "grow_policy": ("depthwise", "lossguide"),
    "sampling_method": ("uniform", "gradient_based"),
}

# Suggested *candidate values* the agent should consider per knob, beyond the
# raw range. These are NOT enforced — the validator still range-checks via
# ``TUNABLE_HP_RANGES_XGB`` — but they are the curated grid the loop's tuning
# playbook points at (e.g. ``min_child_weight`` is the L2 lesson from
# ``docs/gbdt/_187_nasdaq100_h25_xgb_manual_tuning.md``: the manual run
# identified ``mcw=10`` as the only distinct XGBoost knob to nudge val Brier
# on that cell, with ``1`` the XGBoost default and ``5`` the middle anchor).
#
# CatBoost intentionally has no parallel registry — its candidate grids are
# already exhaustively described in ``CATBOOST_HP_REFERENCE.md`` § "Suggested
# per-iteration agent prompt"; this XGBoost-only registry surfaces the lessons
# captured in ``_187`` directly to the agent loop.
KNOB_CANDIDATES_XGB: dict[str, tuple[float | int, ...]] = {
    "min_child_weight": (1, 5, 10),  # _187 L2 — mcw=10 was the val-best
}

# Structured (non-scalar) HPs the agent loop is allowed to propose for the
# XGBoost backend. Unlike ``TUNABLE_HP_RANGES_XGB`` / ``ENUM_HP_VALUES_XGB``
# these carry a *shape* (list-of-list-of-strings, dicts, etc.) and need a
# bespoke structural validator in :func:`gbdt.loop_protocol.validate_decision`
# rather than a numeric-range or enum check. They are deliberately
# **XGBoost-only**; the symmetric CatBoost-only ``monotone_constraints`` is
# already rejected by the loop's whitelist and stays so.
#
# ``interaction_constraints``: list of constraint *groups* (each a list of
# feature names) — features may only co-split within a shared group. Default
# (absent / ``None`` / empty list) ⇒ no constraint (current behaviour). The
# XGBoost capability + per-cell motivation are documented in
# ``docs/gbdt/_175_xgboost_interaction_constraints_h25.md`` (Phase-8 demo) and
# ``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`` § 3.2 / § 8.
STRUCTURED_HP_KEYS_XGB: frozenset[str] = frozenset({"interaction_constraints"})

# Pinned: never overridable from a spec. For XGBoost the load-bearing pins are
# the determinism knobs (``tree_method``/``n_jobs``/``device``) that replace
# ``has_time``'s "never-override" role (plan § 5.1/§ 5.3). The construction-time
# **hard-fail** enforcement (:func:`_validate_hp_xgb`, V1.2 Phase 3 / plan D5)
# raises if any of these is set to a *different* value — passing the same pinned
# value is a no-op. ``seed`` is set from ``random_seed`` at construction, like
# CatBoost's ``random_seed`` — so it is not listed as a fixed-value pin here (a
# caller may pick the seed; it just stays fixed within a run).
PINNED_HPS_XGB: dict[str, Any] = {
    "objective": "binary:logistic",            # ↔ Logloss
    "eval_metric": "logloss",                   # Brier computed in the bundle
    # ``tree_method="hist" + n_jobs=8`` is empirically byte-identical across runs
    # on this machine (booster ``save_raw()`` SHA-256 match across 2 fits × 4
    # configs), and 15–90× faster than the prior ``exact + n_jobs=1`` pin. The
    # determinism contract is now "**at fixed ``n_jobs`` on a fixed machine**",
    # not "at any n_jobs on any machine" — cross-machine or n_jobs changes would
    # produce a different booster blob and require an A6 baseline refresh.
    "tree_method": "hist",                      # speed; byte-identical at fixed n_jobs on a fixed machine
    "n_jobs": 8,                                # speed; pin to a fixed n_jobs for byte-identical multi-thread reductions (8 = saturate the 8-core box)
    "device": "cpu",                            # determinism (GPU reductions are non-deterministic)
}

# The subset of ``PINNED_HPS_XGB`` whose override breaks bit-reproducibility of
# the finalization retrain (plan § 5.1). These get the **determinism-specific**
# hard-fail message in :func:`_validate_hp_xgb`; ``objective``/``eval_metric`` are
# pinned-by-policy (semantics, not determinism) and reuse the generic message.
_DETERMINISM_PINS_XGB: tuple[str, ...] = ("tree_method", "n_jobs", "device")


# Backend → (tunable ranges, enum values, pinned) table triple.
_HP_TABLES: dict[str, tuple[dict, dict, dict]] = {
    "catboost": (TUNABLE_HP_RANGES, ENUM_HP_VALUES, PINNED_HPS),
    "xgboost": (TUNABLE_HP_RANGES_XGB, ENUM_HP_VALUES_XGB, PINNED_HPS_XGB),
}

# Per-backend HP reference doc, surfaced in validation error messages so the
# agent is pointed at the right doc when it requests an out-of-vocab / out-of-
# range HP.
_HP_REFERENCE_DOC: dict[str, str] = {
    "catboost": "docs/gbdt/CATBOOST_HP_REFERENCE.md",
    "xgboost": "docs/gbdt/XGBOOST_HP_REFERENCE.md",
}


# Backend → persisted-model filename. The single source of truth the runner and
# the ``/gbdt-diagnose`` loader both consult so they agree on the on-disk name
# (V1.2 plan § 4.4 / § 6.4): CatBoost writes its ``.cbm`` binary, XGBoost its
# native ``.ubj`` (UBJSON). ``CatBoostModel.save`` / ``XGBoostModel.save`` write
# the format implied here; ``make_model(backend).load`` reads it back.
_MODEL_FILENAME: dict[str, str] = {
    "catboost": "model.cbm",
    "xgboost": "model.ubj",
}


def model_filename(backend: str | Any) -> str:
    """Return the canonical persisted-model filename for ``backend``.

    The single dispatch point for the backend-determined artifact model name
    (V1.2 plan § 4.4): ``"catboost"`` → ``"model.cbm"``, ``"xgboost"`` →
    ``"model.ubj"``. ``backend`` may be a plain string or a spec-like object
    exposing ``backend.library`` (resolved via :func:`_resolve_backend`), so the
    runner can pass the parsed spec straight through. Any other backend raises
    :class:`ValueError`.
    """
    library = _resolve_backend(backend)
    try:
        return _MODEL_FILENAME[library]
    except KeyError:
        raise ValueError(
            f"unknown backend {library!r}; model filenames exist for "
            f"{sorted(_MODEL_FILENAME)} only."
        ) from None


def hp_tables_for(backend: str) -> tuple[dict, dict, dict]:
    """Return ``(tunable_ranges, enum_values, pinned)`` for ``backend``.

    The single dispatch point for the backend-conditional HP-name tables
    (V1.2 plan D3). ``"catboost"`` returns the existing CatBoost tables
    (unchanged behaviour); ``"xgboost"`` returns the ``*_XGB`` siblings. Any
    other backend raises :class:`ValueError`.

    The returned dicts are the module-level singletons — callers must not mutate
    them (they are read-only validation references).
    """
    try:
        return _HP_TABLES[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; HP tables exist for "
            f"{sorted(_HP_TABLES)} only."
        ) from None


def _validate_hp_xgb(hp: dict) -> dict:
    """XGBoost HP validation with the **determinism hard-fail** (V1.2 Phase 3 /
    plan D5 / § 5.1).

    The load-bearing reason this exists: the FS+HP loop's finalization step
    *retrains* a prior ``(features, hp)`` config (``train.py::_fit_one``) and
    assumes it reproduces the in-loop fit **bit-identically** — the checkpoint
    stores no model blob. XGBoost reproduces bit-identically when those
    determinism knobs are held FIXED across the in-loop fit and the
    finalization retrain — the current pins are ``tree_method="hist"`` +
    ``n_jobs=8`` + ``device="cpu"`` (empirically byte-identical on this
    machine; ``hist`` + ``n_jobs=8`` is 15–90× faster than the prior
    ``exact + n_jobs=1`` pin while still hashing the same across runs). The
    cross-machine / cross-n_jobs case is **not** guaranteed bit-identical —
    multi-thread float-reduction order and GPU reductions are otherwise
    run-to-run non-deterministic — so those knobs are pinned in
    ``PINNED_HPS_XGB`` and **cannot be overridden** at construction, exactly
    as CatBoost pins ``has_time`` (``model.py`` ``_validate_hp`` raises on a
    ``has_time`` override). A caller passing the *same* pinned value is a
    no-op; passing a *different* value raises here, at construction, never
    at predict-time.

    Returns a copy of ``hp`` with the pins applied + the tunable/enum tables
    range-checked. Raises :class:`ValueError` on the first violation.
    """
    tunable, enum_values, pinned = hp_tables_for("xgboost")
    out = dict(hp)

    # Determinism knobs: a DIFFERENT value breaks the bit-identical retrain
    # contract — hard-fail with a determinism-specific message (plan § 5.1).
    for k in _DETERMINISM_PINS_XGB:
        if k in out and out[k] != pinned[k]:
            raise ValueError(
                f"XGBoost determinism pin {k!r} is fixed to {pinned[k]!r} and "
                f"cannot be overridden (got {out[k]!r}). This pin is load-bearing: "
                f"the FS+HP loop's finalization step retrains the selected "
                f"(features, hp) config and requires a bit-identical reproduction "
                f"of the in-loop fit. Only tree_method={pinned['tree_method']!r}, "
                f"n_jobs={pinned['n_jobs']!r}, device={pinned['device']!r} guarantee "
                f"that on this machine — a different override (e.g. device='cuda', "
                f"or a different n_jobs value than the pin) would silently make "
                f"the artifact's predictions disagree with the val-Brier the "
                f"checkpoint was selected on. See V1.2 plan D5 / § 5.1 and "
                f"project-xgboost-training-essentials § 1."
            )

    # Remaining pins (objective / eval_metric): policy pins, generic message.
    for k, v in pinned.items():
        if k in _DETERMINISM_PINS_XGB:
            out[k] = v
            continue
        if k in out and out[k] != v:
            raise ValueError(
                f"{k!r} is pinned to {v!r} in v1 and cannot be overridden "
                f"(got {out[k]!r}); see V1_PLAN.md Stage 4."
            )
        out[k] = v

    _range_enum_check(out, tunable, enum_values, _HP_REFERENCE_DOC["xgboost"])
    return out


def _validate_hp(hp: dict, backend: str = "catboost") -> dict:
    """Return a copy of ``hp`` with pinned HPs applied and tunable HPs
    range-checked against ``backend``'s tables. Raise ``ValueError`` on the
    first violation.

    ``backend`` defaults to ``"catboost"`` so every existing caller (and the
    ``GBDTModel`` alias) behaves byte-for-byte as before the V1.2 seam. The
    ``"xgboost"`` path dispatches to :func:`_validate_hp_xgb`, which adds the
    determinism-specific hard-fail message (V1.2 Phase 3 / plan D5).
    """
    if backend == "xgboost":
        return _validate_hp_xgb(hp)

    tunable, enum_values, pinned = hp_tables_for(backend)
    doc = _HP_REFERENCE_DOC.get(backend, "the backend HP reference")
    out = dict(hp)

    # Reject overrides on pinned HPs (except matching value, which is a no-op).
    for k, v in pinned.items():
        if k in out and out[k] != v:
            raise ValueError(
                f"{k!r} is pinned to {v!r} in v1 and cannot be overridden "
                f"(got {out[k]!r}); see V1_PLAN.md Stage 4."
            )
        out[k] = v

    _range_enum_check(out, tunable, enum_values, doc)
    return out


def _interaction_constraints_to_indices(
    groups: Any, feat_names: list[str] | None
) -> list[list[int]]:
    """Translate an ``interaction_constraints`` value from list-of-list-of-strings
    (the agent-loop ergonomic form) to list-of-list-of-ints (the form XGBoost's
    booster resolves against a name-less matrix).

    Already-integer groups pass through; mixed groups (some names, some ints)
    raise. An empty outer list / ``None`` is returned untouched as an empty
    list (XGBoost reads "no constraint" semantics from that). Unknown names
    raise :class:`ValueError` with the offending name embedded so the model.py
    caller can decorate the message.
    """
    if groups is None:
        return []
    if not isinstance(groups, (list, tuple)):
        raise ValueError(
            f"interaction_constraints must be a list of groups, got "
            f"{type(groups).__name__}"
        )
    if len(groups) == 0:
        return []
    name_to_idx: dict[str, int] = {}
    if feat_names is not None:
        name_to_idx = {name: i for i, name in enumerate(feat_names)}
    out: list[list[int]] = []
    for g in groups:
        if not isinstance(g, (list, tuple)):
            raise ValueError(
                f"each interaction_constraints group must be a list, got "
                f"{type(g).__name__}"
            )
        idx_group: list[int] = []
        for item in g:
            if isinstance(item, bool):  # bool is a subclass of int
                raise ValueError(
                    f"interaction_constraints group entries must be feature "
                    f"names or indices, got bool {item!r}"
                )
            if isinstance(item, int):
                idx_group.append(int(item))
            elif isinstance(item, str):
                if item not in name_to_idx:
                    raise ValueError(item)
                idx_group.append(name_to_idx[item])
            else:
                raise ValueError(
                    f"interaction_constraints group entries must be strings "
                    f"or ints, got {type(item).__name__} ({item!r})"
                )
        out.append(idx_group)
    return out


def _range_enum_check(
    out: dict, tunable: dict, enum_values: dict, doc: str
) -> None:
    """Range-check tunable HPs and validate enum HPs in ``out`` against the
    given tables. Raises :class:`ValueError` on the first violation. Mutates
    nothing — ``out`` is read only.

    Shared by the CatBoost (:func:`_validate_hp`) and XGBoost
    (:func:`_validate_hp_xgb`) validators so the range/enum message text is
    identical across backends.
    """
    # Numeric range checks
    for k, (lo, hi) in tunable.items():
        if k not in out or out[k] is None:
            continue
        v = out[k]
        if hi is not None:
            if not (lo <= v <= hi):
                raise ValueError(
                    f"HP {k}={v} is outside the documented range [{lo}, {hi}] "
                    f"(see {doc})."
                )
        else:
            if v < lo:
                raise ValueError(
                    f"HP {k}={v} is below the documented minimum {lo}."
                )

    # Enum checks
    for k, allowed in enum_values.items():
        if k not in out or out[k] is None:
            continue
        if out[k] not in allowed:
            raise ValueError(
                f"HP {k}={out[k]!r} is not one of {allowed}; "
                f"see {doc}."
            )


# ---------------------------------------------------------------------------
# Backend-agnostic interface (the seam)
# ---------------------------------------------------------------------------


class BaseGBDTModel(abc.ABC):
    """Backend-agnostic GBDT classifier interface.

    Captures *exactly* the public surface that the rest of the gbdt codebase
    (``train.py``, ``diagnostics.py``, the FS+HP loop, ``/gbdt-diagnose``)
    calls on a fitted-or-fitting model. A concrete backend (CatBoost today;
    XGBoost in a later V1.2 phase) implements these members.

    Deliberately backend-neutral: no CatBoost-only concept (``has_time``,
    the ``.cbm`` persistence format, the two-column ``predict_proba`` shape,
    ordered boosting) appears here. Those live in the concrete class so a
    second backend can slot in without redesigning this base. ``predict_proba``
    is contracted to return a **1-D** array of ``P(positive)``.
    """

    # ---- accessors ------------------------------------------------------

    @property
    @abc.abstractmethod
    def hp(self) -> dict:
        """The (validated, pin-applied) HP dict this model was built with."""

    @property
    @abc.abstractmethod
    def feature_names(self) -> list[str] | None:
        """Ordered feature names, or ``None`` if not yet known."""

    @property
    @abc.abstractmethod
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""

    @property
    @abc.abstractmethod
    def best_iteration(self) -> int | None:
        """Best boosting iteration after early stopping, or ``None``."""

    @property
    @abc.abstractmethod
    def evals_result(self) -> dict | None:
        """Per-split metric history (learning curves), or ``None``."""

    # ---- core ops -------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        *,
        early_stopping_rounds: int | None = None,
        train_weight: np.ndarray | pd.Series | None = None,
        val_weight: np.ndarray | pd.Series | None = None,
    ) -> "BaseGBDTModel":
        """Fit the model, optionally early-stopping against ``(X_val, y_val)``.

        ``train_weight`` / ``val_weight`` (optional) are per-row sample weights
        (typically the LdP §4.4 uniqueness weights). Returns ``self``.
        """

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ``P(positive)`` as a 1-D array."""

    @abc.abstractmethod
    def feature_importance(
        self,
        kind: str = "native",
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> pd.Series:
        """Return a Series of feature → importance for the given ``kind``."""

    # ---- persistence ----------------------------------------------------

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to ``path`` in the backend's format."""

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        path: str | Path,
        hp: dict | None = None,
        feature_names: list[str] | None = None,
    ) -> "BaseGBDTModel":
        """Load a fitted model previously written by :meth:`save`."""


# ---------------------------------------------------------------------------
# CatBoost backend
# ---------------------------------------------------------------------------


class CatBoostModel(BaseGBDTModel):
    """Thin CatBoost wrapper. ``has_time=True`` is mandatory.

    Build via ``CatBoostModel(hp_dict, feature_names=...)`` (or, equivalently,
    the back-compat ``GBDTModel`` alias / ``make_model("catboost", ...)``); fit
    with ``fit(X_train, y_train, X_val, y_val)``; score with ``predict_proba(X)``
    returning the marginal probability of the positive class.
    """

    def __init__(
        self,
        hp: dict,
        *,
        feature_names: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        hp = _validate_hp(hp)
        hp.setdefault("random_seed", random_seed)
        hp.setdefault("verbose", False)             # quiet by default
        hp.setdefault("allow_writing_files", False)
        self._hp = hp
        self._feature_names = feature_names
        self._model: CatBoostClassifier | None = None
        self._fitted = False

    # ---- accessors ------------------------------------------------------

    @property
    def hp(self) -> dict:
        return dict(self._hp)

    @property
    def feature_names(self) -> list[str] | None:
        return list(self._feature_names) if self._feature_names else None

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def best_iteration(self) -> int | None:
        if not self._fitted:
            return None
        return self._model.get_best_iteration()

    @property
    def evals_result(self) -> dict | None:
        if not self._fitted:
            return None
        return self._model.get_evals_result()

    # ---- core ops -------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        *,
        early_stopping_rounds: int | None = None,
        train_weight: np.ndarray | pd.Series | None = None,
        val_weight: np.ndarray | pd.Series | None = None,
    ) -> "CatBoostModel":
        """Fit CatBoost with optional early stopping against ``(X_val, y_val)``.

        ``train_weight`` / ``val_weight`` (optional) are per-row sample
        weights — typically the LdP §4.4 uniqueness weights produced by
        :func:`gbdt.uniqueness.compute_uniqueness_weights`. When supplied,
        CatBoost weights the gradient by ``w_i`` on training and the val
        loss reported in ``evals_result`` by ``w_i`` on val.
        """
        feat_names = self._feature_names
        if feat_names is None and isinstance(X_train, pd.DataFrame):
            feat_names = list(X_train.columns)
            self._feature_names = feat_names

        train_pool = Pool(
            data=_to_2d(X_train),
            label=np.asarray(y_train).ravel(),
            feature_names=feat_names,
            weight=(
                np.asarray(train_weight, dtype=float).ravel()
                if train_weight is not None else None
            ),
        )
        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(
                data=_to_2d(X_val),
                label=np.asarray(y_val).ravel(),
                feature_names=feat_names,
                weight=(
                    np.asarray(val_weight, dtype=float).ravel()
                    if val_weight is not None else None
                ),
            )

        model_hp = dict(self._hp)
        if early_stopping_rounds is None:
            early_stopping_rounds = model_hp.pop("early_stopping_rounds", None)
        else:
            model_hp.pop("early_stopping_rounds", None)
        model = CatBoostClassifier(**model_hp)
        model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=eval_pool is not None,
            verbose=False,
        )
        self._model = model
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ``P(positive)`` as a 1-D array."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        proba = self._model.predict_proba(_to_2d(X))
        # CatBoostClassifier returns (n, 2) for binary; positive class is col 1.
        return np.asarray(proba)[:, 1]

    def feature_importance(
        self,
        kind: str = "native",
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> pd.Series:
        """Return a Series of feature → importance.

        - ``kind="native"`` (default): CatBoost's native ``PredictionValuesChange``.
        - ``kind="permutation"``: drop-column-style importance against ``(X_val, y_val)``.
          Returns absolute Brier-score increase relative to baseline; non-negative.
        """
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        feat_names = self._feature_names or list(range(self._model.feature_count_))

        if kind == "native":
            imp = self._model.get_feature_importance()
            return pd.Series(imp, index=feat_names, name="importance_native")
        if kind == "permutation":
            if X_val is None or y_val is None:
                raise ValueError("permutation importance needs X_val + y_val")
            from sklearn.metrics import brier_score_loss
            X_arr = _to_2d(X_val).copy()
            y_arr = np.asarray(y_val).ravel()
            base_pred = self.predict_proba(X_arr)
            base_brier = brier_score_loss(y_arr, base_pred)
            rng = np.random.default_rng(self._hp.get("random_seed", 42))
            scores = []
            for j in range(X_arr.shape[1]):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_pred = self.predict_proba(X_perm)
                perm_brier = brier_score_loss(y_arr, perm_pred)
                scores.append(max(0.0, perm_brier - base_brier))
            return pd.Series(scores, index=feat_names, name="importance_permutation")
        raise ValueError(f"unknown importance kind {kind!r}")

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        self._model.save_model(str(path), format="cbm")

    @classmethod
    def load(cls, path: str | Path, hp: dict | None = None,
             feature_names: list[str] | None = None) -> "CatBoostModel":
        m = CatBoostClassifier()
        m.load_model(str(path))
        obj = cls(hp or PINNED_HPS, feature_names=feature_names)
        obj._model = m
        obj._fitted = True
        if feature_names is None:
            obj._feature_names = list(m.feature_names_) if m.feature_names_ else None
        return obj


# Back-compat public alias. ``GBDTModel`` was the CatBoost wrapper's name before
# the V1.2 backend seam; keep it pointing at the concrete CatBoost impl so every
# existing import / spec / test keeps working byte-for-byte.
GBDTModel = CatBoostModel


# ---------------------------------------------------------------------------
# XGBoost backend (V1.2 Phase 1)
# ---------------------------------------------------------------------------


class XGBoostModel(BaseGBDTModel):
    """Thin XGBoost wrapper implementing the :class:`BaseGBDTModel` protocol.

    The second concrete backend behind the V1.2 seam
    (``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`` § 4 / § 8 Phase 1).
    It mirrors :class:`CatBoostModel`'s method signatures + return shapes exactly:
    ``predict_proba`` returns a **1-D** array of ``P(positive)``,
    ``feature_importance`` returns a feature → importance ``Series``,
    ``best_iteration`` / ``evals_result`` expose early-stopping + learning-curve
    state, and ``save`` / ``load`` round-trip the fitted model (XGBoost native
    ``.ubj`` binary, ``project-xgboost-training-essentials`` § 4).

    **Deterministic by construction** (the load-bearing § 5.1 / § 1 contract):
    ``PINNED_HPS_XGB`` supplies the determinism knobs (``tree_method="hist"``,
    ``n_jobs=8``, ``device="cpu"``) and a fixed ``seed``/``random_state`` is
    applied at construction, so a refit on the same ``(features, hp, seed)`` +
    row order — at the *same* ``n_jobs`` on the *same* machine — reproduces the
    prior model bit-identically. (``hist + n_jobs=8`` was empirically verified
    byte-identical across 2 fits × 4 configs on this 8-core box, and is 15–90×
    faster than the prior ``exact + n_jobs=1`` pin; the cross-machine /
    cross-n_jobs guarantee is intentionally NOT made — those would require an
    A6 baseline refresh.) The XGBoost objective is pinned to
    ``binary:logistic`` (raw margin → sigmoid → probability) and
    ``eval_metric="logloss"`` drives early stopping (Brier is computed in the
    diagnostic bundle, not as a native eval metric — V1.2 plan Q1). Missing values
    flow through XGBoost's sparsity-aware split finding — no imputation.

    Calibration is unchanged — :mod:`gbdt.calibration` operates purely on the
    ``(y_val, p_raw)`` arrays ``predict_proba`` produces (§ 4.3).

    .. note::

        Construction runs :func:`_validate_hp_xgb` (V1.2 Phase 3 / plan D5),
        which **hard-fails** if a caller overrides any of the ``PINNED_HPS_XGB``
        determinism knobs (``tree_method``/``n_jobs``/``device``) with a value
        that would break the bit-identical finalization retrain. Passing the
        same pinned value is a no-op; a different value (e.g. ``tree_method=
        "exact"``, ``n_jobs=4``, or ``device="cuda"``) raises at construction.
    """

    def __init__(
        self,
        hp: dict,
        *,
        feature_names: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        hp = _validate_hp(hp, backend="xgboost")
        # Deterministic-by-construction: the seed mirrors CatBoost's random_seed.
        # XGBoost's sklearn wrapper reads ``random_state``; accept either spelling
        # from a spec and keep them consistent.
        seed = hp.pop("seed", None)
        if seed is None:
            seed = hp.get("random_state", random_seed)
        hp.setdefault("random_state", seed)
        hp["seed"] = seed
        hp.setdefault("verbosity", 0)               # quiet by default
        self._hp = hp
        self._feature_names = feature_names
        self._model: xgb.XGBClassifier | None = None
        self._fitted = False

    # ---- accessors ------------------------------------------------------

    @property
    def hp(self) -> dict:
        return dict(self._hp)

    @property
    def feature_names(self) -> list[str] | None:
        return list(self._feature_names) if self._feature_names else None

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def best_iteration(self) -> int | None:
        if not self._fitted:
            return None
        bi = getattr(self._model, "best_iteration", None)
        return int(bi) if bi is not None else None

    @property
    def evals_result(self) -> dict | None:
        """Per-split metric history normalized to CatBoost's nested shape.

        XGBoost nests as ``{"validation_0": {"logloss": [...]}}``. The
        ``diagnostics.py`` learning-curve builder (lines 298–304) iterates
        ``evals.items()`` → ``metrics.items()`` and joins as ``f"{split}_{metric}"``,
        so the nested ``{split: {metric: [...]}}`` shape is exactly what it
        expects. We only relabel ``validation_0`` → ``validation`` so the
        learning-curve keys read the same as CatBoost's (``validation_logloss``).
        """
        if not self._fitted:
            return None
        try:
            raw = self._model.evals_result()
        except xgb.core.XGBoostError:
            # No eval_set was used during training (so no learning curve).
            return {}
        if not raw:
            return {}
        out: dict[str, dict] = {}
        for split, metrics in raw.items():
            label = "validation" if split == "validation_0" else split
            out[label] = {m: list(vals) for m, vals in metrics.items()}
        return out

    # ---- core ops -------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        *,
        early_stopping_rounds: int | None = None,
        train_weight: np.ndarray | pd.Series | None = None,
        val_weight: np.ndarray | pd.Series | None = None,
    ) -> "XGBoostModel":
        """Fit XGBoost with optional early stopping against ``(X_val, y_val)``.

        Mirrors :meth:`CatBoostModel.fit`: ``train_weight`` / ``val_weight``
        (optional) are per-row sample weights (LdP §4.4 uniqueness weights) —
        XGBoost weights the gradient by ``w_i`` on training and the val loss
        reported in ``evals_result`` by ``w_i`` on val.
        """
        feat_names = self._feature_names
        if feat_names is None and isinstance(X_train, pd.DataFrame):
            feat_names = list(X_train.columns)
            self._feature_names = feat_names

        # XGBoost's DMatrix construction rejects ``±inf`` (unlike CatBoost) —
        # map it to ``NaN`` so it flows through XGBoost's native missing-value
        # handling. Applied identically on the predict path (``predict_proba``)
        # so train/inference see the same sanitization. See ``_sanitize_nonfinite``.
        X_tr = _sanitize_nonfinite(_to_2d(X_train))
        y_tr = np.asarray(y_train).ravel()
        w_tr = (
            np.asarray(train_weight, dtype=float).ravel()
            if train_weight is not None else None
        )

        eval_set = None
        eval_weight = None
        if X_val is not None and y_val is not None:
            eval_set = [(_sanitize_nonfinite(_to_2d(X_val)),
                         np.asarray(y_val).ravel())]
            if val_weight is not None:
                eval_weight = [np.asarray(val_weight, dtype=float).ravel()]

        model_hp = dict(self._hp)
        if early_stopping_rounds is None:
            early_stopping_rounds = model_hp.pop("early_stopping_rounds", None)
        else:
            model_hp.pop("early_stopping_rounds", None)
        # Early stopping is only meaningful with an eval set; XGBoost raises if
        # ``early_stopping_rounds`` is set without one.
        if eval_set is None:
            early_stopping_rounds = None

        # ``interaction_constraints`` agent-loop ergonomic form is a list of
        # lists of *feature names* — but ``_to_2d`` strips names off the matrix
        # we feed XGBoost, so the booster's name-based resolution would fail.
        # Translate name-groups → integer-index groups against
        # ``self._feature_names`` here (the JSON-string-of-ints form already
        # produced by ``gbdt.interactions._interaction_constraints_from_forbidden``
        # is left untouched — that path stays exactly as ``_175`` verified).
        ic = model_hp.get("interaction_constraints")
        if ic is not None and not isinstance(ic, str):
            try:
                model_hp["interaction_constraints"] = (
                    _interaction_constraints_to_indices(ic, feat_names)
                )
            except ValueError as exc:
                raise ValueError(
                    f"interaction_constraints contains a feature name not in "
                    f"the model's feature set: {exc}"
                ) from exc

        model = xgb.XGBClassifier(
            early_stopping_rounds=early_stopping_rounds,
            **model_hp,
        )
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weight,
            verbose=False,
        )
        self._model = model
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ``P(positive)`` as a 1-D array."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        # Mirror ``fit``: sanitize ``±inf`` → ``NaN`` before the DMatrix is built
        # on the predict path, so inference tolerates the same matrices as fit.
        proba = self._model.predict_proba(_sanitize_nonfinite(_to_2d(X)))
        # XGBClassifier returns (n, 2) for binary; positive class is col 1.
        return np.asarray(proba)[:, 1]

    def feature_importance(
        self,
        kind: str = "native",
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> pd.Series:
        """Return a Series of feature → importance.

        - ``kind="native"`` (default): XGBoost gain importance (the booster's
          ``feature_importances_`` with ``importance_type="gain"``), aligned to
          the model's feature order. Mirrors CatBoost's native-importance Series.
        - ``kind="permutation"``: drop-column-style importance against
          ``(X_val, y_val)`` — backend-neutral Brier perturbation, identical to
          :meth:`CatBoostModel.feature_importance` (``kind="permutation"``).
        """
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        n_feat = self._model.n_features_in_
        feat_names = self._feature_names or list(range(n_feat))

        if kind == "native":
            booster = self._model.get_booster()
            score = booster.get_score(importance_type="gain")
            # get_score keys are the booster feature names; align to our order.
            booster_names = list(booster.feature_names or [])
            imp = np.zeros(n_feat, dtype=float)
            for j in range(n_feat):
                bname = booster_names[j] if j < len(booster_names) else f"f{j}"
                imp[j] = score.get(bname, 0.0)
            return pd.Series(imp, index=feat_names, name="importance_native")
        if kind == "permutation":
            if X_val is None or y_val is None:
                raise ValueError("permutation importance needs X_val + y_val")
            from sklearn.metrics import brier_score_loss
            X_arr = _to_2d(X_val).copy()
            y_arr = np.asarray(y_val).ravel()
            base_pred = self.predict_proba(X_arr)
            base_brier = brier_score_loss(y_arr, base_pred)
            rng = np.random.default_rng(self._hp.get("seed", 42))
            scores = []
            for j in range(X_arr.shape[1]):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_pred = self.predict_proba(X_perm)
                perm_brier = brier_score_loss(y_arr, perm_pred)
                scores.append(max(0.0, perm_brier - base_brier))
            return pd.Series(scores, index=feat_names, name="importance_permutation")
        raise ValueError(f"unknown importance kind {kind!r}")

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted model as XGBoost native UBJSON (``.ubj``)."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path, hp: dict | None = None,
             feature_names: list[str] | None = None) -> "XGBoostModel":
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        obj = cls(hp or PINNED_HPS_XGB, feature_names=feature_names)
        obj._model = m
        obj._fitted = True
        if feature_names is None:
            booster_names = getattr(m.get_booster(), "feature_names", None)
            obj._feature_names = list(booster_names) if booster_names else None
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_backend(backend: Any) -> str:
    """Coerce a ``backend`` argument to its library string.

    Accepts a plain string (``"catboost"``), or a spec-like object exposing
    ``.library`` directly or nested under ``.backend.library`` (so the runner
    can thread the parsed spec through without unwrapping). Mappings with a
    ``"backend"``/``"library"`` key are also accepted.
    """
    if isinstance(backend, str):
        return backend
    # spec.backend.library or spec.library
    lib = getattr(getattr(backend, "backend", None), "library", None)
    if lib is None:
        lib = getattr(backend, "library", None)
    if lib is None and isinstance(backend, dict):
        inner = backend.get("backend", backend)
        if isinstance(inner, dict):
            lib = inner.get("library")
        else:
            lib = getattr(inner, "library", None)
    if not isinstance(lib, str):
        raise TypeError(
            f"cannot resolve backend library from {backend!r}; pass a string "
            f"(e.g. 'catboost') or a spec exposing backend.library."
        )
    return lib


def make_model(
    backend: str | Any,
    hp: dict,
    *,
    feature_names: list[str] | None = None,
    random_seed: int = 42,
) -> BaseGBDTModel:
    """Construct the GBDT model for the requested ``backend``.

    ``backend`` is the ``backend.library`` value from the experiment spec
    (a string like ``"catboost"``), or a spec-like object exposing it. Dispatch:

    - ``"catboost"`` → :class:`CatBoostModel` (behaviour-identical to the v1
      ``GBDTModel`` — same construction, same ``has_time`` pin, same
      determinism, same save/load).
    - ``"xgboost"`` → :class:`XGBoostModel` (V1.2 — deterministic-by-construction
      pins with a Phase-3 hard-fail on any determinism-knob override,
      ``binary:logistic`` objective, ``.ubj`` save/load).
    - anything else → ``NotImplementedError``.

    .. note::

        Constructing an :class:`XGBoostModel` via this factory is reachable in
        v1 only through unit tests / direct calls — the runner's spec validation
        (``gbdt.__main__``) still rejects ``backend.library != "catboost"``, so
        no end-to-end XGBoost path is wired into the FS+HP loop yet. That wiring
        is V1.2 Phase 5.
    """
    library = _resolve_backend(backend)
    if library == "catboost":
        return CatBoostModel(
            hp, feature_names=feature_names, random_seed=random_seed
        )
    if library == "xgboost":
        return XGBoostModel(
            hp, feature_names=feature_names, random_seed=random_seed
        )
    raise NotImplementedError(
        f"unknown backend.library={library!r}; "
        f"only 'catboost' and 'xgboost' are wired."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_2d(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _sanitize_nonfinite(arr: np.ndarray) -> np.ndarray:
    """Map ``±inf`` to ``NaN`` so a stricter backend treats them as missing.

    XGBoost's ``XGDMatrixCreateFromDense`` **rejects** ``inf`` outright
    (``"Input data contains 'inf' or a value too large, while 'missing' is not
    set to 'inf'"``), whereas CatBoost tolerates ``inf`` / huge values and
    routes ``NaN`` to a dedicated missing-value bucket.

    **As of PR #182, the gbdt feature pipeline is contractually inf-free at
    source** — every ratio/division family in ``gbdt.features`` guards its
    denominator with ``.replace(0, np.nan)`` (zero-denom → NaN, the correct
    "undefined / missing" sentinel), and ``build_feature_matrix`` asserts no
    ``±inf`` survives the build. This function is therefore retained as
    **defense-in-depth**: it ensures the backend never crashes if a future
    feature family forgets the guard (so the model run degrades to "rows with
    inf get routed down the missing branch" rather than a hard ``DMatrix``
    crash). Pre-#182 it was the primary fix for the V1.2 sp500 crash
    (#180) — now it's the safety net behind the source-side guards.

    Replacing ``±inf`` with ``NaN`` is semantically right: ``inf`` means "the
    denominator was (near-)zero, so this ratio is undefined", which is exactly
    the *missing* semantics XGBoost's sparsity-aware split finding already
    handles natively (and what CatBoost's NaN bucket captures). We do **not**
    clip to a finite rail — that would invent a real, learnable value at the
    boundary and let the tree split on an artifact instead of routing the row
    down the missing branch. Returns a float copy with ``±inf`` → ``NaN``
    (finite values and existing ``NaN``s are untouched). Not done per-feature —
    the backend is made robust to non-finite values generally.
    """
    out = np.asarray(arr, dtype=float)
    nonfinite = ~np.isfinite(out)
    # ``~isfinite`` is True for both ``NaN`` and ``±inf``; only the inf subset
    # needs rewriting (NaN is already the missing sentinel). Operate on a copy
    # so the caller's array is never mutated in place.
    if nonfinite.any():
        out = out.copy()
        out[np.isinf(out)] = np.nan
    return out


def count_nonfinite(X: pd.DataFrame | np.ndarray) -> dict[str, int]:
    """Audit a feature matrix for ``±inf`` values (fail-fast helper).

    Returns ``{column_name: inf_count}`` for every column carrying at least one
    ``±inf``, sorted descending by count. Empty dict when the matrix is clean.
    ``NaN`` is **not** counted — it is the legitimate missing sentinel both
    backends accept; only ``±inf`` (which crashes XGBoost's DMatrix
    construction) is flagged. Callable right after the feature build so a
    non-finite matrix surfaces in seconds, not after a multi-hour train.
    """
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        arr = X.values
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"f{j}" for j in range(arr.shape[1])]
    arr = np.asarray(arr, dtype=float)
    per_col = np.isinf(arr).sum(axis=0)
    offenders = {
        cols[j]: int(per_col[j]) for j in range(len(cols)) if per_col[j] > 0
    }
    return dict(sorted(offenders.items(), key=lambda kv: kv[1], reverse=True))


__all__ = [
    "BaseGBDTModel",
    "CatBoostModel",
    "XGBoostModel",
    "GBDTModel",
    "make_model",
    "count_nonfinite",
    "hp_tables_for",
    "model_filename",
    "PINNED_HPS",
    "TUNABLE_HP_RANGES",
    "ENUM_HP_VALUES",
    "PINNED_HPS_XGB",
    "TUNABLE_HP_RANGES_XGB",
    "ENUM_HP_VALUES_XGB",
]
