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
- :func:`make_model` — the factory that dispatches on ``backend.library`` and
  returns the right concrete model. ``"catboost"`` routes to
  :class:`CatBoostModel`; other backends raise ``NotImplementedError`` (the
  XGBoost backend lands in a later V1.2 phase).

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
# Phase 2 ships the **name-mapping data + the resolver + backend-aware
# validation**. The XGBoost *model adapter*, the ``xgboost`` dependency, and the
# determinism hard-fail (``tree_method``/``n_jobs``/``device`` enforcement on
# ``PINNED_HPS_XGB``) land in later V1.2 phases (§ 8 Phases 1/3); the pins below
# are recorded as table *data* (so ``validate_decision`` rejects an agent
# request to change them) but are not yet enforced at model construction.


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

# Pinned: never overridable from a spec. For XGBoost the load-bearing pins are
# the determinism knobs (``tree_method``/``n_jobs``/``device``) that replace
# ``has_time``'s "never-override" role (plan § 5.1/§ 5.3). Phase 2 records these
# as table *data* so :func:`validate_decision` rejects an agent request to change
# them; the construction-time **hard-fail** enforcement (``_validate_hp_xgb``) is
# Phase 3 (plan § 8). ``seed`` is set from ``random_seed`` at construction, like
# CatBoost's ``random_seed`` — so it is not listed as a fixed-value pin here.
PINNED_HPS_XGB: dict[str, Any] = {
    "objective": "binary:logistic",            # ↔ Logloss
    "eval_metric": "logloss",                   # Brier computed in the bundle
    "tree_method": "exact",                     # determinism (plan § 5.1)
    "n_jobs": 1,                                # determinism
    "device": "cpu",                            # determinism
}


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


def _validate_hp(hp: dict, backend: str = "catboost") -> dict:
    """Return a copy of ``hp`` with pinned HPs applied and tunable HPs
    range-checked against ``backend``'s tables. Raise ``ValueError`` on the
    first violation.

    ``backend`` defaults to ``"catboost"`` so every existing caller (and the
    ``GBDTModel`` alias) behaves byte-for-byte as before the V1.2 seam.
    """
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

    return out


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
    - anything else → ``NotImplementedError``. The XGBoost backend lands in a
      later V1.2 phase (``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md``
      § 8); this factory is the dispatch point it will plug into.
    """
    library = _resolve_backend(backend)
    if library == "catboost":
        return CatBoostModel(
            hp, feature_names=feature_names, random_seed=random_seed
        )
    raise NotImplementedError(
        "xgboost backend lands in a later V1.2 phase "
        f"(got backend.library={library!r}); only 'catboost' is wired today."
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


__all__ = [
    "BaseGBDTModel",
    "CatBoostModel",
    "GBDTModel",
    "make_model",
    "hp_tables_for",
    "PINNED_HPS",
    "TUNABLE_HP_RANGES",
    "ENUM_HP_VALUES",
    "PINNED_HPS_XGB",
    "TUNABLE_HP_RANGES_XGB",
    "ENUM_HP_VALUES_XGB",
]
