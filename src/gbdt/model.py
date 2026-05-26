"""CatBoost classifier wrapper for gbdt v1.

Thin layer over ``catboost.CatBoostClassifier`` that:

- Pins the non-tunable HPs (``has_time``, ``loss_function``, ``eval_metric``,
  ``custom_metric``, ``random_seed``) per V1_PLAN.md Stage 4. ``has_time=True``
  is mandatory for walk-forward correctness; the wrapper raises if a caller
  tries to override it.
- Validates tunable HPs against ``CATBOOST_HP_REFERENCE.md`` per-parameter
  ranges. Anything out of range is rejected on construction with a clear
  error.
- Exposes ``fit / predict_proba / feature_importance / save / load``.

This is the v1 *model* surface — calibration lives separately in
``gbdt.calibration``, and the FS+HP iteration loop lives in
``gbdt.train`` / ``gbdt.fs_hp_loop``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool


# ---------------------------------------------------------------------------
# HP validation
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


def _validate_hp(hp: dict) -> dict:
    """Return a copy of ``hp`` with pinned HPs applied and tunable HPs
    range-checked. Raise ``ValueError`` on the first violation.
    """
    out = dict(hp)

    # Reject overrides on pinned HPs (except matching value, which is a no-op).
    for k, v in PINNED_HPS.items():
        if k in out and out[k] != v:
            raise ValueError(
                f"{k!r} is pinned to {v!r} in v1 and cannot be overridden "
                f"(got {out[k]!r}); see V1_PLAN.md Stage 4."
            )
        out[k] = v

    # Numeric range checks
    for k, (lo, hi) in TUNABLE_HP_RANGES.items():
        if k not in out or out[k] is None:
            continue
        v = out[k]
        if hi is not None:
            if not (lo <= v <= hi):
                raise ValueError(
                    f"HP {k}={v} is outside the documented range [{lo}, {hi}] "
                    f"(see docs/gbdt/CATBOOST_HP_REFERENCE.md)."
                )
        else:
            if v < lo:
                raise ValueError(
                    f"HP {k}={v} is below the documented minimum {lo}."
                )

    # Enum checks
    for k, allowed in ENUM_HP_VALUES.items():
        if k not in out or out[k] is None:
            continue
        if out[k] not in allowed:
            raise ValueError(
                f"HP {k}={out[k]!r} is not one of {allowed}; "
                f"see docs/gbdt/CATBOOST_HP_REFERENCE.md."
            )

    return out


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class GBDTModel:
    """Thin CatBoost wrapper. ``has_time=True`` is mandatory.

    Build via ``GBDTModel(hp_dict, feature_names=...)``; fit with
    ``fit(X_train, y_train, X_val, y_val)``; score with ``predict_proba(X)``
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
    ) -> "GBDTModel":
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
             feature_names: list[str] | None = None) -> "GBDTModel":
        m = CatBoostClassifier()
        m.load_model(str(path))
        obj = cls(hp or PINNED_HPS, feature_names=feature_names)
        obj._model = m
        obj._fitted = True
        if feature_names is None:
            obj._feature_names = list(m.feature_names_) if m.feature_names_ else None
        return obj


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
    "GBDTModel",
    "PINNED_HPS",
    "TUNABLE_HP_RANGES",
    "ENUM_HP_VALUES",
]
