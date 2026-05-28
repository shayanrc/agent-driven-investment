"""Conditional isotonic calibration with Spiegelhalter Z-test gating.

Per V1_PLAN.md Stage 5: compute the Spiegelhalter Z statistic on the val
segment's raw predictions; if ``|z| < z_threshold`` (default 2.0) ship the
model's native outputs, else fit ``sklearn.isotonic.IsotonicRegression``
on val and persist the calibrator alongside the model.

Spiegelhalter Z statistic for predicted probabilities ``p_i`` and outcomes
``y_i ∈ {0, 1}``:

    Z = sum((y - p) * (1 - 2p)) / sqrt(sum((1 - 2p)^2 * p * (1 - p)))

Under perfect calibration Z is approximately N(0, 1); |Z| > 2 is the
standard two-sided 5% miscalibration signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Spiegelhalter Z
# ---------------------------------------------------------------------------


def spiegelhalter_z(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float]:
    """Return ``(z, two_sided_p_value)``.

    Robust to extreme p (clamps to ``[eps, 1-eps]``) so the variance term
    stays finite.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p_pred, dtype=float).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size != p.size:
        raise ValueError(f"length mismatch: y={y.size}, p={p.size}")
    if y.size == 0:
        return 0.0, 1.0
    num = float(np.sum((y - p) * (1.0 - 2.0 * p)))
    var = float(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if var <= 0:
        return 0.0, 1.0
    z = num / np.sqrt(var)
    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(pval)


# ---------------------------------------------------------------------------
# Isotonic fit
# ---------------------------------------------------------------------------


def fit_isotonic(y_true: np.ndarray, p_pred: np.ndarray) -> IsotonicRegression:
    """Fit IsotonicRegression(y ~ p) with ``out_of_bounds='clip'``."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p_pred).ravel(), np.asarray(y_true).ravel())
    return iso


# ---------------------------------------------------------------------------
# Platt fit (backend-neutral; V1.2 Phase 6 / plan R7)
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """A 1-D Platt scaling map fit on ``(p_raw, y)`` — backend-neutral.

    Platt scaling (Platt 1999) fits a logistic regression ``sigmoid(a·s + b)``
    from a raw score ``s`` to a calibrated probability. Per V1.2 plan R7
    (``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`` § 4.3 / § 11) the
    fit is done **directly on the raw probability vector + the binary outcome**,
    NOT by routing through ``XGBClassifier`` / ``CatBoostClassifier`` or any
    model-specific sklearn surface. This matches exactly how :func:`fit_isotonic`
    already works — it operates only on the ``(y_true, p_pred)`` arrays the
    model's ``predict_proba`` produces, so the *same* Platt path applies to any
    backend's raw probabilities.

    Exposes a ``.predict(p_raw) -> calibrated_proba`` method (returning the
    positive-class probability, NOT a hard class label), so it composes with
    :func:`apply_calibrator` exactly like an :class:`IsotonicRegression` does
    and pickles cleanly into ``calibration.pkl``.
    """

    def __init__(self, lr: LogisticRegression) -> None:
        self._lr = lr

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        """Map raw probabilities through the fitted logistic to calibrated ones."""
        s = np.asarray(p_raw, dtype=float).ravel().reshape(-1, 1)
        return self._lr.predict_proba(s)[:, 1]


def fit_platt(y_true: np.ndarray, p_pred: np.ndarray) -> PlattCalibrator:
    """Fit a backend-neutral Platt scaler ``sigmoid(a·p + b)`` on ``(p_raw, y)``.

    Mirrors :func:`fit_isotonic`: takes the raw probability vector and the binary
    outcomes, fits a 1-D :class:`~sklearn.linear_model.LogisticRegression`
    (the raw probability is the single regressor), and returns a
    :class:`PlattCalibrator` whose ``.predict`` yields calibrated probabilities.
    No model object is touched (R7) — backend-agnostic.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel().reshape(-1, 1)
    if y.size != p.shape[0]:
        raise ValueError(f"length mismatch: y={y.size}, p={p.shape[0]}")
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(p, y.astype(int))
    return PlattCalibrator(lr)


def apply_calibrator(p_raw: np.ndarray, calibrator: Any | None) -> np.ndarray:
    """Return calibrated probabilities.

    - ``calibrator is None``  → returns ``p_raw`` unchanged (native pass-through).
    - ``IsotonicRegression``  → ``calibrator.predict(p_raw)``.
    - :class:`PlattCalibrator` (or any other object with a ``.predict`` method)
      is also supported — ``.predict`` is contracted to return calibrated
      probabilities, not class labels.
    """
    arr = np.asarray(p_raw, dtype=float).ravel()
    if calibrator is None:
        return arr
    if hasattr(calibrator, "predict"):
        return np.asarray(calibrator.predict(arr), dtype=float)
    raise TypeError(f"unsupported calibrator type: {type(calibrator)!r}")


# ---------------------------------------------------------------------------
# Conditional decision
# ---------------------------------------------------------------------------


@dataclass
class CalibrationDecision:
    """Result of :func:`conditional_isotonic`."""

    method: str                                # "native" | "isotonic" | "platt"
    spiegelhalter_z: float
    spiegelhalter_p: float
    z_threshold: float
    calibrator: IsotonicRegression | None
    rationale: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "spiegelhalter_z": self.spiegelhalter_z,
            "spiegelhalter_p": self.spiegelhalter_p,
            "z_threshold": self.z_threshold,
            "rationale": self.rationale,
        }


def conditional_isotonic(
    y_val: np.ndarray,
    p_val_raw: np.ndarray,
    z_threshold: float = 2.0,
) -> CalibrationDecision:
    """Run the Spiegelhalter gate and either ship native or fit isotonic."""
    z, pval = spiegelhalter_z(y_val, p_val_raw)
    if abs(z) < z_threshold:
        return CalibrationDecision(
            method="native",
            spiegelhalter_z=z,
            spiegelhalter_p=pval,
            z_threshold=z_threshold,
            calibrator=None,
            rationale=(
                f"|z|={abs(z):.3f} < {z_threshold}; "
                f"native CatBoost probabilities are well-calibrated on val."
            ),
        )
    iso = fit_isotonic(y_val, p_val_raw)
    return CalibrationDecision(
        method="isotonic",
        spiegelhalter_z=z,
        spiegelhalter_p=pval,
        z_threshold=z_threshold,
        calibrator=iso,
        rationale=(
            f"|z|={abs(z):.3f} >= {z_threshold}; "
            f"native miscalibrated, fit IsotonicRegression on val."
        ),
    )


def isotonic_always(y_val: np.ndarray, p_val_raw: np.ndarray,
                     z_threshold: float = 2.0) -> CalibrationDecision:
    """Fit isotonic unconditionally; still records the Z for the artifact."""
    z, pval = spiegelhalter_z(y_val, p_val_raw)
    iso = fit_isotonic(y_val, p_val_raw)
    return CalibrationDecision(
        method="isotonic",
        spiegelhalter_z=z,
        spiegelhalter_p=pval,
        z_threshold=z_threshold,
        calibrator=iso,
        rationale="isotonic_always: unconditional isotonic on val",
    )


def platt_calibration(y_val: np.ndarray, p_val_raw: np.ndarray,
                      z_threshold: float = 2.0) -> CalibrationDecision:
    """Fit a backend-neutral Platt scaler unconditionally; records the Z too.

    The ``calibration_method: platt`` path (``EXPERIMENT_SPEC.md``,
    ``__main__.py`` valid-methods). Per V1.2 plan R7, Platt is fit **manually on
    ``(p_raw, y)``** via :func:`fit_platt` — never by routing through an
    ``XGBClassifier`` / ``CatBoostClassifier`` sklearn surface — so the path is
    backend-agnostic, exactly like :func:`isotonic_always`. The Z statistic is
    recorded for the artifact; the gate does not branch on it (Platt is
    unconditional, mirroring ``isotonic_always``).
    """
    z, pval = spiegelhalter_z(y_val, p_val_raw)
    platt = fit_platt(y_val, p_val_raw)
    return CalibrationDecision(
        method="platt",
        spiegelhalter_z=z,
        spiegelhalter_p=pval,
        z_threshold=z_threshold,
        calibrator=platt,
        rationale="platt: unconditional Platt scaling fit on (p_raw, y) — backend-neutral (R7)",
    )


__all__ = [
    "CalibrationDecision",
    "PlattCalibrator",
    "spiegelhalter_z",
    "fit_isotonic",
    "fit_platt",
    "apply_calibrator",
    "conditional_isotonic",
    "isotonic_always",
    "platt_calibration",
]
