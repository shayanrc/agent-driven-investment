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


def apply_calibrator(p_raw: np.ndarray, calibrator: Any | None) -> np.ndarray:
    """Return calibrated probabilities.

    - ``calibrator is None``  → returns ``p_raw`` unchanged (native pass-through).
    - ``IsotonicRegression``  → ``calibrator.predict(p_raw)``.
    - Any other object with a ``.predict`` method is also supported (Platt /
      future calibrators).
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

    method: str                                # "native" | "isotonic"
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


__all__ = [
    "CalibrationDecision",
    "spiegelhalter_z",
    "fit_isotonic",
    "apply_calibrator",
    "conditional_isotonic",
    "isotonic_always",
]
