"""calibration — backend-agnostic probability-calibration toolkit.

See ``docs/calibration/goal.md`` for the charter and
``docs/backtests/V1_cell5_bayesian_kelly_plan.md`` §5.1 for the v1 scope.

The defining rule (goal.md): **the calibrator knows nothing about the
predictor.** Inputs are NumPy arrays of ``(p_raw, y_true)`` for fit and
``p_raw`` for transform; outputs are ``CalibrationOutput`` (NumPy arrays of
``p_mean`` plus optional ``p_low`` / ``p_high``). No predictor imports.

v1 ships the :class:`Calibrator` Protocol + :class:`CalibrationOutput`
dataclass, the :class:`~calibration.bayesian.BetaBinomialBucketed`
calibrator, and the ECE / reliability diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CalibrationOutput:
    """The standard return of :meth:`Calibrator.transform`.

    Attributes
    ----------
    p_mean:
        Point estimate of ``P(y=1 | p_raw)`` per input row.
    p_low, p_high:
        Lower / upper bound of the 95% credible interval per row. Bayesian
        calibrators populate these; non-Bayesian calibrators may set them
        to ``None`` (goal.md: "Non-Bayesian calibrators may return ``None``
        for ``p_low`` / ``p_high``").
    """

    p_mean: np.ndarray
    p_low: np.ndarray | None = None
    p_high: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.p_mean)
        for name in ("p_low", "p_high"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"CalibrationOutput.{name} has length {len(arr)} "
                    f"but p_mean has length {n}"
                )


@runtime_checkable
class Calibrator(Protocol):
    """The minimal fit / transform contract (goal.md: a Protocol, not an ABC).

    Any object exposing ``fit(p_raw, y_true) -> Calibrator`` and
    ``transform(p_raw) -> CalibrationOutput`` satisfies the contract; no
    inheritance required. The v1 fit signature is the minimal one —
    ``(p_raw, y_true)`` plus an optional ``sample_weight`` keyword.
    """

    def fit(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "Calibrator": ...

    def transform(self, p_raw: np.ndarray) -> CalibrationOutput: ...


__all__ = ["CalibrationOutput", "Calibrator"]
