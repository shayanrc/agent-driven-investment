"""trading_strategies — concrete ``backtesting.Strategy`` classes + sizers.

See ``docs/trading_strategies/goal.md`` for the charter and
``docs/backtests/V1_cell5_bayesian_kelly_plan.md`` §5.2 for the v1 scope.

The defining rule (goal.md): **the strategy is backend-agnostic in its
probability contract.** A strategy accepts a
``dict[Timestamp, list[(ticker, p_mean, p_low, p_high)]]`` — never a model
object, a fitted calibrator, or a backend reference. No predictor imports.

Two sizer protocols, deliberately (goal.md):

- :class:`PortfolioSizer` fits on a return series and exposes a single
  fraction (e.g. ``VinceOptimalF``).
- :class:`PerPredictionSizer` computes a fraction per prediction from that
  prediction's probability (e.g. ``DiscreteBoundedLossKelly``).

A strategy dispatches on ``isinstance(sizer, PortfolioSizer)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PortfolioSizer(Protocol):
    """Fit on a realized-return series; expose one portfolio fraction.

    The fraction is the per-position fraction-at-risk applied uniformly.
    Used by ``VinceOptimalF`` (TWR-optimal f fit on the trade-return
    series). Fitting happens once, off-line, before the strategy runs.
    """

    def fit(self, trade_returns: np.ndarray) -> "PortfolioSizer": ...

    @property
    def per_position_fraction_at_risk(self) -> float: ...


@runtime_checkable
class PerPredictionSizer(Protocol):
    """Compute a fraction-at-risk per prediction from its probability.

    Closed-form, no fit data — the sizer just needs the per-cell payoff
    geometry passed at call time. Used by ``DiscreteBoundedLossKelly``.
    """

    def fraction_at_risk(
        self, p: float, *, payoff_win: float, payoff_loss: float
    ) -> float: ...


# Imported at the END so the protocols above are already defined when the
# strategy module does `from trading_strategies import PerPredictionSizer`.
from trading_strategies.topk_daily_kelly_label_exit import (  # noqa: E402
    StrategyEvent,
    TopKDailyKellyLabelExit,
)

__all__ = [
    "PortfolioSizer",
    "PerPredictionSizer",
    "TopKDailyKellyLabelExit",
    "StrategyEvent",
]
