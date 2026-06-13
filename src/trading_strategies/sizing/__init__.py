"""sizing — position-sizing components for trading_strategies.

Sizers are first-class, separable, swappable (goal.md). v1 ships:

- :class:`~trading_strategies.sizing.kelly.DiscreteBoundedLossKelly`
  (``PerPredictionSizer``) — the primary sizer under daily rebalance (D6).
- :class:`~trading_strategies.sizing.vince.VinceOptimalF`
  (``PortfolioSizer``) — §7 ablation only (D7).
- :class:`~trading_strategies.sizing.fixed.FixedFraction` — §7 baseline.
"""

from __future__ import annotations

from trading_strategies.sizing.fixed import FixedFraction
from trading_strategies.sizing.kelly import DiscreteBoundedLossKelly
from trading_strategies.sizing.vince import VinceOptimalF

__all__ = ["DiscreteBoundedLossKelly", "VinceOptimalF", "FixedFraction"]
