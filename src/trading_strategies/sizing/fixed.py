"""FixedFraction — naive constant fraction-at-risk baseline (§7 ablation).

The simplest sizer: every pick gets the same fraction-at-risk, regardless
of probability or payoff. Used as the naive baseline row in the §7
sensitivity table (e.g. ``FixedFraction(0.20)``).

Implements ``PerPredictionSizer`` so it slots into the strategy in place of
the Kelly sizer with no strategy-code change — it simply ignores ``p`` and
the payoffs.
"""

from __future__ import annotations


class FixedFraction:
    """Return a constant fraction-at-risk for every prediction."""

    def __init__(self, fraction: float) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1]; got {fraction}")
        self._fraction = fraction

    @property
    def fraction(self) -> float:
        return self._fraction

    def fraction_at_risk(
        self, p: float, *, payoff_win: float, payoff_loss: float
    ) -> float:
        """Constant fraction-at-risk; ``p`` and payoffs are ignored."""
        return self._fraction
