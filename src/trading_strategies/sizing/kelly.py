"""DiscreteBoundedLossKelly — closed-form per-prediction Kelly sizer (D6).

The primary sizer under daily rebalance. For a binary outcome with a
bounded win payoff ``b_win`` and bounded loss payoff ``b_loss`` (both as
positive fractions of notional, e.g. cell-5: win = 0.10, loss = 0.05), the
Kelly fraction-at-risk is

    f_risk = max(0, (b·p − q) / b),  with b = b_win / b_loss, q = 1 − p

where ``f_risk`` is the fraction of bankroll put at risk per position. The
implied notional is ``f_risk · equity / b_loss`` — it can exceed equity
when ``b_loss < f_risk`` (the strategy's gross cap + per-entry floor manage
that; see plan D9/D23). No fit data: closed-form from ``p`` + payoffs.
"""

from __future__ import annotations


class DiscreteBoundedLossKelly:
    """Closed-form Kelly fraction-at-risk for a bounded-loss binary bet.

    Conforms to the ``PerPredictionSizer`` protocol. Stateless — the same
    instance serves every pick; payoffs are passed per call so one sizer
    works across cells with different win/loss geometry.
    """

    def fraction_at_risk(
        self, p: float, *, payoff_win: float, payoff_loss: float
    ) -> float:
        """Kelly fraction-at-risk, clipped at 0.

        Parameters
        ----------
        p:
            Probability of the win outcome (calibrated ``p_mean``).
        payoff_win, payoff_loss:
            Positive win / loss payoffs as fractions of notional. For
            cell-5 these are the label boundaries (+0.10 target, −0.05 DD).

        Returns
        -------
        float
            ``max(0, (b·p − q) / b)`` with ``b = payoff_win / payoff_loss``.
            Zero when the edge does not clear breakeven ``p = q/(1+b)``.
        """
        if payoff_win <= 0.0 or payoff_loss <= 0.0:
            raise ValueError(
                f"payoffs must be positive; got win={payoff_win}, "
                f"loss={payoff_loss}"
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        b = payoff_win / payoff_loss
        q = 1.0 - p
        return max(0.0, (b * p - q) / b)
