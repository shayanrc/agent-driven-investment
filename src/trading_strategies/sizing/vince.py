"""VinceOptimalF — TWR-optimal fixed fraction fit on a return series (D7).

Ralph Vince's "optimal f": the single fraction ``f ∈ (0, 1)`` that
maximizes terminal wealth relative (TWR) over a realized trade-return
series. Each return is normalized by the worst loss; the holding period
return per trade is ``1 + f · (rᵢ / |max_loss|)``, and TWR is the product.

This is a ``PortfolioSizer`` — fit once on the eval-replay return series
(plan §6.2), expose one fraction. Used only for the §7 ablation row; the
primary sizer under daily rebalance is the closed-form Kelly (D6).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class VinceOptimalF:
    """Fit the TWR-maximizing fixed fraction on a trade-return series."""

    def __init__(self) -> None:
        self._f_star: float | None = None
        self._max_loss: float | None = None
        self._diagnostics: dict[str, float] | None = None

    @property
    def per_position_fraction_at_risk(self) -> float:
        if self._f_star is None:
            raise RuntimeError("VinceOptimalF.fit() must be called first")
        return self._f_star

    @property
    def diagnostics(self) -> dict[str, float]:
        if self._diagnostics is None:
            raise RuntimeError("VinceOptimalF.fit() must be called first")
        return dict(self._diagnostics)

    def fit(self, trade_returns: np.ndarray) -> "VinceOptimalF":
        """Solve ``f* = argmax_f Π(1 + f · rᵢ / |max_loss|)`` over ``f ∈ (0, 1)``.

        Requires at least one losing trade (``min(returns) < 0``) — the
        normalization divides by the worst loss. Raises if the series has
        no loss (optimal f is then undefined / unbounded).
        """
        r = np.asarray(trade_returns, dtype=float)
        if r.size == 0:
            raise ValueError("VinceOptimalF.fit: empty trade_returns")
        worst = r.min()
        if worst >= 0.0:
            raise ValueError(
                "VinceOptimalF.fit: no losing trade (min return >= 0); "
                "optimal f is undefined without a worst-loss normalizer"
            )
        max_loss = -worst
        normalized = r / max_loss

        def neg_log_twr(f: float) -> float:
            hpr = 1.0 + f * normalized
            # Guard the log domain: f ∈ (0, 1) keeps 1 + f·(rᵢ/|max_loss|)
            # > 0 because the worst normalized return is exactly −1.
            if np.any(hpr <= 0.0):
                return np.inf
            return -np.sum(np.log(hpr))

        res = minimize_scalar(
            neg_log_twr, bounds=(1e-6, 1.0 - 1e-6), method="bounded"
        )
        self._f_star = float(res.x)
        self._max_loss = float(max_loss)
        self._diagnostics = {
            "n_trades": float(r.size),
            "mean_return": float(r.mean()),
            "max_loss": float(max_loss),
            "max_gain": float(r.max()),
            "f_star": self._f_star,
        }
        return self
