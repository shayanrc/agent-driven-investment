"""Backtest — the orchestrator.

Per ``spec.md`` § 4.4 + § 5. Wires Stages 1-4 together via a single-phase
``step()`` lifecycle. The engine's ``fill_mode`` (Q6) selects one of two
near-identical paths; there is no per-order branching anywhere.

The five lifecycle steps (PARSE / [FILL or ADVANCE] / [ADVANCE or FILL] /
MARK / RETURN) are deliberately small and explicit. The ``_fill_mode_*``
helpers below are the only place where ``fill_mode`` matters; everything
else (parsing, mark-to-market, state assembly, info packing) is shared.

Look-ahead-bias elimination is structural: the engine never fills against
a bar the caller hasn't yet observed, regardless of mode (B2 enforced by
which bar is handed to ``broker.process_queue``, not by an assertion).
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import pandas as pd

from backtesting.broker import ExecutionBroker
from backtesting.data_handler import DataHandler
from backtesting.portfolio import Portfolio
from backtesting.utils import GapPolicy, parse_action

FillMode = Literal["current_close", "next_open"]


class Backtest:
    """Step-loop backtesting engine.

    Parameters per ``spec.md`` § 6. See ``goal.md`` for the
    structural-look-ahead-elimination rule that shaped every design
    decision; do not "convenience"-modify the lifecycle without surfacing
    it in a doc-level discussion first.
    """

    def __init__(
        self,
        data_feeds: dict[str, dict[str, pd.DataFrame]],
        initial_cash: float = 100_000.0,
        lookback: int = 20,
        lot_sizes: dict[str, int] | None = None,
        default_lot_size: int = 1,
        commission_fn: Callable[[str, float, float], float] | None = None,
        fill_mode: FillMode = "next_open",
        gap_policy: GapPolicy = "ffill_zero_volume",
    ) -> None:
        if fill_mode not in ("current_close", "next_open"):
            raise ValueError(
                f"fill_mode must be 'current_close' or 'next_open', "
                f"got {fill_mode!r}"
            )
        self.data_handler = DataHandler(
            data_feeds, lookback=lookback, gap_policy=gap_policy
        )
        self.portfolio = Portfolio(initial_cash)
        self.broker = ExecutionBroker(commission_fn)
        self.lot_sizes: dict[str, int] = dict(lot_sizes or {})
        self.default_lot_size: int = default_lot_size
        self.fill_mode: FillMode = fill_mode
        # Track most-recent target weights so we can compute realized
        # weight drift after MARK (Q10 weight_drift key).
        self._last_target_weights: dict[str, float] | None = None
        # Initial mark so equity is meaningful from the very first state.
        self._mark_portfolio_at_current_step()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Reset all components; return ``(state, done, info)`` at the
        first valid step. ``done=False``, ``info={}``."""
        self.data_handler.reset()
        self.portfolio.reset()
        self.broker.reset()
        self._last_target_weights = None
        self._mark_portfolio_at_current_step()
        state = self._build_state()
        return state, False, {}

    def step(
        self, action: dict[str, Any] | None
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """One iteration of the step-loop. See ``spec.md`` § 5."""
        # ----- PARSE ------------------------------------------------
        # parse_action raises on invalid weight-sums (Q1) and on extra
        # order fields (Q6); we let those propagate to the caller.
        orders, lot_audit = parse_action(
            action,
            self.portfolio,
            self.data_handler,
            self.lot_sizes,
            self.default_lot_size,
        )
        # Capture target_weights for later weight-drift accounting
        # *before* the FILL/ADVANCE branch runs.
        self._last_target_weights = None
        if action is not None and action.get("type") == "weight":
            self._last_target_weights = dict(action.get("target_weights") or {})

        # Submit valid orders (broker validates again; in practice
        # parse_action has already screened them, so no further
        # rejections expected here unless asset name is unknown).
        invalid_rejections = self.broker.submit_orders(
            orders, known_assets=self.data_handler.get_known_assets()
        )
        # invalid_rejections is also captured inside the broker's
        # _invalid_buffer; the fill log will surface them.
        del invalid_rejections  # silence unused-var; broker tracks them.

        # ----- FILL or ADVANCE (mode-dependent) ----------------------
        if self.fill_mode == "current_close":
            fill_log = self.broker.process_queue(
                self.data_handler.get_current_bar(),
                self.portfolio,
                self.data_handler,
                price_column="close",
            )
            done = self.data_handler.advance_time()
        else:  # "next_open"
            done = self.data_handler.advance_time()
            if done:
                # No T+1 bar exists; pending orders are uniformly
                # rejected as untradeable (B7 contract).
                fill_log = self.broker.drain_pending_as_untradeable()
            else:
                fill_log = self.broker.process_queue(
                    self.data_handler.get_current_bar(),
                    self.portfolio,
                    self.data_handler,
                    price_column="open",
                )

        # ----- MARK --------------------------------------------------
        self._mark_portfolio_at_current_step()

        # ----- RETURN ------------------------------------------------
        info = self._build_info(fill_log, lot_audit)
        state = self._build_state()
        return state, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mark_portfolio_at_current_step(self) -> None:
        """Mark portfolio to close prices at the current step.

        Used by ``__init__`` / ``reset()`` to give the very first state a
        meaningful equity, and by ``step()`` after FILL to mark to T+1
        (or T on the terminal step where ``advance_time`` did not move).

        Builds a price dict containing close for every asset across all
        feeds — only entries for held positions matter (Portfolio
        ignores the rest), but the construction is feed-agnostic.
        """
        bar = self.data_handler.get_current_bar()
        prices: dict[str, float] = {}
        for feed_data in bar.values():
            for asset, row in feed_data.items():
                if "close" in row:
                    prices[asset] = float(row["close"])
        self.portfolio.update_valuations(prices)

    def _build_state(self) -> dict[str, Any]:
        market_data = self.data_handler.get_window()
        ts = self.data_handler.get_current_timestamp()
        pf = self.portfolio.get_state()
        return {
            "market_data": market_data,
            "portfolio": {
                "cash": pf["cash"],
                "equity": pf["equity"],
                "positions": pf["positions"],
                "pending_orders": self.broker.get_pending_count(),
            },
            "step": self.data_handler.current_step,
            "timestamp": ts.isoformat()[:10] if hasattr(ts, "isoformat") else str(ts),
        }

    def _build_info(
        self,
        fill_log: dict[str, list[dict[str, Any]]],
        lot_audit: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Pack the per-step ``info`` dict using the locked Q10 keys.

        Locked keys (from spec.md § 3.2):
        - fills, rejected_overdraw, rejected_untradeable, rejected_invalid
          (list[dict] — omit if empty)
        - weight_drift (dict[asset, float] — omit if empty)
        - rebalance_shortfall (dict[asset, float] — omit if empty)
        - lot_size_audit (dict[asset, {requested_qty, filled_qty}] —
          omit if empty)

        Every key is emitted only when its payload is non-empty.
        """
        info: dict[str, Any] = {}
        if fill_log.get("filled"):
            info["fills"] = fill_log["filled"]
        if fill_log.get("rejected_overdraw"):
            info["rejected_overdraw"] = fill_log["rejected_overdraw"]
        if fill_log.get("rejected_untradeable"):
            info["rejected_untradeable"] = fill_log["rejected_untradeable"]
        if fill_log.get("rejected_invalid"):
            info["rejected_invalid"] = fill_log["rejected_invalid"]
        if lot_audit:
            info["lot_size_audit"] = lot_audit

        # Weight-drift accounting: realized minus target per asset
        # (post-MARK, against the action's target_weights if any).
        weight_drift = self._compute_weight_drift()
        if weight_drift:
            info["weight_drift"] = weight_drift

        # Rebalance shortfall: per-asset (target_qty - filled_qty) when
        # cash overdraw forced a skip. Surfaced from rejected_overdraw on
        # this step, scoped to weight-driven submissions.
        shortfall = self._compute_rebalance_shortfall(
            fill_log.get("rejected_overdraw", [])
        )
        if shortfall:
            info["rebalance_shortfall"] = shortfall

        return info

    def _compute_weight_drift(self) -> dict[str, float]:
        """Return ``{asset: realized_weight - target_weight}`` for every
        asset where the realized weight differs from the target. Returns
        ``{}`` if no weight-action was just submitted, no positions held,
        or all drifts are zero.
        """
        if not self._last_target_weights:
            return {}
        equity = self.portfolio.equity
        if equity <= 0:
            return {}
        # Use current (post-MARK) close prices to compute realized weights.
        bar = self.data_handler.get_current_bar()
        prices: dict[str, float] = {}
        for feed_data in bar.values():
            for asset, row in feed_data.items():
                if "close" in row:
                    prices[asset] = float(row["close"])
        positions = self.portfolio.positions
        relevant = set(self._last_target_weights) | set(positions)
        drift: dict[str, float] = {}
        for asset in relevant:
            price = prices.get(asset, 0.0)
            qty = positions.get(asset, 0.0)
            realized_w = (qty * price) / equity if price > 0 else 0.0
            target_w = self._last_target_weights.get(asset, 0.0)
            d = realized_w - target_w
            if d != 0:
                drift[asset] = d
        return drift

    def _compute_rebalance_shortfall(
        self, rejected_overdraw: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Per-asset shortfall for rebalance-driven overdraw rejections.

        Only meaningful when the just-submitted action was a weight
        rebalance; the rejected order's qty is exactly the shortfall.
        """
        if not self._last_target_weights:
            return {}
        out: dict[str, float] = {}
        for rec in rejected_overdraw:
            asset = rec.get("asset")
            if asset in self._last_target_weights or asset in self.portfolio.positions:
                qty = float(rec.get("qty", 0.0))
                if qty != 0:
                    out[asset] = qty
        return out
