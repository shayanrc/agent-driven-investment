"""ExecutionBroker — the order queue.

Per ``spec.md`` § 4.3. v1 has a single fill path: every order fills at the
bar the orchestrator hands in. ``Backtest.step()`` picks T or T+1 based on
its configured ``fill_mode`` (Q6); the broker is fill_mode-agnostic.

Key design choices reflected here:

- Sells process before buys (stable sort by qty sign) so liquidations
  free cash for subsequent purchases without ever allowing negative cash.
- Direction-aware overdraw guard:
    * buys check ``qty * price + commission <= portfolio.cash``
    * sells check ``commission <= portfolio.cash + abs(qty) * price``
  (post-fill cash covers commission).
- Untradeable assets (pre-IPO / delisted): buys are rejected; sells are
  permitted at the last-known close so existing positions can be closed.
- No partial fills in v1 — order fills completely or is rejected.

The ``process_queue`` return is a fill log dict with four buckets
(``filled``, ``rejected_overdraw``, ``rejected_untradeable``,
``rejected_invalid``). ``rejected_invalid`` is populated at
``submit_orders`` time, returned for the orchestrator to surface in
``info``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from backtesting.data_handler import DataHandler
from backtesting.portfolio import Portfolio

# An order is exactly {"asset": str, "qty": float}. The strict schema is a
# Q6 contract: extra fields (execution, limit_price, time_in_force) get
# rejected at submit time so v1.1 reintroducing those features is a
# deliberate API extension.
REQUIRED_ORDER_FIELDS: frozenset[str] = frozenset({"asset", "qty"})


class ExecutionBroker:
    """Order queue + single-phase fill engine.

    Parameters
    ----------
    commission_fn:
        Optional ``(asset, qty, price) -> float`` callable (Q5 signature).
        Called once per fill; the returned amount is deducted from cash as
        a post-fill adjustment. ``None`` ⇒ zero commission.
    """

    def __init__(self, commission_fn: Callable[[str, float, float], float] | None = None) -> None:
        self.commission_fn = commission_fn
        self.pending_orders: list[dict[str, Any]] = []
        # Invalid-at-submit orders are stored here until process_queue (or
        # the orchestrator) folds them into the fill log. This keeps the
        # invalid-rejection info path consistent with the overdraw /
        # untradeable buckets even though invalid orders never enter the
        # pending queue.
        self._invalid_buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit_orders(
        self,
        orders: Iterable[dict[str, Any]],
        known_assets: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Validate and enqueue orders.

        Validation rules:
        - Exactly the two required fields ``{asset: str, qty: float}``.
          Extra fields (``execution``, ``limit_price``, ``time_in_force``)
          are rejected per Q6.
        - ``qty != 0``.
        - If ``known_assets`` is provided, ``asset`` must be in it (catches
          typos at submit time; untradeable-but-known assets are handled
          at fill time in ``process_queue``).

        Rejected orders are appended to the internal invalid buffer with
        a ``reason`` field and returned for the orchestrator to surface.
        Returns the list of invalid-this-submit records (for convenience
        — the same records are also accumulated in ``_invalid_buffer``).
        """
        rejected_this_submit: list[dict[str, Any]] = []
        for order in orders:
            reason = self._validate_order(order, known_assets)
            if reason is None:
                # Normalize qty to float for consistent downstream math.
                normalized = {"asset": order["asset"], "qty": float(order["qty"])}
                self.pending_orders.append(normalized)
            else:
                rec = {**order, "reason": reason}
                self._invalid_buffer.append(rec)
                rejected_this_submit.append(rec)
        return rejected_this_submit

    @staticmethod
    def _validate_order(
        order: dict[str, Any], known_assets: set[str] | None
    ) -> str | None:
        """Return a reason string if invalid, ``None`` if valid."""
        if not isinstance(order, dict):
            return "order must be a dict"
        keys = set(order.keys())
        if keys != REQUIRED_ORDER_FIELDS:
            extra = keys - REQUIRED_ORDER_FIELDS
            missing = REQUIRED_ORDER_FIELDS - keys
            return (
                f"schema mismatch: missing={sorted(missing)} "
                f"extra={sorted(extra)} (expected exactly "
                f"{sorted(REQUIRED_ORDER_FIELDS)})"
            )
        if not isinstance(order["asset"], str):
            return f"asset must be str, got {type(order['asset']).__name__}"
        qty = order["qty"]
        if not isinstance(qty, (int, float)) or isinstance(qty, bool):
            return f"qty must be int or float, got {type(qty).__name__}"
        if qty == 0:
            return "qty == 0 is not a valid order"
        if known_assets is not None and order["asset"] not in known_assets:
            return f"unknown asset {order['asset']!r}"
        return None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_queue(
        self,
        current_bar: dict[str, dict[str, dict[str, float]]],
        portfolio: Portfolio,
        data_handler: DataHandler,
        price_column: str = "close",
    ) -> dict[str, list[dict[str, Any]]]:
        """Process all pending orders against ``current_bar``.

        Sells (qty < 0) execute before buys (qty > 0). Cash freed by
        liquidations is available for subsequent buys without ever
        allowing the cash guard to trigger.

        Untradeable assets (per ``data_handler.is_active``): buys
        rejected, sells permitted at the bar's price (typically the
        last-known close after forward-fill).

        Returns a fill log dict with keys ``filled`` /
        ``rejected_overdraw`` / ``rejected_untradeable`` /
        ``rejected_invalid``. The queue is empty after this call (every
        pending order is either filled or moved to a rejection bucket).
        """
        # Stable sort: sells (qty < 0) first, then buys (qty > 0).
        ordered = sorted(self.pending_orders, key=lambda o: o["qty"] > 0)

        log: dict[str, list[dict[str, Any]]] = {
            "filled": [],
            "rejected_overdraw": [],
            "rejected_untradeable": [],
            "rejected_invalid": list(self._invalid_buffer),
        }
        self._invalid_buffer = []

        # Build a fast asset -> fill-price lookup from the current bar.
        # The orchestrator hands us the right bar (T for current_close,
        # T+1 for next_open); we just look up the configured price
        # column for each asset.
        price_lookup: dict[str, float] = {}
        for feed_data in current_bar.values():
            for asset_name, row in feed_data.items():
                if price_column in row:
                    price_lookup[asset_name] = float(row[price_column])

        for order in ordered:
            asset = order["asset"]
            qty = order["qty"]

            # Untradeable check (pre-IPO / delisted at current step).
            try:
                active = data_handler.is_active(asset)
            except KeyError:
                # Unknown asset shouldn't reach here (caught at submit),
                # but be defensive.
                log["rejected_untradeable"].append(
                    {**order, "reason": "unknown asset at fill time"}
                )
                continue

            if not active and qty > 0:
                log["rejected_untradeable"].append(
                    {**order, "reason": "asset inactive at fill bar (buy rejected)"}
                )
                continue
            # Inactive sells are permitted (fill at last-known price).

            price = price_lookup.get(asset)
            if price is None or _is_nan(price):
                log["rejected_untradeable"].append(
                    {**order, "reason": "no fill price available at current bar"}
                )
                continue

            commission = (
                float(self.commission_fn(asset, qty, price))
                if self.commission_fn is not None
                else 0.0
            )

            # Direction-aware overdraw guard (spec.md § 4.2).
            if qty > 0:
                # Buy: need cash for trade + commission.
                if qty * price + commission > portfolio.cash:
                    log["rejected_overdraw"].append(
                        {
                            **order,
                            "fill_price": price,
                            "commission": commission,
                            "reason": "insufficient cash for buy",
                        }
                    )
                    continue
            else:
                # Sell: post-fill cash must cover commission.
                if commission > portfolio.cash + abs(qty) * price:
                    log["rejected_overdraw"].append(
                        {
                            **order,
                            "fill_price": price,
                            "commission": commission,
                            "reason": "insufficient cash for sell-side commission",
                        }
                    )
                    continue

            # All checks passed: route to portfolio. Trade first
            # (changes cash by qty*price), then deduct commission as a
            # post-fill cash adjustment so the portfolio sees only the
            # market-price leg.
            portfolio.execute_trade(asset, qty, price)
            portfolio.cash -= commission

            log["filled"].append(
                {
                    "asset": asset,
                    "qty": qty,
                    "fill_price": price,
                    "commission": commission,
                }
            )

        # Queue is fully drained after process_queue (either filled or
        # rejected); orders that need to persist across bars (limit GTC
        # etc.) are v1.1 features.
        self.pending_orders = []
        return log

    def drain_pending_as_untradeable(
        self, reason: str = "terminal step, no future bar to fill against"
    ) -> dict[str, list[dict[str, Any]]]:
        """Move all pending orders into the rejected_untradeable bucket.

        Called by ``Backtest.step()`` for ``fill_mode="next_open"`` when
        the timeline has exhausted and there is no T+1 bar to fill
        against (B7 contract). Also flushes any invalid orders captured
        at submit time.
        """
        log: dict[str, list[dict[str, Any]]] = {
            "filled": [],
            "rejected_overdraw": [],
            "rejected_untradeable": [
                {**order, "reason": reason} for order in self.pending_orders
            ],
            "rejected_invalid": list(self._invalid_buffer),
        }
        self.pending_orders = []
        self._invalid_buffer = []
        return log

    def get_pending_count(self) -> int:
        """Number of orders still in the queue."""
        return len(self.pending_orders)

    def reset(self) -> None:
        """Clear all pending orders and the invalid buffer."""
        self.pending_orders = []
        self._invalid_buffer = []


def _is_nan(value: float) -> bool:
    return value != value  # NaN is the only IEEE-754 value with x != x.
