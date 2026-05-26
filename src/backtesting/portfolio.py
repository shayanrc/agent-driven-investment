"""Portfolio — the ledger primitive.

Per ``spec.md`` § 4.2. Tracks cash, positions, and valuations. Knows nothing
about time, data, or execution logic — those live in DataHandler / Broker /
Backtest. Lot-size enforcement happens upstream in ``Backtest._parse_action``,
not here; the portfolio accepts float qty so a single class can serve both
whole-share and fractional instruments.

Cash cannot go negative. ``execute_trade`` is the last-line-of-defense guard
that ``raise``s on overdraw; the direction-aware pre-fill check in the broker
(``spec.md`` § 4.2 design note) should mean this is never tripped in practice,
but the guard exists to make B4 (portfolio consistency) unviolatable even if
a future caller bypasses the broker.
"""

from __future__ import annotations


class Portfolio:
    """Cash + positions ledger.

    Parameters
    ----------
    initial_cash:
        Starting cash balance. Must be non-negative.
    """

    def __init__(self, initial_cash: float) -> None:
        if initial_cash < 0:
            raise ValueError(
                f"initial_cash must be >= 0, got {initial_cash}"
            )
        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.positions: dict[str, float] = {}
        self.equity: float = float(initial_cash)

    def execute_trade(self, asset: str, qty: float, price: float) -> None:
        """Apply a single fill to the ledger.

        ``cash -= qty * price`` (positive qty = buy = cash outflow;
        negative qty = sell = cash inflow). Positions track net signed
        quantity (long if positive, short if negative).

        Raises ``ValueError`` if the resulting cash would be negative. The
        broker is responsible for the direction-aware pre-fill guard
        (``spec.md`` § 4.2); this raise is the structural last-line defense
        that ensures B4 cannot be violated even if a buggy caller bypasses
        the broker.

        Commission is *not* applied here. The broker deducts commission as a
        post-fill cash adjustment after this method returns; that keeps the
        ledger commission-agnostic and the commission model entirely
        broker-side (per ``spec.md`` § 4.2).
        """
        if qty == 0:
            # No-op fill is not a valid order in the broker pipeline; if it
            # leaks through, just do nothing.
            return
        new_cash = self.cash - qty * price
        if new_cash < 0:
            raise ValueError(
                f"execute_trade would overdraw cash: asset={asset!r} "
                f"qty={qty} price={price} cash_before={self.cash} "
                f"cash_after={new_cash}"
            )
        self.cash = new_cash
        prev = self.positions.get(asset, 0.0)
        new_pos = prev + qty
        if new_pos == 0:
            # Drop zero positions so they don't appear in get_state()'s
            # positions dict — keeps the surface clean and consistent
            # with "absent key ≡ zero position".
            self.positions.pop(asset, None)
        else:
            self.positions[asset] = new_pos

    def update_valuations(self, current_prices: dict[str, float]) -> None:
        """Mark all positions to market.

        Sets ``self.equity = cash + Σ position_qty * mark_price``. Called
        once per step by the orchestrator after all fills are processed.

        ``current_prices`` must contain a price for every asset currently
        held. A missing price for a held asset raises ``KeyError`` — that
        is a programming error, not a recoverable state. (Untradeable
        assets are still forward-filled by DataHandler, so a mark always
        exists for an existing position.)
        """
        equity = self.cash
        for asset, qty in self.positions.items():
            try:
                price = current_prices[asset]
            except KeyError as err:
                raise KeyError(
                    f"update_valuations: no mark price for held asset "
                    f"{asset!r}; positions={list(self.positions)}"
                ) from err
            equity += qty * price
        self.equity = equity

    def get_state(self) -> dict:
        """Return a snapshot dict consumed by ``Backtest.step()``.

        Keys: ``cash``, ``equity``, ``positions`` (shallow copy of the
        internal dict so the caller can't mutate ledger state), and
        ``unrealized_pnl`` (equity − initial_cash; realized vs unrealized
        is not separated in v1 because there are no cost-basis tags on
        positions, per architectural decision IA2).
        """
        return {
            "cash": self.cash,
            "equity": self.equity,
            "positions": dict(self.positions),
            "unrealized_pnl": self.equity - self.initial_cash,
        }

    def reset(self) -> None:
        """Restore to ``initial_cash``; clear all positions and equity."""
        self.cash = self.initial_cash
        self.positions = {}
        self.equity = self.initial_cash
