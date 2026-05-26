"""Shared helpers for the backtesting engine.

This module is extended across stages:
- Stage 2 adds timeline / NaN-aware reindex / gap-policy primitives.
- Stage 4 adds the action parser (``parse_action``) and lot snapper
  (``snap_to_lot``).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

GapPolicy = Literal["raise", "ffill_zero_volume"]


def build_master_timeline(
    asset_frames: Iterable[pd.DataFrame],
) -> pd.DatetimeIndex:
    """Return the sorted, deduplicated union of the DataFrames' indices.

    All DataFrames must have a ``DatetimeIndex``-compatible index. The
    timeline is the single source of truth for the engine's clock
    (``spec.md`` D3); every per-asset frame is reindexed onto it.
    """
    union: pd.Index | None = None
    for frame in asset_frames:
        idx = pd.DatetimeIndex(frame.index)
        if union is None:
            union = idx
        else:
            union = union.union(idx)
    if union is None or len(union) == 0:
        raise ValueError(
            "build_master_timeline: no data frames provided or all empty"
        )
    # Sort + dedupe (``union`` already dedupes).
    return pd.DatetimeIndex(sorted(union))


def reindex_asset_to_timeline(
    frame: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    gap_policy: GapPolicy,
    volume_column: str = "volume",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Reindex a single asset's frame to the master timeline.

    Returns ``(reindexed_frame, active_mask)`` where ``active_mask`` is a
    boolean ndarray of length ``len(timeline)``, True where the asset is
    tradeable on that date.

    Mid-series-start handling:
        Dates before the asset's first observation get NaN rows and
        ``active=False`` (the asset hasn't IPO'd yet; the broker rejects
        all orders).

    Mid-series-end handling:
        Dates after the asset's last observation get forward-filled price
        columns (so existing positions still mark to market) and
        ``active=False`` — the broker rejects buys but permits sells at
        the last-known close (so callers can liquidate phantom positions).

    Internal-gap handling (within the asset's active range):
        ``gap_policy="raise"`` — any missing date inside the active range
        fails construction with a clear error.
        ``gap_policy="ffill_zero_volume"`` (default) — forward-fill price
        columns from the previous bar; zero the volume column. The gap
        day becomes a valid bar with zero traded volume. ``active=True``
        on the gap day (treat as "no trading happened" but still
        tradeable from the engine's perspective).
    """
    if frame.empty:
        raise ValueError(
            "reindex_asset_to_timeline: cannot reindex an empty frame"
        )

    src_idx = pd.DatetimeIndex(frame.index)
    first_obs = src_idx.min()
    last_obs = src_idx.max()

    # Detect internal gaps (timeline dates inside [first_obs, last_obs]
    # that have no source observation).
    active_range_mask = (timeline >= first_obs) & (timeline <= last_obs)
    active_range_dates = timeline[active_range_mask]
    src_set = set(src_idx)
    internal_gaps = [d for d in active_range_dates if d not in src_set]

    if gap_policy == "raise" and internal_gaps:
        raise ValueError(
            f"Internal gap(s) detected for asset (gap_policy='raise'): "
            f"{len(internal_gaps)} missing date(s) inside active range; "
            f"first gap = {internal_gaps[0]}"
        )

    # Reindex; pre-IPO and post-delisting rows initially NaN.
    reindexed = frame.reindex(timeline)

    # Internal-gap fill (only inside the active range, only for
    # ffill_zero_volume mode — "raise" already returned above).
    if gap_policy == "ffill_zero_volume" and internal_gaps:
        gap_positions = timeline.get_indexer(internal_gaps)
        # Forward-fill price columns INSIDE the active range only. We do
        # this by ffilling on a slice and writing back. ffill() over the
        # full reindexed frame would also fill the post-delisting tail,
        # which we want — but it would also "fill" the pre-IPO leading
        # NaN with nothing (no source) so that branch is safe. We do the
        # active-range slice explicitly to keep semantics clear and so
        # the post-delisting forward-fill is applied as a separate step
        # below.
        active_slice = reindexed.loc[active_range_dates].ffill()
        reindexed.loc[active_range_dates] = active_slice
        # Zero the volume column on gap days only (not on real bars).
        if volume_column in reindexed.columns:
            reindexed.loc[internal_gaps, volume_column] = 0

    # Post-delisting forward-fill: extend last known values past last_obs.
    post_mask = timeline > last_obs
    if post_mask.any():
        # Forward-fill from last_obs row through end of timeline.
        last_row = reindexed.loc[last_obs]
        reindexed.loc[post_mask] = last_row.values

    # Active mask: True between first_obs and last_obs (inclusive),
    # False before first_obs (pre-IPO; rows are NaN) and after last_obs
    # (delisted; rows are forward-filled but untradeable for buys).
    active_mask = np.asarray(active_range_mask, dtype=bool)

    return reindexed, active_mask


def slice_window(
    frame: pd.DataFrame,
    end_step: int,
    lookback: int,
) -> np.ndarray:
    """Return ``frame.iloc[end_step - lookback + 1 : end_step + 1]`` as a
    numpy array of shape ``(lookback, n_columns)``.

    The slice is exactly ``lookback`` bars ending at and including
    ``end_step``. Pure integer-position slicing — no date arithmetic.
    """
    start = end_step - lookback + 1
    if start < 0:
        raise ValueError(
            f"slice_window: end_step={end_step} lookback={lookback} "
            f"would slice below 0"
        )
    return frame.iloc[start : end_step + 1].to_numpy(copy=False)


# ---------------------------------------------------------------------------
# Stage 4 — action parsing + lot snapping
# ---------------------------------------------------------------------------
def snap_to_lot(qty: float, lot_size: int) -> float:
    """Truncate ``qty`` toward zero to the nearest valid lot.

    - ``lot_size == 0`` ⇒ fractional, no rounding (return qty unchanged).
    - ``lot_size == 1`` ⇒ whole units: truncate toward zero (``2.7 → 2``,
      ``-2.7 → -2``).
    - ``lot_size == N`` for any positive int ⇒ ``floor(abs(qty)/N)*N*sign(qty)``.

    Raises ``ValueError`` for negative lot sizes.
    """
    if lot_size < 0:
        raise ValueError(f"lot_size must be >= 0, got {lot_size}")
    if lot_size == 0:
        return float(qty)
    if qty == 0:
        return 0.0
    sign = 1.0 if qty > 0 else -1.0
    magnitude = math.floor(abs(qty) / lot_size) * lot_size
    return float(sign * magnitude)


def _lookup_lot_size(
    asset: str, lot_sizes: dict[str, int], default_lot_size: int
) -> int:
    return lot_sizes.get(asset, default_lot_size)


def parse_action(
    action: dict[str, Any] | None,
    portfolio,  # backtesting.portfolio.Portfolio
    data_handler,  # backtesting.data_handler.DataHandler
    lot_sizes: dict[str, int],
    default_lot_size: int,
    price_column: str = "close",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Translate an action dict into a list of normalized orders.

    Returns ``(orders, lot_size_audit)`` where ``orders`` is the list to
    submit to the broker (with ``{asset, qty}`` only — no extra fields,
    per Q6) and ``lot_size_audit`` is the Q10 dict mapping
    ``asset → {"requested_qty": float, "filled_qty": int}`` for every
    asset where the snap changed the requested quantity (including
    snap-to-zero).

    Action types:
    - ``None`` ⇒ ``([], {})``. No-op.
    - ``{"type": "order", "orders": [...]}``: each order has fields
      ``{asset, qty}`` only (extra fields raise via the broker validator
      at submit time, not here — this function is shape-permissive on
      input so callers can also pass typed dicts that we'll trim).
    - ``{"type": "weight", "target_weights": {...}}``: compute deltas
      from pre-fill equity / positions, snap to lot, sequence sells
      before buys. Raises ``ValueError`` if the weights sum > 1.0 (Q1).

    Orders with snapped qty == 0 are dropped (no fill possible) but
    still recorded in ``lot_size_audit`` with ``filled_qty: 0`` (Q10
    unified key).
    """
    if action is None:
        return [], {}
    if not isinstance(action, dict):
        raise ValueError(f"action must be dict or None, got {type(action).__name__}")
    action_type = action.get("type")
    if action_type == "order":
        return _parse_order_action(action, lot_sizes, default_lot_size)
    if action_type == "weight":
        return _parse_weight_action(
            action,
            portfolio,
            data_handler,
            lot_sizes,
            default_lot_size,
            price_column,
        )
    raise ValueError(
        f"action['type'] must be 'order' or 'weight', got {action_type!r}"
    )


def _parse_order_action(
    action: dict[str, Any],
    lot_sizes: dict[str, int],
    default_lot_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    orders_in = action.get("orders")
    if not isinstance(orders_in, list):
        raise ValueError("action['orders'] must be a list")
    out_orders: list[dict[str, Any]] = []
    audit: dict[str, dict[str, float]] = {}
    for order in orders_in:
        if not isinstance(order, dict) or "asset" not in order or "qty" not in order:
            raise ValueError(f"malformed order entry: {order!r}")
        # Reject extra fields up front (Q6) — the broker would catch
        # this too, but failing early gives the caller a cleaner trace.
        extra = set(order.keys()) - {"asset", "qty"}
        if extra:
            raise ValueError(
                f"order has unexpected fields {sorted(extra)} "
                f"(v1 schema is {{asset, qty}} only — Q6)"
            )
        asset = order["asset"]
        requested = float(order["qty"])
        lot = _lookup_lot_size(asset, lot_sizes, default_lot_size)
        snapped = snap_to_lot(requested, lot)
        if snapped != requested:
            audit[asset] = {
                "requested_qty": requested,
                "filled_qty": snapped,
            }
        if snapped != 0:
            out_orders.append({"asset": asset, "qty": snapped})
    return out_orders, audit


def _parse_weight_action(
    action: dict[str, Any],
    portfolio,
    data_handler,
    lot_sizes: dict[str, int],
    default_lot_size: int,
    price_column: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    target_weights = action.get("target_weights")
    if not isinstance(target_weights, dict):
        raise ValueError("action['target_weights'] must be a dict")
    total = sum(target_weights.values())
    # Q1: raise on over-allocation; caller must normalize.
    if total > 1.0 + 1e-12:
        raise ValueError(
            f"sum(target_weights) = {total} exceeds 1.0; "
            "caller must normalize (Q1)"
        )

    # Pre-fill equity using close prices at the current step.
    positions = dict(portfolio.positions)
    price_lookup: dict[str, float] = {}
    relevant = set(target_weights) | set(positions)
    for asset in relevant:
        try:
            price_lookup[asset] = data_handler.get_price(asset, price_column)
        except KeyError as err:
            raise ValueError(
                f"weight action references unknown asset {asset!r}"
            ) from err
    # Pre-fill equity = cash + Σ pos * price.
    equity = portfolio.cash
    for asset, qty in positions.items():
        equity += qty * price_lookup[asset]

    audit: dict[str, dict[str, float]] = {}
    out_orders: list[dict[str, Any]] = []

    # Compute deltas for every asset that's either targeted or currently
    # held. Assets dropped from target_weights ⇒ target_weight 0 ⇒
    # sell-to-zero.
    for asset in relevant:
        target_w = target_weights.get(asset, 0.0)
        price = price_lookup[asset]
        # If price is NaN or 0, can't compute a meaningful target.
        if price <= 0 or price != price:  # NaN check
            # Skip silently; broker will reject as untradeable when it
            # tries to look up the price at fill time.
            continue
        target_qty = (target_w * equity) / price
        current_qty = positions.get(asset, 0.0)
        delta = target_qty - current_qty
        lot = _lookup_lot_size(asset, lot_sizes, default_lot_size)
        snapped = snap_to_lot(delta, lot)
        if snapped != delta:
            audit[asset] = {
                "requested_qty": delta,
                "filled_qty": snapped,
            }
        if snapped != 0:
            out_orders.append({"asset": asset, "qty": snapped})

    # Sells (qty < 0) before buys (qty > 0) so cash freed by sells funds
    # subsequent buys (IA5 in V1_PLAN: ordering happens here AND in the
    # broker — intentional duplication for now).
    out_orders.sort(key=lambda o: o["qty"] > 0)
    return out_orders, audit
