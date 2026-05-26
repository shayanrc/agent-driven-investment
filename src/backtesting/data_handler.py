"""DataHandler — timekeeping, activity masks, and window slicing.

Per ``spec.md`` § 4.1 + D3. Single source of truth for "what step are we
on?" v1 is single-frequency-per-engine: every asset in every feed shares
the same trading calendar; multi-frequency support is deferred to v1.1
(Q3). Internal-gap handling is governed by the engine-level ``gap_policy``
constructor arg (Q9).

The class is deliberately small: timeline construction + reindex happens
once in ``__init__`` via ``utils.reindex_asset_to_timeline``; everything
else is integer indexing on the resulting numpy panels. The 8-method
surface (``advance_time``, ``get_current_bar``, ``get_window``,
``get_price``, ``is_active``, ``get_active_assets``, ``reset``, and the
constructor) is exactly what ``Backtest`` and ``ExecutionBroker`` need;
no backdoor accessors that bypass the timeline contract.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from backtesting.utils import (
    GapPolicy,
    build_master_timeline,
    reindex_asset_to_timeline,
    slice_window,
)


class DataHandler:
    """Timekeeping + windowing across a single-frequency, multi-asset feed set.

    Parameters
    ----------
    data_feeds:
        Nested dict ``{feed_name: {asset_name: DataFrame}}``. Each DataFrame
        must be indexed by a DatetimeIndex-compatible index and carry
        whatever columns the feed's schema dictates (e.g., OHLCV for
        equities). All assets share a single trading calendar in v1.
    lookback:
        Number of historical bars to include in the state window.
        ``current_step`` starts at ``lookback`` so ``get_window()`` returns
        a full-length window from the first call.
    gap_policy:
        How to handle missing dates *inside* an asset's active range. See
        ``utils.reindex_asset_to_timeline`` for semantics.
    """

    def __init__(
        self,
        data_feeds: dict[str, dict[str, pd.DataFrame]],
        lookback: int,
        gap_policy: GapPolicy = "ffill_zero_volume",
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if gap_policy not in ("raise", "ffill_zero_volume"):
            raise ValueError(
                f"gap_policy must be one of 'raise' or 'ffill_zero_volume', "
                f"got {gap_policy!r}"
            )
        if not data_feeds:
            raise ValueError("data_feeds must contain at least one feed")
        for feed_name, assets in data_feeds.items():
            if not assets:
                raise ValueError(
                    f"feed {feed_name!r} contains no assets"
                )

        self.lookback: int = int(lookback)
        self.gap_policy: GapPolicy = gap_policy

        # Build master timeline as union of all asset indices across all
        # feeds (D3: single-frequency, shared trading calendar).
        all_frames: list[pd.DataFrame] = []
        for assets in data_feeds.values():
            for frame in assets.values():
                all_frames.append(frame)
        self.timeline: pd.DatetimeIndex = build_master_timeline(all_frames)
        self.max_steps: int = len(self.timeline)

        if self.max_steps <= self.lookback:
            raise ValueError(
                f"timeline length ({self.max_steps}) must exceed "
                f"lookback ({self.lookback})"
            )

        # Reindex every asset, capturing per-asset active mask.
        # ``self.data[feed][asset]`` is a reindexed DataFrame on the timeline.
        # ``self.active[feed][asset]`` is a bool ndarray of length max_steps.
        self.data: dict[str, dict[str, pd.DataFrame]] = {}
        self.active: dict[str, dict[str, np.ndarray]] = {}
        for feed_name, assets in data_feeds.items():
            self.data[feed_name] = {}
            self.active[feed_name] = {}
            for asset_name, frame in assets.items():
                reindexed, mask = reindex_asset_to_timeline(
                    frame, self.timeline, gap_policy
                )
                self.data[feed_name][asset_name] = reindexed
                self.active[feed_name][asset_name] = mask

        # current_step starts at lookback so get_window() always returns a
        # full-length window (spec.md § 4.1 design note + B1 invariant).
        self.current_step: int = self.lookback

    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------
    def advance_time(self) -> bool:
        """Advance the clock by one bar.

        Q8 / B7 contract: **no-mutate when done**. If the next step would
        exceed the timeline, return ``True`` (done) WITHOUT incrementing
        ``current_step``. Otherwise increment and return ``False``.

        Invariant: ``current_step ∈ [lookback, max_steps - 1]`` for the
        entire engine lifetime.
        """
        if self.current_step + 1 >= self.max_steps:
            return True
        self.current_step += 1
        return False

    def reset(self) -> None:
        """Reset ``current_step`` to ``lookback``."""
        self.current_step = self.lookback

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get_current_bar(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return data for exactly the current bar (one row per asset).

        Shape:
            ``{feed_name: {asset_name: {column_name: value}}}``

        Used by the broker for fill-price lookup and by the orchestrator
        for the mark-to-market price snapshot.
        """
        out: dict[str, dict[str, dict[str, float]]] = {}
        step = self.current_step
        for feed_name, assets in self.data.items():
            feed_out: dict[str, dict[str, float]] = {}
            for asset_name, frame in assets.items():
                row = frame.iloc[step]
                feed_out[asset_name] = row.to_dict()
            out[feed_name] = feed_out
        return out

    def get_window(self) -> dict[str, dict[str, np.ndarray]]:
        """Return the state window for the current step.

        Shape:
            ``{feed_name: {asset_name: ndarray(lookback, n_columns)}}``

        Returns exactly ``lookback`` rows ending at and including
        ``current_step``. Pure integer-position slicing on the reindexed
        panel — by construction, no row at index > current_step appears
        in the returned ndarrays (B1 enforced structurally, not by
        convention).
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for feed_name, assets in self.data.items():
            feed_out: dict[str, np.ndarray] = {}
            for asset_name, frame in assets.items():
                feed_out[asset_name] = slice_window(
                    frame, self.current_step, self.lookback
                )
            out[feed_name] = feed_out
        return out

    def get_price(self, asset: str, field: str = "close") -> float:
        """Return one column's value for one asset at ``current_step``.

        Searches across all feeds for the named asset. Raises KeyError
        if not found in any feed (programming error — caller passed an
        unknown asset).
        """
        for feed_name, assets in self.data.items():
            if asset in assets:
                value = assets[asset].iloc[self.current_step][field]
                return float(value)
        raise KeyError(
            f"get_price: asset {asset!r} not found in any feed; "
            f"feeds={list(self.data)}"
        )

    def get_current_timestamp(self) -> pd.Timestamp:
        """The calendar timestamp corresponding to ``current_step``."""
        return self.timeline[self.current_step]

    # ------------------------------------------------------------------
    # Activity masks
    # ------------------------------------------------------------------
    def is_active(self, asset: str) -> bool:
        """True if the asset is tradeable at the current step.

        An asset is tradeable on dates inside ``[first_obs, last_obs]``
        of its source frame. Pre-IPO and post-delisting dates are inactive.
        The broker uses this for buy-rejection / sell-permitting logic.
        """
        for assets in self.active.values():
            if asset in assets:
                return bool(assets[asset][self.current_step])
        raise KeyError(
            f"is_active: asset {asset!r} not found in any feed"
        )

    def get_active_assets(self) -> set[str]:
        """Return the set of asset names tradeable at the current step.

        Useful as the validation surface in ``ExecutionBroker.submit_orders``.
        """
        step = self.current_step
        out: set[str] = set()
        for assets in self.active.values():
            for asset_name, mask in assets.items():
                if mask[step]:
                    out.add(asset_name)
        return out

    def get_known_assets(self) -> set[str]:
        """Return all asset names across all feeds (active or not).

        Used by ``ExecutionBroker.submit_orders`` to distinguish
        "unknown asset" (validation rejection) from "untradeable asset"
        (a separate rejection bucket).
        """
        out: set[str] = set()
        for assets in self.data.values():
            for asset_name in assets:
                out.add(asset_name)
        return out

    def feeds_for_asset(self, asset: str) -> Iterable[str]:
        """Return the feed name(s) that carry the given asset."""
        for feed_name, assets in self.data.items():
            if asset in assets:
                yield feed_name
