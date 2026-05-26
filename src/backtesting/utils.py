"""Shared helpers for the backtesting engine.

This module is extended across stages:
- Stage 2 adds timeline / NaN-aware reindex / gap-policy primitives.
- Stage 4 adds the action parser (``parse_action``) and lot snapper
  (``snap_to_lot``).
"""

from __future__ import annotations

from typing import Iterable, Literal

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
