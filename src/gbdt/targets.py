"""Binary target construction for gbdt v1.

One spec → one target: ``(direction, threshold_pct, horizon_days)`` and an
optional ``max_drawdown`` path-honesty filter.

Semantics (per ``docs/gbdt/EXPERIMENT_SPEC.md`` § "target"):

**Simple binary mode (no max_drawdown):**
- ``direction=up``  → 1 iff ``max(high[t+1:t+horizon]) >= close[t] * (1 + threshold)``.
- ``direction=down`` → 1 iff ``min(low[t+1:t+horizon])  <= close[t] * (1 - threshold)``.
- Uses HIGH (up) / LOW (down) — the most aggressive intraday extreme.

**Path-honesty mode (max_drawdown set):**
- Scan ``(t, t+horizon]`` for first ``t_breach`` with ``close`` clearing the
  threshold. Switches breach metric from HIGH/LOW to **CLOSE** intentionally
  (operator semantics; see EXPERIMENT_SPEC § "Breach criterion switches…").
- If a breach exists AND the path-excursion bound is honored before breach,
  label 1; else label 0.

Rows in the last ``horizon`` rows per ticker (where forward data is incomplete)
are labelled ``NaN`` and excluded from training; predictions can still be
emitted at those rows downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_target(
    panel: pd.DataFrame,
    *,
    direction: str,
    threshold_pct: float,
    horizon_days: int,
    max_drawdown: float | None = None,
) -> pd.Series:
    """Return a ``0/1/NaN`` Series indexed on the panel's ``(date, ticker)``."""
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    if threshold_pct <= 0:
        raise ValueError(f"threshold_pct must be > 0, got {threshold_pct}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days}")
    if max_drawdown is not None and not (0 < max_drawdown < 1):
        raise ValueError(
            f"max_drawdown must be in (0, 1), got {max_drawdown}"
        )

    threshold = float(threshold_pct) / 100.0

    tickers = panel.index.get_level_values("ticker").unique()
    chunks = []
    for t in tickers:
        sub = panel.xs(t, level="ticker")[["high", "low", "close"]]
        y = _build_target_one_ticker(
            sub, direction=direction, threshold=threshold,
            horizon=horizon_days, max_drawdown=max_drawdown,
        )
        y.index = pd.MultiIndex.from_product([y.index, [t]], names=["date", "ticker"])
        chunks.append(y)

    out = pd.concat(chunks).sort_index()
    # Reindex onto the panel's full (date, ticker) MultiIndex so missing rows
    # stay aligned (in practice they should all match because we used the panel
    # directly, but explicit reindex makes the contract clear).
    out = out.reindex(panel.index)
    return out.rename("target")


def _build_target_one_ticker(
    df: pd.DataFrame,
    *,
    direction: str,
    threshold: float,
    horizon: int,
    max_drawdown: float | None,
) -> pd.Series:
    """Vectorized-ish per-ticker target builder.

    Approach: for each origin row t, look at the forward window
    ``(t, t+horizon]`` (using positional indexing — ``df`` is sorted by date
    per ticker).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    for t in range(n - horizon):
        c0 = close[t]
        # Forward window indices: t+1 .. t+horizon (inclusive)
        end = t + horizon + 1
        fwd_close = close[t + 1: end]
        fwd_high = high[t + 1: end]
        fwd_low = low[t + 1: end]

        if direction == "up":
            target_level = c0 * (1.0 + threshold)
            if max_drawdown is None:
                # Simple binary: HIGH-based breach check
                out[t] = 1.0 if fwd_high.max() >= target_level else 0.0
            else:
                # Path-honesty: CLOSE-based breach + close-excursion floor
                drawdown_floor = c0 * (1.0 - max_drawdown)
                # First index of breach on CLOSE
                hits = np.where(fwd_close >= target_level)[0]
                if hits.size == 0:
                    out[t] = 0.0
                else:
                    first = hits[0]
                    # Path between t+1 and t+1+first (inclusive of breach)
                    path = fwd_close[: first + 1]
                    if path.min() > drawdown_floor:
                        out[t] = 1.0
                    else:
                        out[t] = 0.0
        else:  # direction == "down"
            target_level = c0 * (1.0 - threshold)
            if max_drawdown is None:
                out[t] = 1.0 if fwd_low.min() <= target_level else 0.0
            else:
                rally_ceiling = c0 * (1.0 + max_drawdown)
                hits = np.where(fwd_close <= target_level)[0]
                if hits.size == 0:
                    out[t] = 0.0
                else:
                    first = hits[0]
                    path = fwd_close[: first + 1]
                    if path.max() < rally_ceiling:
                        out[t] = 1.0
                    else:
                        out[t] = 0.0

    return pd.Series(out, index=df.index)


__all__ = ["build_target"]
