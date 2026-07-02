"""V1.6 Phase 2 — date-extendable feature-matrix cache (bounded-lookback families).

The daily `/daily-predictions` cadence rebuilds the whole 7–10-year feature matrix
every run to emit one new day. Because the features are *causal rolling* stats, a
historical row's value does not change when a new bar arrives — only the last
~max-lookback rows of each rolling window do. So we can **freeze the cached matrix
and rebuild only a bounded TAIL of the panel**, then stitch the new dates on.

This module implements the correctness core of that extend for the **bounded-lookback
families (F1–F15)**:

- ``build_tail`` — rebuild features on ``[cached_max_date − warmup_td td .. end]``.
  ``warmup_td`` must exceed the deepest *nested* rolling lookback (vol-of-vol and
  the F16-native z-scores are ``rolling(N)`` of a ``rolling(N)`` stat → ~2× the max
  ``DEFAULT_LOOKBACKS``=200 → ~400 td; the default 500 adds a margin).
- ``seam_ok`` — the **seam-integrity check**: recompute the last ``check_td`` cached
  dates from the tail and require them **bit-identical** to the cached matrix. A
  mismatch means the warmup was too short OR a provider revised a historical bar
  (the `_006`/`_007` gap-fill class) — either way the caller must full-rebuild.
- ``extend_matrix`` — build tail → seam check → append rows ``> cached_max_date``.

**Out of Phase-2 scope (raise/rebuild, not silently wrong):**
- **F16 streak (class B)** — ``signed_days_outside_band`` is an *unbounded* run-length;
  a tail rebuild starts mid-streak and diverges. Handled in Phase 3 via streak-state
  carry. Callers pass ``families`` WITHOUT F16 here (see ``BOUNDED_FAMILIES``).
- **Cross-sectional eligibility boundary (class C)** — a ticker newly crossing the
  1,600-row floor re-ranks past cross-sections. The common case (stable membership
  over the extend window) is bit-identical here; the boundary is Phase 4.

**Contract: matches the full build to ~1e-13, not bit-for-bit.** The extend
recomputes rolling stats over a different series offset, and pandas' online
rolling accumulation (mean/std/var/cov/corr/skew/kurt) rounds differently by
offset — so those columns match to ~1e-13 (max/min/shift ARE exact). That is 9
orders below the cadence's 1e-4 self-check and has zero effect on the tree model.
The seam check (with a tight tolerance) is the runtime guard against a too-short
warmup or a real data revision (which moves values by >> the FP noise); the parity
test (``tests/gbdt/test_incremental_feature_cache.py``) is the CI guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbdt import features as gbdt_features

# Bounded-lookback families safe to extend on a tail (everything except F16, whose
# streak meta-layer carries unbounded state — Phase 3).
BOUNDED_FAMILIES: list[str] = [f for f in gbdt_features._ALL_FAMILIES if f != "F16"]

# Trading days of warmup kept before ``cached_max_date`` when rebuilding the tail.
# > the ~400-td deepest nested rolling depth; the seam check verifies sufficiency
# empirically on every extend (a too-short tail fails the check → full rebuild).
DEFAULT_WARMUP_TD = 500

# Cached dates recomputed from the tail and checked against the cache.
DEFAULT_CHECK_TD = 20

# The extend recomputes rolling stats over a different series OFFSET than the full
# build, and pandas' rolling mean/std/var/cov/corr/skew/kurt use an ONLINE running
# sum that accumulates FP rounding from the series start — so the tail is NOT
# bit-for-bit. Most features match to ~1e-10; the numerically-unstable rolling
# HIGHER MOMENTS (returns_skew / returns_kurt — 3rd/4th moments, catastrophic
# cancellation) reach ~1e-4 worst-case (kurt_5). Even that is at/below the cadence's
# 1e-4 p_raw self-check and is a 0.007%-of-one-feature perturbation with no effect
# on the tree model. rtol=1e-3 passes the kurt FP noise, yet a real data revision
# moves the many FP-STABLE features (returns/vols, ~1e-10) by >> rtol*|v| so it is
# still caught. max/min/shift features are exact. See the Phase-2 memo.
SEAM_RTOL = 1e-3
SEAM_ATOL = 1e-6


class SeamMismatch(RuntimeError):
    """The tail recompute diverged from the cached matrix on the overlap — the
    warmup was too short or a historical bar was revised. The caller must rebuild."""


def _panel_dates(obj: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(obj.index.get_level_values("date").unique()).sort_values()


def _tail_start(dates: pd.DatetimeIndex, cached_max_date: pd.Timestamp, warmup_td: int) -> pd.Timestamp:
    """The trading date ``warmup_td`` sessions before ``cached_max_date`` (clamped to
    the earliest available date)."""
    prior = dates[dates <= cached_max_date]
    if len(prior) <= warmup_td:
        return dates[0]
    return prior[-(warmup_td + 1)]


def build_tail(
    panel: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    annualization: int,
    families: list[str],
    cached_max_date: pd.Timestamp,
    warmup_td: int = DEFAULT_WARMUP_TD,
) -> pd.DataFrame:
    """``build_feature_matrix`` on the tail ``[cached_max_date − warmup_td td .. end]``.

    Slicing an ALREADY-LOADED full panel (not ``load_panel`` with a short window,
    which the 1,600-row eligibility floor would empty) keeps the correct eligible
    ticker set while paying the build only on the tail rows.
    """
    dates = _panel_dates(panel)
    ts = _tail_start(dates, pd.Timestamp(cached_max_date), warmup_td)
    ptail = panel[panel.index.get_level_values("date") >= ts]
    itail = index_df[index_df.index >= ts]
    X = gbdt_features.build_feature_matrix(
        ptail, itail, annualization=annualization, families=families,
    )
    return X.dropna(axis=1, how="all")


def seam_ok(
    cached_X: pd.DataFrame,
    tail_X: pd.DataFrame,
    cached_max_date: pd.Timestamp,
    *,
    check_td: int = DEFAULT_CHECK_TD,
) -> tuple[bool, str]:
    """Bit-identity of the last ``check_td`` cached dates vs their tail recompute."""
    cdates = _panel_dates(cached_X)
    overlap = cdates[cdates <= pd.Timestamp(cached_max_date)][-check_td:]
    if len(overlap) == 0:
        return False, "no overlap dates"
    a = cached_X[cached_X.index.get_level_values("date").isin(overlap)]
    b = tail_X.reindex(a.index)  # align tail rows to the cached (date,ticker) order
    missing = [c for c in a.columns if c not in b.columns]
    if missing:
        return False, f"tail missing {len(missing)} cached column(s), e.g. {missing[:3]}"
    b = b[a.columns]
    if not np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                       rtol=SEAM_RTOL, atol=SEAM_ATOL, equal_nan=True):
        return False, "value/NaN mismatch on overlap (beyond FP tolerance)"
    return True, "ok"


def extend_matrix(
    cached_X: pd.DataFrame,
    panel: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    annualization: int,
    families: list[str],
    cached_max_date: pd.Timestamp | str,
    warmup_td: int = DEFAULT_WARMUP_TD,
    check_td: int = DEFAULT_CHECK_TD,
) -> pd.DataFrame:
    """Return ``cached_X`` extended with the panel's dates ``> cached_max_date``,
    bit-identical to a full rebuild. Raises :class:`SeamMismatch` if the seam check
    fails (the caller falls back to a full rebuild + cache refresh)."""
    cached_max_date = pd.Timestamp(cached_max_date)
    tail_X = build_tail(
        panel, index_df, annualization=annualization, families=families,
        cached_max_date=cached_max_date, warmup_td=warmup_td,
    )
    ok, why = seam_ok(cached_X, tail_X, cached_max_date, check_td=check_td)
    if not ok:
        raise SeamMismatch(f"incremental extend seam check failed: {why}")
    new_rows = tail_X[tail_X.index.get_level_values("date") > cached_max_date]
    new_rows = new_rows.reindex(columns=cached_X.columns)  # exact cached column set/order
    return pd.concat([cached_X, new_rows])
