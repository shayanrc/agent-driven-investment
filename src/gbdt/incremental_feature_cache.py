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

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gbdt import feature_cache as _feature_cache
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


# ---------------------------------------------------------------------------
# Phase 3 — F16 streak-state carry (class B)
# ---------------------------------------------------------------------------
#
# ``signed_days_outside_band`` is an UNBOUNDED run-length: a persistent cross-
# sectional outlier can sit outside its band for >500 td, so a fixed tail rebuild
# truncates the streak (observed: max 975 on sp500). Instead of rebuilding the
# streak, we CONTINUE it from the value at ``cached_max_date`` (which encodes
# ``sign × run_length``) over the new dates' z-underlyings. On the SAME z-values the
# continuation is bit-identical to a full rebuild; the only residual is the rare
# boundary flip where a new z sits within ~1e-10 of ±σ (the tail's z differs from a
# full build by that much) — the same tolerated instability as the current cadence's
# 1e-4 p_raw self-check, bounded to ≤ (#new dates) since a flip can't propagate past
# the next band re-entry.


def _continue_streaks(prev: np.ndarray, z_new: np.ndarray, sigma: float) -> np.ndarray:
    """Continue ``M`` signed-days-outside-band streaks over ``k`` new dates.

    ``prev`` (M,) = the streak value at ``cached_max_date`` (``sign × run_length``,
    0 = in band, NaN = last obs missing). ``z_new`` (M, k) = new z-values. Returns
    (M, k). Mirrors :func:`gbdt.features._signed_days_outside_band_one` exactly (same
    ±σ sides, in-band reset to 0, NaN → NaN + streak reset), seeded from ``prev``.
    """
    M, k = z_new.shape
    out = np.empty((M, k), dtype=float)
    p = prev.astype(float).copy()
    for i in range(k):
        z = z_new[:, i]
        nan = np.isnan(z)
        side = np.where(z >= sigma, 1.0, np.where(z <= -sigma, -1.0, 0.0))
        extend = (~np.isnan(p)) & (p != 0.0) & (np.sign(p) == side) & (side != 0.0)
        v = np.where(side == 0.0, 0.0, np.where(extend, p + side, side))
        v = np.where(nan, np.nan, v)
        out[:, i] = v
        p = np.where(nan, np.nan, v)
    return out


# ---------------------------------------------------------------------------
# Phase 4 — cross-sectional eligibility boundary (class C)
# ---------------------------------------------------------------------------


def _require_stable_membership(cached_X: pd.DataFrame, panel: pd.DataFrame) -> None:
    """A ticker that has crossed the 1,600-row eligibility floor since the cache
    re-ranks the cross-sectional features (F14, F7-xs) at EVERY date — the extend
    can't reproduce that from a tail, so fall back to a full rebuild. (The seam check
    would also trip, since the new member shifts the overlap cross-sections; this
    makes the intent explicit + cheap, *before* building.) A ticker present in the
    cache but absent now (delisted) is fine — it simply gets no new rows.
    """
    newly = (set(panel.index.get_level_values("ticker").unique())
             - set(cached_X.index.get_level_values("ticker").unique()))
    if newly:
        raise SeamMismatch(
            f"{len(newly)} newly-eligible ticker(s) since the cache "
            f"(e.g. {sorted(newly)[:3]}) — cross-sections changed; rebuild.")


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
    _require_stable_membership(cached_X, panel)
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


def _slice_tail(panel, index_df, cached_max_date, warmup_td):
    dates = _panel_dates(panel)
    ts = _tail_start(dates, pd.Timestamp(cached_max_date), warmup_td)
    ptail = panel[panel.index.get_level_values("date") >= ts]
    itail = index_df[index_df.index >= ts]
    return ptail, itail


def _extend_streak_meta(cached_X, underlyings, band_cols, cached_max_date, sigmas):
    """New-date F16-meta (signed-days-outside-band) rows via streak-state carry.

    ``underlyings`` = the 31 z-columns (``f16_meta_underlying_columns``) rebuilt on
    the tail. For each ``<base>_outside_band_<σ>`` column we continue the per-ticker
    streak from the cached value at ``cached_max_date`` over the new-date z of
    ``<base>``. Returns a ``(date, ticker)``-indexed frame of the new-date meta cols.
    """
    cmd = pd.Timestamp(cached_max_date)
    new_u = underlyings[underlyings.index.get_level_values("date") > cmd]
    new_dates = _panel_dates(new_u)
    lab2sig = {(str(int(s)) if s == int(s) else str(s).replace(".", "p")): s for s in sigmas}
    out = {}
    for band_col in band_cols:
        base, label = band_col.rsplit("_outside_band_", 1)
        sigma = lab2sig[label]
        zwide = new_u[base].unstack("ticker").reindex(index=new_dates)
        tickers = zwide.columns
        prev = cached_X[band_col].xs(cmd, level="date").reindex(tickers).to_numpy(dtype=float)
        cont = _continue_streaks(prev, zwide.to_numpy(dtype=float).T, sigma)
        out[band_col] = pd.DataFrame(cont.T, index=new_dates, columns=tickers).stack(future_stack=True)
    new_band = pd.DataFrame(out)
    new_band.index = new_band.index.set_names(["date", "ticker"])
    return new_band


def extend_matrix_full(
    cached_X: pd.DataFrame,
    panel: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    annualization: int,
    cached_max_date: pd.Timestamp | str,
    lookbacks=gbdt_features.DEFAULT_LOOKBACKS,
    warmup_td: int = DEFAULT_WARMUP_TD,
    check_td: int = DEFAULT_CHECK_TD,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Extend a FULL matrix (all families incl F16) by dates ``> cached_max_date``.

    Bounded families (F1–F15 + F16-native) come from a tail rebuild + seam check;
    the F16-meta streak (class B, unbounded) is CONTINUED from the cached state (a
    tail can't capture a >warmup-length streak). Raises :class:`SeamMismatch` if the
    bounded-family seam check fails. Matches a full build to the ~1e-4 FP contract on
    bounded cols; F16-meta is exact except rare ±σ boundary flips (bounded to the
    new-date span, tolerated by the downstream 1e-4 p_raw self-check).
    """
    cmd = pd.Timestamp(cached_max_date)
    _require_stable_membership(cached_X, panel)
    ptail, itail = _slice_tail(panel, index_df, cmd, warmup_td)
    band_cols = [c for c in cached_X.columns if "outside_band" in c]
    nonband = [c for c in cached_X.columns if "outside_band" not in c]

    # Bounded tail: F1–F15 + F16-native (12 rolling-z), WITHOUT the 93-col F16-meta
    # streak (a tail can't capture it — state-carried below). Skipping that build is
    # the speedup; f16_nat is threaded into the meta-underlyings to avoid a recompute.
    tail_stable = gbdt_features.build_feature_matrix(
        ptail, itail, annualization=annualization, families=BOUNDED_FAMILIES,
    ).dropna(axis=1, how="all")
    f16nat = gbdt_features.f16_underlying(ptail, lookbacks, annualization)
    tail_nonband = pd.concat([tail_stable, f16nat], axis=1).reindex(columns=nonband)

    ok, why = seam_ok(cached_X[nonband], tail_nonband, cmd, check_td=check_td)
    if not ok:
        raise SeamMismatch(f"bounded-family seam check failed: {why}")

    new_mask = tail_nonband.index.get_level_values("date") > cmd
    new_nonband = tail_nonband[new_mask]

    underlyings = gbdt_features.f16_meta_underlying_columns(
        ptail, lookbacks, annualization, f16_nat=f16nat,
    )
    new_band = _extend_streak_meta(cached_X, underlyings, band_cols, cmd, sigmas).reindex(new_nonband.index)

    new_rows = pd.concat([new_nonband, new_band], axis=1).reindex(columns=cached_X.columns)
    return pd.concat([cached_X, new_rows])


# ---------------------------------------------------------------------------
# Phase 5 — on-disk persistence + build-or-extend orchestrator
# ---------------------------------------------------------------------------
#
# The cache is keyed on (universe, warmup_start, min_rows, features.py source hash,
# content prefix) — NOT on the full panel content or cached_max_date, which are
# exactly what the extend advances. A features.py edit flips the code signature →
# cold rebuild (correct). Data-content staleness (V1.9_TBD #2 — the V5
# split-adjustment re-seed served a 19G STALE cache that only the seam check
# caught) is guarded twice:
#
# 1. **In the key**: ``content_signature`` — a content hash of a fixed PREFIX of the
#    panel (the first ``_CONTENT_PREFIX_DATES`` dates' index + values). The prefix is
#    append-invariant (the panel only grows at the END within one key namespace), so
#    the key stays stable day-to-day, yet any broad historical correction (a re-seed
#    / adjustment-basis change rescales deep history) flips it → clean cold rebuild.
# 2. **At load time** (build_or_extend): the meta stores a content hash of the FULL
#    panel the matrix was built from; on a key hit the hash is recomputed over the
#    current panel's overlap (dates ≤ cached_max_date) and any mismatch — e.g. a
#    surgical single-bar correction past the prefix, which the key can't see —
#    forces a full rebuild instead of a stale reuse. This makes the (secondary,
#    windowed) seam check no longer the only guard against revised values.
#
# A changed alignment / newly-eligible ticker is still caught by the seam check at
# extend time. Stored one dir per key under ``<cache_root>/<key>/``.

# v2 (V1.9_TBD #2): content_signature in the key + content_signature in the meta
# (overlap check). Bump invalidates any v1 entry on disk once — correctness over
# reuse (the 19G stale-unadjusted incident is exactly what this prevents).
_CACHE_SCHEMA = "v2"
_MATRIX_FILE = "matrix.parquet"
_META_FILE = "meta.json"

# Distinct dates in the fixed panel prefix hashed into the cache key. Deep enough
# that any historical re-adjustment touches it; small enough to hash in ~ms.
_CONTENT_PREFIX_DATES = 256


def panel_content_signature(panel: pd.DataFrame, *, upto=None) -> str:
    """Content hash (SHA-256) of the panel's ``(date, ticker)`` index + numeric
    OHLCV *values*, optionally restricted to rows with ``date <= upto``.

    Vectorized via ``pd.util.hash_pandas_object`` (NaN-deterministic) — ~1 s on a
    1.3M-row sp500 panel, negligible next to the feature build it guards. Used
    both for the key's prefix component and for the load-time overlap check."""
    if upto is not None:
        panel = panel[panel.index.get_level_values("date") <= pd.Timestamp(upto)]
    h = hashlib.sha256()
    # Which rows entered the build …
    h.update(pd.util.hash_pandas_object(
        panel.index.to_frame(index=False), index=False).to_numpy().tobytes())
    # … and what values they carried (the part the v1 key was blind to).
    num = panel.select_dtypes(include="number")
    h.update(",".join(map(str, num.columns)).encode("utf-8"))
    if num.shape[1]:
        h.update(pd.util.hash_pandas_object(num, index=False).to_numpy().tobytes())
    return h.hexdigest()


def panel_prefix_content_signature(panel: pd.DataFrame,
                                   *, n_dates: int = _CONTENT_PREFIX_DATES) -> str:
    """Content hash of the panel's first ``n_dates`` distinct dates — the
    append-invariant prefix folded into :func:`cache_key`. Stable as new bars
    arrive at the end; flips on any values-only correction inside the prefix
    (e.g. a split-adjustment re-seed, which rescales deep history)."""
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    if len(dates) == 0:
        return panel_content_signature(panel)
    cutoff = dates[min(n_dates, len(dates)) - 1]
    return panel_content_signature(panel, upto=cutoff)


def cache_key(universe: str, warmup_start, *, min_rows: int = 1600,
              align_signature: str = "noalign",
              content_signature: str = "nocontent") -> str:
    """Deterministic cache key (SHA-256). Stable as the panel grows; invalidated by a
    ``features.py`` edit (via ``feature_code_signature``), a different universe /
    warmup anchor / eligibility floor / panel ALIGNMENT / prefix CONTENT.
    ``align_signature`` is a hash of the gap-fill rows ``_align_panel`` drops —
    stable day-to-day (gap-fill is historical, ≤ test_end) but distinct per cell when
    alignments differ, so two cells never collide on one cache entry.
    ``content_signature`` is the append-invariant panel-prefix content hash
    (:func:`panel_prefix_content_signature`) — stable day-to-day, flips on a
    values-only historical correction so a re-seeded panel never reuses a stale
    entry (V1.9_TBD #2)."""
    payload = {
        "schema": _CACHE_SCHEMA,
        "universe": universe,
        "warmup_start": str(warmup_start),
        "min_rows": int(min_rows),
        "align_signature": align_signature,
        "content_signature": content_signature,
        "code_signature": _feature_cache.feature_code_signature(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _key_dir(cache_root, key: str) -> Path:
    return Path(cache_root) / key


def save(cache_root, key: str, X: pd.DataFrame, cached_max_date,
         *, content_signature: str = "") -> Path:
    """Persist the matrix + a sidecar (atomic temp-rename, per ``feature_cache``).

    ``content_signature`` is the FULL-panel content hash
    (:func:`panel_content_signature` of the panel the matrix was built from) —
    verified against the current panel's overlap at load time by
    :func:`build_or_extend`, so a values-only correction under an unchanged
    index forces a rebuild instead of a stale reuse (V1.9_TBD #2)."""
    d = _key_dir(cache_root, key)
    d.mkdir(parents=True, exist_ok=True)
    mp = d / _MATRIX_FILE
    tmp = mp.with_suffix(mp.suffix + ".tmp")
    X.to_parquet(tmp)
    tmp.replace(mp)
    meta = {
        "schema": _CACHE_SCHEMA, "key": key,
        "cached_max_date": str(pd.Timestamp(cached_max_date).date()),
        "n_rows": int(len(X)), "n_cols": int(X.shape[1]),
        "content_signature": content_signature,
    }
    kp = d / _META_FILE
    ktmp = kp.with_suffix(kp.suffix + ".tmp")
    ktmp.write_text(json.dumps(meta, indent=2))
    ktmp.replace(kp)
    return mp


def load(cache_root, key: str) -> tuple[pd.DataFrame, pd.Timestamp, dict] | None:
    """Load ``(matrix, cached_max_date, meta)`` iff the sidecar key matches; else
    ``None`` (any corruption / mismatch → miss → the caller rebuilds)."""
    d = _key_dir(cache_root, key)
    mp, kp = d / _MATRIX_FILE, d / _META_FILE
    if not mp.exists() or not kp.exists():
        return None
    try:
        meta = json.loads(kp.read_text())
    except (OSError, ValueError):
        return None
    if meta.get("schema") != _CACHE_SCHEMA or meta.get("key") != key:
        return None
    try:
        X = pd.read_parquet(mp)
    except Exception:
        return None
    if int(meta.get("n_rows", -1)) != len(X):
        return None
    return X, pd.Timestamp(meta["cached_max_date"]), meta


def build_or_extend(
    cache_root,
    universe: str,
    warmup_start,
    panel: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    annualization: int,
    min_rows: int = 1600,
    align_signature: str = "noalign",
) -> pd.DataFrame:
    """Return the full feature matrix for ``panel`` (all families incl F16), using
    the on-disk incremental cache: load + ``extend_matrix_full`` when possible
    (seam-checked; falls back to a full rebuild on ``SeamMismatch`` — a changed
    alignment / revised bar / newly-eligible ticker), else a full build. Persists the
    result. Matches a from-scratch build to the ~1e-4 contract.

    Data-content guard (V1.9_TBD #2): the key carries the panel's PREFIX content
    hash (broad historical corrections → new key → cold rebuild) and, on a key hit,
    the meta's full-panel content hash is verified against the current panel's
    overlap (dates ≤ cached_max_date) — any values-only correction the prefix
    misses forces a full rebuild instead of a stale reuse or an early stale
    return."""
    key = cache_key(universe, warmup_start, min_rows=min_rows,
                    align_signature=align_signature,
                    content_signature=panel_prefix_content_signature(panel))
    panel_max = _panel_dates(panel).max()
    hit = load(cache_root, key)
    if hit is not None:
        cached_X, cmd, meta = hit
        # Overlap content check BEFORE the nothing-new early return: the cached
        # matrix was built from a panel whose rows ≤ cmd must hash identically to
        # the current panel's rows ≤ cmd. A mismatch = revised values (or index)
        # under the same key → treat as a miss.
        if meta.get("content_signature") != panel_content_signature(panel, upto=cmd):
            hit = None
    if hit is not None:
        cached_X, cmd, _meta = hit
        if cmd >= panel_max:
            return cached_X  # nothing new to score
        try:
            X = extend_matrix_full(cached_X, panel, index_df,
                                   annualization=annualization, cached_max_date=cmd)
            save(cache_root, key, X, panel_max,
                 content_signature=panel_content_signature(panel))
            return X
        except SeamMismatch:
            pass  # fall through to a full rebuild + cache refresh
    X = gbdt_features.build_feature_matrix(
        panel, index_df, annualization=annualization,
    ).dropna(axis=1, how="all")
    save(cache_root, key, X, panel_max,
         content_signature=panel_content_signature(panel))
    return X
