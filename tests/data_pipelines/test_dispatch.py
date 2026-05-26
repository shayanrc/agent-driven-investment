"""Stage 4 tests: dispatch.fetch() with fake adapters.

Covers the routing matrix: cold cache → seed; small gap → first update;
big gap → seed; update fail → fallback; all fail → AllProvidersFailed;
unknown prefix → UnknownDomain.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from data_pipelines import fetch, fetch_with_meta
from data_pipelines.adapter import Adapter
from data_pipelines.domain import DomainRegistry
from data_pipelines.errors import (
    AllProvidersFailed,
    EmptyPayload,
    ProviderError,
    UnknownDomain,
)
from data_pipelines.raw_store import write_raw_atomic

from .conftest import FakeCalendar, FakeDomain, make_df


# ---------------------------------------------------------------------------
# Test adapters
# ---------------------------------------------------------------------------

class _ScriptedAdapter(Adapter):
    """Adapter that returns a pre-set DataFrame for any call.

    Tracks all (identifier, start, end) calls in `.calls` for assertions.
    """

    def __init__(self, name: str, df: pd.DataFrame, tmp_root: Path,
                 extra_meta: dict | None = None):
        self.name = name
        self._df = df
        self._tmp_root = tmp_root  # retained for legacy fixture compat; data_root wins
        self.calls: list[tuple[str, date | None, date | None]] = []
        self.extra_meta = extra_meta or {}
        self._counter = 0

    def fetch(self, identifier, start=None, end=None, *, data_root):
        self.calls.append((identifier, start, end))
        self._counter += 1
        ts = datetime(2026, 5, 23, 14, 30, self._counter, tzinfo=timezone.utc)
        rs = start or date(2020, 1, 1)
        re_ = end or date(2026, 12, 31)
        return write_raw_atomic(
            data_root, self.name, "fake", "FAKE", "X",
            payload=b"raw", range_start=rs, range_end=re_, ext="csv",
            timestamp=ts,
        )

    def parse(self, raw_path):
        return self._df.copy()


class _RaisingAdapter(Adapter):
    """Adapter that always raises a specified error on fetch."""

    def __init__(self, name: str, exc: Exception):
        self.name = name
        self._exc = exc
        self.calls: list = []

    def fetch(self, identifier, start=None, end=None, *, data_root):
        self.calls.append((identifier, start, end))
        raise self._exc

    def parse(self, raw_path):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def _register(*adapters, threshold: int = 10):
    dom = FakeDomain(adapters=list(adapters), big_gap_threshold=threshold)
    DomainRegistry.register(dom)
    return dom


# ---------------------------------------------------------------------------
# Routing matrix
# ---------------------------------------------------------------------------

class TestColdCache:
    def test_cold_cache_uses_seed(self, root):
        seed_df = make_df([date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
                          [1, 2, 3])
        seed = _ScriptedAdapter("seed", seed_df, root, {"adjustment_quality": "split_only"})
        upd = _ScriptedAdapter("upd", seed_df, root)
        _register(seed, upd)

        df, meta = fetch_with_meta("FAKE:X", date(2026, 1, 5), date(2026, 1, 7),
                                    data_root=root)
        assert len(seed.calls) == 1
        assert upd.calls == []
        assert list(df["value"]) == [1, 2, 3]
        assert meta.cache_was_cold is True
        assert meta.gaps_filled[0]["provider"] == "seed"


class TestSmallGap:
    def test_small_gap_uses_update_chain(self, root):
        # Pre-seed cache with full coverage of an earlier range.
        seed_df = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1, 2])
        seed = _ScriptedAdapter("seed", seed_df, root)
        upd_df = make_df([date(2026, 1, 7), date(2026, 1, 8)], [3, 4])
        upd = _ScriptedAdapter("upd", upd_df, root)
        _register(seed, upd, threshold=10)

        # First call: cold → seed pulls [01-05, 01-06]
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 6), data_root=root)
        assert len(seed.calls) == 1 and len(upd.calls) == 0

        # Second call: small gap (2 days) → should call updater, not seed
        df = fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 8), data_root=root)
        assert len(seed.calls) == 1  # unchanged
        assert len(upd.calls) == 1
        assert list(df["value"]) == [1, 2, 3, 4]


class TestBigGap:
    def test_big_gap_uses_seed_even_with_cache(self, root):
        seed_df = make_df([date(2026, 1, 5)], [1])
        seed_full = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
             date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 12),
             date(2026, 1, 13), date(2026, 1, 14), date(2026, 1, 15),
             date(2026, 1, 16), date(2026, 1, 19), date(2026, 1, 20),
             date(2026, 1, 21), date(2026, 1, 22), date(2026, 1, 23)],
            list(range(1, 16)),
        )

        seed = _ScriptedAdapter("seed", seed_df, root)
        # Updater would also work but should NOT be called for a big gap.
        upd = _ScriptedAdapter("upd", make_df([], []), root)
        _register(seed, upd, threshold=3)

        # Cold cache, small range — seed fills.
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 5), data_root=root)

        # Now request a much larger range. Gap is > threshold=3 → seed.
        seed._df = seed_full
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 23), data_root=root)
        assert len(seed.calls) == 2
        assert upd.calls == []


class TestFallback:
    def test_update_failure_falls_through_to_fallback(self, root):
        seed_df = make_df([date(2026, 1, 5)], [1])
        seed = _ScriptedAdapter("seed", seed_df, root)
        broken_upd = _RaisingAdapter(
            "tiingo", ProviderError("tiingo", "FAKE:X", "HTTP 503")
        )
        fb_df = make_df([date(2026, 1, 6), date(2026, 1, 7)], [2, 3])
        fb = _ScriptedAdapter("yfinance", fb_df, root)

        dom = FakeDomain(adapters=[seed, broken_upd, fb], big_gap_threshold=20)
        DomainRegistry.register(dom)

        # Seed first to establish cache.
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 5), data_root=root)
        # Now small gap → update chain [tiingo, yfinance]. Tiingo fails, yfinance succeeds.
        df, meta = fetch_with_meta("FAKE:X", date(2026, 1, 5), date(2026, 1, 7),
                                    data_root=root)
        assert list(df["value"]) == [1, 2, 3]
        assert len(broken_upd.calls) == 1
        assert len(fb.calls) == 1
        # providers_failed records the tiingo failure
        assert any(p["provider"] == "tiingo" for p in meta.providers_failed)


class TestAllFail:
    def test_chain_exhaustion_raises(self, root):
        seed = _RaisingAdapter("seed", ProviderError("seed", "FAKE:X", "HTTP 500"))
        upd = _RaisingAdapter("upd", ProviderError("upd", "FAKE:X", "rate limit"))
        _register(seed, upd)

        with pytest.raises(AllProvidersFailed) as exc:
            fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 6), data_root=root)
        assert len(exc.value.failures) == 1  # cold cache → only seed in chain
        assert exc.value.failures[0].provider == "seed"

    def test_empty_payload_treated_as_failure_cold_cache(self, root):
        # Cold cache: all-empty truly means the asset doesn't exist → raise.
        seed = _RaisingAdapter("seed", EmptyPayload("seed", "FAKE:X"))
        _register(seed)
        with pytest.raises(AllProvidersFailed):
            fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 6), data_root=root)


class TestPreCacheGapClip:
    """Requesting earlier than the cache's first date is a no-op: dispatch
    clips effective_start to cache_first_date. Avoids wasteful chain attempts
    on pre-IPO date ranges where the asset didn't exist.
    """

    def test_pre_cache_range_skipped(self, root):
        df_existing = make_df([date(2026, 1, 8), date(2026, 1, 9)], [10, 20])
        seed = _ScriptedAdapter("seed", df_existing, root)
        # Update tier MUST NOT be called — clip should prevent any gap.
        not_called = _RaisingAdapter("upd", EmptyPayload("upd", "FAKE:X"))
        dom = FakeDomain(adapters=[seed, not_called], big_gap_threshold=10)
        DomainRegistry.register(dom)

        # Cold fetch populates cache.
        fetch("FAKE:X", date(2026, 1, 8), date(2026, 1, 9), data_root=root)
        seed_calls_before = len(seed.calls)
        upd_calls_before = len(not_called.calls)

        # Re-fetch with start earlier than cache's first date.
        df = fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 9), data_root=root)

        # No new adapter calls — clip optimization handled it.
        assert len(seed.calls) == seed_calls_before
        assert len(not_called.calls) == upd_calls_before
        assert list(df["value"]) == [10, 20]


class TestInternalGapSoftFail:
    """When a gap is INSIDE the cached date range and providers can't fill it
    (any EmptyPayload), soft-fail instead of raising — preserve the cache.
    Typical case: a one-day NYSE closure we don't have in the calendar yet.
    """

    def test_internal_gap_all_empty_soft_fails(self, root):
        # Cache covers Mon, Wed (skipping Tue intentionally). Re-fetch with
        # range that includes Tue — gap is INTERNAL (after first cache row).
        df_existing = make_df([date(2026, 1, 5), date(2026, 1, 7)], [10, 30])
        seed = _ScriptedAdapter("seed", df_existing, root)
        empty_upd = _RaisingAdapter("upd", EmptyPayload("upd", "FAKE:X"))
        empty_fb = _RaisingAdapter("fb", EmptyPayload("fb", "FAKE:X"))
        dom = FakeDomain(adapters=[seed, empty_upd, empty_fb], big_gap_threshold=10)
        DomainRegistry.register(dom)

        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root)
        df, meta = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root,
        )
        # Cache rows preserved; internal gap (Tue) skipped after soft-fail.
        assert list(df["value"]) == [10, 30]
        assert any("unfillable" in p["reason"] for p in meta.providers_failed)

    def test_internal_gap_mixed_failures_still_soft_fails(self, root):
        # Any EmptyPayload + cache has data → soft fail.
        df_existing = make_df([date(2026, 1, 5), date(2026, 1, 7)], [10, 30])
        seed = _ScriptedAdapter("seed", df_existing, root)
        bad_upd = _RaisingAdapter("upd", ProviderError("upd", "FAKE:X", "HTTP 500"))
        empty_fb = _RaisingAdapter("fb", EmptyPayload("fb", "FAKE:X"))
        dom = FakeDomain(adapters=[seed, bad_upd, empty_fb], big_gap_threshold=10)
        DomainRegistry.register(dom)

        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root)
        df, _ = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root,
        )
        assert list(df["value"]) == [10, 30]

    def test_internal_gap_no_empty_only_errors_still_raises(self, root):
        # All-environmental failures (no EmptyPayload) → no authoritative
        # "no data" signal → hard fail.
        df_existing = make_df([date(2026, 1, 5), date(2026, 1, 7)], [10, 30])
        seed = _ScriptedAdapter("seed", df_existing, root)
        bad_upd = _RaisingAdapter("upd", ProviderError("upd", "FAKE:X", "HTTP 500"))
        bad_fb = _RaisingAdapter("fb", ProviderError("fb", "FAKE:X", "timeout"))
        dom = FakeDomain(adapters=[seed, bad_upd, bad_fb], big_gap_threshold=10)
        DomainRegistry.register(dom)

        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root)
        with pytest.raises(AllProvidersFailed):
            fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root)


class TestUnknownDomain:
    def test_unregistered_prefix_raises(self, root):
        with pytest.raises(UnknownDomain):
            fetch("MARS:ROVER", date(2026, 1, 5), date(2026, 1, 6),
                  data_root=root)


class TestCachePersistence:
    def test_second_call_no_adapter_invocation_when_fully_cached(self, root):
        seed_df = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1, 2])
        seed = _ScriptedAdapter("seed", seed_df, root)
        _register(seed)
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 6), data_root=root)
        assert len(seed.calls) == 1

        # Re-request same range.
        df, meta = fetch_with_meta("FAKE:X", date(2026, 1, 5), date(2026, 1, 6),
                                    data_root=root)
        assert len(seed.calls) == 1  # no new call
        assert meta.cache_was_cold is False
        assert meta.gaps_filled == []
        assert list(df["value"]) == [1, 2]


class TestSliceCorrectness:
    def test_returns_only_requested_range(self, root):
        seed_df = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
             date(2026, 1, 8), date(2026, 1, 9)],
            [1, 2, 3, 4, 5],
        )
        seed = _ScriptedAdapter("seed", seed_df, root)
        _register(seed)
        df = fetch("FAKE:X", date(2026, 1, 6), date(2026, 1, 8),
                   data_root=root)
        assert list(df["value"]) == [2, 3, 4]


class TestInputValidation:
    def test_start_after_end_raises(self, root):
        seed = _ScriptedAdapter("seed", make_df([date(2026, 1, 5)], [1]), root)
        _register(seed)
        with pytest.raises(ValueError, match="start.*after end"):
            fetch("FAKE:X", date(2026, 1, 6), date(2026, 1, 5),
                  data_root=root)

    def test_non_daily_frequency_rejected(self, root):
        seed = _ScriptedAdapter("seed", make_df([date(2026, 1, 5)], [1]), root)
        _register(seed)
        with pytest.raises(NotImplementedError):
            fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 6),
                  frequency="1min", data_root=root)


class TestPartialFillContinuation:
    """Refactor B: when the first provider returns only PART of the requested
    gap (e.g., nselib NIFTY: caps at ~3 fiscal years), the dispatcher must
    re-detect the remaining sub-gap and let the next provider in the chain
    fill it. Cache ends up with the union; both providers appear in sources.
    """

    def test_partial_then_full_yields_union(self, root):
        # Request Mon Jan-5 .. Fri Jan-9 (5 trading days). Adapter A only
        # returns the middle three; adapter B (next in chain) is allowed to
        # fill the wings.
        a_df = make_df([date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)],
                       [20, 30, 40])
        b_df = make_df([date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
                        date(2026, 1, 8), date(2026, 1, 9)],
                       [10, 20, 30, 40, 50])
        a = _ScriptedAdapter("a-partial", a_df, root)
        b = _ScriptedAdapter("b-full", b_df, root)
        _register(a, b, threshold=1000)  # cold cache → first adapter only
        # FakeDomain uses [first] on cold cache; bump so both are in play.
        # Cleaner: re-register with a custom chain.
        DomainRegistry._reset()
        dom = FakeDomain(adapters=[a, b], big_gap_threshold=1)
        # Override chain_for_gap to return both in order on a single call.
        dom.chain_for_gap = lambda ident, gap, has_cache: [a, b]
        DomainRegistry.register(dom)

        df, meta = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 9), data_root=root,
        )
        # Union of both providers' rows.
        assert list(df["date"].dt.date) == [
            date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
            date(2026, 1, 8), date(2026, 1, 9),
        ]
        # Both providers ran; both appear in gaps_filled.
        providers = [g["provider"] for g in meta.gaps_filled]
        assert "a-partial" in providers
        assert "b-full" in providers

    def test_partial_then_empty_soft_fails(self, root):
        # First adapter fills the middle; second returns empty. Cache keeps
        # what got filled; soft-fail message records the residual gap.
        a_df = make_df([date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)],
                       [20, 30, 40])
        a = _ScriptedAdapter("a-partial", a_df, root)
        b = _RaisingAdapter("b-empty", EmptyPayload("b-empty", "FAKE:X"))
        DomainRegistry._reset()
        dom = FakeDomain(adapters=[a, b], big_gap_threshold=1)
        dom.chain_for_gap = lambda ident, gap, has_cache: [a, b]
        DomainRegistry.register(dom)

        df, meta = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 9), data_root=root,
        )
        # Only the three middle rows landed.
        assert len(df) == 3
        # Soft-fail recorded — partial fill, residual unfillable.
        assert any("unfillable" in p["reason"] for p in meta.providers_failed)


class TestBackExtend:
    """back_extend=True bypasses the cache-first cap so pre-cache dates are
    requested from providers. Used for deep-history extension after the cache
    was originally seeded with a shallower range.
    """

    def test_without_back_extend_clips_pre_cache_range(self, root):
        # Mirrors TestPreCacheGapClip but asserts the explicit default.
        df_existing = make_df([date(2026, 1, 8), date(2026, 1, 9)], [10, 20])
        seed = _ScriptedAdapter("seed", df_existing, root)
        not_called = _RaisingAdapter("upd", EmptyPayload("upd", "FAKE:X"))
        dom = FakeDomain(adapters=[seed, not_called], big_gap_threshold=10)
        DomainRegistry.register(dom)

        # Cold fetch populates cache.
        fetch("FAKE:X", date(2026, 1, 8), date(2026, 1, 9), data_root=root)
        seed_calls_before = len(seed.calls)
        upd_calls_before = len(not_called.calls)

        # Re-fetch earlier than cache_first WITHOUT back_extend → clipped.
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 9), data_root=root)
        assert len(seed.calls) == seed_calls_before
        assert len(not_called.calls) == upd_calls_before

    def test_with_back_extend_requests_pre_cache_range(self, root):
        # First seed installs Jan-8..Jan-9. Then back_extend re-fetch with
        # start Jan-5 must result in an adapter call covering the Jan-5..Jan-7
        # pre-cache window.
        df_existing = make_df([date(2026, 1, 8), date(2026, 1, 9)], [10, 20])
        df_extended = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
            [5, 6, 7],
        )
        seed = _ScriptedAdapter("seed", df_existing, root)
        _register(seed, threshold=10)

        # Cold seed.
        fetch("FAKE:X", date(2026, 1, 8), date(2026, 1, 9), data_root=root)
        assert len(seed.calls) == 1

        # Swap the adapter's canned df so the back-extend call returns the
        # earlier rows.
        seed._df = df_extended

        df, meta = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 9),
            data_root=root, back_extend=True,
        )
        # An additional provider call landed.
        assert len(seed.calls) == 2
        # The new call's start was earlier than the cache_first (Jan-8).
        new_call = seed.calls[-1]
        assert new_call[1] == date(2026, 1, 5)
        # Cache now spans Jan-5..Jan-9 (existing rows preserved, new rows
        # merged in).
        assert list(df["date"].dt.date) == [
            date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
            date(2026, 1, 8), date(2026, 1, 9),
        ]
        assert list(df["value"]) == [5, 6, 7, 10, 20]
        assert meta.gaps_filled, "expected a gap to be reported as filled"

    def test_back_extend_noop_when_start_at_or_after_cache_first(self, root):
        # back_extend=True with start >= cache_first must behave exactly like
        # the default path: no additional pre-cache fetch attempts.
        df_existing = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
            [10, 20, 30],
        )
        seed = _ScriptedAdapter("seed", df_existing, root)
        _register(seed)

        # Cold seed installs Jan-5..Jan-7.
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root)
        seed_calls_before = len(seed.calls)

        # Re-request fully-cached range with back_extend=True → no new calls.
        df, _ = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 7),
            data_root=root, back_extend=True,
        )
        assert len(seed.calls) == seed_calls_before
        assert list(df["value"]) == [10, 20, 30]

    def test_back_extend_preserves_existing_cached_rows(self, root):
        # Existing rows must not be corrupted by a back_extend re-fetch even
        # when the new adapter response happens to cover (with the same
        # values) the existing cache range. Use existing_wins merge policy
        # to make the assertion strict: the original values must survive.
        df_existing = make_df([date(2026, 1, 8), date(2026, 1, 9)], [10, 20])
        # New adapter payload covers Jan-5..Jan-9 — overlaps the cache.
        df_overlap = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
             date(2026, 1, 8), date(2026, 1, 9)],
            [5, 6, 7, 999, 999],  # different values on the overlap dates
        )
        seed = _ScriptedAdapter("seed", df_existing, root)
        # existing_wins policy: overlap rows keep their original values.
        dom = FakeDomain(
            adapters=[seed], big_gap_threshold=10,
            overlap_policy="existing_wins",
        )
        DomainRegistry.register(dom)

        fetch("FAKE:X", date(2026, 1, 8), date(2026, 1, 9), data_root=root)
        seed._df = df_overlap

        df = fetch(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 9),
            data_root=root, back_extend=True,
        )
        # Pre-cache rows added; existing Jan-8/9 values preserved (10, 20),
        # NOT clobbered by the 999s the new payload carried.
        assert list(df["date"].dt.date) == [
            date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
            date(2026, 1, 8), date(2026, 1, 9),
        ]
        assert list(df["value"]) == [5, 6, 7, 10, 20]


class TestExtraMetaPropagation:
    def test_extra_meta_in_source_record(self, root):
        seed_df = make_df([date(2026, 1, 5)], [1])
        seed = _ScriptedAdapter("seed", seed_df, root,
                                 extra_meta={"adjustment_quality": "split_only"})
        _register(seed)
        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 5), data_root=root)

        # Re-read processed meta to verify extra_meta propagated.
        from data_pipelines.cache import read_processed
        _, meta = read_processed(root, DomainRegistry.resolve("FAKE:X"), "FAKE:X")
        assert meta["sources"][0]["adjustment_quality"] == "split_only"


class TestEmptyPostNormalizeResponse:
    """Regression: a provider whose parse()+normalize() yields a zero-row
    DataFrame must NOT raise a bare IndexError out of _build_source_meta.

    Schema.validate() accepts an empty frame whose columns/dtypes match, so
    the empty payload slips past the validator. Before the fix, dispatch then
    indexed ``df[time_column].iloc[0]`` to build the source-meta covers range
    and crashed with IndexError, escaping the chain-fallthrough machinery
    (which only catches typed ProviderError subclasses). The fix promotes
    the empty-frame case to EmptyPayload so the existing soft-fail (with
    cache) / hard-fail (cold cache) logic handles it.
    """

    def test_empty_post_normalize_treated_as_empty_payload_cold_cache(self, root):
        # Cold cache + adapter returns a structurally valid but empty frame
        # → should raise AllProvidersFailed (NOT IndexError).
        empty_df = make_df([], [])
        adapter = _ScriptedAdapter("seed", empty_df, root)
        _register(adapter)
        with pytest.raises(AllProvidersFailed) as exc:
            fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 6), data_root=root)
        # The failure must be typed (EmptyPayload), not a stray IndexError
        # wrapped or re-raised. The chain captures it as a ProviderError.
        assert exc.value.failures, "expected at least one captured failure"
        assert exc.value.failures[0].provider == "seed"
        assert "empty payload" in exc.value.failures[0].reason

    def test_empty_post_normalize_soft_fails_when_cache_has_data(self, root):
        # Cache already covers Jan-5..Jan-7. Re-request a range with an
        # internal gap (Tue) where the adapter returns an empty frame —
        # soft-fail path engages, cache preserved, no IndexError.
        existing = make_df([date(2026, 1, 5), date(2026, 1, 7)], [10, 30])
        seed = _ScriptedAdapter("seed", existing, root)
        upd = _ScriptedAdapter("upd", make_df([], []), root)
        dom = FakeDomain(adapters=[seed, upd], big_gap_threshold=10)
        DomainRegistry.register(dom)

        fetch("FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root)
        df, meta = fetch_with_meta(
            "FAKE:X", date(2026, 1, 5), date(2026, 1, 7), data_root=root,
        )
        # Cache preserved; soft-fail recorded for the empty payload.
        assert list(df["value"]) == [10, 30]
        assert any("unfillable" in p["reason"] for p in meta.providers_failed)


class TestBuildSourceMetaUnit:
    """Direct unit tests for ``_build_source_meta`` — both branches (empty and
    non-empty). The happy-path branch is also exercised indirectly by every
    other dispatch test that successfully fills a gap, but a focused unit
    test pins the contract.
    """

    def test_empty_dataframe_raises_empty_payload(self):
        from data_pipelines.dispatch import _build_source_meta

        class _A:
            name = "p"
            extra_meta: dict = {}

        empty_df = make_df([], [])
        with pytest.raises(EmptyPayload) as exc:
            _build_source_meta(
                adapter=_A(), raw_path=Path("ignored.csv"),
                df=empty_df, time_column="date", identifier="FAKE:X",
            )
        assert exc.value.provider == "p"
        assert exc.value.identifier == "FAKE:X"

    def test_non_empty_dataframe_returns_covers_range(self):
        from data_pipelines.dispatch import _build_source_meta

        class _A:
            name = "p"
            extra_meta = {"adjustment_quality": "split_only"}

        df = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
            [1, 2, 3],
        )
        meta = _build_source_meta(
            adapter=_A(), raw_path=Path("file.csv"),
            df=df, time_column="date", identifier="FAKE:X",
        )
        assert meta["provider"] == "p"
        assert meta["raw_file"] == "file.csv"
        assert meta["covers"] == {"start": "2026-01-05", "end": "2026-01-07"}
        assert meta["adjustment_quality"] == "split_only"
