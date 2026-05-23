"""Stage 3 tests: cache.py — read/write atomicity, gap detection, merge."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipelines.cache import (
    META_SCHEMA_VERSION,
    PROCESSED_META_NAME,
    PROCESSED_PARQUET_NAME,
    detect_gaps,
    merge_cache,
    processed_dir,
    read_processed,
    write_processed_atomic,
)

from .conftest import FakeCalendar, FakeDomain, make_df


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def src_meta() -> dict:
    return {"provider": "fake_prov", "raw_file": "20260101T000000Z_a_b.csv",
            "covers": {"start": "2026-01-01", "end": "2026-01-31"}}


class TestPathLayout:
    def test_processed_dir(self, root, fake_domain):
        d = processed_dir(root, fake_domain.name, "FAKE", "X")
        assert d == root / "processed" / "fake" / "FAKE" / "X"


class TestReadProcessed:
    def test_missing_returns_none(self, root, fake_domain):
        assert read_processed(root, fake_domain, "FAKE:X") == (None, None)

    def test_parquet_without_meta_treated_missing(self, root, fake_domain):
        d = processed_dir(root, fake_domain.name, "FAKE", "X")
        d.mkdir(parents=True)
        df = make_df([date(2026, 1, 5)], [1.0])
        df.to_parquet(d / PROCESSED_PARQUET_NAME, engine="pyarrow", index=False)
        # No meta written.
        assert read_processed(root, fake_domain, "FAKE:X") == (None, None)


class TestWriteProcessed:
    def test_round_trip(self, root, fake_domain, src_meta):
        df_in = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1.0, 2.0])
        meta_in = {"schema_version": 1, "domain": "fake", "row_count": 2,
                   "range": {"start": "2026-01-05", "end": "2026-01-06"},
                   "last_fetch_utc": "2026-05-23T14:30:22Z",
                   "sources": [src_meta]}
        write_processed_atomic(root, fake_domain, "FAKE:X", df_in, meta_in)
        df_out, meta_out = read_processed(root, fake_domain, "FAKE:X")
        assert list(df_out["value"]) == [1.0, 2.0]
        assert meta_out == meta_in

    def test_parquet_atomic_no_partial(self, root, fake_domain):
        df = make_df([date(2026, 1, 5)], [1.0])
        with patch("data_pipelines.cache.os.replace",
                   side_effect=RuntimeError("crash")):
            with pytest.raises(RuntimeError):
                write_processed_atomic(root, fake_domain, "FAKE:X", df, {"x": 1})
        d = processed_dir(root, fake_domain.name, "FAKE", "X")
        files = list(d.iterdir()) if d.exists() else []
        assert all(f.suffix == ".tmp" or f.name.startswith(".") is False for f in files)
        # No final parquet, no final meta.
        assert not (d / PROCESSED_PARQUET_NAME).exists()
        assert not (d / PROCESSED_META_NAME).exists()
        # No leftover temp.
        leftovers = [f for f in files if f.name.endswith(".tmp")]
        assert leftovers == []


class TestDetectGaps:
    def test_cold_cache_one_big_gap(self, fake_domain):
        gaps = detect_gaps(None, date(2026, 1, 5), date(2026, 1, 9),
                           fake_domain.calendar)
        # Mon-Fri 2026-01-05..09 → all five days form one gap.
        assert gaps == [(date(2026, 1, 5), date(2026, 1, 9))]

    def test_full_coverage_no_gaps(self, fake_domain):
        df = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
             date(2026, 1, 8), date(2026, 1, 9)],
            [1, 2, 3, 4, 5],
        )
        gaps = detect_gaps(df, date(2026, 1, 5), date(2026, 1, 9),
                           fake_domain.calendar)
        assert gaps == []

    def test_internal_hole(self, fake_domain):
        df = make_df([date(2026, 1, 5), date(2026, 1, 9)], [1, 2])
        gaps = detect_gaps(df, date(2026, 1, 5), date(2026, 1, 9),
                           fake_domain.calendar)
        assert gaps == [(date(2026, 1, 6), date(2026, 1, 8))]

    def test_two_holes(self, fake_domain):
        df = make_df([date(2026, 1, 5), date(2026, 1, 7), date(2026, 1, 9)],
                     [1, 2, 3])
        gaps = detect_gaps(df, date(2026, 1, 5), date(2026, 1, 9),
                           fake_domain.calendar)
        assert gaps == [(date(2026, 1, 6), date(2026, 1, 6)),
                        (date(2026, 1, 8), date(2026, 1, 8))]

    def test_calendar_skips_weekends(self, fake_domain):
        # Jan 5 (Mon) … Jan 12 (Mon) — weekend Jan 10–11 is NOT a trading day.
        df = make_df([date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
                      date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 12)],
                     [1, 2, 3, 4, 5, 6])
        gaps = detect_gaps(df, date(2026, 1, 5), date(2026, 1, 12),
                           fake_domain.calendar)
        assert gaps == []

    def test_calendar_honors_holiday(self):
        cal = FakeCalendar(holidays=frozenset({date(2026, 1, 6)}))
        dom = FakeDomain(calendar=cal)
        df = make_df([date(2026, 1, 5), date(2026, 1, 7)], [1, 2])
        gaps = detect_gaps(df, date(2026, 1, 5), date(2026, 1, 7), dom.calendar)
        assert gaps == []  # Jan 6 is a holiday, not a missing trading day

    def test_prefix_only_cache(self, fake_domain):
        df = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1, 2])
        gaps = detect_gaps(df, date(2026, 1, 5), date(2026, 1, 9),
                           fake_domain.calendar)
        assert gaps == [(date(2026, 1, 7), date(2026, 1, 9))]


class TestMergeCache:
    def test_empty_existing_appends_all(self, fake_domain, src_meta):
        new_df = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1, 2])
        merged, meta = merge_cache(None, new_df, None, src_meta, fake_domain)
        assert list(merged["value"]) == [1, 2]
        assert meta["row_count"] == 2
        assert meta["sources"] == [src_meta]
        assert meta["schema_version"] == META_SCHEMA_VERSION
        assert meta["range"] == {"start": "2026-01-05", "end": "2026-01-06"}

    def test_no_overlap_concat(self, fake_domain, src_meta):
        ex = make_df([date(2026, 1, 5)], [1])
        new = make_df([date(2026, 1, 6)], [2])
        merged, meta = merge_cache(ex, new, {"sources": [{"provider": "prev"}]},
                                    src_meta, fake_domain)
        assert list(merged["value"]) == [1, 2]
        assert len(meta["sources"]) == 2

    def test_overlap_new_wins_default(self, fake_domain, src_meta):
        ex = make_df([date(2026, 1, 5), date(2026, 1, 6)], [10, 20])
        new = make_df([date(2026, 1, 6), date(2026, 1, 7)], [200, 300])
        merged, _ = merge_cache(ex, new, {"sources": []}, src_meta, fake_domain)
        assert dict(zip(merged["date"].dt.date.tolist(),
                        merged["value"].tolist())) == {
            date(2026, 1, 5): 10,
            date(2026, 1, 6): 200,  # new wins on overlap
            date(2026, 1, 7): 300,
        }

    def test_domain_overlap_policy_respected(self, src_meta):
        dom = FakeDomain(overlap_policy="existing_wins")
        ex = make_df([date(2026, 1, 6)], [10])
        new = make_df([date(2026, 1, 6), date(2026, 1, 7)], [200, 300])
        merged, _ = merge_cache(ex, new, {"sources": []}, src_meta, dom)
        # Existing value preserved on overlap; new-only day appended.
        as_dict = dict(zip(merged["date"].dt.date.tolist(),
                            merged["value"].tolist()))
        assert as_dict == {date(2026, 1, 6): 10, date(2026, 1, 7): 300}

    def test_sources_array_grows(self, fake_domain, src_meta):
        ex = make_df([date(2026, 1, 5)], [1])
        ex_meta = {"sources": [{"provider": "stooq", "raw_file": "f1"}]}
        new = make_df([date(2026, 1, 6)], [2])
        _, meta = merge_cache(ex, new, ex_meta, src_meta, fake_domain)
        providers = [s["provider"] for s in meta["sources"]]
        assert providers == ["stooq", "fake_prov"]

    def test_merged_sorted_ascending(self, fake_domain, src_meta):
        # Out-of-order rows on the new side.
        new = make_df([date(2026, 1, 8), date(2026, 1, 5), date(2026, 1, 7)],
                      [3, 1, 2])
        merged, _ = merge_cache(None, new, None, src_meta, fake_domain)
        assert list(merged["value"]) == [1, 2, 3]


class TestReadWriteRoundTrip:
    def test_full_pipeline(self, root, fake_domain, src_meta):
        # First fetch.
        df1 = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1, 2])
        merged1, meta1 = merge_cache(None, df1, None, src_meta, fake_domain)
        write_processed_atomic(root, fake_domain, "FAKE:X", merged1, meta1)

        # Second fetch fills a gap.
        df_read, meta_read = read_processed(root, fake_domain, "FAKE:X")
        new = make_df([date(2026, 1, 7), date(2026, 1, 8)], [3, 4])
        src2 = {"provider": "fake_prov", "raw_file": "20260108T000000Z_a_b.csv"}
        merged2, meta2 = merge_cache(df_read, new, meta_read, src2, fake_domain)
        write_processed_atomic(root, fake_domain, "FAKE:X", merged2, meta2)

        df_final, meta_final = read_processed(root, fake_domain, "FAKE:X")
        assert list(df_final["value"]) == [1, 2, 3, 4]
        assert meta_final["row_count"] == 4
        assert len(meta_final["sources"]) == 2
