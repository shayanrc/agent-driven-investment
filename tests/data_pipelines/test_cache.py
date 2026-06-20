"""cache.py tests — SQLite-backed read/write atomicity, gap detection, merge.

(Swapped from parquet to SQLite in v1.5; the gap-detection and merge tests
are unchanged since those are pure-DataFrame operations.)
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipelines.cache import (
    META_SCHEMA_VERSION,
    detect_gaps,
    list_cached_identifiers,
    merge_cache,
    processed_db_path,
    purge_identifier,
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
    def test_processed_db_default_path(self, root):
        p = processed_db_path(root)
        assert p == root / "processed.db"

    def test_processed_db_custom_subdir(self, root):
        p = processed_db_path(root, processed_subdir="alt.sqlite")
        assert p == root / "alt.sqlite"


class TestReadProcessed:
    def test_missing_db_returns_none(self, root, fake_domain):
        assert read_processed(root, fake_domain, "FAKE:X") == (None, None)

    def test_empty_db_returns_none(self, root, fake_domain):
        # Create empty DB by writing then purging.
        df_in = make_df([date(2026, 1, 5)], [1.0])
        write_processed_atomic(root, fake_domain, "FAKE:X", df_in, {})
        purge_identifier(root, fake_domain, "FAKE:X")
        assert read_processed(root, fake_domain, "FAKE:X") == (None, None)

    def test_meta_row_is_commit_marker(self, root, fake_domain):
        # Insert data rows manually but no meta row → read returns (None, None).
        db = processed_db_path(root)
        db.parent.mkdir(parents=True, exist_ok=True)
        # Trigger table creation via a real write+purge.
        write_processed_atomic(root, fake_domain, "FAKE:OTHER",
                                make_df([date(2026, 1, 5)], [9.0]), {})
        purge_identifier(root, fake_domain, "FAKE:OTHER")
        # Now inject just a data row directly.
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO fake_data (ticker, date, value) VALUES (?, ?, ?)",
                ("FAKE:X", "2026-01-05 00:00:00", 1.0),
            )
            conn.commit()
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
        # Dtypes survive the round-trip — canonical to canonical.
        assert str(df_out["date"].dtype) == "datetime64[ns]"
        assert str(df_out["value"].dtype) == "float64"
        # Meta preserved (with normalization).
        assert meta_out["row_count"] == 2
        assert meta_out["range"] == {"start": "2026-01-05", "end": "2026-01-06"}
        assert meta_out["sources"] == [src_meta]
        assert meta_out["schema_version"] == 1

    def test_overwrite_replaces_data(self, root, fake_domain, src_meta):
        # First write: 2 rows.
        df1 = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1.0, 2.0])
        write_processed_atomic(root, fake_domain, "FAKE:X", df1,
                                {"row_count": 2, "range": {"start": "2026-01-05",
                                                            "end": "2026-01-06"},
                                 "sources": [src_meta], "schema_version": 1,
                                 "last_fetch_utc": "x"})
        # Second write: 1 row only. The data for FAKE:X must be fully replaced.
        df2 = make_df([date(2026, 1, 7)], [3.0])
        write_processed_atomic(root, fake_domain, "FAKE:X", df2,
                                {"row_count": 1, "range": {"start": "2026-01-07",
                                                            "end": "2026-01-07"},
                                 "sources": [src_meta], "schema_version": 1,
                                 "last_fetch_utc": "x"})
        df_out, meta_out = read_processed(root, fake_domain, "FAKE:X")
        assert list(df_out["value"]) == [3.0]
        assert meta_out["row_count"] == 1

    def test_atomic_rollback_on_failure(self, root, fake_domain, src_meta):
        # Establish a known good state.
        df1 = make_df([date(2026, 1, 5)], [1.0])
        meta_in = {"schema_version": 1, "row_count": 1,
                   "range": {"start": "2026-01-05", "end": "2026-01-05"},
                   "last_fetch_utc": "x", "sources": [src_meta]}
        write_processed_atomic(root, fake_domain, "FAKE:X", df1, meta_in)

        # Second write: simulate a crash after _replace_data deleted the prior
        # rows but before the meta upsert lands. The transaction must roll
        # back so the prior data + meta remain intact.
        df2 = make_df([date(2026, 2, 1)], [99.0])
        from data_pipelines import cache as cache_mod
        with patch.object(cache_mod, "_upsert_meta",
                          side_effect=RuntimeError("simulated meta-write crash")):
            with pytest.raises(RuntimeError, match="simulated meta-write crash"):
                write_processed_atomic(root, fake_domain, "FAKE:X", df2, meta_in)

        df_out, meta_out = read_processed(root, fake_domain, "FAKE:X")
        assert list(df_out["value"]) == [1.0]
        assert meta_out["row_count"] == 1


class TestListAndPurge:
    def test_list_empty(self, root, fake_domain):
        assert list_cached_identifiers(root, fake_domain) == []

    def test_list_after_writes(self, root, fake_domain, src_meta):
        for tk in ("FAKE:A", "FAKE:B", "FAKE:C"):
            write_processed_atomic(
                root, fake_domain, tk,
                make_df([date(2026, 1, 5)], [1.0]),
                {"row_count": 1, "range": {"start": "2026-01-05",
                                            "end": "2026-01-05"},
                 "last_fetch_utc": "x", "sources": [src_meta],
                 "schema_version": 1},
            )
        assert list_cached_identifiers(root, fake_domain) == [
            "FAKE:A", "FAKE:B", "FAKE:C",
        ]

    def test_purge_removes_both_tables(self, root, fake_domain, src_meta):
        write_processed_atomic(
            root, fake_domain, "FAKE:X",
            make_df([date(2026, 1, 5)], [1.0]),
            {"row_count": 1, "range": {"start": "2026-01-05",
                                        "end": "2026-01-05"},
             "last_fetch_utc": "x", "sources": [src_meta],
             "schema_version": 1},
        )
        assert purge_identifier(root, fake_domain, "FAKE:X") is True
        assert read_processed(root, fake_domain, "FAKE:X") == (None, None)
        # Subsequent purge returns False (idempotent).
        assert purge_identifier(root, fake_domain, "FAKE:X") is False


class TestDetectGaps:
    def test_cold_cache_one_big_gap(self, fake_domain):
        gaps = detect_gaps(None, date(2026, 1, 5), date(2026, 1, 9),
                           fake_domain.calendar)
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
        assert gaps == []

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
            date(2026, 1, 6): 200,
            date(2026, 1, 7): 300,
        }

    def test_domain_overlap_policy_respected(self, src_meta):
        dom = FakeDomain(overlap_policy="existing_wins")
        ex = make_df([date(2026, 1, 6)], [10])
        new = make_df([date(2026, 1, 6), date(2026, 1, 7)], [200, 300])
        merged, _ = merge_cache(ex, new, {"sources": []}, src_meta, dom)
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
        new = make_df([date(2026, 1, 8), date(2026, 1, 5), date(2026, 1, 7)],
                      [3, 1, 2])
        merged, _ = merge_cache(None, new, None, src_meta, fake_domain)
        assert list(merged["value"]) == [1, 2, 3]


class TestMergeDeDuplicatesTimestamps:
    """Regression for issue #36.

    Some upstream providers (e.g., jugaad-data on a handful of pre-2015 NSE
    bhav rows) return DataFrames with internal duplicate timestamps. Without
    a de-dup pass in ``merge_cache``, those duplicates flow into
    ``write_processed_atomic`` and crash the SQLite write with an opaque
    ``IntegrityError: UNIQUE constraint failed`` because the data table has
    PRIMARY KEY (ticker, <time_column>).

    The de-dup must:
      - keep the LAST occurrence per timestamp (matches "new wins"),
      - emit a WARNING log line naming the source,
      - be a no-op when there are no duplicates.
    """

    def test_internal_dup_in_new_df_cold_cache(self, fake_domain, src_meta, caplog):
        # Cold cache + adapter returns 2 rows for the same date.
        new = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 6)],
            [1.0, 2.0, 2.5],
        )
        with caplog.at_level("WARNING", logger="data_pipelines.cache"):
            merged, meta = merge_cache(None, new, None, src_meta, fake_domain)
        # The duplicate was dropped; the LAST occurrence (2.5) wins.
        assert list(merged["date"].dt.date) == [date(2026, 1, 5), date(2026, 1, 6)]
        assert list(merged["value"]) == [1.0, 2.5]
        assert meta["row_count"] == 2
        # Warning was emitted with the source name visible.
        assert any("dropped 1 duplicate" in r.message and "fake_prov" in r.message
                   for r in caplog.records)

    def test_internal_dup_in_new_df_with_existing_cache(self, fake_domain, src_meta, caplog):
        # Cache has Jan-1..Jan-3; new payload extends backward to Dec-30
        # with an internal duplicate on Dec-31. No overlap with cache.
        # Exercises the non-cold-cache code path that initially exhibited
        # the production bug (back-extend → adapter dup → crash on write).
        existing = make_df(
            [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            [10.0, 20.0, 30.0],
        )
        new = make_df(
            [date(2025, 12, 30), date(2025, 12, 31), date(2025, 12, 31)],
            [98.0, 99.0, 99.5],
        )
        ex_meta = {"sources": [{"provider": "prev"}]}
        with caplog.at_level("WARNING", logger="data_pipelines.cache"):
            merged, meta = merge_cache(existing, new, ex_meta, src_meta, fake_domain)
        assert list(merged["date"].dt.date) == [
            date(2025, 12, 30), date(2025, 12, 31),
            date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),
        ]
        assert list(merged["value"]) == [98.0, 99.5, 10.0, 20.0, 30.0]
        assert meta["row_count"] == 5
        assert any("dropped 1 duplicate" in r.message for r in caplog.records)

    def test_no_dup_no_warning(self, fake_domain, src_meta, caplog):
        new = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1.0, 2.0])
        with caplog.at_level("WARNING", logger="data_pipelines.cache"):
            merged, _ = merge_cache(None, new, None, src_meta, fake_domain)
        assert list(merged["value"]) == [1.0, 2.0]
        # No warning emitted on the clean path.
        assert not any("duplicate" in r.message for r in caplog.records)

    def test_deduped_df_writes_cleanly_via_full_pipeline(
        self, root, fake_domain, src_meta,
    ):
        # End-to-end: a new_df with internal dups merges + writes without
        # triggering a SQLite UNIQUE error. This is the exact path the
        # production back-extend hit.
        existing = make_df(
            [date(2026, 1, 4), date(2026, 1, 5)], [100.0, 101.0],
        )
        write_processed_atomic(
            root, fake_domain, "FAKE:X", existing,
            {"schema_version": 1, "row_count": 2,
             "range": {"start": "2026-01-04", "end": "2026-01-05"},
             "last_fetch_utc": "x", "sources": [src_meta]},
        )
        cached_df, cached_meta = read_processed(root, fake_domain, "FAKE:X")

        # Simulated provider response with an internal duplicate.
        new = make_df(
            [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 2)],
            [200.0, 300.0, 301.0],
        )
        new_source = {"provider": "back_extend_prov",
                      "raw_file": "back_extend.csv",
                      "covers": {"start": "2026-01-01", "end": "2026-01-02"}}
        merged, meta = merge_cache(
            cached_df, new, cached_meta, new_source, fake_domain,
        )
        # No exception — this is the regression.
        write_processed_atomic(root, fake_domain, "FAKE:X", merged, meta)

        df_out, meta_out = read_processed(root, fake_domain, "FAKE:X")
        assert list(df_out["date"].dt.date) == [
            date(2026, 1, 1), date(2026, 1, 2),
            date(2026, 1, 4), date(2026, 1, 5),
        ]
        assert list(df_out["value"]) == [200.0, 301.0, 100.0, 101.0]
        assert meta_out["row_count"] == 4


class TestReadWriteRoundTrip:
    def test_full_pipeline(self, root, fake_domain, src_meta):
        df1 = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1, 2])
        merged1, meta1 = merge_cache(None, df1, None, src_meta, fake_domain)
        write_processed_atomic(root, fake_domain, "FAKE:X", merged1, meta1)

        df_read, meta_read = read_processed(root, fake_domain, "FAKE:X")
        new = make_df([date(2026, 1, 7), date(2026, 1, 8)], [3, 4])
        src2 = {"provider": "fake_prov", "raw_file": "20260108T000000Z_a_b.csv"}
        merged2, meta2 = merge_cache(df_read, new, meta_read, src2, fake_domain)
        write_processed_atomic(root, fake_domain, "FAKE:X", merged2, meta2)

        df_final, meta_final = read_processed(root, fake_domain, "FAKE:X")
        assert list(df_final["value"]) == [1, 2, 3, 4]
        assert meta_final["row_count"] == 4
        assert len(meta_final["sources"]) == 2

    def test_multiple_tickers_coexist(self, root, fake_domain, src_meta):
        meta = {"schema_version": 1, "row_count": 1,
                "range": {"start": "2026-01-05", "end": "2026-01-05"},
                "last_fetch_utc": "x", "sources": [src_meta]}
        write_processed_atomic(root, fake_domain, "FAKE:A",
                                make_df([date(2026, 1, 5)], [100]), meta)
        write_processed_atomic(root, fake_domain, "FAKE:B",
                                make_df([date(2026, 1, 5)], [200]), meta)
        a, _ = read_processed(root, fake_domain, "FAKE:A")
        b, _ = read_processed(root, fake_domain, "FAKE:B")
        assert list(a["value"]) == [100]
        assert list(b["value"]) == [200]


class TestNullableRoundTrip:
    """NaN in a value column survives the write→read round-trip as NaN.

    Regression guard for the iterrows() dtype-coercion bug: a NaN float in a
    row that also carries a datetime column was binding as NaT and crashing
    the SQLite write. Equity domains never store NaN; fred_macro's nullable
    `value` is the first to hit this, so the fix lives in cache.py and is
    pinned here at the framework level.
    """

    def test_nan_value_survives_as_null(self, root, fake_domain):
        df_in = make_df(
            [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
            [1.0, float("nan"), 3.0],
        )
        write_processed_atomic(root, fake_domain, "FAKE:X", df_in, {})
        df_out, _ = read_processed(root, fake_domain, "FAKE:X")
        assert str(df_out["value"].dtype) == "float64"
        assert df_out["value"].isna().tolist() == [False, True, False]
        assert df_out.loc[df_out["value"].notna(), "value"].tolist() == [1.0, 3.0]

    def test_stored_as_sql_null(self, root, fake_domain):
        # The middle row's value must be a genuine SQL NULL, not a NaN-real.
        df_in = make_df([date(2026, 1, 5), date(2026, 1, 6)], [1.0, float("nan")])
        write_processed_atomic(root, fake_domain, "FAKE:X", df_in, {})
        conn = sqlite3.connect(str(processed_db_path(root)))
        try:
            n_null = conn.execute(
                "SELECT COUNT(*) FROM fake_data WHERE value IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        assert n_null == 1
