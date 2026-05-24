"""Stage 2 tests: raw_store atomicity (D2), immutability (D8), filename contract."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipelines.raw_store import (
    encode_filename,
    list_raw,
    parse_filename,
    raw_dir,
    write_raw_atomic,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def fixed_ts() -> datetime:
    return datetime(2026, 5, 23, 14, 30, 22, tzinfo=timezone.utc)


class TestFilenameRoundTrip:
    def test_encode_parse_round_trip(self, fixed_ts):
        name = encode_filename(fixed_ts, date(2010, 1, 4), date(2026, 5, 22), "csv")
        assert name == "20260523T143022Z_2010-01-04_2026-05-22.csv"
        parsed = parse_filename(name)
        assert parsed.timestamp == fixed_ts
        assert parsed.range_start == date(2010, 1, 4)
        assert parsed.range_end == date(2026, 5, 22)
        assert parsed.ext == "csv"

    def test_encode_strips_leading_dot_in_ext(self, fixed_ts):
        name = encode_filename(fixed_ts, date(2026, 1, 1), date(2026, 1, 2), ".json")
        assert name.endswith(".json")
        assert ".." not in name

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="tz-aware"):
            encode_filename(datetime(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 2), "csv")

    def test_non_utc_normalized(self):
        from datetime import timedelta
        tz_plus5 = timezone(timedelta(hours=5))
        # 2026-05-23 19:30:22 +05 == 2026-05-23 14:30:22Z
        ts = datetime(2026, 5, 23, 19, 30, 22, tzinfo=tz_plus5)
        name = encode_filename(ts, date(2026, 1, 1), date(2026, 1, 2), "csv")
        assert name.startswith("20260523T143022Z")

    def test_parse_rejects_garbage(self):
        for bad in ["foo.csv", "20260523_2026-01-01_2026-01-02.csv",
                    "20260523T143022Z_bad_2026-01-02.csv", ""]:
            with pytest.raises(ValueError, match="not a valid raw filename"):
                parse_filename(bad)


class TestPathLayout:
    def test_raw_dir_layout(self, root):
        d = raw_dir(root, "stooq", "us_equities", "NYSE", "AAPL")
        assert d == root / "raw" / "stooq" / "us_equities" / "NYSE" / "AAPL"

    def test_custom_raw_subdir(self, root):
        d = raw_dir(root, "tiingo", "us_equities", "NASDAQ", "MSFT",
                    raw_subdir="raw_alt")
        assert d == root / "raw_alt" / "tiingo" / "us_equities" / "NASDAQ" / "MSFT"


class TestWriteRawAtomic:
    def test_basic_write(self, root, fixed_ts):
        path = write_raw_atomic(
            root, "stooq", "us_equities", "NYSE", "AAPL",
            payload=b"date,close\n2026-01-01,100\n",
            range_start=date(2026, 1, 1), range_end=date(2026, 1, 2),
            ext="csv", timestamp=fixed_ts,
        )
        assert path.exists()
        assert path.read_bytes() == b"date,close\n2026-01-01,100\n"
        assert path.name == "20260523T143022Z_2026-01-01_2026-01-02.csv"
        assert path.parent == root / "raw" / "stooq" / "us_equities" / "NYSE" / "AAPL"

    def test_creates_parent_dirs(self, root, fixed_ts):
        path = write_raw_atomic(
            root, "tiingo", "us_equities", "NYSE", "DEEP",
            payload=b"{}", range_start=date(2026, 1, 1), range_end=date(2026, 1, 2),
            ext="json", timestamp=fixed_ts,
        )
        assert path.parent.is_dir()

    def test_d8_immutability_existing_file_rejected(self, root, fixed_ts):
        kwargs = dict(
            data_root=root, provider="stooq", domain="us_equities",
            exchange="NYSE", ticker="AAPL",
            payload=b"v1", range_start=date(2026, 1, 1), range_end=date(2026, 1, 2),
            ext="csv", timestamp=fixed_ts,
        )
        write_raw_atomic(**kwargs)
        with pytest.raises(FileExistsError, match="immutability"):
            write_raw_atomic(**{**kwargs, "payload": b"v2"})

    def test_no_partial_file_on_crash(self, root, fixed_ts):
        """D2: a crash during the write must not leave a file at the final path."""
        kwargs = dict(
            data_root=root, provider="stooq", domain="us_equities",
            exchange="NYSE", ticker="AAPL",
            payload=b"would be data",
            range_start=date(2026, 1, 1), range_end=date(2026, 1, 2),
            ext="csv", timestamp=fixed_ts,
        )

        with patch("data_pipelines.raw_store.os.replace",
                   side_effect=RuntimeError("simulated crash")):
            with pytest.raises(RuntimeError, match="simulated crash"):
                write_raw_atomic(**kwargs)

        target = raw_dir(root, "stooq", "us_equities", "NYSE", "AAPL")
        # No .csv file at final path.
        files = list(target.iterdir()) if target.exists() else []
        finals = [p for p in files if p.name.endswith(".csv")]
        assert finals == []
        # No leftover .tmp either (cleanup path ran).
        tmps = [p for p in files if p.name.endswith(".tmp")]
        assert tmps == []

    def test_distinct_timestamps_coexist(self, root):
        ts1 = datetime(2026, 5, 23, 14, 30, 22, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 23, 14, 30, 23, tzinfo=timezone.utc)
        p1 = write_raw_atomic(
            root, "stooq", "us_equities", "NYSE", "AAPL",
            payload=b"a", range_start=date(2026, 1, 1),
            range_end=date(2026, 1, 2), ext="csv", timestamp=ts1,
        )
        p2 = write_raw_atomic(
            root, "stooq", "us_equities", "NYSE", "AAPL",
            payload=b"b", range_start=date(2026, 1, 1),
            range_end=date(2026, 1, 2), ext="csv", timestamp=ts2,
        )
        assert p1 != p2
        assert p1.exists() and p2.exists()


class TestListRaw:
    def test_empty_dir_returns_empty(self, root):
        assert list_raw(root, "stooq", "us_equities", "NYSE", "MISSING") == []

    def test_lists_sorted_by_timestamp(self, root):
        for h in (10, 12, 11):
            ts = datetime(2026, 5, 23, h, 0, 0, tzinfo=timezone.utc)
            write_raw_atomic(
                root, "stooq", "us_equities", "NYSE", "AAPL",
                payload=b"x", range_start=date(2026, 1, 1),
                range_end=date(2026, 1, 2), ext="csv", timestamp=ts,
            )
        paths = list_raw(root, "stooq", "us_equities", "NYSE", "AAPL")
        ts_hours = [parse_filename(p.name).timestamp.hour for p in paths]
        assert ts_hours == [10, 11, 12]

    def test_skips_non_matching_files(self, root, fixed_ts):
        write_raw_atomic(
            root, "stooq", "us_equities", "NYSE", "AAPL",
            payload=b"x", range_start=date(2026, 1, 1),
            range_end=date(2026, 1, 2), ext="csv", timestamp=fixed_ts,
        )
        # Stray junk file in the same dir.
        junk = raw_dir(root, "stooq", "us_equities", "NYSE", "AAPL") / "junk.txt"
        junk.write_text("hi")
        paths = list_raw(root, "stooq", "us_equities", "NYSE", "AAPL")
        assert len(paths) == 1
        assert paths[0].name.endswith(".csv")
