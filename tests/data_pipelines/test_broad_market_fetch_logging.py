"""Unit tests for the structured-logging helpers used by the back-extend
driver scripts (``scripts/data_pipelines/_fetch_logging.py``).

These cover the pure-function classifier, the rolling-summary renderer, and
the summary.json writer. They explicitly do NOT invoke the live
``python -m data_pipelines fetch`` subprocess — that path is network- and
provider-dependent and would make the suite slow/flaky.
"""
from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pytest

from scripts.data_pipelines._fetch_logging import (
    ALL_STATUSES,
    STATUS_FAIL_EMPTY,
    STATUS_FAIL_OTHER,
    STATUS_FAIL_OVERLAP,
    STATUS_OK,
    STATUS_OK_NOOP,
    RollingCounter,
    SummaryWriter,
    classify_error,
    classify_outcome,
    extract_err_short,
    format_per_ticker_line,
)


# ---------------------------------------------------------------------------
# classify_error / classify_outcome
# ---------------------------------------------------------------------------


def test_classify_error_empty_payload():
    stderr = (
        "Traceback (most recent call last):\n"
        "  File ...\n"
        "data_pipelines.errors.AllProvidersFailed: NSE:BERGEPAINT: "
        "all providers failed — jugaad: empty payload; nselib: empty "
        "payload; yfinance: empty payload\n"
    )
    assert classify_error(stderr) == STATUS_FAIL_EMPTY


def test_classify_error_overlap_integrity():
    stderr = (
        "sqlite3.IntegrityError: UNIQUE constraint failed: "
        "nse_equities_data.ticker, nse_equities_data.date\n"
    )
    assert classify_error(stderr) == STATUS_FAIL_OVERLAP


def test_classify_error_overlap_lowercase_unique_constraint():
    # The IntegrityError class name may be stripped on some traceback formats;
    # the "UNIQUE constraint failed" substring alone is enough.
    stderr = "UNIQUE constraint failed: nse_equities_data.ticker"
    assert classify_error(stderr) == STATUS_FAIL_OVERLAP


def test_classify_error_other_bucket():
    stderr = "RuntimeError: provider rate-limited\n"
    assert classify_error(stderr) == STATUS_FAIL_OTHER


def test_classify_error_empty_string_is_other():
    assert classify_error("") == STATUS_FAIL_OTHER


def test_classify_outcome_ok_with_rows():
    assert classify_outcome(rc=0, delta=1246, stderr="") == STATUS_OK


def test_classify_outcome_noop_when_no_rows_added():
    assert classify_outcome(rc=0, delta=0, stderr="") == STATUS_OK_NOOP


def test_classify_outcome_failure_dispatches_to_classifier():
    stderr = "AllProvidersFailed: ... empty payload ..."
    assert classify_outcome(rc=1, delta=0, stderr=stderr) == STATUS_FAIL_EMPTY


# ---------------------------------------------------------------------------
# extract_err_short
# ---------------------------------------------------------------------------


def test_extract_err_short_grabs_last_exception_line():
    stderr = (
        "Traceback (most recent call last):\n"
        "  File foo.py\n"
        "data_pipelines.errors.AllProvidersFailed: NSE:FOO: empty payload\n"
    )
    s = extract_err_short(stderr)
    assert "AllProvidersFailed" in s
    assert "empty payload" in s


def test_extract_err_short_truncates_to_max_len():
    long_msg = "x" * 500
    stderr = f"RuntimeError: {long_msg}"
    s = extract_err_short(stderr, max_len=80)
    assert len(s) <= 80


def test_extract_err_short_handles_no_exception_line():
    stderr = "some stderr noise without an exception\n"
    s = extract_err_short(stderr)
    assert "some stderr noise" in s


# ---------------------------------------------------------------------------
# format_per_ticker_line
# ---------------------------------------------------------------------------


def test_format_per_ticker_line_shape():
    line = format_per_ticker_line(
        idx=92, total=697, sym="NSE:BERGEPAINT",
        status="FAIL_EMPTY", before=0, after=0, delta=0,
        took_s=20.7, elapsed_m=35.6, eta_m=234.2,
        err_short="AllProvidersFailed:empty_payload",
        ts="2026-05-27T14:33:44Z",
    )
    assert line == (
        "2026-05-27T14:33:44Z [ 92/697] NSE:BERGEPAINT "
        "status=FAIL_EMPTY rows=0->0(+0) took=20.7s "
        "err=AllProvidersFailed:empty_payload "
        "elapsed=35.6m eta=234.2m"
    )


def test_format_per_ticker_line_ok_no_err_field():
    line = format_per_ticker_line(
        idx=1, total=10, sym="NSE:RELIANCE",
        status="OK", before=0, after=1246, delta=1246,
        took_s=2.3, elapsed_m=0.1, eta_m=0.9,
        ts="2026-05-27T14:00:00Z",
    )
    assert "err=" not in line
    assert "status=OK" in line
    assert "rows=0->1246(+1246)" in line


# ---------------------------------------------------------------------------
# RollingCounter
# ---------------------------------------------------------------------------


def test_rolling_counter_render_includes_all_statuses_and_rate():
    t0 = time.time() - 60.0  # pretend we've been running for 60 seconds
    ctr = RollingCounter(total=697, t_start=t0, interval=50)
    # 100 ticks processed: 12 OK, 3 NOOP, 82 EMPTY, 3 OTHER
    for _ in range(12):
        ctr.bump(STATUS_OK)
    for _ in range(3):
        ctr.bump(STATUS_OK_NOOP)
    for _ in range(82):
        ctr.bump(STATUS_FAIL_EMPTY)
    for _ in range(3):
        ctr.bump(STATUS_FAIL_OTHER)

    line = ctr.render(done=100, ts="2026-05-27T14:33:44Z")
    assert line.startswith(
        "2026-05-27T14:33:44Z [ROLLING 100/697 done=14% elapsed="
    )
    # All bucket names present, with the expected counts.
    assert "OK=12" in line
    assert "OK_NOOP=3" in line
    assert "FAIL_EMPTY=82" in line
    assert "FAIL_OVERLAP=0" in line
    assert "FAIL_OTHER=3" in line
    assert "rate=" in line
    assert "eta=" in line


def test_rolling_counter_bumps_initialise_all_buckets():
    ctr = RollingCounter(total=10, t_start=time.time())
    for s in ALL_STATUSES:
        assert s in ctr.counts


# ---------------------------------------------------------------------------
# SummaryWriter
# ---------------------------------------------------------------------------


def test_summary_writer_shape(tmp_path: Path):
    out = tmp_path / "summary.json"
    sw = SummaryWriter(
        out_path=out,
        total_tickers=3,
        fetch_plan_path="/tmp/test_fetch_plan.json",
    )
    sw.record(status=STATUS_OK, sym="NSE:RELIANCE", extra="+1246 rows")
    sw.record(status=STATUS_FAIL_EMPTY, sym="NSE:360ONE")
    sw.record(status=STATUS_FAIL_OTHER, sym="NSE:FOO",
              extra="RuntimeError: rate limit")

    path = sw.finalize()
    assert path == out
    payload = json.loads(out.read_text())

    # Top-level shape.
    assert payload["total_tickers"] == 3
    assert payload["completed"] == 3
    assert payload["fetch_plan_path"] == "/tmp/test_fetch_plan.json"
    assert payload["fetch_plan_size"] == 3
    assert "start_utc" in payload and "end_utc" in payload
    assert isinstance(payload["elapsed_minutes"], (int, float))
    assert isinstance(payload["wrote_summary_at_pid"], int)

    # Outcomes & examples.
    assert payload["outcomes"]["OK"] == 1
    assert payload["outcomes"]["FAIL_EMPTY"] == 1
    assert payload["outcomes"]["FAIL_OTHER"] == 1
    assert payload["outcomes"]["OK_NOOP"] == 0
    assert payload["outcomes"]["FAIL_OVERLAP"] == 0

    assert payload["per_status_examples"]["OK"] == ["NSE:RELIANCE +1246 rows"]
    assert payload["per_status_examples"]["FAIL_EMPTY"] == ["NSE:360ONE"]
    assert payload["per_status_examples"]["FAIL_OVERLAP"] == []


def test_summary_writer_examples_capped_at_five(tmp_path: Path):
    sw = SummaryWriter(out_path=tmp_path / "s.json", total_tickers=100)
    for i in range(20):
        sw.record(status=STATUS_FAIL_EMPTY, sym=f"NSE:T{i}")
    sw.finalize()
    payload = json.loads((tmp_path / "s.json").read_text())
    assert len(payload["per_status_examples"]["FAIL_EMPTY"]) == 5
    # First five tickers, in insertion order.
    assert payload["per_status_examples"]["FAIL_EMPTY"][0] == "NSE:T0"
    assert payload["per_status_examples"]["FAIL_EMPTY"][4] == "NSE:T4"
    # But the count reflects all 20.
    assert payload["outcomes"]["FAIL_EMPTY"] == 20


def test_summary_writer_finalize_is_idempotent(tmp_path: Path):
    out = tmp_path / "summary.json"
    sw = SummaryWriter(out_path=out, total_tickers=1)
    sw.record(status=STATUS_OK, sym="NSE:X", extra="+1 rows")
    sw.finalize()
    first_mtime = out.stat().st_mtime_ns
    # Second call must NOT rewrite (idempotent).
    sw.finalize()
    assert out.stat().st_mtime_ns == first_mtime


def test_summary_writer_signal_handler_registration(tmp_path: Path):
    """Verify install_signal_handlers actually registers SIGINT+SIGTERM."""
    out = tmp_path / "summary.json"
    sw = SummaryWriter(out_path=out, total_tickers=1)
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        sw.install_signal_handlers(log_fn=lambda _msg: None)
        assert signal.getsignal(signal.SIGINT) is not prev_int
        assert signal.getsignal(signal.SIGTERM) is not prev_term
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
