"""Tests for the daily data-update helpers (no network, no cache)."""
from __future__ import annotations

import sqlite3
import types

import pandas as pd
import pytest

from scripts.data_pipelines.daily_update import (
    BASIS_STEP_TOL,
    detect_basis_steps,
    detect_extreme_moves,
    last_completed_session,
    load_repair_state,
    save_repair_state,
    select_suspects,
    split_into_chunks,
)


class TestLastCompletedSession:
    def test_after_close_same_day_is_complete(self):
        # Tue 2026-10-13, 17:00 ET — the session has closed.
        now = pd.Timestamp("2026-10-13 17:00", tz="America/New_York")
        assert last_completed_session(now) == pd.Timestamp("2026-10-13")

    def test_during_session_is_not_complete(self):
        # 10:00 ET — still trading; the prior session is the last completed one.
        now = pd.Timestamp("2026-10-13 10:00", tz="America/New_York")
        assert last_completed_session(now) == pd.Timestamp("2026-10-12")

    def test_weekend_rolls_back_to_friday(self):
        now = pd.Timestamp("2026-10-17 12:00", tz="America/New_York")  # Saturday
        assert last_completed_session(now) == pd.Timestamp("2026-10-16")

    def test_ist_morning_does_not_count_open_us_session(self):
        """IST is a calendar day ahead of ET for the whole US session.

        At 00:45 IST on the 14th it is still 15:15 ET on the 13th — the US session
        is OPEN. A machine-local date check would wrongly treat the 13th as done;
        this function must not.
        """
        now = pd.Timestamp("2026-10-14 00:45", tz="Asia/Kolkata")
        assert now.tz_convert("America/New_York").date() == pd.Timestamp("2026-10-13").date()
        assert last_completed_session(now) == pd.Timestamp("2026-10-12")

    def test_ist_morning_after_us_close_counts_the_session(self):
        # 06:00 IST on the 14th == 20:30 ET on the 13th (EDT) — session closed.
        now = pd.Timestamp("2026-10-14 06:00", tz="Asia/Kolkata")
        assert last_completed_session(now) == pd.Timestamp("2026-10-13")

    def test_naive_timestamp_treated_as_eastern(self):
        now = pd.Timestamp("2026-10-13 17:00")
        assert last_completed_session(now) == pd.Timestamp("2026-10-13")


class TestSplitIntoChunks:
    def test_exact_multiple(self):
        s = [pd.Timestamp(f"2026-01-{i:02d}") for i in range(1, 5)]
        assert [len(c) for c in split_into_chunks(s, 2)] == [2, 2]

    def test_remainder(self):
        s = [pd.Timestamp(f"2026-01-{i:02d}") for i in range(1, 6)]
        assert [len(c) for c in split_into_chunks(s, 2)] == [2, 2, 1]

    def test_empty(self):
        assert split_into_chunks([], 3) == []

    def test_single_chunk_when_short(self):
        s = [pd.Timestamp("2026-01-01")]
        assert [len(c) for c in split_into_chunks(s, 90)] == [1]

    def test_rejects_nonpositive_max(self):
        with pytest.raises(ValueError):
            split_into_chunks([pd.Timestamp("2026-01-01")], 0)

    def test_chunks_are_contiguous_and_ordered(self):
        s = [pd.Timestamp(f"2026-02-{i:02d}") for i in range(1, 8)]
        flat = [d for c in split_into_chunks(s, 3) for d in c]
        assert flat == s


class TestDetectBasisSteps:
    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows, columns=["ticker", "date", "close", "adj_close"])

    def test_clean_series_is_not_flagged(self):
        # ratio constant at 1.0 across the whole series
        f = self._frame([
            ("NASDAQ:AAA", "2026-05-22", 100.0, 100.0),
            ("NASDAQ:AAA", "2026-05-26", 101.0, 101.0),
            ("NASDAQ:AAA", "2026-05-27", 102.0, 102.0),
        ])
        assert detect_basis_steps(f).empty

    def test_split_basis_step_is_flagged(self):
        # The 2026-10-14 bug shape: close continues, adj_close re-bases by 4x.
        f = self._frame([
            ("NASDAQ:CRWD", "2026-05-22", 663.46, 663.46),
            ("NASDAQ:CRWD", "2026-05-26", 671.55, 671.55),
            ("NASDAQ:CRWD", "2026-05-27", 645.36, 161.34),  # ratio steps 1 -> 4
            ("NASDAQ:CRWD", "2026-05-28", 671.00, 167.75),
        ])
        out = detect_basis_steps(f)
        assert list(out["ticker"]) == ["NASDAQ:CRWD"]
        assert out["date"].iloc[0] == pd.Timestamp("2026-05-27")
        assert out["step"].iloc[0] == pytest.approx(3.0, rel=1e-6)

    def test_reverse_split_is_not_a_ratio_step(self):
        # NYSE:DD shape — both series jump together, so the ratio barely moves.
        # This is why a ratio test alone is insufficient.
        f = self._frame([
            ("NYSE:DD", "2026-05-22", 48.12, 48.12),
            ("NYSE:DD", "2026-05-26", 148.38, 147.73),
        ])
        assert detect_basis_steps(f).empty

    def test_small_dividend_drift_is_not_flagged(self):
        # A sub-tolerance basis wobble (ordinary dividend re-adjustment).
        f = self._frame([
            ("NYSE:XOM", "2026-05-22", 100.0, 100.0),
            ("NYSE:XOM", "2026-05-26", 100.0, 100.0 * (1 - BASIS_STEP_TOL / 2)),
        ])
        assert detect_basis_steps(f).empty

    def test_high_yield_reit_dividend_step_is_not_flagged(self):
        """A real dividend step must not be read as a split.

        A mortgage REIT yielding ~13%/yr steps ~3.4% quarterly. A 2% threshold
        (an earlier calibration) flagged 8 such names in one run, 2026-10-14.
        """
        f = self._frame([
            ("NYSE:NLY", "2026-10-01", 20.0, 20.0),
            ("NYSE:NLY", "2026-10-02", 19.65, 19.65 * (1 - 0.034)),
        ])
        assert detect_basis_steps(f).empty

    def test_large_split_step_still_flagged_at_new_threshold(self):
        # ratio 1.0 -> 4.0 (4:1 forward split); well clear of any dividend.
        f = self._frame([
            ("NASDAQ:AAA", "2026-05-22", 400.0, 400.0),
            ("NASDAQ:AAA", "2026-05-26", 100.5, 100.5 / 4),
        ])
        out = detect_basis_steps(f)
        assert list(out["ticker"]) == ["NASDAQ:AAA"]
        assert out["step"].iloc[0] == pytest.approx(3.0, rel=1e-6)

    def test_per_ticker_isolation(self):
        # A step in one ticker must not flag its neighbour, and tickers must not
        # be compared against each other.
        f = self._frame([
            ("A", "2026-05-22", 10.0, 10.0),
            ("A", "2026-05-26", 10.0, 2.5),
            ("B", "2026-05-22", 500.0, 500.0),
            ("B", "2026-05-26", 505.0, 505.0),
        ])
        out = detect_basis_steps(f)
        assert set(out["ticker"]) == {"A"}

    def test_empty_frame(self):
        f = self._frame([])
        assert detect_basis_steps(f).empty

    def test_zero_and_null_adj_close_ignored(self):
        f = self._frame([
            ("A", "2026-05-22", 10.0, 0.0),
            ("A", "2026-05-26", 10.0, None),
            ("A", "2026-05-27", 10.0, 10.0),
        ])
        assert detect_basis_steps(f).empty


class TestDetectExtremeMoves:
    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows, columns=["ticker", "date", "close", "adj_close"])

    def test_reverse_split_shape_is_flagged(self):
        f = self._frame([
            ("NYSE:DD", "2026-05-22", 48.12, 48.12),
            ("NYSE:DD", "2026-05-26", 148.38, 147.73),  # +207%
        ])
        out = detect_extreme_moves(f)
        assert list(out["ticker"]) == ["NYSE:DD"]
        assert out["ret"].iloc[0] == pytest.approx(2.07, abs=0.01)

    def test_normal_series_not_flagged(self):
        f = self._frame([
            ("A", "2026-05-22", 100.0, 100.0),
            ("A", "2026-05-26", 103.0, 103.0),
            ("A", "2026-05-27", 99.0, 99.0),
        ])
        assert detect_extreme_moves(f).empty

    def test_threshold_is_configurable(self):
        f = self._frame([
            ("A", "2026-05-22", 100.0, 100.0),
            ("A", "2026-05-26", 120.0, 120.0),  # +20%
        ])
        assert detect_extreme_moves(f, threshold=0.40).empty
        assert not detect_extreme_moves(f, threshold=0.15).empty

    def test_per_ticker_isolation(self):
        f = self._frame([
            ("A", "2026-05-22", 100.0, 100.0),
            ("A", "2026-05-26", 300.0, 300.0),
            ("B", "2026-05-22", 100.0, 100.0),
            ("B", "2026-05-26", 101.0, 101.0),
        ])
        assert set(detect_extreme_moves(f)["ticker"]) == {"A"}

    def test_empty(self):
        assert detect_extreme_moves(self._frame([])).empty


class TestSelectSuspects:
    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows, columns=["ticker", "date", "close", "adj_close"])

    def test_union_of_both_detectors(self):
        f = self._frame([
            # forward split -> ratio step
            ("RATIO", "2026-05-22", 100.0, 100.0),
            ("RATIO", "2026-05-26", 100.0, 25.0),
            # reverse split -> extreme move
            ("MOVE", "2026-05-22", 48.0, 48.0),
            ("MOVE", "2026-05-26", 148.0, 147.0),
            ("CLEAN", "2026-05-22", 10.0, 10.0),
            ("CLEAN", "2026-05-26", 10.2, 10.2),
        ])
        suspects, skipped = select_suspects(f, move_threshold=0.40, state={}, lookback_days=45)
        assert sorted(suspects) == ["MOVE", "RATIO"]
        assert skipped == []

    def test_recently_repaired_are_skipped(self):
        f = self._frame([
            ("MOVE", "2026-05-22", 48.0, 48.0),
            ("MOVE", "2026-05-26", 148.0, 147.0),
        ])
        today = pd.Timestamp.now("UTC").normalize().strftime("%Y-%m-%d")
        suspects, skipped = select_suspects(f, move_threshold=0.40,
                                            state={"MOVE": today}, lookback_days=45)
        assert suspects == []
        assert skipped == ["MOVE"]

    def test_stale_repair_entries_do_not_skip(self):
        f = self._frame([
            ("MOVE", "2026-05-22", 48.0, 48.0),
            ("MOVE", "2026-05-26", 148.0, 147.0),
        ])
        suspects, _ = select_suspects(f, move_threshold=0.40,
                                      state={"MOVE": "2000-01-01"}, lookback_days=45)
        assert suspects == ["MOVE"]

    def test_empty_frame(self):
        assert select_suspects(pd.DataFrame(), move_threshold=0.4,
                               state={}, lookback_days=45) == ([], [])


class TestRepairState:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "sub" / "state.json"
        assert load_repair_state(p) == {}
        save_repair_state(p, {"NYSE:DD": "2026-10-14"})
        assert load_repair_state(p) == {"NYSE:DD": "2026-10-14"}

    def test_corrupt_state_is_treated_as_empty(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("{not json")
        assert load_repair_state(p) == {}


class TestRepairCrashSafety:
    """The repair must never leave a ticker with zero rows.

    The naive form (delete every suspect, then re-fetch in a loop) loses data if a
    provider fails or the process is killed mid-loop: the ticker ends up empty,
    invisible to the seam scan, and no scheduled seed fetches enough history to
    restore it. These tests pin the journalled, per-ticker, recoverable behaviour.
    """

    @staticmethod
    def _mkdb(tmp_path):
        db = tmp_path / "processed.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE us_equities_data "
                    "(ticker TEXT, date TEXT, close REAL, adj_close REAL)")
        con.execute("CREATE TABLE us_equities_meta (ticker TEXT, schema_version INTEGER)")
        for t in ("A", "B"):
            con.execute("INSERT INTO us_equities_data VALUES (?, '2026-01-01', 1.0, 1.0)", (t,))
            con.execute("INSERT INTO us_equities_meta VALUES (?, 1)", (t,))
        con.commit()
        con.close()
        return db

    @staticmethod
    def _rows(db, ticker):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return con.execute("SELECT COUNT(*) FROM us_equities_data WHERE ticker=?",
                               (ticker,)).fetchone()[0]
        finally:
            con.close()

    def _call(self, tmp_path, monkeypatch, tickers, *, fetch, reprocess):
        import scripts.data_pipelines.daily_update as du

        monkeypatch.setattr(du, "_full_refetch", fetch)
        monkeypatch.setattr(du, "_reprocess_from_raw", reprocess)
        state, pending = {}, {}
        ok = du.repair_basis_seams(
            tickers, pd.Timestamp("2026-10-13"), tmp_path, repair=True, max_repairs=10,
            state_path=tmp_path / "state.json", state=state,
            pending_path=tmp_path / "pending.json", pending=pending)
        return ok, state, pending

    def test_successful_repair_records_state_and_clears_journal(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path)

        def fetch(ticker, end, data_root):
            con = sqlite3.connect(db)
            con.execute("INSERT INTO us_equities_data VALUES (?, '2026-02-01', 9.0, 9.0)", (ticker,))
            con.commit()
            con.close()
            return 5

        ok, state, pending = self._call(tmp_path, monkeypatch, ["A", "B"],
                                        fetch=fetch, reprocess=lambda t, dr: True)
        assert ok is True
        assert set(state) == {"A", "B"}
        assert pending == {}                       # journal drained
        assert self._rows(db, "A") == 1            # old row replaced, not duplicated
        assert self._rows(db, "B") == 1
        assert load_repair_state(tmp_path / "pending.json") == {}

    def test_failed_fetch_falls_back_to_raw_and_is_not_lost(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path)
        reprocessed = []

        def fetch(ticker, end, data_root):
            if ticker == "B":
                raise RuntimeError("all providers down")
            return 5

        def reprocess(ticker, data_root):
            reprocessed.append(ticker)
            # offline rebuild restores the pre-repair rows
            con = sqlite3.connect(db)
            con.execute("INSERT INTO us_equities_data VALUES (?, '2026-01-01', 2.0, 2.0)", (ticker,))
            con.commit()
            con.close()
            return True

        ok, state, pending = self._call(tmp_path, monkeypatch, ["A", "B"],
                                        fetch=fetch, reprocess=reprocess)
        assert ok is False                          # surfaced as a failure
        assert reprocessed == ["B"]
        assert self._rows(db, "B") == 1             # B is NOT empty
        assert "B" not in pending                   # recovered → journal cleared
        assert state.get("A") and "B" not in state  # B is retried next run

    def test_unrecovered_ticker_stays_journalled(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path)

        def fetch(ticker, end, data_root):
            if ticker == "B":
                raise RuntimeError("all providers down")
            return 5

        ok, state, pending = self._call(tmp_path, monkeypatch, ["A", "B"],
                                        fetch=fetch, reprocess=lambda t, dr: False)
        assert ok is False
        assert "B" in pending
        # the journal on disk is what a later run heals from
        assert "B" in load_repair_state(tmp_path / "pending.json")
        assert self._rows(db, "B") == 0             # exposed, but journalled

    def test_next_run_heals_a_leftover_journal(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path)
        (tmp_path / "pending.json").write_text('{"B": "2026-10-13"}')

        def fetch(ticker, end, data_root):
            con = sqlite3.connect(db)
            con.execute("INSERT INTO us_equities_data VALUES (?, '2026-02-01', 3.0, 3.0)", (ticker,))
            con.commit()
            con.close()
            return 4

        import scripts.data_pipelines.daily_update as du
        monkeypatch.setattr(du, "_full_refetch", fetch)
        monkeypatch.setattr(du, "_reprocess_from_raw", lambda t, dr: True)
        state = {}
        pending = load_repair_state(tmp_path / "pending.json")
        ok = du.repair_basis_seams([], pd.Timestamp("2026-10-13"), tmp_path, repair=True,
                                   max_repairs=10, state_path=tmp_path / "state.json",
                                   state=state, pending_path=tmp_path / "pending.json",
                                   pending=pending)
        assert ok is True
        assert self._rows(db, "B") == 1             # healed
        assert load_repair_state(tmp_path / "pending.json") == {}

    def test_journal_is_written_before_the_rows_are_deleted(self, tmp_path, monkeypatch):
        """A crash between the delete and the re-fetch must still be recoverable."""
        db = self._mkdb(tmp_path)
        observed = {}

        def fetch(ticker, end, data_root):
            observed["journal_during_fetch"] = load_repair_state(tmp_path / "pending.json")
            raise RuntimeError("killed mid-loop")

        ok, state, pending = self._call(tmp_path, monkeypatch, ["B"],
                                        fetch=fetch, reprocess=lambda t, dr: False)
        assert ok is False
        assert observed["journal_during_fetch"].get("B"), \
            "the ticker must be journalled before its rows are touched"

    def test_no_repair_leaves_cache_untouched(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path)
        import scripts.data_pipelines.daily_update as du
        monkeypatch.setattr(du, "_full_refetch",
                            lambda *a, **k: pytest.fail("must not fetch"))
        ok = du.repair_basis_seams(["A", "B"], pd.Timestamp("2026-10-13"), tmp_path,
                                   repair=False, max_repairs=10,
                                   state_path=tmp_path / "state.json", state={},
                                   pending_path=tmp_path / "pending.json", pending={})
        assert ok is True
        assert self._rows(db, "A") == 1 and self._rows(db, "B") == 1

    def test_max_repairs_aborts_before_any_delete(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path)
        import scripts.data_pipelines.daily_update as du
        monkeypatch.setattr(du, "_full_refetch", lambda *a, **k: pytest.fail("must not fetch"))
        ok = du.repair_basis_seams(["A", "B"], pd.Timestamp("2026-10-13"), tmp_path,
                                   repair=True, max_repairs=1,
                                   state_path=tmp_path / "state.json", state={},
                                   pending_path=tmp_path / "pending.json", pending={})
        assert ok is False
        assert self._rows(db, "A") == 1 and self._rows(db, "B") == 1


class TestReprocessRecoverySemantics:
    """`_reprocess_from_raw` must judge success by rows restored, not exit code.

    `data_pipelines reprocess` returns 0 unconditionally — including its
    "no raw files; skipping" branch and the case where every raw file fails to
    parse. Treating that as recovery pops the journal entry and leaves the ticker
    empty and invisible to the seam scan: the same silent-loss class the recovery
    path exists to prevent. These tests exercise the REAL function (the repair
    tests stub it), so its return semantics are actually pinned.
    """

    def _mkdb(self, tmp_path, tickers=("A",)):
        db = tmp_path / "processed.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE us_equities_data "
                    "(ticker TEXT, date TEXT, close REAL, adj_close REAL)")
        con.execute("CREATE TABLE us_equities_meta (ticker TEXT, schema_version INTEGER)")
        for t in tickers:
            con.execute("INSERT INTO us_equities_data VALUES (?, '2026-01-01', 1.0, 1.0)", (t,))
            con.execute("INSERT INTO us_equities_meta VALUES (?, 1)", (t,))
        con.commit()
        con.close()
        return db

    def _stub_run(self, monkeypatch, du, *, returncode, writes_rows_for=None, db=None):
        def fake_run(*a, **k):
            if writes_rows_for and db is not None:
                con = sqlite3.connect(db)
                con.execute("INSERT INTO us_equities_data VALUES (?, '2026-01-01', 5.0, 5.0)",
                            (writes_rows_for,))
                con.commit()
                con.close()
            return types.SimpleNamespace(returncode=returncode,
                                         stdout="reprocess: no raw files; skipping", stderr="")
        monkeypatch.setattr(du.subprocess, "run", fake_run)

    def test_rc0_with_no_rows_written_is_not_recovery(self, tmp_path, monkeypatch):
        self._mkdb(tmp_path, tickers=())
        import scripts.data_pipelines.daily_update as du
        self._stub_run(monkeypatch, du, returncode=0)  # rc=0, writes nothing
        assert du._reprocess_from_raw("B", tmp_path) is False
        assert du._ticker_row_count(tmp_path, "B") == 0

    def test_rc0_with_rows_written_is_recovery(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path, tickers=())
        import scripts.data_pipelines.daily_update as du
        self._stub_run(monkeypatch, du, returncode=0, writes_rows_for="B", db=db)
        assert du._reprocess_from_raw("B", tmp_path) is True
        assert du._ticker_row_count(tmp_path, "B") == 1

    def test_nonzero_rc_is_not_recovery_even_if_rows_exist(self, tmp_path, monkeypatch):
        db = self._mkdb(tmp_path, tickers=("B",))
        import scripts.data_pipelines.daily_update as du
        self._stub_run(monkeypatch, du, returncode=2)
        assert du._reprocess_from_raw("B", tmp_path) is False


class TestHealLoopPerTicker:
    def test_one_failed_heal_does_not_strand_later_successes(self, tmp_path, monkeypatch):
        db = TestRepairCrashSafety._mkdb(tmp_path)
        (tmp_path / "pending.json").write_text('{"A": "2026-10-13", "B": "2026-10-13"}')
        import scripts.data_pipelines.daily_update as du

        def fetch(*a, **k):
            raise RuntimeError("providers down")

        def reprocess(ticker, data_root):
            return ticker == "B"          # A cannot be recovered, B can

        monkeypatch.setattr(du, "_full_refetch", fetch)
        monkeypatch.setattr(du, "_reprocess_from_raw", reprocess)
        state = {}
        pending = load_repair_state(tmp_path / "pending.json")
        ok = du.repair_basis_seams([], pd.Timestamp("2026-10-13"), tmp_path, repair=True,
                                   max_repairs=10, state_path=tmp_path / "state.json",
                                   state=state, pending_path=tmp_path / "pending.json",
                                   pending=pending)
        assert ok is False                       # A is unresolved
        assert "A" in pending                    # stays journalled
        assert "B" not in pending                # NOT stranded by A's failure
        assert load_repair_state(tmp_path / "pending.json") == {"A": "2026-10-13"}
