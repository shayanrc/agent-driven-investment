"""Daily ``us_equities`` cache refresh for the tracked model universes.

Why this exists separately from the /daily-predictions cadence
-------------------------------------------------------------
``scripts/backtests/daily_forward_predictions.py`` seeds the cache as step 2 of
its *scoring* run. That couples two different concerns: keeping the data fresh,
and re-scoring cells. The cadence fails closed — one cell that cannot reproduce
its training book aborts the run — so when a cell is blocked the data refresh is
hostage to it. This script refreshes the data on its own schedule.

What it does
------------
1. Disk pre-flight (>= 10 GB free; the FS wedges near-full per the project's
   wedge pattern).
2. Compute the last **completed** US session from the NYSE calendar in
   America/New_York terms. This is deliberately ET-based, not machine-local: a
   run early in the IST morning can otherwise treat a still-open US session as
   finished (IST is a calendar day ahead of ET for the whole US session).
3. Seed each tracked universe to that session, in <=90-session chunks (see
   ``_chunks``), with parallel fetch workers.
4. Detect and repair **corporate-action basis seams** — see ``detect_basis_steps``.
   This is the bug class found 2026-10-14: rows fetched at time T0 sit on the
   adjustment basis in force then, rows fetched at T1 on the basis in force now,
   so any ticker with a split in between gets a discontinuity *exactly at the
   seam* (close continuous, adj_close stepping by the split factor, or vice
   versa). A daily incremental refresh recreates this continuously, so the guard
   is part of the refresh, not an occasional cleanup. The repair is crash-safe:
   each ticker is journalled to disk before its rows are touched, and a failed
   re-fetch falls back to the immutable raw store offline.
5. Optionally (``--with-valuation``, intended for a weekly timer) refresh the
   ``us_fundamentals`` cache and rebuild the point-in-time valuation panel.

Exit code is non-zero if any universe, repair, or optional step failed, so a
timer surfaces the failure instead of silently drifting stale.

Note on the fundamentals leg: the documented chain is macrotrends -> SEC EDGAR
-> yfinance. macrotrends is HTTP-403 IP-blocked on some hosts; when that happens
EDGAR serves every row (it carries native ``filed_date``, so the point-in-time
lag is unaffected). The provider actually used is reported per run.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# The universes the daily cadence scores cells on. Kept in step with
# daily_forward_predictions.CELLS (sp500 / russell1000 / nasdaq100).
DEFAULT_UNIVERSES = ("sp500", "russell1000", "nasdaq100")

MIN_FREE_GB = 10.0
MAX_SEED_CHUNK_SESSIONS = 90  # > this trips the domain's Stooq-first "big gap" tier
DEFAULT_JOBS = 8

# A close/adj_close ratio step larger than this is a basis change (split /
# reverse split / spin-off rebasing), not a price move.
#
# CALIBRATION MATTERS HERE. `close/adj_close` steps *routinely* at every
# ex-dividend date, because adj_close is reduced for the dividend — that is
# normal, not corruption. The step is the dividend yield per period, so the
# highest-yielding names (mortgage REITs at ~13%/yr) step ~3-4% quarterly. A
# split steps by its factor (2:1 -> 100%, 3:2 -> 50%). The gap between the two
# populations is wide, so the threshold sits in the middle: 10% clears every
# plausible dividend step while catching any real split. A 2% threshold (an
# earlier calibration) false-positived on 8 high-yield REITs in a single run.
BASIS_STEP_TOL = 0.10


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested without network or a cache)
# --------------------------------------------------------------------------- #
def last_completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent NYSE session whose close is already in the past (ET terms).

    Uses America/New_York so a machine-local (e.g. IST) calendar date that has
    already rolled over does not count a still-open US session as complete.
    """
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    now_et = pd.Timestamp.now(tz="America/New_York") if now is None else now
    if now_et.tzinfo is None:
        now_et = now_et.tz_localize("America/New_York")
    else:
        now_et = now_et.tz_convert("America/New_York")

    sched = nyse.schedule(
        start_date=(now_et - pd.Timedelta(days=20)).date(),
        end_date=(now_et + pd.Timedelta(days=1)).date(),
    )
    closed = sched[sched["market_close"] <= now_et]
    if closed.empty:  # pragma: no cover - only on a >20-day market closure
        raise RuntimeError("no completed NYSE session in the last 20 days")
    return pd.Timestamp(closed.index[-1]).normalize()


def split_into_chunks(sessions: list[pd.Timestamp], max_len: int) -> list[list[pd.Timestamp]]:
    """Split a session list into consecutive runs of at most ``max_len``."""
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    return [sessions[i : i + max_len] for i in range(0, len(sessions), max_len)]


def detect_basis_steps(frame: pd.DataFrame, tol: float = BASIS_STEP_TOL) -> pd.DataFrame:
    """Flag tickers whose close/adj_close ratio steps between sessions.

    ``frame`` needs columns ``ticker, date, close, adj_close``. A constant ratio
    means the raw and adjusted series share one basis; a step means one side was
    re-based (a forward split applied to only part of the history). Returns the
    offending rows ordered by ticker/date.

    Catches the *forward*-split shape only. A reverse split can leave the ratio
    untouched while both series jump — see ``detect_extreme_moves``.
    """
    if frame.empty:
        return frame.assign(ratio=pd.Series(dtype=float), step=pd.Series(dtype=float))
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df = df[df["adj_close"].notna() & (df["adj_close"] != 0)]
    df["ratio"] = df["close"] / df["adj_close"]
    df["step"] = df.groupby("ticker")["ratio"].pct_change()
    return df[df["step"].abs() > tol]


def detect_extreme_moves(frame: pd.DataFrame,
                         threshold: float = 0.40) -> pd.DataFrame:
    """Flag sessions with an implausibly large 1-day move in ``adj_close``.

    A forward split is caught by the ratio test above, but a **reverse** split
    re-bases *both* series: the ratio stays ~1 while the price level triples
    (the NYSE:DD 48 -> 148 case, volume ÷3). Nothing in a single price column can
    separate that from a genuine earnings gap — the discriminator is that the
    artifact *disappears* once the whole history is re-fetched from one provider,
    which is exactly what the repair does. This detector is therefore a candidate
    generator, deliberately recall-heavy, not a verdict.
    """
    if frame.empty:
        return frame.assign(ret=pd.Series(dtype=float))
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df = df[df["adj_close"].notna() & (df["adj_close"] != 0)]
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
    return df[df["ret"].abs() > threshold]


def load_repair_state(path: Path) -> dict:
    import json

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_repair_state(path: Path, state: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def _preflight_disk(root: Path) -> None:
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < MIN_FREE_GB:
        sys.exit(f"[ABORT] only {free_gb:.1f} G free (< {MIN_FREE_GB} G) — "
                 "refusing to run (FS-wedge guard).")
    print(f"[preflight] disk OK: {free_gb:.0f} G free", flush=True)


def _universe_identifiers(universe: str) -> list[str]:
    from data_pipelines.domains.us_equities.universe import load_universe

    return load_universe(universe)


def _chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end] into spans of <= MAX_SEED_CHUNK_SESSIONS NYSE sessions.

    Gaps wider than the domain's ``big_gap_threshold_days`` (90) route through
    the Stooq seed tier first, which is IP-gated here and costs ~75 s/ticker in
    failed attempts before falling through. Chunking keeps the much faster
    direct update chain and produces the same rows.
    """
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    sessions = list(nyse.valid_days(start_date=start.date(), end_date=end.date()))
    if not sessions:
        return []
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for run in split_into_chunks(sessions, MAX_SEED_CHUNK_SESSIONS):
        out.append((pd.Timestamp(run[0]).normalize(), pd.Timestamp(run[-1]).normalize()))
    return out


def _cache_max_date(data_root: Path, identifiers: list[str]) -> pd.Timestamp | None:
    """Latest cached session across ``identifiers`` (None on a cold cache)."""
    db = data_root / "processed.db"
    if not db.exists() or not identifiers:
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    try:
        placeholders = ",".join("?" * len(identifiers))
        row = con.execute(
            f"SELECT MAX(date) FROM us_equities_data WHERE ticker IN ({placeholders})",
            tuple(identifiers),
        ).fetchone()
    finally:
        con.close()
    return pd.Timestamp(row[0]).normalize() if row and row[0] else None


def seed_universe(universe: str, end: pd.Timestamp, jobs: int, data_root: Path,
                  *, fallback_days: int = 30) -> bool:
    """Bring one universe up to ``end``, starting from whatever the cache holds.

    The start is derived from the cache rather than a fixed trailing window: a
    fixed window only ever refills that window, so an outage longer than it would
    leave a hole that never closes (each later run would refill the same short
    tail). Starting at the cache's own last session means any outage length is
    caught up, with ``_chunks`` keeping the request bounded.
    """
    identifiers = _universe_identifiers(universe)
    cache_max = _cache_max_date(data_root, identifiers)
    start = (cache_max - pd.Timedelta(days=1)) if cache_max is not None \
        else (end - pd.Timedelta(days=fallback_days))
    if start > end:
        start = end

    spans = _chunks(start, end)
    if not spans:
        print(f"[seed] {universe}: no NYSE sessions in "
              f"{start.date()}..{end.date()} — nothing to do", flush=True)
        return True
    print(f"[seed] {universe}: cache_max={cache_max.date() if cache_max is not None else 'cold'} "
          f"→ spanning {start.date()}..{end.date()} in {len(spans)} chunk(s)", flush=True)
    ok = True
    for lo, hi in spans:
        # NB: --data-root is a GLOBAL option and must precede the subcommand.
        cmd = [sys.executable, "-m", "data_pipelines", "--data-root", str(data_root),
               "seed", "--domain", "us_equities", "--universe", universe,
               "--start", str(lo.date()), "--end", str(hi.date()),
               "--jobs", str(jobs)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"[seed] {universe} {lo.date()}..{hi.date()}: {tail}", flush=True)
        if r.returncode != 0:
            ok = False
            print((r.stderr or r.stdout).strip()[-600:], file=sys.stderr, flush=True)
    return ok


def _scan_window(data_root: Path, identifiers: list[str], lookback_days: int = 45) -> pd.DataFrame:
    db = data_root / "processed.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    try:
        placeholders = ",".join("?" * len(identifiers))
        cutoff = (pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return pd.read_sql(
            f"SELECT ticker,date,close,adj_close FROM us_equities_data "
            f"WHERE ticker IN ({placeholders}) AND date >= ?",
            con, params=(*identifiers, cutoff),
        )
    finally:
        con.close()


def select_suspects(frame: pd.DataFrame, *, move_threshold: float,
                    state: dict, lookback_days: int) -> tuple[list[str], list[str]]:
    """Union of both detectors, minus tickers repaired inside the lookback window.

    Returns ``(suspects, skipped_recent)``. The skip list is what stops a real
    +40% earnings move — which stays after any re-fetch, so it is re-detected
    every run — from triggering a repair 45 days in a row.
    """
    if frame.empty:
        return [], []
    by_ratio = set(detect_basis_steps(frame)["ticker"])
    by_move = set(detect_extreme_moves(frame, move_threshold)["ticker"])
    allf = sorted(by_ratio | by_move)
    cutoff = (pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    suspects, skipped = [], []
    for t in allf:
        if state.get(t, "") >= cutoff:
            skipped.append(t)
        else:
            suspects.append(t)
    if by_ratio:
        print(f"[seam] ratio steps ({len(by_ratio)}): {sorted(by_ratio)}", flush=True)
    if by_move:
        print(f"[seam] large moves >{move_threshold:.0%} ({len(by_move)}): {sorted(by_move)}", flush=True)
    if skipped:
        print(f"[seam] skipped (repaired within {lookback_days}d): {skipped}", flush=True)
    return suspects, skipped


def _delete_ticker(db: Path, ticker: str) -> None:
    con = sqlite3.connect(db, timeout=120)
    try:
        con.execute("DELETE FROM us_equities_data WHERE ticker=?", (ticker,))
        con.execute("DELETE FROM us_equities_meta WHERE ticker=?", (ticker,))
        con.commit()
    finally:
        con.close()


def _ticker_row_count(data_root: Path, ticker: str) -> int:
    db = data_root / "processed.db"
    con = sqlite3.connect(db, timeout=60)
    try:
        return con.execute("SELECT COUNT(*) FROM us_equities_data WHERE ticker=?",
                           (ticker,)).fetchone()[0]
    finally:
        con.close()


def _reprocess_from_raw(ticker: str, data_root: Path) -> bool:
    """Rebuild one ticker's processed rows from the immutable raw store (offline).

    The raw downloads are never mutated, so this restores the pre-repair series
    without any network call — the recovery path when a re-fetch fails.

    Success is judged by whether rows actually came back, NOT by the exit code:
    ``data_pipelines reprocess`` returns 0 unconditionally, including its
    "no raw files; skipping" branch and the case where every raw file fails to
    parse and nothing is written. Trusting the exit code would pop the journal
    entry on a no-op and leave the ticker empty and invisible to the seam scan —
    the very silent-loss class this recovery path exists to prevent.
    """
    cmd = [sys.executable, "-m", "data_pipelines", "--data-root", str(data_root),
           "reprocess", "--identifier", ticker]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    rows = _ticker_row_count(data_root, ticker)
    if r.returncode != 0 or rows == 0:
        detail = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
        print(f"[seam]   recovery FAILED {ticker} (rc={r.returncode}, rows={rows}): {detail}",
              file=sys.stderr, flush=True)
        return False
    print(f"[seam]   recovered {ticker} from raw: {rows} rows", flush=True)
    return True


def _full_refetch(ticker: str, end: pd.Timestamp, data_root: Path) -> int:
    import data_pipelines.domains.us_equities  # noqa: F401  (registers the domain)
    from data_pipelines import fetch_with_meta

    _, meta = fetch_with_meta(identifier=ticker, start="2000-01-01",
                              end=str(end.date()), data_root=data_root)
    return meta.row_count


def _repair_one(ticker: str, db: Path, end: pd.Timestamp, data_root: Path) -> int:
    """Delete one ticker and re-fetch its full history from a single response.

    The delete is unconditional even on the heal path: a crash can land either
    side of it, so healing a journalled ticker must work whether its rows are
    still present (crash before the delete) or already gone (crash after).
    """
    _delete_ticker(db, ticker)
    return _full_refetch(ticker, end, data_root)


def repair_basis_seams(tickers: list[str], end: pd.Timestamp, data_root: Path,
                       *, repair: bool, max_repairs: int, state_path: Path,
                       state: dict, pending_path: Path, pending: dict) -> bool:
    """Re-fetch affected tickers in full so each series has ONE basis.

    Crash-safe by construction. The naive form — delete every suspect, then
    re-fetch in a loop — is a data-loss bug: a provider failure or a kill mid-loop
    leaves tickers with ZERO rows, and they cannot self-heal (absent rows are
    invisible to the seam scan, the state file records only successes, and no
    scheduled seed fetches enough history to restore them).

    Instead each ticker is:
      1. journalled to ``pending`` (fsynced to disk) BEFORE anything is deleted,
      2. deleted on its own,
      3. re-fetched in full; on success the journal entry is cleared,
      4. on failure, recovered offline from the immutable raw store via
         ``data_pipelines reprocess``, and left in the journal if that fails too.
    Any journal entries left by a previous crash are healed at the start of the
    next run, so the window of exposure is one ticker and it is self-closing.
    """
    if not tickers and not pending:
        print("[seam] nothing to repair", flush=True)
        return True
    if not repair:
        print("[seam] --no-repair set; leaving the cache untouched", flush=True)
        return True

    db = data_root / "processed.db"
    today = pd.Timestamp.now("UTC").normalize().strftime("%Y-%m-%d")
    ok = True

    # Heal anything a previous run left half-done.
    for t in sorted(pending):
        print(f"[seam] healing {t} (interrupted by an earlier run)", flush=True)
        try:
            rows = _repair_one(t, db, end, data_root)
            print(f"[seam]   healed {t}: {rows} rows", flush=True)
            state[t] = today
            pending.pop(t, None)
            save_repair_state(pending_path, pending)
        except Exception as e:  # noqa: BLE001
            # Per-ticker result, not the cumulative `ok`: one failed heal must not
            # leave every later successful heal stuck in the journal.
            if _reprocess_from_raw(t, data_root):
                pending.pop(t, None)
                save_repair_state(pending_path, pending)
            else:
                ok = False
                print(f"[seam]   {t} still unresolved: {e}", file=sys.stderr, flush=True)

    if len(tickers) > max_repairs:
        print(f"[seam] ABORT: {len(tickers)} exceeds --max-repairs {max_repairs}; "
              "refusing a bulk rewrite. Investigate before raising the cap.",
              file=sys.stderr, flush=True)
        return False

    for t in tickers:
        pending[t] = today
        save_repair_state(pending_path, pending)  # durable BEFORE the delete
        try:
            rows = _repair_one(t, db, end, data_root)
            print(f"[seam]   re-fetched {t}: {rows} rows", flush=True)
            state[t] = today
            pending.pop(t, None)
        except Exception as e:  # noqa: BLE001
            print(f"[seam]   FAILED {t}: {e}", file=sys.stderr, flush=True)
            ok = False
            if _reprocess_from_raw(t, data_root):
                pending.pop(t, None)  # cache restored (older basis); retried next run
        save_repair_state(pending_path, pending)
    save_repair_state(state_path, state)
    return ok


def refresh_fundamentals_and_valuation(data_root: Path, end: pd.Timestamp,
                                       universe: str = "sp500") -> bool:
    """Weekly leg: refresh us_fundamentals then rebuild the valuation panel."""
    cmd = [sys.executable, "-m", "data_pipelines", "--data-root", str(data_root),
           "seed", "--domain", "us_fundamentals", "--universe", universe,
           "--start", "2015-01-01", "--end", str(end.date()), "--jobs", "1"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(f"[fundamentals] {universe}: "
          f"{(r.stdout.strip().splitlines() or [''])[-1]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or r.stdout).strip()[-600:], file=sys.stderr, flush=True)
        return False

    cmd = [sys.executable, "-m", "scripts.valuation.build_valuation_panel",
           "--domain", "us", "--start", "2018-01-01", "--end", str(end.date())]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    for line in r.stdout.strip().splitlines()[-4:]:
        print(f"[valuation] {line}", flush=True)
    if r.returncode != 0:
        print((r.stderr or r.stdout).strip()[-600:], file=sys.stderr, flush=True)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Daily us_equities cache refresh (+ optional weekly valuation leg).")
    ap.add_argument("--end", default=None,
                    help="as-of session (default: last completed NYSE session)")
    ap.add_argument("--universes", nargs="*", default=list(DEFAULT_UNIVERSES))
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--lookback-days", type=int, default=45,
                    help="window scanned for basis steps (default 45)")
    ap.add_argument("--no-repair", action="store_true",
                    help="detect basis steps but do not rewrite the cache")
    ap.add_argument("--move-threshold", type=float, default=0.40,
                    help="|1-day move| that flags a ticker for a look (default 0.40)")
    ap.add_argument("--max-repairs", type=int, default=150,
                    help="safety cap on how many tickers one run may re-fetch")
    ap.add_argument("--fallback-days", type=int, default=30,
                    help="fetch depth when the cache is cold (default 30)")
    ap.add_argument("--with-valuation", action="store_true",
                    help="also refresh us_fundamentals and rebuild the valuation panel")
    args = ap.parse_args(argv)

    data_root = (REPO / args.data_root) if not Path(args.data_root).is_absolute() else Path(args.data_root)
    end = pd.Timestamp(args.end).normalize() if args.end else last_completed_session()
    print(f"[run] data_root={data_root} end_session={end.date()} "
          f"universes={list(args.universes)}", flush=True)

    _preflight_disk(REPO)

    ok = True
    for u in args.universes:
        # The start is derived from the cache's own last session (see
        # seed_universe), so any outage length is caught up rather than only the
        # last few days. `fetch` fills gaps only — it never rewrites cached rows —
        # so provider re-statements are left to the basis-step guard below, which
        # is the only thing allowed to rewrite history (a partial rewrite is
        # precisely what CREATES a basis seam, so it is never done speculatively).
        ok &= seed_universe(u, end, args.jobs, data_root,
                            fallback_days=args.fallback_days)

    identifiers = sorted({i for u in args.universes for i in _universe_identifiers(u)})
    frame = _scan_window(data_root, identifiers, args.lookback_days)
    state_path = data_root / ".daily_update_repairs.json"
    pending_path = data_root / ".daily_update_pending.json"
    state = load_repair_state(state_path)
    pending = load_repair_state(pending_path)
    suspects, _skipped = select_suspects(frame, move_threshold=args.move_threshold,
                                        state=state, lookback_days=args.lookback_days)
    ok &= repair_basis_seams(suspects, end, data_root, repair=not args.no_repair,
                             max_repairs=args.max_repairs,
                             state_path=state_path, state=state,
                             pending_path=pending_path, pending=pending)

    if args.with_valuation:
        # The panel builder reads `<repo-root>/data` and resolves the fundamentals
        # cache through the library default, so it cannot follow a custom root.
        # Fail loudly rather than silently rebuilding from the wrong cache.
        if data_root.resolve() != (REPO / "data").resolve():
            print(f"[valuation] --with-valuation requires the default data root "
                  f"({REPO / 'data'}); got {data_root}. Run the valuation leg "
                  "separately or use the default root.", file=sys.stderr, flush=True)
            ok = False
        else:
            ok &= refresh_fundamentals_and_valuation(data_root, end)

    print(f"[done] {'OK' if ok else 'WITH FAILURES'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
