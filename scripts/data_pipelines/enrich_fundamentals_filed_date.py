"""One-shot EDGAR `filed_date` enrichment for the us_fundamentals cache.

Macrotrends (the primary provider) has no filing dates, so a cache seeded
mostly by macrotrends carries `filed_date` = NaT on ~95% of rows. `filed_date`
is the point-in-time hook the modeling phase needs (a quarter's numbers are not
public knowledge until filed) — so before building any causally-lagged
fundamentals feature, this pass fills it from SEC EDGAR.

Source: the SEC **submissions** API (`data.sec.gov/submissions/CIK…json`) — one
small (~150 KB) doc per ticker listing every filing's `form`, `filingDate`, and
`reportDate` (period-of-report). We map each period end → the earliest filing
date of the 10-K/10-Q that reported it (`EdgarAdapter.filing_dates`). This is the
**authoritative** "became public" date and, unlike the companyfacts differencing
approach, is exact for derived-Q4 quarters (a fiscal-year 10-K's reportDate is
the Q4 end, so its own filing date lands on Q4 — no differencing-max that reads
weeks late).

Discipline:
  - **filed_date only.** No metric value, no row count, ever changes.
  - **Authoritative rewrite.** filed_date becomes the confirmed submissions
    date where a valid match exists, else **NaT** — an unconfirmed value is
    *cleared*, never left in place. This purges any stale/wrong dates from an
    earlier companyfacts pass or the seed's EDGAR fills (which mis-snap
    off-calendar filers like CAVA's 13-week restaurant quarters or FERG's
    July fiscal year). Every non-null filed_date then traces to a real SEC
    form; nulls are honest "unknown".
  - **Grid match, fiscal-end + causal guarded.** A cached row at grid date G
    takes the filing whose `reportDate` snaps to G and is closest to the row's
    `fiscal_period_end` (within `--fiscal-tolerance-days`) AND whose filing
    date is strictly after that fiscal end — a filing cannot precede the
    period it reports, so a match violating that is a mis-snap and is rejected.
  - **Idempotent** on writes: a ticker already carrying the authoritative dates
    is detected as unchanged and not rewritten (the submissions doc is still
    fetched — it's cheap — so corrections always apply).
  - Tickers with no EDGAR coverage (ADRs / foreign filers / delisted) are left
    as-is and counted.

Usage:
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
      uv run python -m scripts.data_pipelines.enrich_fundamentals_filed_date \
      [--tickers FUND:AAPL ...] [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.cache import (
    list_cached_identifiers,
    read_processed,
    write_processed_atomic,
)
from data_pipelines.domains.us_fundamentals import get_domain
from data_pipelines.domains.us_fundamentals.adapters.edgar import EdgarAdapter
from data_pipelines.domains.us_fundamentals.schema import snap_to_quarter_end
from data_pipelines.errors import ProviderError


def _log(msg: str) -> None:
    print(msg, flush=True)


def _snap(d: pd.Timestamp):
    return pd.Timestamp(snap_to_quarter_end(d.date()))


def resolve_filed_dates(
    cached_df: pd.DataFrame,
    report_to_filed: dict,
    fiscal_tol_days: int,
) -> pd.Series:
    """Pure core: given the cached rows (date grid + fiscal_period_end) and a
    submissions ``reportDate → filingDate`` map, return the authoritative
    filed_date series — confirmed date per row, or NaT where unconfirmed.

    A row is confirmed by the filing whose reportDate snaps to the row's grid
    date and is closest to its fiscal_period_end, subject to BOTH guards:
      - proximity: |reportDate − fiscal_period_end| ≤ fiscal_tol_days, and
      - causal invariant: filingDate > fiscal_period_end (a filing cannot
        precede the period it reports; a violation means a mis-snap, e.g. an
        off-calendar 13-week filer whose provider fiscal_period_end was
        calendar-normalized).
    Everything else → NaT (any stale unconfirmed value is cleared).
    """
    by_grid: dict[pd.Timestamp, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for rd, fd in report_to_filed.items():
        by_grid.setdefault(_snap(pd.Timestamp(rd)), []).append(
            (pd.Timestamp(rd), pd.Timestamp(fd))
        )
    tol = pd.Timedelta(days=fiscal_tol_days)
    out = []
    for grid, fiscal in zip(cached_df["date"], cached_df["fiscal_period_end"]):
        chosen = pd.NaT
        cands = by_grid.get(pd.Timestamp(grid), [])
        if cands:
            rd, fd = min(cands, key=lambda rf: abs(rf[0] - fiscal))
            if abs(rd - fiscal) <= tol and fd > fiscal:
                chosen = fd
        out.append(chosen)
    return pd.Series(out, index=cached_df.index, dtype="datetime64[ns]")


def enrich_one(
    adapter: EdgarAdapter,
    domain,
    data_root: Path,
    identifier: str,
    *,
    fiscal_tol_days: int,
    dry_run: bool,
) -> tuple[str, int]:
    """Returns (status, n_set). status ∈ {set, nochange, no_edgar, error, skip}."""
    cached_df, cached_meta = read_processed(data_root, domain, identifier)
    if cached_df is None or cached_df.empty:
        return "skip", 0

    try:
        report_to_filed = adapter.filing_dates(identifier, data_root=data_root)
    except ProviderError as e:
        _log(f"  ! {identifier}: submissions error — {e.reason}")
        return "error", 0

    out = cached_df.copy()
    # authoritative rewrite: confirmed date, else clear. With an empty map (a
    # foreign filer with a CIK but only 20-F/40-F forms, or no periodic
    # filings), this clears ANY stale unconfirmed value from an earlier
    # companyfacts pass / seed fill — filed_date is never left unauthoritative.
    new_filed = resolve_filed_dates(out, report_to_filed, fiscal_tol_days)
    changed = ~(
        (out["filed_date"].isna() & new_filed.isna())
        | (out["filed_date"] == new_filed)
    )
    n_set = int(changed.sum())
    out["filed_date"] = new_filed

    if n_set and not dry_run:
        meta = dict(cached_meta)
        meta["last_fetch_utc"] = datetime.now(timezone.utc).isoformat()
        sources = [
            s for s in meta.get("sources", [])
            if not (s.get("provider") == "edgar_submissions"
                    and s.get("role") == "filed_date")
        ]
        if report_to_filed:  # only claim submissions as a source if it matched
            sources.append({"provider": "edgar_submissions", "role": "filed_date"})
        meta["sources"] = sources
        meta["row_count"] = len(out)
        write_processed_atomic(data_root, domain, identifier, out, meta)

    if not report_to_filed:
        return "no_edgar", n_set
    return "nochange" if n_set == 0 else "set", n_set


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="explicit FUND: identifiers (default: all cached)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fiscal-tolerance-days", type=int, default=20)
    ap.add_argument("--min-sleep", type=float, default=0.1,
                    help="floor between EDGAR fetches (SEC ≤10 req/s)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    domain = get_domain()
    adapter = EdgarAdapter(domain.config)

    tickers = args.tickers or list_cached_identifiers(root, domain)
    if args.limit:
        tickers = tickers[: args.limit]

    _log(f"enriching filed_date (SEC submissions) for {len(tickers)} ticker(s) "
         f"(dry_run={args.dry_run}, fiscal_tol={args.fiscal_tolerance_days}d)")

    counts = {"set": 0, "nochange": 0, "no_edgar": 0, "error": 0, "skip": 0}
    rows_set = 0
    for i, ident in enumerate(tickers, 1):
        t0 = time.monotonic()
        status, n = enrich_one(
            adapter, domain, root, ident,
            fiscal_tol_days=args.fiscal_tolerance_days, dry_run=args.dry_run,
        )
        counts[status] += 1
        rows_set += n
        if status in ("set", "no_edgar", "error"):
            _log(f"  [{i}/{len(tickers)}] {ident}: {status}"
                 + (f" (set {n} rows)" if n else ""))
        dt = time.monotonic() - t0
        if dt < args.min_sleep:
            time.sleep(args.min_sleep - dt)

    _log("\n=== summary ===")
    _log(f"  set:      {counts['set']} tickers, {rows_set} rows")
    _log(f"  unchanged (already authoritative): {counts['nochange']}")
    _log(f"  no EDGAR coverage: {counts['no_edgar']}")
    _log(f"  skipped (empty cache): {counts['skip']}")
    _log(f"  errors: {counts['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
