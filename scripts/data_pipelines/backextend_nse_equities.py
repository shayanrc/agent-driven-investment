#!/usr/bin/env python
"""Back-extend cached nse_equities history for a universe (deepen shallow seeds).

The nse_equities cache was seeded at two different start dates — the nifty50/100
core from 2015, the broader nifty500 expansion only from 2020 — so ~85% of the
universe has NO pre-2020 history (their entire record is the 2020-2023 rally).
That makes any date-aligned split train-confounded (rally-dominated + staggered
universe entry). This backfills the missing pre-2020 bars so training can anchor
earlier and span pre-rally + rally + normalization regimes.

Two deliberate choices:
  * ``back_extend=True`` — bypass the cache-first cap so providers fetch BEFORE
    the earliest cached bar (the default False refuses and returns empty).
  * ``chain_order=("yfinance",)`` — jugaad/nselib are blocked/shallow on this
    host; the default chain burns ~20s/ticker of retries before falling through
    to yfinance. Forcing yfinance-first (.NS has deep history) makes it fast.
    Gap-detection still only fetches the missing window; already-cached bars are
    untouched (per-cell first-written-wins merge), so the deep cohort no-ops.

Usage:
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
      uv run python -m scripts.data_pipelines.backextend_nse_equities \
      [--universe nifty500] [--start 2015-01-01] [--end 2019-12-31] [--jobs 8]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import data_pipelines.domains.nse_equities  # noqa: F401  (registers default domain)
from data_pipelines.dispatch import fetch_with_meta
from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.nse_equities import NSEDomain
from data_pipelines.domains.nse_equities.config import NSEEquitiesConfig
from data_pipelines.domains.nse_equities.universe import load_universe


def _install_yfinance_first() -> None:
    """Swap the registered nse_equities domain for a yfinance-only one.

    register() rejects re-registering a different instance for a taken prefix,
    so assign into the registry map directly (one-off maintenance script).
    """
    custom = NSEDomain(NSEEquitiesConfig(chain_order=("yfinance",)))
    for prefix, dom in list(DomainRegistry._by_prefix.items()):
        if dom.name == custom.name:
            DomainRegistry._by_prefix[prefix] = custom


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nifty500")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args(argv)

    _install_yfinance_first()
    data_root = Path(args.data_root)

    universe = load_universe(args.universe)
    tickers = [t for t in universe if t.startswith("NSE:")]  # equities only
    print(f"back-extending {len(tickers)} NSE tickers ({args.universe}) "
          f"to [{args.start}, {args.end}] via yfinance, jobs={args.jobs} ...",
          flush=True)

    def _one(ident: str):
        _, meta = fetch_with_meta(
            ident, args.start, args.end, data_root=data_root, back_extend=True,
        )
        return ident, meta.row_count

    ok, failed, added = 0, [], 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_one, t): t for t in tickers}
        for fut in as_completed(futs):
            ident = futs[fut]
            done += 1
            try:
                _, rows = fut.result()
                ok += 1
                added += rows
                if done % 50 == 0 or done == len(tickers):
                    print(f"  [{done}/{len(tickers)}] ok={ok} failed={len(failed)} "
                          f"rows_in_window~{added}", flush=True)
            except Exception as e:  # noqa: BLE001 — per-ticker isolation
                failed.append(ident)
                print(f"  ✗ {ident}: {e}", file=sys.stderr, flush=True)

    print(f"\ndone: {ok}/{len(tickers)} ok, {len(failed)} failed", flush=True)
    if failed:
        print("failed:", ", ".join(failed[:30]) + (" ..." if len(failed) > 30 else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
