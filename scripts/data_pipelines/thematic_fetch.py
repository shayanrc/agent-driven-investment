"""NSE Thematic / Strategy catalog — universe-YAML writer + sequential back-extend driver.

Two-phase, both phases re-runnable and idempotent (mirrors sectoral_fetch.py):

  Phase A: --write-yamls
    For each Thematic/Strategy index in THEMATIC_INDICES, curl the current
    constituent CSV from niftyindices.com/IndexConstituent/<csv_filename>
    (the archives.nseindia.com path is sectoral-only — thematic/strategy
    CSVs are hosted under niftyindices.com with mixed-case + heterogeneous
    slug shapes; each canonical filename is captured here as published).
    Parse the Symbol column, filter DUMMY* placeholders, write one
    configs/data_pipelines/domains/nse_equities/universe_<slug>.yaml.

  Phase B: --back-extend
    Sequentially shell out to `python -m data_pipelines fetch <ticker>
    --start 2015-01-01 --end <today> --back-extend` for every unique
    constituent symbol across all Thematic/Strategy universes. Per-ticker
    120 s hard timeout. NOT wired in by default for this catalog PR —
    back-extends are gated on the cache-availability follow-up so we don't
    create overlapping single-writer contention with experiments in flight.

  Default (no flag): runs Phase A only (Phase B is opt-in for this script).

SQLite is single-writer per data_root; never run two of these in parallel
against the same `data/processed.db`. See CLAUDE.md § Data and configs.
"""
from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "processed.db"
YAML_OUT_DIR = REPO / "configs" / "data_pipelines" / "domains" / "nse_equities"
LOG_PATH = REPO / "logs" / "nse_thematic_fetch.log"

DATESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START = "2015-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PER_TICKER_TIMEOUT = 120  # seconds — matches broad_market_fetch.py

# Canonical NSE Thematic + Strategy index catalog. Tuples of:
#   (universe_slug, index_ticker, description, csv_filename)
# csv_filename is the *exact* file under niftyindices.com/IndexConstituent/
# (case- and underscore-sensitive — the strategy/thematic CSVs use a
# heterogeneous naming convention vs. the sectoral family). Each filename
# was resolved by scraping the per-index detail page on niftyindices.com
# and grepping for the "<page>/IndexConstituent/<file>.csv" link.
#
# index_ticker uses the symbol form already cached under nse_equities for
# the matching benchmark series (verified via
# `SELECT DISTINCT ticker FROM nse_equities_data WHERE ticker LIKE 'NIFTY:%'`).
# Where no cached benchmark exists yet (e.g. NIFTY:CAPMKT), the canonical
# NIFTY: form per niftyindices.com is used and the back-extend of that
# benchmark series is left as a follow-up.
THEMATIC_INDICES: list[tuple[str, str, str, str]] = [
    # --- Strategy / factor indices ------------------------------------------
    ("nifty_alpha_50", "NIFTY:ALPHA50",
     "NIFTY Alpha 50 — top-50 high-alpha stocks (strategy)",
     "ind_nifty_Alpha_Index.csv"),
    ("nifty_low_volatility_50", "NIFTY:LOWVOL50",
     "NIFTY Low Volatility 50 — 50 lowest-volatility stocks (strategy)",
     "nifty_low_Volatility50_Index.csv"),
    ("nifty_high_beta_50", "NIFTY:HIGHBETA50",
     "NIFTY High Beta 50 — 50 highest-beta stocks (strategy)",
     "nifty_High_Beta50_Index.csv"),
    ("nifty_dividend_opportunities_50", "NIFTY:DIVOPP50",
     "NIFTY Dividend Opportunities 50 — high-dividend stocks (strategy)",
     "ind_niftydivopp50list.csv"),
    ("nifty50_value_20", "NIFTY:VALUE20",
     "NIFTY50 Value 20 — value-tilted Nifty 50 subset (strategy)",
     "ind_Nifty50_Value20.csv"),
    ("nifty_growth_sectors_15", "NIFTY:GROWTHSECT15",
     "NIFTY Growth Sectors 15 — growth-sector tilt (strategy)",
     "ind_NiftyGrowth_Sectors15_Index.csv"),
    ("nifty_quality_low_volatility_30", "NIFTY:QUALLOWVOL30",
     "NIFTY Quality Low Volatility 30 — combined quality+lowvol factor",
     "ind_nifty_quality_lowvol30list.csv"),
    ("nifty_alpha_low_volatility_30", "NIFTY:ALPHALOWVOL30",
     "NIFTY Alpha Low Volatility 30 — combined alpha+lowvol factor",
     "ind_nifty_alpha_lowvol30list.csv"),
    ("nifty_alpha_quality_low_volatility_30", "NIFTY:ALPHAQUALLOWVOL30",
     "NIFTY Alpha Quality Low Volatility 30 — 3-factor combo",
     "ind_nifty_alpha_quality_lowvol30list.csv"),
    ("nifty_alpha_quality_value_low_volatility_30", "NIFTY:ALPHAQUALVALLOWVOL30",
     "NIFTY Alpha Quality Value Low Volatility 30 — 4-factor combo",
     "ind_nifty_alpha_quality_value_lowvol30list.csv"),
    ("nifty100_quality_30", "NIFTY:100QUALITY30",
     "NIFTY100 Quality 30 — top-30 quality stocks from Nifty 100",
     "ind_nifty100Quality30list.csv"),
    ("nifty100_low_volatility_30", "NIFTY:100LOWVOL30",
     "NIFTY100 Low Volatility 30 — 30 lowest-vol from Nifty 100",
     "ind_Nifty100LowVolatility30list.csv"),
    ("nifty200_momentum_30", "NIFTY:200MOMENTUM30",
     "NIFTY200 Momentum 30 — top-30 momentum from Nifty 200",
     "ind_nifty200Momentum30_list.csv"),
    ("nifty_midcap150_momentum_50", "NIFTY:MIDCAP150MOM50",
     "NIFTY Midcap150 Momentum 50 — top-50 momentum from Midcap 150",
     "ind_niftymidcap150momentum50_list.csv"),
    ("nifty_midcap150_quality_50", "NIFTY:MIDCAP150QUAL50",
     "NIFTY Midcap150 Quality 50 — top-50 quality from Midcap 150",
     "ind_niftymidcap150quality50list.csv"),
    ("nifty500_multifactor_mqvlv_50", "NIFTY:500MULTIFACTOR50",
     "NIFTY500 Multifactor MQVLv 50 — momentum+quality+value+lowvol combo",
     "ind_nifty500MultifactorMQVLv50_list.csv"),
    # --- Thematic / cross-cutting indices -----------------------------------
    ("nifty_commodities", "NIFTY:COMMODITIES",
     "NIFTY Commodities — diversified commodity-producer theme",
     "ind_niftycommoditieslist.csv"),
    ("nifty_cpse", "NIFTY:CPSE",
     "NIFTY CPSE — central-public-sector enterprises theme",
     "ind_niftycpselist.csv"),
    ("nifty_infrastructure", "NIFTY:INFRA",
     "NIFTY Infrastructure — infrastructure theme",
     "ind_niftyinfralist.csv"),
    ("nifty_mnc", "NIFTY:MNC",
     "NIFTY MNC — multinational-corporation theme",
     "ind_niftymnclist.csv"),
    ("nifty_pse", "NIFTY:PSE",
     "NIFTY PSE — public-sector enterprise theme",
     "ind_niftypselist.csv"),
    ("nifty_services_sector", "NIFTY:SERVICES",
     "NIFTY Services Sector — diversified services theme",
     "ind_niftyservicelist.csv"),
    ("nifty_india_consumption", "NIFTY:CONSUMPTION",
     "NIFTY India Consumption — domestic-consumption theme",
     "ind_niftyconsumptionlist.csv"),
    ("nifty_india_manufacturing", "NIFTY:MANUFACTURING",
     "NIFTY India Manufacturing — manufacturing theme",
     "ind_niftyindiamanufacturing_list.csv"),
    ("nifty_india_defence", "NIFTY:DEFENCE",
     "NIFTY India Defence — defence/aerospace theme",
     "ind_niftyindiadefence_list.csv"),
    ("nifty_india_digital", "NIFTY:DIGITAL",
     "NIFTY India Digital — digital-economy theme",
     "ind_niftyindiadigital_list.csv"),
    ("nifty_india_tourism", "NIFTY:TOURISM",
     "NIFTY India Tourism — tourism/hospitality theme",
     "ind_niftyindiatourism_list.csv"),
    ("nifty_mobility", "NIFTY:MOBILITY",
     "NIFTY Mobility — mobility/transport theme",
     "ind_niftymobility_list.csv"),
    ("nifty_capital_markets", "NIFTY:CAPMKT",
     "NIFTY Capital Markets — exchanges + brokerage + AMC theme",
     "ind_niftyCapitalMarkets_list.csv"),
    ("nifty_housing", "NIFTY:HOUSING",
     "NIFTY Housing — housing-construction theme",
     "ind_niftyhousing_list.csv"),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Phase A — write universe YAMLs
# ---------------------------------------------------------------------------


def fetch_csv(csv_filename: str) -> bytes | None:
    """GET https://niftyindices.com/IndexConstituent/<csv_filename> via curl.

    Returns bytes on HTTP 200 *and* CSV content-type-shaped body, None
    otherwise. niftyindices.com is a soft-404 server (returns the homepage
    HTML with 200 for missing files), so the caller must additionally
    validate that the body parses as CSV with a Symbol column."""
    url = f"https://niftyindices.com/IndexConstituent/{csv_filename}"
    cmd = [
        "curl", "-sL", "-A", "Mozilla/5.0",
        "-w", "%{http_code}",
        "-o", "-",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        log(f"  CSV fetch timeout for {csv_filename}")
        return None
    if proc.returncode != 0:
        log(f"  CSV fetch curl rc={proc.returncode} for {csv_filename}")
        return None
    out = proc.stdout
    if len(out) < 3:
        return None
    code = out[-3:].decode("ascii", errors="replace")
    body = out[:-3]
    if code != "200":
        log(f"  CSV fetch HTTP {code} for {csv_filename}")
        return None
    # Soft-404 guard: niftyindices.com returns its homepage HTML with 200
    # for unknown paths. A real constituent CSV starts with the field-list
    # header that includes "Symbol".
    head = body[:200].decode("utf-8-sig", errors="replace").lower()
    if "<html" in head or "<!doctype" in head:
        log(f"  CSV fetch returned HTML (soft-404) for {csv_filename}")
        return None
    if "symbol" not in head:
        log(f"  CSV fetch missing Symbol header for {csv_filename}")
        return None
    return body


def parse_symbols(csv_bytes: bytes) -> list[str]:
    """Parse the niftyindices.com CSV, extract `Symbol` column,
    filter DUMMY* placeholders (`[[project-nse-data-quirks]]` § 2)."""
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    symbols: list[str] = []
    for row in reader:
        sym = (row.get("Symbol") or "").strip()
        if not sym:
            continue
        if sym.startswith("DUMMY"):
            continue
        symbols.append(sym)
    return symbols


def render_yaml(universe: str, index_ticker: str, description: str,
                csv_filename: str, symbols: list[str]) -> str:
    lines = [
        f"# {description} (as of {DATESTAMP}).",
        f"# Source: https://niftyindices.com/IndexConstituent/{csv_filename}",
        f"# Refresh by re-running scripts/data_pipelines/thematic_fetch.py --write-yamls;",
        f"# NSE rebalances thematic / strategy indices semi-annually (some quarterly).",
        f"# Point-in-time historical membership is explicitly out of scope (V2_TBD open q.1).",
        "",
        f"universe: {universe}",
        f"listed_at: {DATESTAMP}",
        "indices:",
        f'  - "{index_ticker}"',
        "tickers:",
    ]
    for sym in sorted(symbols):
        lines.append(f'  - "NSE:{sym}"')
    return "\n".join(lines) + "\n"


def write_yamls() -> dict[str, list[str]]:
    """Phase A. Returns dict {universe_slug: [symbols]} for successful fetches."""
    YAML_OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    failed: list[tuple[str, str]] = []
    log(f"PHASE A start — {len(THEMATIC_INDICES)} Thematic/Strategy indices")
    for slug, index_ticker, desc, csv_filename in THEMATIC_INDICES:
        csv_bytes = fetch_csv(csv_filename)
        if csv_bytes is None:
            failed.append((slug, csv_filename))
            log(f"  SKIP {slug:<48} (CSV fetch failed: {csv_filename})")
            continue
        symbols = parse_symbols(csv_bytes)
        if not symbols:
            failed.append((slug, csv_filename))
            log(f"  SKIP {slug:<48} (CSV parsed 0 symbols: {csv_filename})")
            continue
        text = render_yaml(slug, index_ticker, desc, csv_filename, symbols)
        out_path = YAML_OUT_DIR / f"universe_{slug}.yaml"
        out_path.write_text(text)
        written[slug] = symbols
        log(f"  WROTE {slug:<48} -> {out_path.name} ({len(symbols)} tickers)")
    log(f"PHASE A done — wrote {len(written)} / failed {len(failed)}")
    if failed:
        for slug, csv_filename in failed:
            log(f"  FAILED: {slug} (csv_filename={csv_filename})")
    return written


# ---------------------------------------------------------------------------
# Phase B — back-extend per-ticker via data_pipelines CLI
# ---------------------------------------------------------------------------


def get_row_count(ticker: str) -> int:
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM nse_equities_data WHERE ticker = ?",
            (ticker,),
        )
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return -1


def fetch_one(sym: str) -> tuple[str, int, int, float]:
    ticker = f"NSE:{sym}"
    before = get_row_count(ticker)
    cmd = [
        "uv", "run", "python", "-m", "data_pipelines", "fetch",
        ticker, "--start", START, "--end", END, "--back-extend",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=PER_TICKER_TIMEOUT, cwd=REPO,
        )
        dt = time.time() - t0
        if proc.returncode == 0:
            after = get_row_count(ticker)
            status = "ok"
        else:
            after = before
            tail = (proc.stderr or proc.stdout or "")[-200:].replace("\n", " | ")
            status = f"err(rc={proc.returncode}):{tail}"
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        after = get_row_count(ticker)
        status = "timeout"
    return status, before, after, dt


def gather_unique_symbols() -> list[str]:
    all_symbols: set[str] = set()
    for slug, _idx, _desc, _csv in THEMATIC_INDICES:
        path = YAML_OUT_DIR / f"universe_{slug}.yaml"
        if not path.exists():
            log(f"  (gather) skipping missing YAML {path.name}")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith('- "NSE:'):
                sym = line.split(":", 1)[1].rstrip('"')
                all_symbols.add(sym)
    return sorted(all_symbols)


def back_extend_all(start_idx: int = 0) -> None:
    symbols = gather_unique_symbols()
    total = len(symbols)
    log(f"PHASE B start — {total} unique symbols across Thematic/Strategy universes "
        f"(start_idx={start_idx})")
    t_start = time.time()
    for i, sym in enumerate(symbols[start_idx:], start=start_idx + 1):
        status, before, after, dt = fetch_one(sym)
        delta = after - before if (before >= 0 and after >= 0) else "?"
        elapsed = time.time() - t_start
        log(f"  [{i:>4}/{total}] {sym:<14} {status:<60} "
            f"rows {before} -> {after} (+{delta}) in {dt:5.1f}s "
            f"(elapsed {elapsed/60:.1f}m)")
    log(f"PHASE B done — {total} tickers in {(time.time()-t_start)/60:.1f}m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-yamls", action="store_true",
                    help="Phase A only: write/refresh universe YAMLs from NSE CSVs.")
    ap.add_argument("--back-extend", action="store_true",
                    help="Phase B only: back-extend cached history for all "
                         "constituent tickers (single-writer; ~120 s/ticker).")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Phase B: resume index for unique-ticker list.")
    args = ap.parse_args()

    if args.write_yamls and args.back_extend:
        write_yamls()
        back_extend_all(start_idx=args.start_idx)
    elif args.back_extend:
        back_extend_all(start_idx=args.start_idx)
    else:
        # default = Phase A only (catalog PR)
        write_yamls()
    return 0


if __name__ == "__main__":
    sys.exit(main())
