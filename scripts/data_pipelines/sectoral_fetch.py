"""NSE Sectoral catalog — universe-YAML writer + sequential back-extend driver.

Two-phase, both phases re-runnable and idempotent:

  Phase A: --write-yamls
    For each Sectoral index in SECTORAL_INDICES, curl the current constituent
    CSV from archives.nseindia.com (the jugaad/nselib path is unreliable
    per `[[project-nse-data-quirks]]`), parse symbols, filter DUMMY*, and
    materialize one configs/data_pipelines/domains/nse_equities/universe_<slug>.yaml.
    Skips indices whose CSV returns non-200 and logs them.

  Phase B: --back-extend
    Sequentially shell out to `python -m data_pipelines fetch <ticker>
    --start 2015-01-01 --end <today> --back-extend` for every unique
    constituent symbol across all Sectoral universes. Per-ticker 120 s hard
    timeout. Logs one line per ticker to logs/nse_sectoral_fetch.log and
    streams to stdout (so a tee'd caller sees progress live).

  Default (no flag): runs Phase A then Phase B.

SQLite is single-writer per data_root; never run two of these in parallel
against the same `data/processed.db`. See CLAUDE.md § Data and configs.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "processed.db"
YAML_OUT_DIR = REPO / "configs" / "data_pipelines" / "domains" / "nse_equities"
LOG_PATH = REPO / "logs" / "nse_sectoral_fetch.log"

DATESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START = "2015-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PER_TICKER_TIMEOUT = 120  # seconds — matches broad_market_fetch.py

# Canonical NSE Sectoral index family. Tuples of:
#   (universe_slug, index_ticker, description, csv_slug)
# csv_slug fits the archives.nseindia.com pattern:
#   https://archives.nseindia.com/content/indices/ind_<csv_slug>.csv
#
# Tickers chosen to match the existing NSE index symbols already cached
# under nse_equities (sqlite SELECT DISTINCT ticker FROM nse_equities_data
# WHERE ticker LIKE 'NIFTY:%'). Where two CSV-slug variants existed (Private
# Bank, Financial Services) the working one was confirmed by probing.
SECTORAL_INDICES: list[tuple[str, str, str, str]] = [
    ("nifty_bank", "NIFTY:BANK",
     "NIFTY Bank — banking sector",
     "niftybanklist"),
    ("nifty_it", "NIFTY:IT",
     "NIFTY IT — information-technology sector",
     "niftyitlist"),
    ("nifty_auto", "NIFTY:AUTO",
     "NIFTY Auto — automotive sector",
     "niftyautolist"),
    ("nifty_pharma", "NIFTY:PHARMA",
     "NIFTY Pharma — pharmaceutical sector",
     "niftypharmalist"),
    ("nifty_fmcg", "NIFTY:FMCG",
     "NIFTY FMCG — fast-moving consumer goods sector",
     "niftyfmcglist"),
    ("nifty_metal", "NIFTY:METAL",
     "NIFTY Metal — metals/mining sector",
     "niftymetallist"),
    ("nifty_realty", "NIFTY:REALTY",
     "NIFTY Realty — real-estate sector",
     "niftyrealtylist"),
    ("nifty_energy", "NIFTY:ENERGY",
     "NIFTY Energy — energy (oil/gas/power) sector",
     "niftyenergylist"),
    ("nifty_media", "NIFTY:MEDIA",
     "NIFTY Media — media/entertainment sector",
     "niftymedialist"),
    ("nifty_psu_bank", "NIFTY:PSUBANK",
     "NIFTY PSU Bank — public-sector banking",
     "niftypsubanklist"),
    ("nifty_private_bank", "NIFTY:PVTBANK",
     "NIFTY Private Bank — private-sector banking",
     "nifty_privatebanklist"),
    ("nifty_financial_services", "NIFTY:FINSVC",
     "NIFTY Financial Services — banks + NBFCs + insurance + capital markets",
     "niftyfinancelist"),
    ("nifty_healthcare", "NIFTY:HEALTHCARE",
     "NIFTY Healthcare — healthcare/hospitals + pharma",
     "niftyhealthcarelist"),
    ("nifty_consumer_durables", "NIFTY:CONSUMERDURABLES",
     "NIFTY Consumer Durables — consumer durables sector",
     "niftyconsumerdurableslist"),
    ("nifty_oil_and_gas", "NIFTY:OILGAS",
     "NIFTY Oil & Gas — oil & gas sector",
     "niftyoilgaslist"),
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


def fetch_csv(csv_slug: str) -> bytes | None:
    """GET https://archives.nseindia.com/content/indices/ind_<slug>.csv via curl.

    Returns bytes on HTTP 200, None on non-200 / failure. Shell-curl is the
    reliable path per `[[project-nse-data-quirks]]` — Python's urllib trips
    SSLCertVerificationError on some hosts, and `nselib`/`jugaad_data` are
    actively blocked by NSE's anti-bot gate.
    """
    url = f"https://archives.nseindia.com/content/indices/ind_{csv_slug}.csv"
    cmd = [
        "curl", "-sL", "-A", "Mozilla/5.0",
        "-w", "%{http_code}",
        "-o", "-",  # body to stdout (then http_code appended via -w)
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        log(f"  CSV fetch timeout for {csv_slug}")
        return None
    if proc.returncode != 0:
        log(f"  CSV fetch curl rc={proc.returncode} for {csv_slug}")
        return None
    # -w writes the http_code as ASCII at the end of stdout (after body).
    out = proc.stdout
    if len(out) < 3:
        return None
    code = out[-3:].decode("ascii", errors="replace")
    body = out[:-3]
    if code != "200":
        log(f"  CSV fetch HTTP {code} for {csv_slug}")
        return None
    return body


def parse_symbols(csv_bytes: bytes) -> list[str]:
    """Parse the archives.nseindia.com CSV, extract `Symbol` column,
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
                csv_slug: str, symbols: list[str]) -> str:
    lines = [
        f"# {description} (as of {DATESTAMP}).",
        f"# Source: https://archives.nseindia.com/content/indices/ind_{csv_slug}.csv",
        f"# Refresh by re-running scripts/data_pipelines/sectoral_fetch.py --write-yamls;",
        f"# NSE rebalances sectoral indices semi-annually.",
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
    log(f"PHASE A start — {len(SECTORAL_INDICES)} Sectoral indices")
    for slug, index_ticker, desc, csv_slug in SECTORAL_INDICES:
        csv_bytes = fetch_csv(csv_slug)
        if csv_bytes is None:
            failed.append((slug, csv_slug))
            log(f"  SKIP {slug:<28} (CSV fetch failed: {csv_slug})")
            continue
        symbols = parse_symbols(csv_bytes)
        if not symbols:
            failed.append((slug, csv_slug))
            log(f"  SKIP {slug:<28} (CSV parsed 0 symbols: {csv_slug})")
            continue
        text = render_yaml(slug, index_ticker, desc, csv_slug, symbols)
        out_path = YAML_OUT_DIR / f"universe_{slug}.yaml"
        out_path.write_text(text)
        written[slug] = symbols
        log(f"  WROTE {slug:<28} -> {out_path.name} ({len(symbols)} tickers)")
    log(f"PHASE A done — wrote {len(written)} / failed {len(failed)}")
    if failed:
        for slug, csv_slug in failed:
            log(f"  FAILED: {slug} (csv_slug={csv_slug})")
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
    """Run the fetch CLI; return (status, before_n, after_n, duration_s)."""
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
    """Walk every Sectoral universe YAML on disk and dedupe the union of
    constituent symbols. Re-uses on-disk YAMLs so this phase is decoupled
    from a fresh Phase A run."""
    all_symbols: set[str] = set()
    for slug, _idx, _desc, _csv in SECTORAL_INDICES:
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
    """Phase B. Sequential single-writer back-extend; idempotent."""
    symbols = gather_unique_symbols()
    total = len(symbols)
    log(f"PHASE B start — {total} unique symbols across Sectoral universes "
        f"(start_idx={start_idx})")
    t_start = time.time()
    for i, sym in enumerate(symbols[start_idx:], start=start_idx + 1):
        status, before, after, dt = fetch_one(sym)
        delta = after - before if after >= 0 and before >= 0 else "?"
        elapsed = time.time() - t_start
        eta_per = elapsed / max(i - start_idx, 1)
        remaining = (total - i) * eta_per
        log(
            f"[{i:>3}/{total}] {sym:<16} {status:<32} "
            f"rows: {before:>5} -> {after:>5} (+{delta}) "
            f"took={dt:>5.1f}s elapsed={elapsed/60:>5.1f}m eta={remaining/60:>5.1f}m"
        )
    log(f"PHASE B done — total_elapsed={(time.time()-t_start)/60:.1f}m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-yamls", action="store_true",
        help="Phase A only: curl Sectoral CSVs and materialize universe YAMLs."
    )
    parser.add_argument(
        "--back-extend", action="store_true",
        help="Phase B only: sequential back-extend fetch for each unique "
             "constituent symbol via data_pipelines CLI."
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="Phase B resume index (for crash-recovery; default 0)."
    )
    args = parser.parse_args()

    do_a = args.write_yamls or not args.back_extend
    do_b = args.back_extend or not args.write_yamls

    if do_a:
        write_yamls()
    if do_b:
        back_extend_all(start_idx=args.start_idx)


if __name__ == "__main__":
    main()
