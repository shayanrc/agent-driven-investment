"""Sequential back-extend driver for the NSE Broad Market catalog task.

Reads /tmp/fetch_plan.json (produced inline by the agent), iterates the
``back_extend`` list, and shells out to ``python -m data_pipelines fetch ...``
for each ticker. Logs one structured line per ticker to
``logs/broad_market_fetch.log`` plus a rolling summary every N tickers, and
writes a final ``logs/broad_market_fetch_summary.json`` artifact on exit
(including Ctrl-C / SIGTERM).

Sequential by design — SQLite is single-writer per data_root.

CLI:
    python -m scripts.data_pipelines.broad_market_fetch [start_idx] [--verbose]

    start_idx: optional resume index (default 0).
    --verbose: also log the raw stderr tail for each failure
               (default off; structured one-line format only).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.data_pipelines._fetch_logging import (
    RollingCounter,
    SummaryWriter,
    classify_outcome,
    extract_err_short,
    format_per_ticker_line,
    utc_ts,
)

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "processed.db"
LOG_PATH = REPO / "logs" / "broad_market_fetch.log"
SUMMARY_PATH = REPO / "logs" / "broad_market_fetch_summary.json"
PLAN_PATH = Path("/tmp/fetch_plan.json")

START = "2015-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PER_TICKER_TIMEOUT = 120  # seconds
ROLLING_INTERVAL = 50


def get_row_count(ticker: str) -> int:
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM nse_equities_data WHERE ticker = ?", (ticker,)
        )
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return -1


def log(msg: str) -> None:
    # If msg already starts with a UTC timestamp (helpers prepend one), don't
    # double-stamp; else prepend.
    if len(msg) >= 11 and msg[:4].isdigit() and msg[4] == "-" and "T" in msg[:11]:
        line = msg
    else:
        line = f"{utc_ts()} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def fetch_one(sym: str) -> tuple[int, int, int, float, str]:
    """Run the fetch CLI; return (rc, before_n, after_n, duration_s, stderr_tail).

    rc convention:
      - 0  : CLI exited cleanly (caller decides OK vs OK_NOOP via row delta)
      - >0 : CLI returned non-zero
      - -1 : subprocess timed out
    """
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
            return 0, before, after, dt, ""
        # Failure — keep enough stderr tail for classify_error + extract_err_short.
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return proc.returncode, before, before, dt, tail
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        after = get_row_count(ticker)
        return -1, before, after, dt, "TimeoutExpired: per-ticker timeout exceeded"


def main(start_idx: int = 0, verbose: bool = False) -> None:
    with open(PLAN_PATH) as f:
        plan = json.load(f)
    backlog = plan["back_extend"]
    total = len(backlog)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary = SummaryWriter(
        out_path=SUMMARY_PATH,
        total_tickers=total,
        fetch_plan_path=str(PLAN_PATH),
    )
    summary.install_signal_handlers(log_fn=log)

    log(f"START broad_market_fetch backlog={total} start_idx={start_idx} "
        f"verbose={verbose}")
    t_start = time.time()
    counter = RollingCounter(total=total, t_start=t_start, interval=ROLLING_INTERVAL)

    try:
        for i, sym in enumerate(backlog[start_idx:], start=start_idx + 1):
            rc, before, after, dt, stderr = fetch_one(sym)
            delta_int = (after - before) if (after >= 0 and before >= 0) else 0
            delta_disp: int | str = delta_int if (after >= 0 and before >= 0) else "?"
            status = classify_outcome(rc, delta_int, stderr)
            err_short = extract_err_short(stderr) if rc != 0 else ""

            elapsed = time.time() - t_start
            eta_per = elapsed / max(i - start_idx, 1)
            remaining = (total - i) * eta_per

            log(format_per_ticker_line(
                idx=i, total=total, sym=f"NSE:{sym}",
                status=status,
                before=before, after=after, delta=delta_disp,
                took_s=dt, elapsed_m=elapsed / 60, eta_m=remaining / 60,
                err_short=err_short,
            ))
            if verbose and stderr:
                # Preserve raw stderr tail at debug verbosity for forensics.
                log(f"  [verbose] stderr tail for NSE:{sym}:")
                for raw in stderr.strip().splitlines()[-20:]:
                    log(f"    {raw}")

            extra = f"+{delta_int} rows" if status == "OK" else (
                err_short if status.startswith("FAIL") else ""
            )
            summary.record(status=status, sym=f"NSE:{sym}", extra=extra)
            counter.bump(status)

            if i % ROLLING_INTERVAL == 0:
                log(counter.render(done=i))
    finally:
        path = summary.finalize()
        log(f"DONE total_elapsed={(time.time()-t_start)/60:.1f}m "
            f"summary_json={path}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "start_idx", nargs="?", type=int, default=0,
        help="Resume index (default 0).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Also dump raw stderr tail per failure (default off — "
             "structured one-line format only).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(start_idx=args.start_idx, verbose=args.verbose)
