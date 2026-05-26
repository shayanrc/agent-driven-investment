"""Sequential back-extend driver for the NSE Broad Market catalog task.

Reads /tmp/fetch_plan.json (produced inline by the agent), iterates the
`back_extend` list, and shells out to `python -m data_pipelines fetch ...`
for each ticker. Logs one line per ticker to logs/broad_market_fetch.log.

Sequential by design — SQLite is single-writer per data_root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "processed.db"
LOG_PATH = REPO / "logs" / "broad_market_fetch.log"
PLAN_PATH = Path("/tmp/fetch_plan.json")

START = "2015-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PER_TICKER_TIMEOUT = 120  # seconds


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
    except Exception as e:
        return -1


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


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
            status = f"err(rc={proc.returncode})"
            # Capture last 200 chars of stderr for diagnostics
            tail = (proc.stderr or proc.stdout or "")[-200:].replace("\n", " | ")
            status = f"{status}:{tail}"
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        after = get_row_count(ticker)
        status = "timeout"
    return status, before, after, dt


def main(start_idx: int = 0) -> None:
    with open(PLAN_PATH) as f:
        plan = json.load(f)
    backlog = plan["back_extend"]
    total = len(backlog)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"START broad_market_fetch backlog={total} start_idx={start_idx}")
    t_start = time.time()
    for i, sym in enumerate(backlog[start_idx:], start=start_idx + 1):
        status, before, after, dt = fetch_one(sym)
        delta = after - before if after >= 0 and before >= 0 else "?"
        elapsed = time.time() - t_start
        eta_per = elapsed / max(i - start_idx, 1)
        remaining = (total - i) * eta_per
        log(
            f"[{i:>3}/{total}] {sym:<14} {status:<32} "
            f"rows: {before:>5} -> {after:>5} (+{delta}) "
            f"took={dt:>5.1f}s elapsed={elapsed/60:>5.1f}m eta={remaining/60:>5.1f}m"
        )
    log(f"DONE total_elapsed={(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(start)
