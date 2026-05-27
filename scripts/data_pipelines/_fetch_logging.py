"""Structured-logging helpers for the back-extend driver scripts.

This module is **script infrastructure**, not part of the `data_pipelines`
library — it lives under `scripts/data_pipelines/` so only the orchestration
scripts depend on it. The underscore prefix marks it as internal to the
scripts directory.

What it provides:

* `classify_error(stderr_or_msg) -> status_bucket` — pure pattern-match
  function mapping a CLI stderr/stdout dump into one of:

      - ``OK``           : provider returned data, rows actually added
      - ``OK_NOOP``      : CLI succeeded but row count unchanged (already cached)
      - ``FAIL_EMPTY``   : `AllProvidersFailed` with every provider returning
                           an empty payload (typical for recently-listed tickers)
      - ``FAIL_OVERLAP`` : `IntegrityError: UNIQUE constraint failed`
                           (the cache-overlap foot-gun fixed in PR #37; bucket
                           kept so a regression would surface)
      - ``FAIL_OTHER``   : any other non-zero-exit / exception

  The ``OK`` vs ``OK_NOOP`` split is decided by the caller (it needs the
  before/after row counts); `classify_error` only buckets failures. Use
  `classify_outcome(rc, delta, stderr)` for the full decision.

* `format_per_ticker_line(...)` — render the one-line structured per-ticker
  log entry.

* `RollingCounter` — accumulates per-status counts; renders the every-N
  rolling-summary line.

* `SummaryWriter` — accumulates per-ticker outcomes and writes a
  `summary.json` artifact when the run ends (normal exit, SIGINT, or
  SIGTERM). Caller wires `install_signal_handlers()`.

Regex patterns are documented inline next to where they're matched.
"""
from __future__ import annotations

import json
import os
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Status buckets
# ---------------------------------------------------------------------------

STATUS_OK = "OK"
STATUS_OK_NOOP = "OK_NOOP"
STATUS_FAIL_EMPTY = "FAIL_EMPTY"
STATUS_FAIL_OVERLAP = "FAIL_OVERLAP"
STATUS_FAIL_OTHER = "FAIL_OTHER"

ALL_STATUSES = (
    STATUS_OK,
    STATUS_OK_NOOP,
    STATUS_FAIL_EMPTY,
    STATUS_FAIL_OVERLAP,
    STATUS_FAIL_OTHER,
)

# Regex patterns — kept simple and documented. Each matches a recognizable
# substring anywhere in the stderr/stdout tail captured from the CLI.
#
#   _RE_EMPTY     : `AllProvidersFailed` exception where every provider
#                   reported "empty payload" (no rows returned). Recently
#                   listed tickers (no history yet at any provider) fall
#                   into this bucket.
#   _RE_OVERLAP   : SQLite `UNIQUE constraint failed: nse_equities_data...`
#                   the integrity error fixed in PR #37 (cache overlap
#                   during back-extend writes). Should be rare post-fix.
_RE_EMPTY = re.compile(
    r"AllProvidersFailed.*empty payload", re.DOTALL
)
_RE_OVERLAP = re.compile(
    r"(IntegrityError|UNIQUE constraint failed)", re.IGNORECASE
)


def classify_error(stderr: str) -> str:
    """Bucket a failure stderr/stdout string into one of the FAIL_* statuses.

    Pure function — no side effects, only string matching. Returns one of
    ``FAIL_EMPTY``, ``FAIL_OVERLAP``, or ``FAIL_OTHER``.

    Callers that want the OK/OK_NOOP buckets should use
    :func:`classify_outcome` which combines this with the row-delta.
    """
    if not stderr:
        return STATUS_FAIL_OTHER
    if _RE_EMPTY.search(stderr):
        return STATUS_FAIL_EMPTY
    if _RE_OVERLAP.search(stderr):
        return STATUS_FAIL_OVERLAP
    return STATUS_FAIL_OTHER


def classify_outcome(rc: int, delta: int, stderr: str) -> str:
    """Combine return-code + row-delta + stderr into a single status bucket.

    * rc == 0 and delta > 0 → ``OK``
    * rc == 0 and delta == 0 → ``OK_NOOP``  (CLI ran, no new rows landed)
    * rc != 0 → :func:`classify_error` on stderr
    """
    if rc == 0:
        return STATUS_OK if (delta is not None and delta > 0) else STATUS_OK_NOOP
    return classify_error(stderr or "")


# ---------------------------------------------------------------------------
# Error-message extraction (short form for the per-ticker line)
# ---------------------------------------------------------------------------

# Match a Python exception class + message dump from the CLI's stderr — the
# last line of a traceback is typically ``<ExceptionClass>: <message>``.
# We collapse newlines and pipes in the captured tail to keep the per-ticker
# log on one line.
_RE_EXC_LINE = re.compile(
    r"([A-Za-z_][\w\.]*Error|AllProvidersFailed|Exception)\s*:\s*(.*)"
)


def extract_err_short(stderr: str, max_len: int = 80) -> str:
    """Return a short ``ExceptionClass:message`` extract for the err= field.

    Falls back to the first 80 chars of the stderr tail if no exception
    line is matched. Always returns a single-line string ≤ max_len chars.
    """
    if not stderr:
        return ""
    # Scan from the end — traceback's "ExceptionClass: ..." is last.
    matches = list(_RE_EXC_LINE.finditer(stderr))
    if matches:
        m = matches[-1]
        cls = m.group(1)
        msg = m.group(2).strip().splitlines()[0]
        out = f"{cls}:{msg}"
    else:
        out = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    out = out.replace("|", "/").replace("\t", " ")
    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out


# ---------------------------------------------------------------------------
# Per-ticker line formatter
# ---------------------------------------------------------------------------


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_per_ticker_line(
    *,
    idx: int,
    total: int,
    sym: str,
    status: str,
    before: int,
    after: int,
    delta: int | str,
    took_s: float,
    elapsed_m: float,
    eta_m: float,
    err_short: str = "",
    ts: str | None = None,
) -> str:
    """Render the structured per-ticker log line.

    Format:
        <ts> [<idx>/<total>] <sym> status=<S> rows=B->A(+D) took=T err=... elapsed=E eta=ETA
    """
    ts = ts or utc_ts()
    err_field = f" err={err_short}" if err_short else ""
    return (
        f"{ts} [{idx:>3}/{total}] {sym} "
        f"status={status} rows={before}->{after}(+{delta}) "
        f"took={took_s:.1f}s{err_field} "
        f"elapsed={elapsed_m:.1f}m eta={eta_m:.1f}m"
    )


# ---------------------------------------------------------------------------
# Rolling summary
# ---------------------------------------------------------------------------


@dataclass
class RollingCounter:
    """Accumulate per-status counts and render the rolling-summary line.

    Use:

        ctr = RollingCounter(total=697, t_start=time.time())
        for i, sym in enumerate(symbols, 1):
            status = ...
            ctr.bump(status)
            if i % ctr.interval == 0:
                log(ctr.render(done=i))
    """
    total: int
    t_start: float
    interval: int = 50
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for s in ALL_STATUSES:
            self.counts.setdefault(s, 0)

    def bump(self, status: str) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1

    def render(self, *, done: int, ts: str | None = None) -> str:
        ts = ts or utc_ts()
        elapsed_s = max(time.time() - self.t_start, 1e-6)
        elapsed_m = elapsed_s / 60.0
        rate = done / (elapsed_s / 60.0) if elapsed_s > 0 else 0.0
        eta_m = ((self.total - done) / rate) if rate > 0 else 0.0
        pct = int(round(100 * done / max(self.total, 1)))
        parts = " ".join(f"{s}={self.counts.get(s, 0)}" for s in ALL_STATUSES)
        return (
            f"{ts} [ROLLING {done}/{self.total} done={pct}% "
            f"elapsed={elapsed_m:.1f}m] {parts} "
            f"eta={eta_m:.1f}m rate={rate:.2f}/min"
        )


# ---------------------------------------------------------------------------
# Summary.json writer
# ---------------------------------------------------------------------------


@dataclass
class SummaryWriter:
    """Accumulate per-ticker outcomes and write a final summary.json artifact.

    Writes on normal completion via :meth:`finalize`, and also on SIGINT /
    SIGTERM if :meth:`install_signal_handlers` is called.
    """
    out_path: Path
    total_tickers: int
    fetch_plan_path: str | None = None
    start_utc: str = field(default_factory=utc_ts)
    _t_start: float = field(default_factory=time.time)
    outcomes: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    completed: int = 0
    _written: bool = False

    def __post_init__(self) -> None:
        for s in ALL_STATUSES:
            self.outcomes.setdefault(s, 0)
            self.examples.setdefault(s, [])

    def record(self, *, status: str, sym: str, extra: str = "") -> None:
        """Record one outcome. extra is e.g. "+1246 rows" or "ProviderError:...".

        Capped at 5 examples per bucket — enough to debug, not enough to bloat.
        """
        self.outcomes[status] = self.outcomes.get(status, 0) + 1
        self.completed += 1
        bucket = self.examples.setdefault(status, [])
        if len(bucket) < 5:
            entry = f"{sym} {extra}".rstrip()
            bucket.append(entry)

    def finalize(self) -> Path:
        """Write summary.json. Idempotent — only writes once."""
        if self._written:
            return self.out_path
        end_utc = utc_ts()
        elapsed_m = (time.time() - self._t_start) / 60.0
        payload = {
            "start_utc": self.start_utc,
            "end_utc": end_utc,
            "elapsed_minutes": round(elapsed_m, 2),
            "total_tickers": self.total_tickers,
            "completed": self.completed,
            "outcomes": dict(self.outcomes),
            "per_status_examples": {k: list(v) for k, v in self.examples.items()},
            "fetch_plan_path": self.fetch_plan_path,
            "fetch_plan_size": self.total_tickers,
            "wrote_summary_at_pid": os.getpid(),
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, self.out_path)
        self._written = True
        return self.out_path

    def install_signal_handlers(
        self, log_fn: Callable[[str], None] | None = None
    ) -> None:
        """Install SIGINT/SIGTERM handlers that flush the summary then re-raise.

        Safe to call multiple times — last writer wins. After flushing, the
        handler restores the default signal disposition and re-raises, so
        the process still terminates promptly with the expected exit code.
        """
        def _handler(signum, _frame):
            try:
                path = self.finalize()
                if log_fn:
                    log_fn(
                        f"INTERRUPTED signal={signum} — wrote partial "
                        f"summary to {path}"
                    )
            except Exception as e:  # pragma: no cover — best-effort flush
                if log_fn:
                    log_fn(f"INTERRUPTED signal={signum} — summary flush failed: {e}")
            # Restore default and re-raise to let the process die naturally.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
