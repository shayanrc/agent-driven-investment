"""Persistent on-disk observability for the agent-driven FS+HP loop (task #177).

When an ``agent_file_protocol`` loop stalls, its only useful signals — the
runner's milestone prints (``[loop] paused …``, ``[resume] …``, the phase
banners) and the 30s ``[heartbeat]`` ticks — go to the driving sub-agent's
stdout/transcript and are NOT persisted on disk. Diagnosing "where is it / is
it alive / why did it stop" then means ``ps`` + file-mtime + transcript
archaeology. Each ``agent_file_protocol`` iteration is moreover a SEPARATE
process invocation (the runner trains one iter, writes ``loop/`` files, prints
``[loop] paused``, and exits 0; the agent relaunches ``--resume``), so stdout
from a prior invocation is already gone by the time the next one runs.

This module persists those signals into the cell's ``loop/`` directory so they
survive across resume subprocesses:

- :class:`ProgressLog` — an **append-only** ``loop/progress.log``. Each run
  invocation appends timestamped (UTC ISO) lines for the milestones it already
  prints AND the periodic heartbeat ticks. ``tail loop/progress.log`` tells the
  whole story of a run across every resume boundary. The append mode is what
  makes it survive the separate-process model: every invocation re-opens the
  same path in ``"a"`` mode, so line N+1's invocation does not clobber line N's.
- :class:`StatusFile` — a single-shot, **overwrite** ``loop/status.json`` giving
  a monitor the current position (iter/phase/awaiting_decision) + liveness
  (``now - last_heartbeat_utc``) from ONE file read.
- :class:`TeeStream` — a tiny multiplexing text stream: writes go to BOTH the
  underlying stream (``sys.stdout``, unchanged) AND the :class:`ProgressLog`.
  The :class:`~gbdt.heartbeat.Heartbeat` already takes a ``stream`` param, so
  pointing it at a :class:`TeeStream` mirrors every ``[heartbeat]`` line into
  ``progress.log`` with zero changes to the heartbeat's stdout behaviour.

Design constraints (task #177):

- Existing stdout prints + the heartbeat's stdout output are unchanged. The tee
  *adds* a destination; it never removes stdout.
- ``GBDT_HEARTBEAT_INTERVAL=0`` (heartbeat disabled) must still work and should
  no-op the file heartbeat too — handled because the heartbeat thread simply
  never ticks, so it never calls into the tee/status; explicit milestone lines
  still get appended by the runner directly.
- Loop semantics, the checkpoint schema, determinism, and the request/decision
  protocol are untouched. ``status.json`` is ADDED beside ``checkpoint.json``;
  it does not repurpose it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbdt.checkpoint import _LOOP_SUBDIR

PROGRESS_LOG_FILENAME = "progress.log"
STATUS_FILENAME = "status.json"

# Bumped on any breaking change to the status.json dict shape. Tracks the
# checkpoint/request schemas independently — status.json is a NEW, read-only
# monitoring surface, not a loop-control file.
STATUS_SCHEMA_VERSION = "v1"


def _utc_now_iso() -> str:
    """Current UTC time as a second-resolution ISO-8601 string with ``Z``-style
    offset (``+00:00``). Used for every timestamp this module writes so a
    monitor parsing ``last_heartbeat_utc`` can ``datetime.fromisoformat`` it
    directly."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Co-located path helpers (plan § 0.6 layout — alongside checkpoint.json)
# ---------------------------------------------------------------------------


def progress_log_path(artifact_dir: str | Path) -> Path:
    """``<artifact_dir>/loop/progress.log``."""
    return Path(artifact_dir) / _LOOP_SUBDIR / PROGRESS_LOG_FILENAME


def status_path(artifact_dir: str | Path) -> Path:
    """``<artifact_dir>/loop/status.json``."""
    return Path(artifact_dir) / _LOOP_SUBDIR / STATUS_FILENAME


# ---------------------------------------------------------------------------
# Append-only progress log
# ---------------------------------------------------------------------------


class ProgressLog:
    """Append-only, UTC-timestamped ``loop/progress.log``.

    Every milestone line and every heartbeat tick is appended with a leading
    ``<utc-iso> `` prefix. The file is opened in append mode on construction and
    kept open for the life of the run invocation; :meth:`close` flushes + closes.
    Because each invocation re-opens the SAME path in append mode, lines from a
    fresh run and from every ``--resume`` relaunch accumulate in one file — the
    log survives the separate-process model of ``agent_file_protocol``.

    Thread-safety: the heartbeat daemon thread and the main thread both append.
    A single ``write`` call on a buffered file object is effectively atomic for
    the short lines here, but we still guard with a lock to avoid interleaved
    partial writes when stdout-tee and the heartbeat race.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self._path = progress_log_path(artifact_dir)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append: each invocation continues the same file.
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        import threading

        self._lock = threading.Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def append(self, message: str) -> None:
        """Append one timestamped line. Newlines in ``message`` are stripped so
        one logical milestone is always one physical line (the heartbeat passes
        a pre-formatted line; milestone callers pass a bare message)."""
        if self._closed:
            return
        line = f"{_utc_now_iso()} {message.rstrip(chr(10))}\n"
        with self._lock:
            try:
                self._fh.write(line)
                self._fh.flush()
            except (ValueError, OSError):
                # A progress-log write must NEVER crash the run (same discipline
                # as the heartbeat's _emit). A closed/failed FH is swallowed.
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.flush()
                self._fh.close()
            except (ValueError, OSError):
                pass

    def __enter__(self) -> "ProgressLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Tee stream — mirrors writes to stdout AND the progress log
# ---------------------------------------------------------------------------


class TeeStream:
    """A minimal text stream that forwards ``write``/``flush`` to an underlying
    stream (``sys.stdout``) AND mirrors each newline-terminated line into a
    :class:`ProgressLog`.

    Used as the :class:`~gbdt.heartbeat.Heartbeat` ``stream`` so every
    ``[heartbeat]`` line lands in ``progress.log`` without changing the
    heartbeat's stdout output. ``print(..., file=tee)`` and the heartbeat's
    ``print(..., file=self._stream, flush=True)`` both work unchanged.

    The tee buffers partial writes until a newline so multi-``write`` ``print``
    calls (text, then ``"\\n"``) record as a single progress-log line.
    """

    def __init__(self, underlying, progress: ProgressLog) -> None:
        self._underlying = underlying
        self._progress = progress
        self._buf = ""

    def write(self, s: str) -> int:
        n = self._underlying.write(s)
        # Mirror complete lines into the progress log; buffer the tail.
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:  # skip blank lines
                self._progress.append(line)
        return n

    def flush(self) -> None:
        self._underlying.flush()

    # Some libraries probe these; forward to the underlying stream.
    def isatty(self) -> bool:
        return getattr(self._underlying, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._underlying, "encoding", "utf-8")


# ---------------------------------------------------------------------------
# Single-shot machine-readable status
# ---------------------------------------------------------------------------


class StatusFile:
    """Overwrite-on-update ``loop/status.json`` — a monitor's one-file read.

    Carries position (``iter_idx`` / ``phase`` / ``awaiting_decision``) +
    liveness (``last_update_utc`` / ``last_heartbeat_utc``) + the headline
    ``best_val_brier`` + a terminal ``stop_reason`` (``None`` until the loop
    completes/stops). Updated at every phase transition and on pause / resume /
    complete. The heartbeat thread refreshes ``last_heartbeat_utc`` each tick so
    ``now - last_heartbeat_utc`` is the staleness signal.

    State is held in memory and re-serialized on every mutation (the file is
    tiny). Writes are guarded by a lock because the heartbeat thread refreshes
    the heartbeat timestamp concurrently with the main thread's phase updates.
    """

    def __init__(self, artifact_dir: str | Path, *, run_id: str) -> None:
        self._path = status_path(artifact_dir)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        import threading

        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "run_id": str(run_id),
            "iter_idx": None,
            "phase": "init",
            "awaiting_decision": False,
            "last_update_utc": _utc_now_iso(),
            "last_heartbeat_utc": None,
            "best_val_brier": None,
            "stop_reason": None,
        }

    @property
    def path(self) -> Path:
        return self._path

    def _flush_locked(self) -> None:
        try:
            self._path.write_text(json.dumps(self._state, indent=2, default=str))
        except OSError:
            # A status write must never crash the run.
            pass

    def update(
        self,
        *,
        iter_idx: int | None = None,
        phase: str | None = None,
        awaiting_decision: bool | None = None,
        best_val_brier: float | None = None,
        stop_reason: str | None = None,
        touch_update: bool = True,
    ) -> None:
        """Mutate the supplied fields and re-serialize.

        Only non-``None`` kwargs overwrite; the rest carry forward. ``stop_reason``
        is special — once set it is sticky-overwritable (a terminal value), but
        passing ``None`` never clears a prior reason. ``touch_update`` bumps
        ``last_update_utc`` (always true for real transitions; the heartbeat
        path uses :meth:`heartbeat` which does NOT bump ``last_update_utc`` so
        a monitor can tell "position changed" from "still alive")."""
        with self._lock:
            if iter_idx is not None:
                self._state["iter_idx"] = int(iter_idx)
            if phase is not None:
                self._state["phase"] = str(phase)
            if awaiting_decision is not None:
                self._state["awaiting_decision"] = bool(awaiting_decision)
            if best_val_brier is not None:
                self._state["best_val_brier"] = float(best_val_brier)
            if stop_reason is not None:
                self._state["stop_reason"] = str(stop_reason)
            if touch_update:
                self._state["last_update_utc"] = _utc_now_iso()
            self._flush_locked()

    def heartbeat(self) -> None:
        """Refresh ``last_heartbeat_utc`` only (called by the heartbeat thread
        each tick). Does NOT bump ``last_update_utc`` — a monitor distinguishes
        liveness (heartbeat advancing) from progress (position changing)."""
        with self._lock:
            self._state["last_heartbeat_utc"] = _utc_now_iso()
            self._flush_locked()

    def read(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


# ---------------------------------------------------------------------------
# Liveness verdict (shared by the runner-side status + the reader script)
# ---------------------------------------------------------------------------


def liveness_verdict(
    status: dict[str, Any],
    *,
    now: datetime | None = None,
    stale_after_sec: float = 90.0,
) -> str:
    """Derive a compact ALIVE/STALE/DONE verdict from a parsed status dict.

    - A terminal ``stop_reason`` -> ``"DONE (reason=...)"``.
    - No ``last_heartbeat_utc`` (heartbeat disabled / not yet ticked) ->
      ``"NO-HEARTBEAT (last_update Ns ago)"`` using ``last_update_utc``.
    - heartbeat within ``stale_after_sec`` -> ``"ALIVE (heartbeat Ns ago)"``.
    - heartbeat older than ``stale_after_sec`` -> ``"STALE (heartbeat Ns ago)"``.

    ``stale_after_sec`` defaults to 90s = 3× the 30s heartbeat cadence (two
    missed ticks before we call it stale, per the heartbeat module's own
    "detectable in ~2 intervals" framing).
    """
    now = now or datetime.now(timezone.utc)
    reason = status.get("stop_reason")
    if reason:
        return f"DONE (reason={reason})"

    def _age(key: str) -> float | None:
        ts = status.get(key)
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt).total_seconds()

    hb_age = _age("last_heartbeat_utc")
    if hb_age is None:
        upd_age = _age("last_update_utc")
        if upd_age is None:
            return "UNKNOWN (no timestamps)"
        return f"NO-HEARTBEAT (last_update {upd_age:.0f}s ago)"
    verdict = "ALIVE" if hb_age <= stale_after_sec else "STALE"
    return f"{verdict} (heartbeat {hb_age:.0f}s ago)"


def tail_lines(path: str | Path, n: int = 15) -> list[str]:
    """Return the last ``n`` lines of a text file (no trailing newlines).

    Reads the whole file (progress.log is small — a handful of lines per
    invocation). Returns ``[]`` if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:] if n > 0 else lines


__all__ = [
    "PROGRESS_LOG_FILENAME",
    "STATUS_FILENAME",
    "STATUS_SCHEMA_VERSION",
    "ProgressLog",
    "TeeStream",
    "StatusFile",
    "progress_log_path",
    "status_path",
    "liveness_verdict",
    "tail_lines",
]
