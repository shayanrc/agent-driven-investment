"""Daemon-thread heartbeat for the gbdt runner — liveness / stuck detection.

Emits a ``[heartbeat]`` line every ``interval`` seconds carrying the current
phase, phase-elapsed, total-elapsed, and RSS. A heartbeat that *stops
advancing* means the process is wedged (e.g. the NTFS near-full D-state hang
documented in ``feedback-disk-wedge-pattern``) — detectable by a monitor in
~2 intervals, long before any backstop timeout. This is what lets the runner
use a large backstop timeout instead of a tight one (V1.1 plan § 0.3).

CatBoost releases the GIL during training, so the heartbeat keeps ticking
through the fit. The thread is a daemon: in the runner's normal use (a CLI /
per-iteration subprocess) it dies with the process, so an unhandled exception
mid-run can't leak it. Call ``stop()`` on the normal path for clean library use.

Disable via ``GBDT_HEARTBEAT_INTERVAL=0`` (e.g. in tests/CI).
"""
from __future__ import annotations

import os
import resource
import sys
import threading
import time

DEFAULT_INTERVAL_SECONDS = 30.0


class Heartbeat:
    def __init__(self, *, interval: float = DEFAULT_INTERVAL_SECONDS, stream=None) -> None:
        self._interval = float(interval)
        self._stream = stream if stream is not None else sys.stdout
        self._phase = "init"
        self._t0 = time.time()
        self._phase_t0 = self._t0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, *, stream=None) -> "Heartbeat":
        """Build with the interval from ``GBDT_HEARTBEAT_INTERVAL`` (default 30s;
        ``0`` or negative disables — ``start()`` becomes a no-op)."""
        raw = os.environ.get("GBDT_HEARTBEAT_INTERVAL")
        try:
            interval = float(raw) if raw is not None else DEFAULT_INTERVAL_SECONDS
        except ValueError:
            interval = DEFAULT_INTERVAL_SECONDS
        return cls(interval=interval, stream=stream)

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._phase_t0 = time.time()

    @staticmethod
    def _rss_mb() -> float:
        # ru_maxrss is in KB on Linux (the project's environment).
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def _emit(self) -> None:
        with self._lock:
            phase = self._phase
            phase_elapsed = time.time() - self._phase_t0
        total = time.time() - self._t0
        print(
            f"[heartbeat] t={time.strftime('%H:%M:%S')} phase={phase} "
            f"elapsed_phase={phase_elapsed:.0f}s elapsed_total={total:.0f}s "
            f"rss_mb={self._rss_mb():.0f}",
            file=self._stream, flush=True,
        )

    def _run(self) -> None:
        # Event.wait returns True only when stop() is called; on timeout it
        # returns False -> emit. So we tick on a fixed cadence and exit promptly.
        while not self._stop_evt.wait(self._interval):
            try:
                self._emit()
            except Exception:
                pass  # a heartbeat must never crash the run

    def start(self) -> "Heartbeat":
        if not self.enabled or self._thread is not None:
            return self
        self._t0 = time.time()
        self._phase_t0 = self._t0
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="gbdt-heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


__all__ = ["Heartbeat", "DEFAULT_INTERVAL_SECONDS"]
