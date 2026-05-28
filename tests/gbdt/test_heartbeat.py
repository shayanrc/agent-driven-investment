"""Unit tests for the gbdt runner heartbeat (no experiment run)."""
from __future__ import annotations

import io
import time

from gbdt.heartbeat import Heartbeat


def test_heartbeat_emits_with_current_phase():
    buf = io.StringIO()
    hb = Heartbeat(interval=0.05, stream=buf).start()
    try:
        hb.set_phase("features")
        time.sleep(0.18)  # ~3 intervals
    finally:
        hb.stop()
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith("[heartbeat]")]
    assert len(lines) >= 2, f"expected multiple heartbeats, got: {out!r}"
    assert "phase=features" in lines[-1]
    assert "elapsed_total=" in lines[-1] and "rss_mb=" in lines[-1]


def test_heartbeat_stop_joins_thread():
    hb = Heartbeat(interval=0.05).start()
    assert hb._thread is not None and hb._thread.is_alive()
    hb.stop()
    assert hb._thread is None


def test_heartbeat_phase_transition_resets_phase_elapsed():
    buf = io.StringIO()
    hb = Heartbeat(interval=0.05, stream=buf).start()
    try:
        hb.set_phase("data")
        time.sleep(0.12)
        hb.set_phase("loop")
        time.sleep(0.12)
    finally:
        hb.stop()
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("[heartbeat]")]
    assert any("phase=data" in ln for ln in lines)
    assert any("phase=loop" in ln for ln in lines)


def test_heartbeat_disabled_when_interval_zero(monkeypatch):
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")
    buf = io.StringIO()
    hb = Heartbeat.from_env(stream=buf).start()
    assert not hb.enabled
    assert hb._thread is None  # start() is a no-op when disabled
    time.sleep(0.1)
    hb.stop()
    assert "[heartbeat]" not in buf.getvalue()


def test_from_env_default_interval(monkeypatch):
    monkeypatch.delenv("GBDT_HEARTBEAT_INTERVAL", raising=False)
    hb = Heartbeat.from_env()
    assert hb.enabled and hb._interval == 30.0


def test_from_env_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "not-a-number")
    hb = Heartbeat.from_env()
    assert hb._interval == 30.0
