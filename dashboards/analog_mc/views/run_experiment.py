"""Streamlit view: launch a walk-forward run as a subprocess and stream progress.

Per [[project-streamlit]], multiple concurrent runs are supported via per-run
lockfiles in their own timestamped directories — no global mutex.

The walk-forward runs in a subprocess (NOT in the Streamlit process — long
loops break the Streamlit rerun model). Progress is shown by polling the
filesystem: each fold writes a ``summary.json`` when it finishes.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from analog_mc import Config

from dashboards.analog_mc.views._shared import list_configs as _list_configs


def _list_active_runs(runs_root: Path) -> list[Path]:
    """Runs with a ``lock`` file are considered active."""
    if not runs_root.exists():
        return []
    return [r for r in sorted(runs_root.iterdir()) if r.is_dir() and (r / "lock").exists()]


def _list_all_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        (r for r in runs_root.iterdir() if r.is_dir() and (r / "config.yaml").exists()),
        key=lambda p: p.name, reverse=True,
    )


def _progress_summary(run_dir: Path) -> tuple[int, list[dict]]:
    """Return (n_completed_folds, list_of_summaries) by scanning the run dir."""
    folds_dir = run_dir / "folds"
    if not folds_dir.exists():
        return 0, []
    out = []
    for fd in sorted(folds_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        sp = fd / "summary.json"
        if sp.exists():
            out.append(json.loads(sp.read_text()))
    return len(out), out


def render(
    configs_root: Path = Path("configs/analog_mc"),
    runs_root: Path = Path("runs/analog_mc"),
) -> None:
    st.title("analog_mc — run experiment")

    configs = _list_configs(configs_root)
    if not configs:
        st.warning(
            f"No configs found in `{configs_root}`. Create one in the *Config editor* view."
        )
        return

    config_name = st.selectbox("config", options=[c.name for c in configs])
    config_path = configs_root / config_name
    cfg = Config.from_yaml(config_path)

    ticker_override = st.text_input("ticker override (optional)", value="", placeholder=cfg.ticker)
    resume = st.checkbox("resume completed folds if run-dir is reused", value=True)
    st.caption(
        f"Run will be persisted under `{runs_root}/<timestamp>/`. "
        f"Multiple concurrent runs are safe — each gets its own directory."
    )

    if st.button("Launch", type="primary"):
        cmd = [
            sys.executable, "-m", "analog_mc", "walk-forward",
            "--config", str(config_path),
        ]
        if ticker_override:
            cmd += ["--ticker", ticker_override]
        if not resume:
            cmd.append("--no-resume")

        st.info(f"Launching: `{shlex.join(cmd)}`")
        # Detach: writes its own log via Python logging to stderr; we capture
        # to a file so the dashboard can tail it later.
        runs_root.mkdir(parents=True, exist_ok=True)
        log_path = runs_root / f"_launch_{int(time.time())}.log"
        log_handle = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        st.session_state["last_pid"] = proc.pid
        st.session_state["last_log"] = str(log_path)
        st.success(f"Launched (pid {proc.pid}). Tail `{log_path}` for logs.")

    # ---- Active runs ----
    st.divider()
    st.subheader("Active runs (with `lock` file present)")
    active = _list_active_runs(runs_root)
    if not active:
        st.caption("None.")
    else:
        for run_dir in active:
            with st.container(border=True):
                n_done, summaries = _progress_summary(run_dir)
                st.write(f"**{run_dir.name}**")
                if summaries:
                    latest = summaries[-1]
                    st.caption(
                        f"completed folds: **{n_done}**  ·  "
                        f"latest test CRPS: **{latest['test_crps']:.5f}**  ·  "
                        f"latest val CRPS: {latest['val_crps']:.5f}"
                    )
                else:
                    st.caption(f"completed folds: **{n_done}**")
                if st.button("refresh", key=f"refresh_{run_dir.name}"):
                    st.rerun()

    # ---- All runs ----
    st.divider()
    st.subheader("All runs in this module's runs directory")
    all_runs = _list_all_runs(runs_root)
    for run_dir in all_runs[:20]:
        n_done, _ = _progress_summary(run_dir)
        locked = (run_dir / "lock").exists()
        st.write(f"- `{run_dir.name}` — {n_done} folds  {'🔒 active' if locked else '✓ complete'}")
