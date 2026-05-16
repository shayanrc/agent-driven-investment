"""Streamlit view: load a completed walk-forward run and render diagnostics.

This view is a thin presentation layer. All pipeline logic lives in
``src/analog_mc/`` — this file only orchestrates UI, never computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analog_mc import (
    Config,
    acf_comparison,
    aggregate_crps_overall,
    aggregate_crps_per_fold,
    aggregate_crps_per_step,
    aggregate_crps_per_vol_regime,
    clip_hit_summary,
    conditional_pit_by_vol_regime,
    crps_surface_plot,
    decision_rules,
    global_pit_histogram,
    load_returns,
    load_run,
    reliability_diagram,
    weight_trajectory_plot,
)


def _list_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and (p / "config.yaml").exists()),
        key=lambda p: p.name, reverse=True,
    )


def _run_status(run_dir: Path) -> str:
    if (run_dir / "lock").exists():
        return "in-progress"
    if (run_dir / "summary.parquet").exists():
        return "complete"
    return "broken"


def render(runs_root: Path = Path("runs/analog_mc")) -> None:
    st.title("analog_mc — diagnostics")

    runs = _list_runs(runs_root)
    if not runs:
        st.warning(
            f"No runs found under `{runs_root}`. Use the *Run experiment* view to start one."
        )
        return

    labels = [f"{r.name}  ({_run_status(r)})" for r in runs]
    choice = st.sidebar.selectbox("run", options=range(len(runs)), format_func=lambda i: labels[i])
    run_dir = runs[choice]
    status = _run_status(run_dir)

    if status == "in-progress":
        st.warning("This run has a `lock` file — it may be in progress. Diagnostics will reflect only the folds completed so far.")

    # ---- Header summary ----
    meta = json.loads((run_dir / "meta.json").read_text())
    config = Config.from_yaml(run_dir / "config.yaml")
    cols = st.columns(4)
    cols[0].metric("ticker", config.ticker)
    cols[1].metric("folds completed", meta.get("n_folds_completed", "?"))
    cols[2].metric("config hash", (meta.get("config_hash") or "?")[:8])
    cols[3].metric(
        "git commit",
        (meta.get("git_commit") or "—")[:7] if meta.get("git_commit") else "—",
    )

    try:
        run = load_run(run_dir)
    except FileNotFoundError as exc:
        st.info(f"No completed folds yet in this run — diagnostics will appear after the first fold finishes. ({exc})")
        return

    # ---- Overall aggregates ----
    overall = aggregate_crps_overall(run)
    st.subheader("Aggregate OOS CRPS")
    cols = st.columns(3)
    cols[0].metric("mean", f"{overall['mean_crps']:.5f}")
    cols[1].metric("median", f"{overall['median_crps']:.5f}")
    cols[2].metric("origin × step pairs", f"{overall['n_origin_step_pairs']:,}")

    # ---- Decision rules verdict ----
    st.subheader("v2-trigger decision rules")
    try:
        returns = load_returns(config)
    except FileNotFoundError:
        st.warning(f"Data file `{config.data_path}` not found; some diagnostics disabled.")
        returns = None

    if returns is not None:
        rules = decision_rules(run, returns, fixed_baseline=None)
        for name, body in rules.items():
            icon = "🔥" if body["fired"] else "✅"
            with st.expander(f"{icon}  {name}  —  metric={body['metric']:.4f}"):
                st.write(body["recommendation"] or "(no recommendation)")

    # ---- Per-fold and per-step tables ----
    st.subheader("Per-fold summary")
    st.dataframe(aggregate_crps_per_fold(run), use_container_width=True)

    st.subheader("Per-horizon-step CRPS")
    st.line_chart(aggregate_crps_per_step(run).set_index("step")[["mean_crps"]])

    if returns is not None:
        st.subheader("Per-vol-regime CRPS")
        st.dataframe(aggregate_crps_per_vol_regime(run, returns), use_container_width=True)

    # ---- Plots ----
    st.subheader("Weight trajectory")
    st.pyplot(weight_trajectory_plot(run))

    st.subheader("CRPS surface (fold 0)")
    st.pyplot(crps_surface_plot(run, fold_index=0))

    st.subheader("Global PIT histogram")
    st.pyplot(global_pit_histogram(run))

    if returns is not None:
        st.subheader("Conditional PIT by vol regime")
        st.pyplot(conditional_pit_by_vol_regime(run, returns))

    st.subheader("Reliability diagram")
    st.pyplot(reliability_diagram(run))

    if returns is not None:
        st.subheader("Squared-return ACF: simulated vs realized")
        st.pyplot(acf_comparison(run, returns))

    st.subheader("Clip-hit summary")
    st.pyplot(clip_hit_summary(run))
