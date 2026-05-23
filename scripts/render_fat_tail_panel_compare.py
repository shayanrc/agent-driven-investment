"""Render side-by-side comparison panels for the 15 fat-tail anchors.

Companion to render_fat_tail_panel.py. Loads cached forecasts.npz from N
canonical runs and emits comparison figures. Defaults to the v4 set
(v2.4, B1, A2.1, B5) but accepts an arbitrary experiment list via
--experiment so new canonicals (V5.A.2, V5.B, future versions) can be folded
in without editing the script.

Output modes (mutually exclusive — pass one):
    (default)          15 per-anchor 2x2 grid PNGs (one per anchor)
    --combined         15-anchor overlay panel (medians + 90% bands per experiment)
    --event YYYY-MM-DD event-level panel: forecasts (top row) + per-event PIT
                       histograms (bottom row), one column per experiment
    --experiment-grid  N rows (experiments) x (n_anchors + 1) cols: 15 forecasts
                       + aggregated PIT histogram per experiment

Default experiment set:
    uv run python scripts/render_fat_tail_panel_compare.py --experiment-grid

Custom experiment set (e.g., adding a V5 canonical):
    uv run python scripts/render_fat_tail_panel_compare.py --experiment-grid \\
        --experiment "v2.4=runs/analog_mc/20260520T045525Z" \\
        --experiment "V5.A.2=runs/analog_mc/20260601T012345Z"

Colors auto-assigned from the matplotlib default cycle when a label isn't in
EXP_COLORS below. Add canonical labels there to lock a color across reruns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import load_close_series

ROOT = Path(__file__).resolve().parents[1]
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"

DEFAULT_EXPERIMENTS = [
    ("v2.4 baseline (Cell-D-s30)",       "runs/analog_mc/20260520T045525Z"),
    ("B1 (Platzer local-linear)",        "runs/analog_mc/20260520T155220Z"),
    ("A2.1 (corrwindow L=100)",          "runs/analog_mc/20260521T061730Z"),
    ("B5 (joint A2.1 + B1)",             "runs/analog_mc/20260521T121025Z"),
]

EXP_COLORS = {
    "v2.4 baseline (Cell-D-s30)": "tab:grey",
    "B1 (Platzer local-linear)":  "tab:blue",
    "A2.1 (corrwindow L=100)":    "tab:red",
    "B5 (joint A2.1 + B1)":       "tab:purple",
}


def parse_experiment_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--experiment expects 'LABEL=RUN_DIR', got: {spec!r}")
    label, run = spec.rsplit("=", 1)
    return label.strip(), run.strip()


def assign_colors(labels: list[str]) -> dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    used: set[str] = set()
    out: dict[str, str] = {}
    fallback_idx = 0
    for lab in labels:
        if lab in EXP_COLORS:
            out[lab] = EXP_COLORS[lab]
            used.add(EXP_COLORS[lab])
    for lab in labels:
        if lab in out:
            continue
        while fallback_idx < len(cycle) and cycle[fallback_idx] in used:
            fallback_idx += 1
        if fallback_idx >= len(cycle):
            fallback_idx = 0
        out[lab] = cycle[fallback_idx]
        used.add(cycle[fallback_idx])
        fallback_idx += 1
    return out


def all_anchor_dates() -> list[str]:
    payload = json.loads(ANCHORS_JSON.read_text())
    return [
        e["anchor_date"]
        for sec in ("positive", "negative", "regime_coverage")
        for e in payload.get(sec, [])
    ]


def load_fold_summaries(run_dir: Path) -> list[dict]:
    return [
        json.loads((run_dir / "folds" / d.name / "summary.json").read_text())
        for d in sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name))
    ]


def find_origin(close: pd.Series, date_str: str) -> tuple[int, pd.Timestamp]:
    target = pd.Timestamp(date_str)
    pos = close.index.searchsorted(target)
    if pos >= len(close):
        raise SystemExit(f"date {date_str} past end of data ({close.index[-1].date()})")
    return pos - 1, close.index[pos]


def quantiles_for_anchor(
    date_str: str, run_dir: Path, close: pd.Series, folds: list[dict], horizon: int,
    keep_paths: bool = False,
) -> dict | None:
    origin_idx, actual_date = find_origin(close, date_str)
    fold = next((f for f in folds if f["test_start"] <= origin_idx <= f["test_end"]), None)
    if fold is None:
        return None
    arr = np.load(run_dir / "folds" / str(fold["fold_index"]) / "forecasts.npz")
    matches = np.where(arr["origin_idx"] == origin_idx)[0]
    if matches.size == 0:
        return None
    pos = int(matches[0])
    fc_returns = arr["paths"][pos]
    rl_returns = arr["realized"][pos]

    anchor_close_pos = origin_idx + 1
    p0 = float(close.iloc[anchor_close_pos])
    fc_prices = p0 * np.exp(np.cumsum(fc_returns, axis=1))
    rl_prices = p0 * np.exp(np.cumsum(rl_returns))
    forward_dates = close.index[anchor_close_pos + 1 : anchor_close_pos + 1 + horizon]

    q05 = np.quantile(fc_prices, 0.05, axis=0)
    q25 = np.quantile(fc_prices, 0.25, axis=0)
    q50 = np.quantile(fc_prices, 0.50, axis=0)
    q75 = np.quantile(fc_prices, 0.75, axis=0)
    q95 = np.quantile(fc_prices, 0.95, axis=0)
    in_50 = int(((rl_prices >= q25) & (rl_prices <= q75)).sum())
    in_90 = int(((rl_prices >= q05) & (rl_prices <= q95)).sum())

    hist_start = max(0, anchor_close_pos - 30)
    out = {
        "actual_date": actual_date,
        "anchor_close_pos": anchor_close_pos,
        "p0": p0,
        "forward_dates": forward_dates,
        "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95,
        "rl_prices": rl_prices,
        "hist_dates": close.index[hist_start : anchor_close_pos + 1],
        "hist_prices": close.iloc[hist_start : anchor_close_pos + 1].to_numpy(),
        "in_50": in_50, "in_90": in_90,
        "fold_index": fold["fold_index"],
    }
    if keep_paths:
        # PIT_t = P(forecast price <= realized price at step t), step-wise across horizon.
        # Randomized PIT for ties using the standard (rank - U) / n convention.
        n_paths, n_steps = fc_prices.shape
        pit = np.empty(n_steps)
        rng = np.random.default_rng(0)
        for t in range(n_steps):
            col = fc_prices[:, t]
            rl = rl_prices[t]
            below = int(np.sum(col < rl))
            ties = int(np.sum(col == rl))
            u = rng.uniform(0.0, 1.0) if ties > 0 else 0.0
            pit[t] = (below + u * ties) / n_paths
        out["pit"] = pit
        out["fc_prices"] = fc_prices
    return out


def render_comparison(date_str: str, panels: list[tuple[str, dict | None]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
    axes_flat = axes.flatten()

    valid = [p for _, p in panels if p is not None]
    if not valid:
        plt.close(fig)
        print(f"  SKIP {date_str}: no data for any experiment")
        return

    ymin = min(min(p["q05"].min(), p["rl_prices"].min(), p["hist_prices"].min()) for p in valid)
    ymax = max(max(p["q95"].max(), p["rl_prices"].max(), p["hist_prices"].max()) for p in valid)
    pad = 0.04 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    actual_date = valid[0]["actual_date"]

    for ax, (label, p) in zip(axes_flat, panels):
        if p is None:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue
        ax.plot(p["hist_dates"], p["hist_prices"], color="black", lw=1.3, label="historical")
        ax.fill_between(p["forward_dates"], p["q05"], p["q95"], color="tab:red", alpha=0.15, label="90% band")
        ax.fill_between(p["forward_dates"], p["q25"], p["q75"], color="tab:red", alpha=0.30, label="50% band")
        ax.plot(p["forward_dates"], p["q50"], color="tab:red", lw=1.4, label="median")
        ax.plot(p["forward_dates"], p["rl_prices"], color="black", lw=1.6, label="realized")
        ax.axvline(p["actual_date"], color="grey", lw=0.5, ls=":")
        ax.scatter([p["actual_date"]], [p["p0"]], color="black", s=14, zorder=5)
        ax.set_ylim(ylim)
        ax.set_title(f"{label} · 50/60={p['in_50']:>2d}, 90/60={p['in_90']:>2d}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    axes_flat[0].legend(loc="best", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"NASDAQ100 — 60-day forecast vs realized (anchored {actual_date.date()})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_combined(
    anchor_panels: list[tuple[str, list[tuple[str, dict | None]]]],
    out_path: Path,
    colors: dict[str, str],
    experiment_labels: list[str],
) -> None:
    from matplotlib.lines import Line2D

    ncols = 3
    nrows = (len(anchor_panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else list(axes)

    for ax, (date_str, panels) in zip(axes_flat, anchor_panels):
        valid = [(lab, p) for lab, p in panels if p is not None]
        if not valid:
            ax.set_title(f"{date_str} (no data)")
            ax.axis("off")
            continue

        ref = valid[0][1]
        realized_end_pct = 100.0 * (ref["rl_prices"][-1] / ref["p0"] - 1.0)

        ax.plot(ref["hist_dates"], ref["hist_prices"], color="black", lw=1.0, alpha=0.6)
        ax.plot(ref["forward_dates"], ref["rl_prices"], color="black", lw=2.0, zorder=10)
        ax.scatter([ref["actual_date"]], [ref["p0"]], color="black", s=14, zorder=11)
        ax.axvline(ref["actual_date"], color="grey", lw=0.5, ls=":")

        cov_chips = []
        for label, p in valid:
            color = colors[label]
            short = label.split()[0]
            ax.fill_between(
                p["forward_dates"], p["q05"], p["q95"], color=color, alpha=0.10,
            )
            ax.plot(
                p["forward_dates"], p["q50"], color=color, lw=1.4,
            )
            cov_chips.append(f"{short} 90={p['in_90']:>2d}/60")

        ax.set_title(
            f"{ref['actual_date'].date()}  ·  realized 60d {realized_end_pct:+.1f}%",
            fontsize=11, loc="left",
        )
        ax.set_title("  ·  ".join(cov_chips), fontsize=8, loc="right", color="dimgrey")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes_flat[len(anchor_panels):]:
        ax.axis("off")

    legend_handles = [Line2D([0], [0], color="black", lw=2.0, label="Realized")] + [
        Line2D([0], [0], color=colors[lab], lw=1.6, label=lab) for lab in experiment_labels
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncols=max(2, len(experiment_labels) + 1), fontsize=10, frameon=False,
    )
    fig.suptitle(
        "NASDAQ100 — 60-day forecast comparison  "
        "(median lines + 90% bands; per-anchor 90-band coverage in upper-right of each subplot)",
        fontsize=13, y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_event(
    date_str: str, panels: list[tuple[str, dict | None]], out_path: Path,
    colors: dict[str, str],
) -> None:
    n_exp = len(panels)
    fig, axes = plt.subplots(2, n_exp, figsize=(5.2 * n_exp, 9), sharey="row")

    valid = [(lab, p) for lab, p in panels if p is not None]
    if not valid:
        plt.close(fig)
        print(f"  SKIP {date_str}: no data")
        return

    ref = valid[0][1]
    realized_end_pct = 100.0 * (ref["rl_prices"][-1] / ref["p0"] - 1.0)

    ymin = min(min(p["q05"].min(), p["rl_prices"].min(), p["hist_prices"].min()) for _, p in valid)
    ymax = max(max(p["q95"].max(), p["rl_prices"].max(), p["hist_prices"].max()) for _, p in valid)
    pad = 0.04 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    for col, (label, p) in enumerate(panels):
        ax_fc = axes[0, col]
        ax_pit = axes[1, col]
        if p is None:
            ax_fc.set_title(f"{label}\n(no data)")
            ax_fc.axis("off")
            ax_pit.axis("off")
            continue
        color = colors[label]

        ax_fc.plot(p["hist_dates"], p["hist_prices"], color="black", lw=1.2, alpha=0.7)
        ax_fc.fill_between(p["forward_dates"], p["q05"], p["q95"], color=color, alpha=0.15, label="90% band")
        ax_fc.fill_between(p["forward_dates"], p["q25"], p["q75"], color=color, alpha=0.30, label="50% band")
        ax_fc.plot(p["forward_dates"], p["q50"], color=color, lw=1.5, label="median")
        ax_fc.plot(p["forward_dates"], p["rl_prices"], color="black", lw=1.8, label="realized")
        ax_fc.axvline(p["actual_date"], color="grey", lw=0.5, ls=":")
        ax_fc.scatter([p["actual_date"]], [p["p0"]], color="black", s=14, zorder=5)
        ax_fc.set_ylim(ylim)
        ax_fc.set_title(
            f"{label}\n50/60={p['in_50']:>2d}  ·  90/60={p['in_90']:>2d}",
            fontsize=11,
        )
        ax_fc.grid(True, alpha=0.3)
        ax_fc.tick_params(axis="x", labelsize=8, rotation=25)
        ax_fc.tick_params(axis="y", labelsize=8)
        if col == 0:
            ax_fc.set_ylabel("NASDAQ100 close")
            ax_fc.legend(loc="best", fontsize=8, framealpha=0.9)

        n_bins = 10
        pit = p["pit"]
        ax_pit.hist(pit, bins=n_bins, range=(0, 1), color=color, alpha=0.7, edgecolor="black")
        ax_pit.axhline(len(pit) / n_bins, color="black", lw=1.0, ls="--",
                       label=f"uniform ({len(pit)//n_bins}/bin)")
        ks = float(np.max(np.abs(np.sort(pit) - np.linspace(1/(2*len(pit)), 1-1/(2*len(pit)), len(pit)))))
        below_5 = int(np.sum(pit < 0.05))
        above_95 = int(np.sum(pit > 0.95))
        ax_pit.set_title(
            f"PIT histogram  ·  KS={ks:.2f}  ·  tails: {below_5}<.05, {above_95}>.95",
            fontsize=10,
        )
        ax_pit.set_xlim(0, 1)
        ax_pit.set_xlabel("PIT value")
        ax_pit.grid(True, alpha=0.3)
        ax_pit.tick_params(labelsize=8)
        if col == 0:
            ax_pit.set_ylabel("Days in bin (60 total)")
            ax_pit.legend(loc="upper center", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"NASDAQ100 — event-level forecast comparison · anchored {ref['actual_date'].date()} · realized 60d {realized_end_pct:+.1f}%",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_experiment_grid(
    runs: list[tuple[str, Path, pd.Series, list[dict], int]],
    anchor_dates: list[str],
    out_path: Path,
    colors: dict[str, str],
) -> None:
    """N rows (one per experiment) x (n_anchors + 1) cols (forecasts + aggregated PIT)."""
    n_exp = len(runs)
    n_anchor = len(anchor_dates)

    panels_by_exp: list[tuple[str, list[tuple[str, dict | None]]]] = []
    for label, run_dir, close, folds, horizon in runs:
        row = []
        for date_str in anchor_dates:
            p = quantiles_for_anchor(date_str, run_dir, close, folds, horizon, keep_paths=True)
            row.append((date_str, p))
        panels_by_exp.append((label, row))

    ylims_per_anchor: dict[str, tuple[float, float]] = {}
    realized_pct_per_anchor: dict[str, float] = {}
    for col, date_str in enumerate(anchor_dates):
        valid_ps = [row[col][1] for _, row in panels_by_exp if row[col][1] is not None]
        if not valid_ps:
            continue
        ymin = min(min(p["q05"].min(), p["rl_prices"].min(), p["hist_prices"].min()) for p in valid_ps)
        ymax = max(max(p["q95"].max(), p["rl_prices"].max(), p["hist_prices"].max()) for p in valid_ps)
        pad = 0.04 * (ymax - ymin)
        ylims_per_anchor[date_str] = (ymin - pad, ymax + pad)
        ref = valid_ps[0]
        realized_pct_per_anchor[date_str] = 100.0 * (ref["rl_prices"][-1] / ref["p0"] - 1.0)

    fig = plt.figure(figsize=(2.4 * n_anchor + 6, 3.6 * n_exp))
    gs = fig.add_gridspec(
        n_exp, n_anchor + 1,
        width_ratios=[1] * n_anchor + [2.6],
        hspace=0.55, wspace=0.18,
        left=0.05, right=0.99, top=0.92, bottom=0.06,
    )

    for row, (label, row_panels) in enumerate(panels_by_exp):
        color = colors[label]
        short = label.split("(")[0].strip()

        for col, (date_str, p) in enumerate(row_panels):
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                rp = realized_pct_per_anchor.get(date_str)
                rp_str = f"\n({rp:+.1f}%)" if rp is not None else ""
                ax.set_title(f"{date_str}{rp_str}", fontsize=8, pad=4)
            if p is None:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.plot(p["hist_dates"], p["hist_prices"], color="black", lw=0.7, alpha=0.6)
            ax.fill_between(p["forward_dates"], p["q05"], p["q95"], color=color, alpha=0.15)
            ax.fill_between(p["forward_dates"], p["q25"], p["q75"], color=color, alpha=0.30)
            ax.plot(p["forward_dates"], p["q50"], color=color, lw=1.0)
            ax.plot(p["forward_dates"], p["rl_prices"], color="black", lw=1.1)
            ax.scatter([p["actual_date"]], [p["p0"]], color="black", s=6, zorder=5)
            if date_str in ylims_per_anchor:
                ax.set_ylim(ylims_per_anchor[date_str])
            ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
            ax.grid(True, alpha=0.2)
            ax.text(
                0.98, 0.04, f"90/60={p['in_90']}", transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2),
            )
            if col == 0:
                ax.set_ylabel(short, fontsize=11, fontweight="bold", color=color, labelpad=8)

        ax_pit = fig.add_subplot(gs[row, n_anchor])
        valid_pits = [p["pit"] for _, p in row_panels if p is not None]
        if not valid_pits:
            ax_pit.axis("off")
            continue
        all_pit = np.concatenate(valid_pits)
        n_bins = 20
        expected = len(all_pit) / n_bins
        ax_pit.hist(
            all_pit, bins=n_bins, range=(0, 1), orientation="horizontal",
            color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        ax_pit.axvline(expected, color="red", lw=1.4, ls="--", label=f"uniform ({expected:.0f}/bin)")
        sorted_pit = np.sort(all_pit)
        n = len(sorted_pit)
        ks = float(np.max(np.abs(sorted_pit - np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))))
        below_5 = float(np.mean(all_pit < 0.05))
        above_95 = float(np.mean(all_pit > 0.95))
        ax_pit.set_title(
            f"Aggregated PIT  ·  N={n}  ·  KS={ks:.2f}\n"
            f"P(<.05)={below_5:.2f}  ·  P(>.95)={above_95:.2f}",
            fontsize=9,
        )
        ax_pit.set_ylim(0, 1)
        ax_pit.set_ylabel("PIT rank", fontsize=9)
        ax_pit.set_xlabel("count", fontsize=9)
        ax_pit.grid(True, alpha=0.3)
        ax_pit.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax_pit.tick_params(labelsize=8)

    fig.suptitle(
        "v4 fat-tail evaluation — 15-anchor forecasts (rows = experiments) + aggregated PIT calibration (last column)",
        fontsize=14, y=0.985,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir", default="docs/analog_mc/experiments/figs/fat_tail_compare",
        help="output dir for comparison PNGs",
    )
    p.add_argument(
        "--combined", action="store_true",
        help="also emit a single combined PNG with all 15 anchors overlaid",
    )
    p.add_argument(
        "--skip-per-anchor", action="store_true",
        help="skip per-anchor 2x2 grid PNGs (useful when only refreshing --combined)",
    )
    p.add_argument(
        "--event", default=None,
        help="render a single event-level panel (forecast + PIT histogram per experiment) for the given anchor date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--experiment-grid", action="store_true",
        help="render one figure with N rows (experiments) x (n_anchors + 1) cols (forecasts + aggregated PIT)",
    )
    p.add_argument(
        "--experiment", action="append", default=None,
        metavar="LABEL=RUN_DIR",
        help="add an experiment to the comparison (repeatable). If omitted, uses the v4 default set.",
    )
    args = p.parse_args()
    out_dir = ROOT / args.out_dir

    spec = [parse_experiment_spec(s) for s in args.experiment] if args.experiment else DEFAULT_EXPERIMENTS
    labels = [lab for lab, _ in spec]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate experiment labels are not allowed")
    colors = assign_colors(labels)

    runs = []
    for label, rel in spec:
        run_dir = ROOT / rel
        cfg = Config.from_yaml(run_dir / "config.yaml")
        close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)
        folds = load_fold_summaries(run_dir)
        runs.append((label, run_dir, close, folds, cfg.forecast_horizon))

    print(f"Comparing {len(runs)} experiment(s):")
    for label, run_dir, *_ in runs:
        print(f"  {label}  [{colors[label]}]  {run_dir.relative_to(ROOT)}")
    print()

    if args.event is not None:
        panels = [
            (label, quantiles_for_anchor(args.event, run_dir, close, folds, horizon, keep_paths=True))
            for label, run_dir, close, folds, horizon in runs
        ]
        out_path = out_dir / f"event_{args.event.replace('-', '')}.png"
        render_event(args.event, panels, out_path, colors)
        valid = [(lab, p) for lab, p in panels if p is not None]
        if valid:
            line = ", ".join(f"{lab.split()[0]} 90={p['in_90']:>2d}" for lab, p in valid)
            print(f"  {args.event}: {line} -> {out_path}")
        return

    if args.experiment_grid:
        anchor_dates = all_anchor_dates()
        out_path = out_dir / "experiment_grid.png"
        render_experiment_grid(runs, anchor_dates, out_path, colors)
        print(f"Experiment grid -> {out_path}")
        return

    all_anchor_panels: list[tuple[str, list[tuple[str, dict | None]]]] = []
    for date_str in all_anchor_dates():
        panels = [
            (label, quantiles_for_anchor(date_str, run_dir, close, folds, horizon))
            for label, run_dir, close, folds, horizon in runs
        ]
        all_anchor_panels.append((date_str, panels))
        if not args.skip_per_anchor:
            out_path = out_dir / f"compare_{date_str.replace('-', '')}.png"
            render_comparison(date_str, panels, out_path)
        valid = [(lab, p) for lab, p in panels if p is not None]
        if valid:
            line = ", ".join(f"{lab.split()[0]} 90={p['in_90']:>2d}" for lab, p in valid)
            print(f"  {date_str}: {line}")

    if args.combined:
        combined_path = out_dir / "compare_all_anchors.png"
        render_combined(all_anchor_panels, combined_path, colors, labels)
        print(f"\nCombined panel -> {combined_path}")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
