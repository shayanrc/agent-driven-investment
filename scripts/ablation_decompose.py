"""Multi-run ablation decomposition: emit comparison tables for paste into ABLATIONS.md.

Given a list of run directories (each is a completed walk_forward output), load
each via analog_mc.diagnostics.load_run and emit markdown-formatted tables for:

  - Aggregate mean/median CRPS
  - Per-step CRPS at h=1, 15, 30, 60
  - Per-vol-regime CRPS (low/mid/high)
  - sloped_global_pit + acf_seam_degradation decision-rule metrics
  - Pairwise per-fold win-rate matrix

Skips the fixed-weight baseline pass that hangs render_diagnostics on conditional
configs (~12 h re-eval). Cell-vs-cell deltas are the attribution mechanism for
ablations.

Usage:
    uv run python scripts/ablation_decompose.py \\
        A-fast:runs/analog_mc/20260516T170018Z \\
        A-canonical:runs/analog_mc/20260516T180000Z \\
        B-fast:runs/analog_mc/20260517T050831Z \\
        B-canonical:runs/analog_mc/20260517T145344Z \\
        D-fast:runs/analog_mc/20260517T070003Z \\
        C-fast:runs/analog_mc/<new>

Each arg is "<label>:<path>". The label is used as the column header in
output tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.data import load_close_series
from analog_mc.diagnostics import (
    aggregate_crps_overall,
    aggregate_crps_per_step,
    aggregate_crps_per_vol_regime,
    decision_rules,
    load_run,
)


def _parse_cell_arg(arg: str) -> tuple[str, Path]:
    """Parse '<label>:<path>'. The label is the column header."""
    if ":" not in arg:
        raise ValueError(f"Cell arg must be 'label:path'; got {arg!r}")
    label, _, path = arg.partition(":")
    return label.strip(), Path(path.strip())


def _md_table(rows: list[list[str]], headers: list[str]) -> str:
    """Render a list of row-lists as a github-flavored markdown table."""
    parts = ["| " + " | ".join(headers) + " |"]
    parts.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts)


def _fmt(x: float, fmt: str = ".5f") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return format(x, fmt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cells", nargs="+",
        help="One or more 'label:path' specs, e.g. 'B-fast:runs/analog_mc/20260517T050831Z'.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional output file to write the full markdown report. "
             "If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    cells = [_parse_cell_arg(a) for a in args.cells]
    print(f"== loading {len(cells)} runs", file=sys.stderr)
    runs = {label: load_run(path) for label, path in cells}
    labels = list(runs.keys())

    # Returns are shared across cells (all use the same NASDAQ100 data); load once
    # from the first cell's config.
    first_cfg = next(iter(runs.values())).config
    print(f"== loading returns from {first_cfg.data_path}", file=sys.stderr)
    prices = load_close_series(first_cfg.data_path, date_col=first_cfg.date_col, close_col=first_cfg.close_col)
    returns = np.log(prices).diff().dropna()
    returns.name = "log_return"

    sections: list[str] = []

    # ---- 1. Aggregate CRPS ----
    print("== aggregate CRPS", file=sys.stderr)
    overall = {label: aggregate_crps_overall(r) for label, r in runs.items()}
    rows = [
        ["mean_crps"]   + [_fmt(overall[l]["mean_crps"])   for l in labels],
        ["median_crps"] + [_fmt(overall[l]["median_crps"]) for l in labels],
        ["n_pairs"]     + [f"{overall[l]['n_origin_step_pairs']:,}" for l in labels],
    ]
    sections.append("## Aggregate CRPS\n\n" + _md_table(rows, ["metric"] + labels))

    # ---- 2. Per-step CRPS at headline horizons ----
    print("== per-step CRPS", file=sys.stderr)
    per_step = {label: aggregate_crps_per_step(r) for label, r in runs.items()}
    headline_steps = [1, 15, 30, 60]
    rows = []
    for h in headline_steps:
        row = [f"h={h}"]
        for l in labels:
            df = per_step[l]
            v = df.loc[df["step"] == h, "mean_crps"]
            row.append(_fmt(float(v.iloc[0]) if len(v) else float("nan")))
        rows.append(row)
    sections.append("## Per-step CRPS\n\n" + _md_table(rows, ["step"] + labels))

    # ---- 3. Per-vol-regime CRPS ----
    print("== per-vol-regime CRPS", file=sys.stderr)
    per_regime = {label: aggregate_crps_per_vol_regime(r, returns) for label, r in runs.items()}
    regimes = ["low_vol", "mid_vol", "high_vol"]
    rows = []
    for reg in regimes:
        row = [reg]
        for l in labels:
            df = per_regime[l]
            v = df.loc[df["regime"] == reg, "mean_crps"]
            row.append(_fmt(float(v.iloc[0]) if len(v) else float("nan")))
        rows.append(row)
    sections.append("## Per-vol-regime mean CRPS\n\n" + _md_table(rows, ["regime"] + labels))

    # ---- 4. Decision rule metrics ----
    print("== decision rules", file=sys.stderr)
    rules_by_cell = {label: decision_rules(r, returns, fixed_baseline=None) for label, r in runs.items()}
    rule_names = ["sloped_global_pit", "u_shaped_high_vol_pit", "acf_seam_degradation", "clip_hit_excessive"]
    rows = []
    for name in rule_names:
        row = [name]
        for l in labels:
            body = rules_by_cell[l].get(name, {})
            metric = body.get("metric", float("nan"))
            fired = body.get("fired", False)
            tag = "🔥" if fired else "✅"
            row.append(f"{tag} {_fmt(metric, '+.4f')}")
        rows.append(row)
    sections.append("## v2-trigger decision rules\n\n" + _md_table(rows, ["rule"] + labels))

    # ---- 5. Per-fold per-pair win-rate matrix ----
    # For each pair (i, j), what fraction of folds has run_i test_crps < run_j test_crps?
    # Only cells with matching fold sets contribute to a pair.
    print("== pair-wise per-fold win-rates", file=sys.stderr)
    summaries = {label: r.summary[["fold_index", "test_crps"]].copy() for label, r in runs.items()}
    rows = []
    for i_label in labels:
        row = [i_label]
        for j_label in labels:
            if i_label == j_label:
                row.append("—")
                continue
            merged = summaries[i_label].merge(
                summaries[j_label], on="fold_index", suffixes=("_i", "_j")
            )
            if merged.empty:
                row.append("n/a")
            else:
                win = float((merged["test_crps_i"] < merged["test_crps_j"]).mean())
                row.append(f"{win*100:.1f}%")
        rows.append(row)
    sections.append(
        "## Per-fold win-rate (row beats column)\n\n"
        + _md_table(rows, ["row \\\\ col"] + labels)
    )

    # ---- 6. Cell-to-cell deltas (relative to first label as baseline) ----
    base_label = labels[0]
    base_mean = overall[base_label]["mean_crps"]
    rows = []
    for l in labels[1:]:
        delta = overall[l]["mean_crps"] - base_mean
        rel = delta / base_mean * 100 if base_mean else float("nan")
        rows.append([l, _fmt(overall[l]["mean_crps"]), _fmt(delta, "+.5f"), _fmt(rel, "+.2f") + "%"])
    sections.append(
        f"## Mean CRPS deltas vs {base_label}\n\n"
        + _md_table(rows, ["cell", "mean_crps", "Δ vs baseline", "rel"])
    )

    output = "\n\n".join(sections) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"== wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
