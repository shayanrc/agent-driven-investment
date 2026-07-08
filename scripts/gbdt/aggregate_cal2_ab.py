"""Aggregate the V1.9 calendar2 (F21) matched A/B and (optionally) register rows.

Reads the 6 cal2_ab experiment artifacts (3 nasdaq100 cells × {base, cal2}),
computes test-segment metrics via the canonical registry helper
(``regenerate_r_precision_at_k_csv.compute_row`` — same min(K, R_q) denominator,
(p_calibrated desc, ticker asc) tie-break), prints the base-vs-cal2 comparison +
delta tables for the memo, and with ``--register`` appends the 6 rows to
``results/gbdt/data/r_precision_at_k.csv`` (existing rows preserved byte-for-byte;
re-sorted by AUC desc so the new rows land at their sorted positions).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.gbdt.regenerate_r_precision_at_k_csv import KS, compute_row

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "results/gbdt/experiments"

# (display cell, threshold/horizon label)
CELLS = [
    ("nasdaq100_up_50pct_25d_dd25pct", "+50% / 25d"),
    ("nasdaq100_up_20pct_50d_dd10pct", "+20% / 50d"),
    ("nasdaq100_up_40pct_200d_dd20pct", "+40% / 200d"),
]
ARMS = ("base", "cal2")

_METRIC_COLS = ["rows", "Q_days", "base_rate", "AUC"] + [f"R_precision_at_{k}" for k in KS]
_REGIME = {"mode": "sweep", "n_iterations_run": "1", "backend": "xgboost"}
_DATE_COLS = ("train_start", "train_end", "val_start", "val_end",
              "eval_start", "eval_end", "test_start", "test_end")


def _load(cell: str, arm: str) -> dict | None:
    name = f"{cell}_cal2ab_{arm}"
    tp = EXP_DIR / name / "predictions" / "test.csv"
    if not tp.is_file():
        return None
    return compute_row(name, tp)


def _fmt(v, nd=4):
    return "—" if v is None else f"{float(v):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true",
                    help="append the 6 rows to r_precision_at_k.csv")
    args = ap.parse_args()

    data = {}
    for cell, _ in CELLS:
        for arm in ARMS:
            data[(cell, arm)] = _load(cell, arm)

    missing = [k for k, v in data.items() if v is None]
    if missing:
        print(f"MISSING artifacts: {missing}")
        return 1

    # --- Per-cell comparison table (raw metric + base_rate; no lift columns) ---
    print("\n### Test-segment metrics (base = all; cal2 = all_calendar2)\n")
    hdr = "| cell | arm | base_rate | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |"
    print(hdr)
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for cell, lbl in CELLS:
        for arm in ARMS:
            r = data[(cell, arm)]
            print(f"| {lbl} | {arm} | {_fmt(r['base_rate'])} | {_fmt(r['AUC'])} | "
                  + " | ".join(_fmt(r[f'R_precision_at_{k}']) for k in KS) + " |")

    # --- Delta table (cal2 − base) ---
    print("\n### Delta (cal2 − base)\n")
    print("| cell | ΔAUC | ΔR-p@1 | ΔR-p@3 | ΔR-p@5 | ΔR-p@10 | ΔR-p@20 |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for cell, lbl in CELLS:
        b, c = data[(cell, "base")], data[(cell, "cal2")]
        dauc = c["AUC"] - b["AUC"]
        drp = [c[f"R_precision_at_{k}"] - b[f"R_precision_at_{k}"] for k in KS]
        print(f"| {lbl} | {dauc:+.4f} | " + " | ".join(f"{x:+.4f}" for x in drp) + " |")

    if args.register:
        csv = REPO / "results/gbdt/data/r_precision_at_k.csv"
        # Byte-append the 6 rows at EOF. The existing file has mixed line
        # endings (some CRLF, some LF from cross-branch appends); reading+
        # rewriting the whole file would normalize those and produce a huge
        # spurious diff. Appending raw bytes leaves every existing line
        # untouched -> the diff is exactly +6 lines. We do NOT re-sort by AUC
        # (the file is only globally sorted right after a full regenerate pass;
        # per-branch appends already break global order — a future regenerate
        # re-sorts everything).
        raw = csv.read_bytes()
        cols = raw.split(b"\n", 1)[0].decode().strip().split(",")

        def render(r: dict) -> str:
            vals = {"experiment": r["experiment"], **_REGIME}
            for mc in _METRIC_COLS:
                vals[mc] = f"{float(r[mc]):.6f}"
            for dc in _DATE_COLS:
                vals[dc] = r.get(dc, "") or ""
            return ",".join(str(vals[c]) for c in cols)

        if not raw.endswith(b"\n"):
            raw += b"\n"
        new = sorted(data.values(), key=lambda r: float(r["AUC"]), reverse=True)
        addition = "".join(render(r) + "\n" for r in new).encode()
        csv.write_bytes(raw + addition)
        print(f"\nRegistered 6 rows (byte-appended) -> {csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
