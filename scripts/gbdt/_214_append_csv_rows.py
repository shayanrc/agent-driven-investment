"""Append canonical R-Precision@K rows for task #214 winners (cells 1+3) +
the two reference rows for the methodology-replication memo:
  - sp500_up_50pct_50d_dd25pct_manual_cells13       (cell-1 winner)
  - sp500_up_20pct_25d_dd10pct_manual_cells13       (cell-3 winner)

Reads the winner test predictions CSV from
  results/gbdt/data/_214_<cell>_winner_test_predictions.csv
and computes the canonical schema row using the
(p_calibrated desc, ticker asc) stable mergesort tie-break + min(K, R_q)
denominator per [[project-r-precision-methodology]].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/mnt/122CEE982CEE765F/Workspace/wt-214-cells13")
CSV_PATH = REPO / "results/gbdt/data/r_precision_at_k.csv"
KS = (1, 3, 5, 10, 20)
COLS = ["experiment", "rows", "Q_days", "base_rate", "AUC",
        "R_precision_at_1", "R_precision_at_3", "R_precision_at_5",
        "R_precision_at_10", "R_precision_at_20"]


def row_from_predictions(name: str, csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    out = {"experiment": name, "rows": len(df), "base_rate": float(df["y_true"].mean())}
    out["AUC"] = float(roc_auc_score(df["y_true"], df["p_calibrated"]))
    by_day = [(d, g.sort_values(by=["p_calibrated", "ticker"],
                                ascending=[False, True], kind="mergesort"))
              for d, g in df.groupby("date")]
    Q = None
    for K in KS:
        ratios = []
        for _d, g in by_day:
            R_q = int(g["y_true"].sum())
            if R_q == 0:
                continue
            r_q = int(g.head(K)["y_true"].sum())
            ratios.append(r_q / min(K, R_q))
        if Q is None:
            Q = len(ratios)
        out[f"R_precision_at_{K}"] = float(np.mean(ratios)) if ratios else float("nan")
    out["Q_days"] = Q
    return out


def main():
    new_rows = []
    for cell_key, cell_name in (("cell1", "sp500_up_50pct_50d_dd25pct"),
                                ("cell3", "sp500_up_20pct_25d_dd10pct")):
        pred_path = REPO / f"results/gbdt/data/_214_{cell_key}_winner_test_predictions.csv"
        if not pred_path.exists():
            print(f"  SKIP {cell_key}: no predictions file at {pred_path}")
            continue
        exp_name = f"{cell_name}_manual_cells13"
        row = row_from_predictions(exp_name, pred_path)
        new_rows.append(row)
        print(f"  {exp_name}: rp@1={row['R_precision_at_1']:.4f}  rp@10={row['R_precision_at_10']:.4f}  AUC={row['AUC']:.4f}")

    if not new_rows:
        print("No predictions found; aborting.")
        return

    # Append rows manually to preserve the 6-decimal precision of existing
    # rows. Using pandas read+write would strip trailing zeros from ALL
    # existing float values, polluting the diff.
    existing_text = CSV_PATH.read_text()
    existing_experiments = {line.split(",", 1)[0] for line in existing_text.splitlines()[1:]}
    rows_to_append = [r for r in new_rows if r["experiment"] not in existing_experiments]
    if not rows_to_append:
        print("\nAll new experiments already present in CSV; nothing to append.")
    else:
        with CSV_PATH.open("a") as f:
            for r in rows_to_append:
                line = ",".join(
                    str(r[c]) if c in ("experiment", "rows", "Q_days")
                    else f"{r[c]:.6f}"
                    for c in COLS
                )
                f.write(line + "\n")
        print(f"\nAppended {len(rows_to_append)} rows; new total: "
              f"{len(existing_text.splitlines()) + len(rows_to_append) - 1} data rows.")

    # Also emit as CSV-format strings for the memo
    print("\n== Canonical CSV row format (for memo / commit) ==")
    for r in new_rows:
        out = ",".join(str(r[c]) if c in ("experiment", "rows", "Q_days")
                       else f"{r[c]:.6f}" for c in COLS)
        print(out)


if __name__ == "__main__":
    main()
