"""_029: produce registry-grade run-dir artifacts for the top-10-AUC K-sweep.

Runs each (cell, K∈{2,3,4,5}) through ``run_backtest_cell`` (champion config: rank/equal/
c=1.0/mean) on the cell's committed ``predictions/test.csv`` — emitting ``summary.json`` +
``equity_curve.csv`` + ``picks.csv`` per run dir under ``results/backtests/_029_k_sweep/
<cell>__k<K>/``. These are what ``regenerate_backtest_performance_csv.py`` (the ``_029`` manifest)
stitches into ``backtest_summary.csv``. The slim analysis numbers come from ``k_sweep_topauc.py``;
this script exists to materialize the full-schema artifacts for the registry. ~40–80 min.

    uv run python -m scripts.backtests.k_sweep_run_artifacts
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/backtests/_029_k_sweep"
KS = [2, 3, 4, 5]
# Pinned top-10-by-AUC cells swept on 2026-06-30 (r_precision_at_k.csv, AUC desc).
CELLS = [
    "nasdaq100_up_40pct_50d_dd20pct_aligned_mixmatch",
    "sp500_up_50pct_25d_dd25pct_daswmacro",
    "sp500_up_50pct_50d_dd25pct_macroreal",
    "sp500_up_50pct_50d_dd25pct_base_v2",
    "sp500_up_40pct_25d_dd20pct_daswbase",
    "sp500_up_50pct_50d_dd25pct_macroproxy",
    "sp500_up_50pct_100d_dd25pct_aligned",
    "sp500_up_50pct_25d_dd25pct_daswbase",
    "sp500_up_50pct_50d_dd25pct_agentloop",
    "sp500_up_20pct_5d_dd10pct_daswmacro",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        for K in KS:
            rd = OUT / f"{cell}__k{K}"
            if (rd / "summary.json").exists():   # idempotent resume — skip completed runs
                print(f"=== skip {cell} K={K} (done)", flush=True)
                continue
            cmd = [sys.executable, "-m", "scripts.backtests.run_backtest_cell",
                   "--cell", f"results/gbdt/experiments/{cell}", "--out", str(rd),
                   "--name", f"029_{cell}_k{K}", "--selection-mode", "rank",
                   "--sizing-mode", "equal", "--k", str(K), "--c", "1.0", "--selection-bound", "mean"]
            print(f">>> {cell} K={K}", flush=True)
            try:
                r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT {cell} K={K} (>900s) — skipping", flush=True)
                continue
            if r.returncode != 0:
                print(f"ERR {cell} K={K}\n{r.stderr[-800:]}", flush=True)
            else:
                tail = r.stdout.strip().splitlines()
                print(tail[-1] if tail else "ok", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
