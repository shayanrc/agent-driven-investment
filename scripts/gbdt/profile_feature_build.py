"""Phase-0 profiler for the gbdt feature build (GBDTPERF / V1.4_TBD row 3).

Loads the same ~7y trailing slice the daily incremental inference path uses and
cProfiles ``build_feature_matrix`` so we optimize the *measured* hot spots
(vectorize vs parallelize) rather than the assumed ones. Read-only; no writes.

    uv run python -m scripts.gbdt.profile_feature_build \
        --cell results/gbdt/experiments/sp500_up_50pct_50d_dd25pct_agentloop \
        --since 2026-06-15
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import pandas as pd
import yaml

from gbdt import features as gbdt_features
from gbdt.data import load_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="results/gbdt/experiments/sp500_up_50pct_50d_dd25pct_agentloop")
    ap.add_argument("--since", default="2026-06-15",
                    help="trailing-slice anchor (warmup_start = since - 2700d), mirrors --since")
    ap.add_argument("--end", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    cell = Path(args.cell)
    universe = yaml.safe_load((cell / "spec.yaml").read_text())["target"]["universe"]
    warmup_start = str((pd.Timestamp(args.since) - pd.Timedelta(days=2700)).date())

    print(f"[profile] universe={universe} slice=[{warmup_start} .. {args.end}] (the --since path)")
    t0 = time.time()
    panel_obj = load_panel(universe, start=warmup_start, end=args.end, cache_only=True)
    panel = panel_obj.panel
    n_tk = panel.index.get_level_values("ticker").nunique()
    n_dt = panel.index.get_level_values("date").nunique()
    print(f"[profile] panel loaded in {time.time()-t0:.1f}s: {len(panel):,} rows "
          f"({n_tk} tickers × {n_dt} dates)")

    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    X = gbdt_features.build_feature_matrix(
        panel, panel_obj.index_series, annualization=panel_obj.annualization_factor,
    )
    prof.disable()
    wall = time.time() - t0
    print(f"[profile] build_feature_matrix: {wall:.1f}s wall (under cProfile — "
          f"real wall is faster), {X.shape[1]} cols × {len(X):,} rows\n")

    def dump(sort_key: str, label: str, top: int) -> None:
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats(sort_key)
        ps.print_stats(top)
        print(f"================= TOP {top} BY {label} =================")
        # Keep the header + rows; drop pstats' noisy preamble.
        for line in s.getvalue().splitlines():
            if line.strip() and ("function calls" in line or "ncalls" in line
                                 or "/" in line or "{" in line or ":" in line):
                print(line)
        print()

    dump("tottime", "SELF-TIME (the actual hot spots to vectorize/parallelize)", args.top)
    dump("cumulative", "CUMULATIVE (which families/ops dominate)", args.top)


if __name__ == "__main__":
    main()
