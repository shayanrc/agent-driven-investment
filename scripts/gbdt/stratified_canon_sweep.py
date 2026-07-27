"""Hyperparameter sweep for group-stratified boosting on canonical periods.

Runs the stratified arm ONLY (no control/baseline — those are already in the
baseline run) across a grid of HP configurations. Evaluates on val+eval;
test is scored for all configs so we can compare fairly against the deployed
champion after the sweep.

The sweep isolates the STRATIFIED arm because that's the one we're trying to
improve — the control and baseline are fixed reference points from the first
run (runs/gbdt/stratified_canon/20/results.json).

Usage:
  uv run python -m scripts.gbdt.stratified_canon_sweep 20

Grid (motivated by the deployed champion's HP d8·ss0.85 and the _284 findings):
  - depth: {6, 8, 10}
  - n_trees × eta: {(800, 0.05), (1200, 0.035), (400, 0.10)}
  - subsample: {1.0, 0.85}
  - family_cap: {2, 3}
  = 36 configs, ~11 min each sequential → too slow.

  Pruned high-prior grid (the configs most likely to beat the baseline):
  - depth 8 (champion's depth) × {ss1.0, ss0.85} × {800/0.05, 1200/0.035}
  - depth 6 × ss0.85 (add row-noise to original recipe)
  - depth 8 × cap3 (wider per-tree family exposure)
  - depth 10 × ss0.85 (push deeper)
  = 7 configs × ~10 min = ~70 min total.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canon_cells import resolve, SPLIT, SNAP, MIN_ROWS_PER_TICKER  # noqa: E402

from gbdt.data import load_panel
from gbdt.targets import build_target

REPO = Path(__file__).resolve().parents[2]
MAXTUNE_CACHE = (REPO / "results/gbdt/experiments"
                 / "sp500_up_20pct_50d_dd10pct_maxtune"
                 / "_feature_matrix_cache.parquet")

SEED = 42
PAIRS = [("dow_sin", "dow_cos"), ("dom_sin", "dom_cos"), ("moy_sin", "moy_cos"),
         ("moq_sin", "moq_cos"), ("qoy_sin", "qoy_cos")]
FLAGS = ["fiscal_year_end_week", "budget_week", "diwali_week", "fomc_week"]
PAIR_COLS = {c for p in PAIRS for c in p}
CAPPED = ["volatility", "returns", "drawdown", "cross-sectional", "persistence",
          "volume", "trend", "vwap"]

# ---- HP grid (pruned to high-prior configs) --------------------------------
GRID = [
    # tag,                depth, n_trees, eta,   ss,   cap
    ("d8",                8,     800,     0.05,  1.0,  2),
    ("d8_ss85",           8,     800,     0.05,  0.85, 2),
    ("d8_1200",           8,     1200,    0.035, 1.0,  2),
    ("d8_ss85_1200",      8,     1200,    0.035, 0.85, 2),
    ("d6_ss85",           6,     800,     0.05,  0.85, 2),
    ("d8_cap3",           8,     800,     0.05,  1.0,  3),
    ("d10_ss85",          10,    800,     0.05,  0.85, 2),
]


def classify(f: str) -> str:
    if f.startswith("fund_"):
        return "F18"
    if f.startswith("vwap_dev"):
        return "vwap"
    if f in PAIR_COLS or f in FLAGS:
        return "calendar"
    if "_xs_" in f:
        return "cross-sectional"
    if "outside" in f:
        return "persistence"
    if any(s in f for s in ("garman_klass", "parkinson", "yang_zhang",
                            "realized_vol", "vol_pct", "volatility", "_vol_",
                            "vol_change", "vol_ret_corr")):
        return "volatility"
    if "drawdown" in f:
        return "drawdown"
    if any(s in f for s in ("volume", "dollar_move", "amihud", "turnover",
                            "illiq", "obv")):
        return "volume"
    if any(s in f for s in ("sma_distance", "ema_", "macd", "sma_",
                            "bollinger", "dist_")):
        return "trend"
    if any(s in f for s in ("stock_return", "rel_strength", "index_return",
                            "_return_", "momentum", "roc_", "reversal",
                            "runup", "beta", "skew", "kurt")):
        return "returns"
    if "zscore" in f:
        return "volatility"
    return "OTHER"


def rpk(index, ytrue, p, ks=(1, 3, 5, 10, 20)):
    t = pd.DataFrame({"date": index.get_level_values("date"),
                      "ticker": index.get_level_values("ticker"),
                      "y": ytrue, "p": p})
    out = {}
    for k in ks:
        ratios = []
        for _, g in t.groupby("date"):
            r_q = int(g["y"].sum())
            if r_q == 0:
                continue
            gg = g.sort_values(["p", "ticker"], ascending=[False, True],
                               kind="mergesort")
            ratios.append(int(gg["y"].head(k).sum()) / min(k, r_q))
        out[k] = float(np.mean(ratios))
    return out


def run_stratified(Xnp, Ynp, DIDX, colpos, groups, feat_cols, segments,
                   depth, n_trees, eta, ss, family_cap, tag):
    """Run one stratified arm and return metrics."""
    eval_segments = [s for s in ("val", "eval", "test") if s in segments]
    rng = np.random.default_rng(SEED)

    def sample_cols():
        cols = []
        for g in CAPPED:
            members = groups.get(g, [])
            cap = min(family_cap, len(members))
            if cap > 0:
                cols += list(rng.choice(members, size=cap, replace=False))
        if "F18" in groups:
            cols += groups["F18"]
        avail_pairs = [i for i, (s, c) in enumerate(PAIRS)
                       if s in colpos and c in colpos]
        if len(avail_pairs) >= 2:
            for pi in rng.choice(avail_pairs, size=2, replace=False):
                cols += list(PAIRS[pi])
        avail_flags = [f for f in FLAGS if f in colpos]
        if len(avail_flags) >= 2:
            cols += list(rng.choice(avail_flags, size=2, replace=False))
        return cols

    p0 = Ynp["train"].mean()
    m0 = float(np.log(p0 / (1 - p0)))
    margins = {s: np.full(len(Ynp[s]), m0, dtype=np.float32) for s in Ynp}
    params = {"max_depth": depth, "eta": eta, "objective": "binary:logistic",
              "tree_method": "hist", "nthread": 8, "seed": SEED,
              "subsample": ss}

    t0 = time.time()
    for t in range(n_trees):
        cols_t = sample_cols()
        idx = [colpos[c] for c in cols_t]
        dtr = xgb.DMatrix(Xnp["train"][:, idx], label=Ynp["train"],
                          base_margin=margins["train"], feature_names=cols_t)
        bst = xgb.train(params, dtr, num_boost_round=1)
        margins["train"] = bst.predict(dtr, output_margin=True)
        for s in eval_segments:
            ds = xgb.DMatrix(Xnp[s][:, idx], base_margin=margins[s],
                             feature_names=cols_t)
            margins[s] = bst.predict(ds, output_margin=True)
        if (t + 1) % 400 == 0:
            print(f"  [{tag}] tree {t+1}/{n_trees} ({time.time()-t0:.0f}s)",
                  flush=True)

    probs = {s: 1 / (1 + np.exp(-margins[s])) for s in eval_segments}
    metrics = {}
    for s, p in probs.items():
        metrics[s] = {"brier": float(brier_score_loss(Ynp[s], p)),
                      "auc": float(roc_auc_score(Ynp[s], p)),
                      **{f"rp{k}": v
                         for k, v in rpk(DIDX[s], Ynp[s], p).items()}}

    elapsed = time.time() - t0
    return metrics, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HP sweep for canonical stratified boosting")
    ap.add_argument("cell", help="canon_cells key (e.g. '20')")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="run only specific configs by tag (default: all)")
    args = ap.parse_args()

    cell = resolve(args.cell)
    t0_global = time.time()

    out_dir = REPO / "runs/gbdt/stratified_canon" / f"{cell['cell']}_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- canonical segments (including test for all configs) ----------------
    segments = {
        "train": (str(SPLIT["train_start"]),
                  str(SPLIT["val_start"] - pd.Timedelta(days=1))),
        "val":   (str(SPLIT["val_start"]),
                  str(SPLIT["eval_start"] - pd.Timedelta(days=1))),
        "eval":  (str(SPLIT["eval_start"]),
                  str(SPLIT["test_start"] - pd.Timedelta(days=1))),
        "test":  (str(SPLIT["test_start"]), str(SPLIT["test_end"])),
    }

    print(f"[cell] {cell['cell']} — {cell['universe']} +{cell['thr']}%/"
          f"{cell['hor']}d dd{cell['dd']}", flush=True)

    # ---- load matrix + target (shared across all configs) ------------------
    if cell["universe"] == "sp500" and MAXTUNE_CACHE.exists():
        print(f"[matrix] reusing sp500 maxtune cache", flush=True)
        X = pd.read_parquet(MAXTUNE_CACHE)
    else:
        raise SystemExit("non-sp500 cells not yet supported in sweep")

    ticker_counts = X.groupby(level="ticker").size()
    keep_tickers = ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index
    n_dropped = len(ticker_counts) - len(keep_tickers)
    X = X.loc[X.index.get_level_values("ticker").isin(keep_tickers)]
    print(f"[debiasing] kept {len(keep_tickers)} tickers, "
          f"dropped {n_dropped}", flush=True)

    panel_result = load_panel(cell["universe"], end=SNAP)
    panel = panel_result.panel
    y = build_target(panel, direction="up", threshold_pct=cell["thr"],
                     horizon_days=cell["hor"],
                     max_drawdown=cell["dd"]).rename("target")
    df = X.join(y, how="left").dropna(subset=["target"])  # NaN-tolerant
    del X, panel
    dates = df.index.get_level_values("date")
    feat_cols = [c for c in df.columns if c != "target"]
    n_feat = len(feat_cols)

    groups: dict[str, list[str]] = {}
    for c in feat_cols:
        groups.setdefault(classify(c), []).append(c)
    unclassified = groups.pop("OTHER", [])
    if unclassified:
        groups.setdefault("returns", []).extend(unclassified)

    print(f"[features] {n_feat} columns", flush=True)

    Xnp, Ynp, DIDX = {}, {}, {}
    for s, (a, b) in segments.items():
        sub = df[(dates >= a) & (dates <= b)]
        Xnp[s] = sub[feat_cols].to_numpy(dtype=np.float32)
        Ynp[s] = sub["target"].to_numpy(dtype=np.int8)
        DIDX[s] = sub.index
        print(f"[seg] {s}: {len(sub):,} rows  prev={Ynp[s].mean():.4f}",
              flush=True)
    del df
    colpos = {c: i for i, c in enumerate(feat_cols)}

    # ---- select configs to run ---------------------------------------------
    configs = GRID
    if args.configs:
        configs = [c for c in GRID if c[0] in args.configs]
        if not configs:
            raise SystemExit(f"no matching configs; available: "
                             f"{[c[0] for c in GRID]}")

    print(f"\n[sweep] running {len(configs)} config(s): "
          f"{[c[0] for c in configs]}\n{'='*70}", flush=True)

    # ---- run sweep ---------------------------------------------------------
    all_results = {}
    for tag, depth, n_trees, eta, ss, cap in configs:
        print(f"\n[{tag}] depth={depth} trees={n_trees} eta={eta} "
              f"ss={ss} cap={cap}", flush=True)
        metrics, elapsed = run_stratified(
            Xnp, Ynp, DIDX, colpos, groups, feat_cols, segments,
            depth, n_trees, eta, ss, cap, tag)
        all_results[tag] = {
            "hp": {"depth": depth, "n_trees": n_trees, "eta": eta,
                   "subsample": ss, "family_cap": cap},
            "metrics": metrics,
            "elapsed_s": elapsed,
        }
        # Print per-config summary
        for seg in ("val", "eval", "test"):
            m = metrics[seg]
            print(f"  {seg}: AUC={m['auc']:.4f}  R-p@1={m['rp1']:.3f}  "
                  f"@3={m['rp3']:.3f}  @5={m['rp5']:.3f}  "
                  f"@10={m['rp10']:.3f}  @20={m.get('rp20',0):.3f}",
                  flush=True)

    # ---- save full results -------------------------------------------------
    sweep_results = {
        "cell": cell["cell"],
        "cell_info": {k: (str(v) if isinstance(v, (pd.Timestamp,)) else v)
                      for k, v in cell.items()},
        "n_features": n_feat,
        "segments": segments,
        "canonical_split": {k: str(v) for k, v in SPLIT.items()},
        "min_rows_per_ticker": MIN_ROWS_PER_TICKER,
        "prevalence": {s: float(Ynp[s].mean()) for s in Ynp},
        "configs": all_results,
    }
    with open(out_dir / "sweep_results.json", "w") as fh:
        json.dump(sweep_results, fh, indent=1)

    # ---- print comparison table --------------------------------------------
    print(f"\n{'='*70}")
    print(f"{'COMPARISON TABLE':^70}")
    print(f"{'='*70}")
    header = (f"{'config':<20} {'seg':>5} {'AUC':>6} {'R-p@1':>6} "
              f"{'R-p@3':>6} {'R-p@5':>6} {'R-p@10':>7} {'R-p@20':>7}")
    print(header)
    print("-" * len(header))

    # Include the original d6 baseline from the first run for reference
    print(f"{'[orig d6 baseline]':<20} {'test':>5} {'0.807':>6} "
          f"{'0.281':>6} {'0.264':>6} {'0.252':>6} {'0.250':>7} "
          f"{'0.308':>7}")
    print(f"{'[deployed champ]':<20} {'test':>5} {'0.823':>6} "
          f"{'0.321':>6} {'0.313':>6} {'0.311':>6} {'0.299':>7} "
          f"{'0.330':>7}")
    print("-" * len(header))

    for tag, _, _, _, _, _ in configs:
        if tag not in all_results:
            continue
        for seg in ("val", "eval", "test"):
            m = all_results[tag]["metrics"][seg]
            print(f"{tag:<20} {seg:>5} {m['auc']:>6.4f} {m['rp1']:>6.3f} "
                  f"{m['rp3']:>6.3f} {m['rp5']:>6.3f} {m['rp10']:>7.3f} "
                  f"{m.get('rp20',0):>7.3f}")

    total_time = time.time() - t0_global
    print(f"\n[done] {len(configs)} configs in {total_time:.0f}s "
          f"({total_time/60:.1f} min)", flush=True)
    print(f"Results: {out_dir / 'sweep_results.json'}", flush=True)


if __name__ == "__main__":
    main()
