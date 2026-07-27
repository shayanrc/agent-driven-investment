"""Group-stratified boosting with Val-AUC early stopping on canonical periods.

Trains a group-stratified ensemble on the canonical periods (2015-anchored split),
tracking Val AUC at each tree step. Outputs metrics for BOTH:
  1. Full Ensemble (at max n_trees)
  2. Early-Stopped Ensemble (at best_t = argmax(val_auc))

Supports any cell in canon_cells.py, including:
  - '20'               : sp500 +20%/25d (the deployed technical champion)
  - 'sp500_40_200_f18' : sp500 +40%/200d F18 (the deployed fundamentals champion)

Usage:
  uv run python -m scripts.gbdt.stratified_canon_earlystop CELL [--depth D] [--subsample SS] [--n-trees N] [--eta ETA] [--family-cap CAP]

Examples:
  uv run python -m scripts.gbdt.stratified_canon_earlystop sp500_40_200_f18 --depth 6 --subsample 0.85
  uv run python -m scripts.gbdt.stratified_canon_earlystop 20 --depth 6 --subsample 0.85
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canon_cells import resolve, SPLIT, SNAP, MIN_ROWS_PER_TICKER  # noqa: E402

from gbdt.data import load_panel
from gbdt.features import build_feature_matrix
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


def classify(f: str) -> str:
    """Semantic feature-family classifier."""
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
    """R-Precision@K, per-day fixed K, macro-averaged (project-canonical)."""
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
        out[k] = float(np.mean(ratios)) if len(ratios) > 0 else 0.0
    return out


def _load_or_build_matrix(cell_info: dict) -> pd.DataFrame:
    """Load the maximal feature matrix, reusing sp500 maxtune cache if sp500."""
    if cell_info["universe"] == "sp500" and MAXTUNE_CACHE.exists():
        print(f"[matrix] reusing sp500 maxtune cache: {MAXTUNE_CACHE}", flush=True)
        return pd.read_parquet(MAXTUNE_CACHE)
    
    universe = cell_info["universe"]
    token = "all_fundamentals_vwap_calendar2"
    print(f"[matrix] building fresh {token} matrix for {universe}...", flush=True)
    panel_result = load_panel(universe, end=SNAP)
    panel = panel_result.panel
    index_df = panel_result.index_df

    fund_df = None
    if "fundamentals" in token:
        try:
            from valuation.panel import load_valuation_panel
            fund_df = load_valuation_panel()
            print(f"[matrix] loaded valuation panel: {fund_df.shape}", flush=True)
        except Exception as e:
            print(f"[warn] could not load valuation panel: {e}", flush=True)

    X = build_feature_matrix(panel, index_df, families=token, fund_df=fund_df)
    return X


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Group-stratified boosting with Val-AUC early stopping")
    ap.add_argument("cell", help="canon_cells key (e.g. '20' or 'sp500_40_200_f18')")
    ap.add_argument("--depth", type=int, default=6, help="tree depth (default 6)")
    ap.add_argument("--subsample", type=float, default=0.85, help="subsample (default 0.85)")
    ap.add_argument("--n-trees", type=int, default=800, help="max tree budget (default 800)")
    ap.add_argument("--eta", type=float, default=0.05, help="learning rate (default 0.05)")
    ap.add_argument("--family-cap", type=int, default=2, help="max cols per capped family (default 2)")
    args = ap.parse_args()

    cell = resolve(args.cell)
    t0 = time.time()

    out_dir = REPO / "runs/gbdt/stratified_canon" / f"{cell['cell']}_earlystop"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    segments = {
        "train": (str(SPLIT["train_start"]), str(SPLIT["val_start"] - pd.Timedelta(days=1))),
        "val":   (str(SPLIT["val_start"]),   str(SPLIT["eval_start"] - pd.Timedelta(days=1))),
        "eval":  (str(SPLIT["eval_start"]),  str(SPLIT["test_start"] - pd.Timedelta(days=1))),
        "test":  (str(SPLIT["test_start"]),  str(SPLIT["test_end"])),
    }

    print(f"============================================================", flush=True)
    print(f"[EARLY-STOP STRATIFIED RUN] {cell['cell']}", flush=True)
    print(f"  Universe: {cell['universe']} | Target: +{cell['thr']}%/{cell['hor']}d dd{cell['dd']}", flush=True)
    print(f"  HP: depth={args.depth}, subsample={args.subsample}, n_trees={args.n_trees}, eta={args.eta}, family_cap={args.family_cap}", flush=True)
    print(f"============================================================", flush=True)

    # ---- load feature matrix & target --------------------------------------
    X = _load_or_build_matrix(cell)

    # Apply de-biasing gate
    ticker_counts = X.groupby(level="ticker").size()
    keep_tickers = ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index
    n_dropped = len(ticker_counts) - len(keep_tickers)
    X = X.loc[X.index.get_level_values("ticker").isin(keep_tickers)]
    print(f"[debiasing] MIN_ROWS_PER_TICKER={MIN_ROWS_PER_TICKER}: kept {len(keep_tickers)} tickers, dropped {n_dropped}", flush=True)

    panel_result = load_panel(cell["universe"], end=SNAP)
    panel = panel_result.panel
    y = build_target(panel, direction="up", threshold_pct=cell["thr"],
                     horizon_days=cell["hor"], max_drawdown=cell["dd"]).rename("target")

    # NaN-tolerant join: drop target-NaN only (XGBoost handles feature-NaN)
    df = X.join(y, how="left").dropna(subset=["target"])
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

    colpos = {c: i for i, c in enumerate(feat_cols)}

    Xnp, Ynp, DIDX = {}, {}, {}
    for s, (a, b) in segments.items():
        sub = df[(dates >= a) & (dates <= b)]
        Xnp[s] = sub[feat_cols].to_numpy(dtype=np.float32)
        Ynp[s] = sub["target"].to_numpy(dtype=np.int8)
        DIDX[s] = sub.index
        print(f"[seg] {s}: {len(sub):,} rows | prevalence {Ynp[s].mean():.4f} | [{a}..{b}]", flush=True)
    del df

    # ---- stratified loop with per-tree Val-AUC tracking --------------------
    rng = np.random.default_rng(SEED)

    def sample_cols():
        cols = []
        for g in CAPPED:
            members = groups.get(g, [])
            cap = min(args.family_cap, len(members))
            if cap > 0:
                cols += list(rng.choice(members, size=cap, replace=False))
        if "F18" in groups:
            cols += groups["F18"]
        avail_pairs = [i for i, (s, c) in enumerate(PAIRS) if s in colpos and c in colpos]
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
    params = {"max_depth": args.depth, "eta": args.eta, "objective": "binary:logistic",
              "tree_method": "hist", "nthread": 8, "seed": SEED, "subsample": args.subsample}

    val_auc_history = []
    best_val_auc = -1.0
    best_t = 0
    best_margins = None

    eval_segments = ["val", "eval", "test"]

    print(f"\n[training] running {args.n_trees} trees with per-step Val-AUC tracking...", flush=True)
    for t in range(args.n_trees):
        cols_t = sample_cols()
        idx = [colpos[c] for c in cols_t]
        dtr = xgb.DMatrix(Xnp["train"][:, idx], label=Ynp["train"],
                          base_margin=margins["train"], feature_names=cols_t)
        bst = xgb.train(params, dtr, num_boost_round=1)
        margins["train"] = bst.predict(dtr, output_margin=True)

        for s in eval_segments:
            ds = xgb.DMatrix(Xnp[s][:, idx], base_margin=margins[s], feature_names=cols_t)
            margins[s] = bst.predict(ds, output_margin=True)

        # Val AUC evaluation
        p_val = 1 / (1 + np.exp(-margins["val"]))
        val_auc_t = float(roc_auc_score(Ynp["val"], p_val))
        val_auc_history.append(val_auc_t)

        if val_auc_t > best_val_auc:
            best_val_auc = val_auc_t
            best_t = t + 1
            best_margins = {s: np.copy(margins[s]) for s in eval_segments}

        if (t + 1) % 100 == 0 or (t + 1) == args.n_trees:
            print(f"  step {t+1:4d}/{args.n_trees} | Val AUC: {val_auc_t:.4f} (best: {best_val_auc:.4f} @ step {best_t})", flush=True)

    print(f"\n[early-stopping] best Val AUC = {best_val_auc:.4f} reached at step {best_t} / {args.n_trees}", flush=True)

    # ---- score both Full Ensemble and Early-Stopped Ensemble --------------
    def calc_metrics(margin_dict):
        res = {}
        for s in eval_segments:
            p = 1 / (1 + np.exp(-margin_dict[s]))
            res[s] = {
                "brier": float(brier_score_loss(Ynp[s], p)),
                "auc": float(roc_auc_score(Ynp[s], p)),
                **{f"rp{k}": v for k, v in rpk(DIDX[s], Ynp[s], p).items()}
            }
        return res

    full_metrics = calc_metrics(margins)
    best_metrics = calc_metrics(best_margins)

    # Save early-stopped predictions
    for s in eval_segments:
        p_best = 1 / (1 + np.exp(-best_margins[s]))
        pdf = pd.DataFrame({
            "date": DIDX[s].get_level_values("date"),
            "ticker": DIDX[s].get_level_values("ticker"),
            "p_calibrated": p_best,
            "y_true": Ynp[s],
        })
        pdf.to_csv(pred_dir / f"stratified_earlystop_{s}.csv", index=False)

    summary = {
        "cell": cell["cell"],
        "target": f"{cell['universe']} +{cell['thr']}%/{cell['hor']}d dd{cell['dd']}",
        "hp": {"depth": args.depth, "subsample": args.subsample, "n_trees": args.n_trees, "eta": args.eta, "family_cap": args.family_cap},
        "early_stopping": {"best_t": best_t, "best_val_auc": best_val_auc},
        "early_stopped_metrics": best_metrics,
        "full_ensemble_metrics": full_metrics,
        "val_auc_history": val_auc_history,
    }

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*75}", flush=True)
    print(f"RESULTS COMPARISON — {cell['cell']} ({cell['universe']} +{cell['thr']}%/{cell['hor']}d)", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"{'Ensemble Mode':<25} {'Segment':>7} {'AUC':>7} {'R-p@1':>7} {'R-p@3':>7} {'R-p@5':>7} {'R-p@10':>7}", flush=True)
    print("-" * 75, flush=True)

    for mode_name, mdict in [("Early-Stopped (t=" + str(best_t) + ")", best_metrics), ("Full Ensemble (t=" + str(args.n_trees) + ")", full_metrics)]:
        for s in ("val", "eval", "test"):
            m = mdict[s]
            print(f"{mode_name:<25} {s:>7} {m['auc']:>7.4f} {m['rp1']:>7.3f} {m['rp3']:>7.3f} {m['rp5']:>7.3f} {m['rp10']:>7.3f}", flush=True)

    print(f"\n[done] summary saved to {out_dir / 'summary.json'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
