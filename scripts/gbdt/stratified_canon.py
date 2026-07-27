"""Canonical-periods group-stratified boosting experiment.

Adapts the _284 recipe (mixed-family trees: <=2 per lookback-ladder family,
F18 uncapped, calendar pairs intact) to the 2015-anchored canonical evaluation
periods so the stratified models can be compared head-to-head against the
deployed canon_ft champions on identical date windows.

Three arms per cell (same as _284):
  A) stratified  — custom XGBoost boosting with per-tree structured feature masks
  B) control     — same budget, random ~35-col subsample (no structure)
  C) baseline    — iter-0 default HP (100 trees, depth 6, eta 0.3)

Usage:
  uv run python -m scripts.gbdt.stratified_canon CELL [--n-trees N] [--test]

  CELL: a canon_cells key (e.g. '20' for sp500 +20%/25d, 'sp500_40_200_f18' for
        the F18 cell, '50' for sp500 +50%/50d, etc.)

  --test: also score the blind test window and include in results.json
          (default: val+eval only, test sealed)
  --n-trees: override tree budget (default 800)

The script builds the maximal 310-column feature matrix from the sp500
maxtune cache (covers 1990→2026, all technical + F18 + F20 + F21). If the
cell uses a smaller universe or different feature token, the matrix is built
fresh from load_panel + build_feature_matrix (with fund_df if needed).

Artifacts (gitignored, under runs/gbdt/stratified_canon/<cell>/):
  results.json     — metrics for all arms on val, eval, (optionally test)
  model.pkl        — serialized ensemble (boosters + cols_per_tree + m0)
  artifacts.pkl    — per-tree anatomy records for analyze_stratified_trees.py
  predictions/     — {val,eval,test}.csv with (date,ticker,p_calibrated,y_true)
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

# ---- project imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canon_cells import resolve, SPLIT, SNAP, MIN_ROWS_PER_TICKER  # noqa: E402

from gbdt.data import load_panel
from gbdt.features import build_feature_matrix
from gbdt.targets import build_target

REPO = Path(__file__).resolve().parents[2]
MAXTUNE_CACHE = (REPO / "results/gbdt/experiments"
                 / "sp500_up_20pct_50d_dd10pct_maxtune"
                 / "_feature_matrix_cache.parquet")

# ---- _284 recipe constants (memo-exact) ------------------------------------
SEED, N_TREES_DEFAULT, ETA, DEPTH = 42, 800, 0.05, 6

PAIRS = [("dow_sin", "dow_cos"), ("dom_sin", "dom_cos"), ("moy_sin", "moy_cos"),
         ("moq_sin", "moq_cos"), ("qoy_sin", "qoy_cos")]
FLAGS = ["fiscal_year_end_week", "budget_week", "diwali_week", "fomc_week"]
PAIR_COLS = {c for p in PAIRS for c in p}
CAPPED = ["volatility", "returns", "drawdown", "cross-sectional", "persistence",
          "volume", "trend", "vwap"]


def classify(f: str) -> str:
    """Semantic feature-family classifier (name-based, memo _284 grouping)."""
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
        out[k] = float(np.mean(ratios))
    return out


def _build_feature_matrix_fresh(cell_info: dict) -> pd.DataFrame:
    """Build the maximal 310-col feature matrix from scratch for a cell.

    Used when the sp500 maxtune cache doesn't match the cell's universe.
    """
    universe = cell_info["universe"]
    token = "all_fundamentals_vwap_calendar2"
    print(f"[build] building fresh {token} matrix for {universe}...",
          flush=True)
    panel_result = load_panel(universe, end=SNAP)
    panel = panel_result.panel
    index_df = panel_result.index_df

    # Check if we need fundamentals
    fund_df = None
    if "fundamentals" in token:
        try:
            from valuation.panel import load_valuation_panel
            fund_df = load_valuation_panel()
            print(f"[build] loaded valuation panel: {fund_df.shape}", flush=True)
        except Exception as e:
            print(f"[warn] could not load valuation panel: {e}", flush=True)

    X = build_feature_matrix(panel, index_df, families=token, fund_df=fund_df)
    print(f"[build] matrix: {X.shape}", flush=True)
    return X


def _load_or_build_matrix(cell_info: dict) -> pd.DataFrame:
    """Load the maximal feature matrix, reusing the sp500 maxtune cache if
    the cell is sp500, otherwise building fresh."""
    if cell_info["universe"] == "sp500" and MAXTUNE_CACHE.exists():
        print(f"[matrix] reusing sp500 maxtune cache: {MAXTUNE_CACHE}",
              flush=True)
        return pd.read_parquet(MAXTUNE_CACHE)
    return _build_feature_matrix_fresh(cell_info)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Canonical-periods group-stratified boosting experiment")
    ap.add_argument("cell", help="canon_cells key (e.g. '20', '50', "
                    "'sp500_40_200_f18')")
    ap.add_argument("--n-trees", type=int, default=N_TREES_DEFAULT,
                    help=f"tree budget (default {N_TREES_DEFAULT})")
    ap.add_argument("--test", action="store_true",
                    help="also score the blind test window")
    args = ap.parse_args()

    cell = resolve(args.cell)
    n_trees = args.n_trees
    t0 = time.time()

    out_dir = REPO / "runs/gbdt/stratified_canon" / cell["cell"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # ---- canonical segments ------------------------------------------------
    segments = {
        "train": (str(SPLIT["train_start"]), str(SPLIT["val_start"] - pd.Timedelta(days=1))),
        "val":   (str(SPLIT["val_start"]),   str(SPLIT["eval_start"] - pd.Timedelta(days=1))),
        "eval":  (str(SPLIT["eval_start"]),  str(SPLIT["test_start"] - pd.Timedelta(days=1))),
    }
    if args.test:
        segments["test"] = (str(SPLIT["test_start"]), str(SPLIT["test_end"]))

    print(f"[cell] {cell['cell']} — {cell['universe']} +{cell['thr']}%/"
          f"{cell['hor']}d dd{cell['dd']}", flush=True)
    print(f"[segments] {json.dumps(segments, indent=2)}", flush=True)

    # ---- load matrix + target ----------------------------------------------
    X = _load_or_build_matrix(cell)

    # Apply MIN_ROWS_PER_TICKER de-biasing gate (keeps ~2015-present tickers)
    ticker_counts = X.groupby(level="ticker").size()
    keep_tickers = ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index
    n_dropped = len(ticker_counts) - len(keep_tickers)
    X = X.loc[X.index.get_level_values("ticker").isin(keep_tickers)]
    print(f"[debiasing] MIN_ROWS_PER_TICKER={MIN_ROWS_PER_TICKER}: "
          f"kept {len(keep_tickers)} tickers, dropped {n_dropped}", flush=True)

    panel_result = load_panel(cell["universe"], end=SNAP)
    panel = panel_result.panel
    y = build_target(panel, direction="up", threshold_pct=cell["thr"],
                     horizon_days=cell["hor"],
                     max_drawdown=cell["dd"]).rename("target")
    df = X.join(y, how="left").dropna(subset=["target"])  # NaN-tolerant: only drop label-NaN rows, keep feature-NaN (XGBoost handles natively)
    del X, panel
    dates = df.index.get_level_values("date")
    feat_cols = [c for c in df.columns if c != "target"]
    n_feat = len(feat_cols)

    # ---- classify features into families -----------------------------------
    groups: dict[str, list[str]] = {}
    for c in feat_cols:
        groups.setdefault(classify(c), []).append(c)
    unclassified = groups.pop("OTHER", [])
    if unclassified:
        print(f"[warn] unclassified features (grouped as 'returns'): "
              f"{unclassified}", flush=True)
        groups.setdefault("returns", []).extend(unclassified)

    # Check all CAPPED families exist in feature pool
    for fam in CAPPED:
        if fam not in groups or len(groups[fam]) < 2:
            print(f"[warn] family '{fam}' has {len(groups.get(fam, []))} "
                  f"features (need >=2 for capped draw); padding or skipping",
                  flush=True)

    print(f"[features] {n_feat} columns, families: "
          f"{', '.join(f'{k}:{len(v)}' for k, v in sorted(groups.items()))}",
          flush=True)

    # ---- segment slicing ---------------------------------------------------
    Xnp, Ynp, DIDX = {}, {}, {}
    for s, (a, b) in segments.items():
        sub = df[(dates >= a) & (dates <= b)]
        Xnp[s] = sub[feat_cols].to_numpy(dtype=np.float32)
        Ynp[s] = sub["target"].to_numpy(dtype=np.int8)
        DIDX[s] = sub.index
        print(f"[seg] {s}: {len(sub):,} rows  prevalence {Ynp[s].mean():.4f}"
              f"  [{a}..{b}]", flush=True)
    del df
    colpos = {c: i for i, c in enumerate(feat_cols)}

    # ---- scoring helper ----------------------------------------------------
    def score(probs: dict[str, np.ndarray]):
        out = {}
        for s, p in probs.items():
            out[s] = {"brier": float(brier_score_loss(Ynp[s], p)),
                      "auc": float(roc_auc_score(Ynp[s], p)),
                      **{f"rp{k}": v
                         for k, v in rpk(DIDX[s], Ynp[s], p).items()}}
        return out

    def save_preds(probs: dict[str, np.ndarray], arm_name: str):
        for s, p in probs.items():
            pdf = pd.DataFrame({
                "date": DIDX[s].get_level_values("date"),
                "ticker": DIDX[s].get_level_values("ticker"),
                "p_calibrated": p,
                "y_true": Ynp[s],
            })
            pdf.to_csv(pred_dir / f"{arm_name}_{s}.csv", index=False)

    eval_segments = [s for s in ("val", "eval", "test") if s in segments]

    results = {"meta": {
        "cell": cell["cell"],
        "cell_info": {k: (str(v) if isinstance(v, (pd.Timestamp,)) else v)
                      for k, v in cell.items()},
        "seed": SEED, "n_trees": n_trees, "eta": ETA, "depth": DEPTH,
        "n_features": n_feat,
        "segments": segments,
        "canonical_split": {k: str(v) for k, v in SPLIT.items()},
        "min_rows_per_ticker": MIN_ROWS_PER_TICKER,
        "prevalence": {s: float(Ynp[s].mean()) for s in Ynp},
    }}

    # ==== A) STRATIFIED =====================================================
    print(f"\n{'='*60}\n[A] STRATIFIED ({n_trees} trees, eta={ETA}, "
          f"depth={DEPTH})\n{'='*60}", flush=True)
    rng = np.random.default_rng(SEED)

    def sample_cols():
        cols = []
        for g in CAPPED:
            members = groups.get(g, [])
            if len(members) >= 2:
                cols += list(rng.choice(members, size=2, replace=False))
            elif len(members) == 1:
                cols += members  # only 1 available
        # F18 fundamentals: uncapped — all offered to every tree
        if "F18" in groups:
            cols += groups["F18"]
        # Calendar pairs: 2 of 5 intact pairs + 2 of 4 flags
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
    params = {"max_depth": DEPTH, "eta": ETA, "objective": "binary:logistic",
              "tree_method": "hist", "nthread": 8, "seed": SEED}
    records, boosters = [], []
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
        # per-tree anatomy
        tdf = bst.trees_to_dataframe()
        id2 = {r.ID: (r.Feature, r.Yes, r.No) for r in tdf.itertuples()}
        path_pairs = set()
        stack = [("0-0", [])]
        while stack:
            nid, acc = stack.pop()
            f, yes, no = id2[nid]
            if f == "Leaf":
                path_pairs.update(combinations(sorted(set(acc)), 2))
                continue
            stack.append((yes, acc + [f]))
            stack.append((no, acc + [f]))
        records.append({
            "cols": cols_t,
            "total_gain": bst.get_score(importance_type="total_gain"),
            "weight": bst.get_score(importance_type="weight"),
            "path_pairs": sorted(path_pairs),
        })
        boosters.append(bst)
        if (t + 1) % 200 == 0:
            print(f"[strat] tree {t+1}/{n_trees} ({time.time()-t0:.0f}s)",
                  flush=True)

    strat_probs = {s: 1 / (1 + np.exp(-margins[s])) for s in eval_segments}
    results["stratified"] = score(strat_probs)
    save_preds(strat_probs, "stratified")
    print(f"[strat] done ({time.time()-t0:.0f}s)", flush=True)
    for s in eval_segments:
        r = results["stratified"][s]
        print(f"  {s}: AUC={r['auc']:.4f}  Brier={r['brier']:.5f}  "
              f"R-p@1={r['rp1']:.3f}  @3={r['rp3']:.3f}  @5={r['rp5']:.3f}  "
              f"@10={r['rp10']:.3f}", flush=True)

    with open(out_dir / "artifacts.pkl", "wb") as fh:
        pickle.dump({"records": records, "feat_cols": feat_cols,
                     "groups": groups, "m0": m0, "params": params}, fh)
    with open(out_dir / "model.pkl", "wb") as fh:
        pickle.dump({"boosters": boosters,
                     "cols_per_tree": [r["cols"] for r in records],
                     "m0": m0}, fh)

    # ==== B) UNSTRUCTURED CONTROL ===========================================
    print(f"\n{'='*60}\n[B] CONTROL (random ~35/{n_feat} cols)\n{'='*60}",
          flush=True)
    m_b = xgb.XGBClassifier(
        n_estimators=n_trees, max_depth=DEPTH, learning_rate=ETA,
        colsample_bytree=35 / n_feat, tree_method="hist",
        random_state=SEED, n_jobs=8, eval_metric="logloss")
    m_b.fit(Xnp["train"], Ynp["train"])
    ctrl_probs = {s: m_b.predict_proba(Xnp[s])[:, 1] for s in eval_segments}
    results["control_rand35"] = score(ctrl_probs)
    save_preds(ctrl_probs, "control")
    print(f"[control] done ({time.time()-t0:.0f}s)", flush=True)
    for s in eval_segments:
        r = results["control_rand35"][s]
        print(f"  {s}: AUC={r['auc']:.4f}  Brier={r['brier']:.5f}  "
              f"R-p@1={r['rp1']:.3f}  @3={r['rp3']:.3f}  @5={r['rp5']:.3f}  "
              f"@10={r['rp10']:.3f}", flush=True)

    # ==== C) BASELINE (iter-0 default HP) ====================================
    print(f"\n{'='*60}\n[C] BASELINE (iter-0: 100 trees, depth 6, "
          f"eta 0.3)\n{'='*60}", flush=True)
    m_c = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.3, tree_method="hist",
        random_state=SEED, n_jobs=8, eval_metric="logloss")
    m_c.fit(Xnp["train"], Ynp["train"])
    base_probs = {s: m_c.predict_proba(Xnp[s])[:, 1] for s in eval_segments}
    results["baseline_iter0"] = score(base_probs)
    save_preds(base_probs, "baseline")
    print(f"[baseline] done ({time.time()-t0:.0f}s)", flush=True)
    for s in eval_segments:
        r = results["baseline_iter0"][s]
        print(f"  {s}: AUC={r['auc']:.4f}  Brier={r['brier']:.5f}  "
              f"R-p@1={r['rp1']:.3f}  @3={r['rp3']:.3f}  @5={r['rp5']:.3f}  "
              f"@10={r['rp10']:.3f}", flush=True)

    # ---- save results & summary --------------------------------------------
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1)

    print(f"\n{'='*60}\n[SUMMARY]\n{'='*60}")
    for arm in ("stratified", "control_rand35", "baseline_iter0"):
        print(f"\n{arm}:")
        for s in eval_segments:
            r = results[arm][s]
            print(f"  {s}: AUC={r['auc']:.4f}  Brier={r['brier']:.5f}  "
                  f"R-p@1={r['rp1']:.3f}  @3={r['rp3']:.3f}  "
                  f"@5={r['rp5']:.3f}  @10={r['rp10']:.3f}  "
                  f"@20={r.get('rp20', 0):.3f}")
    print(f"\n[done] artifacts in {out_dir} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
