"""Group-stratified boosting experiment (memo _284).

Trains three arms on a cell's cached maximal-pool feature matrix and scores
them on the date-aligned val + eval segments (test stays blind until an
explicit one-shot commit):

  A) stratified — custom boosting loop where every tree trains on its own
     structured column subset:
       - 8 lookback-ladder families capped at <=2 distinct features per tree
         (volatility, returns, drawdown, cross-sectional, persistence,
         volume, trend, vwap)
       - F18 fundamentals: uncapped, all 13 columns offered to every tree
       - calendar sin/cos pairs (dow, dom, moy, moq, qoy) sampled INTACT
         (2 of 5 pairs per tree) + 2 of 4 event flags
     ~35 cols/tree, additive continuation via base_margin (mathematically
     standard boosting with a per-tree feature mask).
  B) control — same budget (same n_trees/eta/depth, ~35 random cols/tree via
     colsample_bytree) but NO structure. Isolates "structure helps" from
     "column subsampling helps".
  C) baseline — the cell's iter-0 default HP (100 trees, depth 6, eta 0.3),
     refit in the same harness so all three arms share one code path.

Motivation: rule 14 showed slow-eta smoothing on the full pool anti-selects
the top-of-book by spreading splits into correlated same-family ladders.
Stratification blocks the pile-up structurally, so the smoothing benefits
arrive without the anti-selection (user-driven design, 2026-07-08 session).

Usage:
  uv run python -m scripts.gbdt.stratified_boosting          # defaults below

Artifacts (OUT_DIR, gitignored):
  results.json          — val/eval brier/AUC/R-p@{1,3,5,10} for the 3 arms
  artifacts.pkl         — per-tree records: cols offered, total_gain, weight,
                          same-path feature pairs (for analyze_stratified_trees)
  model.pkl             — pickled per-tree boosters + col lists + base margin
                          (lets a later one-shot test scoring reuse the EXACT
                          ensemble, no retrain drift)
"""
import json
import pickle
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

from gbdt.data import load_panel
from gbdt.targets import build_target

# ---- experiment constants (memo _284 exact recipe) -------------------------
CELL = "sp500_up_20pct_50d_dd10pct_maxtune"
RUN_DIR = Path("results/gbdt/experiments") / CELL
OUT_DIR = Path("runs/gbdt/stratified") / CELL
UNIVERSE, THRESH, HORIZON, MAXDD = "sp500", 20, 50, 0.10
SNAPSHOT_END = "2026-07-06"
SEED, N_TREES, ETA, DEPTH = 42, 800, 0.05, 6
# date_aligned segments (train_start 2019-01-01, NYSE, 800/400/200 rows)
SEGMENTS = {
    "train": ("2019-01-02", "2022-03-04"),
    "val":   ("2022-03-07", "2023-10-06"),
    "eval":  ("2023-10-09", "2024-07-25"),
    # test 2024-07-26..2024-12-16 — BLIND, scored only via an explicit one-shot
}

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


def rpk(index, ytrue, p, ks=(1, 3, 5, 10)):
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


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = pd.read_parquet(RUN_DIR / "_feature_matrix_cache.parquet")
    panel = load_panel(UNIVERSE, end=SNAPSHOT_END).panel
    y = build_target(panel, direction="up", threshold_pct=THRESH,
                     horizon_days=HORIZON, max_drawdown=MAXDD).rename("target")
    df = X.join(y, how="left").dropna()
    del X, panel
    dates = df.index.get_level_values("date")
    feat_cols = [c for c in df.columns if c != "target"]
    groups: dict[str, list[str]] = {}
    for c in feat_cols:
        groups.setdefault(classify(c), []).append(c)
    assert not groups.get("OTHER"), f"unclassified: {groups.get('OTHER')}"

    Xnp, Ynp, DIDX = {}, {}, {}
    for s, (a, b) in SEGMENTS.items():
        sub = df[(dates >= a) & (dates <= b)]
        Xnp[s] = sub[feat_cols].to_numpy(dtype=np.float32)
        Ynp[s] = sub["target"].to_numpy(dtype=np.int8)
        DIDX[s] = sub.index
        print(f"[seg] {s}: {len(sub):,} rows prevalence {Ynp[s].mean():.4f}",
              flush=True)
    del df
    colpos = {c: i for i, c in enumerate(feat_cols)}

    def score(pv, pe):
        out = {}
        for s, p in [("val", pv), ("eval", pe)]:
            out[s] = {"brier": float(brier_score_loss(Ynp[s], p)),
                      "auc": float(roc_auc_score(Ynp[s], p)),
                      **{f"rp{k}": v
                         for k, v in rpk(DIDX[s], Ynp[s], p).items()}}
        return out

    results = {"meta": {
        "cell": CELL, "seed": SEED, "n_trees": N_TREES, "eta": ETA,
        "depth": DEPTH, "segments": SEGMENTS,
        "prevalence": {s: float(Ynp[s].mean()) for s in Ynp},
    }}

    # ---- A) stratified ------------------------------------------------------
    rng = np.random.default_rng(SEED)

    def sample_cols():
        cols = []
        for g in CAPPED:
            cols += list(rng.choice(groups[g], size=2, replace=False))
        cols += groups["F18"]
        for pi in rng.choice(len(PAIRS), size=2, replace=False):
            cols += list(PAIRS[pi])
        cols += list(rng.choice(FLAGS, size=2, replace=False))
        return cols

    p0 = Ynp["train"].mean()
    m0 = float(np.log(p0 / (1 - p0)))
    margins = {s: np.full(len(Ynp[s]), m0, dtype=np.float32) for s in Ynp}
    params = {"max_depth": DEPTH, "eta": ETA, "objective": "binary:logistic",
              "tree_method": "hist", "nthread": 8, "seed": SEED}
    records, boosters = [], []
    for t in range(N_TREES):
        cols_t = sample_cols()
        idx = [colpos[c] for c in cols_t]
        dtr = xgb.DMatrix(Xnp["train"][:, idx], label=Ynp["train"],
                          base_margin=margins["train"], feature_names=cols_t)
        bst = xgb.train(params, dtr, num_boost_round=1)
        margins["train"] = bst.predict(dtr, output_margin=True)
        for s in ("val", "eval"):
            ds = xgb.DMatrix(Xnp[s][:, idx], base_margin=margins[s],
                             feature_names=cols_t)
            margins[s] = bst.predict(ds, output_margin=True)
        # per-tree anatomy for analyze_stratified_trees.py
        tdf = bst.trees_to_dataframe()
        id2 = {r.ID: (r.Feature, r.Yes, r.No) for r in tdf.itertuples()}
        pairs = set()
        stack = [("0-0", [])]
        while stack:
            nid, acc = stack.pop()
            f, yes, no = id2[nid]
            if f == "Leaf":
                pairs.update(combinations(sorted(set(acc)), 2))
                continue
            stack.append((yes, acc + [f]))
            stack.append((no, acc + [f]))
        records.append({
            "cols": cols_t,
            "total_gain": bst.get_score(importance_type="total_gain"),
            "weight": bst.get_score(importance_type="weight"),
            "path_pairs": sorted(pairs),
        })
        boosters.append(bst)
        if (t + 1) % 200 == 0:
            print(f"[strat] tree {t+1}/{N_TREES} ({time.time()-t0:.0f}s)",
                  flush=True)
    pv = 1 / (1 + np.exp(-margins["val"]))
    pe = 1 / (1 + np.exp(-margins["eval"]))
    results["stratified"] = score(pv, pe)

    with open(OUT_DIR / "artifacts.pkl", "wb") as fh:
        pickle.dump({"records": records, "feat_cols": feat_cols,
                     "groups": groups, "m0": m0, "params": params}, fh)
    with open(OUT_DIR / "model.pkl", "wb") as fh:
        pickle.dump({"boosters": boosters,
                     "cols_per_tree": [r["cols"] for r in records],
                     "m0": m0}, fh)

    # ---- B) unstructured control -------------------------------------------
    m_b = xgb.XGBClassifier(
        n_estimators=N_TREES, max_depth=DEPTH, learning_rate=ETA,
        colsample_bytree=35 / len(feat_cols), tree_method="hist",
        random_state=SEED, n_jobs=8, eval_metric="logloss")
    m_b.fit(Xnp["train"], Ynp["train"])
    results["control_rand35"] = score(m_b.predict_proba(Xnp["val"])[:, 1],
                                      m_b.predict_proba(Xnp["eval"])[:, 1])

    # ---- C) iter-0 baseline --------------------------------------------------
    m_c = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.3, tree_method="hist",
        random_state=SEED, n_jobs=8, eval_metric="logloss")
    m_c.fit(Xnp["train"], Ynp["train"])
    results["baseline_iter0"] = score(m_c.predict_proba(Xnp["val"])[:, 1],
                                      m_c.predict_proba(Xnp["eval"])[:, 1])

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "meta"},
                     indent=1), flush=True)
    print(f"[done] artifacts in {OUT_DIR} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
