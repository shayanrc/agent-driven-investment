"""Family-stratified boosting (memo _284 recipe) on the nifty500 F18 cells
(task #30/#31). Adapts the sp500-hardcoded `stratified_boosting.py` to nifty500:
2015 date_aligned segments, NSE calendar flags, and 7 capped ladder families
(no F20 vwap in the `all_fundamentals_calendar2` matrix — the only deviation).

Each tree trains on its own structured column subset: 7 lookback-ladder families
capped at <=2 distinct features/tree, F18 fundamentals UNCAPPED (all 10), calendar
sin/cos pairs kept intact. Compared against a same-budget random-subset control
and the iter-0 default-HP baseline. Scored on val + eval only; test stays BLIND.

Usage:
    uv run python -m scripts.gbdt.stratified_boosting_nifty500 [30_100 | 20_100]
"""
import json
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from gbdt.data import load_panel
from gbdt.targets import build_target

# ---- cell selection --------------------------------------------------------
CELLS = {
    "30_100": ("nifty500_up_30pct_100d_dd15pct_ffund", 30, 100, 0.15),
    "20_100": ("nifty500_up_20pct_100d_dd10pct_ffund", 20, 100, 0.10),
}
KEY = sys.argv[1] if len(sys.argv) > 1 else "30_100"
CELL, THRESH, HORIZON, MAXDD = CELLS[KEY]
RUN_DIR = Path("results/gbdt/experiments") / CELL
OUT_DIR = Path("runs/gbdt/stratified") / CELL
UNIVERSE = "nifty500"
SNAPSHOT_END = "2026-07-06"
SEED, N_TREES, ETA, DEPTH = 42, 800, 0.05, 6
PATIENCE = 50  # early-stopping patience on val logloss (rounds without improvement)

# nifty500 date_aligned segments (train_start 2015-01-01), matching the _285
# sweep boundaries. test 2024-09-04..2025-07-01 stays BLIND (not in SEGMENTS).
SEGMENTS = {
    "train": ("2015-01-01", "2022-03-29"),
    "val":   ("2022-03-30", "2023-11-09"),
    "eval":  ("2023-11-10", "2024-09-03"),
}

PAIRS = [("dow_sin", "dow_cos"), ("dom_sin", "dom_cos"), ("moy_sin", "moy_cos"),
         ("moq_sin", "moq_cos"), ("qoy_sin", "qoy_cos")]
FLAGS = ["fiscal_year_end_week", "budget_week", "diwali_week", "fomc_week"]
PAIR_COLS = {c for p in PAIRS for c in p}
# 7 capped ladder families (sp500's 8 minus vwap — no F20 in this matrix).
CAPPED = ["volatility", "returns", "drawdown", "cross-sectional", "persistence",
          "volume", "trend"]


def classify(f: str) -> str:
    if f.startswith("fund_"):                              return "F18"
    if f.startswith("vwap_dev"):                           return "vwap"
    if f in PAIR_COLS or f in FLAGS:                       return "calendar"
    if "_xs_" in f:                                        return "cross-sectional"
    if "outside" in f:                                     return "persistence"
    if any(s in f for s in ("garman_klass", "parkinson", "yang_zhang",
        "realized_vol", "vol_pct", "volatility", "_vol_", "vol_change",
        "vol_ret_corr")):                                  return "volatility"
    if "drawdown" in f:                                    return "drawdown"
    if any(s in f for s in ("volume", "dollar_move", "amihud", "turnover",
        "illiq", "obv")):                                  return "volume"
    if any(s in f for s in ("sma_distance", "ema_", "macd", "sma_", "bollinger",
        "dist_")):                                         return "trend"
    if any(s in f for s in ("stock_return", "rel_strength", "index_return",
        "_return_", "momentum", "roc_", "reversal", "runup", "beta", "skew",
        "kurt")):                                          return "returns"
    if "zscore" in f:                                      return "volatility"
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
    # Drop only TARGET-NaN rows — NOT feature-NaN. xgboost handles missing
    # features natively (as the real pipeline does); the nifty500 F18 columns are
    # 40-50% NaN (sparse NSE fundamentals), so a full .dropna() would decimate the
    # panel to the F18-dense subset (860k -> 162k) and wreck the segment sizes.
    df = X.join(y, how="left")
    df = df[df["target"].notna()]
    del X, panel
    dates = df.index.get_level_values("date")
    feat_cols = [c for c in df.columns if c != "target"]
    groups: dict[str, list[str]] = {}
    for c in feat_cols:
        groups.setdefault(classify(c), []).append(c)
    assert not groups.get("OTHER"), f"unclassified: {groups.get('OTHER')}"
    for g in CAPPED:
        assert len(groups.get(g, [])) >= 2, f"capped family {g} has <2 cols"

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
                      **{f"rp{k}": v for k, v in rpk(DIDX[s], Ynp[s], p).items()}}
        return out

    results = {"meta": {
        "cell": CELL, "seed": SEED, "n_trees": N_TREES, "eta": ETA,
        "depth": DEPTH, "segments": SEGMENTS, "capped": CAPPED,
        "n_features": len(feat_cols), "group_sizes": {g: len(v) for g, v in groups.items()},
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
    # Early stopping on val logloss (user requirement): keep the best iteration
    # rather than a fixed N_TREES. Margins accumulate in-place via base_margin
    # continuation, so we snapshot the val/eval margins at the best val-logloss
    # point and score from those (not the final, over-trained state).
    best_ll, best_t, no_improve = np.inf, 0, 0
    best_margins = {"val": margins["val"].copy(), "eval": margins["eval"].copy()}
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
        records.append({"cols": cols_t})
        boosters.append(bst)
        ll = log_loss(Ynp["val"], 1.0 / (1.0 + np.exp(-margins["val"])), labels=[0, 1])
        if ll < best_ll - 1e-5:
            best_ll, best_t, no_improve = ll, t + 1, 0
            best_margins = {"val": margins["val"].copy(),
                            "eval": margins["eval"].copy()}
        else:
            no_improve += 1
        if (t + 1) % 200 == 0:
            print(f"[strat] tree {t+1}/{N_TREES} val_ll={ll:.5f} "
                  f"best={best_ll:.5f}@{best_t} ({time.time()-t0:.0f}s)", flush=True)
        if no_improve >= PATIENCE:
            print(f"[strat] early stop @ tree {t+1} — best {best_t} "
                  f"(val_ll {best_ll:.5f})", flush=True)
            break
    boosters = boosters[:best_t]
    records = records[:best_t]
    results["meta"]["stratified_n_trees"] = best_t
    pv = 1 / (1 + np.exp(-best_margins["val"]))
    pe = 1 / (1 + np.exp(-best_margins["eval"]))
    results["stratified"] = score(pv, pe)
    with open(OUT_DIR / "model.pkl", "wb") as fh:
        pickle.dump({"boosters": boosters,
                     "cols_per_tree": [r["cols"] for r in records], "m0": m0}, fh)

    # ---- B) unstructured control (same budget, random 30-col subsets) -------
    # Early-stopped on val too, so the comparison is structure-vs-no-structure at
    # matched early-stopping discipline (not fixed-800 vs early-stopped).
    m_b = xgb.XGBClassifier(
        n_estimators=N_TREES, max_depth=DEPTH, learning_rate=ETA,
        colsample_bytree=30 / len(feat_cols), tree_method="hist",
        random_state=SEED, n_jobs=8, eval_metric="logloss",
        early_stopping_rounds=PATIENCE)
    m_b.fit(Xnp["train"], Ynp["train"],
            eval_set=[(Xnp["val"], Ynp["val"])], verbose=False)
    results["meta"]["control_n_trees"] = int(m_b.best_iteration + 1)
    results["control_rand30"] = score(m_b.predict_proba(Xnp["val"])[:, 1],
                                      m_b.predict_proba(Xnp["eval"])[:, 1])

    # ---- C) iter-0 baseline (default HP, full pool) -------------------------
    m_c = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.3, tree_method="hist",
        random_state=SEED, n_jobs=8, eval_metric="logloss")
    m_c.fit(Xnp["train"], Ynp["train"])
    results["baseline_iter0"] = score(m_c.predict_proba(Xnp["val"])[:, 1],
                                      m_c.predict_proba(Xnp["eval"])[:, 1])

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "meta"}, indent=1),
          flush=True)
    print(f"[done] {CELL} artifacts in {OUT_DIR} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
