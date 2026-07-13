"""HP tuning, corrected: grid over regularization knobs, selected on EVAL R-p@3.

Backtracks the independent-sweep combined config (which was selected on val
R-p@3 — the FS metric, not the HP metric). Per the canonical role convention
HP tuning belongs on EVAL, and the single-knob curves showed val and eval
disagree (regularization hurts val R-p@3 but helps eval R-p@3), and knobs
interact — so this sweeps a small grid over the regularization region and
selects on EVAL R-p@3. Test is NOT touched here (pure holdout): the eval-winner
is evaluated on test separately by final_fit_canon.py.

  CELL=50 -> sp500 +50%/50d/dd25% ;  CELL=20 -> sp500 +20%/25d/dd10%
"""
import itertools
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping

from gbdt.data import load_panel
from gbdt.targets import build_target
from gbdt.uniqueness import compute_uniqueness_weights
from gbdt.train import SplitSpec, segment_bound_indices
from gbdt.universe_calendar import get_calendar

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

CELL = os.environ.get("CELL", "50")
THR, HOR, DD, STEM = {"50": (50, 50, 0.25, "sp500_up_50pct_50d_dd25pct"),
                      "20": (20, 25, 0.10, "sp500_up_20pct_25d_dd10pct")}[CELL]
BASE = f"{STEM}_canon_base"; FT = f"{STEM}_canon_ft"
SNAP = "2026-07-06"; SEED, N_TREES, ES = 42, 500, 50
MATRIX = Path(f"results/gbdt/experiments/{BASE}/_feature_matrix_cache.parquet")
FTDIR = Path(f"results/gbdt/experiments/{FT}"); OUT = FTDIR / "hp"; OUT.mkdir(parents=True, exist_ok=True)

GRID = {"max_depth": [5, 6], "min_child_weight": [1, 10, 20],
        "subsample": [0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0.0], "eta": [0.05]}

feats = json.load(open(FTDIR / "fs" / "fs_locked.json"))["features"]
log(f"CELL={CELL} +{THR}%/{HOR}d/dd{DD}  {len(feats)} feats  grid select on EVAL R-p@3")

X = pd.read_parquet(MATRIX)[feats]
panel = load_panel("sp500", end=SNAP).panel
y = build_target(panel, direction="up", threshold_pct=THR, horizon_days=HOR,
                 max_drawdown=DD).reindex(X.index)
del panel
w = compute_uniqueness_weights(pd.DataFrame(index=X.index), horizon=HOR).reindex(X.index)
split = SplitSpec(mode="date_aligned", train_start=date(2015, 1, 1),
                  val_start=date(2022, 3, 30), eval_start=date(2023, 7, 1),
                  test_start=date(2024, 7, 1), test_end=date(2025, 6, 30))
cal = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in get_calendar("sp500")))
segd = {s: (cal[i0], cal[i1]) for s, (i0, i1) in segment_bound_indices(split, cal).items()}
dates = X.index.get_level_values("date")
def seg(name):
    s, e = segd[name]; m = (dates >= s) & (dates <= e) & y.notna().values
    return X[m], y[m], w[m], X.index[m]
Xtr, ytr, wtr, _ = seg("train"); Xva, yva, wva, iva = seg("val"); Xev, yev, wev, iev = seg("eval")
log(f"train {Xtr.shape[0]:,} val {Xva.shape[0]:,} eval {Xev.shape[0]:,}")
dtr = xgb.DMatrix(Xtr.values, label=ytr.values, weight=wtr.values, feature_names=feats)
dva = xgb.DMatrix(Xva.values, label=yva.values, weight=wva.values, feature_names=feats)
dev = xgb.DMatrix(Xev.values, feature_names=feats)

def rpk(mi, yt, p, ks=(1, 3, 5, 10, 20)):
    df = pd.DataFrame({"d": mi.get_level_values("date"), "t": mi.get_level_values("ticker"),
                       "y": np.asarray(yt), "p": np.asarray(p)})
    out = {}
    for k in ks:
        num = den = 0.0
        for _, g in df.groupby("d", sort=False):
            R = int(g.y.sum())
            if R <= 0: continue
            top = g.sort_values(["p", "t"], ascending=[False, True], kind="mergesort").head(k)
            num += top.y.sum() / min(k, R); den += 1
        out[str(k)] = num / den if den else float("nan")
    return out

keys = list(GRID); results = []
for combo in itertools.product(*[GRID[k] for k in keys]):
    cfg = dict(zip(keys, combo))
    ev = {}
    bst = xgb.train({"objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
                     "tree_method": "hist", "seed": SEED, **cfg},
                    dtr, num_boost_round=N_TREES, evals=[(dva, "val")],
                    callbacks=[EarlyStopping(rounds=ES, metric_name="auc",
                                             data_name="val", maximize=True)],
                    evals_result=ev, verbose_eval=False)
    bi = bst.best_iteration; rng = (0, bi + 1)
    vrp = rpk(iva, yva.values, bst.predict(dva, iteration_range=rng))
    erp = rpk(iev, yev.values, bst.predict(dev, iteration_range=rng))
    results.append({"cfg": cfg, "best_iter": int(bi), "val_auc": float(ev["val"]["auc"][bi]),
                    "val_rp": vrp, "eval_rp": erp})
    log(f"d{cfg['max_depth']} mcw{cfg['min_child_weight']:>2} ss{cfg['subsample']} "
        f"cs{cfg['colsample_bytree']} | eval_rp@3={erp['3']:.3f} eval_rp@5={erp['5']:.3f} "
        f"eval_rp@10={erp['10']:.3f} | val_rp@3={vrp['3']:.3f} it={bi}")

results.sort(key=lambda r: r["eval_rp"]["3"], reverse=True)
json.dump(results, open(OUT / "hp_grid_eval.json", "w"), indent=2)
log("=== TOP 8 by EVAL R-p@3 ===")
for r in results[:8]:
    c = r["cfg"]
    log(f"  d{c['max_depth']} mcw{c['min_child_weight']:>2} ss{c['subsample']} cs{c['colsample_bytree']} "
        f"g{c['gamma']} -> eval_rp@3={r['eval_rp']['3']:.3f} eval_rp@5={r['eval_rp']['5']:.3f} "
        f"eval_rp@10={r['eval_rp']['10']:.3f} | val_rp@3={r['val_rp']['3']:.3f} val_auc={r['val_auc']:.4f}")
log(f"-> {OUT}/hp_grid_eval.json  (winner evaluated on TEST separately)")
