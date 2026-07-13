"""Fit ONE (feature-set, HP) config on the canonical split; report val + eval
R-p@K + val AUC. For reasoned, one-at-a-time exploration over BOTH feature set
and hyperparameters (not a grid).

Prep cache holds ALL 279 feature columns once, so any FS round (or the locked
set, or an explicit list) can be subset on the fly and fit in seconds.
Selection window per the canonical roles: EVAL (HP tuning). Test is NOT touched.

Env:
  CELL   50 | 20
  FEATS  "locked" (default, the FS-locked set) | "rN" (FS trajectory round N) | "all" | comma-list
  HP     JSON hyperparameter dict
  LABEL  free-text tag for the log

  e.g. FEATS=r11 HP='{"max_depth":6,"min_child_weight":10,...}' LABEL="f26_mcw10"
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import roc_auc_score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon_cells import resolve, SPLIT, SNAP  # noqa: E402

CELL = os.environ.get("CELL", "50")
cell = resolve(CELL)
THR, HOR, DD, UNIV = cell["thr"], cell["hor"], cell["dd"], cell["universe"]
BASE, FT = cell["base"], cell["ft"]
SEED, N_TREES, ES = 42, 500, 50
FTDIR = Path(f"results/gbdt/experiments/{FT}")
PREP = FTDIR / "hp" / f"prep_all_{CELL}.npz"
PREP.parent.mkdir(parents=True, exist_ok=True)

if not PREP.exists():
    log("building ALL-feature prep cache (once)")
    from gbdt.data import load_panel
    from gbdt.targets import build_target
    from gbdt.uniqueness import compute_uniqueness_weights
    from gbdt.train import SplitSpec, segment_bound_indices
    from gbdt.universe_calendar import get_calendar
    X = pd.read_parquet(f"results/gbdt/experiments/{BASE}/_feature_matrix_cache.parquet")
    feat_all = list(X.columns)
    panel = load_panel(UNIV, end=SNAP).panel
    y = build_target(panel, direction="up", threshold_pct=THR, horizon_days=HOR,
                     max_drawdown=DD).reindex(X.index)
    del panel
    w = compute_uniqueness_weights(pd.DataFrame(index=X.index), horizon=HOR).reindex(X.index)
    split = SplitSpec(mode="date_aligned", **SPLIT)
    cal = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in get_calendar(UNIV)))
    segd = {s: (cal[i0], cal[i1]) for s, (i0, i1) in segment_bound_indices(split, cal).items()}
    dates = X.index.get_level_values("date")
    def seg(name):
        s, e = segd[name]; m = (dates >= s) & (dates <= e) & y.notna().values
        idx = X.index[m]
        return (X[m].values.astype(np.float32), y[m].values.astype(np.float32),
                w[m].values.astype(np.float32),
                idx.get_level_values("date").values.astype("datetime64[D]").astype(str),
                idx.get_level_values("ticker").values.astype(str))
    Xtr, ytr, wtr, _, _ = seg("train"); Xva, yva, wva, vad, vat = seg("val")
    Xev, yev, wev, evd, evt = seg("eval")
    np.savez(PREP, feat_all=np.array(feat_all), Xtr=Xtr, ytr=ytr, wtr=wtr,
             Xva=Xva, yva=yva, wva=wva, vad=vad, vat=vat, Xev=Xev, yev=yev, evd=evd, evt=evt)
    log(f"prep cached ({len(feat_all)} feats) -> {PREP}")

z = np.load(PREP, allow_pickle=True)
feat_all = list(z["feat_all"])
colpos = {c: i for i, c in enumerate(feat_all)}

# ---- resolve the requested feature set ----
FEATS = os.environ.get("FEATS", "locked")
if FEATS == "all":
    feats = list(feat_all)
elif FEATS == "locked":
    feats = json.load(open(FTDIR / "fs" / "fs_locked.json"))["features"]
elif FEATS.startswith("r"):
    traj = json.load(open(FTDIR / "fs" / "fs_trajectory.json"))
    feats = [r for r in traj if r["round"] == int(FEATS[1:])][0]["features"]
else:
    feats = [f.strip() for f in FEATS.split(",")]
cols = [colpos[f] for f in feats]

dtr = xgb.DMatrix(z["Xtr"][:, cols], label=z["ytr"], weight=z["wtr"], feature_names=feats)
dva = xgb.DMatrix(z["Xva"][:, cols], label=z["yva"], weight=z["wva"], feature_names=feats)
dev = xgb.DMatrix(z["Xev"][:, cols], feature_names=feats)

def rpk(dts, tks, yt, p, ks=(1, 3, 5, 10, 20)):
    df = pd.DataFrame({"d": dts, "t": tks, "y": yt, "p": p})
    out = {}
    for k in ks:
        num = den = 0.0
        for _, g in df.groupby("d", sort=False):
            R = int(g.y.sum())
            if R <= 0: continue
            top = g.sort_values(["p", "t"], ascending=[False, True], kind="mergesort").head(k)
            num += top.y.sum() / min(k, R); den += 1
        out[str(k)] = round(float(num / den), 4) if den else None
    return out

HP = json.loads(os.environ["HP"]); LABEL = os.environ.get("LABEL", "cfg")
ev = {}
bst = xgb.train({"objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
                 "tree_method": "hist", "seed": SEED, **HP},
                dtr, num_boost_round=N_TREES, evals=[(dva, "val")],
                callbacks=[EarlyStopping(rounds=ES, metric_name="auc", data_name="val",
                                         maximize=True)],
                evals_result=ev, verbose_eval=False)
bi = bst.best_iteration; rng = (0, bi + 1)
pva = bst.predict(dva, iteration_range=rng); pev = bst.predict(dev, iteration_range=rng)
val_auc = float(ev["val"]["auc"][bi]); eval_auc = float(roc_auc_score(z["yev"], pev))
vrp = rpk(z["vad"], z["vat"], z["yva"], pva)
erp = rpk(z["evd"], z["evt"], z["yev"], pev)
log(f"[{LABEL}] FEATS={FEATS}({len(feats)}) HP={HP} it={bi}")
log(f"[{LABEL}] EVAL auc={eval_auc:.4f}  R-p@ 1/3/5/10/20 = {erp['1']} {erp['3']} {erp['5']} {erp['10']} {erp['20']}")
log(f"[{LABEL}] val  auc={val_auc:.4f}  R-p@ 1/3/5/10/20 = {vrp['1']} {vrp['3']} {vrp['5']} {vrp['10']} {vrp['20']}")
with open(FTDIR / "hp" / "one_by_one.jsonl", "a") as f:
    f.write(json.dumps({"label": LABEL, "feats": FEATS, "n_features": len(feats), "hp": HP,
                        "best_iter": int(bi), "val_auc": val_auc, "eval_auc": eval_auc,
                        "eval_rp": erp, "val_rp": vrp}) + "\n")
