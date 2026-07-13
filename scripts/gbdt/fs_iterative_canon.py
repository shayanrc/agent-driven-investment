"""Iterative feature selection for the canonical-periods fine-tune.

Per the fine-tune recipe: fit XGBoost with 500 trees + early stopping on val,
rank features by BOTH gain and SHAP (mean|contrib| on a train subsample), then
iteratively drop the least-important features (combined gain+SHAP rank) and
refit, tracking val performance. Runs the full backward-elimination trajectory
(down to MIN_FEATS) so the val-peak / knee is visible, then emits the trajectory
+ the val-optimal feature set for the HP step.

Reuses the warm universe feature matrix from the base run (target-agnostic) and
the canonical explicit-boundary split. Date-masking == the runner's carve here
because min_rows_per_ticker=2591 makes every kept ticker present in every
segment day (base run: 468 tickers x 250 td = 117000 rows/segment).

Usage:
  CELL=50   -> sp500 +50%/50d/dd25%   (default)
  CELL=20   -> sp500 +20%/25d/dd10%
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

from gbdt.data import load_panel
from gbdt.targets import build_target
from gbdt.uniqueness import compute_uniqueness_weights
from gbdt.train import SplitSpec, segment_bound_indices
from gbdt.universe_calendar import get_calendar

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon_cells import resolve, SPLIT, SNAP  # noqa: E402

CELL = os.environ.get("CELL", "50")
cell = resolve(CELL)
THR, HOR, DD, UNIV = cell["thr"], cell["hor"], cell["dd"], cell["universe"]
BASE, FT = cell["base"], cell["ft"]

SEED, N_TREES, ES, ETA, MAXDEPTH = 42, 500, 50, 0.05, 6
DROP_FRAC, MIN_FEATS, SHAP_N = 0.20, 10, 60000
MATRIX = Path(f"results/gbdt/experiments/{BASE}/_feature_matrix_cache.parquet")
OUT = Path(f"results/gbdt/experiments/{FT}/fs"); OUT.mkdir(parents=True, exist_ok=True)

# ---- load matrix + target + uniqueness weights ----
log(f"CELL={CELL} +{THR}%/{HOR}d/dd{DD}  load matrix {MATRIX}")
X = pd.read_parquet(MATRIX)
feat_all = list(X.columns)
log(f"matrix {X.shape}  ({len(feat_all)} features)")
log("load panel + build target + uniqueness weights")
panel = load_panel(UNIV, end=SNAP).panel
y = build_target(panel, direction="up", threshold_pct=THR, horizon_days=HOR,
                 max_drawdown=DD).reindex(X.index)
del panel
w = compute_uniqueness_weights(pd.DataFrame(index=X.index), horizon=HOR).reindex(X.index)

# ---- carve canonical split by date mask (== runner carve for full-membership tickers) ----
split = SplitSpec(mode="date_aligned", **SPLIT)
cal = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in get_calendar(UNIV)))
b = segment_bound_indices(split, cal)
segd = {s: (cal[i0], cal[i1]) for s, (i0, i1) in b.items()}
log(f"segments: " + " ".join(f"{s}={segd[s][0].date()}..{segd[s][1].date()}" for s in segd))
dates = X.index.get_level_values("date")
def seg(name):
    s, e = segd[name]
    m = (dates >= s) & (dates <= e) & y.notna().values
    return X[m], y[m], w[m], X.index[m]
Xtr, ytr, wtr, _ = seg("train")
Xva, yva, wva, iva = seg("val")
Xev, yev, wev, iev = seg("eval")
log(f"train {Xtr.shape[0]:,}  val {Xva.shape[0]:,}  eval {Xev.shape[0]:,}  "
    f"pos-rate tr/va/ev {ytr.mean():.4f}/{yva.mean():.4f}/{yev.mean():.4f}")

def rpk(mi, ytrue, pred, ks=(1, 3, 5, 10, 20)):
    df = pd.DataFrame({"d": mi.get_level_values("date"),
                       "t": mi.get_level_values("ticker"),
                       "y": np.asarray(ytrue), "p": np.asarray(pred)})
    out = {}
    for k in ks:
        num = den = 0.0
        for _, g in df.groupby("d", sort=False):
            R = int(g.y.sum())
            if R <= 0:
                continue
            top = g.sort_values(["p", "t"], ascending=[False, True],
                                kind="mergesort").head(k)
            num += top.y.sum() / min(k, R); den += 1
        out[k] = num / den if den else float("nan")
    return out

def fit_eval(feats):
    dtr = xgb.DMatrix(Xtr[feats].values, label=ytr.values, weight=wtr.values,
                      feature_names=feats)
    dva = xgb.DMatrix(Xva[feats].values, label=yva.values, weight=wva.values,
                      feature_names=feats)
    ev = {}
    # Early stopping EXPLICITLY on val AUC (maximize); logloss kept only for reporting.
    bst = xgb.train(
        {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
         "tree_method": "hist", "eta": ETA, "max_depth": MAXDEPTH, "seed": SEED},
        dtr, num_boost_round=N_TREES, evals=[(dva, "val")],
        callbacks=[EarlyStopping(rounds=ES, metric_name="auc", data_name="val",
                                 maximize=True)],
        evals_result=ev, verbose_eval=False)
    bi = bst.best_iteration
    rng = (0, bi + 1)
    pva = bst.predict(dva, iteration_range=rng)
    pev = bst.predict(xgb.DMatrix(Xev[feats].values, feature_names=feats),
                      iteration_range=rng)
    gain = bst.get_score(importance_type="gain")
    n = min(SHAP_N, Xtr.shape[0])
    samp = np.random.default_rng(SEED).choice(Xtr.shape[0], n, replace=False)
    contribs = bst.predict(xgb.DMatrix(Xtr[feats].values[samp], feature_names=feats),
                           pred_contribs=True)
    shap = dict(zip(feats, np.abs(contribs[:, :-1]).mean(axis=0)))
    return (bi, float(ev["val"]["auc"][bi]), float(ev["val"]["logloss"][bi]),
            rpk(iva, yva.values, pva), rpk(iev, yev.values, pev), gain, shap)

# ---- iterative backward elimination ----
feats = list(feat_all); traj = []; r = 0
while len(feats) >= MIN_FEATS:
    bi, va, vl, vrp, erp, gain, shap = fit_eval(feats)
    traj.append({"round": r, "n_features": len(feats), "best_iter": int(bi),
                 "val_auc": va, "val_logloss": vl,
                 "val_rp": {str(k): v for k, v in vrp.items()},
                 "eval_rp": {str(k): v for k, v in erp.items()},
                 "features": list(feats)})
    log(f"r{r:02d} nfeat={len(feats):3d} it={bi:3d} val_auc={va:.4f} val_ll={vl:.4f} "
        f"val_rp@5={vrp[5]:.3f} val_rp@10={vrp[10]:.3f} | eval_rp@5={erp[5]:.3f} "
        f"eval_rp@10={erp[10]:.3f}")
    if len(feats) <= MIN_FEATS:
        break
    grank = pd.Series([gain.get(f, 0.0) for f in feats]).rank(method="average").values
    srank = pd.Series([shap.get(f, 0.0) for f in feats]).rank(method="average").values
    comb = (grank + srank) / 2.0
    order = np.argsort(comb, kind="mergesort")  # least important first
    ndrop = max(1, int(DROP_FRAC * len(feats)))
    ndrop = min(ndrop, len(feats) - MIN_FEATS)
    drop = set(order[:ndrop].tolist())
    feats = [f for i, f in enumerate(feats) if i not in drop]
    r += 1

json.dump(traj, open(OUT / "fs_trajectory.json", "w"), indent=2)
best_auc = max(traj, key=lambda x: x["val_auc"])
best_rp = max(traj, key=lambda x: x["val_rp"]["10"])
json.dump({"by_val_auc": {"round": best_auc["round"], "n_features": best_auc["n_features"],
                          "features": best_auc["features"]},
           "by_val_rp10": {"round": best_rp["round"], "n_features": best_rp["n_features"],
                           "features": best_rp["features"]}},
          open(OUT / "fs_selected.json", "w"), indent=2)
log(f"DONE. best val_auc: r{best_auc['round']} nfeat={best_auc['n_features']} "
    f"auc={best_auc['val_auc']:.4f} | best val_rp@10: r{best_rp['round']} "
    f"nfeat={best_rp['n_features']} rp@10={best_rp['val_rp']['10']:.3f}")
log(f"-> {OUT}/fs_trajectory.json + fs_selected.json")
