"""Final fit + evaluation for the canonical-periods fine-tune.

Fits the fine-tuned model (locked FS feature set + HP-sweep combined config,
500 trees + early stopping on val AUC), evaluates on eval + TEST (2024-07..
2025-06, the held-out final-comparison window), and emits runner-schema
prediction CSVs for eval / test / backtest (2025-07..2026-06) so run_fresh_oos
can backtest it. Saves model.pkl. Compares test R-p@K / AUC vs the base run.

  CELL=50 -> sp500 +50%/50d/dd25% ;  CELL=20 -> sp500 +20%/25d/dd10%
HP override via env (JSON), else the hp/ combined config is read.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import roc_auc_score, brier_score_loss

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
THR, HOR, DD, UNIV, STEM = cell["thr"], cell["hor"], cell["dd"], cell["universe"], cell["stem"]
TOKEN = cell["token"]
BASE, FT = cell["base"], cell["ft"]
SEED, N_TREES, ES = 42, 500, 50
MATRIX = Path(f"results/gbdt/experiments/{BASE}/_feature_matrix_cache.parquet")
FTDIR = Path(f"results/gbdt/experiments/{FT}")
PRED = FTDIR / "predictions"; PRED.mkdir(parents=True, exist_ok=True)

FEATS = os.environ.get("FEATS", "locked")
Xfull = pd.read_parquet(MATRIX)
if FEATS == "locked":
    feats = json.load(open(FTDIR / "fs" / "fs_locked.json"))["features"]
elif FEATS == "all":
    feats = list(Xfull.columns)
elif FEATS.startswith("r"):
    traj = json.load(open(FTDIR / "fs" / "fs_trajectory.json"))
    feats = [r for r in traj if r["round"] == int(FEATS[1:])][0]["features"]
else:
    feats = [f.strip() for f in FEATS.split(",")]
HP = json.loads(os.environ["HP"]) if os.environ.get("HP") else \
    json.load(open(FTDIR / "hp" / "hp_sweep_curves.json"))["combined"]["cfg"]
log(f"CELL={CELL} +{THR}%/{HOR}d/dd{DD}  FEATS={FEATS}({len(feats)}) HP={HP}")

X = Xfull[feats]; del Xfull
panel = load_panel(UNIV, end=SNAP).panel
y = build_target(panel, direction="up", threshold_pct=THR, horizon_days=HOR,
                 max_drawdown=DD).reindex(X.index)
del panel
w = compute_uniqueness_weights(pd.DataFrame(index=X.index), horizon=HOR).reindex(X.index)
split = SplitSpec(mode="date_aligned", **SPLIT)
cal = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in get_calendar(UNIV)))
segd = {s: (cal[i0], cal[i1]) for s, (i0, i1) in segment_bound_indices(split, cal).items()}
dates = X.index.get_level_values("date")
def seg(a, b, need_y=True):
    m = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    if need_y: m = m & y.notna().values
    return X[m], (y[m] if need_y else None), w[m], X.index[m]
Xtr, ytr, wtr, _ = seg(*[d.date() for d in segd["train"]])
Xva, yva, wva, iva = seg(*[d.date() for d in segd["val"]])
Xev, yev, wev, iev = seg(*[d.date() for d in segd["eval"]])
Xte, yte, wte, ite = seg(*[d.date() for d in segd["test"]])
Xbt, _, _, ibt = seg("2025-07-01", "2026-06-30", need_y=False)  # backtest window (labels partial)
log(f"train {Xtr.shape[0]:,} val {Xva.shape[0]:,} eval {Xev.shape[0]:,} "
    f"test {Xte.shape[0]:,} backtest {Xbt.shape[0]:,}")

dtr = xgb.DMatrix(Xtr.values, label=ytr.values, weight=wtr.values, feature_names=feats)
dva = xgb.DMatrix(Xva.values, label=yva.values, weight=wva.values, feature_names=feats)
ev = {}
bst = xgb.train({"objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
                 "tree_method": "hist", "seed": SEED, **HP},
                dtr, num_boost_round=N_TREES, evals=[(dva, "val")],
                callbacks=[EarlyStopping(rounds=ES, metric_name="auc",
                                         data_name="val", maximize=True)],
                evals_result=ev, verbose_eval=False)
bi = bst.best_iteration; rng = (0, bi + 1)
log(f"fit done: best_iter={bi} val_auc@best={ev['val']['auc'][bi]:.4f}")

def rpk(mi, yt, p, ks=(1, 3, 5, 10, 20)):
    df = pd.DataFrame({"d": mi.get_level_values("date"), "t": mi.get_level_values("ticker"),
                       "y": np.asarray(yt), "p": np.asarray(p)})
    out = {}
    for k in ks:
        num = den = 0.0
        for _, g in df.groupby("d", sort=False):
            R = int(g.y.sum())
            if R <= 0: continue
            out.setdefault(k, 0)
            top = g.sort_values(["p", "t"], ascending=[False, True], kind="mergesort").head(k)
            num += top.y.sum() / min(k, R); den += 1
        out[str(k)] = num / den if den else float("nan")
    return {str(k): out[str(k)] for k in ks}

def emit(name, Xs, ys, mi):
    p = bst.predict(xgb.DMatrix(Xs.values, feature_names=feats), iteration_range=rng)
    df = pd.DataFrame({"date": mi.get_level_values("date"), "ticker": mi.get_level_values("ticker"),
                       "p_raw": p, "p_calibrated": p,
                       "y_true": (np.asarray(ys) if ys is not None else np.nan),
                       "sample_weight": 1.0})
    df.to_csv(PRED / f"{name}.csv", index=False)
    return p

emit("val", Xva, yva, iva)  # needed by the backtest calibrator (fit on VAL, leak-free)
summary = {"cell": STEM, "n_features": len(feats), "hp": HP, "best_iter": int(bi)}
for nm, Xs, ys, mi in [("eval", Xev, yev, iev), ("test", Xte, yte, ite)]:
    p = emit(nm, Xs, ys, mi)
    yv = np.asarray(ys)
    summary[nm] = {"n_rows": int(len(yv)), "base_rate": float(yv.mean()),
                   "auc": float(roc_auc_score(yv, p)),
                   "brier": float(brier_score_loss(yv, p)),
                   "r_precision_at_k": rpk(mi, yv, p)}
    log(f"{nm}: base={summary[nm]['base_rate']:.4f} auc={summary[nm]['auc']:.4f} "
        f"rp@3={summary[nm]['r_precision_at_k']['3']:.3f} rp@10={summary[nm]['r_precision_at_k']['10']:.3f}")
emit("backtest", Xbt, None, ibt)
pickle.dump({"booster": bst, "feats": feats, "best_iter": int(bi), "hp": HP},
            open(FTDIR / "model.pkl", "wb"))
json.dump(summary, open(FTDIR / "final_summary.json", "w"), indent=2)
# Runner-compatible artifacts so scripts.backtests.{run_fresh_oos,infer_fresh_predictions}
# + /daily-predictions can consume this ft dir: model.ubj (native XGBoost, the loader
# infer_fresh dispatches on), features.yaml (exact feature order), hp.yaml, spec.yaml
# (incl. features.candidates token for the F18 dispatch), metrics.json (calibrator col).
bst.save_model(str(FTDIR / "model.ubj"))
import yaml as _yaml
_yaml.safe_dump({"features": list(feats)}, open(FTDIR / "features.yaml", "w"))
_yaml.safe_dump({"hp": dict(HP)}, open(FTDIR / "hp.yaml", "w"))
_yaml.safe_dump({"experiment_name": FT,
    "target": {"universe": UNIV, "direction": "up", "threshold_pct": THR,
               "horizon_days": HOR, "max_drawdown": DD},
    "features": {"candidates": TOKEN}}, open(FTDIR / "spec.yaml", "w"))
json.dump({"calibration": {"decision": "identity"}}, open(FTDIR / "metrics.json", "w"))
log(f"saved model.{{pkl,ubj}} + features.yaml + hp.yaml + predictions/{{val,eval,test,backtest}}.csv "
    f"+ spec.yaml + metrics.json -> {FTDIR}")
