"""HP tuning for the canonical-periods fine-tune: INDEPENDENT single-knob sweeps.

Step 2 (after FS). On the locked feature set, sweep EACH hyperparameter
independently — vary one knob across its grid while holding all others at a
common baseline — producing a response curve per knob. Then combine the
per-knob argmax (by val R-p@3) into a candidate config and re-fit to verify.

Every fit: 500 trees + early stopping EXPLICITLY on val AUC (maximize), same as
the FS step. Selection metric: val R-p@3 (val AUC, val/eval R-p@K all tracked).

  CELL=50 -> sp500 +50%/50d/dd25%   (default);  CELL=20 -> sp500 +20%/25d/dd10%
"""
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
CFG = {"50": (50, 50, 0.25, "sp500_up_50pct_50d_dd25pct"),
       "20": (20, 25, 0.10, "sp500_up_20pct_25d_dd10pct")}[CELL]
THR, HOR, DD, STEM = CFG
BASE = f"{STEM}_canon_base"; FT = f"{STEM}_canon_ft"
SNAP = "2026-07-06"
SEED, N_TREES, ES = 42, 500, 50
MATRIX = Path(f"results/gbdt/experiments/{BASE}/_feature_matrix_cache.parquet")
FTDIR = Path(f"results/gbdt/experiments/{FT}")
OUT = FTDIR / "hp"; OUT.mkdir(parents=True, exist_ok=True)

BASELINE = {"max_depth": 6, "min_child_weight": 1, "subsample": 1.0,
            "colsample_bytree": 1.0, "gamma": 0.0, "eta": 0.05}
SWEEPS = {
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "gamma": [0.0, 0.5, 1.0],
}

# ---- locked feature set (from the FS step) ----
feats = json.load(open(FTDIR / "fs" / "fs_locked.json"))["features"]
log(f"CELL={CELL} +{THR}%/{HOR}d/dd{DD}  locked {len(feats)} features")

# ---- data (same carve as FS / base run) ----
X = pd.read_parquet(MATRIX)[feats]  # only the locked columns (+ MultiIndex)
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
Xtr, ytr, wtr, _ = seg("train"); Xva, yva, wva, iva = seg("val")
Xev, yev, wev, iev = seg("eval")
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

def fit(cfg):
    ev = {}
    params = {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
              "tree_method": "hist", "seed": SEED, **cfg}
    bst = xgb.train(params, dtr, num_boost_round=N_TREES, evals=[(dva, "val")],
                    callbacks=[EarlyStopping(rounds=ES, metric_name="auc",
                                             data_name="val", maximize=True)],
                    evals_result=ev, verbose_eval=False)
    bi = bst.best_iteration; rng = (0, bi + 1)
    vrp = rpk(iva, yva.values, bst.predict(dva, iteration_range=rng))
    erp = rpk(iev, yev.values, bst.predict(dev, iteration_range=rng))
    return {"best_iter": int(bi), "val_auc": float(ev["val"]["auc"][bi]),
            "val_rp": vrp, "eval_rp": erp}

# ---- baseline ----
base_res = fit(BASELINE)
log(f"BASELINE {BASELINE} -> val_rp@3={base_res['val_rp']['3']:.3f} "
    f"val_rp@5={base_res['val_rp']['5']:.3f} val_auc={base_res['val_auc']:.4f}")

# ---- independent single-knob sweeps ----
curves = {"baseline": {"cfg": BASELINE, "res": base_res}, "sweeps": {}}
best_per_knob = {}
for knob, vals in SWEEPS.items():
    curves["sweeps"][knob] = []
    rows = []
    for v in vals:
        cfg = dict(BASELINE); cfg[knob] = v
        r = fit(cfg)
        curves["sweeps"][knob].append({"value": v, **r})
        rows.append((v, r["val_rp"]["3"], r["val_rp"]["5"], r["val_auc"],
                     r["eval_rp"]["3"], r["eval_rp"]["10"], r["best_iter"]))
        log(f"  {knob}={v}: val_rp@3={r['val_rp']['3']:.3f} val_rp@5={r['val_rp']['5']:.3f} "
            f"val_auc={r['val_auc']:.4f} eval_rp@3={r['eval_rp']['3']:.3f} it={r['best_iter']}")
    best = max(curves["sweeps"][knob], key=lambda x: x["val_rp"]["3"])
    best_per_knob[knob] = best["value"]
    log(f"  -> best {knob}={best['value']} (val_rp@3={best['val_rp']['3']:.3f})")

# ---- combine per-knob argmax + verify ----
combined = dict(BASELINE); combined.update(best_per_knob)
comb_res = fit(combined)
log(f"COMBINED {combined} -> val_rp@3={comb_res['val_rp']['3']:.3f} "
    f"val_rp@5={comb_res['val_rp']['5']:.3f} val_auc={comb_res['val_auc']:.4f} "
    f"eval_rp@3={comb_res['eval_rp']['3']:.3f} eval_rp@10={comb_res['eval_rp']['10']:.3f}")
curves["combined"] = {"cfg": combined, "res": comb_res}
curves["best_per_knob"] = best_per_knob
json.dump(curves, open(OUT / "hp_sweep_curves.json", "w"), indent=2)
log(f"baseline val_rp@3={base_res['val_rp']['3']:.3f} -> combined val_rp@3={comb_res['val_rp']['3']:.3f}")
log(f"-> {OUT}/hp_sweep_curves.json")
