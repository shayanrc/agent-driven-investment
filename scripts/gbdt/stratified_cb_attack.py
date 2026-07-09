"""Board-attack round 2: the _284 stratified recipe on a CatBoost backend.

Round-1 read (docs/gbdt/_284): the xgboost stratified ensemble wins on
common+mid-horizon decoupled cells but does NOT transfer to H=200, while
CatBoost dominates xgboost systematically on long horizons (_277/_278) — and
both H=200 incumbents are CatBoost. Round 2 tests whether structure + the
incumbents' backend closes the gap. Pre-declared recipe, _284 constants
unchanged; only the per-tree learner swaps (1-iteration CatBoost fits chained
via Pool ``baseline``, the base_margin equivalent).

Honest protocol: incumbent's exact date-aligned segments; val/eval printed;
TEST metrics written to JSON only (read after the candidate leads on eval —
both sealed test looks remain banked). NaN-tolerant harness
(dropna(subset=["target"]) — the runner never drops feature-NaN rows) +
pyarrow row filters on the matrix read (the OOM lesson).

Usage: uv run python -m scripts.gbdt.stratified_cb_attack {sp500|r1k}
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, roc_auc_score

from gbdt.data import load_panel
from gbdt.targets import build_target

t0 = time.time()
SEED, N_TREES, ETA, DEPTH = 42, 800, 0.05, 6
SEG = {"train": ("2018-01-02", "2021-03-08"), "val": ("2021-03-09", "2022-10-06"),
       "eval": ("2022-10-07", "2023-07-26"), "test": ("2023-07-27", "2024-10-03")}

CELLS = {
    "sp500": {
        "run": "results/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune",
        "universe": "sp500",
        # public registry test metrics of sp500_up_50pct_200d_dd25pct cbagent
        "incumbent": {"basis": "test", "auc": 0.7161, "rp1": 0.930,
                      "rp3": 0.6167, "rp5": 0.5553, "rp10": 0.4873},
    },
    "r1k": {
        "run": "results/gbdt/experiments/russell1000_up_50pct_200d_dd25pct_maxtune",
        "universe": "russell1000",
        # incumbent's unbiased EVAL metrics (from its published eval.csv);
        # its test stays sealed.
        "incumbent": {"basis": "eval", "auc": 0.722, "rp1": 0.745, "rp3": 0.595},
    },
}

PAIRS = [("dow_sin", "dow_cos"), ("dom_sin", "dom_cos"), ("moy_sin", "moy_cos"),
         ("moq_sin", "moq_cos"), ("qoy_sin", "qoy_cos")]
FLAGS = ["fiscal_year_end_week", "budget_week", "diwali_week", "fomc_week"]
PAIR_COLS = {c for p in PAIRS for c in p}
CAPPED = ["volatility", "returns", "drawdown", "cross-sectional", "persistence",
          "volume", "trend", "vwap"]


def classify(f):
    if f.startswith("fund_"):                              return "F18"
    if f.startswith("vwap_dev"):                           return "vwap"
    if f in PAIR_COLS or f in FLAGS:                       return "calendar"
    if "_xs_" in f:                                        return "cross-sectional"
    if "outside" in f:                                     return "persistence"
    if any(s in f for s in ("garman_klass", "parkinson", "yang_zhang", "realized_vol",
        "vol_pct", "volatility", "_vol_", "vol_change", "vol_ret_corr")):
                                                           return "volatility"
    if "drawdown" in f:                                    return "drawdown"
    if any(s in f for s in ("volume", "dollar_move", "amihud", "turnover", "illiq", "obv")):
                                                           return "volume"
    if any(s in f for s in ("sma_distance", "ema_", "macd", "sma_", "bollinger", "dist_")):
                                                           return "trend"
    if any(s in f for s in ("stock_return", "rel_strength", "index_return", "_return_",
        "momentum", "roc_", "reversal", "runup", "beta", "skew", "kurt")):
                                                           return "returns"
    if "zscore" in f:                                      return "volatility"
    return "OTHER"


def rpk(index, ytrue, p, ks=(1, 3, 5, 10)):
    t = pd.DataFrame({"date": index.get_level_values("date"),
                      "ticker": index.get_level_values("ticker"),
                      "y": ytrue, "p": p})
    out = {}
    for k in ks:
        rs = []
        for _, g in t.groupby("date"):
            Rq = int(g["y"].sum())
            if Rq == 0:
                continue
            gg = g.sort_values(["p", "ticker"], ascending=[False, True], kind="mergesort")
            rs.append(int(gg["y"].head(k).sum()) / min(k, Rq))
        out[k] = float(np.mean(rs))
    return out


cell_key = sys.argv[1]
cell = CELLS[cell_key]
# Round-3 arm (pre-declared 2026-07-09, before any results): "double" =
# 2x budget at half the learning rate (1600 trees, eta 0.025), recipe
# otherwise frozen — the standard capacity trade, testing whether the
# H=200 @1 gap responds to ensemble depth.
suffix = ""
if len(sys.argv) > 2 and sys.argv[2] == "double":
    N_TREES, ETA = 1600, 0.025
    suffix = "_double"
OUT = Path(f"runs/gbdt/stratified/{cell_key}_50pct_200d_cb{suffix}")
OUT.mkdir(parents=True, exist_ok=True)

X = pd.read_parquet(
    f"{cell['run']}/_feature_matrix_cache.parquet",
    filters=[("date", ">=", pd.Timestamp(SEG["train"][0])),
             ("date", "<=", pd.Timestamp(SEG["test"][1]))],
)
panel = load_panel(cell["universe"], end="2026-07-06").panel
y = build_target(panel, direction="up", threshold_pct=50, horizon_days=200,
                 max_drawdown=0.25).rename("target")
del panel
df = X.join(y, how="left").dropna(subset=["target"])  # NaN-tolerant (runner-faithful)
del X
dates = df.index.get_level_values("date")
feat_cols = [c for c in df.columns if c != "target"]
groups = {}
for c in feat_cols:
    groups.setdefault(classify(c), []).append(c)

Xnp, Ynp, DIDX = {}, {}, {}
for s, (a, b) in SEG.items():
    sub = df[(dates >= a) & (dates <= b)]
    Xnp[s] = sub[feat_cols].to_numpy(dtype=np.float32)
    Ynp[s] = sub["target"].to_numpy(dtype=np.int8)
    DIDX[s] = sub.index
    print(f"[seg] {s}: {len(sub):,} rows prevalence {Ynp[s].mean():.4f} "
          f"({sub.index.get_level_values('date').min().date()}->"
          f"{sub.index.get_level_values('date').max().date()})", flush=True)
del df
colpos = {c: i for i, c in enumerate(feat_cols)}

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
margins = {s: np.full(len(Ynp[s]), m0, dtype=np.float64) for s in SEG}

# Per-tree learner: 1-iteration CatBoost chained via Pool baseline. has_time
# pinned per project convention (C6); data is date-sorted. Plain boosting —
# ordered boosting is meaningless at depth-1 iteration counts and 10x slower.
CB_PARAMS = dict(iterations=1, depth=DEPTH, learning_rate=ETA,
                 loss_function="Logloss", boosting_type="Plain", has_time=True,
                 random_seed=SEED, thread_count=8, verbose=0,
                 allow_writing_files=False)

# Model objects are NOT retained: 800 fitted CatBoost objects OOM-killed the
# r1k shot at 36.7GB RSS (2026-07-09). The ensemble is seed-deterministic
# (rng consumed identically per tree), so it regenerates on demand; metrics +
# the per-tree column log are the deliverables.
colslog = []
for t in range(N_TREES):
    cols_t = sample_cols()
    idx = [colpos[c] for c in cols_t]
    ptr = Pool(Xnp["train"][:, idx], label=Ynp["train"],
               baseline=margins["train"], feature_names=cols_t)
    m = CatBoostClassifier(**CB_PARAMS)
    m.fit(ptr)
    margins["train"] = m.predict(ptr, prediction_type="RawFormulaVal")
    del ptr
    for s in ("val", "eval", "test"):
        ps = Pool(Xnp[s][:, idx], baseline=margins[s], feature_names=cols_t)
        margins[s] = m.predict(ps, prediction_type="RawFormulaVal")
        del ps
    del m
    colslog.append(cols_t)
    if (t + 1) % 100 == 0:
        print(f"[strat-cb] tree {t+1}/{N_TREES} ({time.time()-t0:.0f}s)", flush=True)

res = {}
for s in ("val", "eval", "test"):
    p = 1 / (1 + np.exp(-margins[s]))
    res[s] = {"brier": float(brier_score_loss(Ynp[s], p)),
              "auc": float(roc_auc_score(Ynp[s], p)),
              **{f"rp{k}": v for k, v in rpk(DIDX[s], Ynp[s], p).items()}}
json.dump({"incumbent": cell["incumbent"], "segments": SEG, "results": res},
          open(OUT / "results.json", "w"), indent=1)
with open(OUT / "recipe.pkl", "wb") as fh:
    pickle.dump({"cols_per_tree": colslog, "m0": m0, "seed": SEED,
                 "cb_params": CB_PARAMS}, fh)

print(f"\n=== {cell_key} val/eval (decision segments) — TEST withheld to JSON ===")
for s in ("val", "eval"):
    r = res[s]
    print(f"  {s:5} brier {r['brier']:.4f} auc {r['auc']:.4f} " +
          " ".join(f"rp{k} {r[f'rp{k}']:.3f}" for k in (1, 3, 5, 10)), flush=True)
inc = cell["incumbent"]
print(f"  incumbent ({inc['basis']}): " +
      " ".join(f"{k} {v}" for k, v in inc.items() if k != "basis"))
print(f"[done] ({time.time()-t0:.0f}s)")
