"""Shot 1 at the board: stratified recipe (pre-declared, _284 unchanged) on
sp500_up_50pct_200d_dd25pct — the R-p@1=0.930 cbagent cell. Incumbent's exact
date-aligned segments. val/eval printed; TEST metrics written to JSON only
(read after the candidate stands on val/eval)."""
import json, pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from gbdt.data import load_panel
from gbdt.targets import build_target

t0 = time.time()
RUN = "results/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune"  # target-agnostic matrix
SEED, N_TREES, ETA, DEPTH = 42, 800, 0.05, 6
SEG = {"train": ("2018-01-02","2021-03-08"), "val": ("2021-03-09","2022-10-06"),
       "eval": ("2022-10-07","2023-07-26"), "test": ("2023-07-27","2024-10-03")}
INCUMBENT = {"auc": 0.7161, "rp1": 0.930, "rp3": 0.6167, "rp5": 0.5553, "rp10": 0.4873}

PAIRS = [("dow_sin","dow_cos"), ("dom_sin","dom_cos"), ("moy_sin","moy_cos"),
         ("moq_sin","moq_cos"), ("qoy_sin","qoy_cos")]
FLAGS = ["fiscal_year_end_week","budget_week","diwali_week","fomc_week"]
PAIR_COLS = {c for p in PAIRS for c in p}
CAPPED = ["volatility","returns","drawdown","cross-sectional","persistence",
          "volume","trend","vwap"]

def classify(f):
    if f.startswith("fund_"):                              return "F18"
    if f.startswith("vwap_dev"):                           return "vwap"
    if f in PAIR_COLS or f in FLAGS:                       return "calendar"
    if "_xs_" in f:                                        return "cross-sectional"
    if "outside" in f:                                     return "persistence"
    if any(s in f for s in ("garman_klass","parkinson","yang_zhang","realized_vol",
        "vol_pct","volatility","_vol_","vol_change","vol_ret_corr")):
                                                           return "volatility"
    if "drawdown" in f:                                    return "drawdown"
    if any(s in f for s in ("volume","dollar_move","amihud","turnover","illiq","obv")):
                                                           return "volume"
    if any(s in f for s in ("sma_distance","ema_","macd","sma_","bollinger","dist_")):
                                                           return "trend"
    if any(s in f for s in ("stock_return","rel_strength","index_return","_return_",
        "momentum","roc_","reversal","runup","beta","skew","kurt")):
                                                           return "returns"
    if "zscore" in f:                                      return "volatility"
    return "OTHER"

X = pd.read_parquet(f"{RUN}/_feature_matrix_cache.parquet")
panel = load_panel("sp500", end="2026-07-06").panel
y = build_target(panel, direction="up", threshold_pct=50, horizon_days=200,
                 max_drawdown=0.25).rename("target")
del panel
df = X.join(y, how="left").dropna()
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

def rpk(index, ytrue, p, ks=(1,3,5,10)):
    t = pd.DataFrame({"date": index.get_level_values("date"),
                      "ticker": index.get_level_values("ticker"),
                      "y": ytrue, "p": p})
    out = {}
    for k in ks:
        rs = []
        for _, g in t.groupby("date"):
            Rq = int(g["y"].sum())
            if Rq == 0: continue
            gg = g.sort_values(["p","ticker"], ascending=[False,True], kind="mergesort")
            rs.append(int(gg["y"].head(k).sum())/min(k,Rq))
        out[k] = float(np.mean(rs))
    return out

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
m0 = float(np.log(p0/(1-p0)))
margins = {s: np.full(len(Ynp[s]), m0, dtype=np.float32) for s in SEG}
params = {"max_depth": DEPTH, "eta": ETA, "objective": "binary:logistic",
          "tree_method": "hist", "nthread": 8, "seed": SEED}
boosters, colslog = [], []
for t in range(N_TREES):
    cols_t = sample_cols()
    idx = [colpos[c] for c in cols_t]
    dtr = xgb.DMatrix(Xnp["train"][:, idx], label=Ynp["train"],
                      base_margin=margins["train"], feature_names=cols_t)
    bst = xgb.train(params, dtr, num_boost_round=1)
    margins["train"] = bst.predict(dtr, output_margin=True)
    for s in ("val","eval","test"):
        ds = xgb.DMatrix(Xnp[s][:, idx], base_margin=margins[s], feature_names=cols_t)
        margins[s] = bst.predict(ds, output_margin=True)
    boosters.append(bst); colslog.append(cols_t)
    if (t+1) % 200 == 0:
        print(f"[strat] tree {t+1}/{N_TREES} ({time.time()-t0:.0f}s)", flush=True)

res = {}
for s in ("val","eval","test"):
    p = 1/(1+np.exp(-margins[s]))
    res[s] = {"brier": float(brier_score_loss(Ynp[s], p)),
              "auc": float(roc_auc_score(Ynp[s], p)),
              **{f"rp{k}": v for k, v in rpk(DIDX[s], Ynp[s], p).items()}}
json.dump({"incumbent_test": INCUMBENT, "segments": SEG, "results": res},
          open("/tmp/strat_sp50200_results.json","w"), indent=1)
with open("/tmp/strat_sp50200_model.pkl","wb") as fh:
    pickle.dump({"boosters": boosters, "cols_per_tree": colslog, "m0": m0}, fh)
print("\n=== val/eval (decision segments) — TEST withheld to JSON ===")
for s in ("val","eval"):
    r = res[s]
    print(f"  {s:5} brier {r['brier']:.4f} auc {r['auc']:.4f} " +
          " ".join(f"rp{k} {r[f'rp{k}']:.3f}" for k in (1,3,5,10)), flush=True)
print(f"[done] ({time.time()-t0:.0f}s)")
