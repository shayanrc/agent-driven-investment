"""ONE-SHOT blind-test commit (memo _284). Pre-declared arms, single look:
  A) stratified (EXACT saved ensemble from /tmp/strat_model.pkl)
  B) control_rand35 (deterministic refit, seed 42)
  C) baseline_iter0 (deterministic refit, seed 42)
Test window: 2024-07-26 -> 2024-12-16 (never touched during tuning)."""
import json, pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from gbdt.data import load_panel
from gbdt.targets import build_target

t0 = time.time()
RUN = "results/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune"
X = pd.read_parquet(f"{RUN}/_feature_matrix_cache.parquet")
panel = load_panel("sp500", end="2026-07-06").panel
y = build_target(panel, direction="up", threshold_pct=20, horizon_days=50,
                 max_drawdown=0.10).rename("target")
df = X.join(y, how="left").dropna()
del X, panel
dates = df.index.get_level_values("date")
feat_cols = [c for c in df.columns if c != "target"]
seg = {}
for s, (a, b) in {"train": ("2019-01-02","2022-03-04"),
                  "test":  ("2024-07-26","2024-12-16")}.items():
    seg[s] = df[(dates >= a) & (dates <= b)]
Xtr = seg["train"][feat_cols].to_numpy(dtype=np.float32)
ytr = seg["train"]["target"].to_numpy(dtype=np.int8)
Xte = seg["test"][feat_cols].to_numpy(dtype=np.float32)
yte = seg["test"]["target"].to_numpy(dtype=np.int8)
te_idx = seg["test"].index
del df, seg
colpos = {c: i for i, c in enumerate(feat_cols)}
qd = te_idx.get_level_values("date").nunique()
print(f"[test] {len(yte):,} rows, {qd} days, base_rate {yte.mean():.4f} "
      f"({time.time()-t0:.0f}s)", flush=True)

def rpk(p, ks=(1,3,5,10,20)):
    t = pd.DataFrame({"date": te_idx.get_level_values("date"),
                      "ticker": te_idx.get_level_values("ticker"),
                      "y": yte, "p": p})
    out = {}
    for k in ks:
        rs = []
        for _, g in t.groupby("date"):
            Rq = int(g["y"].sum())
            if Rq == 0: continue
            gg = g.sort_values(["p","ticker"], ascending=[False,True], kind="mergesort")
            rs.append(int(gg["y"].head(k).sum()) / min(k, Rq))
        out[k] = float(np.mean(rs))
    return out

def score(p):
    return {"brier": float(brier_score_loss(yte, p)),
            "auc": float(roc_auc_score(yte, p)),
            **{f"rp{k}": v for k, v in rpk(p).items()}}

res = {"meta": {"test_rows": int(len(yte)), "Q_days": int(qd),
                "base_rate": float(yte.mean())}}

# A) stratified — exact saved ensemble
mdl = pickle.load(open("/tmp/strat_model.pkl", "rb"))
margin = np.full(len(yte), mdl["m0"], dtype=np.float32)
for bst, cols in zip(mdl["boosters"], mdl["cols_per_tree"]):
    idx = [colpos[c] for c in cols]
    d = xgb.DMatrix(Xte[:, idx], base_margin=margin, feature_names=cols)
    margin = bst.predict(d, output_margin=True)
res["stratified"] = score(1/(1+np.exp(-margin)))
print(f"[done] stratified ({time.time()-t0:.0f}s)", flush=True)

# B) control
mB = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.05,
                       colsample_bytree=35/len(feat_cols), tree_method="hist",
                       random_state=42, n_jobs=8, eval_metric="logloss")
mB.fit(Xtr, ytr)
res["control_rand35"] = score(mB.predict_proba(Xte)[:,1])
print(f"[done] control ({time.time()-t0:.0f}s)", flush=True)

# C) baseline
mC = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                       tree_method="hist", random_state=42, n_jobs=8,
                       eval_metric="logloss")
mC.fit(Xtr, ytr)
res["baseline_iter0"] = score(mC.predict_proba(Xte)[:,1])
print(f"[done] baseline ({time.time()-t0:.0f}s)", flush=True)

json.dump(res, open("/tmp/test_commit_results.json","w"), indent=1)
hdr = f"{'model':16}{'brier':>8}{'auc':>8}" + "".join(f"{'rp'+str(k):>8}" for k in (1,3,5,10,20))
print("\n=== ONE-SHOT TEST (2024-07-26..2024-12-16) ===")
print(hdr)
for m in ("baseline_iter0","control_rand35","stratified"):
    r = res[m]
    print(f"{m:16}{r['brier']:>8.4f}{r['auc']:>8.4f}" +
          "".join(f"{r['rp'+str(k)]:>8.4f}" for k in (1,3,5,10,20)))
