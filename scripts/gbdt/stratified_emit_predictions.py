"""Emit runner-schema prediction CSVs for the _284 stratified ensemble so the
standard backtest harness (run_backtest_cell --pred-csv) can place it on the
backtest leaderboard.

Windows:
  eval.csv  2023-10-09 -> 2024-07-25   (calibrator fit, same as runner layout)
  test.csv  2024-07-26 -> 2026-06-12   (test + FORWARD OOS, matching the
                                        board's long-window convention; rows
                                        after 2024-12-16 were never seen by
                                        any tuning decision)
y_true from the triple-barrier target where labelable (NaN horizon-truncated
rows kept with y_true empty)."""
import pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from gbdt.data import load_panel
from gbdt.targets import build_target

t0 = time.time()
RUN = "results/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune"
OUT = "/tmp/strat_preds"
import os; os.makedirs(OUT, exist_ok=True)

X = pd.read_parquet(f"{RUN}/_feature_matrix_cache.parquet")
panel = load_panel("sp500", end="2026-07-06").panel
y = build_target(panel, direction="up", threshold_pct=20, horizon_days=50,
                 max_drawdown=0.10).rename("target")
del panel
dates = X.index.get_level_values("date")
m = (dates >= "2023-10-09") & (dates <= "2026-06-12")
Xw = X[m]
del X
# feature completeness consistent with training harness: drop rows with any
# NaN features (matches the dropna() training convention)
Xw = Xw.dropna()
feat_cols = list(Xw.columns)
print(f"[data] scoring window {len(Xw):,} rows ({time.time()-t0:.0f}s)", flush=True)

mdl = pickle.load(open("/tmp/strat_model.pkl", "rb"))
colpos = {c: i for i, c in enumerate(feat_cols)}
Xnp = Xw.to_numpy(dtype=np.float32)
margin = np.full(len(Xw), mdl["m0"], dtype=np.float32)
for i, (bst, cols) in enumerate(zip(mdl["boosters"], mdl["cols_per_tree"])):
    idx = [colpos[c] for c in cols]
    d = xgb.DMatrix(Xnp[:, idx], base_margin=margin, feature_names=cols)
    margin = bst.predict(d, output_margin=True)
    if (i+1) % 200 == 0:
        print(f"[score] tree {i+1}/800 ({time.time()-t0:.0f}s)", flush=True)
p = 1/(1+np.exp(-margin))

out = pd.DataFrame({"date": Xw.index.get_level_values("date"),
                    "ticker": Xw.index.get_level_values("ticker"),
                    "p_raw": p})
yy = y.reindex(Xw.index)
out["y_true"] = yy.values
out["p_calibrated"] = out["p_raw"]      # harness refits its own calibrator from eval
out["sample_weight"] = 1.0
out = out[["date","ticker","p_raw","p_calibrated","y_true","sample_weight"]]

ev = out[(out.date >= "2023-10-09") & (out.date <= "2024-07-25")]
te = out[(out.date >= "2024-07-26") & (out.date <= "2026-06-12")]
ev.to_csv(f"{OUT}/eval.csv", index=False)
te.to_csv(f"{OUT}/test.csv", index=False)
print(f"[saved] eval {len(ev):,} rows | test(+fwd) {len(te):,} rows "
      f"{te.date.min().date()}->{te.date.max().date()} "
      f"| labelable {te.y_true.notna().mean():.1%} ({time.time()-t0:.0f}s)", flush=True)
