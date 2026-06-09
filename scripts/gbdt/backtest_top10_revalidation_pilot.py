"""Top-10 OOS back-test — PILOT on cell-5 (nasdaq100 up+10% / 50d / dd5%, V1.3
revalidation). Loads the saved model, builds features through the cache's
right edge, scores on the true OOS window (test_end + 1 BD → right_edge - 50 BD,
so every pick has a realized 50-BD label), and writes:

  results/gbdt/backtest/<cell>/predictions_oos.csv  (date, ticker, p_raw, p_calibrated, y_true, sample_weight)
  results/gbdt/backtest/<cell>/picks_top_k.csv      (K, rank, date, ticker, p_calibrated, y_true, fwd_return, fwd_max_dd)
  results/gbdt/backtest/<cell>/summary.json         (canonical R-p@K via scripts/gbdt/compute_r_precision.py)

Calibration is NATIVE pass-through on this artifact (regen Spiegelhalter z within
the conditional_isotonic native band — see calibration.pkl). For other cells we
will read calibration.pkl and apply.

Cache state caveat: data/raw symlink is currently broken (self-loop); this run
uses cache_only=True with the processed.db right edge (~2026-05-22), so OOS ends
~2026-03-13 instead of today−50BD. Symlink fix is a separate task.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gbdt import data as gbdt_data, features as gbdt_features, targets as gbdt_targets
from gbdt.model import XGBoostModel

CELL = "nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation"
RUN = Path("results/gbdt/experiments") / CELL
OUT = Path("results/gbdt/backtest") / CELL
OUT.mkdir(parents=True, exist_ok=True)

# True OOS boundary — regen's per-ticker trailing test slice end (from metrics.json
# segment_dates.test.end). Anything STRICTLY GREATER than this date is unseen.
TEST_END = pd.Timestamp("2025-12-26")
HORIZON_BD = 50  # cell-5 horizon in business days

# Load artifact metadata
hp = yaml.safe_load((RUN / "hp.yaml").read_text())["hp"]
kept_features = yaml.safe_load((RUN / "features.yaml").read_text())["features"]
print(f"[load] hp={hp}")
print(f"[load] n_kept_features={len(kept_features)}")

# Load panel up to cache right edge (cache_only=True; raw fetch is broken).
# Passing end=None lets the loader use the per-ticker max(date) from the cache.
print(f"[panel] loading nasdaq100 panel (cache_only=True, end=cache right edge)")
panel_obj = gbdt_data.load_panel(
    "nasdaq100", start=None, end=None,
    min_rows=1600, cache_only=True, staleness_days=60,  # loosened because cache stops 2026-05-22
)
panel_right_edge = panel_obj.panel.index.get_level_values("date").max()
print(f"[panel] rows={len(panel_obj.panel)} tickers_kept={len(panel_obj.tickers_kept)} "
      f"right_edge={panel_right_edge.date()}")

print("[features] building feature matrix (default 279 families)")
X = gbdt_features.build_feature_matrix(
    panel_obj.panel, panel_obj.index_series,
    annualization=panel_obj.annualization_factor,
)
X = X.dropna(axis=1, how="all")
print(f"[features] shape={X.shape}")

print(f"[target] building target up +10% / {HORIZON_BD}d / dd 5%")
y = gbdt_targets.build_target(
    panel_obj.panel, direction="up", threshold_pct=10,
    horizon_days=HORIZON_BD, max_drawdown=0.05,
)
print(f"[target] rows={len(y)} positive_prevalence={y.dropna().mean():.4f}")

# Select kept features
missing = [f for f in kept_features if f not in X.columns]
if missing:
    raise RuntimeError(f"missing features in rebuilt matrix: {missing[:5]}")
X_kept = X[kept_features]

# Load model
print(f"[model] loading {RUN}/model.ubj")
model = XGBoostModel.load(RUN / "model.ubj", hp=hp, feature_names=kept_features)

print("[score] running predict_proba on full panel")
p_raw = model.predict_proba(X_kept)
p_calib = p_raw  # native pass-through (Spiegelhalter |z| < 2.0 on this artifact)

dates = X_kept.index.get_level_values("date")
tickers = X_kept.index.get_level_values("ticker")
preds = pd.DataFrame({
    "date": pd.to_datetime(dates),
    "ticker": tickers,
    "p_raw": p_raw,
    "p_calibrated": p_calib,
    "y_true": y.reindex(X_kept.index).values,
    "sample_weight": 1.0,
})
print(f"[score] all rows: {len(preds)}")

# OOS filter: strictly after TEST_END, with realized label available (y_true not NaN).
# The y_true=NaN check implicitly enforces "forward window must be fully observable
# in the cache" — gbdt_targets.build_target only emits labels where the H-BD forward
# window is complete.
oos_mask = (preds["date"] > TEST_END) & preds["y_true"].notna()
oos = preds[oos_mask].copy()
oos["y_true"] = oos["y_true"].astype(int)
print(f"[filter] OOS rows={len(oos)}  "
      f"date_range=[{oos['date'].min().date()}, {oos['date'].max().date()}]  "
      f"Q_days={oos['date'].nunique()}  base_rate={oos['y_true'].mean():.4f}")

# Write canonical predictions CSV (consumed by compute_r_precision.py)
oos_csv = OUT / "predictions_oos.csv"
oos.to_csv(oos_csv, index=False)
print(f"[write] {oos_csv}")

# Compute realized forward close-to-close return + max drawdown per (pick_date, ticker).
# Used for human interpretation of top-K picks (not for the R-p@K metric).
print("[fwd] computing forward returns + max drawdown for OOS rows")
panel = panel_obj.panel.reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)

fwd_records = []
for tkr, g in oos.groupby("ticker"):
    tpanel = panel[panel["ticker"] == tkr].reset_index(drop=True)
    if tpanel.empty:
        continue
    tpanel = tpanel.set_index("date")
    pick_dates = g["date"].tolist()
    for pick_date in pick_dates:
        if pick_date not in tpanel.index:
            continue
        i = tpanel.index.get_loc(pick_date)
        if i + HORIZON_BD >= len(tpanel):
            continue
        c0 = tpanel.iloc[i]["close"]
        fwd_window = tpanel.iloc[i + 1 : i + 1 + HORIZON_BD]
        end_close = fwd_window.iloc[-1]["close"]
        max_high = fwd_window["high"].max()
        min_low = fwd_window["low"].min()
        running_max = fwd_window["high"].cummax()
        dd = (fwd_window["low"] / running_max - 1.0).min()  # most negative low-vs-running-max
        fwd_records.append({
            "date": pick_date, "ticker": tkr,
            "fwd_close_return": float(end_close / c0 - 1.0),
            "fwd_high_return":  float(max_high / c0 - 1.0),
            "fwd_low_return":   float(min_low / c0 - 1.0),
            "fwd_max_dd":       float(dd),
        })
fwd_df = pd.DataFrame(fwd_records)
print(f"[fwd] rows={len(fwd_df)}")

# Build top-K picks per day for K ∈ {1, 3, 5, 10}
print("[topk] building top-K picks per day")
pick_frames = []
for K in [1, 3, 5, 10]:
    daily = (
        oos.sort_values(["date", "p_calibrated", "ticker"], ascending=[True, False, True], kind="mergesort")
           .groupby("date").head(K)
           .copy()
    )
    daily["K"] = K
    daily["rank"] = daily.groupby("date").cumcount() + 1
    pick_frames.append(daily)
picks_df = pd.concat(pick_frames, ignore_index=True)
picks_df = picks_df.merge(fwd_df, on=["date", "ticker"], how="left")
picks_csv = OUT / "picks_top_k.csv"
picks_df.to_csv(picks_csv, index=False)
print(f"[write] {picks_csv}")

# Canonical R-p@K via compute_r_precision.py --json
print("[r_precision] invoking compute_r_precision.py")
summary_path = OUT / "summary.json"
result = subprocess.run(
    ["uv", "run", "python", "scripts/gbdt/compute_r_precision.py", str(oos_csv), "--json"],
    capture_output=True, text=True, check=True,
)
summary_path.write_text(result.stdout)
print(f"[write] {summary_path}")

# Print a quick verdict
summary = json.loads(result.stdout)
rpk = summary["r_precision_at_k"]["by_k"]
print()
print(f"=== OOS R-Precision@K (cell-5 revalidation, OOS=[{oos['date'].min().date()}, {oos['date'].max().date()}]) ===")
print(f"Q_days={summary['r_precision_at_k']['n_days_total']}  base_rate={summary['r_precision_at_k']['base_rate']:.4f}")
for K in ["1", "3", "5", "10", "20"]:
    if K in rpk:
        print(f"  K={K:>2}  R-p@K={rpk[K]['r_precision_at_k']:.4f}  "
              f"(hits={rpk[K]['n_hits']}/denom={rpk[K]['n_denom']})")

# Hit rates of top-K picks (denom = picks made, NOT min(K, R_q) — for human readability)
print()
print(f"=== OOS Top-K hit rates (raw, denom = picks made) ===")
for K in [1, 3, 5, 10]:
    sub = picks_df[picks_df["K"] == K]
    hr = sub["y_true"].mean()
    mean_fwd = sub["fwd_close_return"].mean()
    mean_dd = sub["fwd_max_dd"].mean()
    print(f"  K={K:>2}  hit_rate={hr:.4f}  mean_fwd_close_ret={mean_fwd:+.4f}  mean_fwd_max_dd={mean_dd:+.4f}  n_picks={len(sub)}")
