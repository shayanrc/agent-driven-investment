"""One-shot: re-score the regenerated cell-5 revalidation model on the memo's test
window [2025-06-05, 2026-03-12] so we get apples-to-apples R-Precision@K vs
the canonical CSV row (which was computed on a later snapshot).

Writes:
  results/gbdt/experiments/<run>/predictions/test_memo_window.csv
  results/gbdt/experiments/<run>/rescore_memo_window_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gbdt import data as gbdt_data, features as gbdt_features, targets as gbdt_targets
from gbdt.model import XGBoostModel

RUN = Path("results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation")
SNAPSHOT_END = "2026-05-22"
TEST_START = pd.Timestamp("2025-06-05")
TEST_END = pd.Timestamp("2026-03-12")

hp = yaml.safe_load((RUN / "hp.yaml").read_text())["hp"]
kept_features = yaml.safe_load((RUN / "features.yaml").read_text())["features"]
print(f"[load] hp={hp}")
print(f"[load] n_kept_features={len(kept_features)}")

print(f"[panel] loading nasdaq100 panel up to {SNAPSHOT_END}")
panel_obj = gbdt_data.load_panel(
    "nasdaq100", start=None, end=SNAPSHOT_END,
    min_rows=1600, cache_only=True,
)
print(f"[panel] rows={len(panel_obj.panel)} tickers_kept={len(panel_obj.tickers_kept)}")

print("[features] building feature matrix (default 279 families)")
X = gbdt_features.build_feature_matrix(
    panel_obj.panel, panel_obj.index_series,
    annualization=panel_obj.annualization_factor,
)
X = X.dropna(axis=1, how="all")
print(f"[features] shape={X.shape}")

print("[target] building target up +10% / 50d / dd 5%")
y = gbdt_targets.build_target(
    panel_obj.panel, direction="up", threshold_pct=10,
    horizon_days=50, max_drawdown=0.05,
)
print(f"[target] rows={len(y)} positive_prevalence={y.dropna().mean():.4f}")

print(f"[fs] selecting {len(kept_features)} kept features")
missing = [f for f in kept_features if f not in X.columns]
if missing:
    raise RuntimeError(f"missing features in rebuilt matrix: {missing[:5]}")
X_kept = X[kept_features]

print(f"[model] loading {RUN}/model.ubj")
model = XGBoostModel.load(RUN / "model.ubj", hp=hp, feature_names=kept_features)

print("[score] running predict_proba")
p_raw = model.predict_proba(X_kept)
p_calib = p_raw  # calibration decision was 'native' on this artifact

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

mask = (preds["date"] >= TEST_START) & (preds["date"] <= TEST_END) & preds["y_true"].notna()
preds_win = preds[mask].copy()
preds_win["y_true"] = preds_win["y_true"].astype(int)
print(f"[filter] memo-window rows={len(preds_win)}  "
      f"date_range=[{preds_win['date'].min().date()}, {preds_win['date'].max().date()}]  "
      f"Q_days={preds_win['date'].nunique()}  base_rate={preds_win['y_true'].mean():.4f}")

out_csv = RUN / "predictions" / "test_memo_window.csv"
preds_win.to_csv(out_csv, index=False)
print(f"[write] {out_csv}")

print("\n[r_precision] run `uv run python scripts/gbdt/compute_r_precision.py "
      f"{out_csv} --json > {RUN}/rescore_memo_window_summary.json`")
