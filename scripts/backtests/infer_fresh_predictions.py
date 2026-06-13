"""Score the refreshed panel with an already-trained gbdt model (inference only).

No retraining: load a cell's saved XGBoost booster + selected feature list,
rebuild the causal feature matrix on the now-extended OHLCV panel, and emit
predictions for the dates AFTER the cell's published test window — a genuinely
fresh out-of-sample set the model never saw.

Correctness:
  * Features are causal rolling stats (C1), so scoring later dates introduces
    no look-ahead.
  * The build call mirrors gbdt/__main__.py:1955 exactly
    (build_feature_matrix(...).dropna(axis=1, how="all")), then subset to the
    cell's features.yaml in saved order.
  * Self-check: on the overlap with the cell's predictions/test.csv, the
    reproduced p_raw must match the artifact to < 1e-4, or we abort (the
    feature build / model load diverged and the fresh scores can't be trusted).

    uv run python -m scripts.backtests.infer_fresh_predictions \
        --cell <artifact_dir> --out <fresh_predictions.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gbdt import features as gbdt_features
from gbdt.data import load_panel
from gbdt.model import XGBoostModel

UNIVERSE = "nasdaq100"
VALIDATION_TOL = 1e-4


def build_scores(cell: Path, end: str) -> pd.DataFrame:
    """Return a (date,ticker)-indexed frame with column p_raw over [.., end]."""
    feats = yaml.safe_load((cell / "features.yaml").read_text())["features"]
    hp = yaml.safe_load((cell / "hp.yaml").read_text())["hp"]

    # Full-history panel so the 200-day rolling features are warm; cache_only
    # (the refresh already populated it).
    panel_obj = load_panel(UNIVERSE, start="1990-01-01", end=end, cache_only=True)
    X = gbdt_features.build_feature_matrix(
        panel_obj.panel,
        panel_obj.index_series,
        annualization=panel_obj.annualization_factor,
    ).dropna(axis=1, how="all")

    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise RuntimeError(f"features.yaml columns absent from build: {missing}")
    X = X[feats]  # exact saved order

    model = XGBoostModel.load(cell / "model.ubj", hp=hp, feature_names=feats)
    p_raw = model.predict_proba(X)

    out = X.index.to_frame(index=False)  # date, ticker
    out["p_raw"] = np.asarray(p_raw).ravel()
    return out


def validate_against_test(scores: pd.DataFrame, cell: Path) -> dict:
    """Self-check: reproduced p_raw must match predictions/test.csv on overlap."""
    test = pd.read_csv(cell / "predictions" / "test.csv", parse_dates=["date"])
    m = test.merge(
        scores.assign(date=pd.to_datetime(scores["date"])),
        on=["date", "ticker"], suffixes=("_orig", "_repro"),
    )
    if m.empty:
        raise RuntimeError("no overlap between reproduced scores and test.csv")
    diff = (m["p_raw_orig"] - m["p_raw_repro"]).abs()
    return {"n_overlap": len(m), "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--end", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--fresh-after", default=None,
                    help="emit predictions strictly after this date "
                         "(default: the cell's test.csv max date)")
    args = ap.parse_args()
    cell = Path(args.cell)

    print(f"[infer] building features + scoring {UNIVERSE} through {args.end} ...")
    scores = build_scores(cell, args.end)
    scores["date"] = pd.to_datetime(scores["date"])

    print("[validate] reproducing predictions/test.csv ...")
    v = validate_against_test(scores, cell)
    print(f"          n_overlap={v['n_overlap']} max_abs_diff={v['max_abs_diff']:.2e} "
          f"mean_abs_diff={v['mean_abs_diff']:.2e}")
    if v["max_abs_diff"] > VALIDATION_TOL:
        raise SystemExit(
            f"[ABORT] reproduced p_raw diverges from test.csv "
            f"(max_abs_diff={v['max_abs_diff']:.2e} > {VALIDATION_TOL}). "
            "Feature build or model load is not faithful; not emitting fresh scores."
        )
    print("          self-check PASSED — inference path is faithful.")

    test = pd.read_csv(cell / "predictions" / "test.csv", parse_dates=["date"])
    cutoff = pd.Timestamp(args.fresh_after) if args.fresh_after else test["date"].max()
    fresh = scores[scores["date"] > cutoff].copy()
    # Native isotonic pass-through on this cell → p_calibrated == p_raw. (The
    # backtest's Bayesian recalibrator is fit on the cell's val regardless.)
    fresh["p_calibrated"] = fresh["p_raw"]
    fresh = fresh.sort_values(["date", "ticker"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(args.out, index=False)
    print(f"[out] fresh predictions after {cutoff.date()}: "
          f"{len(fresh)} rows, {fresh['date'].nunique()} dates "
          f"[{fresh['date'].min().date()} .. {fresh['date'].max().date()}], "
          f"{fresh['ticker'].nunique()} tickers → {args.out}")


if __name__ == "__main__":
    main()
