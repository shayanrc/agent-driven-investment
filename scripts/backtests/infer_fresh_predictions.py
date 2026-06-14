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

import glob
import json

from gbdt import features as gbdt_features
from gbdt.data import load_panel
from gbdt.model import XGBoostModel

VALIDATION_TOL = 1e-4
CACHE_DIR = "data/gbdt_feature_cache"


def _training_panel_index(universe: str, test_keys: set) -> tuple[pd.Index, pd.Timestamp] | None:
    """Find the cell's training panel row-set from the universe feature cache.

    A provider gap-fill during a cache refresh can ADD historical (date,ticker)
    rows the model never trained on. For path-dependent features (stateful
    F16 ``*_outside_band`` running counts, cross-sectional ranks/z-scores), one
    inserted row perturbs every downstream value → inference diverges from the
    model's stored predictions (the `_006` ndx40 abort: a single backfilled
    AZN 2026-02-09 bar shifted p_raw by 3.3e-2).

    The cell's cached universe feature-matrix index IS the training panel row
    set (``build_feature_matrix`` preserves the panel index). We locate the
    matching cache parquet (same universe, index ⊇ the cell's test rows) and
    return its (date,ticker) index + max date, so the caller can drop gap-fill
    rows and reproduce the training panel exactly. Returns None if not found
    (caller falls back to the raw panel + the strict self-check as the guard).
    """
    cands = []
    for kf in glob.glob(f"{CACHE_DIR}/*.key.json"):
        try:
            pl = json.load(open(kf)).get("payload", {})
            if pl.get("universe") != universe:
                continue
            pf = kf.replace(".key.json", ".parquet")
            idx = pd.read_parquet(pf, columns=[]).index  # index-only read
            dates = pd.to_datetime(idx.get_level_values("date"))
            keys = set(zip(dates, idx.get_level_values("ticker")))
            if test_keys.issubset(keys):
                cands.append((dates.max(), idx, keys))
        except Exception:
            continue
    if not cands:
        return None
    # Smallest snapshot that still covers the test rows = the training matrix
    # (not a later regen built on a longer panel).
    snap, idx, _ = min(cands, key=lambda c: c[0])
    return idx, snap


def build_scores(cell: Path, end: str, *, align_panel: bool = True) -> pd.DataFrame:
    """Return a (date,ticker)-indexed frame with column p_raw over [.., end]."""
    feats = yaml.safe_load((cell / "features.yaml").read_text())["features"]
    hp = yaml.safe_load((cell / "hp.yaml").read_text())["hp"]
    universe = yaml.safe_load((cell / "spec.yaml").read_text())["target"]["universe"]

    # Full-history panel so the 200-day rolling features are warm; cache_only
    # (the refresh already populated it).
    panel_obj = load_panel(universe, start="1990-01-01", end=end, cache_only=True)
    panel = panel_obj.panel

    # Align the panel to the model's training row-set: drop provider gap-fill
    # rows added after training (≤ snapshot) that the model never saw. Fresh
    # rows (> snapshot) are kept — they are the genuine OOS extension.
    if align_panel:
        test = pd.read_csv(cell / "predictions" / "test.csv", parse_dates=["date"])
        test_keys = set(zip(test["date"], test["ticker"]))
        found = _training_panel_index(universe, test_keys)
        if found is not None:
            train_idx, snap = found
            train_keys = set(zip(pd.to_datetime(train_idx.get_level_values("date")),
                                 train_idx.get_level_values("ticker")))
            pdates = panel.index.get_level_values("date")
            ptk = panel.index.get_level_values("ticker")
            # gap-fill rows: in the panel, dated ≤ snapshot, absent from training
            drop = [(d <= snap) and ((d, t) not in train_keys)
                    for d, t in zip(pdates, ptk)]
            n_drop = sum(drop)
            if n_drop:
                gap = sorted({(str(d.date()), t) for d, t, dr
                              in zip(pdates, ptk, drop) if dr})
                print(f"[align] dropping {n_drop} gap-fill row(s) the model never trained on: "
                      f"{gap[:5]}{'...' if len(gap) > 5 else ''}")
                panel = panel[[not d for d in drop]]
        else:
            print("[align] no training feature-matrix in cache; using raw panel "
                  "(strict self-check still guards faithfulness)")

    X = gbdt_features.build_feature_matrix(
        panel, panel_obj.index_series, annualization=panel_obj.annualization_factor,
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
    ap.add_argument("--no-align", action="store_true",
                    help="skip panel-alignment (don't drop post-training gap-fill rows)")
    args = ap.parse_args()
    cell = Path(args.cell)
    universe = yaml.safe_load((cell / "spec.yaml").read_text())["target"]["universe"]

    print(f"[infer] building features + scoring {universe} through {args.end} ...")
    scores = build_scores(cell, args.end, align_panel=not args.no_align)
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
