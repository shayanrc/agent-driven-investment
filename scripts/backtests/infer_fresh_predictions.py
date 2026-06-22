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
from gbdt.model import CatBoostModel, XGBoostModel

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


def _align_panel(panel: pd.DataFrame, cell: Path, universe: str) -> pd.DataFrame:
    """Drop provider gap-fill rows the model never trained on (≤ snapshot, absent
    from the training row-set); keep fresh rows (> snapshot, the genuine OOS
    extension). One inserted bar perturbs path-dependent / cross-sectional
    features, so this reproduces the training panel exactly. No-op fallback when
    the training feature-matrix isn't cached (the strict self-check still guards).
    """
    test = pd.read_csv(cell / "predictions" / "test.csv", parse_dates=["date"])
    test_keys = set(zip(test["date"], test["ticker"]))
    found = _training_panel_index(universe, test_keys)
    if found is None:
        print("[align] no training feature-matrix in cache; using raw panel "
              "(strict self-check still guards faithfulness)")
        return panel
    train_idx, snap = found
    train_keys = set(zip(pd.to_datetime(train_idx.get_level_values("date")),
                         train_idx.get_level_values("ticker")))
    pdates = panel.index.get_level_values("date")
    ptk = panel.index.get_level_values("ticker")
    drop = [(d <= snap) and ((d, t) not in train_keys) for d, t in zip(pdates, ptk)]
    n_drop = sum(drop)
    if n_drop:
        gap = sorted({(str(d.date()), t) for d, t, dr in zip(pdates, ptk, drop) if dr})
        print(f"[align] dropping {n_drop} gap-fill row(s) the model never trained on: "
              f"{gap[:5]}{'...' if len(gap) > 5 else ''}")
        panel = panel[[not d for d in drop]]
    return panel


def _build_one(cell: Path, end: str, *, align_panel: bool, warmup_start: str,
               panel_cache: dict, feat_cache: dict) -> pd.DataFrame:
    """Score one cell, reusing a shared panel-load + feature-build across cells.

    ``panel_cache`` is keyed by ``(universe, warmup_start)`` (the expensive
    ``load_panel``); ``feat_cache`` by ``(universe, warmup_start, aligned-index
    hash)`` (the expensive ``build_feature_matrix``). Two cells of the same
    universe + same trailing slice + same alignment therefore share ONE feature
    build (the daily cadence's two sp500 champions). The aligned-index hash makes
    sharing exact: any difference in the aligned row-set → a distinct build. The
    per-cell ``feats`` subset, model, and self-check stay independent, so each
    cell's result is byte-identical to building it alone.
    """
    feats = yaml.safe_load((cell / "features.yaml").read_text())["features"]
    hp = yaml.safe_load((cell / "hp.yaml").read_text())["hp"]
    universe = yaml.safe_load((cell / "spec.yaml").read_text())["target"]["universe"]

    # Panel from warmup_start so the 200-day rolling features are warm; cache_only
    # (the refresh already populated it). Shared across same-(universe, slice) cells.
    pk = (universe, warmup_start)
    if pk not in panel_cache:
        panel_cache[pk] = load_panel(universe, start=warmup_start, end=end, cache_only=True)
    panel_obj = panel_cache[pk]
    panel = panel_obj.panel
    if align_panel:
        panel = _align_panel(panel, cell, universe)

    # Feature build keyed by the ALIGNED row-set — identical alignment ⇒ one build.
    fk = (universe, warmup_start, int(pd.util.hash_pandas_object(panel.index, index=False).sum()))
    if fk not in feat_cache:
        feat_cache[fk] = gbdt_features.build_feature_matrix(
            panel, panel_obj.index_series, annualization=panel_obj.annualization_factor,
        ).dropna(axis=1, how="all")
    X = feat_cache[fk]

    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise RuntimeError(f"features.yaml columns absent from build: {missing}")
    Xc = X[feats]  # exact saved order (per-cell subset of the shared build)

    # Dispatch on the saved model file: XGBoost (.ubj) or CatBoost (.cbm). Loading a
    # .cbm through XGBoostModel segfaults (XGBoosterLoadModel on a non-XGBoost blob),
    # so pick the right loader. XGBoost path is unchanged (daily-predictions champions).
    if (cell / "model.ubj").exists():
        model = XGBoostModel.load(cell / "model.ubj", hp=hp, feature_names=feats)
    elif (cell / "model.cbm").exists():
        model = CatBoostModel.load(cell / "model.cbm", hp=hp, feature_names=feats)
    else:
        raise RuntimeError(f"no model.ubj / model.cbm in {cell}")
    p_raw = model.predict_proba(Xc)

    out = Xc.index.to_frame(index=False)  # date, ticker
    out["p_raw"] = np.asarray(p_raw).ravel()
    return out


def build_scores(cell: Path, end: str, *, align_panel: bool = True,
                 warmup_start: str = "1990-01-01") -> pd.DataFrame:
    """Return a (date,ticker)-indexed frame with column p_raw over [.., end].

    ``warmup_start`` bounds how far back the panel is loaded. The default
    (1990) builds the full history. For the daily/incremental path
    (``--since``) it is set to a trailing slice start (~7y before the target
    date) so the build is ~5x cheaper. The slice must be >= the ``min_rows``
    eligibility floor (1600 trading days), because that filter defines the
    cross-sectional ticker set — a shorter slice would admit a different set of
    tickers, shift the cross-sectional ranks, and diverge from the full build.
    At >= 1600 td + the same ``min_rows=1600``, a ticker is kept iff it has
    >= 1600 rows (identical to the full build), and because the slice still
    covers the model's test window, the self-check proves faithfulness
    (insufficient warmup or a changed ticker set makes the test reproduction
    diverge and abort).

    Single-cell wrapper around ``_build_one`` with private (unshared) caches —
    behavior is unchanged from the original implementation.
    """
    return _build_one(cell, end, align_panel=align_panel, warmup_start=warmup_start,
                      panel_cache={}, feat_cache={})


def build_scores_multi(specs: list[tuple[Path, str]], end: str, *,
                       align_panel: bool = True) -> dict[str, pd.DataFrame]:
    """Score many cells, sharing the panel-load + feature-build where possible.

    ``specs`` is a list of ``(cell_path, warmup_start)``. Returns
    ``{str(cell_path): scores_df}``. Cells of the same universe + warmup_start +
    alignment share one ``build_feature_matrix`` (the cadence's two sp500
    champions ⇒ one build instead of two). Each cell's frame is identical to
    ``build_scores()`` called on it alone; callers still run the per-cell
    self-check, so a mis-share can only abort loudly, never produce bad data.
    """
    panel_cache: dict = {}
    feat_cache: dict = {}
    return {
        str(cell): _build_one(cell, end, align_panel=align_panel,
                              warmup_start=warmup_start,
                              panel_cache=panel_cache, feat_cache=feat_cache)
        for cell, warmup_start in specs
    }


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


def self_check(scores: pd.DataFrame, cell: Path, *, incremental: bool,
               label: str = "") -> None:
    """Reproduce predictions/test.csv and abort (SystemExit) if p_raw diverges
    beyond VALIDATION_TOL. In incremental mode a no-overlap (slice predates the
    test window) is a warn-and-proceed, not an abort — faithfulness was proven on
    the initial full run. Shared by main() and the daily cadence so both gate on
    the identical faithfulness check."""
    pre = f"[{label}] " if label else ""
    print(f"{pre}[validate] reproducing predictions/test.csv ...")
    try:
        v = validate_against_test(scores, cell)
        print(f"{pre}          n_overlap={v['n_overlap']} max_abs_diff={v['max_abs_diff']:.2e} "
              f"mean_abs_diff={v['mean_abs_diff']:.2e}")
        if v["max_abs_diff"] > VALIDATION_TOL:
            raise SystemExit(
                f"[ABORT] reproduced p_raw diverges from test.csv "
                f"(max_abs_diff={v['max_abs_diff']:.2e} > {VALIDATION_TOL}). "
                "Feature build or model load is not faithful; not emitting fresh scores."
            )
        print(f"{pre}          self-check PASSED — inference path is faithful.")
    except RuntimeError as exc:
        if not incremental:
            raise
        print(f"{pre}          [warn] self-check skipped ({exc}); slice predates "
              "the test window. Faithfulness was validated on the initial full run.")


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
    ap.add_argument("--since", default=None,
                    help="INCREMENTAL mode (daily/backfill): build features from a "
                         "trailing ~7y slice ending at --end instead of full history, "
                         "and emit predictions for dates strictly after --since. ~5x "
                         "faster; the slice is kept >= the 1600-row eligibility floor "
                         "so the ticker set (and cross-sectional features) match the "
                         "full build, and it still covers the test window so the "
                         "self-check guards faithfulness. Sets the output cutoff "
                         "unless --fresh-after is given.")
    args = ap.parse_args()
    cell = Path(args.cell)
    universe = yaml.safe_load((cell / "spec.yaml").read_text())["target"]["universe"]

    # Incremental: load only a trailing slice (~2y) before --since so the build
    # is cheap. WARMUP_DAYS must exceed the longest feature lookback (200 td) AND
    # reach back to cover the cell's test window so the self-check is meaningful.
    warmup_start = "1990-01-01"
    if args.since is not None:
        # ~7y trailing slice: comfortably exceeds the 1600-td eligibility floor
        # (so the kept-ticker set matches the full build) while skipping the
        # deep history that contributes nothing to ≤200d rolling features.
        warmup_start = str((pd.Timestamp(args.since) - pd.Timedelta(days=2700)).date())

    mode = f"incremental (slice from {warmup_start})" if args.since else "full history"
    print(f"[infer] building features + scoring {universe} through {args.end} "
          f"[{mode}] ...")
    scores = build_scores(cell, args.end, align_panel=not args.no_align,
                          warmup_start=warmup_start)
    scores["date"] = pd.to_datetime(scores["date"])

    self_check(scores, cell, incremental=args.since is not None)

    test = pd.read_csv(cell / "predictions" / "test.csv", parse_dates=["date"])
    cutoff = (pd.Timestamp(args.fresh_after) if args.fresh_after
              else pd.Timestamp(args.since) if args.since
              else test["date"].max())
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
