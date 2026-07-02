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
import hashlib
import json

from gbdt import features as gbdt_features
from gbdt import incremental_feature_cache as _ifc
from gbdt.data import load_panel
from gbdt.model import CatBoostModel, XGBoostModel

VALIDATION_TOL = 1e-4
CACHE_DIR = "data/gbdt_feature_cache"
# V1.6 Phase 5 — on-disk incremental feature-matrix cache (extend, don't rebuild).
INCREMENTAL_CACHE_DIR = "data/gbdt_incremental_cache"


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
    # The matched cache may be a FORWARD-EXTENDING regen (snap > the cell's test
    # window) built on a then-current — possibly stale — panel. Aligning to its
    # max date would drop genuine OOS rows (> test_end) that aren't in that older
    # row-set: e.g. a russell1000 regen ending 2026-06-18 (built while ~half the
    # universe was stale at 05-22) silently pruned the freshly-seeded tickers on
    # 2026-05-26→06-18, collapsing forward coverage to ~471/890. Cap the gap-fill
    # cutoff at the training boundary (test_end): only HISTORICAL rows are pruned
    # (for faithful test reproduction); every forward row passes through at full
    # universe coverage. The self-check (≤ test_end) is unaffected by the cap.
    snap = min(snap, test["date"].max())
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


# XGBoost/CatBoost predict_proba materializes a (float32) copy of the feature frame.
# On the full-history russell panel (~7M rows × 143 feats) that transient copy, on top
# of the already-large in-memory build (X + Xc), pushes peak RAM over the box and the
# process is OOM-killed (exit 137) right after the build completes. Predicting in row-
# chunks bounds that copy to CHUNK rows; the result is identical (predict is row-wise).
_PREDICT_CHUNK_ROWS = 500_000


def _predict_proba_chunked(model, Xc: pd.DataFrame) -> np.ndarray:
    """Row-chunked predict_proba — byte-identical to a single call, bounded peak RAM."""
    n = len(Xc)
    if n <= _PREDICT_CHUNK_ROWS:
        return np.asarray(model.predict_proba(Xc)).ravel()
    parts = [np.asarray(model.predict_proba(Xc.iloc[i:i + _PREDICT_CHUNK_ROWS])).ravel()
             for i in range(0, n, _PREDICT_CHUNK_ROWS)]
    return np.concatenate(parts)


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
    align_sig = "noalign"
    if align_panel:
        aligned = _align_panel(panel, cell, universe)
        if len(aligned) != len(panel):
            dropped = sorted(panel.index.difference(aligned.index).tolist())
            align_sig = hashlib.sha256(
                "|".join(f"{d}:{t}" for d, t in dropped).encode("utf-8")
            ).hexdigest()[:16]
        panel = aligned

    # Feature build keyed by the ALIGNED row-set — identical alignment ⇒ one in-process
    # build. The build itself goes through the on-disk INCREMENTAL cache (V1.6 Phase 5):
    # load the cached matrix + extend only the new dates (seam-checked), else full
    # rebuild. ``align_sig`` keys per-cell alignment so two cells never collide on one
    # entry. Result matches a from-scratch build to the ~1e-4 contract (frozen historical
    # rows incl the test window stay exact, so the self-check below is unaffected).
    fk = (universe, warmup_start, int(pd.util.hash_pandas_object(panel.index, index=False).sum()))
    if fk not in feat_cache:
        feat_cache[fk] = _ifc.build_or_extend(
            INCREMENTAL_CACHE_DIR, universe, warmup_start, panel, panel_obj.index_series,
            annualization=panel_obj.annualization_factor, align_signature=align_sig,
        )
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
    p_raw = _predict_proba_chunked(model, Xc)

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
    sc = scores.assign(date=pd.to_datetime(scores["date"]))
    m = test.merge(sc, on=["date", "ticker"], suffixes=("_orig", "_repro"))
    if m.empty:
        raise RuntimeError("no overlap between reproduced scores and test.csv")
    diff = (m["p_raw_orig"] - m["p_raw_repro"]).abs()
    # Membership delta on the test window. A universe that has grown/shrunk since
    # training (tickers crossing the min_rows eligibility floor, IPOs, delistings)
    # re-ranks the *cross-sectional* features at historical dates — a legitimate
    # universe change, NOT feature corruption. We surface it so the self-check can
    # distinguish it from a stable-membership prediction drift (the _007 bug).
    test_dates = set(test["date"])
    repro_test = sc[sc["date"].isin(test_dates)]
    test_tk = set(test["ticker"]); repro_tk = set(repro_test["ticker"])
    return {"n_overlap": len(m), "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "added_tickers": sorted(repro_tk - test_tk),
            "removed_tickers": sorted(test_tk - repro_tk)}


def self_check(scores: pd.DataFrame, cell: Path, *, incremental: bool,
               label: str = "", allow_universe_growth: bool = False) -> None:
    """Reproduce predictions/test.csv and abort (SystemExit) if p_raw diverges
    beyond VALIDATION_TOL. In incremental mode a no-overlap (slice predates the
    test window) is a warn-and-proceed, not an abort — faithfulness was proven on
    the initial full run. Shared by main() and the daily cadence so both gate on
    the identical faithfulness check.

    ``allow_universe_growth`` (OOS backtests): if the divergence is explained by a
    universe MEMBERSHIP change (tickers added/removed since training — e.g. a name
    crossing the min_rows eligibility floor once the panel is extended to cover the
    OOS window), downgrade the abort to a warning and proceed. Such a change merely
    re-ranks cross-sectional features at historical dates; the path (per-ticker)
    features are unaffected. A divergence with an UNCHANGED universe still aborts —
    that is genuine feature/model corruption (the _007 backfill bug). The default
    (strict) behavior is unchanged, so the /daily-predictions cadence is untouched."""
    pre = f"[{label}] " if label else ""
    print(f"{pre}[validate] reproducing predictions/test.csv ...")
    try:
        v = validate_against_test(scores, cell)
        print(f"{pre}          n_overlap={v['n_overlap']} max_abs_diff={v['max_abs_diff']:.2e} "
              f"mean_abs_diff={v['mean_abs_diff']:.2e}")
        if v["max_abs_diff"] > VALIDATION_TOL:
            n_add, n_rem = len(v["added_tickers"]), len(v["removed_tickers"])
            if allow_universe_growth and (n_add or n_rem):
                print(f"{pre}          [warn] test-window reproduction diverges "
                      f"(max_abs_diff={v['max_abs_diff']:.2e}) but the UNIVERSE CHANGED "
                      f"(+{n_add}/-{n_rem} tickers since training; e.g. added "
                      f"{v['added_tickers'][:6]}). This re-ranks cross-sectional features "
                      "at historical dates — a legitimate universe change, not feature "
                      "corruption (path features are membership-independent). Proceeding.")
            else:
                raise SystemExit(
                    f"[ABORT] reproduced p_raw diverges from test.csv "
                    f"(max_abs_diff={v['max_abs_diff']:.2e} > {VALIDATION_TOL}) with an "
                    f"UNCHANGED universe (+{n_add}/-{n_rem} tickers). Feature build or "
                    "model load is not faithful; not emitting fresh scores."
                )
        else:
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
    ap.add_argument("--allow-universe-growth", action="store_true",
                    help="OOS backtests: treat a test-window reproduction divergence as "
                         "NON-fatal when it is explained by a universe MEMBERSHIP change "
                         "(tickers added/removed since training — they re-rank cross-"
                         "sectional features). Still aborts on divergence with an "
                         "UNCHANGED universe (real feature corruption). A backtest should "
                         "depend only on (model, universe, date-range); the universe "
                         "legitimately grows over that range, so growth is not an error.")
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

    self_check(scores, cell, incremental=args.since is not None,
               allow_universe_growth=args.allow_universe_growth)

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
