# _030 — Fundamentals-aware daily inference + first F18 forward candidate

**What.** Extend the `/daily-predictions` cadence (`_019`) so it can score **F18/F19
fundamentals cells**, and add the top fundamental-feature model
(`sp500_up_40pct_200d_dd20pct_w2ffundtune`) as a `deployed=False` forward-comparison
candidate. Prior to this, every tracked cell was technical-only (`all` token); the
inference path could not build `fund_*` columns at all.

## The blocker

The daily cadence scores via `infer_fresh_predictions._build_one` →
`incremental_feature_cache.build_or_extend`, which calls
`build_feature_matrix(panel, index_df, annualization=…)` — technical-only
(`families="all"`, `fund_df=None`). Only the **training** path
(`gbdt/experiment_runner.py`) ever loads the point-in-time valuation panel and passes
`fund_df` + `all_fundamentals`. So dropping a fundamentals cell into `CELLS` would make
the shared build produce a technical-only matrix and abort the whole cadence with
`RuntimeError: features.yaml columns absent from build: [fund_earnings_yield, …]`.

It was a **code gap, not a data gap**: the valuation panel
(`results/valuation/data/valuation_panel.parquet`) and the `us_fundamentals` cache were
already current (fiscal_period_end 2026-05-31).

## The fix — a fundamentals-aware build branch

In `infer_fresh_predictions._build_one`, read the cell's `features.candidates` token and
branch:

- **technical cells** (`all`) — unchanged, ride the fast on-disk incremental cache
  (byte-identical; the champions are untouched).
- **fundamentals cells** (`all_fundamentals` / `all_fundamentals2`, in
  `FUNDAMENTALS_TOKENS`) — load `load_fundamentals_panel(…)` and build the `fund_*`
  columns **directly** via `build_feature_matrix(panel, index_df, annualization=…,
  families=token, fund_df=fund_df).dropna(axis=1, how="all")`, mirroring the training
  build exactly.

The shared `feat_cache` key gained the **feature token** so a fundamentals cell and the
technical champions of the same universe never collide on one build.

**Correctness is proven by the existing self-check.** On w2ffundtune the fresh F18 build
reproduced the cell's `predictions/test.csv` to **`max_abs_diff = 2.97e-08`** over 48,600
overlapping rows — ~3,000× inside the `1e-4` faithfulness contract. If the build had
diverged (wrong lookbacks / exclude / fund join), it would have aborted loudly.

## The candidate

`sp500_up_40pct_200d_dd20pct_w2ffundtune` — F18 (`all_fundamentals`), xgboost, sp500
+40%/200d, test 2025-01-24 → 2025-06-17. Chosen as the **best fundamental-feature model
by R-Precision@3** in the `_279`/`_280` arc (R-p@3 0.603, AUC 0.727, base 0.221 — strong
across the whole book, and a genuine AUC on a real base, not the rare-event inflation of
the ultra-rare 50% cells that top the by-AUC list). Added to `CELLS` as
`deployed=False`, `backfill_from: 2026-01-01` (test_end 2025-06-17 → January 2026 is
genuine OOS).

**This is forward-comparison tracking, NOT a promotion.** Per `_279`/`_280` the F18/F19
fundamentals edge did **not** replicate across windows; the daily slot exists precisely to
watch whether the window-specific edge holds out-of-sample. A champion swap remains a
separate human decision.

## Costs / caveats

- **Fund cells full-build each run** (~7y warmup slice, sp500 only ~2–3 min); there is no
  incremental fund cache yet. Parked as a V1.8 optimization (`docs/gbdt/V1.8_TBD.md`).
- **Freshness dependency:** the daily seed refreshes only `us_equities`. For a
  fundamentals candidate to stay current, the valuation panel must be rebuilt separately
  (`scripts.valuation.build_valuation_panel`) as the `us_fundamentals` cache grows.
  Because F18 features are point-in-time on `filed_date` and change only ~quarterly, a
  weekly rebuild is sufficient; the self-check will abort if the panel is too stale to
  reproduce the test window.
- 200-day horizon + test_end 2025-06-17 → ~13-month first backfill (fine for a weekly
  `deployed=False` comparison series).

## Artifacts

Code: `scripts/backtests/infer_fresh_predictions.py` (`FUNDAMENTALS_TOKENS` + the
`_build_one` branch), `scripts/backtests/daily_forward_predictions.py::CELLS`
(`sp500_40_200`). Forward log: `results/backtests/data/forward_predictions_log.csv`.
Related: `_019` (the cadence), `docs/gbdt/_279`/`_280` (the fundamentals arc + why
w2ffundtune is the pick).
