# /canonical-finetune

One verb: **finetune a gbdt cell on the canonical evaluation periods** per
`docs/gbdt/CANONICAL_FINETUNE_RECIPE.md` — that doc is the **source of truth** for the
methodology (windows, role discipline, selection rules, mistake-guards); this skill is the
executable wrapper. Input: a cell = `(universe, direction, threshold, horizon, max_drawdown,
feature-token)`. Output: an adopted model (`<stem>_canon_ft` dir with predictions for
val/eval/test/backtest), a backtest read, an `hp/EXPLORATION.md` trail, and registry rows.

## Pre-flight

- Disk ≥ 10 G free (`df --output=avail .`), MemAvailable ≥ ~18 G before heavy fits (fits are
  5–15 GB each — **serialize them**, never two `final_fit` in parallel; recipe guard #6).
- Cache currency: the universe must be seeded through the snapshot (`SNAP` in
  `scripts/gbdt/canon_cells.py`, currently 2026-07-06). Long-horizon (200d) cells need
  prices ~10 months past `test_end` (recipe guard #7).
- Fundamentals tokens (`all_fundamentals*`) need the valuation panel present
  (`results/valuation/data/valuation_panel[_nse].parquet`); for nifty500 mind the F18-IN
  coverage cliff (`[[project-in-fundamentals-coverage-cliff]]` — resolved by the pre-2019
  backfill, verify NaN fraction if regenerating).

## Steps (each maps to a recipe § — read the recipe first)

1. **Register the cell** in `scripts/gbdt/canon_cells.py::CELLS` (id → universe/thr/hor/dd/
   token/stem). All canon tooling resolves through it.
2. **Base spec + build**: write `configs/gbdt/experiments/<stem>_canon_base.yaml` (canonical
   explicit-boundary split: train_start 2015-01-01 · val_start 2022-03-30 · eval_start
   2023-07-01 · test_start 2024-07-01 · test_end 2025-06-30 · min_rows_per_ticker 2591 ·
   `date_range.end` = SNAP; copy an existing one), then
   `uv run python -m gbdt experiment configs/gbdt/experiments/<stem>_canon_base.yaml
   --snapshot-end <SNAP> --overwrite`. Fast on a warm universe cache (~2 min).
3. **Controlled baseline** (the bar — NOT the runner's metrics.json, recipe guard #1):
   `CELL=<id> FEATS=all HP='{"max_depth":6,"min_child_weight":1,"subsample":1,
   "colsample_bytree":1,"gamma":0,"eta":0.05}' uv run python -m scripts.gbdt.final_fit_canon`.
   Record its full test R-p@K via `scripts.gbdt.compute_r_precision` on
   `<ft_dir>/predictions/test.csv`.
4. **Diagnose** (recipe §2): prevalence; **base test @1 headroom** (the real determinant —
   weak @1 below base_rate → deep+bagging likely adopts; high @1 ≥ ~0.45 → base likely
   stands); eval↔test agreement (it has inverted on EVERY canonical cell so far → **select
   on val, never eval**).
5. **HP sweep** (recipe §3): `CELL=<id> bash scripts/gbdt/canon_ft_sweep.sh` — the 16-config
   deep+bagging grid (depth{6,8,10} × ss{0.7,0.85} × cs{0.7,1.0} + mcw{5,10} variants) on
   `FEATS=all` via `hp_one_canon` (fits val+eval only; test untouched). ~30–45 min → run as
   a **background job with a heartbeat** (`[[feedback-longrun-heartbeat]]`); a single
   `hp_one` config is ~1–4 min (foreground OK).
6. **Select on val** (book balance across K, not just @1), take the top 1–2 configs to
   **`final_fit_canon` on test**. **Adopt only if the FT beats the baseline BOOK
   (R-p@3–@20)** — not @1/AUC alone (recipe §4; the `_276` anti-selection trap). Model
   selection ENDS on test — never defer the choice to the backtest window.
7. **Re-save the adopted config LAST** (recipe guard #5 — `final_fit_canon` clobbers the ft
   dir; capture intermediate `test.csv` copies before refitting).
8. **Backtest** (strategy read + deploy cut, NOT a model-selection arbiter):
   `uv run python -m scripts.backtests.run_fresh_oos --cell <ft_dir> --predictions
   <ft_dir>/predictions/backtest.csv --out results/backtests/data/_<tag> --name <tag>
   --selection-mode rank --rank-by raw --sizing-mode equal --horizon <H>`.
   `--sizing-mode equal`, never `rank_kelly` (recipe §6). Note the 200d room caveat and the
   `^NDX`-reference-benchmark caveat for non-US cells.
9. **Record**: `hp/EXPLORATION.md` in the ft dir (windows/prevalence, baseline, path tried,
   test table, verdict); append registry rows (`results/gbdt/data/r_precision_at_k.csv`,
   `mode=canon_ft`, **fill all 8 window-date columns**); memo per `docs/<module>` convention.
   Tables: raw metric + base_rate, NO lift columns.

## Verdict discipline

Judge on the **test book**, never AUC/val-Brier. Expected outcomes by base-@1 headroom
(recipe §"real determinant"): weak @1 → FT wins every K (#50, #52, V1.11 10/25); high @1 →
baseline stands (#51, #54, V1.11 20/100); mid @1 → a trade, decide on the test book (#53,
V1.10 50/200 adopted d10). A finetune result on one test window is NOT a replicated edge —
feature-level claims need an independent second window (the `_272`→`_273` / V1.11 Task-B
pattern).
