# _288 — Canonical-periods retrains of the six champion/candidate cells

All six tracked cells retrained on the **canonical evaluation windows** (CLAUDE.md): train
2015-01-01→2022-03-29 · val→2023-06-30 · eval→2024-06-30 · **test 2024-07-01→2025-06-30** ·
backtest 2025-07-01→2026-06-30. Method + full per-cell narrative:
[`CANONICAL_FINETUNE_RECIPE.md`](CANONICAL_FINETUNE_RECIPE.md) and each ft dir's
`hp/EXPLORATION.md`. Explicit-boundary `date_aligned` split, min_rows_per_ticker 2591.

## Method (one line)
Controlled baseline = `final_fit` all-features / default HP (d6, val-AUC ES, raw p) — the
same code path as the FT, HP-only difference (the runner's `metrics.json` is NOT a valid
bar). Diagnose eval↔test agreement + base @1 headroom; tune deep+bagging (common/low-@1) or
keep the base (high-@1); select on val, confirm on test by the **book (R-p@K)**, not AUC.

## Test R-p@K — controlled baseline vs chosen model (raw p, test window)
| # | cell | prev | base@1 | chosen model | base R-p@3 | chosen R-p@3 | verdict |
|---|---|---|---|---|---|---|---|
| 49 | sp500 +50%/50d | 0.9% | 0.311 | baseline all/d6 | 0.323 | 0.323 | baseline is the #49 model (wins the book on test); c9 (R-p@3 0.306) → sp500_50_c9 candidate |
| 50 | sp500 +20%/25d | 4.8% | 0.253 | 279f·d8·ss0.85 | 0.277 | **0.313** | FT wins EVERY K |
| 51 | nasdaq +40%/50d | 3.0% | 0.564 | baseline all/d6 | 0.462 | 0.462 | base stands |
| 52 | russell +40%/100d | 8% | 0.208 | 279f·d8·ss0.7·cs0.7 | 0.288 | **0.351** | FT wins EVERY K |
| 53 | russell +50%/200d | 12% | 0.516 | 279f·d8·ss0.7·cs0.7 | 0.497 | **0.511** | FT wins @3–@20; loses @1 |
| 54 | sp500 +40%/200d F18 | 17% | 0.628 | baseline all/d6 (292f) | 0.517 | 0.517 | base wins EVERY K |

Key result: **base @1 headroom** (not prevalence) decides deep+bagging's net win — low @1 →
FT wins every K; high @1 → the sharp top is maxed and bagging dilutes it → base stands.

## Backtest — chosen models, backtest window (2025-07-01→2026-07-10)
Equal-weight daily top-3 (rank mode / rank-by raw), +10%/-5%/horizon exits, $100k gross.
Benchmark ^NDX buy-hold +32.1%. Full data + per-cell dirs: `results/backtests/canon/`.

| # | cell | model | return | maxDD | EW basket | target/DD | deploy |
|---|---|---|---|---|---|---|---|
| 49 | sp500 +50%/50d | **baseline all/d6** | **+156.4%** | -12.2% | +26.5% | 99/107 | candidate |
| 49 | sp500 +50%/50d | c9 (candidate) | +64.3% | -15.0% | +26.5% | 99/122 | candidate |
| 50 | sp500 +20%/25d | d8·ss0.85 | +135.1% | -14.4% | +26.5% | 98/83 | **deployed** |
| 51 | nasdaq +40%/50d | baseline | +73.2% | -18.1% | +36.5% | 98/114 | candidate |
| 51 | nasdaq +40%/50d | d8·ss0.85 (alt) | +78.7% | -13.3% | +36.5% | 84/113 | — |
| 52 | russell +40%/100d | d8·ss0.7·cs0.7 | +21.2% | -15.3% | +20.5% | 95/143 | candidate |
| 53 | russell +50%/200d | **stratified (t=13)** ⭐ | **+71.9%** | **-29.6%** | **+31.0%** | **25/16** | **deployed** |
| 53 | russell +50%/200d | d8·ss0.7·cs0.7 | +82.5% | -14.7% | +20.5% | 103/119 | candidate |
| 54 | sp500 +40%/200d F18 | baseline | +134.6% | -11.3% | +26.5% | 92/79 | **deployed** |

**Deployed set (3):** `sp500_20` (98/83), `sp500_f18` (92/79), and **`russell_50_200` Stratified** (25/16) clear the strict target>DD rule.
 **sp500_50 baseline is the top backtest performer (+156.4%) but has MORE DD-stops than
target-hits (99/107), so it stays a candidate** (holding the rule; a user decision reversing the
earlier best-return deploy). c9 (the +50%/50d fine-tune, +64.3%) is also a candidate.

## Tie-break (alternative vs chosen, backtest window)
Backtested the model NOT chosen for the three ambiguous top-vs-book cells:
- **#49: the baseline all/d6 wins the held-out TEST book (R-p@3 0.323 vs c9's 0.306) — that is
  the test-grounded basis** for making it the primary #49 model (`_canon_ft`); the FS+HP c9
  fine-tune becomes the `sp500_50_c9` candidate. The backtest (+156.4% vs c9 +64.3%, DD -12.2%
  vs -15.0%) is *consistent* with this but is NOT the basis (see the canonical-discipline note
  below). Neither is deployed: the baseline fails the strict target>DD rule (99/107).
- #51: baseline retained over d8/ss0.85 (both ≈ on test; backtest +73.2% vs +78.7% a wash).
- #53: d8/ss0.7/cs0.7 retained — it wins the **test** book (R-p@3 0.511 vs baseline 0.497) and
  the backtest agrees (+82.5% vs +16.6%). The controlled-baseline-vs-FT call is test-grounded.
Lesson: on the highest-signal rare cell the FS+HP fine-tune anti-selected vs the plain
full-feature default on **test** — the controlled-baseline discipline mattered.

## Caveats
- Gross (no costs — downstream). Equal-weight, not Kelly (the Kelly gate zeroed on the stale
  eval-R-p@K per-pick prob; actual +10/-5 win rate ~54%). Bull window (NDX +32%).
- **test R-p@K does not perfectly predict backtest return** (russell_40_100 won every K on
  test but lagged the backtest, +21%). Model selection stays on `test`; the backtest is a
  strategy/deploy check, NOT the model arbiter (see the Canonical-discipline note).
- The ambiguous top-vs-book cells (#49, #51, #53) had both the chosen model and the alternative
  backtested (the Tie-break section) to *sanity-check* the @1-vs-book call. The #49 baseline-over-c9
  choice is decided on the **test** book (0.323 vs 0.306); the backtest is consistent, not the basis.
- **#54 F18 is deployed by the backtest criterion despite the prior "F18 not promoted"
  (_279/_280) note** — flagged; a fundamentals model is now `deployed=True` for the first time.
- **F18 self-check (fixed 2026-07-14, commit `ab16851b`): sp500_f18 now serves.** The fund
  full-build in `infer_fresh` had aborted (max_abs_diff 5.4e-2) for two reasons, both specific
  to the fundamentals path (technical cells ride the incremental cache, which replays frozen
  training-time cross-sections): (1) infer resolved the sp500 roster at its ≥1600-td floor
  (~479) vs the cell's trained 2591-gate (~468), and `_align_panel` aligned to the *technical*
  cache of the same universe so it didn't drop the +11 → they re-ranked the cross-sectional
  `fund_*_xs_rank`/`_xs_zscore` columns. Fix: pin the fund build to the cell's OWN trained
  `(date,ticker)` keys (`predictions/test.csv`) → membership exact (+0/−0), mean 7.2e-4→1.6e-5.
  (2) A ~3.4e-2 residual on 504/117000 rows, clustered on ~30 tickers with the step-constant-
  across-a-filing-window signature = point-in-time `fund_*` revision across valuation-panel
  rebuilds — a fund cell can't reproduce `test.csv` byte-identically forever. Fix: a bounded
  fund-drift tolerance (`FUND_VALIDATION_TOL=0.05` max **and** `FUND_MEAN_TOL=1e-3` mean,
  unchanged-universe only) mirroring `allow_universe_growth`; the mean bound is the corruption
  guard (the _007 backfill bug lifts the mean far past it). Forward scores always use the
  current panel. The MODEL itself was always faithful (`final_fit` reproduced `test.csv`).

## Canonical-discipline note (backtest-window usage)
Model **configs/HP were selected purely on train/val/eval/test** — nothing about the fits or feature/HP
choices saw the `backtest` window, so the model-side discipline is clean (the `final_fit` rebuild
proves it). But the **deploy cut** (target-hits > DD-stops) is a strategy-simulation metric computed
**on the backtest window**, so choosing the deployed set (sp500_20, sp500_f18) *does* use it. Per the
canonical role discipline (backtest = "never touched by model selection"), that makes the quoted
backtest returns **retrospectively in-sample for the deploy choice** — treat them as informational,
NOT a clean OOS estimate. The honest out-of-sample record for the deployed models is the forward
**/daily-predictions** log accumulated from 2026-07-14 onward, on data no selection step has seen.
(The #49 model choice is re-grounded on the test book above, so only the deploy cut leans on the backtest.)
