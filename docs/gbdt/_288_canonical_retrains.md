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
| 49 | sp500 +50%/50d | 0.9% | 0.311 | c9 144f·d6·mcw10·ss0.7 | 0.323 | 0.306 | base wins book; c9 wins @1 (kept per user) |
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
| 49 | sp500 +50%/50d | c9 | +64.3% | -15.0% | +26.5% | 99/122 | candidate |
| 50 | sp500 +20%/25d | d8·ss0.85 | +135.1% | -14.4% | +26.5% | 98/83 | **deployed** |
| 51 | nasdaq +40%/50d | baseline | +73.2% | -18.1% | +36.5% | 98/114 | candidate |
| 52 | russell +40%/100d | d8·ss0.7·cs0.7 | +21.2% | -15.3% | +20.5% | 95/143 | candidate |
| 53 | russell +50%/200d | d8·ss0.7·cs0.7 | +82.5% | -14.7% | +20.5% | 103/119 | candidate |
| 54 | sp500 +40%/200d F18 | baseline | +134.6% | -11.3% | +26.5% | 92/79 | **deployed** |

5/6 beat NDX buy-hold; all beat their own EW basket. **Deploy criterion = target hits > DD
stops** (user's rule): sp500_20 (98/83) and sp500_F18 (92/79) clear it → wired into
`/daily-predictions` as `deployed=True`; the other four track as `deployed=False` candidates.

## Caveats
- Gross (no costs — downstream). Equal-weight, not Kelly (the Kelly gate zeroed on the stale
  eval-R-p@K per-pick prob; actual +10/-5 win rate ~54%). Bull window (NDX +32%).
- **test R-p@K does not perfectly predict backtest return** (russell_40_100 won every K on
  test but lagged the backtest, +21%). The backtest window is the independent arbiter.
- The ambiguous top-vs-book cells (#49, #51, #53) would ideally have BOTH the chosen model and
  the alternative backtested to settle the @1-vs-book call — a follow-up.
- **#54 F18 is deployed by the backtest criterion despite the prior "F18 not promoted"
  (_279/_280) note** — flagged; a fundamentals model is now `deployed=True` for the first time.
