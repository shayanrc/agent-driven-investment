# _024 — OOS backtest of the date-aligned agent top-10 + the universe-growth inference fix

**Question:** take the **date-aligned, agent-tuned top-10 cells** (by test R-Precision@3) and back-test each
on its true out-of-sample window — `test_end+1 → 2026-06-01` — under the champion strategy
(`TopKDailyKellyLabelExit`, rank / equal-weight / **K=3**), benchmarked against the index. How do the
leaderboard models actually trade forward?

**Answer:** all **10 scored, 7 beat SPX**. The standout is the H=100 cell (russell +40%/100d, +86.8% on a
shorter window); russell cells dominate the top, the two **sp500 +40%** cells are the soft spot. Getting here
took two tooling fixes — a self-check that wrongly treated **universe growth** as corruption, and a
**chunked predict** to keep the full-history russell builds inside RAM.

## Fix 1 — `--allow-universe-growth` (the bulk of the work)

Fresh OOS inference runs a faithfulness self-check (`infer_fresh_predictions`): rebuild the test-window
features, reproduce the saved `test.csv` predictions, abort if `max_abs_diff > 1e-4` (the `_007` backfill-
corruption guard). Every sp500/russell `_aligned` cell aborted at ~0.04–0.06.

**Diagnosed cause (not corruption):** the `_aligned` cells use `min_rows_per_ticker: 2000`, and that
eligibility filter is computed over the **whole loaded panel `[1990, end]`**. To score the OOS window you
must extend the panel to 2026, which pulls **9 recent listings** — CRWD, DDOG, UBER, MRNA, CTVA, DOW, FOX,
FOXA, VRT (all 2018–2020 IPOs/spin-offs) — over the 2000-row floor. They become eligible and **join the
cross-sectional peer group at the historical test dates**, re-ranking every name's `rank`/`z-score`. Proven
with a feature-level diff vs the cached training matrix:

```
build to test_end (2024-10-03):   every feature Δ = 0.00e+00,  predictions identical  (0.000)
build to OOS end (2026-06-01):    +9 tickers in the cross-section; drift is 100% in cross-sectional
                                  features (path/vol features Δ = 0); max|Δp| 0.057, mean 1e-4
```

A legitimate universe change, not corruption. The self-check now distinguishes a divergence with **changed
membership** (tickers added/removed → warn + proceed) from one with **unchanged membership** (→ still abort,
real `_007` corruption). Default is unchanged (strict), so the `/daily-predictions` cadence is byte-identical.
A backtest should depend only on `(model, universe, date-range)`; the universe legitimately grows over that
range. Validated: sp500 cbagents *warn + proceed* (+9 tickers); russell cbagents *reproduce exactly* (9.7e-17).

## Fix 2 — chunked `predict_proba` (completed the last 3 cells)

The 3 russell **XGBoost-agent** cells survived `--allow-universe-growth` but were **OOM-killed (exit 137)**
right after the feature build: the full-history russell panel (~858 tickers × 36y ≈ 7M rows × 279 features
≈ 16 GB for `X`) plus a 143-feature `Xc` plus XGBoost materializing a float32 prediction copy on top exceeds
the ~39 GB box. (The russell *cbagents* survived — CatBoost predict is lighter, FS'd 53-feature `Xc` is ~⅓.)
Predicting in **500k-row chunks** (`_predict_proba_chunked`) bounds that copy; output is identical (predict is
row-wise). All 3 then scored — and their self-checks **PASS at 2.98e-08**, confirming the failures were
**purely a memory limit, not feature corruption**.

## Results — OOS `test_end+1 → 2026-06-01` (mark-to-market at data end 2026-06-23), rank/equal/K=3

| cell | backend | strategy ret | max_dd | vs index | entries/tk | exits tgt/DD/hzn | beat index |
|---|---|--:|--:|--:|--:|---|:--:|
| russell1000 +40%/100d v14p1 | xgboost | **+86.8%** | −10.9% | +61.5 | 60/29 | 26/16/7 | ✓ |
| russell1000 +40%/200d cbagent | catboost | +50.4% | −29.0% | +22.0 | 69/29 | 28/29/4 | ✓ |
| russell1000 +50%/200d agent | xgboost | +46.1% | −27.1% | +17.7 | 46/30 | 12/11/12 | ✓ |
| russell1000 +40%/200d agent | xgboost | +42.8% | −26.3% | +14.4 | 60/27 | 22/21/8 | ✓ |
| russell1000 +50%/200d cbagent | catboost | +38.2% | −32.7% | +9.7 | 54/28 | 20/20/7 | ✓ |
| sp500 +50%/200d cbagent | catboost | +38.1% | −29.2% | +9.6 | 51/25 | 18/17/8 | ✓ |
| sp500 +50%/200d agent | xgboost | +31.0% | −30.2% | +2.6 | 52/28 | 17/19/7 | ✓ |
| russell1000 +50%/200d v14p1 | xgboost | +24.9% | −30.5% | −3.6 | 53/31 | 14/19/8 | ✗ |
| sp500 +40%/200d cbagent | catboost | +22.2% | −26.0% | −6.2 | 56/25 | 19/24/5 | ✗ |
| sp500 +40%/200d agent | xgboost | +20.7% | −30.1% | −7.7 | 57/27 | 19/26/3 | ✗ |

Benchmark: **SPX**, the 200d cells over `2024-10-04→2026-06-01` = **+28.4% (DD −18.9%)**; the H=100 cell
(`russell +40%/100d`, test_end 2025-05-13) over its shorter `2025-05-14→2026-06-01` window = **+25.3% (DD
−9.1%)** — its `vs index` and shallow DD reflect that shorter window. EW basket +29% (sp500) / +19%
(russell1000). russell1000 is benched against **SPX as a proxy** (`^RUI` uncached).

## Reading

- **7 of 10 beat SPX.** The 3 misses are both **sp500 +40%** cells (~+21%) and **russell/50 v14p1**. Excess
  spans −7.7 to +61.5 pts.
- **CatBoost-agent ≥ its plain XGBoost-agent sibling in 3 of 4** matchups (sp500/50 +38.1 vs +31.0; sp500/40
  +22.2 vs +20.7; russell/40 +50.4 vs +42.8) — **but russell/50 is the exception**: the plain agent (+46.1)
  beats the cbagent (+38.2). So the R-p@3 cbagent edge mostly carries into realized PnL, not universally.
- **`v14p1` is the high-variance variant**: best overall (russell +40%/100d +86.8%) *and* a clear loss
  (russell +50%/200d +24.9%, the worst russell cell).
- **russell cells dominate** (4 of the top 5; +42% to +87%) and beat their own EW basket (+19%) by wide
  margins — selection, not beta. **sp500 +40% is the weak corner** (both losses).
- **Every strategy except the H=100 cell runs a deeper drawdown than SPX** (−26% to −33% vs −19%). This is the
  **ungated** strategy; the `_017` SMA200 regime overlay (harness-level, not in `run_backtest_cell`) is the
  known drawdown-cap and would tighten these.
- Single window, small trade counts (46–69 entries), overlapping horizons — directional, not a Sharpe claim
  (same caveats as `_008`). The H=100 standout especially: shorter window, fewer independent bets.

## Verdict / recommendation

The date-aligned agent leaderboard is **tradeable OOS** — 7/10 beat SPX on realized return and the CatBoost-
agent edge mostly carries from R-p@3 into PnL (russell/50 the lone reversal). Deployment would pair these with
the **SMA200 gate** (`_017`) for the drawdown. Follow-up: **rolling validation** (`_008`-style) before any
alpha claim — single-window results, and the +86.8% H=100 number rests on a short window.

## Artifacts

- Fix 1: `scripts/backtests/infer_fresh_predictions.py` `--allow-universe-growth` (commit `c16c6a7`, PR #218).
- Fix 2: `scripts/backtests/infer_fresh_predictions.py` `_predict_proba_chunked` (commit `27a68bf`).
- Sidecar: `results/backtests/data/_024_oos_backtest_top10_data.json`. Per-cell summaries: `/tmp/bt_<cell>/`
  (regenerable; gitignored).
