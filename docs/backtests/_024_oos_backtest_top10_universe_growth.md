# _024 — OOS backtest of the date-aligned agent top-10 + the universe-growth inference fix

**Question:** take the **date-aligned, agent-tuned top-10 cells** (by test R-Precision@3) and back-test each
on its true out-of-sample window — `test_end+1 → 2026-06-01` — under the champion strategy
(`TopKDailyKellyLabelExit`, rank / equal-weight / **K=3**), benchmarked against the index. How do the
leaderboard models actually trade forward?

**Answer:** of the 10, **7 scored and 3 OOM'd** (the russell XGBoost-agent variants — a memory limit on the
full-history build, not a model/data problem). **5 of the 7 beat SPX**; the CatBoost-agent beat its XGBoost-
agent sibling in **every** head-to-head; the only losers are the two **sp500 +40%** cells. Getting here first
required fixing a self-check that wrongly treated **universe growth** as feature corruption.

## The blocker, and the fix (the bulk of the work)

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

So it is a **panel-extent-dependent cross-sectional eligibility set** — a legitimate universe change, not
corruption. **The fix** (`--allow-universe-growth`, `infer_fresh_predictions`): the self-check now
distinguishes a divergence with **changed membership** (tickers added/removed → warn + proceed) from one with
**unchanged membership** (→ still abort — real `_007`-style corruption). Default is unchanged (strict), so the
`/daily-predictions` cadence is byte-identical. A backtest should depend only on `(model, universe,
date-range)`; the universe legitimately grows over that range. The discriminator validated cleanly: sp500
cbagents *warn + proceed* (+9 tickers); russell cbagents *reproduce exactly* (9.7e-17 — russell membership
didn't move).

## Results — OOS `test_end+1 → 2026-06-01` (mark-to-market at data end 2026-06-23), rank/equal/K=3

| cell | backend | strategy ret | max_dd | entries/tickers | exit triggers (tgt/DD/hzn) | beat SPX |
|---|---|--:|--:|--:|---|:--:|
| russell1000 +40%/200d cbagent | catboost | **+50.4%** | −29.0% | 69/29 | 28/29/4 | ✓ |
| russell1000 +40%/200d agent | xgboost | +42.8% | −26.3% | 60/27 | 22/21/8 | ✓ |
| russell1000 +50%/200d cbagent | catboost | +38.2% | −32.7% | 54/28 | 20/20/7 | ✓ |
| sp500 +50%/200d cbagent | catboost | +38.1% | −29.2% | 51/25 | 18/17/8 | ✓ |
| sp500 +50%/200d agent | xgboost | +31.0% | −30.2% | 52/28 | 17/19/7 | ✓ |
| sp500 +40%/200d cbagent | catboost | +22.2% | −26.0% | 56/25 | 19/24/5 | ✗ |
| sp500 +40%/200d agent | xgboost | +20.7% | −30.1% | 57/27 | 19/26/3 | ✗ |
| russell1000 +50%/200d v14p1 | xgboost | — OOM — | | | | — |
| russell1000 +50%/200d agent | xgboost | — OOM — | | | | — |
| russell1000 +40%/100d v14p1 | xgboost | — OOM — | | | | — |

Benchmarks (same window): **SPX +28.4%, DD −18.9%**; EW basket +29% (sp500) / +19% (russell1000).
russell1000 is benched against **SPX as a proxy** (`^RUI` uncached). The H=100 cell (`russell +40%/100d`,
test_end 2025-05-13, a shorter ~1yr window) is among the OOM'd.

## The 3 OOM'd cells

All three are russell **XGBoost-agent** variants. The feature build *completed* (F16) then the process was
**SIGKILL'd (exit 137)** during the post-build model-load/predict step. The russell full-history panel
(~858 tickers × 36y ≈ 7M rows × 279 features ≈ 16 GB for `X`) plus a 143-feature `Xc` plus XGBoost
materializing a prediction DMatrix on top exceeds the ~39 GB box. The russell **cbagents** survived the same
panel — CatBoost's predict is lighter and the FS'd 53-feature `Xc` is ~⅓ the size. It is a **resource limit,
not a faithfulness or data issue**. A chunked-predict (or a `min_rows`-aware bounded warmup) would complete
them on the full universe — left as a follow-up; the xgb-agent variants are the R-p-leaderboard-losing
siblings of cells already scored here.

## Reading

- **5 of 7 beat SPX** (+28.4%). The two losers are the **sp500 +40%** cells (~+21%); every **+50%** cell and
  both **russell +40%** cells beat — the +40%/sp500 corner is the soft spot.
- **CatBoost-agent ≥ its XGBoost-agent sibling in all 3 head-to-heads** (sp500/50 +38.1 vs +31.0; sp500/40
  +22.2 vs +20.7; russell/40 +50.4 vs +42.8). The R-Precision@3 leaderboard edge carries into realized OOS
  return, not just ranking.
- **russell/40 cbagent +50.4%** leads — and beats its own EW basket (+19%) by 31 pts, so it is selection, not
  beta.
- **Every strategy runs a deeper drawdown than SPX** (−26% to −33% vs −19%). This is the **ungated** strategy;
  the `_017` SMA200 regime overlay (a harness-level gate, not in `run_backtest_cell`) is the known drawdown-
  cap and would tighten these — an add-on, not part of this run.
- Single window, small trade counts (51–69 entries), overlapping horizons — directional, not a Sharpe claim
  (same caveats as `_008`).

## Verdict / recommendation

The date-aligned agent leaderboard is **tradeable OOS** — the top cells beat SPX on realized return, and the
CatBoost-agent edge is real in PnL, not just R-p. Deployment would pair these with the **SMA200 gate**
(`_017`) for the drawdown. Two follow-ups: (1) a memory-bounded inference path (chunked predict) to complete
the 3 russell xgb-agent cells; (2) rolling validation (`_008`-style) before any alpha claim.

## Artifacts

- Fix: `scripts/backtests/infer_fresh_predictions.py` `--allow-universe-growth` (commit `c16c6a7`).
- Sidecar: `results/backtests/data/_024_oos_backtest_top10_data.json`. Per-cell summaries: `/tmp/bt_<cell>/`
  (regenerable; gitignored).
