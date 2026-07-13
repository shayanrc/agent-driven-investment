# Canonical retrained models — backtest on the backtest window (2025-07-01 → 2026-07-10)

Strategy: TopKDailyKellyLabelExit, **equal-weight** daily top-3 (rank mode, rank-by raw),
+10%/-5%/horizon exits. Predictions = each cell's chosen canonical model, scored on the
backtest window (never touched during train/val/eval/test selection). $100k start, gross
(no costs). Benchmark ^NDX buy-hold = +32.1%; EW basket per universe.

| # | cell | model | return | maxDD | NDX b&h | EW basket | entries | target/DD |
|---|---|---|---|---|---|---|---|---|
| 49 | sp500 +50%/50d | baseline all/d6 (candidate) | +156.4% | -12.2% | +32.1% | +26.5% | 207 | 99/107 |
| 49 | sp500 +50%/50d | c9 144f·d6·mcw10·ss0.7 (candidate) | +64.3% | -15.0% | +32.1% | +26.5% | 223 | 99/122 |
| 50 | sp500 +20%/25d | 279f·d8·ss0.85 | +135.1% | -14.4% | +32.1% | +26.5% | 186 | 98/83 |
| 51 | nasdaq +40%/50d | baseline all/d6 | +73.2% | -18.1% | +32.1% | +36.5% | 212 | 98/114 |
| 52 | russell +40%/100d | 279f·d8·ss0.7·cs0.7 | +21.2% | -15.3% | +32.1% | +20.5% | 239 | 95/143 |
| 53 | russell +50%/200d | 279f·d8·ss0.7·cs0.7 | +82.5% | -14.7% | +32.1% | +20.5% | 223 | 103/119 |
| 54 | sp500 F18 +40%/200d | baseline all/d6 (292f) | +134.6% | -11.3% | +32.1% | +26.5% | 172 | 92/79 |

5/6 beat NDX buy-hold; all beat their own EW basket. Laggard: russell_40_100 (+21.2%,
below NDX; ~40% win rate). Bull window (NDX +32%).

Caveats: gross; equal-weight not Kelly (Kelly gate zeroed on the stale eval-R-p@K per-pick
prob — actual +10/-5 win rate ~54%); ^NDX benchmark for all cells (reference); 200d cells
exit mostly via target/stop before 200d.
