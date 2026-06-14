# _008: rolling the high-AUC rare-event cells (sp500 ×2, russell1000)

## TL;DR

With `_007`'s faithful-inference fix in place, we refreshed the sp500 +
russell1000 caches and rolled the three remaining strong agent cells. **All three
show a positive, consistent rolling edge vs their index** — and combined with `_007`
(ndx40) and `_006` (cell-5), a clean pattern emerges: **rank/equal deployment of
high-AUC rare-event gbdt cells beats the index across rolling windows; the low-AUC
cell-5 family does not.**

| Cell | AUC | Full-OOS strat / index | Rolling: % windows beat | median excess | OOS span / windows |
|---|---|---|---|---|---|
| sp500 +20%/25d | 0.76 | +73.7% / SPX +7.8% | **100%** | +9.5% | ~6mo / 18 |
| sp500 +50%/50d | **0.90** | +34.2% / SPX +7.8% | **92%** | +7.3% | ~6mo / 13 |
| russell1000 +50%/200d | 0.73 | +135.9% / SPX +63.8% | **90%** | +14.2% | **~3yr / 105** |
| — ndx40 +40%/50d (`_007`) | 0.74 | +121.7% / NDX +37.5% | 57% | +6.6% | ~1yr / 42 |
| — cell-5 family (`_006`) | ~0.52 | ≈ index | 38–46% | −0.6% / −4.4% | ~1yr / 42–52 |

The **russell1000** result is the most credible: a **~3-year OOS, 105 windows, 90%
beating the index, median excess +14.2%, worst window only −7.1%**. The sp500 cells
are the most *consistent* (92–100% of windows positive) but on a shorter ~6-month OOS.

## The pattern (and why it's coherent)

- **AUC tracks the rolling edge.** The high-AUC cells (sp500 0.90/0.76, r1k 0.73,
  ndx40 0.74) all beat the index in a strong majority of windows; the anti-AUC cell-5
  family (~0.52) does not. This is the expected direction: AUC measures ranking quality,
  and rank/equal deploys exactly the ranking. `_004`'s base-rate gate hid this — those
  cells never traded under absolute-Kelly; rank-sizing (`_005`) unlocked them, and
  rolling (`_006`–`_008`) shows the unlocked signal is real on the high-AUC ones.
- **The faithful-inference fix (`_007`) scaled.** russell1000 alignment dropped **56,681**
  gap-fill rows (large membership drift since training — e.g. NYSE:BJ backfilled from 2018)
  and the self-check still PASSED (1e-8). Without the fix none of these could be rolled.

## Methodology

Per `_006`: refresh caches (cache_only=False) → faithful inference (gap-fill aligned,
universe from spec) → one full-OOS rank/equal (c=1.0) back-test → rolling H-day
excess-return distribution vs the universe index at stride 5. Calibrator fit on each
cell's VAL. russell1000 index is a **^SPX proxy** (^RUI uncached).

## Caveats (the edge is promising, NOT yet a deployable-alpha claim)

- **C1: small N.** 9 / 19 / 22 entries over each OOS. The rare-event labels (+50%/+20%
  thresholds) fire seldom; returns ride a handful of big winners. The equity curve — and
  thus most overlapping rolling windows — is dominated by those few names.
- **C2: overlapping windows** (stride 5 ≪ window 50/25/200) → highly autocorrelated. "100%
  / 90% of windows" is NOT 100/90 independent trials — with 9–22 trades the effective
  sample is far smaller. Descriptive, not iid significance.
- **C3: favorable regimes.** sp500 windows are a flattish SPX (+7.8%, ~6mo) where a few
  big winners dominate; r1k spans a strong 3-yr bull (SPX +63.8%). The strategy beat both,
  but regime-conditioning isn't tested.
- **C4: russell1000 index is a proxy** (^SPX, not ^RUI); the fresh region scores the grown
  universe (890 vs 858 training tickers) — a genuine forward state, slightly different XS
  context.
- **C5/C6:** zero costs (low turnover, immaterial), DD-not-bounded on gap-downs — as prior.

**Net:** four high-AUC cells across three universes and three horizons all show positive
rolling edges, with the longest/most-robust (r1k, 3yr) at +14.2% median / 90% of windows.
That is materially stronger evidence than any single-window result — but the small trade
counts + overlapping windows mean the right next step is more trades / disjoint windows /
cost+regime stress, not deployment.

## Reproducibility

- Branch `backtests-v12-roll-sp500-r1k`. Refreshed sp500∪russell1000 (900/1012 kept; fails
  are <1600-row short-history tickers).
- Per cell: `infer_fresh_predictions --cell <cell> --out <fresh.csv>` (alignment automatic) →
  `run_rolling_validation --cell <cell> --fresh <fresh.csv> --out <dir> --name <n>`.
- Artifacts under `results/backtests/_008_roll/`.

## Open questions / follow-ups

- **Disjoint (non-overlapping) windows + a block bootstrap** to get an honest effective-N
  and confidence band on the edge (the overlapping-window % is inflated).
- **Transaction-cost + slippage stress** and **sector-neutralization** on the survivors
  (sp500_20 looked semis-heavy in `_005`).
- **Longer OOS for the sp500 cells** (only ~6mo) — re-score forward as the cache ages.
- **Why does ndx40 (57%) lag the sp500/r1k cells (90–100%)** despite similar AUC? Likely the
  +40%/50d label's rarer fires + the specific window; worth a closer look.
