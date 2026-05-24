# v0.3 — Drawdown-filtered opportunity scan (NIFTY 50)

**Task:** v0.2's grid, but filter out events whose price path took an adverse half-threshold excursion before reaching the target. Quantifies how many of v0.2's raw events are "clean" (monotone-ish path to target) vs noisy (significant intermediate adverse move).

**Filter rule:**
- UP target at `+thr`: clean iff `min(adj_close ∈ (t, t_breach]) > (1 − thr/2) · adj_close[t]`
- DOWN target at `-thr`: clean iff `max(adj_close ∈ (t, t_breach]) < (1 + thr/2) · adj_close[t]`

Both flavors of "clean" mean: the path to the first breach never made a half-target adverse move first.

**Spec:** same grid as v0.2 — `{up, down} × {5, 10, 20, 30, 50%} × {10, 20, 50, 100 days}` = 40 cells per stock.

- Script: [`scripts/gbdt/v0_opportunity_scan_filtered.py`](../../scripts/gbdt/v0_opportunity_scan_filtered.py).
- JSON: [`results/gbdt/data/_v0_opportunity_scan_filtered_data.json`](../../results/gbdt/data/_v0_opportunity_scan_filtered_data.json).
- Data window: 2020-01-01 → 2025-12-31 (~1,492 trading rows per stock).
- This is a price-path descriptor, NOT a strategy backtest. No positions, no PnL, no transaction costs (project-wide anti-rule).

## Raw → clean rate, UP direction

|  thr | H=10 raw → clean | H=20 raw → clean | H=50 raw → clean | H=100 raw → clean |
|---:|---:|---:|---:|---:|
| **+5%** | 29.34% → **26.38%** | 47.81% → **38.09%** | 69.99% → **43.89%** | 82.18% → **44.21%** |
| **+10%** | 8.41% → **8.13%** | 21.25% → **19.81%** | 47.17% → **38.65%** | 66.13% → **46.66%** |
| **+20%** | 1.22% → **1.20%** | 4.23% → **4.13%** | 18.66% → **17.80%** | 37.98% → **34.54%** |
| **+30%** | 0.27% → **0.27%** | 1.13% → **1.12%** | 7.50% → **7.38%** | 21.38% → **20.65%** |
| **+50%** | 0.03% → **0.03%** | 0.17% → **0.17%** | 1.44% → **1.44%** | 7.83% → **7.79%** |

## Raw → clean rate, DOWN direction

|  thr | H=10 raw → clean | H=20 raw → clean | H=50 raw → clean | H=100 raw → clean |
|---:|---:|---:|---:|---:|
| **−5%** | 20.99% → **18.90%** | 33.45% → **26.48%** | 48.96% → **30.11%** | 57.62% → **30.58%** |
| **−10%** | 4.99% → **4.78%** | 11.36% → **10.49%** | 24.22% → **19.24%** | 34.09% → **22.80%** |
| **−20%** | 0.98% → **0.97%** | 2.36% → **2.31%** | 7.19% → **6.59%** | 12.49% → **10.49%** |
| **−30%** | 0.46% → **0.46%** | 1.24% → **1.23%** | 3.83% → **3.63%** | 6.07% → **5.32%** |
| **−50%** | 0.17% → **0.17%** | 0.40% → **0.40%** | 1.27% → **1.21%** | 2.40% → **2.08%** |

## Filter ratio = (filtered events) / (raw events)

UP direction:

|  thr | H=10 | H=20 | H=50 | H=100 |
|---:|---:|---:|---:|---:|
| 5% | 10.0% | 20.3% | 37.2% | **46.2%** |
| 10% | 3.2% | 6.7% | 18.0% | **29.4%** |
| 20% | 1.3% | 2.2% | 4.5% | 8.9% |
| 30% | 1.5% | 0.7% | 1.5% | 3.3% |
| 50% | 0.0% | 0.0% | 0.2% | 0.4% |

DOWN direction:

|  thr | H=10 | H=20 | H=50 | H=100 |
|---:|---:|---:|---:|---:|
| 5% | 10.0% | 20.9% | 38.6% | **46.9%** |
| 10% | 4.2% | 7.6% | 20.6% | **33.1%** |
| 20% | 0.4% | 1.9% | 8.2% | **15.9%** |
| 30% | 0.6% | 0.9% | 5.3% | **12.3%** |
| 50% | 0.0% | 0.0% | 4.9% | **13.3%** |

## Findings

1. **The filter bites hardest at low thresholds + long horizons.** A +5% in 100 days has a 46% filter ratio — almost half of those raw "opportunities" took a ≥2.5% drawdown before reaching the target. A +50% in 10 days has a 0% filter ratio — when the price moves 50% in 10 days, it does so monotonically (no choppy paths).

2. **Big-and-fast moves are clean; small-and-slow moves are noisy.** Diagonal pattern: filter ratio is bounded by **min(threshold-distance, time-window-slack)**. A path that travels a long way fast has no room to wander; a path that has a long time to cover a small distance has lots of room.

3. **DOWN paths are systematically choppier than UP paths.** Compare the filter ratios cell-by-cell:
   - 10% × 100d: UP 29.4% filtered, DOWN 33.1% filtered (DOWN higher).
   - 50% × 100d: UP 0.4% filtered, DOWN 13.3% filtered (**huge asymmetry**).
   - The 50% × 100d asymmetry says: large up-moves over 100 days are nearly monotonic (markets grind up), while large down-moves over 100 days frequently include a >25% rally somewhere in the middle (bear-market rallies / dead-cat bounces).

4. **The v1 18-cell lattice cells are mostly clean.** At the v1 targets (`{±10, ±20, ±50%} × {10, 20, 50}`):
   - UP: filter ratios all ≤ 18.0% (worst is +10% × 50d).
   - DOWN: filter ratios all ≤ 20.6% (worst is −10% × 50d).
   - So v1's lattice is biased toward the "clean" side — the targets being predicted aren't dominated by choppy false-positive events. Good for the model: each labeled "1" is mostly a real monotone move toward target, not a near-miss path.

5. **For pre-prediction filtering, the rule is informative but mostly small-magnitude.** Across the v1 lattice, the maximum re-rating of base rate from filtering is `47% → 38%` (UP +5% × 50d, ~20% relative reduction). For the rarer cells (±50%), filtering barely matters because there's no room for adverse moves on the way to a big-and-fast target.

## What this implies for v1 / v2

- **v1 should consider both raw-rate and clean-rate baselines for Brier.** Predicting the raw base rate is the easier baseline; predicting the clean rate is what a "path-quality-aware" model would do. Reporting both in Stage 9's acceptance demo would tell us whether the model is just predicting "did the target ever fire" or "did the target fire cleanly." Cheap to add: it's a different labeling of the same events.
- **Clean-rate target = a richer downstream signal.** If the project eventually consumes these probabilities for any decision (per the anti-rule, *not* in this module), a clean-event probability is more useful than a raw-event probability — paths matter, not just touches. Adding a `clean_<dir>_<thr>_h<H>` family alongside the existing `<dir>_<thr>_h<H>` family is a v1.1 / v2 consideration. Don't pre-engineer; surface as an option once v1 lands.
- **The DOWN asymmetry is mechanistically interesting.** v2 features should probably include realized-vol skew (whether downside vol exceeds upside vol over rolling windows) as a candidate predictor — it's a feature that could directly distinguish stocks/regimes where DOWN targets fire cleanly vs choppily.

## Caveats

- **Same window caveats as v0.1 / v0.2** — 2020-2025 bull-regime-dominated; survivorship-biased universe; `adj_close`-based.
- **First-breach timing.** The filter looks at the path UP TO the first target breach. After the breach, what happens is irrelevant. This is the natural interpretation for a "did you have to suffer to get here" question.
- **Strict inequality** in the filter (`> -thr/2`, not `>= -thr/2`). At thr=0 the question is degenerate; for the thresholds in this scan it's a non-issue (no event lives at the exact half-threshold line for both target and adverse simultaneously, since they're on opposite sides of origin).

## Relationship to v0.1 / v0.2

- v0.1 and v0.2's "base rate" = this report's "raw_rate." Numerically identical at the overlapping cells.
- v0.3 adds the `clean_rate` column and the `filter_ratio`. v0.2 stands as the full unfiltered grid; v0.3 is the filtered companion.

## Re-run

```bash
uv run python -m scripts.gbdt.v0_opportunity_scan_filtered
```
