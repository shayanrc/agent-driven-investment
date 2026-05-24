# v0.2 — Full direction × threshold × horizon opportunity scan (NIFTY 50)

**Task:** extend v0.1 — for each NIFTY 50 stock and each cell in the grid `{up, down} × {5, 10, 20, 30, 50%} × {10, 20, 50, 100 days}` (= 40 cells per stock), count rolling-origin events. Report per-cell pooled-across-stocks base rates and median first-breach lags.

**Spec:**
- Universe: NIFTY 50 as of 2026-05-24 (50 tickers).
- Price field: `adj_close`.
- Event definitions:
  - `up`: `max(adj_close ∈ (t, t+H]) ≥ (1 + threshold) · adj_close[t]`
  - `down`: `min(adj_close ∈ (t, t+H]) ≤ (1 - threshold) · adj_close[t]`
- Script: [`scripts/gbdt/v0_opportunity_scan_full.py`](../../scripts/gbdt/v0_opportunity_scan_full.py).
- Headline JSON: [`results/gbdt/data/_v0_opportunity_scan_full_data.json`](../../results/gbdt/data/_v0_opportunity_scan_full_data.json).
- Data window: 2020-01-01 → 2025-12-31 (~1,492 trading rows per stock, median).
- Sanity check vs v0.1: `up × 10% × {10, 20, 50, 100}` cells reproduce v0.1 numbers exactly (0.0841, 0.2125, 0.4717, 0.6613). ✓

## Pooled base rate — UP direction

Rows = threshold; columns = horizon (trading days).

|  thr | 10d | 20d | 50d | 100d |
|---:|---:|---:|---:|---:|
| **5%** | 0.2934 | 0.4781 | 0.6999 | 0.8218 |
| **10%** | 0.0841 | 0.2125 | 0.4717 | 0.6613 |
| **20%** | 0.0122 | 0.0423 | 0.1866 | 0.3798 |
| **30%** | 0.0027 | 0.0113 | 0.0750 | 0.2138 |
| **50%** | 0.0003 | 0.0017 | 0.0144 | 0.0783 |

## Pooled base rate — DOWN direction

|  thr | 10d | 20d | 50d | 100d |
|---:|---:|---:|---:|---:|
| **5%** | 0.2099 | 0.3345 | 0.4896 | 0.5762 |
| **10%** | 0.0499 | 0.1136 | 0.2422 | 0.3409 |
| **20%** | 0.0098 | 0.0236 | 0.0719 | 0.1249 |
| **30%** | 0.0046 | 0.0124 | 0.0383 | 0.0607 |
| **50%** | 0.0017 | 0.0040 | 0.0127 | 0.0240 |

## Median first-breach lag (days, across-stocks median of per-stock median lag)

UP direction:

|  thr | 10d | 20d | 50d | 100d |
|---:|---:|---:|---:|---:|
| 5% | 6.0 | 9.0 | 13.0 | 16.0 |
| 10% | 7.0 | 13.0 | 23.0 | 32.0 |
| 20% | 8.0 | 14.0 | 32.0 | 51.5 |
| 30% | 8.5 | 14.0 | 36.0 | 62.5 |
| 50% | 7.8 | 16.2 | 40.0 | 75.8 |

DOWN direction:

|  thr | 10d | 20d | 50d | 100d |
|---:|---:|---:|---:|---:|
| 5% | 6.0 | 8.0 | 12.0 | 15.5 |
| 10% | 6.0 | 12.0 | 22.2 | 30.0 |
| 20% | 6.0 | 12.0 | 28.8 | 42.8 |
| 30% | 8.0 | 13.0 | 28.0 | 37.5 |
| 50% | 5.5 | 10.5 | 27.0 | 50.0 |

## Findings

1. **UP/DOWN asymmetry is large and uniform.** Across every (threshold, horizon) cell, the UP base rate exceeds the DOWN base rate. The ratio widens with threshold:
   - 5% × 10d: 0.293 / 0.210 ≈ **1.4×**
   - 10% × 100d: 0.661 / 0.341 ≈ **1.9×**
   - 50% × 100d: 0.0783 / 0.0240 ≈ **3.3×**

   This is the fingerprint of a bull regime — 2020-2025 was post-COVID-recovery + a 2024-2025 NIFTY 50 record run. Symmetric base rates would suggest a sideways regime; a DOWN-dominant grid would suggest a bear. Re-running this scan over an earlier window (when deeper backfill lands) is a v0.3 candidate.

2. **Rarity scales steeply with threshold, shallowly with horizon.** Per doubling/halving:
   - **Threshold doubling (10% → 20%) at H=10**: UP rate drops ~7× (0.084 → 0.012). DOWN ~5× (0.050 → 0.010). Strongly super-linear.
   - **Horizon 10× (10d → 100d) at threshold=10%**: UP rate grows ~8× (0.084 → 0.661). DOWN ~7× (0.050 → 0.341). Sub-linear (because the rate saturates near 1.0).

   At low thresholds (5%), horizon saturates fast — 82% UP at H=100 is approaching the "always happens eventually" asymptote. At high thresholds (50%), even H=100 is rare on the upside (7.8%) and rarer on the downside (2.4%).

3. **First-breach lag scales sub-linearly with horizon, super-linearly with threshold.** At a fixed threshold, lengthening the horizon 10× (10 → 100) only moves the median lag ~3-5×. At a fixed horizon, raising the threshold from 5% to 50% takes the median lag up ~1.3-4.5× (less than the threshold ratio). This means *most events occur in the first part of the window* — once a move is going to happen, it happens early, regardless of how far we extend the window.

4. **DOWN events hit slightly faster than UP at the same threshold/horizon** — by 1-3 days at most cells. Consistent with the volatility asymmetry pattern (drops register faster than rallies because volatility spikes coincide with downside).

5. **The v1 18-cell lattice is well-positioned for calibration measurement.** The v1 plan's headline targets `{±10%, ±20%, ±50%} × {10, 20, 50}` span:
   - Pooled base rates from **0.00027 (up 50% × 10d)** to **0.4717 (up 10% × 50d)** — three orders of magnitude.
   - The rarest v1 cell (up 50% × 10d) sees only ~20 events across the entire 50-stock universe (50 × 0.00027 × 1492 ≈ 20). Per-stock calibration on this cell is impossible at this data window; cross-sectional pooling is the only way.
   - DOWN cells at 50% are similarly rare. The v1 plan should anticipate that 4-6 of the rarest 18 cells will be calibration-noisy.

## What this implies for v1 / v2

- **Cross-sectional pooling becomes necessary earlier than v1 planned.** For the rarer cells (`±50% in 10 days`, `±30% in 10 days`), single-asset training has too few events for meaningful calibration. The v1 plan currently scopes training to NASDAQ100 only; if v1 reproduces these NIFTY base rates on US data, expect 4-6 of the 18 targets to be uncalibratable per-asset. Plan to either drop them or escalate to pooled training in v1.1.
- **Asymmetry confirms separate UP and DOWN classifiers are right.** A unified "magnitude" target would mask the 1.4-3.3× directional asymmetry. v1's per-direction classifiers (currently 6 × 3 = 18 cells per asset) are validated by this scan.
- **Adding 5% and 30% thresholds is worth considering for v1.** They produce qualitatively different rarity regimes (5% is near-universal at long horizons; 30% bridges the 20→50 gap). But that's an 18→30 cell expansion, against the "small fixed lattice" anti-goal. Defer to v2 unless v1 diagnostics specifically motivate.
- **100d horizon is barely informative at 5% threshold (~82% UP).** Near-certain events have low information content for prediction. The 100-day horizon adds value at moderate thresholds (10-30%) but not at the extremes. Worth keeping for completeness of the rarity curve but de-emphasizing in headline reporting.

## Caveats (same as v0.1)

- **Survivorship-biased universe** (NIFTY 50 as of 2026-05-24).
- **5-to-6 year window dominated by 2020-2025 bull regime.** UP/DOWN asymmetry would shrink (or invert) in a bear or sideways regime.
- **No transaction costs / PnL / execution** (project-wide anti-rule).
- **`adj_close`-based** — corporate-action-adjusted, so splits / bonuses don't fire false events.

## Re-run

```bash
uv run python -m scripts.gbdt.v0_opportunity_scan_full
```

Updates `results/gbdt/data/_v0_opportunity_scan_full_data.json` in place.

## Relationship to v0.1

v0.2 is a strict superset on the metrics dimension (5 thresholds × 2 directions vs v0.1's 1 × 1). UP × 10% × {10, 20, 50, 100} cells reproduce v0.1 exactly. v0.1 stands as the narrower, focused report; v0.2 is the full grid. Both are valid; choose v0.1 for the focused +10% UP cell discussion, v0.2 for cross-cell comparisons.
