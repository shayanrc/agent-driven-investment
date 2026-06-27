# _026: min entry-`p` gate on sp500_50 — does a probability threshold break the leaderboard top-10?

## TL;DR (mandatory)

Added one knob to the sp500_50 champion (rank/equal, K=3, c=1.0) — *only candidates with `p ≥ threshold` may be traded* — and swept it, asking whether any threshold lifts sp500_50 into the back-test leaderboard top-10. **Two scales, two answers, one verdict.** On the **calibrated** `p` the strategy ranks on (a single isotonic plateau ≈0.047 in this bull window, per `_021`), a 0.10 threshold drops *every* candidate → **0 trades, flat \$100K → \$100K → the gate disables the strategy.** On the model's **raw** `p_raw` (0.04–0.41), a threshold of **0.10–0.12 nominally doubles** the bull-window return (\$100K → ~\$226–228K, +67% → **+126–128%**; ~#7/118 by total return, ~#1 by excess) by concentrating the top-3 on the highest-conviction names — **but it does not survive**: it is knife-edge (raw 0.15 → +2.6% in the same bull) and **inverts in the 2022 bear** (raw 0.10 → **−30.7%** vs −9.2% baseline, DD −34%). It is a **high-beta amplifier** — more upside in bulls, more downside in bears — not a robust edge. **No genuine top-10 break; keep the champion as-is.** Single most important caveat: the bull is one ~6-month window of **10–17 trades** — tiny-N and regime-conditional, gross of costs.

## Spec (mandatory)

```yaml
prediction_source:
  module: gbdt
  bull_cell: sp500_up_50pct_50d_dd25pct_agentloop           # standard champion (2022 in-sample)
  bear_cell: sp500_up_50pct_50d_dd25pct_bear2022            # --snapshot-end 2022-12-31 retrain (_016); 2022 genuinely OOS
  bull_predictions: faithful inference, 2025-12-30→2026-06-12 (build_scores; p_calibrated=p_raw, daily convention)
  bear_predictions: <bear_cell>/predictions/test.csv

calibrator: BetaBinomial/isotonic fit on each cell's VAL split (cal.transform(p_calibrated))

strategy:                                                   # CHAMPION — only the gate differs
  class: TopKDailyKellyLabelExit
  K: 3
  selection_mode: rank
  sizing_mode: equal
  c: 1.0
  rank_by: calibrated

new_knob:                                                   # added to scripts/backtests/run_backtest_cell.py
  --min-entry-p: float       # gate on the calibrated selection bound (p_mean) — plateaued
  --min-entry-p-raw: float   # gate on the model's raw p_raw (the substantive variant)
                             # filters candidate (date,ticker) rows BEFORE selection; 0.0 = off

sweep:
  raw_threshold: [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
  calibrated_threshold: [0.06 .. 0.20]   # all >= 0.06 -> 0 trades (plateau ~0.047)

windows:
  bull: 2025-12-30 .. 2026-06-26 (comparison_end clipped to data end)
  bear: 2021-12-21 .. 2022-10-19
benchmark: INDEX:^SPX
```

## Pipeline (mandatory)

`predictions(test.csv | faithful inference) → [min-entry-p[-raw] candidate filter] → calibrator → TopKDailyKellyLabelExit (rank/equal K=3) → Backtest engine → equity/metrics`. The gate is a pure candidate filter inserted between prediction load and `_predictions_dict`; everything downstream is the unchanged champion.

## Methodology (mandatory)

- **Calibrated gate is a day-filter, not a name-selector.** The calibrated `p` is an isotonic step function whose top plateau is ~102 names wide (`_021`); with K=3 ≪ 102 the strategy only ever trades that one plateau, so a threshold either sits below it (no-op) or above it (excludes the whole day). The bull plateau ≈0.047 < 0.10 → every day excluded → 0 trades.
- **Raw gate is a name-selector.** `p_raw` has fine resolution; only ~1.7% (bull) / 0.8% (bear) of (day,ticker) rows clear 0.10, so a raw gate keeps the daily high-conviction names and the champion takes its top-3 among them (ranked by the plateaued calibrated `p` → alphabetical among survivors, i.e. selection within the survivor pool is semi-arbitrary).
- **Bear OOS correctness.** The 2022 bear is in the standard champion's training data, so robustness uses the `_016` pre-bear retrain (`--snapshot-end 2022-12-31`), for which 2022 is genuinely OOS.
- **Vintage caveat.** The bull baseline regenerated to +67% on the current cache + a window extended to 2026-06-26 (the committed champion id 13 = +34%, to 06-12). So leaderboard placement is **indicative**, not a like-for-like row.

## Results (mandatory)

Sweep (`sweep_results.csv`; figure `figs/_026_threshold_bull_vs_bear.png`):

| regime | raw thr | total return | excess vs SPX | max DD | entries | target hits |
|---|--:|--:|--:|--:|--:|--:|
| bull 2026-H1 | — (base) | +67.1% | +60.6% | −15.4% | 10 | 2 |
| bull 2026-H1 | 0.05 | +103.8% | +97.3% | −17.3% | 10 | 4 |
| bull 2026-H1 | 0.08 | +118.5% | +112.0% | −18.5% | 11 | 5 |
| bull 2026-H1 | **0.10** | **+126.3%** | +119.8% | −20.3% | 11 | 5 |
| bull 2026-H1 | **0.12** | **+128.0%** | +121.5% | −20.4% | 11 | 5 |
| bull 2026-H1 | 0.15 | +2.6% | −3.9% | −32.0% | 10 | 3 |
| bull 2026-H1 | 0.20 | +12.8% | +6.1% | −10.6% | 6 | 2 |
| bear 2022 | — (base) | −9.2% | +8.2% | −32.8% | 15 | 0 |
| bear 2022 | 0.05 | −26.0% | −8.6% | −32.1% | 16 | 0 |
| bear 2022 | 0.08 | −35.5% | −18.1% | −45.8% | 17 | 0 |
| bear 2022 | **0.10** | **−30.7%** | −13.3% | −33.9% | 17 | 0 |
| bear 2022 | 0.12 | −18.4% | +0.3% | −29.4% | 15 | 0 |
| bear 2022 | 0.15 | +23.1% | +40.4% | −21.2% | 11 | 0 |
| bull CALIBRATED thr 0.10 | — | **0.0%** | −6.5% | 0.0% | **0** | 0 |

Leaderboard placement (committed `backtest_summary.csv`, top-10 bars: total 1.205, excess 0.721): bull raw 0.10/0.12 → ~#7 by total return, ~#1 by excess. **Indicative only** (vintage/window differ).

Reproduce: add `p_calibrated=p_raw` to the faithful-inference CSV, then
`uv run python -m scripts.backtests.run_backtest_cell --cell <cell> --predictions <csv> --c 1.0 --k 3 --selection-mode rank --sizing-mode equal --min-entry-p-raw <thr>`.

## Caveats (mandatory)

- **Tiny-N / knife-edge.** 10–17 trades per window; ~5 target-hits drive the bull win. Bull collapses +128%→+2.6% between raw 0.12 and 0.15; bear swings −18%→+23% between 0.12 and 0.15. Non-monotonic cliffs = fitting to a handful of trades, not signal.
- **Regime-conditional.** The bull gain inverts in the bear (raw 0.10: +126% → −31%) — the gate selects high-`p_raw` = high-beta names that reach +50% more often in bulls *and* crash hardest in bears (zero bear target-hits; DD worsens to −46% at raw 0.08). Matches `_016` (edge is bull-only) and `_021` (raw selection adds beta, doesn't pay).
- **Vintage-inconsistent leaderboard comparison** (above).
- Single bull + single bear window; gross of costs; survivor-pool selection semi-arbitrary (calibrated plateau).

## Verdict (mandatory)

**The min-`p` entry gate is not an improvement and is not deployed.** On the calibrated `p` the strategy uses it disables sp500_50 (0 trades at 0.10); on the raw scale it is a regime-conditional high-beta amplifier that *nominally* tops the bull leaderboard but collapses in the bear — a fragile single-window artifact, not a genuine top-10 break. Champion stays as-is. The `--min-entry-p` / `--min-entry-p-raw` infra is retained for future use. **These sweep runs are intentionally NOT added to the canonical registry** — they are vintage-inconsistent and fragile, and listing them would misrepresent the leaderboard. Closes the entry-`p`-threshold thread opened by `_021`.
