# _002: cell5b_fresh_oos

## TL;DR

A genuinely **fresh out-of-sample** back-test: we took the already-trained
`nasdaq100_up_10pct_50d_dd5pct_b_acceptance_agent` model (the V1.3 Option-B,
scout+FS-prefit sibling of the `_001` cell), refreshed the OHLCV cache to
2026-06-12, and **scored the new dates with the trained model — no retraining**
(inference self-check reproduced the published test.csv `p_raw` to 1.5e-8). That
produced 64 fresh signal days the model never saw (`2026-03-13 → 2026-06-12`),
which we ran through the same half-Kelly strategy as `_001`.

**$100,000 → $108,960 (+9.0%)**, vs **NDX buy-and-hold +21.6%** and **EW basket
+15.4%**; it beats the no-Kelly top-K cohort (+7.1%). So the strategy
underperforms passive NASDAQ-100 again — but this window is **contiguous (no
132-day forced-cash gap)**, so the `_001` confound is gone, and the result is the
cleaner evidence. **The signal itself persists OOS**: realized win rate **66.7%**
(10/15 closed, near the published R-p@3=0.638), target exits +16.7%. The strategy
trails the index for a structural-design reason, not signal decay: half-Kelly +
breakeven exits keep it **~59% in cash** (avg gross exposure 0.41) and the +10%/−5%
label exits cap winners while the index compounds the full bull move.

## Spec

```yaml
prediction_source:
  module: gbdt
  cell_or_preset: nasdaq100_up_10pct_50d_dd5pct_b_acceptance_agent
  model_artifact_path: results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_b_acceptance_agent/
  predictions_csv: results/backtests/_002_fresh/fresh_predictions.csv   # INFERRED, not from the artifact
  inference: trained model scored on refreshed panel (no retrain); self-check vs test.csv max_abs_diff=1.5e-8

calibrator:
  class: BetaBinomialBucketed
  params: {n_bins: 10, alpha_prior: 1.0, beta_prior: 1.0}
  fit_set_source: val   # the cell's VAL split (leak-free); native isotonic pass-through so p_calibrated≈p_raw
sizer:   {class: DiscreteBoundedLossKelly, c: 0.5, gross_cap: 1.0, cash_buffer: 0.02, payoffs: {win: 0.10, loss: 0.05}, breakeven_p: 0.3333}
strategy: {class: TopKDailyKellyLabelExit, K: 3, target_return: 0.10, stop_drawdown: 0.05, horizon_days: 50, anchor: signal_day_close}
engine:  {fill_mode: next_open, lookback: 5, gap_policy: ffill_zero_volume, commission_fn: None, initial_cash: 100000.0}

oos_window:
  fresh_start: 2026-03-13          # day after the cell's published test_end
  fresh_end: 2026-06-12            # data end (cache refreshed to here)
  comparison_end: 2026-06-12
  full_50bd_resolution_cutoff: 2026-04-01   # signals after this are open/MTM at the data end
  rationale: dates strictly after the model's published test window → never seen in train/val/eval/test

universe: {name: nasdaq100, n_tickers_used: 92, price_basis: split-adjusted close}
```

## Pipeline

```
refresh cache → 2026-06-12 (provider gap-fill; fixed a circular data/raw symlink first — see Repro)
   │
trained model (model.ubj) + features.yaml + hp.yaml
   │  scripts.backtests.infer_fresh_predictions: build_feature_matrix on refreshed panel (causal, C1),
   │  subset to features.yaml, XGBoost.predict_proba → p_raw  [self-check vs test.csv: max_abs_diff 1.5e-8]
   ▼
fresh_predictions.csv (2026-03-13..2026-06-12, 5872 rows, 92 tickers)
   │  calibrator fit on cell VAL → (p_mean,p_low,p_high)
   ▼
TopKDailyKellyLabelExit (K=3, c=0.5) → Backtest(next_open) → results/backtests/_002_cell5b_fresh_oos/
```

## Methodology

**Inference (the new piece).** No retraining. `infer_fresh_predictions.py` rebuilds the
279-candidate causal feature matrix on the refreshed full-history panel (mirroring
`gbdt/__main__.py:1955` — `build_feature_matrix(...).dropna(axis=1, how="all")`),
subsets to the cell's `features.yaml` in saved order, loads `model.ubj` via
`XGBoostModel.load`, and scores. A **mandatory self-check** reproduces the cell's
`predictions/test.csv` `p_raw` on the overlap and aborts if `max_abs_diff > 1e-4`;
here it was **1.5e-8**, confirming the build + load are faithful. Causal features mean
scoring later dates has no look-ahead (C1).

**Calibration / sizing / strategy / engine** are identical to `_001` (see
`_001_cell5_bayesian_kelly.md`). Calibrator fit on this cell's VAL split.

## Data

92-ticker nasdaq100 roster, split-adjusted `close` from the data_pipelines cache,
refreshed to 2026-06-12. `INDEX:^NDX` for benchmark #1. **ANSS is now genuinely
delisted** (provider 404 — the Synopsys merger finally propagated; its cache freezes
at 2026-05-22) — the first time the plan's R5 delisting case is live rather than
hypothetical; it simply stops being a tradeable candidate after its last date.

## Results

### Headline

| Strategy / Benchmark | End $ | Total % | Max DD | n_trades |
|---|---|---|---|---|
| Strategy (Bayesian + Kelly c=0.5) | $108,960 | +9.0% | −7.3% | 32 |
| NDX buy-and-hold | $121,555 | +21.6% | −7.4% | 1 |
| 92-ticker equal-weight basket | $115,413 | +15.4% | −6.5% | 92 |
| Equal-weight top-K (no Kelly) | $107,073 | +7.1% | −5.2% | ~40 |

Equity overlay: `results/backtests/_002_cell5b_fresh_oos/figs/equity_overlay.png`.

### Post-cost reality check

Turnover is low (16 entries / 15 exits / 1 trim) and the verdict is unchanged by costs:
at 10 bps/side the strategy loses only ~$50 (32 trades × ~$16k avg notional × 10bps),
so net ≈ $108,910 — still well below NDX. The ranking (NDX > basket > strategy >
no-Kelly) is cost-insensitive.

### Detail

- **Realized closed positions (15): win rate 66.7%** (10/15), mean realized +4.4% —
  much closer to the published R-p@3=0.638 than `_001`'s 33%, because this window is
  contiguous and the book diversified across **14 tickers** (vs `_001`'s 4).
- By exit trigger: **target 4 (+16.7% mean)**, **breakeven 9 (+2.7% mean)**, **DD 2
  (−12.6% mean)**. The breakeven exits dominate — the model's conviction drops below
  breakeven often, cycling the position to cash. DD exits again realize well past −5%
  on gap-downs (R3).
- **Avg gross exposure 0.41** (59% cash) — the under-investment is now driven by
  breakeven-exit cycling + small near-breakeven Kelly fractions, NOT a data gap.
- 1 position still open (unresolved) at the data end (opened after the 2026-04-01
  full-resolution cutoff); it is marked-to-market, not realized.

## Caveats

- **C1: short, recent window.** 64 signal days (~3 months); only signals through
  2026-04-01 complete a full 50-BD horizon by the data end, so the tail is partially
  marked-to-market. Smaller sample than `_001`, but free of the 132-day gap confound.
- **C2: under-investment is the headline drag.** At avg 41% gross exposure the strategy
  cannot keep pace with a +21.6% always-invested index even with a 67% pick win rate.
  This is the half-Kelly-near-breakeven + breakeven-exit interaction, a real design
  property (quarter-Kelly / a `p_low` selection filter are the levers — see `_001`
  sensitivity + follow-ups).
- **C3: label exits cap winners.** Target exits realized +16.7% mean while NDX ran
  further; the +10%/−5% exit frame is structurally upside-capped in a bull market.
- **C4: DD not bounded at −5%** (gap-downs, −12.6% mean on 2 exits) — R3, as in `_001`.
- **C5: zero costs / price-return basis** — same as `_001` (D18/D24); immaterial here.

## Reproducibility

- Branch `backtests-v11-fresh-oos`. Today 2026-06-13.
- **Cache infra fix (one-time, outside the repo):** the scratch cache had a circular
  `data/raw` symlink (`cache_data/raw → …/data/raw → cache_data/raw`, ELOOP) that
  silently blocked all provider downloads since 2026-05-27 (cache frozen at 2026-05-22).
  Replaced the loop with a real directory, then refreshed via
  `ensure_universe_cached(..., cache_only=False)` to 2026-06-12 (93/101 kept; the 8
  drops are the cell's already-excluded short-history tickers).
- Commands:
  - `uv run python -m scripts.backtests.infer_fresh_predictions --cell <b_acceptance_agent dir> --out results/backtests/_002_fresh/fresh_predictions.csv`
  - `uv run python -m scripts.backtests.run_fresh_oos --cell <…> --predictions results/backtests/_002_fresh/fresh_predictions.csv --out results/backtests/_002_cell5b_fresh_oos --name cell5b_fresh_oos`
- **Action chart** (strategy equity + NDX buy-hold + labeled buy/sell points; added retroactively via `scripts/backtests/plot_actions.py`, see `_020`): `results/backtests/_002_cell5b_fresh_oos/figs/actions.png`.
- Deterministic; no RNG in the inference/strategy/engine path.

## Open questions / follow-ups

- **Re-run at quarter-Kelly + `p_low` selection** to test whether higher invested
  exposure closes the gap to the index without blowing up drawdown (the `_001`
  sensitivity hinted c=0.25 helps).
- **Signal-gap-aware evaluation** (the `_001` C1 item) is moot here (contiguous window)
  but still wanted as a general harness.
- As the cache ages forward, re-score monthly to extend the fresh-OOS series — this is
  now a one-command inference + back-test (no retrain).
