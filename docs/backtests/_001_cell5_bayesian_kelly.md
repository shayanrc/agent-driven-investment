# _001: cell5_bayesian_kelly

## TL;DR

We back-tested gbdt cell `nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen`
(published R-Precision@3 = 0.7556) by operationalizing its top-K daily picks into a
half-Kelly long-only strategy with a Bayesian recalibrator and label-mirroring exits,
then running it through the `backtesting` engine on the test slice
`[2025-03-26 → 2025-12-26]` (positions resolved through 2026-03-12). **$100,000 → $106,728
(+6.7%)**, versus **NDX buy-and-hold +23.2%** and a **92-ticker equal-weight basket +15.6%**.
So on the literal question asked — "would following this model have beaten the index?" — the
answer for this window is **no, the strategy materially underperformed passive NASDAQ-100**.

**But the dominant cause is not signal quality — it is a structural hole in the test slice.**
The per-ticker staggered-cohort test windows cluster into two disjoint calendar bands
(Mar 26–Jun 5 and Oct 15–Dec 26) separated by a **132-day signal gap during which the
strategy holds 100% cash for 67 trading days while NDX rose +14.8%.** When the strategy
*had* signals (cluster 1) it returned **+7.6% vs NDX +8.2% — essentially a match.** The
single most important caveat (C1) is therefore that this is **not an apples-to-apples
$-comparison**: the benchmarks are invested for the whole window; the strategy can only be
invested for the ~2/3 of it that carries signals.

## Spec

```yaml
prediction_source:
  module: gbdt
  cell_or_preset: nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen
  model_artifact_path: results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen/
  predictions_csv: predictions/{val,test}.csv

calibrator:
  class: BetaBinomialBucketed
  params: {n_bins: 10, alpha_prior: 1.0, beta_prior: 1.0, min_bin_size: 20, min_effective_bins: 3}
  fit_set_window: [2023-… , 2024-…]   # cell-5 VAL split (541 days / 92 tickers / 36,800 rows)
  fit_set_source: val
  input_column: p_calibrated   # native pass-through on cell-5 → ≡ p_raw to 1.5e-8

sizer:
  class: DiscreteBoundedLossKelly   # primary (D6); closed-form, no fit
  params: {c: 0.5, gross_cap: 1.0, cash_buffer: 0.02}
  payoffs: {win: 0.10, loss: 0.05}
  breakeven_p: 0.3333
  ablation_sizers: [VinceOptimalF (f*=0.047, fit on eval-replay r_i), FixedFraction]

strategy:
  class: TopKDailyKellyLabelExit
  K: 3
  target_return: 0.10
  stop_drawdown: 0.05
  horizon_days: 50
  anchor: signal_day_close
  re_rank_policy: daily_kelly_ratchet_down   # Path A (D13)
  re_entry_policy: allowed_after_exit_future_day   # D14

engine:
  fill_mode: next_open
  lookback: 5
  gap_policy: ffill_zero_volume
  commission_fn: None   # gross of costs (D18); post-cost reality check below
  lot_sizes: {default: 1}
  initial_cash: 100000.0

oos_window:
  test_start: 2025-03-26
  test_end: 2025-12-26
  comparison_end: 2026-03-12   # test_end + 50 BD; all 4 lines measured here
  rationale: apples-to-apples with the canonical R-p@3 = 0.7556 row

universe:
  name: nasdaq100
  n_tickers_used: 92
  price_basis: split-adjusted close (yfinance `close`; = the column the gbdt label used; D24 price-return)
  in_window_delistings: 0   # ANSS (the plan's R5 example) has full data through 2026-03-09 in this cache
```

## Pipeline

```
cell-5 predictions/{val,test}.csv
   │
   ├─ calibration.BetaBinomialBucketed.fit(val)  →  (p_mean, p_low, p_high) per (date,ticker)
   │
   ▼
trading_strategies.TopKDailyKellyLabelExit (K=3, c=0.5)
   ├─ SIZER DiscreteBoundedLossKelly (per-pick, closed-form)
   ├─ SELECT top-K by p_mean, filter p_mean > breakeven_p (1/3); tie-break (p desc, ticker asc) = D21
   ├─ REBALANCE (DD→target→horizon→[skip if no p_today, D22]→breakeven→ratchet-down trim)
   └─ ENTER highest-p first, min(intended_f, room), drop < floor; cash_buffer guards next_open overdraw
   │
   ▼
backtesting.Backtest(fill_mode=next_open, lookback=5, initial_cash=100_000, gap_policy=ffill_zero_volume)
   │
   ▼
results/backtests/_001_cell5_bayesian_kelly/{equity_curve,picks,fills,gross_exposure}.csv + summary.json + figs/
```

## Methodology

### Calibration

`BetaBinomialBucketed` (10 quantile bins, `Beta(1,1)` prior) fit on the cell's **VAL** split
(D5 — closes the calibrator-sizer double-use leak: calibrator on val, Vince ablation on eval,
test is the back-test). Input column `p_calibrated`; on cell-5 the V1.3 native isotonic
pass-through means `p_calibrated ≡ p_raw` to 1.5e-8, so the Bayesian layer's value-add is the
credible band, not point recalibration.

**Stage-5 checkpoint (gate cleared):** `effective_n_bins=9`, `ECE_val≈7e-5`, `ECE_eval=0.033`
(< 0.10), `|ECE_val−ECE_eval|=0.033` (< 0.05 → val→eval generalizes), `max_band_width=0.035`
(< 0.15). Reliability figure: `figs/reliability.png`.

> **Calibrator bug caught by the checkpoint (and fixed).** The first checkpoint tripped the
> band-width gate at 0.33. Cause: gbdt emits a *constant* down-weight (~0.0101/row) as
> `sample_weight`; feeding it as the Beta trial count deflated the per-bin effective `n` ~100×
> (a 36,800-row fit reporting the uncertainty of 372 obs). Fixed by normalizing `sample_weight`
> to preserve the true sample size (sum=N) — recovers unweighted counts for constant weights and
> matches the plan's "~900 obs/bin" expectation. Bands tightened 10× (0.33 → 0.035).

### Sizing

`DiscreteBoundedLossKelly`: `f_risk = max(0, (b·p − q)/b)`, `b = win/loss = 2`, breakeven
`p = q/(1+b) = 1/3`. Notional fraction = `c · f_risk / payoff_loss`. At cell-5's median top-pick
`p ≈ 0.385`, `f_risk = 0.0775` → notional ≈ 0.78·equity; at `p ≥ 0.41` the intended notional ≥ 1.0
and the gross cap binds. **`cash_buffer = 0.02`** is a Stage-7 addition (not in the plan): under
`next_open`, an order sized to ~all cash at the signal-day close overdraws at the higher next open
and the engine rejects the *whole* order — the buffer reserves cash so near-cap entries fill
(0 overdraw rejections observed). Vince ablation `f* = 0.047` (fit on 30 eval-replay realized returns).

### Strategy

K=3; exit priority per open position (D11): (1) DD `close ≤ 0.95·anchor` → full exit; (2) target
`close ≥ 1.10·anchor`; (3) horizon 50 BD; (4) **if ticker absent from today's predictions, skip
breakeven+trim** (D22 — normal under staggered cohorts); (5) `p_today ≤ breakeven` → full exit;
(6) ratchet-down trim if `new_kelly_f < cur_f` (never adds up, D13). **Anchor = signal-day close**,
set once at entry, never moved by trims (D12). Re-entry allowed on a *future* signal day, never the
same day as the exit (D14). Order actions (not weight — weight liquidates unnamed positions and
continuously rebalances held shares, which conflicts with "hold shares fixed until a trigger").

### Engine

`next_open` (MOO) per B1/B2; `lookback=5`; `gap_policy="ffill_zero_volume"`. Current close is read
from `state["market_data"]["equities"][ticker][-1, 3]` — the plan's `info["last_close"]` (§6.5) does
not exist in the engine's `info` (spec §3.2); this is a documented plan-vs-spec delta. Gross of
costs (D18); see the post-cost reality check.

## Data

92-ticker NASDAQ-100 roster from the cell's test predictions, split-adjusted `close` read straight
from the data_pipelines SQLite cache via `gbdt.data._cache_read` (no domain registration needed; the
same `close` column the gbdt label consumed → D12/D24 satisfied simultaneously). NDX benchmark from
the real `INDEX:^NDX` index (price-return). **Pre-flight roster coverage (R5): 0 in-window
delistings** — notably `NASDAQ:ANSS` (the plan's canonical Synopsys-merger example) has full OHLCV
through 2026-03-09 in this cache, so the delisting mitigation was exercised but not needed
(`delisted_in_window.json` written, empty).

**Critical structural property — the 132-day signal gap.** The per-ticker trailing-50-day test
slices fall into two disjoint calendar bands:

| Cluster | Signal dates | # distinct dates |
|---|---|---|
| 1 | 2025-03-26 → 2025-06-05 | 50 |
| (gap) | 2025-06-05 → 2025-10-15 (132 days) | 0 |
| 2 | 2025-10-15 → 2025-12-26 | 51 |

During the gap the strategy holds **100% cash for 67 trading days** (mean gross exposure 0.000).

## Results

### Headline

| Strategy / Benchmark | End $ | Total % | CAGR | Max DD | n_trades |
|---|---|---|---|---|---|
| Strategy (Bayesian + Kelly c=0.5) | $106,728 | +6.7% | +6.9% | −11.6% | 28 |
| NDX buy-and-hold (cap-weighted) | $123,179 | +23.2% | +24.1% | −14.2% | 1 |
| 92-ticker equal-weight basket | $115,572 | +15.6% | +16.2% | −14.0% | 92 |
| Equal-weight top-K (no Kelly) | $101,366 | +1.4% | +1.5% | −9.7% | 16 |

Equity overlay: `figs/equity_overlay.png`. Drawdown: `figs/drawdown.png`. Gross exposure
(the Path-A sawtooth, median 0.11): `figs/gross_exposure.png`.

**Return decomposition by signal cluster (the key table):**

| Period | Strategy | NDX | note |
|---|---|---|---|
| Cluster 1 (Mar 26 – Jun 5) | +7.6% | +8.2% | **near-match while invested** |
| 132-day gap (forced cash) | +2.3%¹ | +14.8% | strategy in 100% cash; NDX compounds |
| Cluster 2 + resolve (Oct 15 – Mar 12) | −3.1% | −0.9% | label exits sting in a flat/down tape |
| **Full window** | **+6.7%** | **+23.2%** | |

¹ residual from cluster-1 positions resolving in early July (horizon 50 BD); flat thereafter.
The +14.8% NDX gain during the gap alone exceeds the entire +16.5pp full-window shortfall.

### Post-cost reality check

Drag = `n_trades · bps · avg_notional_per_trade`, avg notional/trade = $37,487 (18 fills).

| Strategy / Benchmark | Gross End $ | At 5 bps/side | At 10 bps/side |
|---|---|---|---|
| Strategy (Bayesian + Kelly c=0.5) | $106,728 | $106,203 | $105,678 |
| NDX buy-and-hold | $123,179 | $123,160 | $123,142 |
| 92-ticker equal-weight basket | $115,572 | $113,848 | $112,123 |
| Equal-weight top-K (no Kelly) | $101,366 | $101,066 | $100,766 |

Costs do **not** change the verdict — even at 10 bps/side the strategy ($105,678) trails NDX
($123,142) and the basket ($112,123). The shortfall is structural (forced cash), not cost-driven.

### Detail

- **Realized closed positions (6):** win rate 2/6 = 33% (vs published R-p@3 = 0.7556). 2 target
  exits (mean +10.7%), 4 DD exits (mean −6.7%). **DD exits realize worse than the −5% label floor**
  (R3 confirmed): ANSS −9.5% on a gap-down; others −5.1/−5.4/−6.9%. The entered set ≠ the top-3
  cohort the published metric scores — see C2.
- **Exposure:** median gross 0.11, mean 0.27; 50% of days < 10% invested. Entries at `p` barely
  above breakeven (0.34–0.37) get tiny Kelly notional, so the book is small even when it has signals.
- **Turnover:** 6 entries / 6 exits / 16 trims = 28 trade-events. The 16 trims are Kelly maintenance
  (selling winners down to hold the target fraction as price rises) + R11 oscillation (one position
  entered at 100% on a high-`p` day, slashed to 11% two days later when `p` reverted to breakeven).

### Sensitivity

| Knob | Setting | Total % | Max DD | entries |
|---|---|---|---|---|
| c | 0.25 | **+11.5%** | −11.6% | 11 |
| c | 0.5 (headline) | +6.7% | −11.6% | 6 |
| c | 1.0 | +8.2% | −12.7% | 6 |
| K | 1 / 3 / 5 | +6.7% / +6.7% / +6.7% | −11.6% | 6 |
| daily rebalance | ON (headline) | +6.7% | −11.6% | 6 |
| daily rebalance | **OFF** | **−13.1%** | **−26.9%** | 2 |
| sizer | FixedFraction(0.20) | +2.1% | −13.0% | 5 |

Two findings carry weight: **(a) K is irrelevant** (1=3=5, identical) — the gross cap collapses the
book to ~1 position regardless of K, so the back-test does *not* trade the "top-3 per day" cohort the
R-p@3 metric describes (C2); **(b) daily rebalance is decisive and protective** — turning it OFF drops
the result to −13.1% and doubles max-DD to −26.9% (two un-trimmed positions rode down hard). Path A's
ratchet-down machinery is doing real downside work here, validating its design intent. Quarter-Kelly
(c=0.25, the reviewer's §10-Q2 instinct) is the best headline at +11.5% — but still far under NDX.

## Caveats

- **C1 (verdict-qualifying): the $-comparison is not apples-to-apples.** The test slice has a 132-day
  signal gap (67 trading days, 100% forced cash) inside the comparison window; NDX rose +14.8% there.
  The benchmarks are invested for the full window, the strategy for ~2/3 of it. A signal-availability-
  matched comparison (cluster 1) shows +7.6% vs NDX +8.2% — a near-match. The full-window underperformance
  is dominated by forced cash, not ranking failure.
- **C2: the back-test does not trade the R-p@3 = 0.7556 cohort.** The half-Kelly gross cap collapses K=3
  to ~1 concentrated position (K-sweep is flat across 1/3/5), and entries cluster at `p` just above
  breakeven, so realized win rate (33%, n=6) is far below the published top-3 metric. The cleaner cohort
  test is the EW top-K benchmark (+1.4%) — which *also* trails passive, because the +10%/−5% label exits
  cap upside in a bull tape.
- **C3: DD exits are not bounded at −5%.** Close-trigger + next-open fill + gap-downs realize losses
  beyond the label's −5% floor (worst −9.5%, ANSS). Kelly's binary `payoff_loss = 0.05` assumption is
  therefore optimistic on the loss side (R12); the realized-return distribution is in `picks.csv`.
- **C4: zero costs (D18).** At observed turnover (28 events, ~$37.5k avg notional) the 10-bps/side drag
  is ~$1,050 over 9 months — immaterial to the verdict (see post-cost table). Engine-side cost simulation
  is a V1.1 follow-up.
- **C5: price-return basis, no dividends (D24).** All four lines exclude dividends → all biased down
  ~0.4% over the window, internally consistent. NDX total-return would widen its lead slightly.

## Reproducibility

- Branch `backtests-v1-scaffold`, commit `706ba1e` (run-time); plan merged via PR #160.
- Command: `uv run python -m scripts.backtests.run_cell5_bayesian_kelly`
- Runtime ≈ 1–2 min (cache-only OHLCV reads). Outputs under `results/backtests/_001_cell5_bayesian_kelly/`.
- **Action chart** (strategy equity + NDX buy-hold + labeled buy/sell points; added retroactively via `scripts/backtests/plot_actions.py`, see `_020`): `results/backtests/_001_cell5_bayesian_kelly/figs/actions.png`.
- Deterministic — no RNG in the strategy/sizer/engine path (calibrator is closed-form).
- Cache state: `data/processed.db` `us_equities` table covering the 92-ticker roster + `INDEX:^NDX`
  through ≥ 2026-03-12.

## Open questions / follow-ups

- **Signal-gap-aware evaluation.** The 132-day hole makes the full-window $-comparison structurally
  unfair to a signal-gated strategy. A future memo should report a cash-adjusted or invested-days-only
  benchmark alongside the literal one — promote to `docs/backtests/V1.1_TBD.md`.
- **Quarter-Kelly headline.** c=0.25 beats c=0.5 here (+11.5% vs +6.7%) at equal max-DD; revisit the
  D8 half-Kelly default (§10 Q2).
- **`_b_acceptance_agent` cell** (§10 Q5) and **fresh post-snapshot OOS** (§10 Q4) remain the natural
  next back-tests once this plumbing is merged.
- **`trim_threshold` (R11).** Positions oscillate from 100% to 11% on single-day `p` reversion; an
  ε-band on trims would damp churn — V1.1 if the cost matters once engine-side costs land.
