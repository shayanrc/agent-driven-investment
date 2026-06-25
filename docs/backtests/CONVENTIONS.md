# Back-Test Conventions

This document defines the conventions for back-tests living under `docs/backtests/`, `results/backtests/`, and `scripts/backtests/`. Back-tests are cross-module — they consume `calibration`, `trading_strategies` (with `sizing/`), `backtesting`, and any prediction backend (`gbdt`, `analog_mc`, future). Memos and results don't belong under any one module's docs/.

## File-system layout

```
docs/backtests/
├── CONVENTIONS.md                            # this file
├── INDEX.md                                  # one-line per memo, scannable, capped at ~200 lines
├── V<N>_<short-name>_plan.md                 # plan docs (e.g., V1_cell5_bayesian_kelly_plan.md)
└── _<NNN>_<short-name>.md                    # memo per back-test (e.g., _001_cell5_bayesian_kelly.md)

results/backtests/
├── data/
│   └── backtest_summary.csv                  # canonical registry (one row per back-test)
└── _<NNN>_<short-name>/                      # per-run artifact directory
    ├── equity_curve.csv
    ├── fills.csv
    ├── picks.csv
    ├── headline.csv                          # rendered as memo's headline table
    ├── summary.json
    ├── calibrator/
    │   ├── artifact.pkl
    │   └── bins.csv
    ├── sizer/
    │   └── fit.json
    └── figs/
        ├── reliability.png
        ├── equity_overlay.png
        ├── drawdown.png
        └── ...

scripts/backtests/
├── benchmarks.py                             # shared benchmark utilities
└── run_<short-name>.py                       # per-back-test orchestrator
```

**Figures live under `results/backtests/<id>/figs/`.** The memo references them by relative path. **Do not duplicate figures to `docs/backtests/figs/`** — single source of truth.

## Memo numbering

- Format: `_<NNN>_<short-name>.md`. Three-digit zero-padded, sequential.
- Counter is **cross-back-test** — sequential regardless of which prediction backend or strategy variant feeds the back-test.
- Short-name is `lower_snake_case`, descriptive (`cell5_bayesian_kelly`, `nasdaq_topk_continuous_kelly`, etc.).
- Per-run artifact directory uses the same numeric prefix: `results/backtests/_<NNN>_<short-name>/`.

## Memo template (mandatory)

Every back-test memo MUST include the following sections in this order. Sections marked **(mandatory)** cannot be omitted; sections marked *(omit if N/A)* may be left out if genuinely not applicable, with an explicit note saying so.

```markdown
# _<NNN>: <short name>

## TL;DR (mandatory)
One paragraph: what was tested, the headline number ($ start → $ end), the verdict
(strategy beats / matches / underperforms the universe benchmark), and the single
most important caveat that qualifies the verdict.

## Spec (mandatory)
A YAML-shaped block — every knob explicitly listed. A reader should be able to
reproduce the entire run from this block + the runner script. Include AT MINIMUM:

```yaml
prediction_source:
  module: gbdt | analog_mc | ...
  cell_or_preset: <name>
  model_artifact_path: <path>
  predictions_csv: <path>

calibrator:
  class: BetaBinomialBucketed | None
  params: {n_bins: 10, alpha_prior: 1.0, beta_prior: 1.0}
  fit_set_window: [start, end]
  fit_set_source: <which split>

sizer:
  class: VinceOptimalF | DiscreteBoundedLossKelly | FixedFraction
  params: {c: 0.5, gross_cap: 1.0}
  fit_set_window: [start, end]   # if applicable
  payoffs: {win: 0.10, loss: 0.05}   # if applicable

strategy:
  class: TopKDailyKellyLabelExit
  K: 3
  target_return: 0.10
  stop_drawdown: 0.05
  horizon_days: 50
  anchor: signal_day_close
  re_rank_policy: daily_kelly_rebalance_ratchet_down  # Path A; "none" for hold-until-exit
  re_entry_policy: allowed_after_full_exit            # trims do NOT count as exits
  tie_break: p_calibrated_desc_ticker_asc_mergesort   # source: src/gbdt/topk_diagnostics.py:85-87
  missing_p_today_policy: skip_breakeven_and_trim     # DD/target/horizon still fire

engine:
  fill_mode: next_open | current_close
  lookback: 5
  gap_policy: ffill_zero_volume | raise
  commission_fn: None | <name>
  lot_sizes: {default: 1}
  initial_cash: 100000.0

oos_window:
  start: 2025-03-26
  end: 2025-12-26
  rationale: <why this window>

universe:
  name: nasdaq100
  n_tickers_used: 92
  n_tickers_excluded: 8
  exclusion_reasons: <brief>
```

## Pipeline (mandatory)
ASCII or mermaid block: predictions → calibrator → sizer → strategy → engine → results.
Should match the runner script's call graph.

## Methodology (mandatory)
### Calibration
Method, prior choice, binning, fit-set choice. **Reference reliability figure** with
path. Report ECE before vs after.

### Sizing
Formula DERIVATION (don't just state — show the math). Eval-period replay summary
(n trades, mean/std return, max loss, max gain, breakeven). Solved f*. Fractional
c. Gross cap behavior.

### Strategy
K, exit triggers IN PRIORITY ORDER, anchor convention (signal-day close vs fill
open — be explicit), re-ranking policy, re-entry policy.

### Engine
fill_mode rationale (reference B1/B2 of backtesting/spec.md), lookback, gap
handling, cost model (or "gross of costs — see caveat C-N" with a real estimate).

## Data (mandatory)
Universe roster (sample if >20 tickers; full roster in an appendix if needed),
date window, source (data_pipelines provider + cache state + snapshot_end if
pinned), exclusion list with reasons per ticker.

## Results (mandatory)
### Headline
**This is the first thing under Results. Lead with the dollar table — include `n_trades` per line so the post-cost reality check has its input:**

| Strategy / Benchmark | End $ | Total % | CAGR | Max DD | n_trades |
|---|---|---|---|---|---|
| Strategy (...) | $X | +X.X% | X.X% | -X.X% | N₁ |
| NDX buy-and-hold | $Y | +Y.Y% | Y.Y% | -Y.Y% | 1 |
| 92-ticker equal-weight basket | $Z | +Z.Z% | Z.Z% | -Z.Z% | N₂ |
| Equal-weight top-K (no Kelly) | $W | +W.W% | W.W% | -W.W% | N₃ |

Plus equity-curve overlay figure (all lines, same start, same time axis).

### Post-cost reality check (mandatory)

**Apply per-line bps drag at 5 bps/side and 10 bps/side** to convert each gross result to a net result. The strategy's turnover is structurally different from buy-and-hold benchmarks, so a flat cost assumption would mislead. Drag = `n_trades · bps · avg_notional_per_trade`.

| Strategy / Benchmark | Gross End $ | At 5 bps/side | At 10 bps/side |
|---|---|---|---|
| Strategy (...) | $X | $X − N₁·5bps·$ | $X − N₁·10bps·$ |
| NDX buy-and-hold | $Y | $Y − 1·5bps·$ | $Y − 1·10bps·$ |
| 92-ticker equal-weight basket | $Z | $Z − N₂·5bps·$ | $Z − N₂·10bps·$ |
| Equal-weight top-K (no Kelly) | $W | $W − N₃·5bps·$ | $W − N₃·10bps·$ |

The robustness verdict comes from "still beats at 10 bps/side?" — gross headline alone is insufficient. (Engine-side cost simulation via `commission_fn` is a future v1.1 milestone; until shipped, every memo MUST emit this post-hoc reality-check table.)

### Detail
- Drawdown trajectory figure
- Gross exposure trajectory figure
- Win rate vs base rate
- R-Precision@K of REALIZED picks vs the published metric being validated
- Per-pick sample table: signal-date, ticker, p_raw, p_mean, p_low95, p_high95,
  f_kelly, f_used, fill_open, exit_date, exit_trigger, days_held, realized_return

### Sensitivity (omit if N/A)
Vary c ∈ {0.25, 0.5, 1.0}, K ∈ {1, 3, 5}, or other applicable knobs.

## Caveats (mandatory)
Numbered (C1, C2, ...). **Be specific — quantify where possible:**

❌ "C1: no transaction costs."
✅ "C1: zero commission/slippage modeled. At 5 bps/side and observed turnover of
   ~85 trades over 9 months, expected drag ≈ 17 bps/yr — small relative to the
   X% CAGR but not zero."

## Reproducibility (mandatory)
- Branch + commit SHA at run time
- Exact command(s) to reproduce: `uv run python -m scripts.backtests.run_<name>`
- Expected runtime + output paths
- Random seed (if any)
- Required data-cache state (snapshot_end if pinned, raw-provider state)

## Open questions / follow-ups (omit if N/A)
Bullet list. Promote substantive items to a V<N+1>_TBD.md in the relevant module
if they warrant their own plan.
```

## Registry CSV schema (`results/backtests/data/backtest_summary.csv`)

One row per back-test run — the back-test analog of `results/gbdt/data/r_precision_at_k.csv`.
**Regenerated by `scripts/backtests/regenerate_backtest_performance_csv.py`** (mirrors the
gbdt R-p regenerator): run-dir rows are rebuilt from per-run artifacts (`summary.json` +
`equity_curve.csv` + `picks.csv` written by `run_backtest_cell.py`); the pre-artifact
`_001`–`_023` curated rows are carried + schema-migrated. Idempotent.

Columns are grouped (78 total). Decimals are fractions (`0.123` = 12.3%); `max_dd` is
negative; missing values are blank/NaN, never `0`.

```
# identity / provenance
id name prediction_source prediction_module cell_or_preset experiment runner
model_artifact_path calibrator_class commit_sha run_timestamp notes
#   experiment = bare cell name (suffix-stripped); runner = run_backtest_cell | legacy_survey
#   JOIN KEY into r_precision_at_k.csv = (experiment, train_start, oos_end[=test_end])

# strategy config (decomposed — never regex the class strings)
strategy_class sizer_class sizer_c K selection_mode sizing_mode selection_bound rank_by
regime_gate cost_bps prob_weight_alpha vol_window
#   regime_gate/cost_bps are run_rolling_validation overlays → none/0 for run_backtest_cell rows

# dates
oos_start oos_end                 # = the model's test_start / test_end (the scored window)
comparison_end                    # mark-to-market end = test_end + horizon bdays, clipped to data_vintage
data_vintage                      # OHLCV cache as-of date at run time (= summary window.data_end)
train_start n_days mtm_truncated  # mtm_truncated = comparison_end clipped before the horizon resolved

# universe / capital
universe_n initial_cash           # universe_n = eligible tickers at scoring (NaN if not retained)

# strategy performance
final_equity_strategy total_return_strategy cagr_strategy max_dd_strategy
sharpe_strategy_gross             # legacy placeholder (always NaN) — superseded by sharpe_active_strategy
calmar_strategy vol_annual_strategy sortino_strategy ulcer_index_strategy
worst_day_strategy sharpe_active_strategy   # risk metrics on the ACTIVE window (>= test_start)

# exposure & turnover
avg_gross_exposure pct_days_invested n_picks n_unique_tickers
n_entries n_exits n_trims open_at_end n_exit_target n_exit_dd n_exit_horizon

# benchmark identity + relative  (NaN if benchmark unavailable)
benchmark_index benchmark_is_proxy benchmark_proxy_for
idx_bh_final_equity idx_bh_total_return idx_bh_cagr idx_bh_max_dd
excess_return_total               # = total_return_strategy - idx_bh_total_return (a difference, NOT a lift)
final_equity_ew_basket total_return_ew_basket cagr_ew_basket max_dd_ew_basket
final_equity_ew_topk_no_kelly total_return_ew_topk_no_kelly cagr_ew_topk_no_kelly max_dd_ew_topk_no_kelly

# data vintage / faithfulness  (run_backtest_cell rows only; legacy NaN)
selfcheck_status selfcheck_max_abs_diff universe_delta   # PASS | WARN_UNIVERSE_GROWTH | ...

# ranking-skill link
r_precision_at_1_realized r_precision_at_3_realized r_precision_at_5_realized
#   REALIZED on the scored window; test R-p is JOINED from r_precision_at_k.csv, not duplicated.
#   NaN on horizon-truncated OOS windows (labels not yet resolved — see mtm_truncated).

# quality
caveat                            # True if n_days < 120 or avg_gross_exposure < 0.4 (thin/low-exposure)
```

**Default sort: chronological** (`run_timestamp`, then `id`).

**Benchmark columns (`idx_bh_*`) renamed from the legacy `*_ndx_bh`** (2026-06-25): those
columns held each universe's *actual* benchmark — SPX for sp500/russell, NIFTY for nifty —
stored under an NDX-named column (94/107 rows were non-nasdaq; verified by code +
recompute). The rename is value-preserving; `benchmark_index` now records the index per row
and `benchmark_is_proxy`/`benchmark_proxy_for` flag the russell1000→SPX proxy (`^RUI`
uncached).

**Append-only**; don't rewrite past rows when conventions evolve — add new columns and leave
old rows NaN. (The one-time `*_ndx_bh` → `idx_bh_*` rename above is the sole exception: it
corrected an actively-wrong column *name*, not the data.)

## INDEX.md format

One line per memo, format: `- [_<NNN>: <short name>](_<NNN>_<short-name>.md) — <one-line hook>`. Capped at ~200 lines (oldest entries roll off into `INDEX_archive.md` if needed).

## Naming conventions

| Thing | Convention | Example |
|---|---|---|
| Plan doc | `V<N>_<short-name>_plan.md` | `V1_cell5_bayesian_kelly_plan.md` |
| Memo | `_<NNN>_<short-name>.md` | `_001_cell5_bayesian_kelly.md` |
| Run artifact dir | `_<NNN>_<short-name>/` | `_001_cell5_bayesian_kelly/` |
| Runner script | `run_<short-name>.py` | `run_cell5_bayesian_kelly.py` |

The short-name is consistent across plan doc / memo / artifact dir / runner — so `grep cell5_bayesian_kelly` finds all four.

## What goes in `docs/backtests/` vs `docs/backtesting/`

| Subtree | Purpose |
|---|---|
| `docs/backtesting/` | The `backtesting` MODULE's design docs (goal.md, spec.md, V<N>_PLAN.md per the module's own evolution) |
| `docs/backtests/` | Back-test MEMOS — outputs of running back-tests on real predictions |

Easy to confuse; they are completely different. The module docs describe how the engine works; the memo docs describe what happened when we ran the engine on a real strategy.
