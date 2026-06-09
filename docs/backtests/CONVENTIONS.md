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
  class: TopKWithLabelExit
  K: 3
  target_return: 0.10
  stop_drawdown: 0.05
  horizon_days: 50
  anchor: signal_day_close
  re_rank_policy: none
  re_entry_policy: allowed_after_exit

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
**This is the first thing under Results. Lead with the dollar table:**

| Strategy / Benchmark | End $ | Total % | CAGR | Max DD |
|---|---|---|---|---|
| Strategy (...) | $X | +X.X% | X.X% | -X.X% |
| NDX buy-and-hold | $Y | +Y.Y% | Y.Y% | -Y.Y% |
| 92-ticker equal-weight basket | $Z | +Z.Z% | Z.Z% | -Z.Z% |
| Equal-weight top-K (no Kelly) | $W | +W.W% | W.W% | -W.W% |

Plus equity-curve overlay figure (all lines, same start, same time axis).

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

One row per back-test run. Columns (in this order):

```
id                              # e.g., 001
name                            # e.g., cell5_bayesian_kelly
prediction_source               # e.g., gbdt
prediction_module               # e.g., gbdt
cell_or_preset                  # e.g., nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation
model_artifact_path             # relative to repo root
calibrator_class                # e.g., BetaBinomialBucketed
sizer_class                     # e.g., VinceOptimalF
sizer_c                         # e.g., 0.5
strategy_class                  # e.g., TopKWithLabelExit
K
oos_start
oos_end
n_picks
n_unique_tickers
initial_cash
final_equity_strategy
total_return_strategy           # decimal, e.g., 0.123 for 12.3%
cagr_strategy
max_dd_strategy                 # negative number
sharpe_strategy_gross
final_equity_ndx_bh             # NaN if benchmark unavailable
total_return_ndx_bh
cagr_ndx_bh
max_dd_ndx_bh
final_equity_ew_basket
total_return_ew_basket
cagr_ew_basket
max_dd_ew_basket
final_equity_ew_topk_no_kelly
total_return_ew_topk_no_kelly
cagr_ew_topk_no_kelly
max_dd_ew_topk_no_kelly
r_precision_at_1_realized       # of realized picks; for cross-check vs published
r_precision_at_3_realized
r_precision_at_5_realized
commit_sha                      # full SHA at run time
run_timestamp                   # ISO 8601
```

Append-only; don't rewrite past rows when conventions evolve (add new columns and leave old rows with NaN).

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
