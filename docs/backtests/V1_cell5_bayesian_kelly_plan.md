# V1 — Cell-5 Bayesian + Kelly Back-Test

**Status**: Draft for review.
**Branch**: `backtests-cell5-bayesian-kelly`
**Date**: 2026-06-09

This is the master plan for the inaugural back-test under a new cross-module subtree. It references the supporting docs listed below — read this plan first, then dig into the supporting docs as needed.

---

## 1. Documents in this plan

| Doc | Role | Read when |
|---|---|---|
| **`docs/backtests/V1_cell5_bayesian_kelly_plan.md`** *(this doc)* | The master plan — purpose, decisions, pipeline, sequencing, risks, open questions | Start here |
| [`docs/backtests/CONVENTIONS.md`](CONVENTIONS.md) | Memo template (mandatory sections); registry CSV schema; numbering + naming; file-system layout for `docs/backtests/`, `results/backtests/`, `scripts/backtests/` | Evaluating reporting standards; understanding how the memo will read |
| [`docs/backtests/INDEX.md`](INDEX.md) | Scannable index for the `docs/backtests/` tree (plans + memos, one-line each) | Navigating the subtree over time |
| [`docs/calibration/goal.md`](../calibration/goal.md) | New module charter — what `src/calibration/` is for, what it is *not*, anti-patterns | Evaluating whether the new module's scope is right; reviewing the backend-agnostic contract rule |
| [`docs/trading_strategies/goal.md`](../trading_strategies/goal.md) | New module charter — what `src/trading_strategies/` is for, its two sizer protocols, anti-patterns | Evaluating module scope; reviewing the backend-agnostic probability contract |

**Other docs referenced** (existing, not created here):

| Doc | Why it matters |
|---|---|
| [`docs/backtesting/spec.md`](../backtesting/spec.md) | The engine contract — fill modes, lifecycle, B1–B7 constraints. The pipeline ends with this engine. |
| [`docs/backtesting/goal.md`](../backtesting/goal.md) | Why the engine is strategy-agnostic — informs our strategy-side responsibilities |
| [`docs/gbdt/goal.md`](../gbdt/goal.md) | What cell-5 is + why it's not wired into `forecasters` (informs why calibration is separate, not nested under forecasters) |
| [`src/gbdt/targets.py`](../../src/gbdt/targets.py) (`L107, L116`) | The exact label drawdown + target semantics our strategy mirrors |
| `results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen/` | Cell-5 artifact: `predictions/{eval,test}.csv`, `spec.yaml`, `hp.yaml`, `model.ubj`. The strategy consumes these. (Originally at `..._v1.3_revalidation/`; renamed when the regen got its own canonical CSV row distinct from memo _223's preserved historical row.) |
| `docs/gbdt/_223_cell5_loop_v1.3_revalidation.md` | The memo behind the lost ORIGINAL artifact (canonical row `_v1.3_revalidation`: R-p@3 = 0.5381 preserved as historical reference). The REGENERATED artifact this plan validates has its own canonical row `_v1.3_revalidation_regen`: published R-p@3 = 0.7556. |

## 2. Purpose

Validate gbdt cell `nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen` (canonical CSV R-Precision@3 = 0.7556 on its test slice) by running an actual back-test that answers two concrete questions:

1. **If $100,000 had been invested following this model's top-3 picks per day over the test window, what would the ending equity be at OOS-end?**
2. **How does that compare to investing the same $100,000 in NASDAQ-100 over the same window?**

The published R-Precision@3 says "of the top-3 picks each day, what fraction were positive labels." This plan operationalizes that into a tradeable strategy with realistic position sizing and exit rules, then measures realized dollar return against passive benchmarks.

This is also the inaugural back-test under a new cross-module subtree (`docs/backtests/` + `results/backtests/` + `scripts/backtests/`). Decisions made here set conventions for future back-tests across all prediction backends — see [`CONVENTIONS.md`](CONVENTIONS.md).

## 3. Decisions locked

These were each negotiated explicitly. Sub-decisions had reviewer input where noted in §11.

| # | Area | Decision | Rationale | Reference |
|---|---|---|---|---|
| D1 | Module layout | New top-level `src/calibration/` (flat); new top-level `src/trading_strategies/` with `sizing/` subpackage | Reviewer's Option D. Avoids `forecasters` charter violation. | §11; [calibration/goal.md](../calibration/goal.md); [trading_strategies/goal.md](../trading_strategies/goal.md) |
| D2 | Strategy contract | Strategies accept `dict[Timestamp, list[(ticker, p_mean, p_low, p_high)]]`, not model objects | Backend-agnostic — prevents re-coupling when analog_mc-quantile strategies arrive | [trading_strategies/goal.md](../trading_strategies/goal.md) "backend-agnostic probability contract" rule |
| D3 | gbdt's existing isotonic | NOT migrated in this branch | Avoids the "two-location calibration state" but accepts the cost as a future migration | [calibration/goal.md](../calibration/goal.md) "What this module is *not*" |
| D4 | Calibrator (v1) | Beta-Binomial bucketed Bayesian calibrator; `Beta(1,1)` prior; M=10 quantile bins | Gives `P(y=1 \| p_raw)` with credible bands; uniform prior is weakly informative | §5.1 + checkpoint in §9 |
| D5 | Calibrator fit set | Cell-5's `predictions/val.csv` (revised post-review) | Closes the calibrator-sizer double-use leak: calibrator fits on val, sizer replay fits on eval, test is the back-test. Val has 36,800 rows / 541 days / 92 tickers — 2× eval. Per-ticker val is disjoint from per-ticker eval even when calendar windows overlap, so honest. Our Bayesian recalibrator fits on top of gbdt's already-isotonic-calibrated `p_calibrated`; the value-add is the credible band, not the point recalibration. | §6.2 + R9 |
| D6 | Sizer (primary) | **`DiscreteBoundedLossKelly`** (closed-form, per-pick) — `f_risk = max(0, (b·p − q)/b)` with `b = win/loss` | Path A (Issue #5): the strategy reads today's p_mean for both new picks AND open positions to ratchet down sizing. That requires a per-pick sizer (not Vince's single portfolio fraction). | §6.4 |
| D7 | Sizer (ablation) | `VinceOptimalF` (portfolio TWR fit) + `FixedFraction` baseline | TWR-fit sanity check + naive baseline. Vince still uses the eval-replay return series (§6.2) but isn't the primary sizer under daily rebalance. | §6.3 |
| D8 | Fractional `c` | **0.5 (half-Kelly)** headline; sweep `c ∈ {0.25, 0.5, 1.0}` | Half-Kelly retains ~75% of full-Kelly growth at much lower variance (derived property) | §11 |
| D9 | Multi-position cap + capital allocation | Gross exposure cap = 1.0 (no leverage). **No cross-position pro-rate**: daily rebalance (D13-revised) releases room as conviction drifts; new entries take their intended size from remaining room sequentially (highest-p first); floor `max(0.05·equity, 0.10·room)` drops below-floor entries. | Path A (Issue #5) rewrites the original "pro-rate new entries" rule. Pro-rate is obviated because daily Kelly rebalance does the room management continuously; new entries either fit at intended Kelly size or get dropped (no proportional shrink across competing same-day picks). Multi-asset Kelly w/ covariance still v1.1 (Σ not estimable on this data). | §6.5 |
| D10 | Strategy class | `TopKDailyKellyLabelExit`, K=3 | Mirrors R-p@3 cohort selection; "daily Kelly" because positions are reassessed each signal day against today's p_mean; "label exit" because DD/target/horizon triggers still mirror `src/gbdt/targets.py:107,116`. | §6.5 |
| D11 | Exit + trim triggers (priority within `__call__`) | Per open position, check in order: (1) DD floor `close ≤ 0.95·anchor` → full exit (mandatory); (2) target `close ≥ 1.10·anchor` → full exit; (3) horizon 50 BD → full exit; (4) `p_today ≤ breakeven_p` → full exit (Issue #5; model no longer believes); (5) `new_kelly_f < cur_f` → **trim** to new_kelly_f (ratchet-down only, no add-up). | DD remains mandatory + the label-mirroring set; breakeven-exit + daily-trim are the Path A additions. | §6.5 |
| D12 | Exit anchor | Signal-day close (T's close, NOT T+1 fill open). Anchor is set once at entry and does NOT update when a position is trimmed — only fully reset on full exit + re-entry. | Mirrors label exactly — else back-test and R-p@K diverge by one tick. Trim is a partial sell, not a reset event. | §6.5 |
| D13 | Re-ranking / re-sizing | **Daily Kelly rebalance, ratchet-down only.** Every signal day: read today's p_mean for each open position; compute today's Kelly notional fraction; if smaller than current → trim; if larger → hold (do NOT add to existing positions). Full exit if `p_today ≤ breakeven_p` or any D11 trigger fires. | Path A (Issue #5) replaces the original "no re-ranking — hold until exit." Continuous reassessment captures conviction decay; ratchet-down-only avoids anchor-reset complexity from add-ups. | §6.5 |
| D14 | Re-entry | Allowed after full exit on a future signal day (new anchor = new signal-day close). Trims do NOT count as exits for re-entry purposes — the position remains open at the smaller size with its original anchor. | — | §6.5 |
| D15 | Engine fill mode | `next_open` (default MOO) | B1/B2 of [backtesting/spec.md](../backtesting/spec.md) | — |
| D16 | OOS window | Test slice `[2025-03-26, 2025-12-26]`; positions resolve through ~2026-03-05 | Apples-to-apples with R-p@3 = 0.7556 we're validating | §10 Q1 |
| D17 | Starting capital | $100,000 | Headline framing the user asked for | §6 |
| D18 | Costs | None (commission_fn=None; engine v1 has no slippage) | Engine v1 limitation; quantify estimated drag in memo Caveats | §7 R-cost |
| D19 | Reporting subtree | `docs/backtests/` + `results/backtests/` + `scripts/backtests/` | Cross-module; not nested under any backend | [CONVENTIONS.md](CONVENTIONS.md) |
| D20 | Memo convention | See [`CONVENTIONS.md`](CONVENTIONS.md) | Numbering `_<NNN>_<short>.md`; mandatory sections; registry CSV | [CONVENTIONS.md](CONVENTIONS.md) |

## 4. Pipeline

```
cell-5 predictions (predictions/{val,eval,test}.csv)
                                              │
                                              ▼
calibration.BetaBinomialBucketed.fit(val)   ──►  artifact (bin edges + (α, β) per bin) + reliability figure
                                              │
                                              ▼
                .transform(eval, test) ──► (p_mean, p_low, p_high) per (date, ticker)
                                              │
                                              ▼
trading_strategies.TopKDailyKellyLabelExit (K=3)
  │── SIZER: DiscreteBoundedLossKelly (per-pick); Vince/Fixed are §7 ablations
  │── SELECTION: top-K by p_mean per day; filter to p_mean > breakeven_p
  │── DAILY REBALANCE (each signal day, BEFORE new entries, per open position):
  │     (1) DD: close ≤ 0.95 × signal-day-close      → FULL exit
  │     (2) Target: close ≥ 1.10 × signal-day-close  → FULL exit
  │     (3) Horizon: 50 BD held                       → FULL exit
  │     (4) p_today ≤ breakeven_p                     → FULL exit
  │     (5) new_kelly_f < cur_f                       → TRIM to new_kelly_f
  │── ENTRY: highest-p first, take intended_f from remaining room,
  │           drop entries below floor = max(0.05·equity, 0.10·room)
                                              │
                                              ▼
backtesting.Backtest(fill_mode="next_open", lookback=5, initial_cash=100_000)
                                              │
                                              ▼
results/backtests/_001_cell5_bayesian_kelly/
  ├─ equity_curve.csv, fills.csv, picks.csv, summary.json
  ├─ headline.csv, equity_overlay.png
  └─ calibrator/, sizer/, figs/
```

Each box maps to a module. The boundaries are deliberate: calibration knows nothing about strategies; strategies know nothing about predictors; the engine knows nothing about either. See the goal docs for each module's charter.

## 5. New module scope (V1)

Each new module gets a `goal.md` (what / why / anti-patterns) — read those for the full charter. V1 scope summary below.

### 5.1 `src/calibration/`

Full charter: [`docs/calibration/goal.md`](../calibration/goal.md). Backend-agnostic probability-calibration toolkit.

**V1 ships**:

- `Calibrator` Protocol + `CalibrationOutput` dataclass in `src/calibration/__init__.py`
- `BetaBinomialBucketed` in `src/calibration/bayesian.py`:
  ```python
  class BetaBinomialBucketed:
      def __init__(
          self,
          n_bins=10,
          alpha_prior=1.0,
          beta_prior=1.0,
          min_bin_size=20,           # collapse adjacent bins until each has ≥ this many obs
          min_effective_bins=3,      # raise ValueError if duplicate-drop + min-size collapse leaves fewer bins
      ): ...
      def fit(self, p_raw, y_true, *, sample_weight=None) -> "BetaBinomialBucketed":
          # 1. n_bins quantile edges of p_raw via pd.cut(..., duplicates='drop')
          #    — tiny models (e.g. cell-5: 226 distinct p_raw across 18,400 rows) tie
          #    at quantile boundaries, producing non-unique edges. Drop duplicates;
          #    record effective_n_bins = len(unique_edges) - 1 in fit_diagnostics_.
          # 2. Merge adjacent bins with n < min_bin_size to keep posterior widths
          #    interpretable in the tail.
          # 3. If effective_n_bins < min_effective_bins after merging, raise
          #    ValueError("calibrator: only {N} bins survived dedup+merge; check
          #    p_raw distribution"). Caller decides whether to widen the fit set or
          #    fall back to a simpler calibrator.
          # 4. Per surviving bin i: k_i = sum(y), n_i = count → alpha_i = α₀+k_i,
          #    beta_i = β₀ + n_i − k_i. Store (edges_, alphas_, betas_,
          #    fit_diagnostics_).
      def transform(self, p_raw) -> CalibrationOutput:
          # p_mean = α/(α+β); p_low/p_high via scipy.stats.beta.ppf(0.025/0.975, α, β)
  ```
  **Tie-handling discovered during plan review** (cell-5 regen eval, n=18,400):
  the XGBoost tiny-model (6 trees, depth 2) emits only 226 distinct `p_raw`
  values. Naive `np.quantile(p_raw, 10 quantiles)` produces a duplicate edge
  pair `(0.3852, 0.3852)` because >10% of rows sit at one exact `p_raw`, and
  `pd.cut` rejects non-unique edges. The `duplicates='drop'` + `min_bin_size`
  fields above keep the calibrator usable on any tiny-tree gbdt cell — and on
  analog_mc fan-quantile outputs which also discretize naturally. `R8` in §8
  tracks the residual risk.
- `diagnostics.py`: `expected_calibration_error(p_calibrated, y_true, n_bins=10)`; `reliability_diagram(...)` with 95% credible bands
- `isotonic.py`: **placeholder only**; gbdt's existing internal isotonic NOT migrated

**V1 does NOT ship**: Platt scaling; temperature scaling; conformal prediction; migration of gbdt's `conditional_isotonic`.

### 5.2 `src/trading_strategies/`

Full charter: [`docs/trading_strategies/goal.md`](../trading_strategies/goal.md). Concrete `backtesting.Strategy`-conformant classes; sizing as subpackage.

**V1 ships**:

- `PortfolioSizer` + `PerPredictionSizer` Protocols in `src/trading_strategies/__init__.py`
- `src/trading_strategies/sizing/`:
  - `kelly.py` — `DiscreteBoundedLossKelly` (PerPredictionSizer; closed-form `(b·p − q)/b` in fraction-at-risk framing) — **primary sizer under daily rebalance (D6)**
  - `vince.py` — `VinceOptimalF` (PortfolioSizer; fit on returns; single fraction) — §7 ablation only (D7)
  - `fixed.py` — `FixedFraction` baseline — §7 ablation
- `topk_daily_kelly_label_exit.py`: `TopKDailyKellyLabelExit` class (D10; renamed from V1-draft `TopKWithLabelExit`)

**V1 does NOT ship**: multi-asset Kelly with covariance; risk-sensitive Kelly (Davis-Lleo); continuous Kelly (μ/σ²); forecast-quantile strategies for analog_mc.

## 6. Component design

### 6.1 `BetaBinomialBucketed`

See §5.1 for the API skeleton. Key choices:

- **Quantile binning** (not uniform): equal-mass bins → posteriors with comparable confidence widths
- **Prior `Beta(1, 1)`**: weakly informative (uniform); with eval n ≈ 9K rows / 10 bins ≈ 900 obs/bin, prior contribution is negligible except in sparse upper bins
- **Credible interval via `scipy.stats.beta.ppf`**: 2.5/97.5 percentiles of the posterior Beta — closed-form, no MCMC

### 6.2 Strategy-side calibration replay (for Vince ablation only)

The primary sizer (`DiscreteBoundedLossKelly`, D6) is closed-form — it needs no fit data, just `payoff_win`/`payoff_loss` per cell. The strategy replay below is **only** needed for the Vince ablation row (D7) in the §7 sensitivity table.

1. Apply the val-fit calibrator (D5) to eval-window predictions
2. Replay `TopKDailyKellyLabelExit` (selection + daily rebalance + exits, K=3) over the EVAL period with the closed-form Kelly sizer in primary mode (Vince hasn't been fit yet)
3. Collect per-pick realized returns `rᵢ` from the eval-replay
4. Fit `VinceOptimalF` on `rᵢ` → produces the ablation sizer for §7

This is a separate driver from the test back-test — same strategy code, different window, different sizer assignment.

**Why this is honest** (closes the calibrator-sizer double-use leak): the calibrator's bucketed posterior is fit on val (D5). When applied to eval predictions, each row gets a `p_mean` derived from val's per-bucket hit rate — eval's labels never touched the calibrator. The strategy's eval picks therefore reflect the calibrator's GENERALIZATION from val to eval, not its overfit to eval. Under Path A (Issue #5), the **primary** sizer is closed-form Kelly (D6) — no fitting at all, so no leak; the Vince ablation fits on the eval-replay return series, which inherits val-fit calibrator honesty. Per-ticker val and per-ticker eval are disjoint by construction (each ticker's val ends before its eval starts), so calendar overlap between the global val/eval windows does not introduce leakage.

### 6.3 `VinceOptimalF`

```python
class VinceOptimalF:
    @property
    def per_position_fraction_at_risk(self) -> float: return self._f_star
    def fit(self, trade_returns: np.ndarray) -> "VinceOptimalF":
        max_loss = -trade_returns.min()
        normalized = trade_returns / max_loss
        # f* = argmax over f ∈ (0,1) of  prod(1 + f * normalized)
        # equivalently: argmin of -sum(log(1 + f*normalized))
        self._f_star = ...   # scipy.optimize.minimize_scalar
        self._max_loss = max_loss
        self._diagnostics = {"n_trades": ..., "mean_return": ..., "max_loss": ..., "max_gain": ...}
```

### 6.4 `DiscreteBoundedLossKelly`

```python
class DiscreteBoundedLossKelly:
    def fraction_at_risk(self, p: float, *, payoff_win: float, payoff_loss: float) -> float:
        b = payoff_win / payoff_loss
        return max(0.0, (b*p - (1-p)) / b)   # clipped at 0
```

Interpretation: `f_risk` = fraction-of-bankroll-at-risk per position. Notional implied = `f_risk · equity / payoff_loss` — can exceed equity when `payoff_loss < f_risk`. In cell-5 (loss=0.05), modest `f_risk` implies levered notional; under Path A (D9-revised, D13-revised), the **daily Kelly rebalance + per-entry floor** keep total notional ≤ gross_cap without resorting to cross-position pro-rate. The cap is enforced as a hard check: if entries would push total exposure past 1.0, the lowest-ranked entries are dropped (not proportionally shrunk).

### 6.5 `TopKDailyKellyLabelExit`

```python
class TopKDailyKellyLabelExit:
    def __init__(
        self,
        predictions: dict[Timestamp, list[tuple[str, float, float, float]]],
        K: int,
        target_return: float, stop_drawdown: float, horizon_days: int,
        sizer: PerPredictionSizer,                # closed-form Kelly (D6)
        sizer_payoffs: tuple[float, float],       # (win, loss) per cell
        breakeven_p: float,                       # 1/(1 + win/loss)
        fractional_c: float = 0.5,
        gross_cap: float = 1.0,
        floor_pct_equity: float = 0.05,           # D9: 5% equity floor
        floor_pct_room: float = 0.10,             # D9: 10% room floor
    ): ...

    def __call__(self, state: dict, info: dict) -> dict | None:
        # Path A algorithm (Issue #5). Within each signal day T:
        #
        # === REBALANCE PASS (existing positions) — runs first; releases room ===
        # For each open position p:
        #   last_close = info["last_close"][p.ticker]
        #   bd_held    = days since signal_date
        #
        #   # D11 priority — FULL EXIT triggers
        #   if last_close <= 0.95 * p.anchor_close:        FULL exit (DD, mandatory)
        #   elif last_close >= 1.10 * p.anchor_close:      FULL exit (target)
        #   elif bd_held >= horizon_days:                   FULL exit (horizon)
        #   elif p_today(p.ticker) <= breakeven_p:          FULL exit (model dropped conviction)
        #   else:
        #     # TRIM pass — ratchet-down only (D13)
        #     f_risk_today = sizer.fraction_at_risk(p=p_today, win, loss)
        #     new_f        = fractional_c * f_risk_today / payoff_loss   # notional fraction
        #     cur_f        = (p.current_shares * last_close) / state.equity   # MTM
        #     if new_f < cur_f:  TRIM to new_f (sell delta next_open)
        #     # else: hold; no add-up
        #
        # === ENTRY PASS (new picks) — uses freed room ===
        # updated_exposure = Σ over open positions of (shares × last_close / equity)
        # room = gross_cap - updated_exposure
        # if room <= 0: emit current actions; return
        #
        # candidates = [(ticker, p_mean) from today's predictions
        #               where p_mean > breakeven_p
        #               AND ticker not in open positions]
        # candidates.sort by p_mean desc
        # for ticker, p_mean in candidates[:K]:    # highest-p first
        #     if room < floor: break
        #     f_risk = sizer.fraction_at_risk(p=p_mean, win, loss)
        #     intended_f = fractional_c * f_risk / payoff_loss
        #     actual_f   = min(intended_f, room)
        #     floor      = max(floor_pct_equity, floor_pct_room * room)
        #     if actual_f < floor: continue        # drop this entry, try next
        #     OPEN at actual_f; anchor = last_close (D12); room -= actual_f
        # Emit combined action
```

**Internal state**:

```python
self._open_positions: dict[str, dict] = {}
# ticker -> {
#   "signal_date":     Timestamp,    # set once at entry, never updated by trim
#   "anchor_close":    float,        # signal-day close; D12 — never updated by trim
#   "current_shares":  int,          # mutated on trim
#   "bd_held":         int,          # incremented each call
# }
```

**Key invariants**:
- `anchor_close` is **signal-day close** (T's close) — set at entry, never updated by trims (D12). Trims are a partial sell, not a reset event.
- Daily rebalance is **ratchet-down only**: a position whose `p_today` rises ABOVE its entry-day Kelly stays at the smaller `cur_f` until the model lets up. No add-ups to existing positions.
- Full exit triggers (DD/target/horizon/breakeven) clear `_open_positions[ticker]`, freeing the ticker for re-entry on a future signal day (D14). A re-entered position gets a NEW anchor at the new signal-day close.
- Entry sequence is **sequential, highest-p first** — no cross-pick pro-rate. Floor `max(0.05·equity, 0.10·room)` (D9) drops sub-floor picks; later picks may still fit.

### 6.6 Runner — `scripts/backtests/run_cell5_bayesian_kelly.py`

```
1. Load cell-5 artifact (spec.yaml, predictions/{val,eval,test}.csv)
2. Fit BayesianCalibrator on VAL (D5); save artifact + reliability figure
3. Transform eval + test → (p_mean, p_low, p_high) per (date, ticker)
4. Instantiate primary sizer `DiscreteBoundedLossKelly(payoff_win=0.10, payoff_loss=0.05)` (D6 — closed-form, no fit)
5. (Ablation path) Replay strategy on EVAL with primary sizer → realized r_i; fit `VinceOptimalF` on r_i for the §7 ablation row only
6. Fetch OHLCV via data_pipelines for 92 tickers, window [test_start - 5 BD, test_end + 50 BD]
7. Build DataHandler-compatible feed dict
8. Construct Backtest(fill_mode="next_open", lookback=5, initial_cash=100_000)
9. Run via run_strategy()
10. Compute 3 benchmarks (§6.7) over same window from $100K
11. Persist all outputs + append registry row (CONVENTIONS § "Registry CSV")
```

### 6.7 Benchmarks — `scripts/backtests/benchmarks.py`

Three benchmarks, same window, same $100K start. Trade counts disclosed per line so the post-cost reality check (§7) can apply bps drag proportionally.

| # | Benchmark | Trades | Notes |
|---|---|---|---|
| 1 | **NDX cap-weighted buy-and-hold** (QQQ proxy if NDX unavailable, §10 Q3) | 1 | Single entry at `test_start`, no rebalance during window |
| 2 | **92-ticker equal-weight basket** | ~92 | $100K/92 to each strategy-universe ticker at `test_start`, hold. Lot-rounding may leave some cash unspent (track + disclose). |
| 3 | **Event-driven top-K (no Kelly, no rebalance)** | ~50-100 | Top-K by p_mean each signal day; uniform 1/K sizing; DD/target/horizon exits only. **No breakeven exit, no daily trim** — this is the V1-pre-Path-A counterfactual structurally. |

Each emits: `start_$, end_$, total_return_pct, cagr, max_dd, n_trades, gross_exposure_avg`.

The strategy emits the same fields plus the daily-rebalance specifics from §7 (trim events, full-exits, etc.). The post-cost reality check (§7) consumes `n_trades` from each row.

## 7. Reporting

Full memo template, registry CSV schema, numbering rules, and file-system layout are in [`CONVENTIONS.md`](CONVENTIONS.md). Summary of what the memo for THIS back-test (`_001_cell5_bayesian_kelly.md`) will deliver:

**Headline table** (leads the Results section):

```
Starting capital: $100,000  |  Window: 2025-03-26 → 2025-12-26 (~9 months)

                                  End $        Total %    CAGR    Max DD    n_trades
─────────────────────────────────────────────────────────────────────────────────────
Strategy (Bayesian + Kelly c=0.5) $X         +X.X%      X.X%     -X.X%     N₁
NDX buy-and-hold (cap-weighted)   $Y         +Y.Y%      Y.Y%     -Y.Y%     1
92-ticker equal-weight basket     $Z         +Z.Z%      Z.Z%     -Z.Z%     N₂
Equal-weight top-K (no Kelly)     $W         +W.W%      W.W%     -W.W%     N₃  (ablation)
```

**Post-cost reality check** (mandatory subsection beneath the gross headline — Issue #6): all four lines are computed gross of commission/slippage per D18 (engine v1 limitation). The strategy has structurally higher turnover than the benchmarks because of daily rebalance (R10). To make the comparison defensible, the memo emits a second table that subtracts a per-line cost drag using each line's realized `n_trades`:

```
                                  Gross End $   At 5 bps/side   At 10 bps/side
─────────────────────────────────────────────────────────────────────────────────
Strategy (Bayesian + Kelly c=0.5) $X            $X − N₁·5bps·$  $X − N₁·10bps·$
NDX buy-and-hold (cap-weighted)   $Y            $Y − 1·5bps·$   $Y − 1·10bps·$
92-ticker equal-weight basket     $Z            $Z − N₂·5bps·$  $Z − N₂·10bps·$
Equal-weight top-K (no Kelly)     $W            $W − N₃·5bps·$  $W − N₃·10bps·$
```

Drag computation: `n_trades · bps · avg_notional_per_trade`. The 5 / 10 bps assumptions cover the realistic range for US-large-cap liquid names (5 bps ≈ marketable commission + tight-spread cost; 10 bps ≈ retail-broker + modest spread). If the strategy still beats benchmarks at 10 bps/side, the result is robust to plausible cost models; if it underperforms at 5 bps, the gross headline is misleading.

Plus equity-curve overlay (4 lines, same $100K start, gross), drawdown trajectory, **gross-exposure trajectory** (Path A: critical — shows the cap rarely binds because daily rebalance releases room continuously), per-pick sample table augmented with **trim events** (date, p_today, cur_f, new_f, sold_delta) per position, **turnover summary** (entries, full-exits, trims; cf. R10), and the §7 sensitivity table:

- **Sizer axis** (D6 vs D7): primary `DiscreteBoundedLossKelly` vs ablations `VinceOptimalF`, `FixedFraction(0.20)`
- **Fractional c**: c ∈ {0.25, 0.5, 1.0}
- **K**: K ∈ {1, 3, 5}
- **Daily rebalance**: ON (headline) vs OFF (rebalance disabled → strategy reverts to hold-until-exit for direct comparison; this is the V1-pre-Path-A counterfactual)

Meta-validations (both still applicable):
- **R-Precision@K of REALIZED entered picks** vs canonical CSV R-p@3 = 0.7556 (does the metric translate to entered-pick quality?)
- **R-Precision@K of REALIZED entered picks AFTER full-exits-due-to-trims/breakeven** (does Path A's filtering improve over the published rank?)

Registry row appended to `results/backtests/data/backtest_summary.csv` per the schema in [CONVENTIONS.md](CONVENTIONS.md).

## 8. Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Bayesian calibrator's upper bin (`p_raw > 0.7`) sparse on val | M=10 → M=5 quantile bins if `n_i < 50` for upper bin. **Checkpoint after first fit** (§9 step 5). |
| R2 | Cell-5's `p_raw` overconfident → Bayes shrinks `p_mean` toward base rate → fewer picks pass Kelly breakeven | Expected outcome. Surface in memo. If <10 picks total, sensitivity sweep on `c` or breakeven. |
| R3 | DD exits trigger on close, fill at next open → gap-down realizes loss > 5% (not strictly bounded as label assumes) | Document realized vs label-assumed loss distribution in memo. Daily rebalance trims partially absorb conviction-driven losses BEFORE DD fires (catching them at the trim, not the DD), but gap-down DD exits remain a real tail risk. |
| R4 | NDX identifier not in `data_pipelines` | Fall back to QQQ ETF as proxy. If neither, document the gap; ship basket + ablation. See §10 Q3. |
| R5 | Universe survivorship — roster as-of `test_start` ≠ as-of-fit | Use 92-ticker roster from cell-5's `metrics.json::data.n_tickers_used`. Don't refresh. Known gbdt v1 limitation, not back-test scope. |
| R6 | Look-ahead in calibrator fit or Vince fit | Verify per-ticker val_end < per-ticker eval_start < `test_start` (segment hierarchy honored, not just calendar-window check). Assert in code on each ticker. |
| R7 | Lot-size rounding zeros out small picks | Engine emits `lot_size_audit` in `info`; track rounded-to-zero count; document if non-trivial. |
| R8 | Tiny-model `p_raw` ties → quantile bins collapse | Surfaced during plan review on cell-5 regen eval: 18,400 rows over 226 distinct `p_raw` values; naive `np.quantile(p_raw, 10)` produces a duplicate edge `(0.3852, 0.3852)` because >10% of rows sit at one exact `p_raw`. Calibrator handles via `pd.cut(duplicates='drop')` + `min_bin_size=20` merge + `min_effective_bins=3` floor (see §5.1). If fewer than 3 bins survive on a future cell, the calibrator raises and the caller falls back to a simpler scheme (e.g. one bin per distinct `p_raw` for very tiny models, or a non-Bayesian calibrator). **Checkpoint output** at §9 step 5 must report `effective_n_bins`. |
| R9 | Bayesian recalibrator stacked on gbdt's isotonic-calibrated `p_calibrated` (val-fit) | `p_calibrated` in `predictions/val.csv` is what gbdt's internal `conditional_isotonic` produced from `p_raw` (fit on val itself — leak-prone for the isotonic, but that's the gbdt module's concern, not ours). Our Bayesian recalibrator fits on top of `p_calibrated` (or `p_raw` if isotonic was pass-through per V1.3 anti-AUC handling — check `metrics.json::calibration.fitted` flag and pick the right input column). On cell-5 specifically, V1.3 native pass-through was active so `p_calibrated == p_raw` and there's no stacking. On future cells where isotonic IS active, the Bayesian recalibrator's job is "produce credible bands on already-shrunk probabilities" — the bands will be narrower (less posterior uncertainty after isotonic smoothing), which is correct. **Checkpoint output** at §9 step 5 should report which input column was consumed + the ECE delta val-vs-eval to flag if val→eval generalization is broken. |
| R10 | Daily rebalance amplifies turnover (Path A) | Each signal day can produce trims (partial sells) + full exits + new entries. Turnover grows roughly with # open positions × # signal days. Memo Caveats quantifies: "Strategy made N entries / M trims / P exits over the 9-month window vs V1-pre-Path-A counterfactual of N′ entries / 0 trims / P′ exits → ~K× turnover ratio → expected cost drag ≈ K · 17 bps/yr = X bps/yr (compared to baseline V1 estimate)." See §7 sensitivity row "Daily rebalance OFF" for direct comparison. |
| R11 | Sizing oscillation if `p_today` is noisy day-over-day | Daily Kelly trim fires every time `p_today < cur_f_implied_p`. If predictions are noisy (e.g. small p_mean changes from feature-day boundary effects), positions could trim down on day T and have free room re-filled by a new entry on day T+1, generating churn. Surface as a **stability check** in the memo: compute std(p_today) per ticker over its held window — if > some threshold (say 0.03 std), flag oscillation risk. Mitigation, if needed: introduce a `trim_threshold` (only trim if `new_f < (1 − ε)·cur_f` where ε = 0.05) — V1.1 if observed. |
| R-cost | Zero-cost back-test inflates returns vs reality — and **unevenly across the headline lines** (Issue #6) | Engine v1 emits `commission_fn=None` for ALL lines (strategy + 3 benchmarks). Strategy has 200+ events post-Path A; NDX buy-and-hold has 1; basket has ~92; ablation has ~50-100. A casual reader applying mental cost would penalize the strategy MUCH more than the benchmarks → the gross headline misleads. Mitigation has TWO parts: (1) **disclose per-line `n_trades`** in the headline table (§7) and per-benchmark in §6.7; (2) emit the **mandatory post-cost reality check table** in §7 that subtracts `n_trades · bps · avg_notional_per_trade` from each line at 5 bps/side and 10 bps/side. The strategy's robustness verdict comes from "still beats at 10 bps/side?" — gross headline alone is insufficient. Engine-side cost simulation is a v1.1 follow-up (§10 Q6). |

## 9. Sequencing

| # | Task | Output | Task-list ID |
|---|---|---|---|
| 1 | This branch + this plan + supporting docs | The 4 docs already on this branch | #12 (done in plan-only mode) |
| 2 | Scaffold `src/calibration/` | Package skeleton + pyproject entry + tests dir + importable | #13 |
| 3 | Scaffold `src/trading_strategies/` | Same + `sizing/` subpackage | #14 |
| 4 | Implement `BetaBinomialBucketed` + diagnostics + tests | `src/calibration/` v1 complete | #15 |
| 5 | **CHECKPOINT** — fit on cell-5 VAL, generate reliability figure + report `effective_n_bins` (R8) + ECE on val vs eval (R9) | Review credible-band quality + bin-survival count + whether val→eval generalization is honest before continuing | #16 |
| 6 | Implement sizers (`VinceOptimalF`, `DiscreteBoundedLossKelly`, `FixedFraction`, `caps`) + tests | `src/trading_strategies/sizing/` v1 complete | #17 |
| 7 | Implement `TopKDailyKellyLabelExit` + tests (rebalance + trim + breakeven exit + floor — Path A per Issue #5) | Strategy class complete | #18 |
| 8 | Implement `scripts/backtests/run_cell5_bayesian_kelly.py` | Runnable end-to-end | #19 |
| 9 | Implement `scripts/backtests/benchmarks.py` (3 benchmarks) | Headline table + overlay figure | #20 |
| 10 | Run; write `_001_cell5_bayesian_kelly.md` memo | Memo + figures + registry row | #21 |
| 11 | Commit + open PR + auto-fire review/merge pipeline | PR merged | #22 |

## 10. Open questions for reviewer

1. **OOS window: test slice vs memo calendar window.** ~~The published R-p@3 = 0.7556 was computed on the per-ticker trailing test slice `[2025-03-26, 2025-12-26]` (Q=101). The memo `_223` reports on calendar window `[2025-06-05, 2026-03-12]` (Q=192, cross-section). The plan uses the test slice for apples-to-apples with the canonical CSV. Trivial to switch.~~ **Resolved (2026-06-10):** The regenerated artifact now has its own canonical CSV row `_v1.3_revalidation_regen` carrying the 0.7556 number (Q=90 days with R_q > 0). The back-test runs on the regen's `predictions/test.csv` window `[2025-03-26, 2025-12-26]` — apples-to-apples with that canonical row. Memo _223's original `_v1.3_revalidation` row (R-p@3 = 0.5381) is preserved as the historical reference for the lost original artifact.

2. **Half-Kelly as the headline `c`.** Half-Kelly retains ~75% of full-Kelly growth at lower variance, but quarter-Kelly is more conservative and was my initial recommendation. Sensitivity sweep covers both. Headline `c` choice affects only the lead number.

3. **NDX vs QQQ.** NDX is the cap-weighted index; QQQ tracks NDX (~3 bps fee). Functionally equivalent for a 9-month window. If `data_pipelines` has neither, plan ships without this benchmark (basket + ablation still present). Acceptable?

4. **Fresh-OOS variant in this branch?** Plan covers `[2025-03-26, 2025-12-26]` (test slice). A second back-test on `[2025-12-27, today − 50 BD]` (post-snapshot, fresh-OOS, ~6 months) would answer "does the signal persist." Cleaner as a separate memo per the "one back-test per memo" convention — but bundling is an option.

5. **`_b_acceptance_agent` as a V1.1 follow-up back-test.** The cell `nasdaq100_up_10pct_50d_dd5pct_b_acceptance_agent` is the V1.3-Option-B (scout + FS-prefit) counterpart of the `_v1.3_revalidation_regen` cell this V1 plan back-tests. On its own test window `[2025-06-05, 2026-03-12]` (Q=70) it carries canonical R-p@3 = 0.638 + R-p@1 = 0.800 — tied with the regen on top-1, lower on R-p@3 native, but the regen's `rescore_memo_window_summary.json` shows it scores only R-p@3 = 0.571 on this same window. So on apples-to-apples, `_b_acceptance_agent` is the **stronger model in the cell-5 family** under the legacy methodology, and it exercises the V1.3 **Option B** pipeline (scout response curves + FS-prefit cliff-cut) that `_v1.3_revalidation_regen` does not. Out of V1 scope per the "one back-test per memo" convention; V1 validates the end-to-end plumbing on the cell with the richest documentation lineage (memo _223 → restore → pilot _257 → this memo). V1.1 should swap the cell path + rerun once V1 plumbing is in. Memo will live at `_002_cell5_b_acceptance_bayesian_kelly.md` (or similar) per [`CONVENTIONS.md`](CONVENTIONS.md) numbering.

6. **Engine-side cost simulation as a V1.1 follow-up.** V1 emits gross-of-costs equity curves for all four headline lines (D18 + R-cost) and applies a **post-hoc per-line bps drag** in the memo's reality-check table (§7) to make the comparison defensible despite the strategy's ~3-5× turnover under Path A (R10). The proper fix is engine-side commission/slippage simulation wired through `commission_fn` — that's a `src/backtesting/` change, not a back-test plan change. V1.1 should ship it: a configurable `LinearCommissionModel(bps_per_side=5)` slotted into the engine so each Backtest emits net-of-costs equity natively, and the memo's headline collapses from "gross + reality-check table" to just "net headline." Until then, the §7 reality check is the gating verdict for the strategy's robustness ("still beats at 10 bps/side?").

## 11. Decisions that already had reviewer input

**D1 (module placement)** — Reviewed independently by Plan-agent (opus). Recommended Option A (three flat modules); user chose Option D (calibration flat + sizing nested under trading_strategies). The reviewer's strongest argument — `forecasters` charter violation if calibration nests there — is honored by Option D. Reviewer's critique inline:
- "Putting general post-processing machinery under `forecasters` inflates its surface from 'dispatch + preset persistence' to 'dispatch + preset persistence + ML calibration toolkit'. That's mission creep dressed up as an organizing principle."
- "The naming says 'this belongs to forecasters' but the actual call graph says 'this is a generic calibration utility used between any probabilistic model and any consumer'."

**D6, D8 (Kelly sizing)** — Verified via web search across 8 sources including:
- Frontiers in Applied Math & Stats — "Practical Implementation of the Kelly Criterion" (long-only no-leverage equity formulation)
- IEEE — empirical comparison of Kelly vs Vince's Optimal f (~2% gap in practice)
- arXiv 1806.05293 — "Generalized framework for applying the Kelly criterion to stock markets"
- Wiley 2021 — "Stock Trading System Based on ML and Kelly Criterion"
- Multiple practitioner references on half-Kelly being the modern conservative default

Half-Kelly's 75% growth retention is a **derived mathematical property**, not a heuristic. Updated my initial 0.25 recommendation accordingly.

**D19, D20 (memo conventions)** — Reviewed inline with user. Confirmed `docs/backtests/` location (distinct from `docs/backtesting/` for the module). Confirmed "complete explanation of what was run and how" content bar — formalized in [CONVENTIONS.md](CONVENTIONS.md) mandatory-sections list.

---

*End of master plan. Begin with §1 (Documents in this plan) to navigate; read the goal docs for module charters and [CONVENTIONS.md](CONVENTIONS.md) for the memo template.*
