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
| D6 | Sizer (primary) | **Vince Empirical Optimal f** — argmax over `f ∈ (0,1)` of `∏(1 + f·rᵢ/max\|loss\|)` | Handles uneven payoffs; ~2% gap vs closed-form Kelly in practice | §11 (web search references) |
| D7 | Sizer (ablation) | `DiscreteBoundedLossKelly` (closed-form) + `FixedFraction` baseline | Sanity check on Vince + naive baseline | §5.3 |
| D8 | Fractional `c` | **0.5 (half-Kelly)** headline; sweep `c ∈ {0.25, 0.5, 1.0}` | Half-Kelly retains ~75% of full-Kelly growth at much lower variance (derived property) | §11 (web search updated my initial 0.25 recommendation) |
| D9 | Multi-position cap | Gross exposure cap = 1.0 (no leverage); pro-rate new entries | Stable Σ estimate not feasible on 100-day eval × 92 tickers; multi-asset Kelly w/ covariance is v1.1 | §5.4 |
| D10 | Strategy class | `TopKWithLabelExit`, K=3 | Mirrors R-p@3 metric being validated | §5.4 |
| D11 | Exit triggers (priority) | (1) DD floor `close ≤ 0.95·anchor` (mandatory); (2) target `close ≥ 1.10·anchor`; (3) horizon 50 BD | User-specified DD mandatory; mirrors label `src/gbdt/targets.py:107,116` | §5.4 |
| D12 | Exit anchor | Signal-day close (T's close, NOT T+1 fill open) | Mirrors label exactly — else back-test and R-p@K diverge by one tick | §5.4 |
| D13 | Re-ranking | None — hold until exit | Matches label semantics; re-ranking conflates stability with realization | §5.4 |
| D14 | Re-entry | Allowed after exit on a future signal day | — | §5.4 |
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
trading_strategies.TopKWithLabelExit (K=3)
  │── SIZING: fit trading_strategies.sizing.VinceOptimalF on eval-period
  │           strategy-replay returns (calibrator from val is honest here)
  │── SELECTION: top-K by p_mean per day; filter to p_mean > breakeven_p
  │── ENTRY SIZE: f_used = c × f*_emp; pro-rate against gross cap
  │── EXIT (each close, priority order):
  │     (1) DD: close ≤ 0.95 × signal-day-close      → sell next open (mandatory)
  │     (2) Target: close ≥ 1.10 × signal-day-close  → sell next open
  │     (3) Horizon: 50 BD held                       → sell next open
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
  - `vince.py` — `VinceOptimalF` (PortfolioSizer; fit on returns; single fraction)
  - `kelly.py` — `DiscreteBoundedLossKelly` (PerPredictionSizer; closed-form `(b·p − q)/b` in fraction-at-risk framing)
  - `fixed.py` — `FixedFraction` baseline
  - `caps.py` — `apply_gross_exposure_cap(current_exposure, new_fractions, cap=1.0)` pro-rate utility
- `topk_label_exit.py`: `TopKWithLabelExit` class

**V1 does NOT ship**: multi-asset Kelly with covariance; risk-sensitive Kelly (Davis-Lleo); continuous Kelly (μ/σ²); forecast-quantile strategies for analog_mc.

## 6. Component design

### 6.1 `BetaBinomialBucketed`

See §5.1 for the API skeleton. Key choices:

- **Quantile binning** (not uniform): equal-mass bins → posteriors with comparable confidence widths
- **Prior `Beta(1, 1)`**: weakly informative (uniform); with eval n ≈ 9K rows / 10 bins ≈ 900 obs/bin, prior contribution is negligible except in sparse upper bins
- **Credible interval via `scipy.stats.beta.ppf`**: 2.5/97.5 percentiles of the posterior Beta — closed-form, no MCMC

### 6.2 Strategy-side calibration replay for Vince

The Vince sizer needs **realized strategy returns** to fit on. Procedure:

1. Apply the val-fit calibrator (D5) to eval-window predictions
2. Replay `TopKWithLabelExit` (selection + exits, K=3) over the EVAL period with **uniform 1/K sizing** (no Kelly yet — we're generating the return distribution Kelly will then size against)
3. Collect per-pick realized returns `rᵢ`
4. Fit `VinceOptimalF` on `rᵢ`

This is a separate driver from the test back-test — same strategy code, different window, different sizing.

**Why this is honest** (post-review, closes the calibrator-sizer double-use leak): the calibrator's bucketed posterior is fit on val (D5). When applied to eval predictions, each row gets a `p_mean` derived from val's per-bucket hit rate — eval's labels never touched the calibrator. The strategy's eval picks therefore reflect the calibrator's GENERALIZATION from val to eval, not its overfit to eval. Vince fits on returns that estimate honest out-of-val strategy performance, which is the right reference for sizing the test back-test. Per-ticker val and per-ticker eval are disjoint by construction (each ticker's val ends before its eval starts), so calendar overlap between the global val/eval windows does not introduce leakage.

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

Interpretation: `f_risk` = fraction-of-bankroll-at-risk per position. Notional implied = `f_risk · equity / payoff_loss` — can exceed equity when `payoff_loss < f_risk`. In cell-5 (loss=0.05), modest `f_risk` implies levered notional; **the gross-exposure cap is what enforces no-leverage in the strategy** (per D9).

### 6.5 `TopKWithLabelExit`

```python
class TopKWithLabelExit:
    def __init__(
        self,
        predictions: dict[Timestamp, list[tuple[str, float, float, float]]],
        K: int,
        target_return: float, stop_drawdown: float, horizon_days: int,
        sizer: PortfolioSizer | PerPredictionSizer,
        sizer_payoffs: tuple[float, float] | None = None,
        breakeven_p: float | None = None,
        fractional_c: float = 0.5,
        gross_cap: float = 1.0,
    ): ...

    def __call__(self, state: dict, info: dict) -> dict | None:
        # 1. EXIT pass — per open position, check close vs anchor; priority: DD → target → horizon
        # 2. ENTRY pass — top-K by p_mean filtered by Kelly breakeven; skip already-held; compute f_used; pro-rate cap; snap to lots
        # 3. Emit combined order action
```

**Internal state**:

```python
self._open_positions: dict[str, dict] = {}
# ticker -> {"signal_date": Timestamp, "anchor_close": float, "bd_held": int, "entry_fill": float | None}
```

`anchor_close` is **signal-day close** (T's close), recorded when the order is submitted — NOT the fill open of T+1. This is the most subtle correctness issue in the strategy; see D12.

### 6.6 Runner — `scripts/backtests/run_cell5_bayesian_kelly.py`

```
1. Load cell-5 artifact (spec.yaml, predictions/{val,eval,test}.csv)
2. Fit BayesianCalibrator on VAL (D5); save artifact + reliability figure
3. Transform eval + test → (p_mean, p_low, p_high) per (date, ticker)
4. Replay strategy on EVAL period (uniform 1/K) → realized return series r_i  (§6.2)
5. Fit VinceOptimalF on r_i
6. Fetch OHLCV via data_pipelines for 92 tickers, window [test_start - 5 BD, test_end + 50 BD]
7. Build DataHandler-compatible feed dict
8. Construct Backtest(fill_mode="next_open", lookback=5, initial_cash=100_000)
9. Run via run_strategy()
10. Compute 3 benchmarks (§6.7) over same window from $100K
11. Persist all outputs + append registry row (CONVENTIONS § "Registry CSV")
```

### 6.7 Benchmarks — `scripts/backtests/benchmarks.py`

Three benchmarks, same window, same $100K start:

1. **NDX cap-weighted buy-and-hold** — fetch NDX or QQQ ETF as proxy; single position. *Risk*: NDX availability in `data_pipelines` — see §10 Q3.
2. **92-ticker equal-weight basket** — $100K/92 to each strategy-universe ticker at `test_start`; hold.
3. **Equal-weight top-K (Kelly ablation)** — same selection + exits as strategy, but uniform 1/K sizing.

Each: `start_$, end_$, total_return_pct, cagr, max_dd`.

## 7. Reporting

Full memo template, registry CSV schema, numbering rules, and file-system layout are in [`CONVENTIONS.md`](CONVENTIONS.md). Summary of what the memo for THIS back-test (`_001_cell5_bayesian_kelly.md`) will deliver:

**Headline table** (leads the Results section):

```
Starting capital: $100,000  |  Window: 2025-03-26 → 2025-12-26 (~9 months)

                                  End $        Total %    CAGR    Max DD
─────────────────────────────────────────────────────────────────────────
Strategy (Bayesian + Kelly c=0.5) $X         +X.X%      X.X%     -X.X%
NDX buy-and-hold (cap-weighted)   $Y         +Y.Y%      Y.Y%     -Y.Y%
92-ticker equal-weight basket     $Z         +Z.Z%      Z.Z%     -Z.Z%
Equal-weight top-K (no Kelly)     $W         +W.W%      W.W%     -W.W%  (ablation)
```

Plus equity-curve overlay (4 lines, same $100K start), drawdown trajectory, gross-exposure trajectory, per-pick sample table, sensitivity sweep over `c ∈ {0.25, 0.5, 1.0}` and `K ∈ {1, 3, 5}`, and **R-Precision@K of REALIZED picks vs the published R-p@3 = 0.7556** (the meta-validation: does the metric translate to a tradeable strategy?).

Registry row appended to `results/backtests/data/backtest_summary.csv` per the schema in [CONVENTIONS.md](CONVENTIONS.md).

## 8. Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Bayesian calibrator's upper bin (`p_raw > 0.7`) sparse on val | M=10 → M=5 quantile bins if `n_i < 50` for upper bin. **Checkpoint after first fit** (§9 step 5). |
| R2 | Cell-5's `p_raw` overconfident → Bayes shrinks `p_mean` toward base rate → fewer picks pass Kelly breakeven | Expected outcome. Surface in memo. If <10 picks total, sensitivity sweep on `c` or breakeven. |
| R3 | DD exits trigger on close, fill at next open → gap-down realizes loss > 5% (not strictly bounded as label assumes) | Vince's empirical f handles this naturally (uses realized eval losses). Document as a real risk in memo. |
| R4 | NDX identifier not in `data_pipelines` | Fall back to QQQ ETF as proxy. If neither, document the gap; ship basket + ablation. See §10 Q3. |
| R5 | Universe survivorship — roster as-of `test_start` ≠ as-of-fit | Use 92-ticker roster from cell-5's `metrics.json::data.n_tickers_used`. Don't refresh. Known gbdt v1 limitation, not back-test scope. |
| R6 | Look-ahead in calibrator fit or Vince fit | Verify per-ticker val_end < per-ticker eval_start < `test_start` (segment hierarchy honored, not just calendar-window check). Assert in code on each ticker. |
| R7 | Lot-size rounding zeros out small picks | Engine emits `lot_size_audit` in `info`; track rounded-to-zero count; document if non-trivial. |
| R8 | Tiny-model `p_raw` ties → quantile bins collapse | Surfaced during plan review on cell-5 regen eval: 18,400 rows over 226 distinct `p_raw` values; naive `np.quantile(p_raw, 10)` produces a duplicate edge `(0.3852, 0.3852)` because >10% of rows sit at one exact `p_raw`. Calibrator handles via `pd.cut(duplicates='drop')` + `min_bin_size=20` merge + `min_effective_bins=3` floor (see §5.1). If fewer than 3 bins survive on a future cell, the calibrator raises and the caller falls back to a simpler scheme (e.g. one bin per distinct `p_raw` for very tiny models, or a non-Bayesian calibrator). **Checkpoint output** at §9 step 5 must report `effective_n_bins`. |
| R9 | Bayesian recalibrator stacked on gbdt's isotonic-calibrated `p_calibrated` (val-fit) | `p_calibrated` in `predictions/val.csv` is what gbdt's internal `conditional_isotonic` produced from `p_raw` (fit on val itself — leak-prone for the isotonic, but that's the gbdt module's concern, not ours). Our Bayesian recalibrator fits on top of `p_calibrated` (or `p_raw` if isotonic was pass-through per V1.3 anti-AUC handling — check `metrics.json::calibration.fitted` flag and pick the right input column). On cell-5 specifically, V1.3 native pass-through was active so `p_calibrated == p_raw` and there's no stacking. On future cells where isotonic IS active, the Bayesian recalibrator's job is "produce credible bands on already-shrunk probabilities" — the bands will be narrower (less posterior uncertainty after isotonic smoothing), which is correct. **Checkpoint output** at §9 step 5 should report which input column was consumed + the ECE delta val-vs-eval to flag if val→eval generalization is broken. |
| R-cost | Zero-cost back-test inflates returns vs reality | Memo Caveats section quantifies: "At 5 bps/side and observed turnover of N trades, expected drag ≈ X bps/yr." |

## 9. Sequencing

| # | Task | Output | Task-list ID |
|---|---|---|---|
| 1 | This branch + this plan + supporting docs | The 4 docs already on this branch | #12 (done in plan-only mode) |
| 2 | Scaffold `src/calibration/` | Package skeleton + pyproject entry + tests dir + importable | #13 |
| 3 | Scaffold `src/trading_strategies/` | Same + `sizing/` subpackage | #14 |
| 4 | Implement `BetaBinomialBucketed` + diagnostics + tests | `src/calibration/` v1 complete | #15 |
| 5 | **CHECKPOINT** — fit on cell-5 VAL, generate reliability figure + report `effective_n_bins` (R8) + ECE on val vs eval (R9) | Review credible-band quality + bin-survival count + whether val→eval generalization is honest before continuing | #16 |
| 6 | Implement sizers (`VinceOptimalF`, `DiscreteBoundedLossKelly`, `FixedFraction`, `caps`) + tests | `src/trading_strategies/sizing/` v1 complete | #17 |
| 7 | Implement `TopKWithLabelExit` + tests | Strategy class complete | #18 |
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
