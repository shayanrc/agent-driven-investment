# Task #138 — H=25 cross-market combined memo (4 cells)

**Cells**: 4 H=25 cross-market replications of the H=25 short-horizon signal pattern first surfaced in PR #28.

| Cell | Universe | Direction | Threshold | Horizon | Drawdown | Status |
|---|---|---:|---:|---:|---:|---|
| A | nasdaq100 | up | +10% | 25d | 5% | landed (PR #28) |
| B | sp500    | up | +10% | 25d | 5% | landed (retry #3 — see `[[feedback-agent-pkill-antipattern]]` for the cross-process kill that took 2 prior attempts) |
| C | nifty50  | up | +10% | 25d | 5% | landed (this run) |
| D | nifty100 | up | +10% | 25d | 5% | landed (this run) |

**Why this run set**: PR #28's H=25 nasdaq cell revealed *top-tail signal* (test AUC 0.51 but per-day P@1 = 1.78× lift) hidden behind a null AUC. The question that drove these 4 cells: does the short-horizon signal generalize across **markets** (US vs NSE) and across **panel sizes** within each market?

**Result location**: `results/gbdt/experiments/{nasdaq100,nifty50,nifty100,sp500}_up_10pct_25d_dd5pct/`.

## Headline metrics — test segment

R-precision (per-day picks = R(d); equals recall@R = precision@R) is the **primary cross-cell metric** because fixed-k P@k unfairly penalizes small panels (NIFTY-50 has ~8 positives/day on average; nasdaq has ~25 → P@5 captures wildly different fractions of available signal in each).

| Cell | Test AUC | Test Brier vs base | **Test R-prec mean** | **Test R-prec weighted** | **Lift weighted** |
|---|---:|---:|---:|---:|---:|
| A nasdaq H=25 | 0.5111 | -0.005 | 0.464 | 0.399 | **1.46×** |
| **B sp500 H=25** | **0.5836** | **+0.003** ✓ | **0.361** | **0.407** | **1.54×** |
| C nifty50 H=25 | 0.7327 | +0.009 ✓ | 0.211 | 0.416 | **2.12×** |
| D nifty100 H=25 | 0.6890 | +0.011 ✓ | 0.230 | 0.419 | **2.06×** |

**Reading**: all 4 cells show meaningful lift over baseline by weighted R-precision (1.46–2.12×). The NSE cells beat US on weighted R-precision (opposite ranking from per-day P@1, which had favored US). Cell B (sp500) is the **strongest AUC of the 4** (0.58, above the [0.45, 0.55] null band) — the bigger panel + more positives per day make the bulk-rank task easier; not coincidentally, sp500 has the cleanest per-day rprec distribution (see next table).

## R-precision vs P@k — the metric framing reset

The P@1 numbers from the earlier per-cell memos painted a misleading "NSE anti-predictive top" story:

| Cell | Test per-day P@1 | Test per-day P@5 |
|---|---:|---:|
| A nasdaq | 0.486 (1.78× lift) | 0.430 (1.57× lift) |
| C nifty50 | 0.119 (0.67× lift — below base!) | 0.206 (1.15× lift) |
| D nifty100 | 0.106 (0.56× lift) | 0.246 (1.30× lift) |

By P@1, nasdaq looked dominant. By R-precision (variable per-day K), all three show real signal of roughly comparable magnitude. **Mechanism**: P@1 isolates the single most-confident pick per day. On a panel with very few positives per day (nifty50: ~8/day, nasdaq: ~25/day), one wrong pick is much more visible in a P@1 average than in an R-precision average over all R(d) picks. The R-precision framing matches actual trading rule semantics more closely ("pick the model's top items each day, sized to the day's signal").

## Per-day rprec distribution (test segment)

| Cell | p10 | p25 | p50 | p75 | p90 | R(d) mean |
|---|---:|---:|---:|---:|---:|---:|
| A nasdaq | 0.095 | 0.182 | 0.367 | 0.692 | 1.000 | ~20 |
| **B sp500** | **0.198** | **0.270** | **0.350** | **0.453** | **0.557** | **128** |
| C nifty50 | 0.000 | 0.000 | 0.125 | 0.362 | 0.605 | ~9 |
| D nifty100 | 0.000 | 0.000 | 0.143 | 0.429 | 0.565 | ~9 |

**Reading**:
- **sp500 has the tightest, most uniform distribution** (p10=0.20, p90=0.56). Bigger panel + higher per-day positive count smooths out the per-day variance. No "zero-signal days" at all.
- **nasdaq is fan-shaped** — wide range (0.10–1.00) because R(d) is small (~20) so per-day rprec is noisy.
- **NSE has a sharp U-shape** — bottom 25% of test days score literally 0 (model picks nothing right); top 25% score 0.36+. The big mean-vs-weighted R-precision gap on NSE is driven by this U-shape.
- The **practical implication**: sp500 → strategy can be sized any day. nasdaq → strategy gets best days some of the time. NSE → strategy needs a regime-filter to skip the zero-days.

## Per-ticker pattern — cross-cell recurrence

When sized by R-precision picks (not fixed-K), the systematic anti-predictive tickers from EVAL recur across BOTH NSE cells:

| Ticker | nifty50 eval picks/hit_rate/base | nifty100 eval picks/hit_rate/base | Verdict |
|---|---:|---:|---|
| NSE:HDFCBANK | 43 / 0.000 / 0.075 | 63 / 0.000 / 0.075 | Persistent over-pick, 0 hits in 106 combined picks |
| NSE:WIPRO    | 30 / 0.000 / 0.040 | 31 / 0.000 / 0.040 | Persistent over-pick, 0 hits in 61 combined picks |
| NSE:COALINDIA | 51 / 0.020 / 0.105 | 23 / 0.000 / 0.105 | Persistent over-pick, 1 hit in 74 combined picks |

**These are all low-volatility, range-bound large-caps that rarely move ±10% in 25 trading days.** Model assigns them moderate calibrated probability (~0.2–0.3) on many days; reality almost never delivers. The model isn't "anti-predictive" in a malicious sense — it's *miscalibrated upward* on this specific cohort.

**Cross-market: the same cohort exists in sp500.** Top-anti-predictive sp500 test tickers (picks ≥ 20, anti_score > 0.3): NYSE:IQV (54/0/0.027 = 100% anti), NYSE:CRM (31/0/0.053 = 100% anti), NYSE:SYF (22/1/0.293), NYSE:EPAM (57/1/0.053), NYSE:VEEV (28/1/0.107), NYSE:NTAP, NYSE:NOW, NASDAQ:INTU, NASDAQ:VRTX. These are **software / healthcare-data / payments large-caps** — structurally similar to the NSE low-vol cohort (range-bound, moderate-probability over-picks).

**Confirms the V2 per-ticker-features hypothesis is cross-market**, not NSE-specific. The same calibration failure mode (range-bound stable large-caps get systematically over-picked at moderate probability) appears in both markets. Asset-agnostic features cannot fix this — the model needs per-ticker context (realized-volatility percentile, max-move-in-25d percentile, sector indicator) to know that "this stock is structurally range-bound" and downweight its probability accordingly.

In the TEST segment, no ticker meets `anti_score > 0.5 & picks ≥ 10` for either nifty50 or nifty100 — the eval-time anti-picks didn't survive into test under R-precision selection. This is consistent with isotonic calibration working as designed (both NSE cells fired the isotonic correction; nasdaq did not).

**Most-picked tickers in nifty100 test that DO work**:
- NSE:ABB — 45 picks, **73% hit rate** (61% base, +12pp). Strong positive.
- NSE:ADANIPORTS — 37 picks, **70% hit rate** (35% base, +35pp). Strong positive.
- NSE:TRENT — 46 picks, **52% hit rate** (32% base, +20pp). Strong positive.
- NSE:BAJFINANCE — 38 picks, **45% hit rate** (23% base, +22pp). Strong positive.

When the model picks volatile mid/large-caps, it's right substantially above base rate. The signal is real; the failure mode is one persistent over-pick cohort.

## Mechanistic reading

1. **The H=25 signal generalizes across markets** — all 3 landed cells show lift > 1.4× on weighted R-precision. SP500 retry pending.
2. **AUC alone is a misleading null-signal flag.** The CLAUDE.md rule "AUC ∈ [0.45, 0.55] is a null-signal flag" misclassified nasdaq H=25 as null (it has top-tail signal). Proposed amendment per PR #28 memo stands; need to ALSO add R-precision as a non-null signal check.
3. **Calibration matters more at higher base rates.** NSE cells fired isotonic correction (Spiegelhalter z = +5.93 / -4.86 → both well outside the ±2 acceptable band). Nasdaq stayed on native sigmoid (z = 1.59). The NSE under-/over-confidence drives the P@1 wobble that R-precision smooths over.
4. **Low-volatility large-caps are a recurring feature pathology.** HDFCBANK, WIPRO, COALINDIA recur across both NSE cells with ~0% hit rates on 30–100 combined picks. The F-family features (run-up, volatility, beta) likely fire moderate-confidence signals on these tickers based on noise, and the model has no way to learn "this stock is structurally range-bound" without per-ticker features (gbdt v1 is asset-agnostic by design).

## Implications — revised plan vs PR #28's recommendations

### 1. Keep the short-horizon US sweep pivot (was PR #28 recommendation #1)

Still valid. H ∈ {5, 10, 25, 50} × {nasdaq100, sp500, russell1000} × +10%/dd5% = 12 cells. Sequencing per task #107.

### 2. Amend the CLAUDE.md null-signal rule — UPDATED proposal

PR #28 proposed:
> AUC ∈ [0.45, 0.55] AND per-day P@5 lift < 1.2x = null signal flagged.

This memo updates the rule to use R-precision:
> AUC ∈ [0.45, 0.55] AND **weighted R-precision lift < 1.2x** = null signal flagged.
> AUC ∈ [0.45, 0.55] AND **weighted R-precision lift > 1.5x** = **top-tail signal; AUC understates; investigate the prediction-extreme regime.**

R-precision is the better diagnostic because P@k requires choosing K (sensitivity to panel size); R-precision is panel-invariant.

### 3. V2 per-ticker features become the natural follow-up

The HDFCBANK/WIPRO/COALINDIA persistent over-pick cohort is a textbook case where per-ticker baseline features (historical realized-volatility percentile, historical max-move-in-25d percentile, sector indicator) would directly fix the calibration. Promote V2 TBD (docs/gbdt/V2_TBD.md) to a real V2_PLAN if the sweep results across H ∈ {5, 10, 50} confirm the same pathology.

### 4. R-precision into runner standard report — optional follow-up

This memo computes R-precision post-hoc via `scripts/gbdt/compute_r_precision.py`. Baking it into `src/gbdt/topk_diagnostics.py` (alongside the existing P@k computation) would make every future run emit R-precision in `metrics.json::segment_diagnostics`. Useful but not urgent — separate PR if/when we want it.

## Verdict

- **Cell A nasdaq verdict**: TOP-TAIL signal (R-prec weighted 1.46×); confirmed PR #28's earlier P@1 finding. AUC 0.51 in [0.45, 0.55] null band — by the old single-AUC rule this would have been dismissed.
- **Cell B sp500 verdict**: SIGNAL (R-prec weighted 1.54×, AUC 0.58 — cleanest of the 4). Largest panel (486 tickers) + most positives per day (R(d) mean 128) → uniform per-day distribution (no zero-signal days). Strongest cell for a trading rule. **Retry #3 landed cleanly after 2 prior attempts**: attempt #1 died at 2h25m with signal 16; attempt #2 was killed at ~5min by a cross-process pkill from a concurrent sub-agent (see `[[feedback-agent-pkill-antipattern]]`).
- **Cell C nifty50 verdict**: SIGNAL (R-prec weighted 2.12×). The P@1 anti-predictive picture was a fixed-K artifact.
- **Cell D nifty100 verdict**: SIGNAL (R-prec weighted 2.06×). Confirms the H=25 short-horizon hypothesis transfers to NSE.

- **Methodology verdict**: R-precision belongs at the top of the diagnostic stack; demote per-day P@k to a secondary metric. Update the null-signal rule in CLAUDE.md.

- **Plan verdict**: 12-cell short-horizon US sweep proceeds. V2 per-ticker features become the prime candidate for the next architecture iteration based on the HDFCBANK/WIPRO/COALINDIA pattern.

Cross-links: PR #28 (nasdaq H=25), PR #27 (uniqueness-fix Sweep #1 rerun), task #107 (sweep — revise scope), task #113 (re-run experiments — partially fulfilled), `[[feedback-agent-pkill-antipattern]]` (concurrent-process lesson from this work).
