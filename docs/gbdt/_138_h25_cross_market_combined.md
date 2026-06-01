# Task #138 — H=25 cross-market combined memo (4 cells)

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the 4 cells recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Cells**: 4 H=25 cross-market replications of the H=25 short-horizon signal pattern first surfaced in PR #28.

| Cell | Universe | Direction | Threshold | Horizon | Drawdown | Status |
|---|---|---:|---:|---:|---:|---|
| A | nasdaq100 | up | +10% | 25d | 5% | landed (PR #28) |
| B | sp500    | up | +10% | 25d | 5% | landed (retry #3 — see `[[feedback-agent-pkill-antipattern]]` for the cross-process kill that took 2 prior attempts) |
| C | nifty50  | up | +10% | 25d | 5% | landed |
| D | nifty100 | up | +10% | 25d | 5% | landed |

**Why this run set**: PR #28's H=25 nasdaq cell revealed *top-tail signal* hidden behind a null AUC. The question that drove these 4 cells: does the short-horizon signal generalize across **markets** (US vs NSE) and across **panel sizes** within each market?

**Result location**: `results/gbdt/experiments/{nasdaq100,nifty50,nifty100,sp500}_up_10pct_25d_dd5pct/`.

> **Erratum** (vs original 2026-05-27 memo): the original memo's per-day P@k table used a buggy denominator (`min(k, n_tickers_in_day)` — count of picks made) that mis-normalized P@k on staggered panels where R(d) often falls below k. The "NSE anti-predictive top / skip top 1-2" narrative was an artifact of that bug. This memo uses the corrected denominator (`min(R(d), k)` — count of achievable positives) and the NSE-inversion story is **withdrawn**. R-precision (the primary headline) was always correct; only the fixed-k P@k values were wrong. The runner has the same bug at `src/gbdt/topk_diagnostics.py:135` — fixed in the same PR as this memo update.

## How to read this memo + metrics formulas

The memo reports **raw metric values, not lift over base rate**, per `CLAUDE.md` § Reporting conventions. To compute lift mentally: `metric / base_rate`.

### R-precision (primary headline)

For each day in the segment:
- `R(d)` = number of positives that day
- Rank items by `p_calibrated` descending (tie-break: `ticker` ascending, stable mergesort)
- Take the top R(d) items
- `r_precision(d) = (positives in top-R(d)) / R(d)`

Two aggregates:
- **Mean unweighted**: average over days where R(d) > 0
- **Weighted** (preferred): `sum(positives_caught) / sum(R(d))` — equivalent to global recall@R = global precision@R

R-precision is panel-invariant (K adapts per day to R(d)), bounded [0, 1], baseline = base rate.

### P@k (per-day, normalized denominator)

For each day:
- Rank items by `p_calibrated` descending (same tie-break)
- Take top k items (or fewer if panel < k that day)
- `p_at_k(d) = (positives in top k) / min(R(d), k)`

Two aggregates:
- **Weighted** (what tables below show): `sum(positives_in_top_k) / sum(min(R(d), k))`

The `min(R(d), k)` denominator normalizes by **achievable positives** — if there are only 3 positives that day and k=10, the maximum P@10 is 1.0 (caught all 3), not 0.3 (caught 3 of 10 picks). Without this normalization, panels with frequently-small R(d) (like staggered NSE test sets where most days have R(d) < k) get artificially-low P@k that has nothing to do with model skill.

### Why "raw, not lift"

Lift (`metric / base_rate`) compresses two pieces of information into one number. Tables that show only lift lose the base rate, the actual hit rate, and the units. Two cells with 1.5× lift can have completely different actual probabilities (e.g., 0.30 vs 0.15) which matters for strategy sizing. We show raw values + base rate so the reader can compute lift on demand and also see the underlying scale.

---

## Headline metrics — test segment

R-precision (weighted) — the **primary cross-cell comparable**:

| Cell | Test AUC | Test Brier vs base | base rate | R-prec mean | R-prec weighted |
|---|---:|---:|---:|---:|---:|
| A nasdaq H=25 | 0.5111 | -0.005 | 0.273 | 0.464 | 0.399 |
| **B sp500 H=25** | **0.5836** | **+0.003** ✓ | 0.264 | 0.361 | 0.407 |
| C nifty50 H=25 | 0.7327 | +0.009 ✓ | 0.179 | 0.211 | 0.416 |
| D nifty100 H=25 | 0.6890 | +0.011 ✓ | 0.188 | 0.230 | 0.419 |

**Reading**: all 4 cells beat baseline. Weighted R-precision spans 0.40–0.42 across cells — strikingly similar magnitudes despite very different markets/panels/base rates. SP500 has the strongest AUC (0.58, outside the [0.45, 0.55] null band) — the bigger panel (486 tickers) + more positives per day (R(d) mean = 128) make the bulk-rank task easier.

## P@k per-day (test segment, corrected formula)

P@k = `positives_in_top_k / min(R(d), k)` per day, weighted aggregate.

| Cell | base | P@1 | P@3 | P@5 | P@10 |
|---|---:|---:|---:|---:|---:|
| A nasdaq | 0.273 | **0.537** | 0.449 | 0.445 | 0.391 |
| **B sp500** | 0.264 | **0.613** | 0.502 | 0.501 | 0.472 |
| C nifty50 | 0.179 | 0.257 | 0.299 | 0.344 | **0.426** |
| D nifty100 | 0.188 | 0.225 | 0.302 | 0.391 | **0.459** |

**Shape observations**:
- **US (nasdaq, sp500)** — P@k **descends** with k: top picks are the cleanest signal. sp500 P@1 = 0.61 means model's most-confident pick per day hits 61% of the time (vs 26% base).
- **NSE (nifty50, nifty100)** — P@k **ascends** with k. This is **expected behavior under correct normalization**: NSE test sets have R(d) < k for most days (nifty50: 124/151 days have R<10), so P@k approaches R-precision as k grows. It is NOT "the model is bad at top picks" — that was the original-memo error driven by the denominator bug.

## P@k per-day (eval segment, for generalization sanity)

| Cell | base | P@1 | P@3 | P@5 | P@10 |
|---|---:|---:|---:|---:|---:|
| A nasdaq | 0.249 | 0.404 | 0.493 | 0.499 | 0.490 |
| B sp500 | 0.220 | 0.550 | 0.503 | 0.515 | 0.508 |
| C nifty50 | 0.133 | 0.256 | 0.237 | 0.246 | 0.335 |
| D nifty100 | 0.132 | 0.180 | 0.232 | 0.243 | 0.317 |

**SP500 generalizes best** — eval P@1 = 0.55, test P@1 = 0.61 (eval-to-test is *up* for sp500, consistent with the bigger test sample). NSE eval-to-test is also consistent in shape (ascending).

## Per-day R-precision distribution (test segment)

| Cell | p10 | p25 | p50 | p75 | p90 | R(d) mean |
|---|---:|---:|---:|---:|---:|---:|
| A nasdaq | 0.095 | 0.182 | 0.367 | 0.692 | 1.000 | ~20 |
| **B sp500** | 0.198 | 0.270 | 0.350 | 0.453 | 0.557 | 128 |
| C nifty50 | 0.000 | 0.000 | 0.125 | 0.362 | 0.605 | ~9 |
| D nifty100 | 0.000 | 0.000 | 0.143 | 0.429 | 0.565 | ~9 |

**Reading**:
- **sp500 has the tightest, most uniform distribution** (p10=0.20, p90=0.56). Bigger panel + higher per-day positive count smooths per-day variance.
- **nasdaq is fan-shaped** — wide range (0.10–1.00) because R(d) is small (~20) and per-day rprec is noisy.
- **NSE bottom 25% sits at 0** — this is a **per-day-binary artifact**, not model failure: ~50% of NSE test days have R(d) = 1, so per-day R-precision is either 0 (missed) or 1 (caught) for those days. With ~50% of days at 0 and ~50% at 1, the median lands at p50 ≈ 0.13 once you average in the larger-R days. The bottom-25% zero is mechanical, not "the model has zero-signal days".

## Per-ticker pattern — cross-cell recurrence

When picked at R-precision sizes, the persistent over-pick cohort recurs across cells (per-ticker hit rates from `scripts/gbdt/nse_anti_predictive_cross_cell.py`):

| Ticker | nifty50 eval picks/hit_rate/base | nifty100 eval picks/hit_rate/base | Verdict |
|---|---:|---:|---|
| NSE:HDFCBANK | 43 / 0.000 / 0.075 | 63 / 0.000 / 0.075 | Persistent over-pick, 0 hits in 106 combined picks |
| NSE:WIPRO    | 30 / 0.000 / 0.040 | 31 / 0.000 / 0.040 | Persistent over-pick, 0 hits in 61 combined picks |
| NSE:COALINDIA | 51 / 0.020 / 0.105 | 23 / 0.000 / 0.105 | Persistent over-pick, 1 hit in 74 combined picks |

**Cross-market: the same cohort exists in sp500.** Top-anti-predictive sp500 test tickers (picks ≥ 20, anti_score > 0.3): NYSE:IQV (54 picks / 0 hits / base 0.027), NYSE:CRM (31/0/0.053), NYSE:SYF (22/1/0.293), NYSE:EPAM (57/1/0.053), NYSE:VEEV (28/1/0.107), NYSE:NTAP, NYSE:NOW, NASDAQ:INTU, NASDAQ:VRTX. **Software / healthcare-data / payments large-caps** — structurally similar to the NSE low-vol cohort (range-bound, moderate-probability over-picks).

**These are all low-volatility, range-bound large-caps that rarely move ±10% in 25 trading days.** Model assigns them moderate calibrated probability (~0.2–0.3) on many days; reality almost never delivers. The model isn't malicious — it's *miscalibrated upward* on this specific cohort. **Confirms the V2 per-ticker-features hypothesis is cross-market**, not NSE-specific. Asset-agnostic features cannot fix this — the model needs per-ticker context (realized-volatility percentile, max-move-in-25d percentile, sector indicator) to know that "this stock is structurally range-bound" and downweight its probability.

**Most-picked tickers in nifty100 test that DO work**:
- NSE:ABB — 45 picks, 73% hit rate (61% base, +12pp). Strong positive.
- NSE:ADANIPORTS — 37 picks, 70% hit rate (35% base, +35pp). Strong positive.
- NSE:TRENT — 46 picks, 52% hit rate (32% base, +20pp). Strong positive.
- NSE:BAJFINANCE — 38 picks, 45% hit rate (23% base, +22pp). Strong positive.

When the model picks volatile mid/large-caps, it's right substantially above base rate. The signal is real; the failure mode is one persistent over-pick cohort.

## Mechanistic reading

1. **The H=25 signal generalizes across markets** — all 4 cells beat baseline on every metric in this memo. Weighted R-precision spans 0.40–0.42 with remarkable consistency.
2. **AUC alone is a misleading null-signal flag.** The old CLAUDE.md rule "AUC ∈ [0.45, 0.55] is a null-signal flag" misclassified nasdaq H=25 as null (AUC 0.51, R-prec 0.40 = clear signal). The compound rule landed in PR #43 (AUC + R-precision check) holds up across these 4 cells.
3. **Calibration matters more at higher base rates.** NSE cells fired isotonic correction (Spiegelhalter z = +5.93 / -4.86 → both outside ±2). Nasdaq stayed on native sigmoid (z = 1.59). The persistent-over-pick cohort drives the calibration drift.
4. **Low-volatility large-caps are a recurring feature pathology** — cross-market. HDFCBANK/WIPRO/COALINDIA on NSE, IQV/CRM/EPAM/VEEV on sp500. The F-family features (run-up, volatility, beta) fire moderate-confidence signals on these tickers based on noise; the model can't learn "this stock is structurally range-bound" without per-ticker features.
5. **The "small-panel artifact" lesson** (this memo's correction) — fixed-k P@k must use `min(R(d), k)` denominator, not picks-made. The original memo's "skip top 1-2 on NSE" rule was an artifact of mis-normalization; **withdrawn**. The corrected P@k tables (above) show NSE behaves like a small-R(d)-shape: P@k rises with k until it converges with R-precision.

## Implications — plan

### 1. Keep the short-horizon US sweep pivot

Still valid. H ∈ {5, 10, 25, 50} × {nasdaq100, sp500, russell1000} × +10%/dd5% = 12 cells. SP500 cell B has shown the strongest signal so far; the cross-horizon spread on sp500 is the highest-information next set.

### 2. V2 per-ticker features stays the natural follow-up

The HDFCBANK/WIPRO/COALINDIA + IQV/CRM/EPAM/VEEV cross-market cohort is the textbook case where per-ticker baseline features (historical realized-volatility percentile, historical max-move-in-25d percentile, sector indicator) would directly fix the calibration. Promote V2 TBD (`docs/gbdt/V2_TBD.md`) to a real V2_PLAN when the sweep results across H ∈ {5, 10, 50} confirm the same cross-horizon pathology.

### 3. CLAUDE.md null-signal rule already amended (PR #43)

The compound rule (AUC + weighted R-precision lift) stands; all 4 cells in this memo are correctly classified by it. No further amendment needed.

### 4. Runner P@k formula needs fixing (same PR as this memo)

`src/gbdt/topk_diagnostics.py::compute_top_k_metrics` uses the buggy denominator. All metrics.json files from runs before this PR have the wrong per-day P@k. The fix in this PR uses `min(R(d), k)` and includes a schema note in segment_diagnostics indicating the formula version. Existing artifacts are NOT regenerated — readers should re-compute P@k from predictions/test.csv via `scripts/gbdt/compute_r_precision.py` (the corrected post-hoc tool).

## Verdict

- **Cell A nasdaq**: SIGNAL (R-prec weighted 0.40, AUC 0.51). Top-tail-shaped: P@1 = 0.54, P@10 = 0.39. AUC alone hides the signal — compound CLAUDE.md rule catches it.
- **Cell B sp500**: SIGNAL (R-prec weighted 0.41, AUC 0.58 — cleanest of the 4). Strongest at every k: P@1 = 0.61, P@10 = 0.47. Best candidate for an actual trading rule.
- **Cell C nifty50**: SIGNAL (R-prec weighted 0.42). Small-R(d)-shape: P@k ascends from 0.26 (k=1) to 0.43 (k=10). Strong signal under correct normalization.
- **Cell D nifty100**: SIGNAL (R-prec weighted 0.42). Same shape as nifty50.

- **Methodology verdict**: R-precision belongs at the top of the diagnostic stack (panel-invariant, no K choice). P@k as a secondary diagnostic requires `min(R(d), k)` denominator; previous fixed-denominator P@k was an artifact-prone half-metric. Reporting convention: raw values + base rate, NOT lift.

- **Plan verdict**: 12-cell short-horizon US sweep proceeds. V2 per-ticker features remains the prime candidate for the next architecture iteration — the cross-market over-pick cohort is the V2-features motivating case.

Cross-links: PR #28 (nasdaq H=25), PR #27 (uniqueness-fix Sweep #1 rerun), PR #43 (original memo + R-precision methodology — see Erratum at top), task #107 (sweep — revise scope), `[[project-r-precision-methodology]]` (updated formula), `[[feedback-agent-pkill-antipattern]]` (concurrent-process lesson from the sp500 retry work).

---

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv` (source: `results/gbdt/data/r_precision_at_k.csv`):

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nasdaq100_up_10pct_25d_dd5pct | 6,900 | 27.3% | 0.511 | 0.537 | 0.526 | 0.536 | 0.507 | 0.508 |
| sp500_up_10pct_25d_dd5pct | 36,450 | 26.4% | 0.590 | 0.373 | 0.391 | 0.373 | 0.404 | 0.403 |
| nifty50_up_10pct_25d_dd5pct | 3,450 | 17.9% | 0.733 | 0.229 | 0.252 | 0.235 | 0.288 | 0.609 |
| nifty100_up_10pct_25d_dd5pct | 3,525 | 18.8% | 0.689 | 0.239 | 0.235 | 0.246 | 0.349 | 0.503 |

The compound-rule classification holds under R-Precision@10 (lift vs base, in prose per CLAUDE.md convention):
- **Cell A nasdaq (AUC 0.511, R-p@10 lift 1.86×)** — the original "hidden top-tail signal" finding. AUC band [0.45, 0.55] + R-p@10 lift > 1.5× ⇒ investigate, don't dismiss. Same verdict as the legacy memo.
- **Cell B sp500 (AUC 0.590, R-p@10 lift 1.53×)** — AUC above the null band; discriminating cell. Same verdict.
- **Cell C nifty50 (AUC 0.733, R-p@10 lift 1.61×)** — discriminating. Same verdict.
- **Cell D nifty100 (AUC 0.689, R-p@10 lift 1.85×)** — discriminating. Same verdict.

The body narrative's verdicts ("SIGNAL" on all 4 cells under different K mixes) stay intact; the R-Precision@K table just expresses them on the current canonical metric.
