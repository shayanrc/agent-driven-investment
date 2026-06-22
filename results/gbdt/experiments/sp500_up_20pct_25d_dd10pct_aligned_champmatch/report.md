# gbdt experiment — sp500_up_20pct_25d_dd10pct_aligned_champmatch

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=1 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `25`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 503
- tickers used: 486
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:PLTR, NYSE:CARR, NYSE:COIN, NYSE:EXE, NYSE:GEV, NYSE:HOOD, NYSE:KVUE, NYSE:OTIS, NYSE:Q, NYSE:SNDK, NYSE:SOLV, NYSE:VLTO
- train rows: 388173 (independent events ≈ 7923.3; overlap-inflation 48.99×)
- val rows: 194400 (independent events ≈ 3967.3; overlap-inflation 49.00×)
- eval rows: 97200 (independent events ≈ 1983.7; overlap-inflation 49.00×)
- test rows: 48600 (independent events ≈ 991.8; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.063
- positive prevalence (eval): 0.051

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2019-01-01`
- train: `2019-01-02` → `2022-03-04`
- val: `2022-03-07` → `2023-10-06`
- eval: `2023-10-09` → `2024-07-25`
- test: `2024-07-26` → `2024-12-16`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0341 | 0.0467 | 0.0126 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`
- tie-break path: `strict_val_brier` — Strict val_brier argmin (no tie-break entered)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -36.313
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0465 | 0.0480 | +0.0015 | 0.2016 | 0.8100 |
| test | 0.0374 | 0.0386 | +0.0012 | 0.1614 | 0.8044 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=97200, base_rate=0.0506

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4550 | 0.0506 | 200 | 91 | 200 | 0 / 200 / 200 |
| 5 | 0.3075 | 0.0506 | 1000 | 301 | 979 | 11 / 200 / 200 |
| 10 | 0.2872 | 0.0506 | 2000 | 525 | 1828 | 43 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0506 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0506 | 5 | 0 | 5 |
| 10 | 0.2000 | 0.0506 | 10 | 2 | 10 |

### test — n_rows=48600, base_rate=0.0402

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1700 | 0.0402 | 100 | 17 | 100 | 0 / 100 / 100 |
| 5 | 0.2312 | 0.0402 | 500 | 114 | 493 | 6 / 100 / 100 |
| 10 | 0.2334 | 0.0402 | 1000 | 204 | 874 | 34 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0402 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.0402 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.0402 | 10 | 2 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=97200, Q_days=200, base_rate=0.0506

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.4550 | 0.0506 | 200 |
| 3 | 0.3475 | 0.0506 | 200 |
| 5 | 0.3054 | 0.0506 | 200 |
| 10 | 0.2826 | 0.0506 | 200 |
| 20 | 0.2789 | 0.0506 | 200 |

### test — n_rows=48600, Q_days=100, base_rate=0.0402

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.1700 | 0.0402 | 100 |
| 3 | 0.2567 | 0.0402 | 100 |
| 5 | 0.2280 | 0.0402 | 100 |
| 10 | 0.2211 | 0.0402 | 100 |
| 20 | 0.2861 | 0.0402 | 100 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 198 | 97 | 0.4899 |
| NYSE:SMCI | 142 | 49 | 0.3451 |
| NASDAQ:TSLA | 82 | 3 | 0.0366 |
| NYSE:CCL | 77 | 10 | 0.1299 |
| NYSE:PSKY | 65 | 20 | 0.3077 |
| NASDAQ:WBD | 53 | 12 | 0.2264 |
| NYSE:SATS | 46 | 20 | 0.4348 |
| NASDAQ:AMD | 36 | 5 | 0.1389 |
| NYSE:XYZ | 36 | 4 | 0.1111 |
| NYSE:APA | 26 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:APA | 26 | 0 | 0.0000 |
| NYSE:HAL | 26 | 0 | 0.0000 |
| NYSE:PODD | 14 | 0 | 0.0000 |
| NASDAQ:ON | 10 | 0 | 0.0000 |
| NYSE:EQT | 5 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:SMCI | 85 | 13 | 0.1529 |
| NYSE:MRNA | 61 | 3 | 0.0492 |
| NYSE:CVNA | 47 | 33 | 0.7021 |
| NYSE:ALB | 39 | 9 | 0.2308 |
| NYSE:EL | 26 | 15 | 0.5769 |
| NASDAQ:AVGO | 25 | 1 | 0.0400 |
| NYSE:SATS | 25 | 1 | 0.0400 |
| NYSE:DLTR | 24 | 1 | 0.0417 |
| NYSE:PSKY | 24 | 0 | 0.0000 |
| NYSE:DELL | 22 | 8 | 0.3636 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:PSKY | 24 | 0 | 0.0000 |
| NASDAQ:AMD | 8 | 0 | 0.0000 |
| NYSE:FSLR | 8 | 0 | 0.0000 |
| NYSE:ALGN | 6 | 0 | 0.0000 |
| NASDAQ:AVGO | 25 | 1 | 0.0400 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 110 | 0.3793 | 0.0506 |
| 2024Q1 | 305 | 95 | 0.3115 | 0.0506 |
| 2024Q2 | 315 | 84 | 0.2667 | 0.0506 |
| 2024Q3 | 90 | 12 | 0.1333 | 0.0506 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 46 | 0.2000 | 0.0402 |
| 2024Q4 | 270 | 68 | 0.2519 | 0.0402 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 97200 | 0.0000 | 0.6452 | 0.0149 | 0.0354 | `True` |
| test | 48600 | 0.0000 | 0.3611 | 0.0152 | 0.0287 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=36.31); shipped as `isotonic`. Brier vs base-rate: +0.0015 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
