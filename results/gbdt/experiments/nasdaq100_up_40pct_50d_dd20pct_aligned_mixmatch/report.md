# gbdt experiment — nasdaq100_up_40pct_50d_dd20pct_aligned_mixmatch

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=1 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `50`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73309 (independent events ≈ 740.9; overlap-inflation 98.95×)
- val rows: 36800 (independent events ≈ 371.7; overlap-inflation 99.00×)
- eval rows: 18400 (independent events ≈ 185.9; overlap-inflation 99.00×)
- test rows: 9200 (independent events ≈ 92.9; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.049
- positive prevalence (eval): 0.028

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
| 0 | 30 | 0.0346 | 0.0402 | 0.0056 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`
- tie-break path: `strict_val_brier` — Strict val_brier argmin (no tie-break entered)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -19.080
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0267 | 0.0275 | +0.0007 | 0.1244 | 0.8100 |
| test | 0.0315 | 0.0354 | +0.0038 | 0.1133 | 0.9228 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0283

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4518 | 0.0283 | 200 | 75 | 166 | 34 / 200 / 200 |
| 5 | 0.2699 | 0.0283 | 1000 | 129 | 478 | 164 / 200 / 200 |
| 10 | 0.4489 | 0.0283 | 2000 | 233 | 519 | 199 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0283 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0283 | 5 | 2 | 5 |
| 10 | 0.4000 | 0.0283 | 10 | 4 | 10 |

### test — n_rows=9200, base_rate=0.0367

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3333 | 0.0367 | 100 | 30 | 90 | 10 / 100 / 100 |
| 5 | 0.3583 | 0.0367 | 500 | 110 | 307 | 62 / 100 / 100 |
| 10 | 0.7071 | 0.0367 | 1000 | 239 | 338 | 100 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0367 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0367 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0367 | 10 | 0 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=18400, Q_days=166, base_rate=0.0283

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.4518 | 0.0283 | 166 |
| 3 | 0.2560 | 0.0283 | 166 |
| 5 | 0.2520 | 0.0283 | 166 |
| 10 | 0.4259 | 0.0283 | 166 |
| 20 | 0.7485 | 0.0283 | 166 |

### test — n_rows=9200, Q_days=90, base_rate=0.0367

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.3333 | 0.0367 | 90 |
| 3 | 0.3130 | 0.0367 | 90 |
| 5 | 0.3674 | 0.0367 | 90 |
| 10 | 0.7346 | 0.0367 | 90 |
| 20 | 0.9261 | 0.0367 | 90 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 200 | 81 | 0.4050 |
| NASDAQ:WBD | 145 | 0 | 0.0000 |
| NASDAQ:DXCM | 122 | 0 | 0.0000 |
| NASDAQ:MDB | 106 | 0 | 0.0000 |
| NASDAQ:DDOG | 80 | 15 | 0.1875 |
| NASDAQ:CRWD | 64 | 13 | 0.2031 |
| NASDAQ:TSLA | 61 | 4 | 0.0656 |
| NASDAQ:ON | 59 | 0 | 0.0000 |
| NASDAQ:TEAM | 41 | 3 | 0.0732 |
| NASDAQ:TTD | 33 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:WBD | 145 | 0 | 0.0000 |
| NASDAQ:DXCM | 122 | 0 | 0.0000 |
| NASDAQ:MDB | 106 | 0 | 0.0000 |
| NASDAQ:ON | 59 | 0 | 0.0000 |
| NASDAQ:TTD | 33 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 86 | 55 | 0.6395 |
| NASDAQ:AMD | 78 | 0 | 0.0000 |
| NASDAQ:MDB | 66 | 0 | 0.0000 |
| NASDAQ:NVDA | 55 | 0 | 0.0000 |
| NASDAQ:WBD | 44 | 20 | 0.4545 |
| NASDAQ:CRWD | 34 | 4 | 0.1176 |
| NASDAQ:MRVL | 32 | 6 | 0.1875 |
| NASDAQ:TSLA | 28 | 14 | 0.5000 |
| NASDAQ:DXCM | 21 | 0 | 0.0000 |
| NASDAQ:TEAM | 21 | 9 | 0.4286 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMD | 78 | 0 | 0.0000 |
| NASDAQ:MDB | 66 | 0 | 0.0000 |
| NASDAQ:NVDA | 55 | 0 | 0.0000 |
| NASDAQ:DXCM | 21 | 0 | 0.0000 |
| NASDAQ:PDD | 9 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 64 | 0.2207 | 0.0283 |
| 2024Q1 | 305 | 48 | 0.1574 | 0.0283 |
| 2024Q2 | 315 | 16 | 0.0508 | 0.0283 |
| 2024Q3 | 90 | 1 | 0.0111 | 0.0283 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 35 | 0.1522 | 0.0367 |
| 2024Q4 | 270 | 75 | 0.2778 | 0.0367 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.1342 | 0.0087 | 0.0184 | `True` |
| test | 9200 | 0.0009 | 0.1342 | 0.0190 | 0.0340 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=19.08); shipped as `isotonic`. Brier vs base-rate: +0.0007 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
