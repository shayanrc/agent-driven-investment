# gbdt experiment — sp500_up_50pct_50d_dd25pct_aligned_champmatch

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=1 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `50`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 503
- tickers used: 486
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:PLTR, NYSE:CARR, NYSE:COIN, NYSE:EXE, NYSE:GEV, NYSE:HOOD, NYSE:KVUE, NYSE:OTIS, NYSE:Q, NYSE:SNDK, NYSE:SOLV, NYSE:VLTO
- train rows: 388173 (independent events ≈ 3922.4; overlap-inflation 98.96×)
- val rows: 194400 (independent events ≈ 1963.6; overlap-inflation 99.00×)
- eval rows: 97200 (independent events ≈ 981.8; overlap-inflation 99.00×)
- test rows: 48600 (independent events ≈ 490.9; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.017
- positive prevalence (eval): 0.010

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
| 0 | 279 | 0.0101 | 0.0070 | -0.0031 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`
- tie-break path: `strict_val_brier` — Strict val_brier argmin (no tie-break entered)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -23.506
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0095 | 0.0100 | +0.0005 | 0.0612 | 0.8846 |
| test | 0.0095 | 0.0096 | +0.0001 | 0.0694 | 0.8484 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=97200, base_rate=0.0101

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5325 | 0.0101 | 200 | 90 | 169 | 31 / 200 / 200 |
| 5 | 0.3809 | 0.0101 | 1000 | 251 | 659 | 108 / 200 / 200 |
| 10 | 0.3742 | 0.0101 | 2000 | 336 | 898 | 175 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0101 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.0101 | 5 | 3 | 5 |
| 10 | 0.7000 | 0.0101 | 10 | 7 | 10 |

### test — n_rows=48600, base_rate=0.0097

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1099 | 0.0097 | 100 | 10 | 91 | 9 / 100 / 100 |
| 5 | 0.1142 | 0.0097 | 500 | 41 | 359 | 49 / 100 / 100 |
| 10 | 0.2044 | 0.0097 | 1000 | 92 | 450 | 92 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0097 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0097 | 5 | 0 | 5 |
| 10 | 0.4000 | 0.0097 | 10 | 4 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=97200, Q_days=169, base_rate=0.0101

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.5325 | 0.0101 | 169 |
| 3 | 0.4181 | 0.0101 | 169 |
| 5 | 0.3394 | 0.0101 | 169 |
| 10 | 0.3536 | 0.0101 | 169 |
| 20 | 0.5980 | 0.0101 | 169 |

### test — n_rows=48600, Q_days=91, base_rate=0.0097

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.1099 | 0.0097 | 91 |
| 3 | 0.0842 | 0.0097 | 91 |
| 5 | 0.1256 | 0.0097 | 91 |
| 10 | 0.1953 | 0.0097 | 91 |
| 20 | 0.2604 | 0.0097 | 91 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 200 | 118 | 0.5900 |
| NYSE:SMCI | 181 | 65 | 0.3591 |
| NYSE:PSKY | 76 | 8 | 0.1053 |
| NASDAQ:WBD | 68 | 0 | 0.0000 |
| NYSE:SATS | 61 | 23 | 0.3770 |
| NYSE:VRT | 61 | 4 | 0.0656 |
| NASDAQ:TSLA | 59 | 2 | 0.0339 |
| NYSE:ALB | 44 | 0 | 0.0000 |
| NYSE:KEY | 40 | 2 | 0.0500 |
| NYSE:CCL | 38 | 3 | 0.0789 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:WBD | 68 | 0 | 0.0000 |
| NYSE:ALB | 44 | 0 | 0.0000 |
| NASDAQ:CPRT | 30 | 0 | 0.0000 |
| NYSE:PODD | 8 | 0 | 0.0000 |
| NASDAQ:TSLA | 59 | 2 | 0.0339 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:SMCI | 100 | 10 | 0.1000 |
| NYSE:MRNA | 91 | 0 | 0.0000 |
| NYSE:CVNA | 74 | 19 | 0.2568 |
| NYSE:SATS | 74 | 8 | 0.1081 |
| NASDAQ:INTC | 38 | 0 | 0.0000 |
| NASDAQ:AVGO | 34 | 0 | 0.0000 |
| NYSE:VRT | 31 | 4 | 0.1290 |
| NYSE:ALB | 18 | 0 | 0.0000 |
| NYSE:NCLH | 12 | 0 | 0.0000 |
| NYSE:MPWR | 7 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:MRNA | 91 | 0 | 0.0000 |
| NASDAQ:INTC | 38 | 0 | 0.0000 |
| NASDAQ:AVGO | 34 | 0 | 0.0000 |
| NYSE:ALB | 18 | 0 | 0.0000 |
| NYSE:NCLH | 12 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 139 | 0.4793 | 0.0101 |
| 2024Q1 | 305 | 88 | 0.2885 | 0.0101 |
| 2024Q2 | 315 | 24 | 0.0762 | 0.0101 |
| 2024Q3 | 90 | 0 | 0.0000 | 0.0101 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 31 | 0.1348 | 0.0097 |
| 2024Q4 | 270 | 10 | 0.0370 | 0.0097 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 97200 | 0.0000 | 0.2959 | 0.0014 | 0.0088 | `True` |
| test | 48600 | 0.0000 | 0.1508 | 0.0016 | 0.0087 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=23.51); shipped as `isotonic`. Brier vs base-rate: +0.0005 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
