# gbdt experiment — nasdaq100_up_10pct_50d_dd5pct_aligned_baccmatch

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=1 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `50`
- max_drawdown: `0.05`
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
- positive prevalence (train): 0.407
- positive prevalence (eval): 0.370

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
| 0 | 190 | 0.2371 | 0.2286 | -0.0086 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`
- tie-break path: `strict_val_brier` — Strict val_brier argmin (no tie-break entered)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -32.687
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2425 | 0.2331 | -0.0093 | 0.6846 | 0.5337 |
| test | 0.2419 | 0.2335 | -0.0085 | 0.6829 | 0.5374 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.3702

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3150 | 0.3702 | 200 | 63 | 200 | 0 / 200 / 200 |
| 5 | 0.3950 | 0.3702 | 1000 | 395 | 1000 | 0 / 200 / 200 |
| 10 | 0.3731 | 0.3702 | 2000 | 745 | 1997 | 3 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3702 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.3702 | 5 | 2 | 5 |
| 10 | 0.2000 | 0.3702 | 10 | 2 | 10 |

### test — n_rows=9200, base_rate=0.3714

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2500 | 0.3714 | 100 | 25 | 100 | 0 / 100 / 100 |
| 5 | 0.2700 | 0.3714 | 500 | 135 | 500 | 0 / 100 / 100 |
| 10 | 0.2780 | 0.3714 | 1000 | 278 | 1000 | 0 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.3714 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.3714 | 5 | 0 | 5 |
| 10 | 0.5000 | 0.3714 | 10 | 5 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=18400, Q_days=200, base_rate=0.3702

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.3150 | 0.3702 | 200 |
| 3 | 0.4233 | 0.3702 | 200 |
| 5 | 0.3950 | 0.3702 | 200 |
| 10 | 0.3725 | 0.3702 | 200 |
| 20 | 0.3788 | 0.3702 | 200 |

### test — n_rows=9200, Q_days=100, base_rate=0.3714

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.2500 | 0.3714 | 100 |
| 3 | 0.2500 | 0.3714 | 100 |
| 5 | 0.2700 | 0.3714 | 100 |
| 10 | 0.2780 | 0.3714 | 100 |
| 20 | 0.3627 | 0.3714 | 100 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMAT | 200 | 105 | 0.5250 |
| NASDAQ:AMD | 200 | 105 | 0.5250 |
| NASDAQ:ADBE | 159 | 32 | 0.2013 |
| NASDAQ:AVGO | 154 | 73 | 0.4740 |
| NASDAQ:ANSS | 130 | 21 | 0.1615 |
| NASDAQ:ADSK | 55 | 26 | 0.4727 |
| NASDAQ:ASML | 35 | 0 | 0.0000 |
| NASDAQ:BKR | 29 | 9 | 0.3103 |
| NASDAQ:AMZN | 23 | 12 | 0.5217 |
| NASDAQ:CDNS | 12 | 12 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ASML | 35 | 0 | 0.0000 |
| NASDAQ:ANSS | 130 | 21 | 0.1615 |
| NASDAQ:ADBE | 159 | 32 | 0.2013 |
| NASDAQ:BKR | 29 | 9 | 0.3103 |
| NASDAQ:ADSK | 55 | 26 | 0.4727 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMAT | 100 | 26 | 0.2600 |
| NASDAQ:AMD | 100 | 25 | 0.2500 |
| NASDAQ:ASML | 100 | 32 | 0.3200 |
| NASDAQ:ADI | 93 | 7 | 0.0753 |
| NASDAQ:AVGO | 59 | 21 | 0.3559 |
| NASDAQ:ADBE | 48 | 24 | 0.5000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADI | 93 | 7 | 0.0753 |
| NASDAQ:AMD | 100 | 25 | 0.2500 |
| NASDAQ:AMAT | 100 | 26 | 0.2600 |
| NASDAQ:ASML | 100 | 32 | 0.3200 |
| NASDAQ:AVGO | 59 | 21 | 0.3559 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 153 | 0.5276 | 0.3702 |
| 2024Q1 | 305 | 110 | 0.3607 | 0.3702 |
| 2024Q2 | 315 | 127 | 0.4032 | 0.3702 |
| 2024Q3 | 90 | 5 | 0.0556 | 0.3702 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 77 | 0.3348 | 0.3714 |
| 2024Q4 | 270 | 58 | 0.2148 | 0.3714 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.1707 | 0.3535 | 0.2778 | 0.0716 | `False` |
| test | 9200 | 0.1707 | 0.3535 | 0.2822 | 0.0735 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=32.69); shipped as `isotonic`. Brier vs base-rate: -0.0093 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
