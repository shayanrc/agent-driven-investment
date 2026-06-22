# gbdt experiment — nasdaq100_up_10pct_50d_dd5pct_aligned_revalmatch

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
| 0 | 190 | 0.2383 | 0.2278 | -0.0104 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`
- tie-break path: `strict_val_brier` — Strict val_brier argmin (no tie-break entered)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -33.341
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2452 | 0.2331 | -0.0121 | 0.6935 | 0.5229 |
| test | 0.2383 | 0.2335 | -0.0048 | 0.6744 | 0.5532 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.3702

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1550 | 0.3702 | 200 | 31 | 200 | 0 / 200 / 200 |
| 5 | 0.3550 | 0.3702 | 1000 | 355 | 1000 | 0 / 200 / 200 |
| 10 | 0.3851 | 0.3702 | 2000 | 769 | 1997 | 3 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3702 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.3702 | 5 | 3 | 5 |
| 10 | 0.7000 | 0.3702 | 10 | 7 | 10 |

### test — n_rows=9200, base_rate=0.3714

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2800 | 0.3714 | 100 | 28 | 100 | 0 / 100 / 100 |
| 5 | 0.3040 | 0.3714 | 500 | 152 | 500 | 0 / 100 / 100 |
| 10 | 0.3210 | 0.3714 | 1000 | 321 | 1000 | 0 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.3714 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.3714 | 5 | 0 | 5 |
| 10 | 0.4000 | 0.3714 | 10 | 4 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=18400, Q_days=200, base_rate=0.3702

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.1550 | 0.3702 | 200 |
| 3 | 0.2750 | 0.3702 | 200 |
| 5 | 0.3550 | 0.3702 | 200 |
| 10 | 0.3845 | 0.3702 | 200 |
| 20 | 0.3594 | 0.3702 | 200 |

### test — n_rows=9200, Q_days=100, base_rate=0.3714

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.2800 | 0.3714 | 100 |
| 3 | 0.2067 | 0.3714 | 100 |
| 5 | 0.3040 | 0.3714 | 100 |
| 10 | 0.3210 | 0.3714 | 100 |
| 20 | 0.3636 | 0.3714 | 100 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADBE | 200 | 32 | 0.1600 |
| NASDAQ:ADSK | 200 | 61 | 0.3050 |
| NASDAQ:AMAT | 200 | 105 | 0.5250 |
| NASDAQ:ADI | 164 | 59 | 0.3598 |
| NASDAQ:AMD | 144 | 67 | 0.4653 |
| NASDAQ:AAPL | 31 | 9 | 0.2903 |
| NASDAQ:AMZN | 28 | 21 | 0.7500 |
| NASDAQ:ADP | 27 | 1 | 0.0370 |
| NASDAQ:AMGN | 4 | 0 | 0.0000 |
| NASDAQ:AEP | 2 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADP | 27 | 1 | 0.0370 |
| NASDAQ:ADBE | 200 | 32 | 0.1600 |
| NASDAQ:AAPL | 31 | 9 | 0.2903 |
| NASDAQ:ADSK | 200 | 61 | 0.3050 |
| NASDAQ:ADI | 164 | 59 | 0.3598 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADBE | 100 | 30 | 0.3000 |
| NASDAQ:ADI | 100 | 9 | 0.0900 |
| NASDAQ:AMAT | 100 | 26 | 0.2600 |
| NASDAQ:ADSK | 84 | 64 | 0.7619 |
| NASDAQ:AAPL | 69 | 20 | 0.2899 |
| NASDAQ:AMD | 31 | 0 | 0.0000 |
| NASDAQ:AMGN | 11 | 0 | 0.0000 |
| NASDAQ:AMZN | 5 | 3 | 0.6000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMD | 31 | 0 | 0.0000 |
| NASDAQ:AMGN | 11 | 0 | 0.0000 |
| NASDAQ:ADI | 100 | 9 | 0.0900 |
| NASDAQ:AMAT | 100 | 26 | 0.2600 |
| NASDAQ:AAPL | 69 | 20 | 0.2899 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 128 | 0.4414 | 0.3702 |
| 2024Q1 | 305 | 100 | 0.3279 | 0.3702 |
| 2024Q2 | 315 | 125 | 0.3968 | 0.3702 |
| 2024Q3 | 90 | 2 | 0.0222 | 0.3702 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 79 | 0.3435 | 0.3714 |
| 2024Q4 | 270 | 73 | 0.2704 | 0.3714 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.1751 | 0.3490 | 0.2780 | 0.0855 | `False` |
| test | 9200 | 0.1751 | 0.3490 | 0.2896 | 0.0825 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=33.34); shipped as `isotonic`. Brier vs base-rate: -0.0121 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
