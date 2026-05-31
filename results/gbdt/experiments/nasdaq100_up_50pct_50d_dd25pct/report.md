# gbdt experiment — nasdaq100_up_50pct_50d_dd25pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `50`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 743.4; overlap-inflation 99.00×)
- val rows: 36800 (independent events ≈ 371.7; overlap-inflation 99.00×)
- eval rows: 18400 (independent events ≈ 185.9; overlap-inflation 99.00×)
- test rows: 4600 (independent events ≈ 46.5; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.023
- positive prevalence (eval): 0.034

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0165 | 0.0128 | -0.0037 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 58/27 |  |
| 1 | 58 | 0.0158 | 0.0130 | -0.0028 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 47/58 features |  |
| 2 | 47 | 0.0157 | 0.0127 | -0.0030 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -9.247
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0303 | 0.0332 | +0.0028 | 0.1235 | 0.8564 |
| test | 0.0363 | 0.0388 | +0.0025 | 0.1680 | 0.7476 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0343

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0529 | 0.0343 | 343 | 10 | 189 | 154 / 343 / 343 |
| 5 | 0.3135 | 0.0343 | 1144 | 169 | 539 | 293 / 200 / 343 |
| 10 | 0.6435 | 0.0343 | 2144 | 399 | 620 | 336 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0343 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0343 | 5 | 2 | 5 |
| 10 | 0.6000 | 0.0343 | 10 | 6 | 10 |

### test — n_rows=4600, base_rate=0.0404

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2353 | 0.0404 | 80 | 12 | 51 | 29 / 80 / 80 |
| 5 | 0.5504 | 0.0404 | 280 | 71 | 129 | 66 / 50 / 80 |
| 10 | 0.5838 | 0.0404 | 530 | 101 | 173 | 75 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0404 | 1 | 0 | 1 |
| 5 | 0.8000 | 0.0404 | 5 | 4 | 5 |
| 10 | 0.9000 | 0.0404 | 10 | 9 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 200 | 3 | 0.0150 |
| NASDAQ:MRVL | 183 | 6 | 0.0328 |
| NASDAQ:MCHP | 152 | 22 | 0.1447 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MDB | 82 | 15 | 0.1829 |
| NASDAQ:MU | 82 | 58 | 0.7073 |
| NASDAQ:INTC | 65 | 26 | 0.4000 |
| NASDAQ:AMD | 63 | 4 | 0.0635 |
| NASDAQ:TSLA | 58 | 10 | 0.1724 |
| NASDAQ:AVGO | 39 | 8 | 0.2051 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MSTR | 200 | 3 | 0.0150 |
| NASDAQ:MRVL | 183 | 6 | 0.0328 |
| NASDAQ:AMD | 63 | 4 | 0.0635 |
| NASDAQ:MCHP | 152 | 22 | 0.1447 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:INTC | 50 | 21 | 0.4200 |
| NASDAQ:MSTR | 50 | 3 | 0.0600 |
| NASDAQ:AMD | 44 | 21 | 0.4773 |
| NASDAQ:MRVL | 33 | 16 | 0.4848 |
| NASDAQ:MU | 31 | 9 | 0.2903 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MCHP | 14 | 0 | 0.0000 |
| NASDAQ:MDB | 14 | 0 | 0.0000 |
| NASDAQ:AMAT | 4 | 0 | 0.0000 |
| NASDAQ:AZN | 4 | 1 | 0.2500 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MCHP | 14 | 0 | 0.0000 |
| NASDAQ:MDB | 14 | 0 | 0.0000 |
| NASDAQ:MSTR | 50 | 3 | 0.0600 |
| NASDAQ:MU | 31 | 9 | 0.2903 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0343 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0343 | 0.000 |
| 2025Q1 | 109 | 7 | 0.0642 | 0.0343 | 1.870 |
| 2025Q2 | 310 | 66 | 0.2129 | 0.0343 | 6.198 |
| 2025Q3 | 320 | 54 | 0.1688 | 0.0343 | 4.913 |
| 2025Q4 | 310 | 42 | 0.1355 | 0.0343 | 3.944 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0404 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0404 | 0.000 |
| 2025Q4 | 11 | 1 | 0.0909 | 0.0404 | 2.248 |
| 2026Q1 | 240 | 70 | 0.2917 | 0.0404 | 7.213 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.2357 | 0.0243 | 0.0534 | `False` |
| test | 4600 | 0.0000 | 0.2357 | 0.0244 | 0.0507 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=9.25); shipped as `isotonic`. Brier vs base-rate: +0.0028 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
