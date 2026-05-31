# gbdt experiment — nasdaq100_up_40pct_10d_dd20pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `10`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 3873.7; overlap-inflation 19.00×)
- val rows: 36800 (independent events ≈ 1936.8; overlap-inflation 19.00×)
- eval rows: 18400 (independent events ≈ 968.4; overlap-inflation 19.00×)
- test rows: 8280 (independent events ≈ 435.8; overlap-inflation 19.00×)
- sample uniqueness weighting: `on` (horizon_days=10)
- positive prevalence (train): 0.002
- positive prevalence (eval): 0.002

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0018 | 0.0015 | -0.0004 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 32/27 |  |
| 1 | 32 | 0.0019 | 0.0015 | -0.0005 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 28/32 features |  |
| 2 | 28 | 0.0018 | 0.0015 | -0.0004 | iteration 2 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -5.186
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0023 | 0.0022 | -0.0001 | 0.0148 | 0.8734 |
| test | 0.0072 | 0.0072 | -0.0000 | 0.0536 | 0.7512 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0022

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1750 | 0.0022 | 343 | 7 | 40 | 303 / 343 / 343 |
| 5 | 0.3250 | 0.0022 | 1144 | 13 | 40 | 343 / 200 / 343 |
| 10 | 0.4500 | 0.0022 | 2144 | 18 | 40 | 343 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0022 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0022 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0022 | 10 | 0 | 10 |

### test — n_rows=8280, base_rate=0.0072

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0303 | 0.0072 | 120 | 1 | 33 | 87 / 120 / 120 |
| 5 | 0.2667 | 0.0072 | 480 | 16 | 60 | 119 / 90 / 120 |
| 10 | 0.5667 | 0.0072 | 930 | 34 | 60 | 120 / 90 / 120 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0072 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0072 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0072 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMD | 155 | 2 | 0.0129 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:INTC | 110 | 4 | 0.0364 |
| NASDAQ:MSTR | 67 | 0 | 0.0000 |
| NASDAQ:AVGO | 55 | 0 | 0.0000 |
| NASDAQ:MRVL | 53 | 0 | 0.0000 |
| NASDAQ:MDB | 49 | 7 | 0.1429 |
| NASDAQ:CRWD | 48 | 0 | 0.0000 |
| NASDAQ:WBD | 48 | 0 | 0.0000 |
| NASDAQ:BIIB | 44 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MSTR | 67 | 0 | 0.0000 |
| NASDAQ:AVGO | 55 | 0 | 0.0000 |
| NASDAQ:MRVL | 53 | 0 | 0.0000 |
| NASDAQ:CRWD | 48 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 73 | 0 | 0.0000 |
| NASDAQ:MU | 56 | 6 | 0.1071 |
| NASDAQ:TEAM | 54 | 0 | 0.0000 |
| NASDAQ:DDOG | 46 | 6 | 0.1304 |
| NASDAQ:INTC | 40 | 3 | 0.0750 |
| NASDAQ:MDB | 33 | 0 | 0.0000 |
| NASDAQ:TTD | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:AMD | 28 | 0 | 0.0000 |
| NASDAQ:CSGP | 24 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 73 | 0 | 0.0000 |
| NASDAQ:TEAM | 54 | 0 | 0.0000 |
| NASDAQ:MDB | 33 | 0 | 0.0000 |
| NASDAQ:TTD | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0022 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0022 | 0.000 |
| 2025Q1 | 109 | 0 | 0.0000 | 0.0022 | 0.000 |
| 2025Q2 | 310 | 0 | 0.0000 | 0.0022 | 0.000 |
| 2025Q3 | 320 | 12 | 0.0375 | 0.0022 | 17.250 |
| 2025Q4 | 310 | 1 | 0.0032 | 0.0022 | 1.484 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0072 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0072 | 0.000 |
| 2025Q4 | 11 | 0 | 0.0000 | 0.0072 | 0.000 |
| 2026Q1 | 305 | 4 | 0.0131 | 0.0072 | 1.810 |
| 2026Q2 | 135 | 12 | 0.0889 | 0.0072 | 12.267 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.0846 | 0.0035 | 0.0142 | `True` |
| test | 8280 | 0.0000 | 0.0846 | 0.0052 | 0.0178 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=5.19); shipped as `isotonic`. Brier vs base-rate: -0.0001 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
