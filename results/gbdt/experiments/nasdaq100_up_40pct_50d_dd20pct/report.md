# gbdt experiment — nasdaq100_up_40pct_50d_dd20pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

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
- train rows: 73600 (independent events ≈ 743.4; overlap-inflation 99.00×)
- val rows: 36800 (independent events ≈ 371.7; overlap-inflation 99.00×)
- eval rows: 18400 (independent events ≈ 185.9; overlap-inflation 99.00×)
- test rows: 4600 (independent events ≈ 46.5; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.040
- positive prevalence (eval): 0.060

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0277 | 0.0229 | -0.0048 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 61/27 |  |
| 1 | 61 | 0.0285 | 0.0232 | -0.0054 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -2.866
- Spiegelhalter p: 0.0042

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0524 | 0.0563 | +0.0039 | 0.1918 | 0.8480 |
| test | 0.0486 | 0.0524 | +0.0038 | 0.2391 | 0.7533 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0598

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1206 | 0.0598 | 343 | 24 | 199 | 144 / 343 / 343 |
| 5 | 0.3631 | 0.0598 | 1144 | 268 | 738 | 247 / 200 / 343 |
| 10 | 0.5367 | 0.0598 | 2144 | 541 | 1008 | 314 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0598 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.0598 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.0598 | 10 | 10 | 10 |

### test — n_rows=4600, base_rate=0.0554

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5490 | 0.0554 | 80 | 28 | 51 | 29 / 80 / 80 |
| 5 | 0.6258 | 0.0554 | 280 | 97 | 155 | 63 / 50 / 80 |
| 10 | 0.5748 | 0.0554 | 530 | 123 | 214 | 70 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0554 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.0554 | 5 | 2 | 5 |
| 10 | 0.6000 | 0.0554 | 10 | 6 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 200 | 11 | 0.0550 |
| NASDAQ:TSLA | 160 | 33 | 0.2062 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:INTC | 140 | 56 | 0.4000 |
| NASDAQ:MRVL | 132 | 21 | 0.1591 |
| NASDAQ:MCHP | 95 | 22 | 0.2316 |
| NASDAQ:MDB | 81 | 51 | 0.6296 |
| NASDAQ:WBD | 59 | 19 | 0.3220 |
| NASDAQ:MU | 56 | 53 | 0.9464 |
| NASDAQ:NVDA | 33 | 2 | 0.0606 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:AVGO | 24 | 0 | 0.0000 |
| NASDAQ:ON | 19 | 0 | 0.0000 |
| NASDAQ:MSTR | 200 | 11 | 0.0550 |
| NASDAQ:NVDA | 33 | 2 | 0.0606 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:INTC | 50 | 28 | 0.5600 |
| NASDAQ:MSTR | 48 | 16 | 0.3333 |
| NASDAQ:MRVL | 39 | 26 | 0.6667 |
| NASDAQ:AMD | 32 | 24 | 0.7500 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MDB | 26 | 0 | 0.0000 |
| NASDAQ:WBD | 18 | 0 | 0.0000 |
| NASDAQ:TSLA | 14 | 0 | 0.0000 |
| NASDAQ:MU | 12 | 2 | 0.1667 |
| NASDAQ:ON | 10 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MDB | 26 | 0 | 0.0000 |
| NASDAQ:WBD | 18 | 0 | 0.0000 |
| NASDAQ:TSLA | 14 | 0 | 0.0000 |
| NASDAQ:ON | 10 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0598 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0598 | 0.000 |
| 2025Q1 | 109 | 11 | 0.1009 | 0.0598 | 1.687 |
| 2025Q2 | 310 | 75 | 0.2419 | 0.0598 | 4.043 |
| 2025Q3 | 320 | 145 | 0.4531 | 0.0598 | 7.573 |
| 2025Q4 | 310 | 37 | 0.1194 | 0.0598 | 1.995 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0554 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0554 | 0.000 |
| 2025Q4 | 11 | 3 | 0.2727 | 0.0554 | 4.920 |
| 2026Q1 | 240 | 94 | 0.3917 | 0.0554 | 7.065 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.4539 | 0.0388 | 0.0784 | `False` |
| test | 4600 | 0.0000 | 0.4539 | 0.0357 | 0.0760 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.87); shipped as `isotonic`. Brier vs base-rate: +0.0039 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
