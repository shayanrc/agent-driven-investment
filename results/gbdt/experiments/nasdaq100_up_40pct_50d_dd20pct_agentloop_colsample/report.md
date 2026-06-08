# gbdt experiment — nasdaq100_up_40pct_50d_dd20pct_agentloop_colsample

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=4 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `50`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `agent_file_protocol`

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
| 3 | 30 | 0.0325 | 0.0236 | -0.0089 | iteration 3 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 3
- iterations run: 1
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -2.645
- Spiegelhalter p: 0.0082

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0512 | 0.0563 | +0.0051 | 0.2072 | 0.8443 |
| test | 0.0495 | 0.0524 | +0.0029 | 0.2150 | 0.7004 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0598

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4422 | 0.0598 | 343 | 88 | 199 | 144 / 343 / 343 |
| 5 | 0.3780 | 0.0598 | 1144 | 279 | 738 | 247 / 200 / 343 |
| 10 | 0.5308 | 0.0598 | 2144 | 535 | 1008 | 314 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0598 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.0598 | 5 | 4 | 5 |
| 10 | 0.7000 | 0.0598 | 10 | 7 | 10 |

### test — n_rows=4600, base_rate=0.0554

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4314 | 0.0554 | 80 | 22 | 51 | 29 / 80 / 80 |
| 5 | 0.4129 | 0.0554 | 280 | 64 | 155 | 63 / 50 / 80 |
| 10 | 0.5607 | 0.0554 | 530 | 120 | 214 | 70 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0554 | 1 | 0 | 1 |
| 5 | 0.6000 | 0.0554 | 5 | 3 | 5 |
| 10 | 0.8000 | 0.0554 | 10 | 8 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 192 | 11 | 0.0573 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:INTC | 134 | 40 | 0.2985 |
| NASDAQ:MU | 134 | 112 | 0.8358 |
| NASDAQ:MRVL | 125 | 15 | 0.1200 |
| NASDAQ:MDB | 96 | 50 | 0.5208 |
| NASDAQ:TSLA | 96 | 4 | 0.0417 |
| NASDAQ:TTD | 79 | 0 | 0.0000 |
| NASDAQ:MCHP | 51 | 0 | 0.0000 |
| NASDAQ:AVGO | 49 | 27 | 0.5510 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TTD | 79 | 0 | 0.0000 |
| NASDAQ:MCHP | 51 | 0 | 0.0000 |
| NASDAQ:NVDA | 21 | 0 | 0.0000 |
| NASDAQ:TSLA | 96 | 4 | 0.0417 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 50 | 17 | 0.3400 |
| NASDAQ:TTD | 46 | 0 | 0.0000 |
| NASDAQ:INTC | 44 | 28 | 0.6364 |
| NASDAQ:MU | 41 | 10 | 0.2439 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MRVL | 23 | 5 | 0.2174 |
| NASDAQ:TEAM | 19 | 0 | 0.0000 |
| NASDAQ:MDB | 15 | 0 | 0.0000 |
| NASDAQ:AMD | 6 | 3 | 0.5000 |
| NASDAQ:WBD | 6 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 46 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:TEAM | 19 | 0 | 0.0000 |
| NASDAQ:MDB | 15 | 0 | 0.0000 |
| NASDAQ:WBD | 6 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0598 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0598 |
| 2025Q1 | 109 | 8 | 0.0734 | 0.0598 |
| 2025Q2 | 310 | 85 | 0.2742 | 0.0598 |
| 2025Q3 | 320 | 118 | 0.3688 | 0.0598 |
| 2025Q4 | 310 | 68 | 0.2194 | 0.0598 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0554 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0554 |
| 2025Q4 | 11 | 5 | 0.4545 | 0.0554 |
| 2026Q1 | 240 | 59 | 0.2458 | 0.0554 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.3019 | 0.0321 | 0.0601 | `False` |
| test | 4600 | 0.0000 | 0.3019 | 0.0378 | 0.0601 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.65); shipped as `isotonic`. Brier vs base-rate: +0.0051 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
