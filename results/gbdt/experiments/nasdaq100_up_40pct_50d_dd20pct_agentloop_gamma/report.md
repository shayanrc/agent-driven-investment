# gbdt experiment — nasdaq100_up_40pct_50d_dd20pct_agentloop_gamma

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
| 5 | 30 | 0.0341 | 0.0234 | -0.0107 | iteration 5 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 5
- iterations run: 1
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: -0.379
- Spiegelhalter p: 0.7050

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0511 | 0.0563 | +0.0051 | 0.1870 | 0.8515 |
| test | 0.0494 | 0.0524 | +0.0030 | 0.1954 | 0.7301 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0598

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1910 | 0.0598 | 343 | 38 | 199 | 144 / 343 / 343 |
| 5 | 0.3699 | 0.0598 | 1144 | 273 | 738 | 247 / 200 / 343 |
| 10 | 0.5337 | 0.0598 | 2144 | 538 | 1008 | 314 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0598 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0598 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0598 | 10 | 0 | 10 |

### test — n_rows=4600, base_rate=0.0554

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3529 | 0.0554 | 80 | 18 | 51 | 29 / 80 / 80 |
| 5 | 0.4839 | 0.0554 | 280 | 75 | 155 | 63 / 50 / 80 |
| 10 | 0.5981 | 0.0554 | 530 | 128 | 214 | 70 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0554 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.0554 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.0554 | 10 | 1 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 200 | 11 | 0.0550 |
| NASDAQ:TSLA | 189 | 39 | 0.2063 |
| NASDAQ:MRVL | 181 | 19 | 0.1050 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MU | 91 | 74 | 0.8132 |
| NASDAQ:MDB | 87 | 49 | 0.5632 |
| NASDAQ:INTC | 83 | 23 | 0.2771 |
| NASDAQ:TTD | 68 | 0 | 0.0000 |
| NASDAQ:WBD | 56 | 41 | 0.7321 |
| NASDAQ:AVGO | 41 | 17 | 0.4146 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TTD | 68 | 0 | 0.0000 |
| NASDAQ:MSTR | 200 | 11 | 0.0550 |
| NASDAQ:MRVL | 181 | 19 | 0.1050 |
| NASDAQ:TSLA | 189 | 39 | 0.2063 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 50 | 17 | 0.3400 |
| NASDAQ:TTD | 44 | 0 | 0.0000 |
| NASDAQ:MU | 36 | 8 | 0.2222 |
| NASDAQ:INTC | 34 | 26 | 0.7647 |
| NASDAQ:MRVL | 32 | 14 | 0.4375 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MDB | 24 | 0 | 0.0000 |
| NASDAQ:AMD | 13 | 9 | 0.6923 |
| NASDAQ:WBD | 12 | 0 | 0.0000 |
| NASDAQ:TSLA | 5 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 44 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MDB | 24 | 0 | 0.0000 |
| NASDAQ:WBD | 12 | 0 | 0.0000 |
| NASDAQ:TSLA | 5 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0598 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0598 |
| 2025Q1 | 109 | 11 | 0.1009 | 0.0598 |
| 2025Q2 | 310 | 66 | 0.2129 | 0.0598 |
| 2025Q3 | 320 | 133 | 0.4156 | 0.0598 |
| 2025Q4 | 310 | 63 | 0.2032 | 0.0598 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0554 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0554 |
| 2025Q4 | 11 | 2 | 0.1818 | 0.0554 |
| 2026Q1 | 240 | 73 | 0.3042 | 0.0554 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0064 | 0.2388 | 0.0341 | 0.0493 | `True` |
| test | 4600 | 0.0074 | 0.2388 | 0.0338 | 0.0457 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=0.38<2). Brier vs base-rate: +0.0051 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
