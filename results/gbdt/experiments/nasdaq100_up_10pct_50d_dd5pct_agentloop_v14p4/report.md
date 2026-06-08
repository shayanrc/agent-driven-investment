# gbdt experiment — nasdaq100_up_10pct_50d_dd5pct_agentloop_v14p4

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `50`
- max_drawdown: `0.05`
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
- positive prevalence (train): 0.367
- positive prevalence (eval): 0.354

## Segment windows

- split mode: `trailing`
- train: `2019-11-06` → `2023-08-07`
- val: `2023-01-11` → `2025-03-12`
- eval: `2024-08-15` → `2025-12-26`
- test: `2025-06-04` → `2026-03-11`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|

## Final checkpoint

- best iteration: 1
- iterations run: 0
- inner stop signal: `agent_should_stop`
- fs_hp_loop callback_mode: `agent_file_protocol`
- tie-break path: `anti_auc_eval_rp1` — Anti-AUC fallback: tie set picked by eval R-Precision@1 (V1.3 Option A)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -3.341
- Spiegelhalter p: 0.0008

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2262 | 0.2286 | +0.0023 | 0.6448 | 0.5536 |
| test | 0.2065 | 0.1957 | -0.0108 | 0.6043 | 0.4626 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.3536

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5820 | 0.3536 | 343 | 142 | 244 | 99 / 343 / 343 |
| 5 | 0.4103 | 0.3536 | 1144 | 423 | 1031 | 149 / 200 / 343 |
| 10 | 0.4658 | 0.3536 | 2144 | 926 | 1988 | 154 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.3536 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.3536 | 5 | 2 | 5 |
| 10 | 0.3000 | 0.3536 | 10 | 3 | 10 |

### test — n_rows=4600, base_rate=0.2670

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4085 | 0.2670 | 81 | 29 | 71 | 10 / 81 / 81 |
| 5 | 0.3000 | 0.2670 | 281 | 81 | 270 | 32 / 50 / 81 |
| 10 | 0.3392 | 0.2670 | 531 | 173 | 510 | 34 / 50 / 81 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.2670 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.2670 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.2670 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADBE | 200 | 31 | 0.1550 |
| NASDAQ:ADI | 200 | 102 | 0.5100 |
| NASDAQ:ADSK | 198 | 50 | 0.2525 |
| NASDAQ:AAPL | 185 | 98 | 0.5297 |
| NASDAQ:AMAT | 152 | 86 | 0.5658 |
| NASDAQ:ANSS | 143 | 44 | 0.3077 |
| NASDAQ:ADP | 48 | 7 | 0.1458 |
| NASDAQ:AMD | 17 | 5 | 0.2941 |
| NASDAQ:AZN | 1 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADP | 48 | 7 | 0.1458 |
| NASDAQ:ADBE | 200 | 31 | 0.1550 |
| NASDAQ:ADSK | 198 | 50 | 0.2525 |
| NASDAQ:AMD | 17 | 5 | 0.2941 |
| NASDAQ:ANSS | 143 | 44 | 0.3077 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADBE | 50 | 4 | 0.0800 |
| NASDAQ:ADI | 50 | 32 | 0.6400 |
| NASDAQ:ADSK | 50 | 9 | 0.1800 |
| NASDAQ:ADP | 49 | 0 | 0.0000 |
| NASDAQ:AMAT | 32 | 11 | 0.3438 |
| NASDAQ:ANSS | 30 | 20 | 0.6667 |
| NASDAQ:AAPL | 19 | 4 | 0.2105 |
| NASDAQ:AZN | 1 | 1 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADP | 49 | 0 | 0.0000 |
| NASDAQ:ADBE | 50 | 4 | 0.0800 |
| NASDAQ:ADSK | 50 | 9 | 0.1800 |
| NASDAQ:AAPL | 19 | 4 | 0.2105 |
| NASDAQ:AMAT | 32 | 11 | 0.3438 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 32 | 22 | 0.6875 | 0.3536 |
| 2024Q4 | 64 | 22 | 0.3438 | 0.3536 |
| 2025Q1 | 113 | 4 | 0.0354 | 0.3536 |
| 2025Q2 | 310 | 158 | 0.5097 | 0.3536 |
| 2025Q3 | 320 | 130 | 0.4062 | 0.3536 |
| 2025Q4 | 305 | 87 | 0.2852 | 0.3536 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q2 | 18 | 18 | 1.0000 | 0.2670 |
| 2025Q3 | 12 | 2 | 0.1667 | 0.2670 |
| 2025Q4 | 16 | 4 | 0.2500 | 0.2670 |
| 2026Q1 | 235 | 57 | 0.2426 | 0.2670 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.1148 | 0.3745 | 0.3578 | 0.0357 | `True` |
| test | 4600 | 0.2800 | 0.3745 | 0.3569 | 0.0337 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=3.34); shipped as `isotonic`. Brier vs base-rate: +0.0023 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
