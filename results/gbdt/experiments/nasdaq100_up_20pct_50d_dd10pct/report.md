# gbdt experiment — nasdaq100_up_20pct_50d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `50`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.176
- positive prevalence (eval): 0.198

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1170 | 0.1182 | 0.0012 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 61/27 |  |
| 1 | 61 | 0.1169 | 0.1187 | 0.0018 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 48/61 features |  |
| 2 | 48 | 0.1193 | 0.1190 | -0.0003 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -6.290
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1420 | 0.1588 | +0.0168 | 0.4666 | 0.7502 |
| test | 0.1193 | 0.1206 | +0.0012 | 0.4377 | 0.6654 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.1980

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4523 | 0.1980 | 343 | 90 | 199 | 144 / 343 / 343 |
| 5 | 0.4350 | 0.1980 | 1144 | 425 | 977 | 150 / 200 / 343 |
| 10 | 0.4523 | 0.1980 | 2144 | 868 | 1919 | 156 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1980 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.1980 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.1980 | 10 | 1 | 10 |

### test — n_rows=4600, base_rate=0.1402

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4510 | 0.1402 | 80 | 23 | 51 | 29 / 80 / 80 |
| 5 | 0.3506 | 0.1402 | 280 | 88 | 251 | 30 / 50 / 80 |
| 10 | 0.2851 | 0.1402 | 530 | 134 | 470 | 42 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1402 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1402 | 5 | 5 | 5 |
| 10 | 0.8000 | 0.1402 | 10 | 8 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:INTC | 200 | 90 | 0.4500 |
| NASDAQ:AVGO | 177 | 107 | 0.6045 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:AMD | 110 | 66 | 0.6000 |
| NASDAQ:MSTR | 109 | 16 | 0.1468 |
| NASDAQ:CRWD | 94 | 25 | 0.2660 |
| NASDAQ:MCHP | 59 | 13 | 0.2203 |
| NASDAQ:MDB | 44 | 33 | 0.7500 |
| NASDAQ:DXCM | 38 | 19 | 0.5000 |
| NASDAQ:LRCX | 34 | 13 | 0.3824 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:PDD | 10 | 0 | 0.0000 |
| NASDAQ:TTD | 6 | 0 | 0.0000 |
| NASDAQ:TEAM | 20 | 1 | 0.0500 |
| NASDAQ:LULU | 10 | 1 | 0.1000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:INTC | 50 | 22 | 0.4400 |
| NASDAQ:MDB | 46 | 1 | 0.0217 |
| NASDAQ:AMD | 35 | 26 | 0.7429 |
| NASDAQ:LULU | 30 | 0 | 0.0000 |
| NASDAQ:MRVL | 30 | 23 | 0.7667 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:TTD | 21 | 4 | 0.1905 |
| NASDAQ:MCHP | 17 | 4 | 0.2353 |
| NASDAQ:MSTR | 7 | 0 | 0.0000 |
| NASDAQ:TSLA | 4 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:LULU | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MSTR | 7 | 0 | 0.0000 |
| NASDAQ:MDB | 46 | 1 | 0.0217 |
| NASDAQ:TTD | 21 | 4 | 0.1905 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.1980 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.1980 | 0.000 |
| 2025Q1 | 109 | 1 | 0.0092 | 0.1980 | 0.046 |
| 2025Q2 | 310 | 165 | 0.5323 | 0.1980 | 2.688 |
| 2025Q3 | 320 | 178 | 0.5563 | 0.1980 | 2.809 |
| 2025Q4 | 310 | 81 | 0.2613 | 0.1980 | 1.319 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.1402 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.1402 | 0.000 |
| 2025Q4 | 11 | 6 | 0.5455 | 0.1402 | 3.890 |
| 2026Q1 | 240 | 82 | 0.3417 | 0.1402 | 2.437 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0083 | 1.0000 | 0.1783 | 0.1330 | `False` |
| test | 4600 | 0.0083 | 1.0000 | 0.1781 | 0.1300 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=6.29); shipped as `isotonic`. Brier vs base-rate: +0.0168 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
