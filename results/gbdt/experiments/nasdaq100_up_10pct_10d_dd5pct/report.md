# gbdt experiment — nasdaq100_up_10pct_10d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `10`
- max_drawdown: `0.05`
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
- positive prevalence (train): 0.111
- positive prevalence (eval): 0.109

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0709 | 0.0740 | 0.0030 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 160/2 |  |
| 1 | 160 | 0.0772 | 0.0741 | -0.0030 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 130/160 features |  |
| 2 | 130 | 0.0757 | 0.0740 | -0.0017 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 12.329
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0887 | 0.0973 | +0.0086 | 0.3009 | 0.7741 |
| test | 0.1259 | 0.1286 | +0.0027 | 0.4211 | 0.6572 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.1092

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2211 | 0.1092 | 343 | 44 | 199 | 144 / 343 / 343 |
| 5 | 0.3232 | 0.1092 | 1144 | 276 | 854 | 205 / 200 / 343 |
| 10 | 0.4152 | 0.1092 | 2144 | 563 | 1356 | 267 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1092 | 1 | 0 | 1 |
| 5 | 0.6000 | 0.1092 | 5 | 3 | 5 |
| 10 | 0.6000 | 0.1092 | 10 | 6 | 10 |

### test — n_rows=8280, base_rate=0.1516

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4787 | 0.1516 | 120 | 45 | 94 | 26 / 120 / 120 |
| 5 | 0.4787 | 0.1516 | 480 | 202 | 422 | 43 / 90 / 120 |
| 10 | 0.4571 | 0.1516 | 930 | 346 | 757 | 64 / 90 / 120 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1516 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.1516 | 5 | 4 | 5 |
| 10 | 0.4000 | 0.1516 | 10 | 4 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 196 | 35 | 0.1786 |
| NASDAQ:ANSS | 143 | 4 | 0.0280 |
| NASDAQ:MRVL | 132 | 45 | 0.3409 |
| NASDAQ:TSLA | 116 | 29 | 0.2500 |
| NASDAQ:INTC | 82 | 20 | 0.2439 |
| NASDAQ:ON | 72 | 20 | 0.2778 |
| NASDAQ:AMD | 59 | 27 | 0.4576 |
| NASDAQ:TEAM | 57 | 9 | 0.1579 |
| NASDAQ:MDB | 54 | 23 | 0.4259 |
| NASDAQ:WBD | 48 | 15 | 0.3125 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:CRWD | 29 | 0 | 0.0000 |
| NASDAQ:ANSS | 143 | 4 | 0.0280 |
| NASDAQ:DDOG | 14 | 2 | 0.1429 |
| NASDAQ:TEAM | 57 | 9 | 0.1579 |
| NASDAQ:TTD | 29 | 5 | 0.1724 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MU | 69 | 41 | 0.5942 |
| NASDAQ:INTC | 68 | 43 | 0.6324 |
| NASDAQ:MSTR | 67 | 29 | 0.4328 |
| NASDAQ:TEAM | 45 | 12 | 0.2667 |
| NASDAQ:MDB | 43 | 14 | 0.3256 |
| NASDAQ:DDOG | 32 | 13 | 0.4062 |
| NASDAQ:MRVL | 32 | 14 | 0.4375 |
| NASDAQ:ANSS | 29 | 4 | 0.1379 |
| NASDAQ:TTD | 26 | 5 | 0.1923 |
| NASDAQ:AMD | 20 | 13 | 0.6500 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AVGO | 16 | 1 | 0.0625 |
| NASDAQ:ANSS | 29 | 4 | 0.1379 |
| NASDAQ:LULU | 6 | 1 | 0.1667 |
| NASDAQ:TTD | 26 | 5 | 0.1923 |
| NASDAQ:TEAM | 45 | 12 | 0.2667 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.1092 | 0.000 |
| 2024Q4 | 64 | 4 | 0.0625 | 0.1092 | 0.572 |
| 2025Q1 | 109 | 10 | 0.0917 | 0.1092 | 0.840 |
| 2025Q2 | 310 | 122 | 0.3935 | 0.1092 | 3.603 |
| 2025Q3 | 320 | 81 | 0.2531 | 0.1092 | 2.317 |
| 2025Q4 | 310 | 59 | 0.1903 | 0.1092 | 1.742 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 2 | 0.1176 | 0.1516 | 0.776 |
| 2025Q3 | 12 | 2 | 0.1667 | 0.1516 | 1.100 |
| 2025Q4 | 11 | 6 | 0.5455 | 0.1516 | 3.599 |
| 2026Q1 | 305 | 89 | 0.2918 | 0.1516 | 1.925 |
| 2026Q2 | 135 | 103 | 0.7630 | 0.1516 | 5.034 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.8297 | 0.1178 | 0.1010 | `False` |
| test | 8280 | 0.0000 | 0.6735 | 0.1144 | 0.0910 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=12.33); shipped as `isotonic`. Brier vs base-rate: +0.0086 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
