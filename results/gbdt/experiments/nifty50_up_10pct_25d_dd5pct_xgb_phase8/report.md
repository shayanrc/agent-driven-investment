# gbdt experiment — nifty50_up_10pct_25d_dd5pct_xgb_phase8

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=1 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty50`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `25`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 50
- tickers used: 46
- tickers excluded: NSE:ETERNAL, NSE:JIOFIN, NSE:MAXHEALTH, NSE:SHRIRAMFIN
- train rows: 36800 (independent events ≈ 751.0; overlap-inflation 49.00×)
- val rows: 18400 (independent events ≈ 375.5; overlap-inflation 49.00×)
- eval rows: 9200 (independent events ≈ 187.8; overlap-inflation 49.00×)
- test rows: 3450 (independent events ≈ 70.4; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.280
- positive prevalence (eval): 0.132

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1726 | 0.1637 | -0.0089 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -5.042
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1167 | 0.1146 | -0.0021 | 0.4094 | 0.5986 |
| test | 0.1457 | 0.1470 | +0.0013 | 0.4665 | 0.6389 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=9200, base_rate=0.1321

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2275 | 0.1321 | 299 | 48 | 211 | 88 / 299 / 299 |
| 5 | 0.2155 | 0.1321 | 1102 | 161 | 747 | 211 / 200 / 299 |
| 10 | 0.3536 | 0.1321 | 2102 | 355 | 1004 | 266 / 200 / 299 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1321 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.1321 | 5 | 2 | 5 |
| 10 | 0.4000 | 0.1321 | 10 | 4 | 10 |

### test — n_rows=3450, base_rate=0.1791

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3143 | 0.1791 | 153 | 22 | 70 | 83 / 153 / 153 |
| 5 | 0.3444 | 0.1791 | 455 | 93 | 270 | 109 / 75 / 153 |
| 10 | 0.3651 | 0.1791 | 830 | 157 | 430 | 126 / 75 / 153 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1791 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.1791 | 5 | 2 | 5 |
| 10 | 0.6000 | 0.1791 | 10 | 6 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ADANIENT | 196 | 33 | 0.1684 |
| NSE:ADANIPORTS | 196 | 24 | 0.1224 |
| NSE:APOLLOHOSP | 146 | 15 | 0.1027 |
| NSE:NTPC | 105 | 21 | 0.2000 |
| NSE:BEL | 95 | 22 | 0.2316 |
| NSE:ASIANPAINT | 82 | 9 | 0.1098 |
| NSE:BAJFINANCE | 70 | 7 | 0.1000 |
| NSE:COALINDIA | 49 | 0 | 0.0000 |
| NSE:AXISBANK | 48 | 8 | 0.1667 |
| NSE:BAJAJ-AUTO | 41 | 1 | 0.0244 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:COALINDIA | 49 | 0 | 0.0000 |
| NSE:ONGC | 16 | 0 | 0.0000 |
| NSE:BAJAJFINSV | 5 | 0 | 0.0000 |
| NSE:SBIN | 5 | 0 | 0.0000 |
| NSE:BAJAJ-AUTO | 41 | 1 | 0.0244 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ADANIENT | 75 | 22 | 0.2933 |
| NSE:ADANIPORTS | 75 | 26 | 0.3467 |
| NSE:NTPC | 75 | 0 | 0.0000 |
| NSE:APOLLOHOSP | 45 | 11 | 0.2444 |
| NSE:ASIANPAINT | 45 | 13 | 0.2889 |
| NSE:AXISBANK | 35 | 9 | 0.2571 |
| NSE:BAJFINANCE | 25 | 1 | 0.0400 |
| NSE:BAJAJ-AUTO | 20 | 2 | 0.1000 |
| NSE:BEL | 18 | 6 | 0.3333 |
| NSE:BAJAJFINSV | 14 | 1 | 0.0714 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 75 | 0 | 0.0000 |
| NSE:INDIGO | 5 | 0 | 0.0000 |
| NSE:BAJFINANCE | 25 | 1 | 0.0400 |
| NSE:BAJAJFINSV | 14 | 1 | 0.0714 |
| NSE:BAJAJ-AUTO | 20 | 2 | 0.1000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q4 | 51 | 0 | 0.0000 | 0.1321 | 0.000 |
| 2025Q1 | 127 | 46 | 0.3622 | 0.1321 | 2.743 |
| 2025Q2 | 305 | 60 | 0.1967 | 0.1321 | 1.490 |
| 2025Q3 | 320 | 50 | 0.1562 | 0.1321 | 1.183 |
| 2025Q4 | 299 | 5 | 0.0167 | 0.1321 | 0.127 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q3 | 38 | 0 | 0.0000 | 0.1791 | 0.000 |
| 2025Q4 | 53 | 1 | 0.0189 | 0.1791 | 0.105 |
| 2026Q1 | 305 | 58 | 0.1902 | 0.1791 | 1.062 |
| 2026Q2 | 59 | 34 | 0.5763 | 0.1791 | 3.217 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 9200 | 0.0000 | 0.7884 | 0.1924 | 0.0560 | `False` |
| test | 3450 | 0.1447 | 0.6000 | 0.2150 | 0.0316 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=5.04); shipped as `isotonic`. Brier vs base-rate: -0.0021 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
