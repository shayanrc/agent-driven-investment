# gbdt experiment — nifty50_up_10pct_25d_dd5pct_catboost_monotone_probe

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=1 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty50`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `25`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `agent_file_protocol`

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
- positive prevalence (eval): 0.133

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1687 | 0.1650 | -0.0037 | iteration 0 — full feature pool, default HPs :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 0
- iterations run: 1
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 5.560
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1177 | 0.1149 | -0.0027 | 0.4008 | 0.6258 |
| test | 0.1398 | 0.1470 | +0.0073 | 0.4504 | 0.7147 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=9200, base_rate=0.1325

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3270 | 0.1325 | 297 | 69 | 211 | 86 / 297 / 297 |
| 5 | 0.2396 | 0.1325 | 1098 | 179 | 747 | 209 / 200 / 297 |
| 10 | 0.2719 | 0.1325 | 2098 | 273 | 1004 | 264 / 200 / 297 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1325 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.1325 | 5 | 2 | 5 |
| 10 | 0.6000 | 0.1325 | 10 | 6 | 10 |

### test — n_rows=3450, base_rate=0.1791

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2429 | 0.1791 | 151 | 17 | 70 | 81 / 151 / 151 |
| 5 | 0.3889 | 0.1791 | 451 | 105 | 270 | 107 / 75 / 151 |
| 10 | 0.4209 | 0.1791 | 826 | 181 | 430 | 124 / 75 / 151 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1791 | 1 | 0 | 1 |
| 5 | 0.8000 | 0.1791 | 5 | 4 | 5 |
| 10 | 0.7000 | 0.1791 | 10 | 7 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ADANIENT | 200 | 34 | 0.1700 |
| NSE:ADANIPORTS | 196 | 27 | 0.1378 |
| NSE:BEL | 191 | 56 | 0.2932 |
| NSE:COALINDIA | 130 | 0 | 0.0000 |
| NSE:NTPC | 97 | 17 | 0.1753 |
| NSE:TRENT | 75 | 26 | 0.3467 |
| NSE:DRREDDY | 67 | 2 | 0.0299 |
| NSE:WIPRO | 62 | 0 | 0.0000 |
| NSE:BAJAJ-AUTO | 31 | 6 | 0.1935 |
| NSE:BAJFINANCE | 20 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:COALINDIA | 130 | 0 | 0.0000 |
| NSE:WIPRO | 62 | 0 | 0.0000 |
| NSE:BAJFINANCE | 20 | 0 | 0.0000 |
| NSE:NESTLEIND | 8 | 0 | 0.0000 |
| NSE:DRREDDY | 67 | 2 | 0.0299 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 75 | 0 | 0.0000 |
| NSE:ADANIENT | 74 | 21 | 0.2838 |
| NSE:ADANIPORTS | 60 | 14 | 0.2333 |
| NSE:BAJFINANCE | 43 | 15 | 0.3488 |
| NSE:BEL | 42 | 11 | 0.2619 |
| NSE:NESTLEIND | 33 | 14 | 0.4242 |
| NSE:TMPV | 30 | 9 | 0.3000 |
| NSE:BAJAJ-AUTO | 25 | 3 | 0.1200 |
| NSE:DRREDDY | 23 | 0 | 0.0000 |
| NSE:HDFCBANK | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 75 | 0 | 0.0000 |
| NSE:DRREDDY | 23 | 0 | 0.0000 |
| NSE:HDFCBANK | 16 | 0 | 0.0000 |
| NSE:BAJAJ-AUTO | 25 | 3 | 0.1200 |
| NSE:ADANIPORTS | 60 | 14 | 0.2333 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q4 | 51 | 0 | 0.0000 | 0.1325 | 0.000 |
| 2025Q1 | 127 | 36 | 0.2835 | 0.1325 | 2.139 |
| 2025Q2 | 305 | 100 | 0.3279 | 0.1325 | 2.474 |
| 2025Q3 | 320 | 36 | 0.1125 | 0.1325 | 0.849 |
| 2025Q4 | 295 | 7 | 0.0237 | 0.1325 | 0.179 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q3 | 38 | 0 | 0.0000 | 0.1791 | 0.000 |
| 2025Q4 | 53 | 6 | 0.1132 | 0.1791 | 0.632 |
| 2026Q1 | 305 | 66 | 0.2164 | 0.1791 | 1.208 |
| 2026Q2 | 55 | 33 | 0.6000 | 0.1791 | 3.350 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 9200 | 0.0000 | 0.5185 | 0.1991 | 0.0211 | `True` |
| test | 3450 | 0.1708 | 0.5185 | 0.2157 | 0.0486 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=5.56); shipped as `isotonic`. Brier vs base-rate: -0.0027 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
