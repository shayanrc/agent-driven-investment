# gbdt experiment — nifty50_up_10pct_25d_dd5pct_catboost_loop

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

## Final checkpoint

- best iteration: 1
- iterations run: 0
- inner stop signal: `agent_should_stop`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 5.918
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1157 | 0.1149 | -0.0007 | 0.3934 | 0.6338 |
| test | 0.1418 | 0.1470 | +0.0052 | 0.4556 | 0.7063 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=9200, base_rate=0.1325

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2322 | 0.1325 | 297 | 49 | 211 | 86 / 297 / 297 |
| 5 | 0.2195 | 0.1325 | 1098 | 164 | 747 | 209 / 200 / 297 |
| 10 | 0.3008 | 0.1325 | 2098 | 302 | 1004 | 264 / 200 / 297 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1325 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.1325 | 5 | 1 | 5 |
| 10 | 0.5000 | 0.1325 | 10 | 5 | 10 |

### test — n_rows=3450, base_rate=0.1791

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3000 | 0.1791 | 151 | 21 | 70 | 81 / 151 / 151 |
| 5 | 0.3556 | 0.1791 | 451 | 96 | 270 | 107 / 75 / 151 |
| 10 | 0.4395 | 0.1791 | 826 | 189 | 430 | 124 / 75 / 151 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1791 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1791 | 5 | 5 | 5 |
| 10 | 0.9000 | 0.1791 | 10 | 9 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ADANIENT | 198 | 32 | 0.1616 |
| NSE:ADANIPORTS | 188 | 28 | 0.1489 |
| NSE:BEL | 134 | 34 | 0.2537 |
| NSE:NTPC | 122 | 18 | 0.1475 |
| NSE:COALINDIA | 92 | 0 | 0.0000 |
| NSE:ONGC | 63 | 4 | 0.0635 |
| NSE:TRENT | 61 | 13 | 0.2131 |
| NSE:HDFCBANK | 45 | 0 | 0.0000 |
| NSE:BAJAJ-AUTO | 39 | 7 | 0.1795 |
| NSE:BAJFINANCE | 36 | 11 | 0.3056 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:COALINDIA | 92 | 0 | 0.0000 |
| NSE:HDFCBANK | 45 | 0 | 0.0000 |
| NSE:HCLTECH | 15 | 0 | 0.0000 |
| NSE:WIPRO | 8 | 0 | 0.0000 |
| NSE:POWERGRID | 33 | 2 | 0.0606 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 75 | 0 | 0.0000 |
| NSE:ADANIENT | 65 | 17 | 0.2615 |
| NSE:ADANIPORTS | 57 | 21 | 0.3684 |
| NSE:BEL | 51 | 10 | 0.1961 |
| NSE:TMPV | 39 | 9 | 0.2308 |
| NSE:BAJFINANCE | 29 | 4 | 0.1379 |
| NSE:ASIANPAINT | 18 | 6 | 0.3333 |
| NSE:BAJAJ-AUTO | 18 | 1 | 0.0556 |
| NSE:M&M | 15 | 3 | 0.2000 |
| NSE:DRREDDY | 11 | 3 | 0.2727 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 75 | 0 | 0.0000 |
| NSE:HDFCBANK | 10 | 0 | 0.0000 |
| NSE:BAJAJFINSV | 8 | 0 | 0.0000 |
| NSE:NESTLEIND | 8 | 0 | 0.0000 |
| NSE:BAJAJ-AUTO | 18 | 1 | 0.0556 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q4 | 51 | 0 | 0.0000 | 0.1325 | 0.000 |
| 2025Q1 | 127 | 44 | 0.3465 | 0.1325 | 2.615 |
| 2025Q2 | 305 | 65 | 0.2131 | 0.1325 | 1.608 |
| 2025Q3 | 320 | 49 | 0.1531 | 0.1325 | 1.156 |
| 2025Q4 | 295 | 6 | 0.0203 | 0.1325 | 0.154 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q3 | 38 | 0 | 0.0000 | 0.1791 | 0.000 |
| 2025Q4 | 53 | 3 | 0.0566 | 0.1791 | 0.316 |
| 2026Q1 | 305 | 62 | 0.2033 | 0.1791 | 1.135 |
| 2026Q2 | 55 | 31 | 0.5636 | 0.1791 | 3.147 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 9200 | 0.0684 | 0.4615 | 0.1957 | 0.0451 | `True` |
| test | 3450 | 0.0684 | 1.0000 | 0.2194 | 0.0437 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=5.92); shipped as `isotonic`. Brier vs base-rate: -0.0007 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
