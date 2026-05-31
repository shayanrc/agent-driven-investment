# gbdt experiment — nasdaq100_up_20pct_10d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `10`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.022
- positive prevalence (eval): 0.019

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0184 | 0.0132 | -0.0052 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 25/27 |  |
| 1 | 25 | 0.0183 | 0.0131 | -0.0052 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 23/25 features |  |
| 2 | 23 | 0.0184 | 0.0131 | -0.0053 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: 0.457
- Spiegelhalter p: 0.6477

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0180 | 0.0190 | +0.0010 | 0.0784 | 0.8628 |
| test | 0.0403 | 0.0411 | +0.0009 | 0.1669 | 0.7807 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0193

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0815 | 0.0193 | 343 | 11 | 135 | 208 / 343 / 343 |
| 5 | 0.3192 | 0.0193 | 1144 | 98 | 307 | 320 / 200 / 343 |
| 10 | 0.5597 | 0.0193 | 2144 | 197 | 352 | 340 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0193 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.0193 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.0193 | 10 | 1 | 10 |

### test — n_rows=8280, base_rate=0.0430

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1974 | 0.0430 | 120 | 15 | 76 | 44 / 120 / 120 |
| 5 | 0.2769 | 0.0430 | 480 | 67 | 242 | 89 / 90 / 120 |
| 10 | 0.4711 | 0.0430 | 930 | 163 | 346 | 112 / 90 / 120 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0430 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.0430 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.0430 | 10 | 1 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 200 | 10 | 0.0500 |
| NASDAQ:MRVL | 185 | 13 | 0.0703 |
| NASDAQ:TSLA | 175 | 20 | 0.1143 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:ON | 85 | 10 | 0.1176 |
| NASDAQ:INTC | 77 | 10 | 0.1299 |
| NASDAQ:WBD | 77 | 5 | 0.0649 |
| NASDAQ:TTD | 71 | 0 | 0.0000 |
| NASDAQ:MDB | 53 | 10 | 0.1887 |
| NASDAQ:MU | 35 | 11 | 0.3143 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TTD | 71 | 0 | 0.0000 |
| NASDAQ:TEAM | 10 | 0 | 0.0000 |
| NASDAQ:MSTR | 200 | 10 | 0.0500 |
| NASDAQ:WBD | 77 | 5 | 0.0649 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MU | 85 | 24 | 0.2824 |
| NASDAQ:TTD | 80 | 3 | 0.0375 |
| NASDAQ:TEAM | 75 | 15 | 0.2000 |
| NASDAQ:MSTR | 72 | 2 | 0.0278 |
| NASDAQ:INTC | 54 | 21 | 0.3889 |
| NASDAQ:MDB | 48 | 1 | 0.0208 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:TSLA | 12 | 0 | 0.0000 |
| NASDAQ:CHTR | 8 | 0 | 0.0000 |
| NASDAQ:MRVL | 5 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:TSLA | 12 | 0 | 0.0000 |
| NASDAQ:CHTR | 8 | 0 | 0.0000 |
| NASDAQ:MRVL | 5 | 0 | 0.0000 |
| NASDAQ:MDB | 48 | 1 | 0.0208 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0193 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0193 | 0.000 |
| 2025Q1 | 109 | 5 | 0.0459 | 0.0193 | 2.371 |
| 2025Q2 | 310 | 44 | 0.1419 | 0.0193 | 7.336 |
| 2025Q3 | 320 | 30 | 0.0938 | 0.0193 | 4.846 |
| 2025Q4 | 310 | 19 | 0.0613 | 0.0193 | 3.168 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0430 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0430 | 0.000 |
| 2025Q4 | 11 | 2 | 0.1818 | 0.0430 | 4.229 |
| 2026Q1 | 305 | 20 | 0.0656 | 0.0430 | 1.525 |
| 2026Q2 | 135 | 45 | 0.3333 | 0.0430 | 7.753 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0003 | 0.4132 | 0.0225 | 0.0313 | `True` |
| test | 8280 | 0.0005 | 0.2724 | 0.0220 | 0.0284 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=0.46<2). Brier vs base-rate: +0.0010 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
