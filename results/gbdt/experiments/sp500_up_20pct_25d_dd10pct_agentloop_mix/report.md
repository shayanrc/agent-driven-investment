# gbdt experiment — sp500_up_20pct_25d_dd10pct_agentloop_mix

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=4 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `25`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Data

- tickers in universe: 503
- tickers used: 486
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:PLTR, NYSE:CARR, NYSE:COIN, NYSE:EXE, NYSE:GEV, NYSE:HOOD, NYSE:KVUE, NYSE:OTIS, NYSE:Q, NYSE:SNDK, NYSE:SOLV, NYSE:VLTO
- train rows: 388800 (independent events ≈ 7934.7; overlap-inflation 49.00×)
- val rows: 194400 (independent events ≈ 3967.3; overlap-inflation 49.00×)
- eval rows: 97200 (independent events ≈ 1983.7; overlap-inflation 49.00×)
- test rows: 36450 (independent events ≈ 743.9; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.054
- positive prevalence (eval): 0.066

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
- Spiegelhalter Z: 44.098
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0584 | 0.0614 | +0.0030 | 0.2132 | 0.7896 |
| test | 0.0756 | 0.0807 | +0.0051 | 0.2711 | 0.7571 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=97200, base_rate=0.0657

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3668 | 0.0657 | 201 | 73 | 199 | 2 / 201 / 201 |
| 5 | 0.3719 | 0.0657 | 1001 | 360 | 968 | 12 / 200 / 201 |
| 10 | 0.3676 | 0.0657 | 2001 | 697 | 1896 | 18 / 200 / 201 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0657 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0657 | 5 | 2 | 5 |
| 10 | 0.6000 | 0.0657 | 10 | 6 | 10 |

### test — n_rows=36450, base_rate=0.0886

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2400 | 0.0886 | 75 | 18 | 75 | 0 / 75 / 75 |
| 5 | 0.3867 | 0.0886 | 375 | 145 | 375 | 0 / 75 / 75 |
| 10 | 0.3547 | 0.0886 | 750 | 266 | 750 | 0 / 75 / 75 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0886 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.0886 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.0886 | 10 | 1 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 128 | 39 | 0.3047 |
| NYSE:MRNA | 128 | 32 | 0.2500 |
| NYSE:SMCI | 94 | 37 | 0.3936 |
| NYSE:COHR | 92 | 44 | 0.4783 |
| NASDAQ:TSLA | 70 | 16 | 0.2286 |
| NYSE:VRT | 59 | 17 | 0.2881 |
| NASDAQ:WBD | 51 | 23 | 0.4510 |
| NYSE:ALB | 44 | 26 | 0.5909 |
| NASDAQ:ON | 40 | 7 | 0.1750 |
| NYSE:SATS | 30 | 10 | 0.3333 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 19 | 0 | 0.0000 |
| NYSE:FICO | 8 | 0 | 0.0000 |
| NYSE:ANET | 5 | 0 | 0.0000 |
| NYSE:FSLR | 12 | 1 | 0.0833 |
| NASDAQ:ON | 40 | 7 | 0.1750 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 54 | 17 | 0.3148 |
| NYSE:COHR | 46 | 29 | 0.6304 |
| NYSE:SMCI | 46 | 19 | 0.4130 |
| NYSE:ALB | 38 | 12 | 0.3158 |
| NYSE:MRNA | 36 | 18 | 0.5000 |
| NASDAQ:TSLA | 24 | 0 | 0.0000 |
| NYSE:LITE | 19 | 12 | 0.6316 |
| NYSE:VRT | 16 | 11 | 0.6875 |
| NYSE:FSLR | 15 | 0 | 0.0000 |
| NASDAQ:ON | 14 | 5 | 0.3571 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TSLA | 24 | 0 | 0.0000 |
| NYSE:FSLR | 15 | 0 | 0.0000 |
| NYSE:BLDR | 5 | 0 | 0.0000 |
| NASDAQ:TTD | 12 | 1 | 0.0833 |
| NASDAQ:INTC | 11 | 3 | 0.2727 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 61 | 8 | 0.1311 | 0.0657 |
| 2025Q2 | 310 | 161 | 0.5194 | 0.0657 |
| 2025Q3 | 320 | 97 | 0.3031 | 0.0657 |
| 2025Q4 | 310 | 94 | 0.3032 | 0.0657 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 10 | 6 | 0.6000 | 0.0886 |
| 2026Q1 | 305 | 101 | 0.3311 | 0.0886 |
| 2026Q2 | 60 | 38 | 0.6333 | 0.0886 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 97200 | 0.0000 | 0.8235 | 0.0560 | 0.0803 | `False` |
| test | 36450 | 0.0011 | 0.7600 | 0.0634 | 0.0563 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=44.10); shipped as `isotonic`. Brier vs base-rate: +0.0030 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
