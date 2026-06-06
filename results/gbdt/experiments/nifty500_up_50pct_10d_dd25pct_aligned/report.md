# gbdt experiment — nifty500_up_50pct_10d_dd25pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `10`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300996 (independent events ≈ 15843.8; overlap-inflation 19.00×)
- val rows: 150755 (independent events ≈ 7934.5; overlap-inflation 19.00×)
- eval rows: 75907 (independent events ≈ 3995.1; overlap-inflation 19.00×)
- test rows: 37600 (independent events ≈ 1978.9; overlap-inflation 19.00×)
- sample uniqueness weighting: `on` (horizon_days=10)
- positive prevalence (train): 0.006
- positive prevalence (eval): 0.002

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2019-01-01`
- train: `2019-01-01` → `2022-03-29`
- val: `2022-03-30` → `2023-11-09`
- eval: `2023-11-10` → `2024-09-03`
- test: `2024-09-04` → `2025-01-27`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0008 | 0.0011 | 0.0003 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 140/2 |  |
| 1 | 140 | 0.0008 | 0.0011 | 0.0002 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 117/140 features |  |
| 2 | 117 | 0.0009 | 0.0011 | 0.0002 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 7.719
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0023 | 0.0023 | -0.0000 | 0.0158 | 0.8040 |
| test | 0.0003 | 0.0003 | -0.0000 | 0.0026 | 0.8658 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.0023

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0889 | 0.0023 | 202 | 8 | 90 | 112 / 202 / 202 |
| 5 | 0.1105 | 0.0023 | 1010 | 19 | 172 | 199 / 202 / 202 |
| 10 | 0.1570 | 0.0023 | 2020 | 27 | 172 | 202 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0023 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0023 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0023 | 10 | 0 | 10 |

### test — n_rows=37600, base_rate=0.0003

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0003 | 100 | 0 | 11 | 89 / 100 / 100 |
| 5 | 0.0000 | 0.0003 | 500 | 0 | 11 | 100 / 100 / 100 |
| 10 | 0.0000 | 0.0003 | 1000 | 0 | 11 | 100 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0003 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0003 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0003 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ADANIENSOL | 93 | 2 | 0.0215 |
| NSE:ADANIGREEN | 82 | 7 | 0.0854 |
| NSE:AEGISLOG | 81 | 0 | 0.0000 |
| NSE:CENTRALBK | 59 | 0 | 0.0000 |
| NSE:ANANTRAJ | 56 | 0 | 0.0000 |
| NSE:ATGL | 56 | 4 | 0.0714 |
| NSE:BBTC | 51 | 0 | 0.0000 |
| NSE:FACT | 32 | 0 | 0.0000 |
| NSE:ACE | 31 | 0 | 0.0000 |
| NSE:CHENNPETRO | 31 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:AEGISLOG | 81 | 0 | 0.0000 |
| NSE:CENTRALBK | 59 | 0 | 0.0000 |
| NSE:ANANTRAJ | 56 | 0 | 0.0000 |
| NSE:BBTC | 51 | 0 | 0.0000 |
| NSE:FACT | 32 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:AARTIIND | 48 | 0 | 0.0000 |
| NSE:BSE | 43 | 0 | 0.0000 |
| NSE:ADANIENSOL | 35 | 0 | 0.0000 |
| NSE:ADANIGREEN | 27 | 0 | 0.0000 |
| NSE:AMBER | 27 | 0 | 0.0000 |
| NSE:BBTC | 27 | 0 | 0.0000 |
| NSE:CEMPRO | 24 | 0 | 0.0000 |
| NSE:AEGISLOG | 19 | 0 | 0.0000 |
| NSE:ADANIPOWER | 16 | 0 | 0.0000 |
| NSE:CDSL | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:AARTIIND | 48 | 0 | 0.0000 |
| NSE:BSE | 43 | 0 | 0.0000 |
| NSE:ADANIENSOL | 35 | 0 | 0.0000 |
| NSE:ADANIGREEN | 27 | 0 | 0.0000 |
| NSE:AMBER | 27 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 16 | 0.0941 | 0.0023 |
| 2024Q1 | 310 | 0 | 0.0000 | 0.0023 |
| 2024Q2 | 305 | 1 | 0.0033 | 0.0023 |
| 2024Q3 | 225 | 2 | 0.0089 | 0.0023 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 0 | 0.0000 | 0.0003 |
| 2024Q4 | 310 | 0 | 0.0000 | 0.0003 |
| 2025Q1 | 95 | 0 | 0.0000 | 0.0003 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.4286 | 0.0009 | 0.0026 | `True` |
| test | 37600 | 0.0000 | 0.4286 | 0.0008 | 0.0032 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=7.72); shipped as `isotonic`. Brier vs base-rate: -0.0000 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
