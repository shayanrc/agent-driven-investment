# gbdt experiment — nifty500_up_50pct_50d_dd25pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `50`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300996 (independent events ≈ 3042.4; overlap-inflation 98.93×)
- val rows: 150755 (independent events ≈ 1522.8; overlap-inflation 99.00×)
- eval rows: 75907 (independent events ≈ 766.7; overlap-inflation 99.00×)
- test rows: 37600 (independent events ≈ 379.8; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.065
- positive prevalence (eval): 0.061

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
| 0 | 279 | 0.0249 | 0.0435 | 0.0186 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 172/2 |  |
| 1 | 172 | 0.0246 | 0.0434 | 0.0188 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 164/172 features |  |
| 2 | 164 | 0.0250 | 0.0433 | 0.0183 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 103.689
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0562 | 0.0577 | +0.0015 | 0.2176 | 0.7152 |
| test | 0.0073 | 0.0049 | -0.0024 | 0.0590 | 0.8893 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.0615

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1238 | 0.0615 | 202 | 25 | 202 | 0 / 202 / 202 |
| 5 | 0.1152 | 0.0615 | 1010 | 115 | 998 | 9 / 202 / 202 |
| 10 | 0.1339 | 0.0615 | 2020 | 259 | 1934 | 21 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0615 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0615 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0615 | 10 | 0 | 10 |

### test — n_rows=37600, base_rate=0.0049

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0471 | 0.0049 | 100 | 4 | 85 | 15 / 100 / 100 |
| 5 | 0.1038 | 0.0049 | 500 | 19 | 183 | 97 / 100 / 100 |
| 10 | 0.1630 | 0.0049 | 1000 | 30 | 184 | 100 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0049 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0049 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0049 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:INOXWIND | 85 | 19 | 0.2235 |
| NSE:APARINDS | 75 | 0 | 0.0000 |
| NSE:ADANIENSOL | 74 | 8 | 0.1081 |
| NSE:CEMPRO | 39 | 13 | 0.3333 |
| NSE:FACT | 32 | 0 | 0.0000 |
| NSE:IIFL | 32 | 0 | 0.0000 |
| NSE:BBTC | 30 | 2 | 0.0667 |
| NSE:IDEA | 27 | 0 | 0.0000 |
| NSE:ANANTRAJ | 26 | 1 | 0.0385 |
| NSE:ADANIGREEN | 24 | 11 | 0.4583 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:APARINDS | 75 | 0 | 0.0000 |
| NSE:FACT | 32 | 0 | 0.0000 |
| NSE:IIFL | 32 | 0 | 0.0000 |
| NSE:IDEA | 27 | 0 | 0.0000 |
| NSE:BSE | 24 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:GRSE | 40 | 0 | 0.0000 |
| NSE:IDEA | 37 | 0 | 0.0000 |
| NSE:COCHINSHIP | 35 | 0 | 0.0000 |
| NSE:AEGISLOG | 31 | 0 | 0.0000 |
| NSE:CAPLIPOINT | 23 | 0 | 0.0000 |
| NSE:ADANIENSOL | 22 | 0 | 0.0000 |
| NSE:ABREL | 21 | 0 | 0.0000 |
| NSE:360ONE | 19 | 0 | 0.0000 |
| NSE:BBTC | 19 | 0 | 0.0000 |
| NSE:SCHNEIDER | 19 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:GRSE | 40 | 0 | 0.0000 |
| NSE:IDEA | 37 | 0 | 0.0000 |
| NSE:COCHINSHIP | 35 | 0 | 0.0000 |
| NSE:AEGISLOG | 31 | 0 | 0.0000 |
| NSE:CAPLIPOINT | 23 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 60 | 0.3529 | 0.0615 |
| 2024Q1 | 310 | 22 | 0.0710 | 0.0615 |
| 2024Q2 | 305 | 33 | 0.1082 | 0.0615 |
| 2024Q3 | 225 | 0 | 0.0000 | 0.0615 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 1 | 0.0105 | 0.0049 |
| 2024Q4 | 310 | 11 | 0.0355 | 0.0049 |
| 2025Q1 | 95 | 7 | 0.0737 | 0.0049 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.1402 | 0.0453 | 0.0332 | `True` |
| test | 37600 | 0.0000 | 0.1402 | 0.0458 | 0.0343 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=103.69); shipped as `isotonic`. Brier vs base-rate: +0.0015 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
