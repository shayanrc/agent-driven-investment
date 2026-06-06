# gbdt experiment — nifty500_up_10pct_25d_dd5pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `25`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300996 (independent events ≈ 6144.8; overlap-inflation 48.98×)
- val rows: 150755 (independent events ≈ 3076.6; overlap-inflation 49.00×)
- eval rows: 75907 (independent events ≈ 1549.1; overlap-inflation 49.00×)
- test rows: 37600 (independent events ≈ 767.3; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.334
- positive prevalence (eval): 0.363

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
| 0 | 279 | 0.1370 | 0.2305 | 0.0935 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 212/2 |  |
| 1 | 212 | 0.1370 | 0.2306 | 0.0935 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 212/212 features;  |  |
| 2 | 212 | 0.1370 | 0.2306 | 0.0935 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 140.545
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2461 | 0.2312 | -0.0149 | 0.7148 | 0.5465 |
| test | 0.1599 | 0.1386 | -0.0213 | 0.5044 | 0.6008 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.3630

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4109 | 0.3630 | 202 | 83 | 202 | 0 / 202 / 202 |
| 5 | 0.4356 | 0.3630 | 1010 | 440 | 1010 | 0 / 202 / 202 |
| 10 | 0.4168 | 0.3630 | 2020 | 842 | 2020 | 0 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.3630 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.3630 | 5 | 1 | 5 |
| 10 | 0.3000 | 0.3630 | 10 | 3 | 10 |

### test — n_rows=37600, base_rate=0.1663

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2100 | 0.1663 | 100 | 21 | 100 | 0 / 100 / 100 |
| 5 | 0.2380 | 0.1663 | 500 | 119 | 500 | 0 / 100 / 100 |
| 10 | 0.2350 | 0.1663 | 1000 | 235 | 1000 | 0 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1663 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.1663 | 5 | 0 | 5 |
| 10 | 0.1000 | 0.1663 | 10 | 1 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:IIFL | 57 | 27 | 0.4737 |
| NSE:JWL | 56 | 24 | 0.4286 |
| NSE:CGCL | 46 | 12 | 0.2609 |
| NSE:GVT&D | 43 | 33 | 0.7674 |
| NSE:NEWGEN | 33 | 25 | 0.7576 |
| NSE:CANBK | 31 | 5 | 0.1613 |
| NSE:MMTC | 31 | 17 | 0.5484 |
| NSE:JBCHEPHARM | 29 | 16 | 0.5517 |
| NSE:IRCON | 24 | 14 | 0.5833 |
| NSE:IREDA | 24 | 1 | 0.0417 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ITI | 14 | 0 | 0.0000 |
| NSE:NESTLEIND | 12 | 0 | 0.0000 |
| NSE:NAVINFLUOR | 11 | 0 | 0.0000 |
| NSE:BDL | 10 | 0 | 0.0000 |
| NSE:SARDAEN | 7 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:360ONE | 30 | 24 | 0.8000 |
| NSE:ABREL | 24 | 5 | 0.2083 |
| NSE:CEMPRO | 24 | 0 | 0.0000 |
| NSE:COCHINSHIP | 19 | 1 | 0.0526 |
| NSE:MMTC | 16 | 0 | 0.0000 |
| NSE:CHOLAHLDNG | 15 | 5 | 0.3333 |
| NSE:GALLANTT | 15 | 2 | 0.1333 |
| NSE:BBTC | 14 | 1 | 0.0714 |
| NSE:AEGISLOG | 12 | 7 | 0.5833 |
| NSE:IDEA | 12 | 3 | 0.2500 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:CEMPRO | 24 | 0 | 0.0000 |
| NSE:MMTC | 16 | 0 | 0.0000 |
| NSE:JUBLPHARMA | 10 | 0 | 0.0000 |
| NSE:ADANIGREEN | 8 | 0 | 0.0000 |
| NSE:INOXWIND | 8 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 104 | 0.6118 | 0.3630 |
| 2024Q1 | 310 | 115 | 0.3710 | 0.3630 |
| 2024Q2 | 305 | 149 | 0.4885 | 0.3630 |
| 2024Q3 | 225 | 72 | 0.3200 | 0.3630 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 22 | 0.2316 | 0.1663 |
| 2024Q4 | 310 | 83 | 0.2677 | 0.1663 |
| 2025Q1 | 95 | 14 | 0.1474 | 0.1663 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.6512 | 0.2387 | 0.0772 | `False` |
| test | 37600 | 0.0936 | 0.8333 | 0.3193 | 0.0402 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=140.54); shipped as `isotonic`. Brier vs base-rate: -0.0149 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
