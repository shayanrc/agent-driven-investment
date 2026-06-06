# gbdt experiment — nifty500_up_10pct_200d_dd5pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `200`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 357
- tickers excluded: NSE:360ONE, NSE:AADHARHFC, NSE:AAVAS, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:ADANIGREEN, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AFFLE, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATGL, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHALET, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:CREDITACC, NSE:DALBHARAT, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:FLUOROCHEM, NSE:FORCEMOT, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:GRSE, NSE:HDBFS, NSE:HDFCAMC, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:INDIAMART, NSE:IRCON, NSE:IRCTC, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:KPITTECH, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POLYCAB, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RITES, NSE:RRKABEL, NSE:RVNL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 286254 (independent events ≈ 719.8; overlap-inflation 397.66×)
- val rows: 143505 (independent events ≈ 359.7; overlap-inflation 399.00×)
- eval rows: 71387 (independent events ≈ 178.9; overlap-inflation 399.00×)
- test rows: 107804 (independent events ≈ 270.2; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.391
- positive prevalence (eval): 0.540

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2018-01-01`
- train: `2018-01-01` → `2021-03-30`
- val: `2021-03-31` → `2022-11-14`
- eval: `2022-11-15` → `2023-09-01`
- test: `2023-09-04` → `2024-11-21`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1562 | 0.2715 | 0.1153 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 183/2 |  |
| 1 | 183 | 0.1558 | 0.2725 | 0.1167 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 181/183 features;  |  |
| 2 | 181 | 0.1558 | 0.2725 | 0.1167 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 166.604
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2689 | 0.2484 | -0.0205 | 0.7317 | 0.5147 |
| test | 0.2532 | 0.2478 | -0.0054 | 0.6999 | 0.4651 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=71387, base_rate=0.5400

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6750 | 0.5400 | 200 | 135 | 200 | 0 / 200 / 200 |
| 5 | 0.5630 | 0.5400 | 1000 | 563 | 1000 | 0 / 200 / 200 |
| 10 | 0.5530 | 0.5400 | 2000 | 1106 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.5400 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.5400 | 5 | 1 | 5 |
| 10 | 0.4000 | 0.5400 | 10 | 4 | 10 |

### test — n_rows=107804, base_rate=0.4536

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4338 | 0.4536 | 302 | 131 | 302 | 0 / 302 / 302 |
| 5 | 0.4430 | 0.4536 | 1510 | 669 | 1510 | 0 / 302 / 302 |
| 10 | 0.4639 | 0.4536 | 3020 | 1401 | 3020 | 0 / 302 / 302 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.4536 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.4536 | 5 | 0 | 5 |
| 10 | 0.4000 | 0.4536 | 10 | 4 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:IREDA | 86 | 62 | 0.7209 |
| NSE:SUNPHARMA | 36 | 30 | 0.8333 |
| NSE:AXISBANK | 34 | 34 | 1.0000 |
| NSE:CASTROLIND | 31 | 24 | 0.7742 |
| NSE:MARUTI | 24 | 24 | 1.0000 |
| NSE:AIAENG | 22 | 5 | 0.2273 |
| NSE:EICHERMOT | 22 | 8 | 0.3636 |
| NSE:HINDUNILVR | 22 | 6 | 0.2727 |
| NSE:ULTRACEMCO | 22 | 22 | 1.0000 |
| NSE:GILLETTE | 21 | 1 | 0.0476 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ESCORTS | 10 | 0 | 0.0000 |
| NSE:NHPC | 9 | 0 | 0.0000 |
| NSE:CUB | 8 | 0 | 0.0000 |
| NSE:UPL | 8 | 0 | 0.0000 |
| NSE:KIRLOSENG | 7 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:HINDUNILVR | 57 | 11 | 0.1930 |
| NSE:ABB | 36 | 15 | 0.4167 |
| NSE:BHARTIARTL | 31 | 22 | 0.7097 |
| NSE:CUMMINSIND | 31 | 2 | 0.0645 |
| NSE:3MINDIA | 30 | 4 | 0.1333 |
| NSE:BRIGADE | 30 | 7 | 0.2333 |
| NSE:IIFL | 28 | 11 | 0.3929 |
| NSE:BRITANNIA | 27 | 4 | 0.1481 |
| NSE:BATAINDIA | 26 | 16 | 0.6154 |
| NSE:BSE | 26 | 4 | 0.1538 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:MGL | 15 | 0 | 0.0000 |
| NSE:AARTIIND | 14 | 0 | 0.0000 |
| NSE:CIEINDIA | 14 | 0 | 0.0000 |
| NSE:ATUL | 12 | 0 | 0.0000 |
| NSE:NATCOPHARM | 10 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 170 | 57 | 0.3353 | 0.5400 |
| 2023Q1 | 310 | 129 | 0.4161 | 0.5400 |
| 2023Q2 | 300 | 250 | 0.8333 | 0.5400 |
| 2023Q3 | 220 | 127 | 0.5773 | 0.5400 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 95 | 33 | 0.3474 | 0.4536 |
| 2023Q4 | 305 | 195 | 0.6393 | 0.4536 |
| 2024Q1 | 310 | 122 | 0.3935 | 0.4536 |
| 2024Q2 | 305 | 193 | 0.6328 | 0.4536 |
| 2024Q3 | 320 | 91 | 0.2844 | 0.4536 |
| 2024Q4 | 175 | 35 | 0.2000 | 0.4536 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 71387 | 0.3789 | 0.4694 | 0.3952 | 0.0241 | `True` |
| test | 107804 | 0.3789 | 0.4694 | 0.4028 | 0.0286 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=166.60); shipped as `isotonic`. Brier vs base-rate: -0.0205 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
