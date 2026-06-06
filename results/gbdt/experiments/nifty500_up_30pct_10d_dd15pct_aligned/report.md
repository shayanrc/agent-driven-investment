# gbdt experiment — nifty500_up_30pct_10d_dd15pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `30`
- horizon_days: `10`
- max_drawdown: `0.15`
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
- positive prevalence (train): 0.019
- positive prevalence (eval): 0.011

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
| 0 | 279 | 0.0074 | 0.0066 | -0.0008 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 189/2 |  |
| 1 | 189 | 0.0070 | 0.0067 | -0.0004 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 156/189 features |  |
| 2 | 156 | 0.0075 | 0.0067 | -0.0008 | iteration 2 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 11.724
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0111 | 0.0112 | +0.0001 | 0.0582 | 0.7852 |
| test | 0.0032 | 0.0033 | +0.0000 | 0.0202 | 0.8303 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.0113

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0632 | 0.0113 | 202 | 12 | 190 | 12 / 202 / 202 |
| 5 | 0.0801 | 0.0113 | 1010 | 52 | 649 | 137 / 202 / 202 |
| 10 | 0.1099 | 0.0113 | 2020 | 89 | 810 | 187 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0113 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0113 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0113 | 10 | 0 | 10 |

### test — n_rows=37600, base_rate=0.0033

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0448 | 0.0033 | 100 | 3 | 67 | 33 / 100 / 100 |
| 5 | 0.1230 | 0.0033 | 500 | 15 | 122 | 97 / 100 / 100 |
| 10 | 0.1789 | 0.0033 | 1000 | 22 | 123 | 100 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0033 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0033 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0033 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:FACT | 58 | 8 | 0.1379 |
| NSE:GVT&D | 54 | 3 | 0.0556 |
| NSE:ADANIGREEN | 47 | 1 | 0.0213 |
| NSE:BBTC | 29 | 0 | 0.0000 |
| NSE:BHEL | 29 | 0 | 0.0000 |
| NSE:ADANIENSOL | 28 | 2 | 0.0714 |
| NSE:ADANIPOWER | 28 | 0 | 0.0000 |
| NSE:CENTRALBK | 27 | 3 | 0.1111 |
| NSE:GICRE | 26 | 0 | 0.0000 |
| NSE:SUZLON | 26 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:BBTC | 29 | 0 | 0.0000 |
| NSE:BHEL | 29 | 0 | 0.0000 |
| NSE:ADANIPOWER | 28 | 0 | 0.0000 |
| NSE:GICRE | 26 | 0 | 0.0000 |
| NSE:SUZLON | 26 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:TARIL | 49 | 0 | 0.0000 |
| NSE:GVT&D | 45 | 0 | 0.0000 |
| NSE:ITI | 34 | 7 | 0.2059 |
| NSE:COCHINSHIP | 33 | 0 | 0.0000 |
| NSE:JUBLPHARMA | 29 | 0 | 0.0000 |
| NSE:RPOWER | 25 | 4 | 0.1600 |
| NSE:HSCL | 23 | 0 | 0.0000 |
| NSE:CREDITACC | 21 | 0 | 0.0000 |
| NSE:CAPLIPOINT | 20 | 0 | 0.0000 |
| NSE:IFCI | 20 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:TARIL | 49 | 0 | 0.0000 |
| NSE:GVT&D | 45 | 0 | 0.0000 |
| NSE:COCHINSHIP | 33 | 0 | 0.0000 |
| NSE:JUBLPHARMA | 29 | 0 | 0.0000 |
| NSE:HSCL | 23 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 4 | 0.0235 | 0.0113 |
| 2024Q1 | 310 | 26 | 0.0839 | 0.0113 |
| 2024Q2 | 305 | 15 | 0.0492 | 0.0113 |
| 2024Q3 | 225 | 7 | 0.0311 | 0.0113 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 4 | 0.0421 | 0.0033 |
| 2024Q4 | 310 | 9 | 0.0290 | 0.0033 |
| 2025Q1 | 95 | 2 | 0.0211 | 0.0033 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.1040 | 0.0056 | 0.0063 | `True` |
| test | 37600 | 0.0000 | 0.1040 | 0.0057 | 0.0063 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=11.72); shipped as `isotonic`. Brier vs base-rate: +0.0001 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
