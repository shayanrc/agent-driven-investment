# gbdt experiment — nifty500_up_30pct_50d_dd15pct

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `30`
- horizon_days: `50`
- max_drawdown: `0.15`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300800 (independent events ≈ 3038.4; overlap-inflation 99.00×)
- val rows: 150400 (independent events ≈ 1519.2; overlap-inflation 99.00×)
- eval rows: 75200 (independent events ≈ 759.6; overlap-inflation 99.00×)
- test rows: 18800 (independent events ≈ 189.9; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.180
- positive prevalence (eval): 0.064

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1255 | 0.1098 | -0.0157 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 39/27 |  |
| 1 | 39 | 0.1168 | 0.1084 | -0.0084 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 39/39 features |  |
| 2 | 39 | 0.1168 | 0.1084 | -0.0084 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 12.377
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0581 | 0.0597 | +0.0016 | 0.2219 | 0.7207 |
| test | 0.0554 | 0.0534 | -0.0020 | 0.2169 | 0.7154 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75200, base_rate=0.0638

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2426 | 0.0638 | 299 | 49 | 202 | 97 / 299 / 299 |
| 5 | 0.1882 | 0.0638 | 1112 | 189 | 1004 | 100 / 203 / 299 |
| 10 | 0.1780 | 0.0638 | 2124 | 329 | 1848 | 149 / 202 / 299 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0638 | 1 | 0 | 1 |
| 5 | 0.8000 | 0.0638 | 5 | 4 | 5 |
| 10 | 0.7000 | 0.0638 | 10 | 7 | 10 |

### test — n_rows=18800, base_rate=0.0566

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4314 | 0.0566 | 104 | 22 | 51 | 53 / 104 / 104 |
| 5 | 0.2360 | 0.0566 | 316 | 59 | 250 | 55 / 53 / 104 |
| 10 | 0.2203 | 0.0566 | 577 | 104 | 472 | 61 / 52 / 104 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0566 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0566 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0566 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:MMTC | 128 | 0 | 0.0000 |
| NSE:NTPC | 96 | 0 | 0.0000 |
| NSE:GRSE | 95 | 23 | 0.2421 |
| NSE:HBLENGINE | 66 | 18 | 0.2727 |
| NSE:ITI | 66 | 43 | 0.6515 |
| NSE:FORCEMOT | 60 | 18 | 0.3000 |
| NSE:NEWGEN | 50 | 5 | 0.1000 |
| NSE:GALLANTT | 49 | 11 | 0.2245 |
| NSE:WELSPUNLIV | 40 | 0 | 0.0000 |
| NSE:GMDCLTD | 39 | 30 | 0.7692 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:MMTC | 128 | 0 | 0.0000 |
| NSE:NTPC | 96 | 0 | 0.0000 |
| NSE:WELSPUNLIV | 40 | 0 | 0.0000 |
| NSE:TTML | 39 | 0 | 0.0000 |
| NSE:GABRIEL | 29 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 50 | 0 | 0.0000 |
| NSE:GALLANTT | 47 | 30 | 0.6383 |
| NSE:TARIL | 35 | 8 | 0.2286 |
| NSE:DCMSHRIRAM | 31 | 0 | 0.0000 |
| NSE:FORCEMOT | 30 | 0 | 0.0000 |
| NSE:KIRLOSENG | 28 | 8 | 0.2857 |
| NSE:IDEA | 19 | 0 | 0.0000 |
| NSE:AEGISLOG | 14 | 0 | 0.0000 |
| NSE:GMDCLTD | 11 | 3 | 0.2727 |
| NSE:RPOWER | 10 | 6 | 0.6000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:NTPC | 50 | 0 | 0.0000 |
| NSE:DCMSHRIRAM | 31 | 0 | 0.0000 |
| NSE:FORCEMOT | 30 | 0 | 0.0000 |
| NSE:IDEA | 19 | 0 | 0.0000 |
| NSE:AEGISLOG | 14 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q4 | 51 | 0 | 0.0000 | 0.0638 | 0.000 |
| 2025Q1 | 131 | 41 | 0.3130 | 0.0638 | 4.907 |
| 2025Q2 | 305 | 72 | 0.2361 | 0.0638 | 3.701 |
| 2025Q3 | 320 | 56 | 0.1750 | 0.0638 | 2.744 |
| 2025Q4 | 305 | 20 | 0.0656 | 0.0638 | 1.028 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q3 | 38 | 0 | 0.0000 | 0.0566 | 0.000 |
| 2025Q4 | 33 | 0 | 0.0000 | 0.0566 | 0.000 |
| 2026Q1 | 245 | 59 | 0.2408 | 0.0566 | 4.255 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75200 | 0.0000 | 0.4550 | 0.0861 | 0.0644 | `False` |
| test | 18800 | 0.0000 | 0.3409 | 0.1045 | 0.0715 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=12.38); shipped as `isotonic`. Brier vs base-rate: +0.0016 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
