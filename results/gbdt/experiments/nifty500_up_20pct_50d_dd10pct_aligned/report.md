# gbdt experiment — nifty500_up_20pct_50d_dd10pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `50`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.291
- positive prevalence (eval): 0.310

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
| 0 | 279 | 0.1126 | 0.2042 | 0.0915 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 191/2 |  |
| 1 | 191 | 0.1126 | 0.2042 | 0.0915 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 191/191 features;  |  |
| 2 | 191 | 0.1126 | 0.2042 | 0.0915 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 157.840
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2237 | 0.2141 | -0.0096 | 0.6521 | 0.5359 |
| test | 0.1142 | 0.0766 | -0.0376 | 0.3992 | 0.6452 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.3105

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5297 | 0.3105 | 202 | 107 | 202 | 0 / 202 / 202 |
| 5 | 0.4465 | 0.3105 | 1010 | 451 | 1010 | 0 / 202 / 202 |
| 10 | 0.4158 | 0.3105 | 2020 | 840 | 2020 | 0 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3105 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.3105 | 5 | 2 | 5 |
| 10 | 0.3000 | 0.3105 | 10 | 3 | 10 |

### test — n_rows=37600, base_rate=0.0836

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1900 | 0.0836 | 100 | 19 | 100 | 0 / 100 / 100 |
| 5 | 0.2000 | 0.0836 | 500 | 100 | 500 | 0 / 100 / 100 |
| 10 | 0.1897 | 0.0836 | 1000 | 187 | 986 | 7 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0836 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.0836 | 5 | 2 | 5 |
| 10 | 0.4000 | 0.0836 | 10 | 4 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:BEML | 64 | 25 | 0.3906 |
| NSE:GVT&D | 51 | 27 | 0.5294 |
| NSE:COCHINSHIP | 49 | 13 | 0.2653 |
| NSE:IREDA | 44 | 21 | 0.4773 |
| NSE:NLCINDIA | 38 | 3 | 0.0789 |
| NSE:NEWGEN | 36 | 19 | 0.5278 |
| NSE:FACT | 32 | 11 | 0.3438 |
| NSE:KPIL | 32 | 3 | 0.0938 |
| NSE:HUDCO | 29 | 14 | 0.4828 |
| NSE:CEMPRO | 27 | 27 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:RKFORGE | 12 | 0 | 0.0000 |
| NSE:360ONE | 10 | 0 | 0.0000 |
| NSE:PNBHOUSING | 8 | 0 | 0.0000 |
| NSE:POLYMED | 6 | 0 | 0.0000 |
| NSE:DIXON | 5 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:GALLANTT | 70 | 17 | 0.2429 |
| NSE:360ONE | 32 | 8 | 0.2500 |
| NSE:ADANIGREEN | 32 | 0 | 0.0000 |
| NSE:TARIL | 29 | 20 | 0.6897 |
| NSE:HEG | 26 | 10 | 0.3846 |
| NSE:RPOWER | 22 | 9 | 0.4091 |
| NSE:SUZLON | 22 | 2 | 0.0909 |
| NSE:PHOENIXLTD | 18 | 10 | 0.5556 |
| NSE:GRSE | 17 | 1 | 0.0588 |
| NSE:GVT&D | 16 | 1 | 0.0625 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ADANIGREEN | 32 | 0 | 0.0000 |
| NSE:SCHNEIDER | 16 | 0 | 0.0000 |
| NSE:ELECON | 14 | 0 | 0.0000 |
| NSE:LTFOODS | 13 | 0 | 0.0000 |
| NSE:CHOLAHLDNG | 9 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 112 | 0.6588 | 0.3105 |
| 2024Q1 | 310 | 99 | 0.3194 | 0.3105 |
| 2024Q2 | 305 | 181 | 0.5934 | 0.3105 |
| 2024Q3 | 225 | 59 | 0.2622 | 0.3105 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 2 | 0.0211 | 0.0836 |
| 2024Q4 | 310 | 85 | 0.2742 | 0.0836 |
| 2025Q1 | 95 | 13 | 0.1368 | 0.0836 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.1001 | 0.4540 | 0.2235 | 0.0839 | `False` |
| test | 37600 | 0.1001 | 0.5500 | 0.2777 | 0.0700 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=157.84); shipped as `isotonic`. Brier vs base-rate: -0.0096 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
