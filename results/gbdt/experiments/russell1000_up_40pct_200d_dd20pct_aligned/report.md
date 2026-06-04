# gbdt experiment — russell1000_up_40pct_200d_dd20pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `200`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 858
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:ASTS, NYSE:AUR, NYSE:AVTR, NYSE:BAM, NYSE:BEPC, NYSE:BILL, NYSE:BIRK, NYSE:BJ, NYSE:BLSH, NYSE:BRBR, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CHWY, NYSE:CLVT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:CTVA, NYSE:DJT, NYSE:DKNG, NYSE:DOCS, NYSE:DOW, NYSE:DT, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ELAN, NYSE:ESAB, NYSE:ESTC, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FOX, NYSE:FOXA, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:LYFT, NYSE:MDLN, NYSE:MP, NYSE:MRNA, NYSE:MRP, NYSE:NCNO, NYSE:NET, NYSE:NIQ, NYSE:NU, NYSE:NVST, NYSE:OGN, NYSE:ONON, NYSE:ONTO, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:PINS, NYSE:PSN, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TIGO, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:TW, NYSE:U, NYSE:UBER, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VRT, NYSE:VSNT, NYSE:WFRD, NYSE:XP, NYSE:YETI, NYSE:ZM
- train rows: 685740 (independent events ≈ 1721.4; overlap-inflation 398.37×)
- val rows: 343200 (independent events ≈ 860.2; overlap-inflation 399.00×)
- eval rows: 171600 (independent events ≈ 430.1; overlap-inflation 399.00×)
- test rows: 257400 (independent events ≈ 645.1; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.242
- positive prevalence (eval): 0.195

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2018-01-01`
- train: `2018-01-02` → `2021-03-08`
- val: `2021-03-09` → `2022-10-06`
- eval: `2022-10-07` → `2023-07-26`
- test: `2023-07-27` → `2024-10-03`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1251 | 0.1117 | -0.0134 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 53/27 |  |
| 1 | 53 | 0.1239 | 0.1154 | -0.0085 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -134.584
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1491 | 0.1570 | +0.0080 | 0.4766 | 0.6920 |
| test | 0.2024 | 0.1806 | -0.0218 | 0.6954 | 0.6618 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=171600, base_rate=0.1951

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6750 | 0.1951 | 200 | 135 | 200 | 0 / 200 / 200 |
| 5 | 0.5450 | 0.1951 | 1000 | 545 | 1000 | 0 / 200 / 200 |
| 10 | 0.5040 | 0.1951 | 2000 | 1008 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1951 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.1951 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.1951 | 10 | 0 | 10 |

### test — n_rows=257400, base_rate=0.2365

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7167 | 0.2365 | 300 | 215 | 300 | 0 / 300 / 300 |
| 5 | 0.5067 | 0.2365 | 1500 | 760 | 1500 | 0 / 300 / 300 |
| 10 | 0.4647 | 0.2365 | 3000 | 1394 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2365 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.2365 | 5 | 4 | 5 |
| 10 | 0.5000 | 0.2365 | 10 | 5 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MDB | 122 | 79 | 0.6475 |
| NASDAQ:TTD | 117 | 95 | 0.8120 |
| NASDAQ:MSTR | 99 | 75 | 0.7576 |
| NYSE:CELH | 94 | 65 | 0.6915 |
| NYSE:CCL | 69 | 36 | 0.5217 |
| NYSE:CAR | 60 | 15 | 0.2500 |
| NYSE:GME | 55 | 8 | 0.1455 |
| NASDAQ:TEAM | 48 | 27 | 0.5625 |
| NYSE:SMCI | 47 | 38 | 0.8085 |
| NYSE:AA | 41 | 1 | 0.0244 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ENPH | 10 | 0 | 0.0000 |
| NYSE:XYZ | 9 | 0 | 0.0000 |
| NYSE:DOCU | 5 | 0 | 0.0000 |
| NYSE:AA | 41 | 1 | 0.0244 |
| NASDAQ:ZS | 21 | 1 | 0.0476 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 285 | 229 | 0.8035 |
| NASDAQ:MSTR | 223 | 137 | 0.6143 |
| NYSE:SMCI | 220 | 88 | 0.4000 |
| NYSE:SMMT | 166 | 99 | 0.5964 |
| NYSE:VKTX | 121 | 48 | 0.3967 |
| NYSE:GME | 112 | 62 | 0.5536 |
| NYSE:W | 96 | 20 | 0.2083 |
| NYSE:MPT | 79 | 16 | 0.2025 |
| NYSE:ROKU | 58 | 1 | 0.0172 |
| NYSE:ZION | 46 | 29 | 0.6304 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:RNG | 18 | 0 | 0.0000 |
| NYSE:ENPH | 5 | 0 | 0.0000 |
| NYSE:ROKU | 58 | 1 | 0.0172 |
| NYSE:RH | 9 | 1 | 0.1111 |
| NASDAQ:TEAM | 6 | 1 | 0.1667 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 116 | 0.3932 | 0.1951 |
| 2023Q1 | 310 | 187 | 0.6032 | 0.1951 |
| 2023Q2 | 310 | 228 | 0.7355 | 0.1951 |
| 2023Q3 | 85 | 14 | 0.1647 | 0.1951 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 63 | 0.2739 | 0.2365 |
| 2023Q4 | 315 | 229 | 0.7270 | 0.2365 |
| 2024Q1 | 305 | 178 | 0.5836 | 0.2365 |
| 2024Q2 | 315 | 117 | 0.3714 | 0.2365 |
| 2024Q3 | 320 | 164 | 0.5125 | 0.2365 |
| 2024Q4 | 15 | 9 | 0.6000 | 0.2365 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 171600 | 0.0000 | 0.4837 | 0.1275 | 0.0972 | `False` |
| test | 257400 | 0.0000 | 0.4837 | 0.0654 | 0.0512 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=134.58); shipped as `isotonic`. Brier vs base-rate: +0.0080 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
