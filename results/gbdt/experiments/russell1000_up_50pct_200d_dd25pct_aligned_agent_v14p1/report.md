# gbdt experiment — russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `200`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Data

- tickers in universe: 1002
- tickers used: 858
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:ASTS, NYSE:AUR, NYSE:AVTR, NYSE:BAM, NYSE:BEPC, NYSE:BILL, NYSE:BIRK, NYSE:BJ, NYSE:BLSH, NYSE:BRBR, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CHWY, NYSE:CLVT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:CTVA, NYSE:DJT, NYSE:DKNG, NYSE:DOCS, NYSE:DOW, NYSE:DT, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ELAN, NYSE:ESAB, NYSE:ESTC, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FOX, NYSE:FOXA, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:LYFT, NYSE:MDLN, NYSE:MP, NYSE:MRNA, NYSE:MRP, NYSE:NCNO, NYSE:NET, NYSE:NIQ, NYSE:NU, NYSE:NVST, NYSE:OGN, NYSE:ONON, NYSE:ONTO, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:PINS, NYSE:PSN, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TIGO, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:TW, NYSE:U, NYSE:UBER, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VRT, NYSE:VSNT, NYSE:WFRD, NYSE:XP, NYSE:YETI, NYSE:ZM
- train rows: 685740 (independent events ≈ 1721.4; overlap-inflation 398.37×)
- val rows: 343200 (independent events ≈ 860.2; overlap-inflation 399.00×)
- eval rows: 171600 (independent events ≈ 430.1; overlap-inflation 399.00×)
- test rows: 257400 (independent events ≈ 645.1; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.181
- positive prevalence (eval): 0.125

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

## Final checkpoint

- best iteration: 0
- iterations run: 0
- inner stop signal: `agent_should_stop`
- fs_hp_loop callback_mode: `agent_file_protocol`
- tie-break path: `strict_val_brier` — Strict val_brier argmin (no tie-break entered)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -121.492
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1019 | 0.1093 | +0.0074 | 0.3555 | 0.7532 |
| test | 0.1350 | 0.1288 | -0.0062 | 0.5553 | 0.7256 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=171600, base_rate=0.1249

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.8650 | 0.1249 | 200 | 173 | 200 | 0 / 200 / 200 |
| 5 | 0.6450 | 0.1249 | 1000 | 645 | 1000 | 0 / 200 / 200 |
| 10 | 0.5360 | 0.1249 | 2000 | 1072 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1249 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1249 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1249 | 10 | 10 | 10 |

### test — n_rows=257400, base_rate=0.1518

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6467 | 0.1518 | 300 | 194 | 300 | 0 / 300 / 300 |
| 5 | 0.5467 | 0.1518 | 1500 | 820 | 1500 | 0 / 300 / 300 |
| 10 | 0.5087 | 0.1518 | 3000 | 1526 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1518 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1518 | 5 | 5 | 5 |
| 10 | 0.7000 | 0.1518 | 10 | 7 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 159 | 130 | 0.8176 |
| NYSE:CCL | 117 | 87 | 0.7436 |
| NYSE:W | 109 | 84 | 0.7706 |
| NASDAQ:MDB | 85 | 76 | 0.8941 |
| NYSE:CELH | 60 | 60 | 1.0000 |
| NYSE:ROKU | 52 | 49 | 0.9423 |
| NYSE:NCLH | 43 | 26 | 0.6047 |
| NYSE:CROX | 32 | 1 | 0.0312 |
| NASDAQ:TSLA | 25 | 14 | 0.5600 |
| NYSE:CVNA | 25 | 10 | 0.4000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CAR | 22 | 0 | 0.0000 |
| NYSE:GME | 20 | 0 | 0.0000 |
| NYSE:CF | 10 | 0 | 0.0000 |
| NYSE:CHRD | 10 | 0 | 0.0000 |
| NYSE:ETSY | 10 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 271 | 207 | 0.7638 |
| NYSE:SMMT | 241 | 154 | 0.6390 |
| NYSE:CVNA | 228 | 202 | 0.8860 |
| NYSE:SMCI | 192 | 51 | 0.2656 |
| NYSE:W | 182 | 26 | 0.1429 |
| NYSE:VKTX | 155 | 73 | 0.4710 |
| NASDAQ:META | 32 | 32 | 1.0000 |
| NYSE:CROX | 16 | 14 | 0.8750 |
| NYSE:AMKR | 15 | 0 | 0.0000 |
| NYSE:INSP | 14 | 10 | 0.7143 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AMKR | 15 | 0 | 0.0000 |
| NYSE:ETSY | 11 | 0 | 0.0000 |
| NYSE:SMG | 9 | 0 | 0.0000 |
| NYSE:UAA | 9 | 0 | 0.0000 |
| NASDAQ:MDB | 5 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 185 | 0.6271 | 0.1249 |
| 2023Q1 | 310 | 243 | 0.7839 | 0.1249 |
| 2023Q2 | 310 | 204 | 0.6581 | 0.1249 |
| 2023Q3 | 85 | 13 | 0.1529 | 0.1249 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 123 | 0.5348 | 0.1518 |
| 2023Q4 | 315 | 251 | 0.7968 | 0.1518 |
| 2024Q1 | 305 | 150 | 0.4918 | 0.1518 |
| 2024Q2 | 315 | 131 | 0.4159 | 0.1518 |
| 2024Q3 | 320 | 162 | 0.5062 | 0.1518 |
| 2024Q4 | 15 | 3 | 0.2000 | 0.1518 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 171600 | 0.0000 | 1.0000 | 0.0703 | 0.0881 | `False` |
| test | 257400 | 0.0000 | 0.6279 | 0.0361 | 0.0462 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=121.49); shipped as `isotonic`. Brier vs base-rate: +0.0074 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
