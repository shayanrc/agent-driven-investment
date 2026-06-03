# gbdt experiment — russell1000_up_20pct_10d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `10`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 37431.6; overlap-inflation 19.00×)
- val rows: 355600 (independent events ≈ 18715.8; overlap-inflation 19.00×)
- eval rows: 177800 (independent events ≈ 9357.9; overlap-inflation 19.00×)
- test rows: 80010 (independent events ≈ 4211.1; overlap-inflation 19.00×)
- sample uniqueness weighting: `on` (horizon_days=10)
- positive prevalence (train): 0.017
- positive prevalence (eval): 0.014

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0150 | 0.0105 | -0.0045 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 33/27 |  |
| 1 | 33 | 0.0150 | 0.0105 | -0.0045 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 30/33 features |  |
| 2 | 30 | 0.0151 | 0.0105 | -0.0046 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -2.628
- Spiegelhalter p: 0.0086

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0130 | 0.0137 | +0.0007 | 0.0601 | 0.8572 |
| test | 0.0231 | 0.0243 | +0.0012 | 0.0991 | 0.8178 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0139

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3198 | 0.0139 | 216 | 63 | 197 | 19 / 216 / 216 |
| 5 | 0.1909 | 0.0139 | 1017 | 172 | 901 | 55 / 200 / 216 |
| 10 | 0.1904 | 0.0139 | 2017 | 288 | 1513 | 112 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0139 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.0139 | 5 | 3 | 5 |
| 10 | 0.6000 | 0.0139 | 10 | 6 | 10 |

### test — n_rows=80010, base_rate=0.0249

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2556 | 0.0249 | 106 | 23 | 90 | 16 / 106 / 106 |
| 5 | 0.2227 | 0.0249 | 466 | 100 | 449 | 17 / 90 / 106 |
| 10 | 0.2249 | 0.0249 | 916 | 188 | 836 | 34 / 90 / 106 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0249 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.0249 | 5 | 4 | 5 |
| 10 | 0.8000 | 0.0249 | 10 | 8 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 200 | 67 | 0.3350 |
| NYSE:SMMT | 177 | 17 | 0.0960 |
| NYSE:SRPT | 108 | 27 | 0.2500 |
| NYSE:SMCI | 87 | 13 | 0.1494 |
| NASDAQ:MSTR | 86 | 9 | 0.1047 |
| NYSE:VKTX | 82 | 10 | 0.1220 |
| NYSE:MRNA | 47 | 0 | 0.0000 |
| NYSE:RH | 39 | 1 | 0.0256 |
| NYSE:LITE | 34 | 15 | 0.4412 |
| NYSE:ENPH | 27 | 1 | 0.0370 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:MRNA | 47 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:QXO | 12 | 0 | 0.0000 |
| NASDAQ:TSLA | 9 | 0 | 0.0000 |
| NYSE:ACHC | 8 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 90 | 25 | 0.2778 |
| NYSE:LITE | 90 | 35 | 0.3889 |
| NYSE:COHR | 72 | 25 | 0.3472 |
| NYSE:SRPT | 28 | 0 | 0.0000 |
| NASDAQ:MSTR | 27 | 1 | 0.0370 |
| NYSE:CIEN | 27 | 9 | 0.3333 |
| NYSE:SMMT | 20 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CAR | 16 | 1 | 0.0625 |
| NYSE:CORT | 15 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:SRPT | 28 | 0 | 0.0000 |
| NYSE:SMMT | 20 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CORT | 15 | 0 | 0.0000 |
| NYSE:INSP | 13 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 5 | 0.0649 | 0.0139 |
| 2025Q2 | 310 | 67 | 0.2161 | 0.0139 |
| 2025Q3 | 320 | 56 | 0.1750 | 0.0139 |
| 2025Q4 | 310 | 44 | 0.1419 | 0.0139 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 2 | 0.0769 | 0.0249 |
| 2026Q1 | 305 | 72 | 0.2361 | 0.0249 |
| 2026Q2 | 135 | 26 | 0.1926 | 0.0249 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.7500 | 0.0176 | 0.0267 | `True` |
| test | 80010 | 0.0000 | 0.2527 | 0.0198 | 0.0284 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.63); shipped as `isotonic`. Brier vs base-rate: +0.0007 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
