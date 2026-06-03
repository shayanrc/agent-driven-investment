# gbdt experiment — russell1000_up_20pct_5d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `5`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 79022.2; overlap-inflation 9.00×)
- val rows: 355600 (independent events ≈ 39511.1; overlap-inflation 9.00×)
- eval rows: 177800 (independent events ≈ 19755.6; overlap-inflation 9.00×)
- test rows: 84455 (independent events ≈ 9383.9; overlap-inflation 9.00×)
- sample uniqueness weighting: `on` (horizon_days=5)
- positive prevalence (train): 0.005
- positive prevalence (eval): 0.004

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0045 | 0.0035 | -0.0010 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 43/27 |  |
| 1 | 43 | 0.0046 | 0.0035 | -0.0011 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 39/43 features |  |
| 2 | 39 | 0.0045 | 0.0035 | -0.0010 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 8.230
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0040 | 0.0041 | +0.0001 | 0.0223 | 0.8662 |
| test | 0.0068 | 0.0069 | +0.0001 | 0.0359 | 0.8327 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0041

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1461 | 0.0041 | 216 | 26 | 178 | 38 / 216 / 216 |
| 5 | 0.1630 | 0.0041 | 1017 | 89 | 546 | 163 / 200 / 216 |
| 10 | 0.1941 | 0.0041 | 2017 | 125 | 644 | 207 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0041 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.0041 | 5 | 1 | 5 |
| 10 | 0.4000 | 0.0041 | 10 | 4 | 10 |

### test — n_rows=84455, base_rate=0.0070

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1075 | 0.0070 | 111 | 10 | 93 | 18 / 111 / 111 |
| 5 | 0.1485 | 0.0070 | 491 | 53 | 357 | 61 / 95 / 111 |
| 10 | 0.1719 | 0.0070 | 966 | 88 | 512 | 91 / 95 / 111 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0070 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.0070 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.0070 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 200 | 42 | 0.2100 |
| NASDAQ:MSTR | 159 | 2 | 0.0126 |
| NYSE:SMMT | 151 | 9 | 0.0596 |
| NYSE:SRPT | 118 | 12 | 0.1017 |
| NYSE:SMCI | 61 | 6 | 0.0984 |
| NYSE:LITE | 33 | 7 | 0.2121 |
| NASDAQ:TSLA | 32 | 0 | 0.0000 |
| NYSE:CAR | 30 | 1 | 0.0333 |
| NYSE:CVNA | 30 | 2 | 0.0667 |
| NASDAQ:MRVL | 23 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TSLA | 32 | 0 | 0.0000 |
| NASDAQ:MRVL | 23 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ELF | 9 | 0 | 0.0000 |
| NYSE:ENPH | 8 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 94 | 15 | 0.1596 |
| NYSE:LITE | 91 | 17 | 0.1868 |
| NYSE:COHR | 64 | 5 | 0.0781 |
| NASDAQ:MSTR | 30 | 1 | 0.0333 |
| NYSE:CAR | 30 | 10 | 0.3333 |
| NYSE:CORT | 28 | 0 | 0.0000 |
| NYSE:SRPT | 20 | 0 | 0.0000 |
| NYSE:INSP | 19 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:MRNA | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CORT | 28 | 0 | 0.0000 |
| NYSE:SRPT | 20 | 0 | 0.0000 |
| NYSE:INSP | 19 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:MRNA | 16 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 2 | 0.0260 | 0.0041 |
| 2025Q2 | 310 | 38 | 0.1226 | 0.0041 |
| 2025Q3 | 320 | 21 | 0.0656 | 0.0041 |
| 2025Q4 | 310 | 28 | 0.0903 | 0.0041 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 3 | 0.1154 | 0.0070 |
| 2026Q1 | 305 | 30 | 0.0984 | 0.0070 |
| 2026Q2 | 160 | 20 | 0.1250 | 0.0070 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.7500 | 0.0054 | 0.0152 | `True` |
| test | 84455 | 0.0000 | 0.7500 | 0.0060 | 0.0130 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=8.23); shipped as `isotonic`. Brier vs base-rate: +0.0001 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
