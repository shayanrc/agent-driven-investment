# gbdt experiment — russell1000_up_50pct_50d_dd25pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `50`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 7183.8; overlap-inflation 99.00×)
- val rows: 355600 (independent events ≈ 3591.9; overlap-inflation 99.00×)
- eval rows: 177800 (independent events ≈ 1796.0; overlap-inflation 99.00×)
- test rows: 44450 (independent events ≈ 449.0; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.018
- positive prevalence (eval): 0.021

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0147 | 0.0102 | -0.0045 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 27/27 |  |
| 1 | 27 | 0.0147 | 0.0102 | -0.0045 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 20/27 features |  |
| 2 | 20 | 0.0148 | 0.0102 | -0.0046 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 8.283
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0188 | 0.0203 | +0.0015 | 0.0775 | 0.8951 |
| test | 0.0230 | 0.0242 | +0.0013 | 0.0956 | 0.8575 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0207

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2500 | 0.0207 | 216 | 50 | 200 | 16 / 216 / 216 |
| 5 | 0.2480 | 0.0207 | 1017 | 248 | 1000 | 16 / 200 / 216 |
| 10 | 0.2346 | 0.0207 | 2017 | 454 | 1935 | 53 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0207 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0207 | 5 | 0 | 5 |
| 10 | 0.1000 | 0.0207 | 10 | 1 | 10 |

### test — n_rows=44450, base_rate=0.0248

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1400 | 0.0248 | 66 | 7 | 50 | 16 / 66 / 66 |
| 5 | 0.2600 | 0.0248 | 266 | 65 | 250 | 16 / 50 / 66 |
| 10 | 0.1800 | 0.0248 | 516 | 90 | 500 | 16 / 50 / 66 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0248 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0248 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0248 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 187 | 115 | 0.6150 |
| NYSE:SMMT | 181 | 20 | 0.1105 |
| NASDAQ:MSTR | 179 | 3 | 0.0168 |
| NYSE:MRNA | 90 | 10 | 0.1111 |
| NYSE:SMCI | 89 | 12 | 0.1348 |
| NYSE:RH | 59 | 8 | 0.1356 |
| NYSE:ENPH | 58 | 23 | 0.3966 |
| NYSE:VKTX | 39 | 18 | 0.4615 |
| NYSE:GME | 34 | 6 | 0.1765 |
| NYSE:W | 29 | 9 | 0.3103 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CAR | 8 | 0 | 0.0000 |
| NASDAQ:MSTR | 179 | 3 | 0.0168 |
| NYSE:SMMT | 181 | 20 | 0.1105 |
| NYSE:MRNA | 90 | 10 | 0.1111 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 50 | 2 | 0.0400 |
| NYSE:SMMT | 47 | 25 | 0.5319 |
| NASDAQ:MSTR | 40 | 3 | 0.0750 |
| NYSE:LITE | 40 | 33 | 0.8250 |
| NYSE:SMCI | 28 | 0 | 0.0000 |
| NYSE:SRPT | 23 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CVNA | 14 | 0 | 0.0000 |
| NYSE:ELF | 5 | 0 | 0.0000 |
| NYSE:ENPH | 2 | 2 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:SMCI | 28 | 0 | 0.0000 |
| NYSE:SRPT | 23 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CVNA | 14 | 0 | 0.0000 |
| NYSE:ELF | 5 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 12 | 0.1558 | 0.0207 |
| 2025Q2 | 310 | 96 | 0.3097 | 0.0207 |
| 2025Q3 | 320 | 61 | 0.1906 | 0.0207 |
| 2025Q4 | 310 | 79 | 0.2548 | 0.0207 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 6 | 0.2308 | 0.0248 |
| 2026Q1 | 240 | 59 | 0.2458 | 0.0248 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.6066 | 0.0172 | 0.0320 | `True` |
| test | 44450 | 0.0000 | 0.6066 | 0.0168 | 0.0321 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=8.28); shipped as `isotonic`. Brier vs base-rate: +0.0015 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
