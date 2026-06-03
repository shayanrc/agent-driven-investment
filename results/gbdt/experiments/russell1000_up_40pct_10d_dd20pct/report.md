# gbdt experiment — russell1000_up_40pct_10d_dd20pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `10`
- max_drawdown: `0.2`
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
- positive prevalence (train): 0.002
- positive prevalence (eval): 0.001

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0015 | 0.0012 | -0.0004 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 20/27 |  |
| 1 | 20 | 0.0015 | 0.0011 | -0.0003 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 20/20 features |  |
| 2 | 20 | 0.0015 | 0.0011 | -0.0003 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -11.559
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0012 | 0.0012 | +0.0000 | 0.0078 | 0.8692 |
| test | 0.0024 | 0.0024 | +0.0000 | 0.0149 | 0.8613 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0012

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2647 | 0.0012 | 216 | 36 | 136 | 80 / 216 / 216 |
| 5 | 0.2558 | 0.0012 | 1017 | 55 | 215 | 216 / 200 / 216 |
| 10 | 0.2884 | 0.0012 | 2017 | 62 | 215 | 216 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0012 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0012 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0012 | 10 | 0 | 10 |

### test — n_rows=80010, base_rate=0.0024

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0290 | 0.0024 | 106 | 2 | 69 | 37 / 106 / 106 |
| 5 | 0.1488 | 0.0024 | 466 | 25 | 168 | 96 / 90 / 106 |
| 10 | 0.2316 | 0.0024 | 916 | 44 | 190 | 106 / 90 / 106 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0024 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0024 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0024 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 200 | 36 | 0.1800 |
| NYSE:SMMT | 184 | 8 | 0.0435 |
| NASDAQ:MSTR | 144 | 1 | 0.0069 |
| NYSE:SMCI | 138 | 6 | 0.0435 |
| NYSE:SRPT | 132 | 1 | 0.0076 |
| NYSE:LITE | 27 | 3 | 0.1111 |
| NYSE:QXO | 22 | 0 | 0.0000 |
| NYSE:INSP | 18 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:VKTX | 15 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:QXO | 22 | 0 | 0.0000 |
| NYSE:INSP | 18 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:VKTX | 15 | 0 | 0.0000 |
| NYSE:ELF | 12 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 90 | 2 | 0.0222 |
| NYSE:LITE | 73 | 8 | 0.1096 |
| NYSE:COHR | 42 | 0 | 0.0000 |
| NYSE:SRPT | 31 | 2 | 0.0645 |
| NYSE:CAR | 28 | 13 | 0.4643 |
| NYSE:CORT | 21 | 0 | 0.0000 |
| NASDAQ:MSTR | 20 | 0 | 0.0000 |
| NYSE:CIEN | 18 | 0 | 0.0000 |
| NYSE:SMMT | 18 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:COHR | 42 | 0 | 0.0000 |
| NYSE:CORT | 21 | 0 | 0.0000 |
| NASDAQ:MSTR | 20 | 0 | 0.0000 |
| NYSE:CIEN | 18 | 0 | 0.0000 |
| NYSE:SMMT | 18 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 0 | 0.0000 | 0.0012 |
| 2025Q2 | 310 | 29 | 0.0935 | 0.0012 |
| 2025Q3 | 320 | 13 | 0.0406 | 0.0012 |
| 2025Q4 | 310 | 13 | 0.0419 | 0.0012 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 0 | 0.0000 | 0.0024 |
| 2026Q1 | 305 | 12 | 0.0393 | 0.0024 |
| 2026Q2 | 135 | 13 | 0.0963 | 0.0024 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.1355 | 0.0015 | 0.0070 | `True` |
| test | 80010 | 0.0000 | 0.1355 | 0.0017 | 0.0060 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=11.56); shipped as `isotonic`. Brier vs base-rate: +0.0000 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
