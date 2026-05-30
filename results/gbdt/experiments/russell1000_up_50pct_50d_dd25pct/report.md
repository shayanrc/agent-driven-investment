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

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: 0.655
- Spiegelhalter p: 0.5122

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0190 | 0.0203 | +0.0013 | 0.0800 | 0.8873 |
| test | 0.0231 | 0.0242 | +0.0011 | 0.0995 | 0.8294 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0207

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4450 | 0.0207 | 216 | 89 | 200 | 16 / 216 / 216 |
| 5 | 0.2490 | 0.0207 | 1017 | 249 | 1000 | 16 / 200 / 216 |
| 10 | 0.2238 | 0.0207 | 2017 | 433 | 1935 | 53 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0207 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.0207 | 5 | 5 | 5 |
| 10 | 0.6000 | 0.0207 | 10 | 6 | 10 |

### test — n_rows=44450, base_rate=0.0248

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0400 | 0.0248 | 66 | 2 | 50 | 16 / 66 / 66 |
| 5 | 0.2960 | 0.0248 | 266 | 74 | 250 | 16 / 50 / 66 |
| 10 | 0.2340 | 0.0248 | 516 | 117 | 500 | 16 / 50 / 66 |

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
| NYSE:ASTS | 200 | 118 | 0.5900 |
| NYSE:SMMT | 186 | 22 | 0.1183 |
| NYSE:SMCI | 174 | 16 | 0.0920 |
| NASDAQ:MSTR | 146 | 3 | 0.0205 |
| NYSE:VKTX | 82 | 21 | 0.2561 |
| NYSE:RH | 69 | 8 | 0.1159 |
| NYSE:MRNA | 31 | 4 | 0.1290 |
| NYSE:SRPT | 29 | 7 | 0.2414 |
| NYSE:W | 26 | 24 | 0.9231 |
| NYSE:AL | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CVNA | 16 | 0 | 0.0000 |
| NYSE:CAR | 8 | 0 | 0.0000 |
| NASDAQ:MSTR | 146 | 3 | 0.0205 |
| NYSE:SMCI | 174 | 16 | 0.0920 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 50 | 2 | 0.0400 |
| NYSE:SMMT | 50 | 28 | 0.5600 |
| NYSE:LITE | 49 | 41 | 0.8367 |
| NYSE:VKTX | 38 | 0 | 0.0000 |
| NYSE:SRPT | 32 | 0 | 0.0000 |
| NASDAQ:MSTR | 19 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:INSP | 4 | 0 | 0.0000 |
| NYSE:CORT | 3 | 3 | 1.0000 |
| NYSE:RH | 2 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:VKTX | 38 | 0 | 0.0000 |
| NYSE:SRPT | 32 | 0 | 0.0000 |
| NASDAQ:MSTR | 19 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ASTS | 50 | 2 | 0.0400 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 13 | 0.1688 | 0.0207 | 8.144 |
| 2025Q2 | 310 | 117 | 0.3774 | 0.0207 | 18.205 |
| 2025Q3 | 320 | 61 | 0.1906 | 0.0207 | 9.195 |
| 2025Q4 | 310 | 58 | 0.1871 | 0.0207 | 9.025 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 4 | 0.1538 | 0.0248 | 6.194 |
| 2026Q1 | 240 | 70 | 0.2917 | 0.0248 | 11.743 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0006 | 0.5183 | 0.0152 | 0.0260 | `True` |
| test | 44450 | 0.0008 | 0.4388 | 0.0157 | 0.0251 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=0.66<2). Brier vs base-rate: +0.0013 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
