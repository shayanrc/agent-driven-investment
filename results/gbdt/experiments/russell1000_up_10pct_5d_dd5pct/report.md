# gbdt experiment — russell1000_up_10pct_5d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `5`
- max_drawdown: `0.05`
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
- positive prevalence (train): 0.040
- positive prevalence (eval): 0.035

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0346 | 0.0263 | -0.0082 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 67/27 |  |
| 1 | 67 | 0.0349 | 0.0264 | -0.0085 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 48/67 features |  |
| 2 | 48 | 0.0349 | 0.0263 | -0.0086 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: -1.211
- Spiegelhalter p: 0.2259

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0307 | 0.0335 | +0.0027 | 0.1268 | 0.8244 |
| test | 0.0449 | 0.0473 | +0.0024 | 0.1765 | 0.7624 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0347

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2800 | 0.0347 | 216 | 56 | 200 | 16 / 216 / 216 |
| 5 | 0.2315 | 0.0347 | 1017 | 225 | 972 | 31 / 200 / 216 |
| 10 | 0.2371 | 0.0347 | 2017 | 429 | 1809 | 60 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0347 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.0347 | 5 | 2 | 5 |
| 10 | 0.6000 | 0.0347 | 10 | 6 | 10 |

### test — n_rows=84455, base_rate=0.0498

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2632 | 0.0498 | 111 | 25 | 95 | 16 / 111 / 111 |
| 5 | 0.2737 | 0.0498 | 491 | 130 | 475 | 16 / 95 / 111 |
| 10 | 0.2358 | 0.0498 | 966 | 224 | 950 | 16 / 95 / 111 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0498 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.0498 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.0498 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 182 | 65 | 0.3571 |
| NYSE:VKTX | 133 | 22 | 0.1654 |
| NYSE:SRPT | 114 | 24 | 0.2105 |
| NYSE:SMMT | 113 | 26 | 0.2301 |
| NYSE:ENPH | 68 | 8 | 0.1176 |
| NYSE:RH | 67 | 13 | 0.1940 |
| NYSE:MRNA | 54 | 4 | 0.0741 |
| NYSE:SMCI | 52 | 9 | 0.1731 |
| NASDAQ:MSTR | 37 | 0 | 0.0000 |
| NYSE:W | 33 | 15 | 0.4545 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 37 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ELF | 15 | 0 | 0.0000 |
| NYSE:NWL | 9 | 0 | 0.0000 |
| NYSE:MRNA | 54 | 4 | 0.0741 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 89 | 24 | 0.2697 |
| NYSE:LITE | 85 | 34 | 0.4000 |
| NYSE:COHR | 59 | 28 | 0.4746 |
| NYSE:SRPT | 46 | 5 | 0.1087 |
| NYSE:INSP | 24 | 1 | 0.0417 |
| NASDAQ:MSTR | 21 | 4 | 0.1905 |
| NYSE:SMMT | 19 | 1 | 0.0526 |
| NYSE:WING | 18 | 4 | 0.2222 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CORT | 15 | 3 | 0.2000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:BRBR | 14 | 0 | 0.0000 |
| NYSE:INSP | 24 | 1 | 0.0417 |
| NYSE:SMMT | 19 | 1 | 0.0526 |
| NYSE:SRPT | 46 | 5 | 0.1087 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 11 | 0.1429 | 0.0347 |
| 2025Q2 | 310 | 86 | 0.2774 | 0.0347 |
| 2025Q3 | 320 | 71 | 0.2219 | 0.0347 |
| 2025Q4 | 310 | 57 | 0.1839 | 0.0347 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 2 | 0.0769 | 0.0498 |
| 2026Q1 | 305 | 83 | 0.2721 | 0.0498 |
| 2026Q2 | 160 | 45 | 0.2812 | 0.0498 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0044 | 0.6820 | 0.0438 | 0.0463 | `True` |
| test | 84455 | 0.0043 | 0.2956 | 0.0459 | 0.0409 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=1.21<2). Brier vs base-rate: +0.0027 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
