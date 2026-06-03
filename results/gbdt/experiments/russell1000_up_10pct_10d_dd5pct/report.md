# gbdt experiment — russell1000_up_10pct_10d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `10`
- max_drawdown: `0.05`
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
- positive prevalence (train): 0.101
- positive prevalence (eval): 0.093

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0802 | 0.0684 | -0.0119 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 60/27 |  |
| 1 | 60 | 0.0809 | 0.0685 | -0.0125 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 47/60 features |  |
| 2 | 47 | 0.0803 | 0.0684 | -0.0119 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -18.996
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0767 | 0.0846 | +0.0079 | 0.2805 | 0.7890 |
| test | 0.1043 | 0.1098 | +0.0055 | 0.3547 | 0.6935 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0933

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3186 | 0.0933 | 216 | 65 | 204 | 12 / 216 / 216 |
| 5 | 0.3300 | 0.0933 | 1017 | 331 | 1003 | 17 / 200 / 216 |
| 10 | 0.3456 | 0.0933 | 2017 | 687 | 1988 | 20 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0933 | 1 | 0 | 1 |
| 5 | 0.8000 | 0.0933 | 5 | 4 | 5 |
| 10 | 0.7000 | 0.0933 | 10 | 7 | 10 |

### test — n_rows=80010, base_rate=0.1256

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3222 | 0.1256 | 106 | 29 | 90 | 16 / 106 / 106 |
| 5 | 0.3844 | 0.1256 | 466 | 173 | 450 | 16 / 90 / 106 |
| 10 | 0.3344 | 0.1256 | 916 | 301 | 900 | 16 / 90 / 106 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1256 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.1256 | 5 | 3 | 5 |
| 10 | 0.5000 | 0.1256 | 10 | 5 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 177 | 69 | 0.3898 |
| NASDAQ:MSTR | 139 | 24 | 0.1727 |
| NYSE:MRNA | 111 | 32 | 0.2883 |
| NYSE:RH | 96 | 27 | 0.2812 |
| NYSE:VKTX | 68 | 22 | 0.3235 |
| NYSE:SMMT | 64 | 17 | 0.2656 |
| NYSE:SMCI | 49 | 9 | 0.1837 |
| NYSE:ENPH | 33 | 12 | 0.3636 |
| NYSE:FTAI | 31 | 12 | 0.3871 |
| NYSE:CLF | 27 | 15 | 0.5556 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:INSP | 9 | 1 | 0.1111 |
| NYSE:COHR | 13 | 2 | 0.1538 |
| NASDAQ:MSTR | 139 | 24 | 0.1727 |
| NYSE:SMCI | 49 | 9 | 0.1837 |
| NYSE:SRPT | 19 | 4 | 0.2105 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 88 | 24 | 0.2727 |
| NYSE:COHR | 73 | 49 | 0.6712 |
| NYSE:LITE | 52 | 29 | 0.5577 |
| NYSE:SRPT | 37 | 10 | 0.2703 |
| NYSE:SMMT | 35 | 8 | 0.2286 |
| NASDAQ:MSTR | 34 | 12 | 0.3529 |
| NYSE:ELF | 23 | 9 | 0.3913 |
| NYSE:INSP | 18 | 3 | 0.1667 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CLF | 13 | 2 | 0.1538 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:VKTX | 8 | 1 | 0.1250 |
| NYSE:RH | 7 | 1 | 0.1429 |
| NYSE:CLF | 13 | 2 | 0.1538 |
| NYSE:INSP | 18 | 3 | 0.1667 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 19 | 0.2468 | 0.0933 |
| 2025Q2 | 310 | 133 | 0.4290 | 0.0933 |
| 2025Q3 | 320 | 100 | 0.3125 | 0.0933 |
| 2025Q4 | 310 | 79 | 0.2548 | 0.0933 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 5 | 0.1923 | 0.1256 |
| 2026Q1 | 305 | 120 | 0.3934 | 0.1256 |
| 2026Q2 | 135 | 48 | 0.3556 | 0.1256 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 1.0000 | 0.1184 | 0.1083 | `False` |
| test | 80010 | 0.0000 | 0.4234 | 0.1071 | 0.0790 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=19.00); shipped as `isotonic`. Brier vs base-rate: +0.0079 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
