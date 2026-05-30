# gbdt experiment — russell1000_up_40pct_50d_dd20pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `50`
- max_drawdown: `0.2`
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
- positive prevalence (train): 0.034
- positive prevalence (eval): 0.039

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0273 | 0.0207 | -0.0067 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 39/27 |  |
| 1 | 39 | 0.0274 | 0.0206 | -0.0068 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 29/39 features |  |
| 2 | 29 | 0.0274 | 0.0207 | -0.0067 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -12.947
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0341 | 0.0378 | +0.0036 | 0.1305 | 0.8586 |
| test | 0.0398 | 0.0430 | +0.0033 | 0.1579 | 0.8032 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0393

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3050 | 0.0393 | 216 | 61 | 200 | 16 / 216 / 216 |
| 5 | 0.3520 | 0.0393 | 1017 | 352 | 1000 | 16 / 200 / 216 |
| 10 | 0.3145 | 0.0393 | 2017 | 628 | 1997 | 18 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0393 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.0393 | 5 | 1 | 5 |
| 10 | 0.3000 | 0.0393 | 10 | 3 | 10 |

### test — n_rows=44450, base_rate=0.0451

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1200 | 0.0451 | 66 | 6 | 50 | 16 / 66 / 66 |
| 5 | 0.4000 | 0.0451 | 266 | 100 | 250 | 16 / 50 / 66 |
| 10 | 0.3560 | 0.0451 | 516 | 178 | 500 | 16 / 50 / 66 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0451 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0451 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0451 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 198 | 111 | 0.5606 |
| NASDAQ:MSTR | 186 | 10 | 0.0538 |
| NYSE:SMMT | 172 | 38 | 0.2209 |
| NYSE:SMCI | 117 | 29 | 0.2479 |
| NYSE:RH | 59 | 25 | 0.4237 |
| NYSE:MRNA | 52 | 2 | 0.0385 |
| NYSE:CVNA | 37 | 21 | 0.5676 |
| NYSE:VKTX | 33 | 19 | 0.5758 |
| NYSE:QXO | 27 | 26 | 0.9630 |
| NYSE:ENPH | 24 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ENPH | 24 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:MRNA | 52 | 2 | 0.0385 |
| NASDAQ:MSTR | 186 | 10 | 0.0538 |
| NYSE:SMMT | 172 | 38 | 0.2209 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 50 | 4 | 0.0800 |
| NYSE:LITE | 50 | 47 | 0.9400 |
| NYSE:SMMT | 50 | 29 | 0.5800 |
| NYSE:SRPT | 50 | 6 | 0.1200 |
| NYSE:VKTX | 28 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:COHR | 15 | 12 | 0.8000 |
| NYSE:W | 4 | 0 | 0.0000 |
| NASDAQ:MSTR | 2 | 2 | 1.0000 |
| NYSE:ELF | 1 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:VKTX | 28 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ASTS | 50 | 4 | 0.0800 |
| NYSE:SRPT | 50 | 6 | 0.1200 |
| NYSE:SMMT | 50 | 29 | 0.5800 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 25 | 0.3247 | 0.0393 | 8.262 |
| 2025Q2 | 310 | 163 | 0.5258 | 0.0393 | 13.380 |
| 2025Q3 | 320 | 65 | 0.2031 | 0.0393 | 5.169 |
| 2025Q4 | 310 | 99 | 0.3194 | 0.0393 | 8.127 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 4 | 0.1538 | 0.0451 | 3.414 |
| 2026Q1 | 240 | 96 | 0.4000 | 0.0451 | 8.877 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.9538 | 0.0330 | 0.0494 | `True` |
| test | 44450 | 0.0000 | 0.3935 | 0.0313 | 0.0467 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=12.95); shipped as `isotonic`. Brier vs base-rate: +0.0036 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
