# gbdt experiment — russell1000_up_10pct_25d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `25`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 14514.3; overlap-inflation 49.00×)
- val rows: 355600 (independent events ≈ 7257.1; overlap-inflation 49.00×)
- eval rows: 177800 (independent events ≈ 3628.6; overlap-inflation 49.00×)
- test rows: 66675 (independent events ≈ 1360.7; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.250
- positive prevalence (eval): 0.237

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1686 | 0.1660 | -0.0026 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 33/27 |  |
| 1 | 33 | 0.1711 | 0.1661 | -0.0050 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 29/33 features |  |
| 2 | 29 | 0.1700 | 0.1657 | -0.0043 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -15.706
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1684 | 0.1809 | +0.0125 | 0.5124 | 0.6890 |
| test | 0.2019 | 0.2038 | +0.0019 | 0.5947 | 0.5821 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.2372

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5463 | 0.2372 | 216 | 112 | 205 | 11 / 216 / 216 |
| 5 | 0.4458 | 0.2372 | 1017 | 448 | 1005 | 16 / 200 / 216 |
| 10 | 0.4401 | 0.2372 | 2017 | 881 | 2002 | 17 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.2372 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.2372 | 5 | 2 | 5 |
| 10 | 0.5000 | 0.2372 | 10 | 5 | 10 |

### test — n_rows=66675, base_rate=0.2851

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5200 | 0.2851 | 91 | 39 | 75 | 16 / 91 / 91 |
| 5 | 0.4213 | 0.2851 | 391 | 158 | 375 | 16 / 75 / 91 |
| 10 | 0.4240 | 0.2851 | 766 | 318 | 750 | 16 / 75 / 91 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.2851 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.2851 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.2851 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 164 | 46 | 0.2805 |
| NASDAQ:MDB | 130 | 72 | 0.5538 |
| NASDAQ:INTC | 128 | 65 | 0.5078 |
| NASDAQ:MRVL | 109 | 56 | 0.5138 |
| NASDAQ:MCHP | 60 | 8 | 0.1333 |
| NYSE:ASTS | 54 | 19 | 0.3519 |
| NASDAQ:AVGO | 51 | 45 | 0.8824 |
| NYSE:SMCI | 32 | 3 | 0.0938 |
| NASDAQ:MU | 29 | 19 | 0.6552 |
| NASDAQ:TSLA | 28 | 12 | 0.4286 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CAR | 18 | 0 | 0.0000 |
| NYSE:ELF | 6 | 0 | 0.0000 |
| NYSE:CLVT | 18 | 1 | 0.0556 |
| NYSE:SMCI | 32 | 3 | 0.0938 |
| NASDAQ:MCHP | 60 | 8 | 0.1333 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:INTC | 59 | 19 | 0.3220 |
| NASDAQ:MDB | 47 | 11 | 0.2340 |
| NASDAQ:MSTR | 41 | 16 | 0.3902 |
| NASDAQ:MRVL | 24 | 9 | 0.3750 |
| NASDAQ:MU | 19 | 11 | 0.5789 |
| NYSE:W | 19 | 3 | 0.1579 |
| NYSE:COHR | 18 | 16 | 0.8889 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NASDAQ:AMD | 15 | 11 | 0.7333 |
| NYSE:VRT | 15 | 11 | 0.7333 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NASDAQ:TSLA | 11 | 0 | 0.0000 |
| NYSE:RH | 7 | 0 | 0.0000 |
| NASDAQ:MCHP | 7 | 1 | 0.1429 |
| NYSE:W | 19 | 3 | 0.1579 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 18 | 0.2338 | 0.2372 | 0.986 |
| 2025Q2 | 310 | 199 | 0.6419 | 0.2372 | 2.707 |
| 2025Q3 | 320 | 120 | 0.3750 | 0.2372 | 1.581 |
| 2025Q4 | 310 | 111 | 0.3581 | 0.2372 | 1.510 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 6 | 0.2308 | 0.2851 | 0.810 |
| 2026Q1 | 305 | 102 | 0.3344 | 0.2851 | 1.173 |
| 2026Q2 | 60 | 50 | 0.8333 | 0.2851 | 2.923 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.7162 | 0.2398 | 0.0848 | `False` |
| test | 66675 | 0.0462 | 0.5334 | 0.2495 | 0.0718 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=15.71); shipped as `isotonic`. Brier vs base-rate: +0.0125 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
