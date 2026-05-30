# gbdt experiment — russell1000_up_50pct_200d_dd25pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=200 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `200`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 1782.5; overlap-inflation 398.98×)
- val rows: 355600 (independent events ≈ 891.2; overlap-inflation 399.00×)
- eval rows: 88900 (independent events ≈ 222.8; overlap-inflation 399.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.158
- positive prevalence (eval): 0.201

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1025 | 0.1095 | 0.0070 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 24/27 |  |
| 1 | 24 | 0.1033 | 0.1096 | 0.0063 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 21/24 features |  |
| 2 | 21 | 0.1018 | 0.1096 | 0.0078 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -8.100
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1382 | 0.1604 | +0.0222 | 0.4293 | 0.7810 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=88900, base_rate=0.2007

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5100 | 0.2007 | 116 | 51 | 100 | 16 / 116 / 116 |
| 5 | 0.6160 | 0.2007 | 517 | 308 | 500 | 16 / 100 / 116 |
| 10 | 0.6260 | 0.2007 | 1017 | 626 | 1000 | 16 / 100 / 116 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2007 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.2007 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.2007 | 10 | 10 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 93 | 13 | 0.1398 |
| NASDAQ:TSLA | 63 | 51 | 0.8095 |
| NYSE:ASTS | 62 | 35 | 0.5645 |
| NASDAQ:MRVL | 42 | 42 | 1.0000 |
| NYSE:CAR | 42 | 11 | 0.2619 |
| NASDAQ:MU | 32 | 32 | 1.0000 |
| NYSE:CLF | 31 | 25 | 0.8065 |
| NASDAQ:INTC | 21 | 21 | 1.0000 |
| NASDAQ:MDB | 21 | 21 | 1.0000 |
| NYSE:CVNA | 20 | 8 | 0.4000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ELF | 9 | 0 | 0.0000 |
| NASDAQ:TTD | 7 | 0 | 0.0000 |
| NASDAQ:MSTR | 93 | 13 | 0.1398 |
| NYSE:CAR | 42 | 11 | 0.2619 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 43 | 0.5584 | 0.2007 | 2.783 |
| 2025Q2 | 310 | 221 | 0.7129 | 0.2007 | 3.552 |
| 2025Q3 | 130 | 44 | 0.3385 | 0.2007 | 1.687 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 88900 | 0.0459 | 0.5143 | 0.1855 | 0.1049 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=8.10); shipped as `isotonic`. Brier vs base-rate: +0.0222 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
