# _258 — Macro-augmented sp500 cells (F17 FRED macro): proxy A/B

**Headline:** Adding a macro-regime feature family (F17) to the two sp500 champion
cells produces a **real, model-used top-K signal** in a clean A/B — but on
**Yahoo proxy data (4 of 9 daily series)** and against a **matched untuned
baseline**, NOT a head-to-head vs the deployed champions. Net-positive on the
headline @1/@10 metrics on both cells; encouraging enough to justify the
definitive real-FRED re-run. **The "beat the champion" objective is not yet
demonstrated** (see Verdict).

## Setup

- **Hypothesis:** macro-regime context (yield level/curve, short-rate dynamics,
  vol, USD) conditions whether a +20%/+50% breakout sustains, so F17 should lift
  top-K skill.
- **F17 (new, opt-in):** daily FRED series broadcast to every (date, ticker) row
  — `<id>_level`, `chg_{20,60}`, `z_{60,120}` — 1-trading-day lag (causal C1,
  proven by a macro-perturbation leakage test). Opt-in via the `all_macro` token;
  `"all"` stays byte-identical F1–F16, so existing models are untouched.
- **⚠️ Proxy data:** FRED egress (stlouisfed.org) was unavailable this session, so
  the macro panel uses **Yahoo proxies** cached under the FRED ids (provider
  `yahoo_proxy`): `DGS10←^TNX`, `DGS3MO←^IRX`, `VIXCLS←^VIX`, `DTWEXBGS←DX-Y.NYB`
  (daily, 2017→2026). The 5 FRED-only series (`T10Y2Y, DFF, BAMLH0A0HYM2,
  T10YIE, DFII10` — curve slope, fed funds, HY credit OAS, breakeven, real yield)
  have no clean free proxy and are skipped. So F17 here is **20 columns from 4
  series**, not the full 9.
- **A/B design:** `base_v2` (candidates `all`, F1–F16) vs `macroproxy` (candidates
  `all_macro`). **Identical** otherwise: xgboost, `conditional_isotonic`,
  `split.mode: date_aligned`, `train_start 2019-01-01`, `--snapshot-end 2026-06-20`,
  **`max_iterations: 1`** (iter_0 only — same HP, no FS-divergence, so the *only*
  difference is the 20 macro columns). This isolates the macro contribution; it is
  deliberately NOT the champion config.
- All four arms scored on identical windows: train 2019-01-02→2022-03-04, val
  →2023-10-06, eval →2024-07-25, **test 2024-07-26→2024-12-16** (486 tickers,
  48,600 rows). Sweep wall-clock ≈ 20 min total (cold sp500 build ≈ 7 min ea.).

## Results (test segment; raw R-Precision@K + base rate, per reporting convention)

### sp500 +20%/25d (dd10%) — base_rate 0.0402, Q=100 days

| K | base_v2 | macroproxy |
|---|---|---|
| R-Precision@1 | 0.1000 | 0.1000 |
| R-Precision@3 | 0.1367 | 0.1633 |
| R-Precision@5 | 0.1660 | 0.1980 |
| R-Precision@10 | 0.1879 | 0.2157 |
| R-Precision@20 | 0.2570 | 0.2358 |
| eval Brier | 0.04675 | 0.04661 |
| eval AUC | 0.7912 | 0.7986 |

Macro lifts the mid band — @3/@5/@10 up ~+15–19% — is exactly tied at @1, and is
~8% worse at @20.

### sp500 +50%/50d (dd25%) — base_rate 0.0097, Q=91 days

| K | base_v2 | macroproxy |
|---|---|---|
| R-Precision@1 | 0.1099 | 0.1319 |
| R-Precision@3 | 0.1062 | 0.0916 |
| R-Precision@5 | 0.1454 | 0.1176 |
| R-Precision@10 | 0.2182 | 0.2736 |
| R-Precision@20 | 0.3912 | 0.4234 |
| eval Brier | 0.00927 | 0.00948 |
| eval AUC | 0.8654 | 0.8676 |

Macro lifts the top pick (@1 ~+20%) and the deeper band (@10 ~+25%, @20 ~+8%),
but dips at @3/@5. (Only 91 days-with-positives + many low-R days → @1/@3/@5 are
small-sample; @10/@20 are the stable readings here.)

## F17 feature usage — the model genuinely uses macro

- **sp500_20:** all 20 macro features nonzero; macro = **10.75%** of total
  importance. `macro_DGS3MO_chg_60` is **rank #4 overall** (3.73%);
  `macro_VIXCLS_z_120` #8.
- **sp500_50:** 18/20 nonzero; macro = **5.99%**. `macro_DGS10_level` #16,
  `macro_VIXCLS_chg_60` #20.
- Most-used: short-rate dynamics (DGS3MO/DGS10) + VIX z-scores. The dollar index
  (DTWEXBGS) ranks lowest — the weakest of the four proxies.

Mechanistically consistent with the hypothesis: the model leans on **rate-regime
momentum and the vol regime**, not the level of the dollar.

## Verdict

1. **Macro features add real, model-used top-K signal** in a controlled A/B — net
   positive on the two headline metrics (@1 and @10) on both cells (sp500_20 @1 is
   a tie). This holds even with only 4/9 proxy series. The necessary precursor
   ("does macro help?") is **yes**.
2. **This is NOT a win over the deployed champions.** The committed champions
   (`sp500_up_50pct_50d_dd25pct_agentloop` R-p@1 0.640; `..._20pct_25d_..._agentloop`
   R-p@1 0.413) were scored on a **different, later test window** (2025-12→2026,
   trailing split, base_rate 0.026/0.089) and were **agent-tuned (5 iters)**. My
   `max_iter:1` date-aligned base_v2 (R-p@1 0.110/0.100, base_rate 0.010/0.040) is
   a much weaker, different-window baseline — the absolute numbers are not
   comparable. The objective "outperform the existing models" is **not yet met**;
   the A/B only shows macro is additive.

## Next steps (to actually beat the champion)

- **Definitive real-FRED run** (gated on egress recovery): seed the full 9 daily
  series, re-run the committed `..._macro` specs (full macro panel incl. credit
  OAS / breakevens / real yield / fed funds / curve slope). Those FRED-only
  series are plausibly the highest-signal regime indicators and are absent here.
- **Champion head-to-head:** run the macro variant under the *champion config*
  (trailing split + `agent_file_protocol` 5-iter tuning, `plateau_threshold 0.0001`)
  and compare to the committed champion on the same window. This is the Phase-3
  "beat-the-champion" run — deferred until real FRED so it's done once, definitively.
- Promote a macro model to the deployed champion / `/daily-predictions` **only if**
  it beats the committed champion on its own window.

## Artifacts

- Specs: `configs/gbdt/experiments/sp500_up_{20pct_25d_dd10pct,50pct_50d_dd25pct}_{base_v2,macroproxy}.yaml`
- Registry rows appended to `results/gbdt/data/r_precision_at_k.csv` (4 `_base_v2`/`_macroproxy` rows).
- Data sidecar: `results/gbdt/data/_258_macro_proxy_ab_data.json`.
- Run artifacts (untracked, regenerable): `results/gbdt/experiments/sp500_up_*_{base_v2,macroproxy}/`.
