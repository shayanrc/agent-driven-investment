# _289 — nifty500 canonical scan + finetune (de-confounded F18-IN)

**Plan:** `docs/gbdt/V1.10_nifty500_canonical_scan_plan.md` · **Task:** #55 · **Branch:** `nifty500-canonical-scan`
**Supersedes** the INVALIDATED `_285` (train_start 2015 with F18-IN ~48–58% NaN).

**One-liner:** After a screener.in pre-2019 backfill de-confounds F18-IN (train-window NaN
48–58% → **17%**), the nifty500 base-vs-fund lattice on canonical periods shows F18-IN is a
**broad, robust positive** — 7/20 cells beat technical at every K (vs _285's single-horizon
claim). The strongest fund cell (**+50%/200d ffund**) survives a finetune on the held-out
**test** book AND replicates on the independent **backtest** window (+18.3% strategy /
+42.3% raw top-K vs +6.2% EW basket). The technical champion (+30%/50d) does not finetune
and lags on backtest. **F18-IN on nifty500 is the first genuinely positive fundamentals
result in this project's F17/F18 arc** — but single-test-window, not promoted (no nifty500
deploy path).

## Phase 0 — de-confound F18-IN (the reason _285 was invalid)
- Built the 2019+ NSE valuation panel (`valuation_panel_nse.parquet`, 688k rows, 482 tickers) —
  it did not previously exist.
- **screener.in annual backfill** (→FY2015, 500/500 tickers, 5,217 rows) → concat pre-2019
  daily panel → extended panel 964,779 rows (2014-06 → 2026-07), 339 tickers back-extended.
- **Gate:** canonical train-window (2015–2022) F18-IN NaN **0.006 at the panel level / 17% at
  the feature-matrix level** (was ~48–58%). Residual seams: 2015 first-year ramp (47%), 2019
  screener→XBRL handoff (39%). See `[[project-in-fundamentals-coverage-cliff]]`.

## Phase 1 — scan (40 cells, single-fit, test window 2024-07→2025-06)
Matched base(all_calendar2)-vs-fund(all_fundamentals_calendar2) A/B, identical HP.

Fund−base delta by horizon (mean across thresholds): **25d + 100d positive at every K; 200d
strong @3–@20; 50d the top-1 anti-pattern.** 7 cells beat at every K (10%/10d, 10%/25d,
30%/25d, 20%/50d, 10%/100d, 20%/100d, **50%/200d** +0.341 @1). Materially broader than _285's
"100d only" — de-confounding matters. Registry: `results/gbdt/data/r_precision_at_k.csv`
(`nifty500_up_*_{fbase,ffund}_canon`) + `_nifty500_canon_base_vs_fund.csv`.

## Phase 2 — finetune (canonical recipe: select on val, confirm on test)
Two user-chosen cells. Both show eval↔test prevalence inversion → selected on val.

| cell | prev | controlled baseline test @1/3/5/10/20 | verdict |
|---|--:|---|---|
| **+50%/200d ffund** | 0.121 | 0.414 / 0.340 / 0.293 / 0.246 / 0.230 | **ADOPT d10/ss0.7/cs1.0** |
| +30%/50d fbase | 0.072 | 0.242 / 0.203 / 0.190 / 0.187 / 0.233 (AUC 0.663) | baseline stands |

- **ffund adopted** `d10 mcw1 ss0.7 cs1.0`: test 0.406 / **0.347 / 0.329 / 0.289 / 0.260** —
  beats baseline book @3–@20 (+0.007/+0.036/+0.043/+0.030), @1 flat (−0.008). Cleaner than
  #53 (same 50%/200d shape lost @1).
- **fbase baseline stands**: deep+bagging collapsed @1 (0.242→0.161) for noise-level @3/@5
  gains — val lift did not transfer. The #49/#51/#54 outcome.

## Backtest (2025-07-01 → 2026-07-13, untouched window, `--sizing-mode equal`)
| cell | strategy | maxDD | EW basket | raw top-K (no-Kelly) | exits target/DD |
|---|--:|--:|--:|--:|--:|
| **+50%/200d ffund** | **+18.3%** | −10.9% | +6.2% | **+42.3%** | 31 / 49 |
| +30%/50d fbase | −2.5% | −19.2% | +6.2% | +0.0% | 37 / 72 |

ffund beats its universe by +12pp with lower drawdown; the raw top-K signal is +42.3%. The
strict deploy cut (target-hits > DD-stops) is NOT met (31 vs 49) and the 200d room caveat
truncates late entries — so this is a strong signal read, not a deploy trigger. fbase lags
the basket. Benchmark in the harness is `^NDX` (US reference; no NIFTY index cached).

## Caveats / not-done
- **Single test window** — the F17/F18 history is single-window fund wins that failed
  replication (sp500 F18 _279/_280, macro _264, _285). The backtest replication is
  encouraging but is one forward window; a second independent test window is the real bar.
- **NOT promoted.** nifty500 has no `/daily-predictions` deploy path today; parity/research.
- Follow-ups (`V1.10_TBD`): second-window replication of the +50%/200d ffund win; the broader
  25d/100d every-K fund cells (unfinetuned); a NIFTY index benchmark in run_fresh_oos.
