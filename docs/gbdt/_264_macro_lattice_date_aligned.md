# _264 — Date-aligned macro lattice: the macro edge does NOT replicate (validation fails)

**Headline:** Re-running the `_263` macro lattice on an **independent window**
(date-aligned split, test 2024-07-26 → 2024-12-16) is the second-window validation
of F17 — and it **fails**. Of the 12 cells testable in both windows, **only 4/12
agree in sign at R-Precision@1; 8/12 flip.** The biggest trailing winners flip
hardest (`+20%/50d` +0.30 → **−0.11**; `+50%/50d` +0.14 → **0.00**; `+20%/25d`
+0.09 → −0.04). The net macro effect on the date-aligned window is **~neutral**
(@1 mean +0.004, 5 help / 8 hurt). **Macro is not a robust, deployable edge** — the
`_262`/`_263` win was window-specific.

The mechanical goal succeeded: date-aligned **closes the long-horizon gap** — all 17
cells test (Q=100 each), including the 5 (100d/200d) cells that were Q=0 under the
trailing snapshot.

Heatmap: `results/gbdt/data/_264_macro_lattice_heatmap.png` (compare to `_263`'s).

## Setup

Identical to `_263` **except the split**: matched xgboost `mcw=10` single fit, base
(`all`) vs `+macro` (`all_macro`), only `features.candidates` differs. `split.mode:
date_aligned`, `train_start: 2019-01-01`, snapshot 2026-06-20. So `_263` (trailing,
test ≈ 2026-Q1) vs `_264` (date-aligned, test = 2024-H2) is a clean two-window read of
the same matched A/B. Tooling extended: `gen_macro_sweep_specs.py` (`date_aligned` →
`dasw` arm prefix), `aggregate_macro_sweep.py` (`dasw _264`). 34 registry rows added.

## Cross-window replication — Δ R-Precision@1 (macro − base)

| cell | trailing `_263` | date-aligned `_264` | |
|---|---|---|---|
| +10%/5d | +0.021 | +0.051 | agree |
| +10%/10d | +0.122 | +0.070 | agree |
| +10%/25d | +0.080 | +0.050 | agree |
| +10%/50d | −0.100 | +0.090 | **flip** |
| +20%/5d | +0.060 | −0.016 | **flip** |
| +20%/10d | +0.146 | −0.033 | **flip** |
| +20%/25d | +0.093 | −0.040 | **flip** |
| +20%/50d | **+0.300** | **−0.110** | **flip** |
| +40%/25d | −0.160 | −0.069 | agree |
| +40%/50d | +0.040 | +0.010 | flip (→neutral) |
| +50%/25d | +0.113 | +0.000 | **flip** |
| +50%/50d | **+0.140** | **+0.000** | **flip** |

**4/12 agree, 8/12 flip.** A robust edge would mostly agree; this is the signature of
a window-specific effect. The only consistent positive is the **low-threshold +10%**
family (3 of 4 agree positive) — everything at +20%/+50% is unstable across windows.

## Date-aligned summary (all 17 cells, now testable)

| K | mean Δ | helps | hurts | neutral |
|---|---|---|---|---|
| @1 | +0.004 | 5 | 8 | 4 |
| @5 | +0.006 | 7 | 5 | 5 |
| @10 | +0.009 | 8 | 5 | 4 |

Essentially noise around zero — no systematic top-of-book benefit, in sharp contrast
to `_263`'s clean @1 help (10/12, mean +0.07). The long-horizon cells (newly testable)
are mixed: `+40%/100d` +0.22 (the one big macro win here) but `+40%/200d` −0.06,
`+50%/200d` −0.08.

## Verdict

- **Macro fails second-window validation.** The `_262`/`_263` top-of-book edge does
  not replicate on the date-aligned window; 8/12 sign flips and a ~zero net effect.
  This **reaffirms the `_258`–`_261` conclusion** ("contextually additive but not a
  robust edge") with a clean two-window view, and walks back the `_262`/`_263`
  optimism — which was favorable to a single (trailing 2026-Q1) window.
- **Do NOT wire macro into `/daily-predictions`.** The objective "a macro model that
  robustly outperforms" is **not met**.
- The two windows differ in regime and base rate, so cell-for-cell deltas aren't
  perfectly comparable — but that is precisely the point: a deployable always-on
  feature must help out-of-sample regardless of regime, and macro does not.
- F17 stays a **merged, opt-in** feature (it helps in *some* regimes, notably
  low-threshold +10% cells); it is not promoted. The validation machinery did its job —
  it caught a window-specific effect before deployment.

## Artifacts

- Heatmap: `results/gbdt/data/_264_macro_lattice_heatmap.{png,svg}`
- Data: `results/gbdt/data/_264_macro_lattice_data.json` (+ `_raw.json`, cross-window deltas)
- Registry: 34 `*_dasw{base,macro}` rows in `results/gbdt/data/r_precision_at_k.csv`
- Tooling: `gen_macro_sweep_specs.py` + `aggregate_macro_sweep.py` (date-aligned support)
