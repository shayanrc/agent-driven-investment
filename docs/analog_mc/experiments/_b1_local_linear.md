# B1 — Platzer–Yiou local-linear correction

## Status

**Closed (did not promote).** Canonical run complete; v2.4 Cell-D-s30 remains the default. See [`V4_RESULTS.md`](../V4_RESULTS.md) for the full v4 synthesis.

## Setup

- Spec: [`_b1_design.md`](_b1_design.md) — decisions D1–D10.
- Config: `configs/analog_mc/ablation_B1_localreg.yaml` (v2.4 baseline + `local_linear_correction: true`).
- Implementation: `src/analog_mc/local_linear.py` (WLS fit, scale-aware Tikhonov, extrapolation clamp, NaN-forward drop, `LocalLinearDiagnostics` dataclass). Hooks into `simulate.forecast()` immediately after probability resolution; correction enters via `drift_target += correction / forecast_horizon`.
- Tests: `tests/analog_mc/test_local_linear.py` — 10 tests, all pass. Bit-identical to v2.4 when knob is off.
- Canonical run: `runs/analog_mc/20260520T155220Z` — 76 folds, 1000 paths, 66×5 weight grid, ~11.8h compute (2026-05-20 21:22 → 2026-05-21 09:08).
- Sanity precursor: `scripts/analog_mc/v4_b1_sanity.py`, results at `results/analog_mc/data/v4_b1_sanity.json`, write-up at [`_b1_sanity_v0.md`](_b1_sanity_v0.md). Predicted −6.7% failure CRPS / 4-of-5 sign-agreement isolated effect; canonical confirmed the broad shape.

## Headline numbers

| Metric | v2.4 baseline | B1 canonical | Δ |
|---|---:|---:|---:|
| Mean walk-forward test_crps (4500 origins) | 0.04755 | 0.05021 | +5.6% |
| 15-anchor mean CRPS | 0.06111 | 0.05705 | **−6.6%** ✅ |
| 5 V3.5 failure-anchor mean CRPS | 0.09471 | 0.08841 | **−6.6%** ✅ |
| 5 control-anchor mean CRPS | 0.02051 | 0.01859 | **−9.3%** ✅ |
| V3.5 failures recovered (90-band ≥45/60) | 0/5 | 1/5 | +1 (2001-10-02) |
| Anchors regressing CRPS >5% | — | 5/15 | over the ≤2 bar ❌ |

Per-anchor failure detail in [`_b1_local_linear_fat_tail.md`](_b1_local_linear_fat_tail.md). Per-fold test_crps comparison is in `logs/b1_canonical_20260520T212218Z.log`.

## Mechanistic reading

B1 added a per-day uniform drift correction `correction / H` after the C3 vol-scaling, computed once per origin from a locally-weighted regression of analog forward-cum-logret on analog z-scores. Decisions D4 + D6: scalar terminal-cum-logret target, frozen at block 0.

The canonical search re-tuned `(weights, n_eff)` under B1 active (decision D10). Per-fold weight choices were materially different from v2.4 (e.g. fold 2: v2.4 had `[0.408, 0.001, 0.591]`; B1 picked `[0.00, 1.00, 0.00]`). This is the search-time effect — B1's drift correction makes a different `(weights, n_eff)` configuration optimal on the val set.

**Where B1 wins.** Mostly small-but-broad: 8 of the first 21 folds improved by 0.5–8% test_crps; the V3.5 failure anchors 24 (2001-10-02 V) and 42 (2010-04-23 crash) improved by 14% and 13% respectively. The correction acts exactly as theory predicts at high-Lyapunov regimes — small magnitude, large positive impact when the matcher's mean is biased.

**Where B1 loses.** 13 of the first 21 folds got worse; some by a lot (fold 1: +42%, fold 3: +61%, fold 11: +85%). These are folds where the matcher was already well-calibrated and the regression's bias estimate was wrong about which direction to drift. The most striking single loss is **1990-09-24 control**: 90-band coverage collapsed from 55 → 11 days. Mechanism not fully understood — worth a per-fold β-inspection in v5.

COVID 2020-03-16 was a 0/5 from the start: B1's correction was +0.7% per day, but the matcher under-estimated by 37% terminally. No drift correction can close that gap.

## Decision-rule verdict

Against the V4 plan §B1 decision rules:
- **`acf_seam_degradation` closure**: not directly assessed at this stage; the headline question was the failure-anchor signal.
- **High-vol regime CRPS gain of ≥1-2%**: confirmed (failure mean CRPS −6.6%).
- **Promotion to v2.5 (≥3/5 failures recovered + ≤2 anchors regressing)**: **fails** — only 1/5 recovered and 5/15 regressed.

Against the V4 mandatory fat-tail criterion: **does not pass**. Solo B1 is not promotable.

## Implication for v5 roadmap

B1 is *real* — the only experiment with both failure AND control aggregate improvement, and the only one where the failure-anchor coverage wins (2001-10-02 90-band 44→55) don't come paired with catastrophic regressions elsewhere. Two carry-overs:

1. **1990-09-24 mechanism investigation.** Why did the local-linear regression destabilize there? Inspect β at that origin and compare against the analog cluster's forward distribution. Possibly a small-cluster sensitivity issue.
2. **B1 may be useful as a stabilizer for A2.1-class experiments** — but the v4 B5 (joint) result showed B1 did NOT effectively stabilize A2.1's tight bands. The interaction is more complex; β estimated under corrwindow's analog set produced unhelpful drifts. A v5 design choice: should B1 use the *original* matcher's analog set (v2.4-style probabilities) rather than the new distance's?

## Deliverables

- `src/analog_mc/local_linear.py`, `src/analog_mc/simulate.py` (B1 hook), `src/analog_mc/config.py` (`local_linear_correction` knob)
- `configs/analog_mc/ablation_B1_localreg.yaml`
- `tests/analog_mc/test_local_linear.py` (10 tests)
- `scripts/analog_mc/v4_b1_sanity.py` + `_b1_sanity_v0.md` + `v4_b1_sanity.json`
- `runs/analog_mc/20260520T155220Z/` (canonical artefacts)
- `results/analog_mc/data/fat_tail_b1_local_linear.json` + `_diff.json`
- `docs/analog_mc/experiments/figs/b1_local_linear_fat_tail/` (15-anchor panel)
- This narrative + auto-generated [`_b1_local_linear_fat_tail.md`](_b1_local_linear_fat_tail.md)
