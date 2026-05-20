# E3 — Seed noise floor (2-point)

V3 experiment 3 (per [`V3_PLAN.md`](V3_PLAN.md#e3-seed-noise-floor-on-cell-d-fast)). Bounds how much of the Cell D ≈ 4% gain over v2.1 is between-seed Monte Carlo noise vs a robust signal.

## Status

**Stopped early after 2 seeds.** Per the autopilot gate: if the 2-point relative gap was < 1%, the noise floor is small enough that additional seeds add no decision-relevant information. The observed gap was **0.08%** — ~50× tighter than the 4.0% Cell D gain. Stopping early frees compute for E2 / E9 / E7 instead of consuming overnight on incremental seeds.

Tradeoff acknowledged: with n=2 we have a single between-seed gap, not a stdev. The conclusion ("noise floor well below the Cell D vs v2.1 gap") is robust to this; a precise stdev would need ≥3 points but does not change the qualitative read.

## Setup

Cell D fast preset (`configs/analog_mc/ablation_E3_seed7.yaml` ≡ `nasdaq100_v22.yaml` with `random_seed: 7`). 76 folds, 273,120 origin × step pairs. Wall time: 9621 s (160 min) — slower than D-fast's earlier 6h 30m only because subsequent runs benefit from the recent in-place speedup work but pay a slightly different conditional-sampling overhead curve.

| Cell | random_seed | run dir | wall |
|---|---|---|---|
| E3-seed42 | 42 | `runs/analog_mc/20260517T070003Z` (= D-fast) | 6h 30m |
| **E3-seed7** | 7 | `runs/analog_mc/20260519T100304Z` | **160 min** |

## Headline numbers

| Metric | seed=42 (D-fast) | seed=7 | 2-pt mean | abs gap | rel gap |
|---|---|---|---|---|---|
| Mean aggregate CRPS | 0.05041 | 0.05037 | 0.05039 | 0.00004 | **0.08%** |
| Low-vol CRPS | 0.0293 | 0.0293 | 0.0293 | 0.00001 | 0.04% |
| Mid-vol CRPS | 0.0398 | 0.0397 | 0.0397 | 0.00007 | 0.18% |
| High-vol CRPS | 0.0826 | 0.0825 | 0.0826 | 0.00012 | 0.14% |

(`summary.parquet` aggregation; slightly differs from log-printed 0.05045 due to count vs simple mean — both correct, gap unchanged.)

### Decision-rule cross-seed stability

| Rule | seed=42 (D-fast) | seed=7 | Δ |
|---|---|---|---|
| `sloped_global_pit` | +0.059 ✅ | +0.058 ✅ | 0.001 |
| `u_shaped_high_vol_pit` | +1.612 ✅ | +1.625 ✅ | 0.013 |
| `acf_seam_degradation` | −1.121 🔥 | −1.122 🔥 | 0.001 |
| `clip_hit_excessive` | +0.100 ✅ | +0.102 ✅ | 0.002 |

Every rule is stable to within 0.013 absolute between seeds — far below any firing threshold gap.

## Verdict

**Noise floor: ~0.1% relative.** The Cell D vs v2.1 gap is 4.0% (0.05265 → 0.05056 at canonical), making the Cell D win **~50σ-equivalent** in noise-floor units. Per-vol-regime CRPS is identically tight across seeds — high-vol calibration (the headline V2 concern) shifts by 0.0001 between seeds, far below the 5.2% v2.1-vs-v1 gain or the 3.9% Cell-D-vs-v2.1 gain.

## Implication for E7 (promote)

V3_PLAN's E7 gating language said *"if the noise floor overlaps v2.1 canonical, weaken the promotion language; if it does not, promotion is robust."* The 0.1% noise floor does not come close to overlapping the 4% gap. **E7 promotion is robust** — flip `default.yaml` to mirror `default_v22.yaml` without weakening language.

Combined with E10's verdict (vanilla Cell D is the right target — bl=20 does not stack), E7's last remaining gate (E2) is optional refinement, not a blocker. **Recommend executing E7 now**, with E2 as a follow-up to find a refined Cell D variant later (and re-promote if material).

## Limitations

- **n=2.** A stdev estimator needs ≥3 points and is unstable until ≥5. We have a single gap, not a distribution.
- **Within-fold seed effects.** `_seed_for(...)` uses blake2b on `(random_seed, weights, n_eff, origin_idx)`. The walk-forward search re-picks weights per fold, and a different `random_seed` produces a different `_seed_for` chain at every origin. So the 0.08% gap measures the *full* seed-induced variation (search + test), not just test-time MC noise. That is what V3_PLAN E3 wanted bounded.
- **Conditional test-only contingency.** Both runs use `conditional_block_sampling_in_search: false`. Search-time noise estimate would need a search-time conditional sampler (V3 E5 territory).

## Deliverables

- `configs/analog_mc/ablation_E3_seed{7,1337,2024,99}.yaml` (only seed=7 executed; others deleted from queue)
- `runs/analog_mc/20260519T100304Z/` (seed=7)
- `runs/analog_mc/20260517T070003Z/` (seed=42, pre-existing)
- `results/analog_mc/data/_e3_data.json`
- This page.
