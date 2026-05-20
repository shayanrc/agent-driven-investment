# analog_mc v3 — experiment plan

Companion to [`RESULTS.md`](RESULTS.md), [`ABLATION_STUDIES_REPORT.md`](ABLATION_STUDIES_REPORT.md), and [`V2_PLAN.md`](V2_PLAN.md). This file is the **spec** for the next round of experiments — what to run, what each cell answers, and what acceptance looks like. Results live in a future `V3_RESULTS.md` (not created yet).

Like the prior plan docs: every decision below was made for a stated reason. Surface deviations rather than silently change.

---

## Starting state (as of 2026-05-19, late)

| Item | State (2026-05-20, end of v3 fast-preset phase) |
|---|---|
| Current default | **v2.3 — Cell D** (shipped 2026-05-19 via E7 flip; drift+cond, shrinkage=0.50, bl=10) |
| **v2.4 promotion candidate** | **Cell-D-s30** — drift+cond with `momentum_shrinkage=0.30`. Fast preset: 0.04765 mean CRPS (**−5.5% vs v2.3 default, −9.5% vs v2.1**), PIT slope +0.0958 (passes +0.10 threshold). Canonical confirmation pending. See [`_e2_momentum.md`](experiments/_e2_momentum.md). |
| **E1 (block-length sweep)** | ✅ done — [`_e1_block_length.md`](experiments/_e1_block_length.md). ACF flat across bl ∈ {5, 10, 20}; block geometry is not the ACF lever. |
| **E10 (Cell D × bl=20)** | ✅ done — [`_e10_celld_bl20.md`](experiments/_e10_celld_bl20.md). bl=20 doesn't stack with Cell D. |
| **E3 (seed noise floor)** | ✅ done (2-pt, early stop) — [`_e3_seed_noise.md`](experiments/_e3_seed_noise.md). 0.08% noise floor. |
| **E11 (Cell B × bl=20)** | ✅ done — [`_e11_cellB_bl20.md`](experiments/_e11_cellB_bl20.md). bl=20 stacks with drift but loses to Cell D by 2.7%; doesn't displace conditional sampling. |
| **E2 + extension (momentum sweep)** | ✅ done — [`_e2_momentum.md`](experiments/_e2_momentum.md). 5+3 cell Pareto sweep. **s=0.30 found as Pareto sweet-spot** at Cell D config. |
| **E9-A (GARCH-conditional, zero drift)** | ✅ done — [`_e9_v3b.md`](experiments/_e9_v3b.md). GARCH gives +2.2% mean CRPS but **does NOT close the ACF rule**. Lag-1 ACF nudges from −0.004 to +0.016 vs realized +0.27. **`acf_seam_degradation` structurally unfixable within the analog-block primitive.** E9-D skipped (would inherit same ACF ceiling). v3b ships as opt-in `vol_model: "garch"`. |
| Decision rules that still fire | `acf_seam_degradation` everywhere (now confirmed structural — v4 scope) |
| Decision rules creeping toward firing | `u_shaped_high_vol_pit` (cleanest at Cell-D-sNN variants: +1.50 to +1.65; threshold +2.50) |
| Open V2_PLAN carryovers | ~~v3c (E1)~~, ~~v3b (E9)~~, ~~v3a~~ (held — same ceiling), tail-inflator re-check, multi-asset robustness |
| Open questions from ablations | ~~momentum sweep~~ (E2 done), ~~Monte Carlo noise floor~~ (E3 done) |

## v3 scope and non-scope

**In scope:** experiments that either (a) decide promotion of Cell D as default, or (b) move the residual `acf_seam_degradation` failure, or (c) keep `u_shaped_high_vol_pit` from firing on new datasets.

**Out of scope:** transaction-cost / PnL modeling, downstream allocation logic, deep-learning replacements for the matcher, multi-asset *joint* forecasting. All four are downstream concerns or v4+ rewrites.

---

## Experiment inventory

| ID | Title | Tier | Status | Motivation | Wall time |
|---|---|---|---|---|---|
| E1 | Block-length sweep | 1 | ✅ done | v3c carryover — decisive diagnostic for the ACF ceiling | ~90 min |
| E2 | Momentum shrinkage sweep (5-cell partial, Cell B) | 1 | pending (compressed post-E3) | Ablation S1/S2 — recover CRPS that drift gives up; noise floor 0.08% makes small gains detectable | ~2.5 h |
| E3 | Seed-noise floor | 1 | ✅ done (2-pt, early stop) | Phase 4 — bound the 4% Cell D gain against MC noise | 160 min |
| **E11** | **Cell B × bl=20 (drift, no conditional)** | **1** | **new (post-E10)** | E10 showed bl=20 ⊥ Cell D. E11 tests whether bl=20 stacks with *drift* alone — if yes, Cell B+bl=20 may be a simpler default than Cell D. | **~1 h** |
| E10 | Cell D × bl=20 | 1 | ✅ done | E1 surprise: longer blocks → lower CRPS. Did NOT stack with Cell D. | 84 min |
| E9 | v3b — GARCH-conditional resampling | 2 | promoted (post-E1) | E1 falsified block-geometry path to closing ACF rule — σ-scaling is the lever. | **~5 h end-to-end** (2 h code + 3 h compute, see E9 section) |
| E6 | Vol-level feature in distance | 2 | pending | Tighten high-vol matcher — addresses creeping U-shape | ~1 day impl + ~1 h run |
| E5 | Search-time conditional sampling | 2 | downgraded (post-E10) | Mechanism overlap E10 surfaced reduces motivation; defer until E9 lands | ~1 day impl + ~3 h run |
| E7 | Promote vanilla Cell D as default | 3 | unblocked (E3 + E10) | All gates passed; flip default.yaml | ~30 min flip + commit |
| E8 | Multi-asset robustness | 3 | pending | V2_PLAN carryover-3 — rule re-firing on different assets | ~3× fast preset |
| E4 | v3a — per-step σ injection | 3 | held | E1 + E10 weakened the design hypothesis: per-step σ inside any block is a third variance-leak fix duplicative with conditional sampling and bl=20. Revive only if E9 fails in a way that motivates it. | ~1 day impl + 1 h run |

Tier-4 speedups (`max_iter=14`, float32 flag, CuPy port) are deferred to [`SPEEDUP_PLAN.md`](SPEEDUP_PLAN.md); they only matter if compute becomes binding again.

---

## Tier 1 — cheap diagnostics that reshape the v3 roadmap

### E1. Block-length sweep (v3c) — ✅ done

**Status (2026-05-19).** All three cells ran fast preset (76 folds each). Full results in [`_e1_block_length.md`](experiments/_e1_block_length.md); summary:

| | bl=5 | bl=10 | bl=20 |
|---|---|---|---|
| Mean CRPS | 0.05438 | 0.05215 | **0.05063** |
| Lag-1 sim ACF | −0.002 | −0.004 | −0.004 |
| `acf_seam_degradation` | −1.056 🔥 | −1.053 🔥 | −1.115 🔥 |

**Verdict: outcome 2.** Simulated ACF is flat across all block lengths — block geometry is not the lever. **v3b (E9) jumps the queue ahead of v3a (E4).**

**Bonus finding.** Mean CRPS improves monotonically with longer blocks (−2.9% bl=20 vs bl=10). Mechanism unconfirmed; motivates **E10 (Cell D × bl=20)**. This was not anticipated by V3_PLAN; treat as a serendipitous finding to capitalise on, not a blocker for the v3b direction.

---

### E10. Cell D × bl=20 (post-E1 follow-up)

**Hypothesis.** E1 showed mean CRPS improves monotonically with longer blocks at the v1 baseline (zero drift, no conditional). If the effect stacks with Cell D (drift + conditional sampling) — i.e. is mechanistically independent — then bl=20 × Cell D should beat the current Cell D fast (0.05045) by a similar ~3% margin.

**Cells.** Single new run, fast preset, Cell D config with `block_length=20, n_blocks=3`.

| Cell | block_length | n_blocks | drift | conditional | preset |
|---|---|---|---|---|---|
| E10 | 20 | 3 | trailing_momentum | true | fast (21×2, n_paths=500) |

Compare to:
- D-fast (`runs/analog_mc/20260517T070003Z`): mean CRPS 0.05045, high-vol 0.0826
- E1-bl20 (`runs/analog_mc/20260519T071520Z`): mean CRPS 0.05063, zero drift

**What it answers.**

- Does the bl=20 CRPS gain stack with the Cell D mechanism, or are they overlapping?
- If E10 ≤ 0.0490 (additive, ~3% below D-fast): run E10 at canonical resolution before E7 (promote that variant instead of vanilla Cell D).
- If E10 ≈ D-fast: bl=20 gain and Cell D gain share a mechanism; promotion stays on vanilla Cell D.
- If E10 > D-fast: bl=20 hurts under drift+conditional; document and drop.

**Acceptance / read.** Compare mean CRPS, per-vol-regime CRPS, and the four decision-rule metrics against D-fast. PASS = strict improvement on at least mean CRPS without regressing any rule from passing to firing.

**Cost.** ~1 h fast preset run + diagnostics. Cheapest experiment that can change E7's outcome.

**Deliverables.** `configs/analog_mc/ablation_E10_celld_bl20.yaml`, run dir, `_e10_celld_bl20.md`.

---

### E2. Momentum shrinkage sweep — compressed to 5-cell partial at Cell B

**Background.** Original V3 design called for a 3×5 = 15 cells at Cell D (drift+cond). At ~3 h per Cell-D fast run (test-eval conditional sampling scales with fold count), full grid = ~45 h. With E3's noise floor at 0.08%, that resolution is overkill.

**Compressed design.** 5-cell shrinkage sweep at **Cell B** (drift on, conditional off) — Cell B cells run at v1-fast pace (~30 min) instead of ~3 h. Five cells × 30 min = **2.5 h** total. If a cell beats the current shrinkage=0.5 Cell B baseline by ≥0.5%, re-run that single cell with `conditional_block_sampling: true` to get the Cell D refinement candidate (~3 h). Worst case end-to-end: 5.5 h. Best case: 2.5 h.

**Cells.** Fast preset, Cell B config (`drift_mode: trailing_momentum`, `conditional_block_sampling: false`):

| Cell | `momentum_shrinkage` | `momentum_lookback` |
|---|---|---|
| E2-s00 | 0.00 (≡ Cell A, v1 fast, anchor) | 20 |
| E2-s25 | 0.25 | 20 |
| E2-s50 | 0.50 (current default) | 20 |
| E2-s75 | 0.75 | 20 |
| E2-s100 | 1.00 (full half-Kelly) | 20 |

`momentum_lookback` is held at 20 — the 3-lookback dimension is a secondary sweep, deferred until shrinkage sweep finds a candidate.

**What it answers.**

- Pareto frontier of (mean CRPS) vs (`sloped_global_pit` metric) for the drift component alone, controlling for conditional sampling's separate variance-leak effect.
- Whether a shrinkage value exists with `sloped_global_pit` inside ±0.10 AND mean CRPS closer to v1 fast's 0.05210 than to Cell B's 0.05313.

**Acceptance / read.** Tabulate mean CRPS, `sloped_global_pit`, high-vol CRPS, low-vol CRPS for each shrinkage. Highlight Pareto-optimal. Follow-up cell at Cell D if a winner emerges.

**Cost.** ~30 min per Cell B cell × 5 = ~2.5 h. Plus optional ~3 h Cell D follow-up.

**Deliverables.** `configs/analog_mc/ablation_E2_s{00,25,50,75,100}.yaml`, results table in `_e2_momentum.md`.

---

### E11. Cell B × bl=20 (drift + bl=20, no conditional)

**Hypothesis.** E10 showed bl=20 doesn't stack with Cell D — drift+cond and bl=20 share a variance-leakage mechanism. But what if bl=20 stacks with **drift alone**? Cell B (drift, no conditional) at bl=10 = 0.0531; if bl=20 closes the gap to Cell D-level (0.0504) while preserving PIT correction, then a simpler **Cell B × bl=20** default may make conditional-block-sampling redundant. Conditional sampling adds ~2× test-eval cost and 100+ lines of complexity in `sampling.py`; if E11 lands a simpler config at parity with Cell D, that's a material simplification candidate.

**Cells.** Single new run, fast preset, Cell B config with `block_length=20, n_blocks=3`.

| Cell | block_length | n_blocks | drift | conditional | preset |
|---|---|---|---|---|---|
| E11 | 20 | 3 | trailing_momentum | false | fast (21×2, n_paths=500) |

Compare against:
- B-fast (`runs/analog_mc/20260517T050831Z`): mean CRPS 0.05313, PIT clean
- D-fast (`runs/analog_mc/20260517T070003Z`): mean CRPS 0.05041, PIT clean
- E1-bl20 (`runs/analog_mc/20260519T071520Z`): mean CRPS 0.05063, PIT slope 🔥

**What it answers.**

- Does bl=20 stack with drift alone (when conditional is off)?
- If E11 ≈ 0.0506 with PIT slope passing: drift fully captures bl=20's gain AND the PIT correction, making Cell D's conditional sampling potentially redundant.
- If E11 ≈ B-fast (0.0531): bl=20's gain at zero drift was a noise effect or specifically blocked by drift; Cell D remains optimal.
- If E11 < 0.0506: a new minimum-CRPS configuration; promote E11 over Cell D.

**Acceptance / read.** Compare mean CRPS, per-vol-regime CRPS, all 4 decision-rule metrics against B-fast and D-fast.

PASS criteria for promoting E11 over Cell D:
- Mean CRPS ≤ 0.0510 (within 1% of Cell D)
- `sloped_global_pit` ≤ +0.10 (PIT calibration preserved)
- High-vol CRPS ≤ 0.0835 (within 1% of Cell D's 0.0826)
- No decision-rule regression from passing to firing

If all 4 PASS: rerun E11 at canonical resolution, then revisit E7 with E11 as alternative promotion target.

**Cost.** ~1 h fast preset (no conditional → v1-pace folds).

**Deliverables.** `configs/analog_mc/ablation_E11_cellB_bl20.yaml`, run dir, `_e11_cellB_bl20.md`.

---

### E3. Seed-noise floor on Cell D fast

**Hypothesis.** Phase 4 (Monte Carlo noise floor) was deferred under the heuristic that within-cell noise is small because `_seed_for(...)` uses blake2b on `(random_seed, weights, n_eff, origin_idx)`. Untested. The Cell D vs v2.1 gap is 4% — if between-seed noise is 1-2%, the gap is 2-4σ rather than a robust signal.

**Cells.** Cell D fast preset, 5 independent seeds.

| Cell | random_seed |
|---|---|
| E3-s42 | 42 (existing run `20260517T070003Z`) |
| E3-s7 | 7 |
| E3-s1337 | 1337 |
| E3-s2024 | 2024 |
| E3-s99 | 99 |

**What it answers.**

- Sample stdev of mean CRPS across seeds → noise floor.
- Whether the 4% Cell D gain over v2.1 is many σ or one σ.
- Whether per-fold weight selections are stable across seeds (sanity on the search-time RNG path).

**Acceptance / read.** Report mean ± stdev of mean CRPS, high-vol CRPS, and `sloped_global_pit`. **Promotion decision (E7) is gated on this**: if the noise floor overlaps v2.1 canonical, weaken the promotion language; if it does not, promotion is robust.

**Cost.** 4 new runs × ~1h each = overnight.

**Deliverables.** `_e3_seed_noise.md` with the table + recommendation for E7.

---

## Tier 2 — target the remaining diagnostic failures

### E4. v3a prototype — per-step σ injection

**Hypothesis.** v2.2 conditional re-matching nudged seam lags but didn't touch within-block dynamics, because `scale_block` computes one block-constant `ratio = σ_current / σ_historical_analog`. If the same ratio is computed per *step* — using a causal EWMA on the simulated path so far AND on the historical analog window indexed forward by step — the within-block squared-return ACF should track the analog's σ dynamics rather than collapse to flat.

**Algorithm sketch.** Replace inside `scale_block`:

```python
# v2.x (block-constant ratio)
ratio = clip(sigma_current / sigma_hist_at_analog_origin, lo, hi)
scaled = demeaned * ratio + drift

# v3a (per-step ratio)
for l in range(block_length):
    sigma_path_l = causal_ewma_running(path_so_far, halflife)
    sigma_hist_l = causal_ewma(real_returns, halflife, at=analog_origin + l)
    ratio_l = clip(sigma_path_l / sigma_hist_l, lo, hi)
    scaled[l] = demeaned[l] * ratio_l + drift
```

The per-path running σ buffer that v2.2 already wires up (`sampling.py::generate_paths_conditional`) is the natural carrier — only the inside of the block loop changes.

**New correctness constraint (C11).** The per-step `sigma_hist_l` must be computed causally from real returns at indices `≤ analog_origin + l - 1` only. The same `causal_ewma_vol` function used for `σ_historical_analog` applies; no leakage risk if reused.

**Cells.**

| Cell | drift | conditional | per_step_sigma | preset |
|---|---|---|---|---|
| E4-A | zero | false | **true** | fast (compare to A-fast) |
| E4-D | trailing_momentum | true | **true** | fast (compare to D-fast) |

**Acceptance.**

| Criterion | Target | Verdict source |
|---|---|---|
| `acf_seam_degradation` no longer fires | metric ≥ −0.30 | rule |
| Simulated lag-1 ACF(r²) within 30% of realized | sim ∈ [+0.19, +0.35] | comparison plot |
| Mean CRPS within +5% of A-fast / D-fast (no regression from σ scaling instability) | E4-A ≤ 0.0548; E4-D ≤ 0.0530 | aggregate |
| `u_shaped_high_vol_pit` does not regress | metric ≤ +1.82 (Cell D baseline) | rule |

If both cells pass: v3a is the v3 win. Run E4-D canonically before promoting.

If E4-A passes but E4-D regresses: per-step σ and conditional re-matching interact poorly; ship E4-A as the v3 default candidate, deprecate conditional sampling.

If both fail: v3b is next.

**Cost.** ~1 day implementation + tests (3 new tests: causality of per-step σ_hist, determinism, equivalence to v2.x when per_step_sigma=false). Run time ~1h × 2 fast presets.

**Deliverables.** Config flag `per_step_sigma_injection: bool = false` in `Config`; `scale_block_per_step` in `sampling.py`; tests in `test_sampling.py`; results in `_e4_v3a.md`.

---

### E5. Search-time conditional sampling

**Background.** V2_PLAN open-question-7 was answered with the test-only contingency because search-time conditional sampling was projected at ~19 days. SPEEDUP_PLAN explicitly identifies search-time as the next natural lever for the same process-pool pattern that landed in `walk_forward.py`.

**Hypothesis.** Test-time conditional sampling re-uses weights chosen under v1 sampling. The conditional sampler's optimal `(weights, n_eff)` region may differ — search-time agreement was an assumption, not a verified fact (only the 76/76 weight match between v2.1 and v2.2 was verified, and v2.2 uses v1 search by construction).

**Cells.** Cell D fast preset with `conditional_block_sampling_in_search: true` after the process-pool extension to `search.py`.

**What it answers.**

- Are search-selected weights stable when conditional sampling is in the inner loop?
- Does search-time conditional sampling unlock additional CRPS beyond the 5% test-only Cell D gain?

**Acceptance.**

- Per-fold weight comparison plot vs test-only Cell D (D-fast).
- Mean CRPS delta: if ≤ −2% beyond D-fast, search-time conditional is a real win; if within ±1%, it's not worth the complexity.

**Cost.** ~2 days to extend `ProcessPoolExecutor` through `search.py` (mirrors Fix 3 in SPEEDUP_PLAN). Search runs ~3-4× slower than non-conditional even with the pool (one conditional forecast per grid × n_eff × val origin).

**Deliverables.** Pool wiring in `search.py`, new test for serial-vs-parallel search-result identity, results in `_e5_search_conditional.md`.

---

### E6. Vol-level feature in distance metric

**Hypothesis.** The matcher matches on z-score *shape* only — three rolling-mean z-scores. Absolute vol level is currently encoded only via σ-ratio scaling, which is downstream of analog selection. Adding `σ_EWMA(t)` (z-scored across history) as a fourth distance feature lets the matcher prefer regime-matched analogs at selection time, which should tighten high-vol calibration.

**Architectural cost.** The grid in `search.py::generate_weight_grid` is hard-coded for the 3-simplex. Extending to a 4-simplex roughly quadruples grid points at resolution 0.1 (~286 from ~66). Nelder-Mead local refine still works in any dimension.

**Cells.** Cell D fast preset, with `zscore_horizons = (20, 50, 200, "ewma_vol")` and a 4-weight grid.

**Acceptance.**

| Criterion | Target |
|---|---|
| `u_shaped_high_vol_pit` regresses toward threshold | metric ≤ +1.50 (vs Cell D's +1.66) |
| High-vol CRPS improves | ≤ 0.0810 (vs Cell D fast's 0.0826) |
| Low-vol CRPS does not regress | ≤ 0.0270 (vs Cell D fast's 0.0293, ie. parity) |
| Weight trajectory shows non-trivial `w_vol` mass | median w_vol ≥ 0.10 across folds |

If the last criterion fails, the matcher is ignoring vol level even when offered — drop and document.

**Cost.** ~2 days to plumb a configurable feature set through `features.py` → `distances.py` → `search.py`. ~3h fast preset (4× grid size).

**Deliverables.** `zscore_horizons` type relaxed to accept named string features; `features.py::compute_extra_features`; 4-simplex grid in `search.py`; tests; results in `_e6_vol_feature.md`.

---

## Tier 3 — promotion + robustness

### E7. Promote Cell D (or E2 variant) as default

Outstanding decision from [`RESULTS.md`](RESULTS.md). Gated on:

1. E3 (seed noise floor) lands. If noise overlaps v2.1, weaken language but still promote on per-vol-regime CRPS evidence.
2. E2 (momentum sweep) lands. If it finds a strictly-better Cell-D variant, promote that instead of vanilla Cell D.

**Mechanics.** Flip `configs/analog_mc/default.yaml` to mirror `default_v22.yaml` (or the E2 variant). Update `IMPLEMENTATION_PLAN.md` revision history (v2.3 entry). Keep `default_v22.yaml` as the archived acceptance run config (mirrors how `default_v21.yaml` was kept after v2.1).

### E8. Multi-asset robustness

**Hypothesis.** All current evidence is NASDAQ100. V2_PLAN carryover-3 explicitly notes `u_shaped_high_vol_pit` re-triggers across datasets. Tail inflator is deferred only while the rule stays quiet.

**Cells.** Cell D fast preset (post-E7 default) on 2-3 contrasting series:

- A single-name large-cap equity (e.g., AAPL daily closes).
- An FX cross (e.g., EUR/USD).
- A commodity or commodity ETF (e.g., GLD).

**What it answers.**

- Which decision rules fire per asset class.
- Whether the asymmetric vol clip bounds `(0.5, 3.0)` are right for FX (V2_PLAN suggested `(0.4, 2.5)` for symmetric-vol assets).
- Whether the matcher's 3 z-score horizons `(20, 50, 200)` are still right for shorter-history or higher-vol assets.

**Cost.** Each fast preset ~1 h × 3 = ~3 h, plus per-asset data ingestion (which is the real cost — see project_data_source memory on the v1 single-CSV decision; this experiment is the first that materially needs the deferred loader module).

**Deliverables.** New CSVs in `data/`, an `_e8_multi_asset.md` per-asset summary.

### E9. v3b — GARCH-conditional resampling (**promoted by E1**)

**Promoted from tier 3 to tier 2 after E1 (2026-05-19).** E1 showed simulated ACF is flat across bl ∈ {5, 10, 20}, falsifying the "block geometry is the ACF lever" hypothesis that motivated v3a (E4) as the minimal lift. The remaining mechanism candidate is σ-scaling itself — exactly what GARCH-conditional resampling addresses.

**Hypothesis.** The σ-ratio scaling step (demean → clip ratio → rescale → add drift) destroys GARCH-like vol clustering inside every block, regardless of block length. Replacing the per-block constant σ ratio with a GARCH(1,1)-simulated vol path should restore the autocorrelation structure.

**Algorithm sketch.**

1. Fit GARCH(1,1) on real returns once per fold (or once globally, with refits at fold boundaries).
2. For each Monte Carlo path: simulate a vol trajectory σ_t parametrically from the GARCH dynamics; sample 60-day analog blocks as today; rescale each *step* (not block) by `σ_t / σ_analog_at_step`.
3. Keep the analog's sign/shape structure (the non-parametric ingredient); only the magnitude is parametric.

This is the FHS sign/shape × parametric scale decomposition from Ortega et al. (`RELATED_WORKS.md` tier 4), applied to our analog-block primitive.

**Cells.** Single experiment, fast preset, two variants:

| Cell | drift | conditional | vol model | preset |
|---|---|---|---|---|
| E9-A | zero | false | GARCH(1,1) | fast |
| E9-D | trailing_momentum | true | GARCH(1,1) | fast |

**Acceptance.**

| Criterion | Target | Source |
|---|---|---|
| `acf_seam_degradation` no longer fires | metric ≥ −0.30 | rule |
| Simulated lag-1 ACF(r²) within 30% of realized | sim ∈ [+0.19, +0.35] | comparison plot |
| Mean CRPS within +5% of A-fast / D-fast | E9-A ≤ 0.0548; E9-D ≤ 0.0530 | aggregate |
| `u_shaped_high_vol_pit` does not regress | metric ≤ +1.82 (Cell D baseline) | rule |

If both cells pass: v3b is the v3 win. Run E9-D canonically before E7 (promote).

**Cost — revised estimate (2026-05-19, post-E10).** Two prior numbers were wrong:
- The original "3–5 days" inflated for a green-field `vol.py` module and broader refactor that aren't needed.
- The "1.5 days" revision applied human-dev hours to an LLM-paced session.

LLM-paced breakdown:

| Step | Active code time | Compute |
|---|---|---|
| `arch` wrapper + `simulate_garch_path` + config plumbing + walk-forward refit | ~20 min total | — |
| Refactor `scale_block` for per-step σ (keep EWMA branch identical) | ~30 min | — |
| 4 tests (causality, determinism, EWMA-equivalence, GARCH residual sanity) | ~15 min | — |
| Configs + aggregation script | ~10 min | — |
| Debugging buffer (test failures, GARCH fit edge cases) | ~45 min | — |
| Plain-FHS baseline run (single-asset GARCH + residual bootstrap, no analog blocks — isolates analog contribution) | — | ~1 h |
| E9-A (zero drift, GARCH-conditional, no analog conditional) | — | ~1 h |
| E9-D (Cell D + GARCH-conditional) | — | ~1 h |
| Diagnostics + `_e9_v3b.md` writeup | ~30 min | — |

**Total: ~2 h active coding + ~3 h compute = ~5 h wall.** Compute can overlap with downstream prep (e.g. starting on E2 / E11 analysis while runs execute). With concurrency, this fits inside a single afternoon.

~1 h fast preset × 3 cells (2 E9 + 1 plain-FHS baseline) for runs themselves.

**Risk.** Per `RELATED_WORKS.md` tier 4 — GARCH-residual nonstationarity (rare-regime residuals dominating the pool) and Cornish-Fisher-style tail breakdown. Run a textbook FHS baseline (single-asset GARCH + residual bootstrap, no analog block) alongside E9-A to isolate whether the analog-block primitive adds anything to plain GARCH FHS — without it, E9 wins are unattributable.

**Deliverables.** `src/analog_mc/vol.py` (GARCH wrapper), `src/analog_mc/sampling.py::scale_block_garch`, `configs/analog_mc/ablation_E9_{A,D}.yaml`, `configs/analog_mc/baseline_fhs.yaml` (plain FHS), `_e9_v3b.md` results.

---

## Suggested ordering (post-E9 re-estimate, 2026-05-19 late)

LLM-paced re-estimate of E9 (5h instead of 1.5 days) changes the calculus: coding E9 is cheap enough to overlap with E11 + E2 compute. Optimized timeline:

1. ~~E1, E10, E3~~ — ✅ done.
2. **E7** (promote vanilla Cell D as default, ~30 min) — no compute, all gates satisfied. **Execute next.**
3. **E11 || E9 code** — launch E11 (~1 h compute) in background; in parallel write E9 code (~1 h active). Both finish around the same time. **Net wall: ~1 h.**
4. **E2 partial || E9 tests** — launch E2 5-cell sweep (~2.5 h compute) in background; in parallel finish E9 tests + plain-FHS baseline scaffolding (~1 h active). **Net wall: ~2.5 h.**
5. **E9 fast runs** (3 × ~1 h compute = ~3 h) — by this point E11 and E2 results are in hand, so E9 diagnostics can compare against the refined Cell D variant from E2 (if any) as well as vanilla Cell D. **Net wall: ~3 h.**
6. **E9 analysis + writeup** (~30 min active).

**End-to-end: ~7 h wall, of which most is compute** (E11 1h + E2 2.5h + E9 runs 3h ≈ 6.5h compute, ~0.5h pure coding overlap). Compared to the strict-sequential ~9 h: the parallelism saves the wall time of one of the cheap experiments.

7. **E6 / E8 / E5 / E4** — sequenced by what E9 reveals (unchanged):
   - E9 closes ACF + `u_shaped_high_vol_pit` quiet → E8 (multi-asset) → v3 done.
   - E9 closes ACF, `u_shaped_high_vol_pit` creeps up → E6.
   - E9 fails → revisit E4 with revised hypothesis or escalate to E5.

**One caveat on parallelism:** each walk-forward uses a 6-worker test-eval pool. Running two compute jobs concurrently oversaturates 8 cores. So **E11 and E2 must run sequentially** between themselves — but coding E9 in the same wall-clock window is fine (single-threaded Python development is non-saturating).

---

## Decision rules read after each tier

| After | Decision |
|---|---|
| E1 | ✅ done — v3b promoted over v3a (E9 over E4) as the primary ACF-fix direction; E10 added as a quick CRPS-stacking probe. |
| E10 | ✅ done — bl=20 did NOT stack with Cell D. Promotion target = vanilla Cell D. |
| E3 | ✅ done — 0.08% seed gap, Cell D 4% gain is ~50× the noise floor. Promotion robust. |
| E2 | Optional refinement; not a blocker for E7. May yield a Cell-D variant for a follow-up re-promotion. |
| E9 | If ACF rule turns green: declare ACF carryover resolved. Else: revisit E4 with revised hypothesis. |
| E8 | Re-check all 5 decision rules per asset; resurface tail inflator if `u_shaped_high_vol_pit` fires anywhere. |

---

## What not to do

- **Do not** bundle v3 features in a single PR (same reason as V2_PLAN: destroys diagnostic attribution).
- **Do not** start E4 (v3a per-step σ) on the original hypothesis. E1 partially falsified it; if E4 ever runs, the design needs to address what E1 showed (σ-scaling is the lever, not block geometry).
- **Do not** sweep `vol_clip_lower` / `vol_clip_upper` in v3 — `clip_hit_excessive` still passes. Re-open only after E8 if a non-equity asset trips it.
- **Do not** swap the matcher for a learned similarity (transformer-encoder, etc.) in v3. That is a v4 architectural conversation, not a v3 experiment.
- **Do not** add a tail inflator preemptively. `u_shaped_high_vol_pit` is creeping but not firing; ship the trigger-gated discipline V2_PLAN established.

---

## Deliverables checklist

| Path | Purpose | Status |
|---|---|---|
| `configs/analog_mc/ablation_E1_bl{5,10,20}.yaml` | E1 cells | ✅ |
| `runs/analog_mc/20260519T{060335,064102,071520}Z` | E1 run dirs | ✅ |
| `docs/analog_mc/experiments/_e1_block_length.md` | E1 results | ✅ |
| `docs/analog_mc/figs/e1_block_length_acf.png` | E1 ACF comparison plot | ✅ |
| `scripts/_e1_aggregate.py` | E1 aggregator (reusable for E10/E9 ACF curves) | ✅ |
| `configs/analog_mc/ablation_E10_celld_bl20.yaml` | E10 cell | ✅ |
| `docs/analog_mc/experiments/_e10_celld_bl20.md` + figs | E10 results | ✅ |
| `configs/analog_mc/ablation_E3_seed{7,1337,2024,99}.yaml` | E3 seed configs (only seed7 ran) | ✅ |
| `docs/analog_mc/experiments/_e3_seed_noise.md` | E3 results | ✅ |
| `configs/analog_mc/ablation_E11_cellB_bl20.yaml` | E11 (post-E10 add) | pending |
| `configs/analog_mc/ablation_E2_s{00,25,50,75,100}.yaml` | E2 5-cell partial | pending |
| `src/analog_mc/vol.py` + `src/analog_mc/sampling.py::scale_block_garch` | E9 (v3b) implementation | pending |
| `configs/analog_mc/ablation_E9_{A,D}.yaml` + `baseline_fhs.yaml` | E9 cells | pending |
| `src/analog_mc/{features,distances,search}.py` 4-feature extension | E6 implementation | pending |
| `src/analog_mc/search.py` pool wiring | E5 implementation | pending |
| `configs/analog_mc/ablation_E4_{A,D}.yaml` + `src/analog_mc/sampling.py::scale_block_per_step` | E4 (held; design needs revisit post-E1) | held |
| `docs/analog_mc/V3_RESULTS.md` | Aggregated results — created when first v3 variant is canonically run | pending |
| `docs/analog_mc/V3_PLAN.md` | **This file** — the spec | ✅ |
