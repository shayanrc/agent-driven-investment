# analog_mc — v2 ablation studies plan

Companion to `ABLATION_STUDIES_REPORT.md` (which holds the *results*). This file is the *spec* — the design of which cells get run, what each cell answers, and how we'll know the study is done.

## Motivation

After v2.1 was shipped as the canonical default and v2.2 was deferred (see `RESULTS.md` v2.2 audit), two questions remained unanswered:

1. **Decompose the v2.2 CRPS gain.** v2.2 fast (0.05045) beat v2.1 fast (0.05313) by 5% with identical per-fold weights — a clean A/B isolated to test-time sampling. But the mechanism is still unclear: is it the conditional re-matching itself, or an interaction with the trailing-momentum drift? We don't have the *(drift = zero, conditional = true)* cell to disambiguate.
2. **Confirm v2.1 wins at full-grid resolution.** v2.1's promotion to default was justified by `runs/analog_mc/20260517T145344Z/` (`default_v21.yaml`, canonical resolution, mean CRPS 0.05265), which ties v1 canonical (0.05246) within +0.36% on aggregate, wins high-vol by 5.2%, and eliminates the sloped-PIT firing. This study formalises that as the Phase 2 result and surfaces it next to the Phase 1 attribution.

Scope of this plan: Phases 1 and 2 only. Phases 3 (tunable sweeps) and 4 (seed-noise floor) are deferred unless the Phase 1 results are ambiguous.

## Cell inventory

The walk-forward runs already on disk plus the one new run this plan adds:

| Cell | Run dir | Config | drift | conditional | n_paths | Grid | Mean CRPS | Status |
|---|---|---|---|---|---|---|---|---|
| A-fast | `runs/analog_mc/20260516T170018Z/` | `nasdaq100_fast.yaml` | zero | false | 500 | 21×2 | 0.0521 | exists |
| A-canonical | `runs/analog_mc/20260516T180000Z/` | `default.yaml` (v1, archived) | zero | false | 1000 | 66×5 | 0.05246 | exists |
| B-fast | `runs/analog_mc/20260517T050831Z/` | `nasdaq100_v21.yaml` | trailing_momentum | false | 500 | 21×2 | 0.05313 | exists |
| B-canonical | `runs/analog_mc/20260517T145344Z/` | `default_v21.yaml` | trailing_momentum | false | 1000 | 66×5 | **0.05265** | exists (Phase 2) |
| **C-fast** | TBD | `ablation_C_cond_only.yaml` (new) | **zero** | **true (test-only)** | 500 | 21×2 | — | **NEW** |
| D-fast | `runs/analog_mc/20260517T070003Z/` | `nasdaq100_v22.yaml` | trailing_momentum | true (test-only) | 500 | 21×2 | 0.05045 | exists |

Phase 1 needs Cell C-fast. Phase 2 is the A-canonical vs B-canonical comparison and is already done.

## Phase 1 — 2×2 decomposition at fast resolution

**Run to add.** `configs/analog_mc/ablation_C_cond_only.yaml` — copy of `nasdaq100_v22.yaml` with `drift_mode: zero`. All other knobs (n_paths=500, 21×2 grid, `conditional_block_sampling: true`, `conditional_block_sampling_in_search: false`) are identical to the v22 preset so the comparison with D-fast is one variable.

**Cost.** ~6 hours wall time. Test-eval dominated because conditional sampling runs at test time only (search uses the v1 sampler via the `_in_search` contingency).

**What the 2×2 answers.**

| Quantity | Computation | Interpretation |
|---|---|---|
| Drift effect, no conditional | B-fast − A-fast = 0.05313 − 0.0521 = **+0.0010** | Drift adds ~1.9% CRPS in low/mid-vol regimes; offsets the high-vol gain |
| Drift effect, with conditional | D-fast − C-fast (needs C) | Does drift still hurt or now help when stacked with conditional? |
| Conditional effect, no drift | C-fast − A-fast (needs C) | Isolates conditional sampling's contribution |
| Conditional effect, with drift | D-fast − B-fast = 0.05045 − 0.05313 = **−0.00268** | Conditional saves 5% on top of drift |
| **Interaction** | (D − B) − (C − A) | ≈ 0 → effects additive; significantly negative → conditional rescues drift miscalibration; significantly positive → effects cancel |

## Phase 2 — canonical search-resolution control

**Run already done.** `runs/analog_mc/20260517T145344Z/` is the canonical v2.1 run (drift on, conditional off, full 66×5 grid, 1000 paths). It ties v1 canonical (0.05246) on aggregate at 0.05265 (+0.36% within Monte Carlo noise), wins high-vol-regime CRPS by 5.2%, and eliminates the sloped-PIT firing (+0.158 → +0.053). On the strength of those numbers, `default.yaml` was flipped to `drift_mode: trailing_momentum` and `default_v21.yaml` preserved as the like-for-like reproduction.

Phase 2's only remaining task is documentation: include the A-canonical ↔ B-canonical comparison in the headline 2×2 table in `ABLATION_STUDIES_REPORT.md` so the search-resolution-controlled drift effect (+0.00019 ≈ +0.36%) sits alongside the fast-preset deltas.

## Decomposition analyses (no new runs)

For each cell, compute and tabulate using `scripts/analog_mc/ablation_decompose.py`:

| Decomposition | What it answers |
|---|---|
| Mean / median aggregate CRPS | Top-line cell ranking |
| Per-step CRPS at h=1, 15, 30, 60 | Does conditional help more at later horizons (more re-matching opportunities)? |
| Per-vol-regime CRPS (low / mid / high) | Where does drift help? Where does conditional help? |
| `sloped_global_pit` metric | Is the PIT correction from drift alone, or does conditional move it too? |
| `acf_seam_degradation` metric | Confirms the structural ceiling from the v2.2 audit — conditional shouldn't materially help here |
| Per-fold win-rate matrix | Robustness — when one cell beats another, how often (and is the average driven by a few outliers)? |
| Mean CRPS deltas vs first cell | Quick numeric summary |

All five decompositions run cheaply on persisted artifacts; the script skips the fixed-weight baseline pass that hangs `render_diagnostics` on conditional configs (see RESULTS.md v2.2 audit).

## Sequencing

1. Author `configs/analog_mc/ablation_C_cond_only.yaml`. ✅
2. Launch Cell C walk-forward in background; arm a 15-min cron status check. ✅
3. **In parallel:** author `scripts/analog_mc/ablation_decompose.py` and dry-run it on the 5 existing cells (A-fast, A-canonical, B-fast, B-canonical, D-fast) to catch any plumbing issue before C lands. ⏳
4. When Cell C completes (~6 h), render diagnostic figs for it and rerun decompose with all 6 cells.
5. Author `docs/analog_mc/ABLATION_STUDIES_REPORT.md` — the results report.
6. Update `RESULTS.md` (one-line index pointer to ABLATIONS) and `V2_PLAN.md` (ablation-conclusion paragraph under the v2.2 audit).
7. Commit granularly (preset, script, results doc, doc pointers).

All runs are crash-resumable via `walk_forward.run_walk_forward(resume=True)` so interruptions are recoverable.

## Deliverables

| Path | Purpose |
|---|---|
| `configs/analog_mc/ablation_C_cond_only.yaml` | Cell C preset |
| `scripts/analog_mc/ablation_decompose.py` | Multi-run decomposition table generator |
| `docs/analog_mc/ABLATION_STUDIES_REPORT.md` | Results report — headline 2×2 + Phase-2 row + all decompositions + conclusion |
| `docs/analog_mc/ABLATION_STUDIES_PLAN.md` | **This file** — the spec |
| Pointer updates in `RESULTS.md` and `V2_PLAN.md` | Cross-references; not duplicates of ABLATION_STUDIES_REPORT.md content |

## Existing helpers to reuse

- `analog_mc.diagnostics.{load_run, aggregate_crps_overall, aggregate_crps_per_step, aggregate_crps_per_vol_regime, decision_rules}` — used by `ablation_decompose.py`
- `scripts/analog_mc/render_diagnostics.py` — render Cell C figures
- `scripts/analog_mc/plot_forecast_vs_realized.py` — could be extended to a 4-cell fan comparison if visualisation helps the writeup

## Verification

For Cell C walk-forward:

1. `runs/analog_mc/<new>/lock` removed and `Walk-forward complete: 76 folds` log line appears in `runs/analog_mc/_ablation_C.log`.
2. `uv run python scripts/analog_mc/render_diagnostics.py runs/analog_mc/<new>` runs to completion. **Will hang on the fixed-weight baseline re-eval** because the preset has `conditional_block_sampling=true` — same workaround as the v2.2 audit: either skip the fixed-baseline section by reading the persisted `summary.parquet` directly, or temporarily flip the run dir's `config.yaml` for the diagnostics pass. The 2×2 attribution does not depend on it.
3. `uv run python scripts/analog_mc/ablation_decompose.py <cells ...>` emits the comparison tables with no crashes.

## Decision rules (read after Phase 1 completes)

| Question | Evidence | Action |
|---|---|---|
| Keep v2.1 as default? | Phase 2 done: B-canonical 0.05265 vs A-canonical 0.05246 ties on aggregate, wins on high-vol + PIT | **Already shipped.** |
| Keep v2.2 as opt-in? | Conditional effect (C − A) and (D − B) both negative within Monte Carlo noise | Keep `nasdaq100_v22.yaml` documented as opt-in (~12× test-eval cost). |
| Either delta within noise? | (one of) (C − A) ≈ 0 or (D − B) ≈ 0 | Conditional is not statistically supported; remove the preset and document the negative result. |
| Open question for v3? | Interaction term (D − B) − (C − A) | If non-zero: per-path re-matching specifically rescues drift-induced miscalibration — worth investigating in v3 alongside the structural ACF ceiling. |
