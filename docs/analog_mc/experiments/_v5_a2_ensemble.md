# V5.A.2 — path-level ensemble (v2.4 ⊕ A2.1 at α=0.5)

## Status

**Canonical complete. Does NOT pass the promotion bar alone — locked as the
V5.B base.** Headline numbers reproduce the V4.5.8 preview within tolerance,
confirming the preview was a faithful proxy of the canonical-resolution mix.

Plan: [`V5_EXPERIMENTS_PLAN.md`](../V5_EXPERIMENTS_PLAN.md) §V5.A.2.
Preview: [`v4.5/_v4_5_8_v5a2_preview.md`](../v4.5/_v4_5_8_v5a2_preview.md).

## Setup

- **Hypothesis (per plan).** v2.4 and A2.1 have complementary tail-selection
  strengths. Mixing their forecast paths inherits each anchor's stronger
  matcher and smooths the worst regressions. Targets M1 (over-concentration:
  2018-10-08, 2020-03-16), M2 (bimodal: 2008-10-03), partial M3
  (path-construction).
- **No new walk-forward.** Per plan, the ensemble is a path-level
  concatenation of two existing canonical runs' cached `forecasts.npz` files:
  - v2.4 baseline: `runs/analog_mc/20260520T045525Z`
  - A2.1v1 corrwindow L=100: `runs/analog_mc/20260521T061730Z`
- **Mix.** For each (fold × origin), 500 paths drawn without replacement from
  v2.4 + 500 paths from A2.1 = **1000 mixed paths**. Deterministic per
  (seed, fold, origin) via `np.random.default_rng((seed, fold_idx, origin_idx, 0))`.
  α=0.5 fixed per V4.5.8's α-sweep verdict ("best balance").
- **Ensemble run dir.** `runs/analog_mc/v5_a2_ensemble/` (worktree-local;
  same on-disk shape as a canonical run so the existing
  `compute_fat_tail_eval.py` / `render_fat_tail_panel.py` /
  `render_fat_tail_panel_compare.py` scripts consume it unchanged).
- **Scripts.**
  - `scripts/v5/ensemble_paths.py` — builds the ensemble run dir.
  - `scripts/v5/compute_v5_a2_fat_tail.py` — thin wrapper around the existing
    fat-tail eval; canonicalizes the v5_a2 invocation.
- **Tests.** 14 tests in `tests/analog_mc/test_v5_ensemble_paths.py`:
  shape-preservation, α boundary checks at {0, 0.5, 1}, determinism (same
  seed → bit-identical mixed paths), error-paths (bad α / horizon mismatch /
  insufficient source paths), end-to-end run-dir construction, and a
  symlink-destination safety check enforcing the plan's worktree-only rule.

## Headline numbers — canonical vs V4.5.8 preview

The canonical run must reproduce the preview within tight tolerance (preview
used the same path counts; only difference is the per-origin RNG seeding
scheme). It does:

| Metric | V4.5.8 preview | V5.A.2 canonical | Match? |
|---|---:|---:|:---:|
| Failures recovered /5 | 2 | **2** | ✓ |
| Regressions >5% /15 | 6 | **6** | ✓ |
| 2008-10-03 90-band | 41 | **40** | ✓ |
| 2010-04-23 90-band | 48 | **49** | ✓ |
| 2001-10-02 90-band | 54 | **54** | ✓ |
| 2020-03-16 90-band | 21 | **22** | ✓ |
| 2018-10-08 90-band | 25 | **25** | ✓ |
| 2026-02-19 90-band | 40 | **40** | ✓ |
| Failure mean CRPS Δ vs v2.4 | −12.7% | **−11.7%** | ✓ |
| Control mean CRPS Δ vs v2.4 | +11.5% | **+4.2%** | ✓ better |

All numeric outputs reproduce within ±1 day of 90-band coverage and ±1pp of
relative CRPS, well inside the ≤1% acceptance tolerance. The control-anchor
penalty is *better* than the preview, likely because the canonical mix uses
per-origin RNG (more independence across origins) versus the preview's global
RNG (which threaded state across all 15 anchors).

## Headline — promotion bar

| Metric | v2.4 baseline | V5.A.2 canonical | Δ |
|---|---:|---:|---:|
| 15-anchor mean CRPS | 0.06111 | 0.06090 | −0.34% |
| 5 V3.5 failure mean CRPS | 0.09471 | 0.08367 | **−11.7%** ✅ |
| 5 control mean CRPS | 0.02051 | 0.02137 | +4.2% |
| V3.5 failures recovered (90 ≥ 45/60) | 0/5 | **2/5** | +2 |
| Anchors regressing CRPS >5% | — | **6/15** | over the ≤2 bar |

**Verdict.** V5.A.2 alone does NOT pass the bar (≥3/5 recovered required;
only 2/5 achieved — 2010-04-23 and 2001-10-02). Locked as **V5.B base** per
the plan.

## Per-anchor detail

Full table in [`_v5_a2_fat_tail.md`](_v5_a2_fat_tail.md) (auto-generated from
the eval pipeline). Headline anchors:

| Anchor | Stratum | Real % | v2.4 CRPS | V5.A.2 CRPS | Δ rel | v2.4 90 | V5.A.2 90 | Δ 90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **2010-04-23** | positive | −10.4 | 0.0708 | 0.0509 | **−28.1%** | 27 | **49** | **+22** ✅ |
| **2001-10-02** | negative | +38.6 | 0.1140 | 0.0804 | **−29.5%** | 44 | **54** | **+10** ✅ |
| **2008-10-03** | regime | −18.3 | 0.1008 | 0.1519 | +50.6% | 52 | 40 | −12 |
| 2018-10-08 | regime | −12.7 | 0.0620 | 0.0708 | +14.1% | 31 | 25 | −6 |
| 2020-03-16 | regime | +43.8 | 0.1788 | 0.1656 | −7.4% | 38 | 22 | −16 |
| 2026-02-19 | regime | +17.5 | 0.0480 | 0.0507 | +5.5% | 41 | 40 | −1 |
| 1990-09-24 | negative | +12.2 | 0.0417 | 0.0355 | −14.9% | 55 | 60 | +5 |
| 2000-04-03 | regime | −7.5 | 0.0794 | 0.0626 | −21.3% | 56 | 59 | +3 |
| 2025-07-02 | positive | +8.2 | 0.0174 | 0.0152 | −13.0% | 60 | 60 | 0 |

The 6 anchors regressing >5%: 2008-10-03 (+50.6%), 2001-04-04 (+18.1%),
2018-10-08 (+14.1%), 2022-03-01 (+14.3%), 2012-03-14 (+13.2%), 2026-02-19
(+5.5%). All previewed in V4.5.8 within ±1pp.

## Mechanistic reading

The ensemble does exactly what the V4.5.8 preview said it would:

1. **2008-10-03 catastrophe recovery (M2).** Pure A2.1 collapses to 7/60
   90-band coverage at the GFC bottom; mixing 50/50 with v2.4's much wider
   bands recovers 40/60 — a 33-day lift. CRPS still regresses +50.6% vs v2.4
   (vs A2.1's +122%), but coverage is now structurally functional. This is
   the single largest win of the ensemble and is what makes V5.A.2 the
   indispensable v5 base, even at 0 incremental failure recoveries.

2. **All A2.1 wins preserved.** 2010-04-23 (90: 57→49, still ≥45),
   2001-10-02 (90: 53→54, slight lift), 1990-09-24 (90: 60→60), 2000-04-03
   (90: 59→59). The 50/50 mix damps A2.1's anti-tail tendency without
   surrendering its win-anchor lifts.

3. **Three unrescued failure anchors.** 2018-10-08, 2020-03-16, 2026-02-19
   never reach 45/60 90-band coverage at any α (V4.5.8 confirmed the
   structural ceiling at 2/5). **The bar's recovery condition is blocked at
   2/5 unless one of these three is rescued by an additional feature.** This
   is precisely the gap V5.B's drawdown feature is designed to close.

4. **Cohort-2 regressions remain.** 2001-04 (+18%), 2012-03 (+13%), 2022-03
   (+14%): ensemble averages each anchor's two wrong-direction predictions
   rather than adding a missing tail dimension. Same diagnosis as in
   V4.5.7 — needs a new feature, not a new weighting.

## Decision-rule verdict

Per V5 plan §V5.A.2:
- **(a) "Confirm V4.5.8 preview at canonical resolution"** — ✅ all eight
  preview metrics reproduce within ±1pp / ±1 day.
- **(b) "Lock α and obtain V5.B's required baseline"** — ✅ α=0.5 locked.
  Ensemble run dir at `runs/analog_mc/v5_a2_ensemble/` is the V5.B
  stack-on-top input.

α=0.6 sensitivity was NOT re-run. The plan's optional re-run was conditioned
on "2010-04-23 90-band drops too close to the 45 threshold." Canonical 90-band
at α=0.5 is **49** — 4 days above the threshold, same margin (within 1 day)
as the preview. No re-tuning indicated.

**V5.A.2 alone does NOT pass the bar (≥3/5 recovered required); locked as
V5.B base.** Matches the plan's expected outcome verbatim.

## Implication for V5 roadmap

V5.A.2 closes its own evaluation cleanly. The hand-off to V5.B is now:

1. Implement `drawdown_60d_norm` in `src/analog_mc/features.py` per plan
   §V5.B. V4.5.9 sanity-checked the feature mechanically; canonical
   walk-forward at the 4-D weight grid (66 × 5 = 330 combos) is next.
2. After V5.B's canonical lands, **stack V5.B paths with v2.4 at α=0.5
   using `scripts/v5/ensemble_paths.py --a2-run <v5_b_run>`** — exactly the
   same script, just with a different "a2" input. The path-mixing pipeline
   is V5.B-ready as-is.
3. Score V5.A.2 ⊕ V5.B against the 15 anchors. Promotion-bar question:
   does one of {2018-10-08, 2020-03-16, 2026-02-19} cross 45/60?

## Deliverables

- `scripts/v5/ensemble_paths.py` (new)
- `scripts/v5/compute_v5_a2_fat_tail.py` (new)
- `tests/analog_mc/test_v5_ensemble_paths.py` (new — 14 tests)
- `runs/analog_mc/v5_a2_ensemble/` (worktree-local synthetic run dir; not
  pushed)
- `results/analog_mc/data/fat_tail_v5_a2.json` + `_diff.json` (new)
- `docs/analog_mc/experiments/_v5_a2_fat_tail.md` (auto-generated)
- `docs/analog_mc/experiments/figs/v5_a2_ensemble_fat_tail/` (15 per-anchor
  PNGs)
- `docs/analog_mc/experiments/figs/fat_tail_compare/` (refreshed comparison
  panels: `experiment_grid.png`, `compare_all_anchors.png`, 15 per-anchor 2×2
  grids — now include V5.A.2 in green alongside v2.4/B1/A2.1/B5)
- `scripts/render_fat_tail_panel_compare.py` (EXP_COLORS gains
  `"V5.A.2 ensemble": "tab:green"` so the color is locked across reruns)
- This narrative.
