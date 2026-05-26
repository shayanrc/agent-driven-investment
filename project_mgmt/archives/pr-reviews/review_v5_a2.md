# V5.A.2 Review — branch v5-experiments → main

## Summary

Approve with minor comments. The implementation faithfully realizes the V5.A.2
plan: a path-level mixture (concatenate 500 v2.4 paths + 500 A2.1 paths per
origin) which is the correct operationalization of "ensemble at α=0.5" as
specified in the plan. Headline numbers reproduce within tolerance and the
sub-agent's reported failures/regressions counts verify exactly. Six low/medium
notes below — none are blocking. C1–C6 are not violated by the ensemble step
itself.

## Verified self-reports

All sub-agent claims reproduce from the on-disk artifacts on `v5-experiments`:

- Failures recovered 2/5 ✓ — `fat_tail_v5_a2_diff.json` shows 2010-04-23
  (in_90=49) and 2001-10-02 (in_90=54) ≥45; 2018-10-08 (25), 2020-03-16 (22),
  2026-02-19 (40) all <45.
- Regressions >5% CRPS = 6/15 ✓ — counted from `per_anchor[*].delta_mean_crps_rel`.
- 2008-10-03 90-band 7→40 ✓ — but note the *baseline* in the diff JSON is v2.4
  (52/60), not A2.1 (which the sub-agent's note "v2.4=7/60" mis-states). The
  "7" comes from pure A2.1 in V4.5.8 preview. V5.A.2 vs v2.4 is 52→40 (i.e., a
  regression vs v2.4) but a *recovery* vs A2.1. This is mentioned correctly in
  `_v5_a2_ensemble.md` §"Mechanistic reading" but the one-line summary handed
  back is muddled.
- Failure mean CRPS −11.66% vs v2.4 ✓.
- Control mean CRPS +4.18% (vs preview's +11.5%) ✓ — confirmed in diff JSON
  `control_anchors.delta_mean_crps_rel`.
- 14 ensemble tests present ✓.
- `render_fat_tail_panel_compare.py` 2×2 → auto 3×2 ✓ — diff is 14 lines.
- Per-(seed, fold, origin) RNG ✓ — at `scripts/v5/ensemble_paths.py:201`.

## Comments

### [Low] `mix_paths` is exported and heavily tested, but dead in the canonical path
**File:** `scripts/v5/ensemble_paths.py:50-110`, `tests/analog_mc/test_v5_ensemble_paths.py:35-115`
**Observation:** `ensemble_one_fold` does NOT call `mix_paths`. It re-implements
the same logic inline (lines 197-220) to keep paths and ratios paired via a
single index draw per origin. As a result, 9 of the 14 tests exercise
`mix_paths`, which is dead code as far as the canonical artifacts are
concerned. The 5 end-to-end tests do cover the real path through
`ensemble_one_fold` / `write_run_dir`, so behavior is verified — but the test
weighting is misleading ("14 tests" sounds heavier than the 5 that matter).
**Suggested action:** Either (a) refactor `ensemble_one_fold` to delegate to
`mix_paths` for paths and a paired helper for ratios (so the test coverage is
real), or (b) delete `mix_paths` + `mix_ratios` and rewrite those 9 tests
against `ensemble_one_fold` with synthetic mini-runs. Option (b) also kills
the misleading `mix_ratios` docstring (see next comment).

### [Low] `mix_ratios` is unreachable and its docstring describes a contract the caller cannot keep
**File:** `scripts/v5/ensemble_paths.py:112-140`
**Observation:** `mix_ratios` is exported but never called. Its docstring
says "the caller must structure RNG draws to align" — meaning callers must
feed it RNG state that produces the same `idx_a`/`idx_b` as a sibling
`mix_paths` call. This is impossible to use safely: `rng.choice(N, k)`
mutates RNG state, so a second call with the same RNG won't produce the same
indices. The actual production code (inlined in `ensemble_one_fold`)
correctly draws indices once and reuses them for both paths and ratios. So
the function is both unused and unsound as written.
**Suggested action:** Delete `mix_ratios`. If the abstraction is wanted,
replace with a single helper that returns `(paths_mixed, ratios_mixed)` from
one pair of index draws.

### [Low] `mix_paths` docstring claims V4.5.8 bit-equivalence; canonical isn't bit-equivalent
**File:** `scripts/v5/ensemble_paths.py:70-73`
**Observation:** The docstring says the index permutation is "identical to
the V4.5.8 preview's approach." The `_v5_a2_ensemble.md` narrative
correctly contradicts this — the canonical uses per-(seed, fold, origin)
RNG while the preview used a shared global RNG, and that's *why* control
CRPS is +4.2% canonical vs +11.5% preview. The docstring should say the
function is bit-identical *given the same rng*, not that it matches the
preview.
**Suggested action:** Update the docstring to clarify the seeding scheme
difference vs preview, or drop the V4.5.8 comparison entirely from the
function-level docs (the narrative covers it).

### [Low] No test for one-source-only fold or NaN/Inf inputs
**File:** `tests/analog_mc/test_v5_ensemble_paths.py`
**Observation:** The reviewer prompt asked about edge cases including "one
branch missing for a fold" and "NaN/Inf handling". Neither is tested.
`ensemble_one_fold` `raise SystemExit` paths (origin_idx mismatch, realized
mismatch, test_range mismatch) and `list_folds(a2_run) != fold_indices` in
`write_run_dir` are also untested. These are the failure modes most likely
to hit a future user mixing newly-built runs.
**Suggested action:** Add at minimum (a) a test that mismatched
`origin_idx` arrays raise, (b) a test that mismatched fold sets raise, and
(c) a test that NaN-containing input paths are propagated faithfully (or
explicitly rejected). The asymmetric-`n_paths` case (e.g., v2.4 has 1000
paths, A2.1 has 500) is also worth a single test since
`mix_paths`/`ensemble_one_fold` already check this with `ValueError` /
`SystemExit`.

### [Low] `mix_ratios` parameter `rng` is in the signature but mutation order doesn't match `mix_paths`
**File:** `scripts/v5/ensemble_paths.py:112-140`
**Observation:** Already covered by the "delete it" suggestion above, but
worth flagging separately for the reader: even if a caller passed a
*fresh* RNG with the same seed as the one given to `mix_paths`, the indices
would match — but they'd then NOT be the same as what `mix_paths` drew
(because by then `mix_paths`'s RNG has advanced). The function as written
cannot correctly pair ratios with a path mixture done by `mix_paths`. The
inline code in `ensemble_one_fold` avoids this trap by drawing indices once.
**Suggested action:** Confirms the "delete it" recommendation.

### [Low] `render_fat_tail_panel_compare.py` change is minimal but lacks a test
**File:** `scripts/render_fat_tail_panel_compare.py:178-225`
**Observation:** The 2×2 → auto-sized grid change is correct and minimal:
- N≤4 preserves the prior 2×2 behavior exactly (ncols=2, nrows=2).
- N=5 widens to 3×2 with one hidden axis.
- N=6 still 3×2.
- N=7 → 3×3 with two hidden axes. Etc.
The `hasattr(axes, "flatten")` guard handles `ncols=1` degenerate cases
(though current code can't produce ncols=1 since `ncols = 2 if n<=4 else 3`).
`axes_flat[len(panels):]` correctly hides leftover axes. The legend on
`axes_flat[0]` still works. The change matches the user-flagged justification
("regenerate cross-experiment fat-tail figures" implies the grid must grow).
There's no test of the rendering function but the script has no test
coverage on main either, so this is a pre-existing gap not a regression.
**Suggested action:** Accept as-is. If a regression test is wanted, a tiny
smoke test that calls `render_comparison` with 5 dummy panels and asserts a
file is written would suffice.

### [Low] `scripts/v5/__init__.py` is empty but `scripts/` is not a package
**File:** `scripts/v5/__init__.py`
**Observation:** The test file inserts `scripts/` into `sys.path` and imports
`v5.ensemble_paths`. The `__init__.py` exists but `scripts/__init__.py`
does not — so `v5` works only because `sys.path` is pointed at `scripts/`.
This is fine and consistent with how `tests/analog_mc/` already imports
from `scripts/` elsewhere in the codebase, but readers may be momentarily
confused by the standalone `__init__.py`. Not actionable.

## Plan-match check

V5.A.2 §Method bullets 1-5 (lines 53-57 of V5_EXPERIMENTS_PLAN.md):
1. "Take v2.4 and A2.1v1 forecasts.npz from existing canonical runs" — ✓
   `--v24-run runs/analog_mc/20260520T045525Z`,
   `--a2-run runs/analog_mc/20260521T061730Z` defaults in the script match the
   plan's "Starting state" table exactly.
2. "concatenate path arrays at α = 0.5 (500 + 500 = 1000)" — ✓ `n_target=1000`,
   `alpha=0.5` default; the math is `n_b_take = round(0.5 * 1000) = 500`,
   `n_a_take = 500`. This is **mixture of distributions** (concatenate samples,
   compute distributional metrics on the union), NOT moment-averaging
   (averaging quantiles or means). The plan's wording is unambiguously the
   mixture, and the code does the mixture. ✓
3. "Recompute all CRPS / coverage / PIT / diagnostics on the mixed path set" —
   ✓ delegated to existing `compute_fat_tail_eval.py` which reads
   `forecasts.npz` and computes everything from the path matrix.
4. "Run the 15-anchor fat-tail panel" — ✓ figures present in
   `docs/analog_mc/experiments/figs/v5_a2_ensemble_fat_tail/` (15 PNGs).
5. "(Optional) Re-do at α=0.6 if 2010-04-23 90-band coverage drops too close
   to 45 threshold" — not run. Justified in `_v5_a2_ensemble.md`: canonical
   2010-04-23 = 49/60, margin 4 days, same as preview. Optional condition not
   triggered. ✓

V5.A.2 decision rule (lines 61): "(a) confirm V4.5.8 preview at canonical
resolution, (b) lock α and obtain V5.B's required baseline." Both met per the
narrative's verification table. ✓

No silent scope drift detected.

## C1–C6 compliance

The ensemble step combines two cached `forecasts.npz` artifacts that
already passed C1–C6. The relevant question is whether the path-loading or
metadata-merging step breaks anything:

- **C1 (causal features):** Inputs are already-computed forecasts. The
  ensemble does not re-touch features. ✓
- **C2 (n_eff parameterization):** `n_eff` from v2.4 fold is carried into the
  synthesized summary under `ensemble_source.v24_n_eff`. The `n_eff` at the
  fold-summary top level also stays as v2.4's value (line 222 of
  `ensemble_paths.py`: `synth = dict(s24)`). Since A2.1 used a different
  n_eff (50 vs v2.4's tuned value), downstream plot titles will show v2.4's
  n_eff for the ensemble, which is a minor display issue but does not affect
  any metric. The narrative explicitly tags this as "carry v24's values for
  downstream plot titles." Accept. ✓
- **C3 (per-analog vol scaling):** Each path was already vol-scaled in its
  source run. The ensemble preserves path-to-ratio pairing by using the
  single `idx_a`/`idx_b` draw for both arrays in `ensemble_one_fold`
  (lines 201-220), so paths and their corresponding pre-clip σ ratios stay
  together. ✓
- **C4 (running EWMA σ):** Per-run property, untouched. ✓
- **C5 (forward sampling):** The ensemble draws a *permutation* of path
  rows; it does NOT re-shuffle timesteps within a path. Horizon ordering
  preserved. ✓
- **C6 (walk-forward boundary discipline):** Origin metadata invariants are
  *checked*: `np.array_equal(origins24, originsa2)` and
  `np.allclose(realized_24, realized_a2)` at lines 168-171. If the two
  source runs disagree on fold boundaries or realized arrays, the ensemble
  script aborts. ✓

No C1–C6 violation introduced.

## Reports check

`_v5_a2_ensemble.md`:
- "Does NOT pass the promotion bar alone — locked as the V5.B base" ✓ matches
  the plan's expected outcome.
- Headline table matches `fat_tail_v5_a2_diff.json` numerics.
- "α=0.5 locked" ✓.
- The mechanistic reading correctly distinguishes that 2008-10-03 is a
  *recovery vs A2.1* but a *regression vs v2.4* (the table shows v2.4 90=52,
  V5.A.2 90=40; the +50% CRPS regression vs v2.4 is acknowledged). Not
  over-claiming. ✓

`_v5_a2_fat_tail.md`:
- Auto-generated table; numerics match the JSON. ✓
- Headline questions section correctly reports 2/5 recovered and 6/15
  regressing. ✓

Neither report claims promotion. Both correctly position V5.A.2 as a base for
V5.B.

## Out of scope / parking lot

- The two dead helper functions (`mix_paths` exported but unused in canonical
  path, `mix_ratios` unsound + unused) should probably be cleaned up before
  V5.B builds on this script. Defer to V5.B's PR if not done here.
- `compute_v5_a2_fat_tail.py` is a thin wrapper around
  `compute_fat_tail_eval.py` whose only value is baking in the defaults. The
  plan's deliverables manifest required it, so it's appropriate to keep, but
  a one-line shell alias would do the same job. No action.
- The `--baseline-json` default in `compute_v5_a2_fat_tail.py` points at
  `results/analog_mc/data/fat_tail_baseline_v24.json` which is assumed to
  already exist. If a future contributor wipes the baseline JSON, the script
  errors. Worth a one-line existence check, but not blocking.
