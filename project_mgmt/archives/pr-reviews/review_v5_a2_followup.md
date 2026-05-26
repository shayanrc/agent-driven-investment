# V5.A.2 Cleanup — Final Review

## Verdict
APPROVE (with one Low-severity follow-up to park in V6_TBD).

## Verification of original 6 findings

**#1 `mix_paths` dead in canonical path.** Addressed. `git show 92d1752` deletes `mix_paths` (lines 51-111 of the pre-cleanup file) and rewrites the test suite. Every test in `tests/analog_mc/test_v5_ensemble_paths.py` now exercises `ensemble_one_fold` or `write_run_dir`, both of which are the real CLI entry points. The "fake coverage" smell is gone.

**#2 `mix_ratios` unsound + unreachable.** Addressed by deletion. The original review's preferred remedy ("Delete `mix_ratios`") is exactly what was shipped — no replacement helper introduced, which matches the recommendation ("If the abstraction is wanted, replace with a single helper" was conditional). `ensemble_one_fold`'s inline single-index-draw remains the only path.

**#3 `mix_paths` docstring V4.5.8 claim wrong.** Resolved by deletion (docstring no longer exists). No surviving reference to "identical to the V4.5.8 preview's approach" in `scripts/v5/ensemble_paths.py`.

**#4 Missing edge-case tests.** Addressed and then some. New tests cover: `origin_idx` mismatch, realized mismatch, fold-set mismatch, asymmetric `n_paths`, and NaN/Inf propagation. The original review's "(a)(b)(c)" minimum is met, and the optional asymmetric-paths case is also covered. Pairing test (`test_ensemble_one_fold_paths_and_ratios_paired`) is a nice unrequested bonus that pins C3-relevant behavior.

**#5 `mix_ratios` RNG contract impossible.** Resolved by deletion.

**#6 `render_fat_tail_panel_compare.py` lacks test.** Correctly left as-is. `git show 92d1752 --stat` shows only `scripts/v5/ensemble_paths.py` and `tests/analog_mc/test_v5_ensemble_paths.py` were modified; the render script is untouched, matching the original review's "Accept as-is."

## Byte-identical artifacts

Computed in worktree at HEAD `92d1752`:

```
b6c24c290a4c88a2baf3d0bc811b2d8e7b032bcd163dd8f8262ce07a94af40ec  results/analog_mc/data/fat_tail_v5_a2.json
d21b5227ac62d3391bb9ec6626c7e66fe52284c45dbb998e3bf8dd015152f24a  results/analog_mc/data/fat_tail_v5_a2_diff.json
```

Both match expected exactly. `git show 92d1752 --stat` confirms no file under `results/analog_mc/data/` is in the diff — invariance is structural, not coincidental. V5.B baseline is safe.

## New tests scrutiny

**`test_ensemble_one_fold_rejects_origin_idx_mismatch`.** Constructs both runs with matching summaries (`test_start`/`test_end` agree, so the prior gate passes), then mutates `a2/folds/0/forecasts.npz`'s `origin_idx` array (`bad_origins[0] += 1`, keeps endpoint matching). Walking the code: `load_fold_summary` test-range gate passes, then `np.array_equal(origins24, originsa2)` fires because `origins24=[1000,1001]` vs `originsa2=[1001,1001]`. The match string `"origin_idx arrays differ"` uniquely identifies the intended raise. No false-positive risk.

**`test_ensemble_one_fold_rejects_realized_mismatch`.** Mutates `realized` to `realized + 0.1`. `origin_idx` unchanged, summary unchanged. Walking the code: test-range and `origin_idx` gates pass; `np.allclose(npz24["realized"], npza2["realized"], atol=1e-9)` fails by 0.1. Match string `"realized arrays differ"` uniquely identifies the right raise. Solid.

**`test_write_run_dir_rejects_fold_set_mismatch`.** v24 has folds `[0, 1]`, a2 has fold `[0]`. `write_run_dir` calls `list_folds(v24_run) → [0,1]`, `list_folds(a2_run) → [0]`, comparison fails, raises `SystemExit("non-matching fold sets")`. Match string is specific. The earlier `out_dir.exists()` check passes (no `out` dir yet), `mkdir`'s `folds/` dir is created — fold-set check happens after that but before any fold work, so the raise is from the intended gate. Good.

**`test_ensemble_one_fold_propagates_nan_and_inf`.** Pins propagation, not rejection — explicitly documented as "pins current behavior." Asserts `np.isnan(paths).sum() == 1` and `np.isinf(paths).sum() == 1`. Uses `alpha=0.0, n_target=2, n_paths=2` so both rows of v24 are necessarily included regardless of RNG permutation (no false-positive from RNG luck). Matches production behavior (no NaN/Inf scrubbing in `ensemble_one_fold`). Behaviorally a regression-pin rather than a contract test — appropriate label, no over-claim.

**`test_ensemble_one_fold_asymmetric_n_paths_raises`.** v24 has 100 paths, a2 has 500, `n_target=1000`, `alpha=0.5` → needs 500 from each. `path_rng.choice(100, size=500, replace=False)` raises `ValueError`. Only covers one direction (v24 < needed). A second direction (a2 < needed) would also raise from the same code path on `idx_b`, so the coverage gap is symmetric/cosmetic. Acceptable; no flag.

## Test suite result

`uv run pytest`: `573 passed, 4 skipped in 270.19s` (exit 0). Matches the claimed 570→573. New file has exactly 17 `def test_` definitions (claim: 14→17). All consistent.

## Alpha-validation regression

Confirmed: no `0.0 <= alpha <= 1.0` check survives anywhere in `scripts/v5/ensemble_paths.py` (grep on `alpha` shows only arithmetic uses, JSON serialization, CLI default, and the function parameter — no validation). Pre-cleanup, `mix_paths` had `if not (0.0 <= alpha <= 1.0): raise ValueError(...)`. That guard is now gone.

Behavior of `ensemble_one_fold` under `alpha=-0.1`: `n_b_take = round(-0.1*1000) = -100`, `n_a_take = 1100`. `path_rng.choice(n_paths_24, size=-100, replace=False)` would raise `ValueError` (numpy rejects negative size). So out-of-range alpha doesn't silently corrupt — it crashes deep in the loop with a less informative error. For `alpha=1.5`: `n_b_take = 1500`, `n_a_take = -500`, same crash mode.

**Recommendation: ACCEPT (option a).** The CLI default is α=0.5 (plan-mandated, locked) and there are no out-of-tree callers. The cost of fixing now is a 2-line check, but adding it would require either a touched-file scope creep on the cleanup commit (which currently has the appealing property of being "deletions + tests only") or a separate amendment commit. Park as a follow-up — V6_TBD entry: "Re-add `0.0 <= alpha <= 1.0` validation at the top of `ensemble_one_fold` when `ensemble_paths.py` is next edited for substantive reasons; current crash mode is loud but unhelpful." This is genuinely low priority given the plan-locked α.

## New observations

No unmotivated changes. The diff is exactly: (1) deletion of `mix_paths` (lines 51-110) + `mix_ratios` (lines 112-140) from `scripts/v5/ensemble_paths.py`, (2) complete rewrite of the test file from a mix of helper-unit + end-to-end tests to all-end-to-end tests with the `_make_mini_run` helper. No default values changed, no functions renamed, no logging dropped, no production code paths altered. The `import` block in the test file correctly drops `mix_paths`/`mix_ratios` from `from v5.ensemble_paths import ...`. No stale references anywhere in the repo (grep `mix_paths|mix_ratios` across `*.py`/`*.md`/`*.yaml` returns nothing).

Commit message has no AI attribution, in line with project policy.

## Recommendation to parent
Open PR; add a one-line entry to `docs/analog_mc/V6_TBD.md` (or note in the PR body for the next maintainer) about re-plugging alpha validation when the file is next touched.
