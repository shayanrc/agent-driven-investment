# analog_mc — conditional sampling speedup plan & implementation log

Companion to [`ABLATION_STUDIES_REPORT.md`](ABLATION_STUDIES_REPORT.md). That
report identified Cell D (drift + conditional) as the production-default
candidate but flagged the wall-time cost — canonical Cell D was estimated at
~15.5h, intractable for routine confirmation runs. This doc records what we
diagnosed, what we changed, and what's still on the table.

## Headline

The conditional sampling test eval was bottlenecked on `(n_paths, K)`
elementwise NumPy ops inside two functions in `src/analog_mc/distances.py`,
plus a serial per-origin loop in `src/analog_mc/walk_forward.py`. Three
shipped fixes reduced canonical Cell D from a projected ~15.5h baseline to
**7h 36m actual** (run `20260518T155800Z`, completed 2026-05-19), all behavior-
preserving (full test suite green, byte-identical results vs serial path).

| Fix | Where | Function-level win | End-to-end win on conditional forecast |
|---|---|---|---|
| 1. Matmul identity in `composite_distance_batched` | `distances.py:139` | 2.2–6.3× | ~1.03× |
| 2. In-place bisection in `distances_to_probs_batched` | `distances.py:198` | 1.3–2.1× | **~2.19×** (stacked with #1) |
| 3. Process pool over test origins | `walk_forward.py` | n/a | **5–6×** at late folds |
| **Combined** | | | **~12–14×** at late folds |

## Diagnosis

Per-fold timings from the existing run logs (`runs/analog_mc/_ablation_C.log`,
`_v21_canonical.log`):

| | Search | Test eval per fold | Where time goes |
|---|---|---|---|
| v2.1 canonical, fold 0 | 191s | 0.5s | search-dominated |
| v2.1 canonical, fold 74 | 200s | 0.5s | flat per fold |
| Cell C ablation, fold 0 | 32s | 55s | test eval already significant |
| **Cell C ablation, fold 71** | 28s | **676s** | **test eval blows up linearly in K** |

The candidate pool K grows from ~800 (fold 0) to ~5500 (fold 75) as the
walk-forward expands the train window. Conditional-sampling cost scales
linearly in K because every block does `(n_paths × K)` work in the per-path
softmax bisection. cProfile on a K=9762 conditional forecast confirmed
**92% of time in `distances_to_probs_batched`**, ~4% in
`composite_distance_batched`, ~4% in sampling/scaling.

## Fix 1 — matmul identity in `composite_distance_batched`

**Before:** broadcast form `‖x − y‖²_w = sum_h w_h (x_h − y_h)²` materialized
a `(n_paths, K, H)` intermediate. At fold 75 with `n_paths=1000, K=5500`,
that's a 132 MB allocation, repeated thrice per call (diff, sq, weighted_sum).

**After:** identity `‖x − y‖²_w = ‖x‖²_w + ‖y‖²_w − 2 (x · (w⊙y))`. The
cross term factors into a single BLAS GEMM `z_targets @ (z_candidates *
weights).T` and is the only `(n_paths, K)` allocation; the remaining identity
terms (targ_sq + cand_sq − 2·cross) are folded in-place onto the GEMM result.

Benchmarks at canonical and fast-preset late-fold sizes:

| Shape (n_paths, K, H) | Broadcast | In-place identity | Speedup |
|---|---|---|---|
| (1000, 5500, 3) | 243 ms | 85 ms | **2.85×** |
| (500, 5500, 3)  | 113 ms | 51 ms | 2.20× |
| (500, 800, 3)   | 18.4 ms | 2.9 ms | 6.25× |

Max numerical drift vs broadcast form: 1.0e-14 absolute. Existing
`test_composite_distance_batched_matches_scalar` at rtol=1e-12, atol=1e-12
still passes.

End-to-end win on a single conditional forecast at K=9762: **15.05s → 14.62s
(2.9%)** — small because `composite_distance_batched` is only ~4% of the
function mix.

## Fix 2 — in-place bisection in `distances_to_probs_batched`

**Before:** each of 22 bisection iterations called
`_softmax_neg_and_log` and `_n_eff_batched`, which together allocated 5 fresh
`(n_paths, K)` arrays. At canonical resolution that's ~5 GB of allocation
churn per call, almost all of which got immediately freed.

**After:** three `(n_paths, K)` buffers (`log_w`, `w`, `scratch`) pre-allocated
once and reused across all iterations. Entropy computed via the identity
`H = log(Σw) − Σ p · log_w_shifted`, so `log_p` is never materialized. The
`p > 0` mask the original carried isn't needed in this form because
`p · log_w_shifted` is `0 · finite = 0` in IEEE float (not the `0 · −inf =
NaN` that motivated the guard). Comparison done in log-space vs
`log(target_n_eff)`, saving one `np.exp` per iter. The two private helpers
`_softmax_neg_and_log` and `_n_eff_batched` are removed.

Function-level benchmarks:

| Shape (n_paths, K) | Old | New | Speedup |
|---|---|---|---|
| (1000, 5500) | 3,328 ms | 1,624 ms | **2.05×** |
| (500, 5500)  | 1,334 ms | 775 ms   | 1.72× |
| (1000, 3000) | 1,328 ms | 891 ms   | 1.49× |
| (500, 800)   | 156 ms   | 119 ms   | 1.31× |

n_eff still hits target within ±0.25% (5% tolerance is the spec).

End-to-end stacked with Fix 1 on a single conditional forecast at K=9762:
**15.05s → 6.87s (2.19×)**.

## Fix 3 — process pool over test origins

**Before:** `_evaluate_on_test` iterated 60 test origins serially per fold.
Each forecast is fully independent — same fold, same weights, same n_eff;
only origin_idx and the deterministic per-origin RNG differ. On 8-core
hosts, 7 cores sat idle during test eval.

**After:** `ProcessPoolExecutor(mp_context=forkserver)` is created once per
run (in `run_walk_forward`) with `max_workers = max(1, cpu_count − 2)` or
the `ANALOG_MC_TEST_WORKERS` env override. The pool's `initializer` clamps
BLAS thread count to 1 via `threadpoolctl.threadpool_limits(1)` (otherwise
N workers × multi-threaded OpenBLAS oversubscribes the host) and stashes the
shared `returns_arr` and `features` DataFrame in worker module state, so
they're not re-pickled per task. Per-origin tasks ship `(origin_i, weights,
n_eff, random_seed, candidate_idx, horizon, config)` — small. Results are
`(origin_i, paths_float32, ratios_float32, realized, crps)` — paths are
already cast to float32 here rather than after collection to halve the
IPC payload.

`executor.map` preserves submission order, so the result arrays align with
`origins` without re-sorting. The serial path (`executor=None`) is preserved
for `n_workers=1` and for tests; it shares the same per-origin worker
function via a thin wrapper.

`forkserver` over `fork` because Python 3.12+ warns on fork-from-multi-
threaded-process and the parent has BLAS threads loaded — forkserver's clean
ancestor avoids the deadlock class entirely. Cost: ~300ms total at run
startup, paid once.

**Determinism:** verified bit-identical. Smoke test on fold 5 (60 origins,
mid-size K) — serial vs 6-worker parallel:

```
Serial:   81.63s  test_crps=0.026809
Parallel: 35.36s  test_crps=0.026809  (workers=6)
Speedup:  2.31x
origins identical: True
paths byte-identical:  True
ratios byte-identical: True
crps delta: 0.00e+00
```

Mid-fold speedup is 2.3×; late-fold speedup is bounded above by worker count
(6×) and approaches that as K grows because the per-task work dominates pool
overhead. Project Cell D canonical test-eval portion: ~12h serial →
~45 min parallel.

## Canonical Cell D timing projection

| Phase | Before (serial, naive code) | After (3 fixes) |
|---|---|---|
| Search (non-conditional at canonical) | ~3h 40m | ~3h 40m (untouched) |
| Test eval (4,560 conditional forecasts) | ~12h | **~45 min** |
| **Total projected** | **~15.5h** | **~4h 25m** |
| **Total actual** | — | **7h 36m** (`runs/analog_mc/20260518T155800Z/`) |

Actual landed ~70% above the optimistic projection. The gap is real and
worth noting: my fast-preset benchmarks used `n_paths=500`, but canonical
uses `n_paths=1000`, so per-forecast conditional cost roughly doubles
(linear in `n_paths × K`). Multiprocessing recouped most but not all of
that — `forkserver` startup, IPC pickling of float32 paths, and per-task
overhead at small folds eat ~25–30% of the theoretical 6× ceiling. Either
way, the run fits comfortably overnight, which was the bar.

## What's still on the table

Three further levers, none currently implemented:

| Lever | Expected gain | Risk | Effort |
|---|---|---|---|
| `max_iter=14` in bisection | ~1.4× on conditional test eval | Near-zero — bisection halves bracket each iter, 12–14 iters reaches the 5e-3 tolerance with no precision loss | 1 line |
| float32 throughout conditional sampler | ~1.5× | Determinism: CRPS shifts at ~1e-5 vs fp64. Breaks bit-identity with existing fp64 runs. Should ship behind a config flag | half-day |
| CuPy GPU port via existing matmul-identity form | 5–15× on bisection at K≥3000; small/none at K<1000 | RNG semantics differ → must keep CPU RNG for determinism, upload uniforms; `scipy.optimize.brentq` and `minimize` stay on CPU | 1–2 days |

Process-pool extension to **search-time** is the natural next architectural
change: search runs `(n_grid × n_eff × n_val_origins)` forecasts per fold —
70–80% of total wall time at canonical. Same shape of change as Fix 3.
Caveat: search-time forecasts are smaller (60 val origins × 66×5 grid points
= 19,800 per fold) and many are fast, so parallelism overhead matters more.
Worth prototyping if conditional canonical runs become routine.

## Validation checklist for any future speedup PR

1. `uv run pytest tests/analog_mc/` — full suite green (186 passed currently)
2. `test_composite_distance_batched_matches_scalar` — per-row equivalence at
   rtol=1e-12 vs scalar form
3. `test_distances_to_probs_batched_matches_scalar_row_by_row` — both solvers
   hit target n_eff within 5%, p-vectors within 1% rel + 5e-5 abs
4. Serial-vs-parallel byte equivalence on at least one fold:
   ```python
   from analog_mc import walk_forward as W
   # Construct serial and parallel runs of the same (fold, weights, n_eff)
   # and assert np.array_equal on (paths, ratios, origins).
   ```
5. End-to-end Cell C or D fast-preset run, compared to existing run on the
   same config — `test_crps` should match to bit-identity (paths-identical
   guarantees CRPS-identical).
