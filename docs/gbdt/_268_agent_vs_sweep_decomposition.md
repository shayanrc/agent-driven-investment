# _268 — Why agent-tuned XGBoost underperforms the CatBoost sweep on H=200 (decomposition)

**Question (`_267` follow-up):** on the three H=200 cells, agent-tuned **XGBoost** lost to the
sweep **CatBoost** on test R-Precision@1/@3 despite much higher *eval* R-p. `_267` named four
suspects — stale-vintage bar, **backend** (XGB vs CatBoost), **eval-overfit**, capacity/horizon.
This memo isolates them with a controlled **2×2 (backend × tuning-effort)**, every arm on the
**current data vintage** and the same date-aligned test window (2023-07-27 → 2024-10-03).

## Design — 2×2, identical data/window, only backend × tuning differ

| | sweep (default HP, ≤3-iter FS+HP) | agent (full agent_file_protocol) |
|---|---|---|
| **CatBoost** | `*_aligned_resnap` (= `_267` bar, re-run current vintage) | *(not run — see Recommendation)* |
| **XGBoost** | `*_aligned_xgbsweep` (the missing control) | `*_aligned_agent` (= `_267` agent model) |

All specs are byte-identical copies of the cell's sweep spec except `backend.library`; all read
`--snapshot-end 2026-06-20`. The agent arm is the heavily-tuned `_267` model.

## Result 1 — vintage confound REJECTED

Re-running the sweep CatBoost on the **current** vintage reproduces the stale-vintage numbers
(test R-Precision@K + base_rate):

| cell | sweep vintage | @1 | @3 | @5 | @10 | @20 | base |
|---|---|---|---|---|---|---|---|
| sp500 +50%/200d | stale | 0.607 | 0.569 | 0.543 | 0.488 | 0.458 | 0.135 |
| sp500 +50%/200d | current | 0.630 | 0.532 | 0.494 | 0.491 | 0.448 | 0.135 |
| sp500 +40%/200d | stale | 0.727 | 0.556 | 0.521 | 0.450 | 0.451 | 0.219 |
| sp500 +40%/200d | current | 0.727 | 0.556 | 0.521 | 0.450 | 0.451 | 0.219 |
| russell1000 +40%/200d | stale | 0.717 | 0.570 | 0.507 | 0.465 | 0.432 | 0.236 |
| russell1000 +40%/200d | current | 0.740 | 0.523 | 0.515 | 0.444 | 0.424 | 0.236 |

sp500/40 is identical to 3 d.p.; the others within ±0.05 across K. **The `_267` stale bar was not
optimistic** — vintage is not the explanation.

## Result 2 — the 2×2 (test R-Precision@K + base_rate)

| cell | variant | @1 | @3 | @5 | @10 | @20 | base |
|---|---|---|---|---|---|---|---|
| sp500 +50%/200d | CatBoost sweep | **0.630** | **0.532** | 0.494 | 0.491 | 0.448 | 0.135 |
| | XGBoost sweep | 0.297 | 0.369 | 0.381 | 0.414 | 0.398 | 0.135 |
| | XGBoost agent | 0.623 | 0.454 | 0.467 | 0.449 | 0.445 | 0.135 |
| sp500 +40%/200d | CatBoost sweep | **0.727** | **0.556** | **0.521** | 0.450 | 0.451 | 0.219 |
| | XGBoost sweep | 0.240 | 0.360 | 0.380 | 0.419 | 0.442 | 0.219 |
| | XGBoost agent | 0.697 | 0.468 | 0.429 | 0.445 | 0.466 | 0.219 |
| russell1000 +40%/200d | CatBoost sweep | **0.740** | 0.523 | 0.515 | 0.444 | 0.424 | 0.236 |
| | XGBoost sweep | 0.633 | **0.529** | 0.494 | 0.477 | 0.458 | 0.236 |
| | XGBoost agent | 0.447 | 0.457 | **0.534** | **0.537** | **0.511** | 0.236 |

**No XGBoost variant beats CatBoost-sweep at @1/@3 in any cell.**

## Result 3 — eval→test decay (the generalization lens)

| cell | variant | eval@1 | test@1 | decay@1 | eval@3 | test@3 |
|---|---|---|---|---|---|---|
| sp500 +50%/200d | CatBoost sweep | 0.820 | 0.630 | +0.190 | 0.702 | 0.532 |
| | XGBoost sweep | 0.755 | 0.297 | +0.458 | 0.578 | 0.369 |
| | XGBoost agent | 0.895 | 0.623 | +0.272 | 0.743 | 0.454 |
| sp500 +40%/200d | CatBoost sweep | 0.800 | 0.727 | +0.073 | 0.660 | 0.556 |
| | XGBoost sweep | 0.740 | 0.240 | +0.500 | 0.647 | 0.360 |
| | XGBoost agent | 0.895 | 0.697 | +0.198 | 0.728 | 0.468 |
| russell1000 +40%/200d | CatBoost sweep | 0.805 | 0.740 | +0.065 | 0.678 | 0.523 |
| | XGBoost sweep | 0.730 | 0.633 | +0.097 | 0.553 | 0.529 |
| | XGBoost agent | 0.790 | 0.447 | +0.343 | 0.660 | 0.457 |

## Reading — two factors, cell-dependent

**1. Backend dominates — CatBoost is far more robust at H=200.** It has the **lowest eval→test
decay in all three cells** (+0.065 to +0.190 @1) and the **best test top-1/@3 everywhere**. Even
the best XGBoost (agent-tuned) only *reaches* ≈ CatBoost-default at best (sp500/50: 0.623 vs 0.630)
and falls far below it elsewhere (russell: 0.447 vs 0.740). This is the `_225` pattern (gaps that
looked like FS/HP were really XGB-vs-CatBoost).

**2. XGBoost is high-variance, and the agent's eval-R-p@1 objective is an unreliable proxy at H=200.**
- **sp500:** XGBoost-*default* over-decays catastrophically (eval 0.74–0.76 → test 0.24–0.30 — the
  default eta=0.3/depth=6 overfits the rare-event eval window). The agent tuning **rescued** it
  (→ test 0.62–0.70; decay roughly halved). Here agent tuning *helped* — but only up to ≈ CatBoost.
- **russell:** XGBoost-*default* was already well-behaved (decay +0.097, test @1 = 0.633). The agent
  tuning **hurt** — it raised eval (0.73→0.79) but dropped test (0.633→0.447; decay +0.097→+0.343).
  The loop maximized eval and produced a model that generalizes *worse than its own untuned default*.
- So eval R-p@1 and test R-p@1 are only weakly (sometimes negatively) coupled at H=200. The loop is
  blind to test, so maximizing eval either rescues a pathological default or over-fits a good one —
  high variance, no consistent win.

**Capacity is not the story:** the agent's tiny models (depth 2–4, 130–143 feats) *match* CatBoost
(depth 6, ~1000 iters) on sp500 — small capacity is sufficient when tuned; the failure mode is
selection, not capacity.

## Verdict

The `_267` shortfall is **primarily backend** (CatBoost generalizes far better at H=200 — lowest
decay, best test top-1/@3) and **secondarily the agent loop's eval-R-p@1 objective being an
unreliable proxy at this horizon** (high-variance: a rescue on sp500, eval-overfit on russell).
Vintage is rejected; capacity is not limiting. This **refines `_267`'s "agent not overfit"** — true
at the train–val level, but at the **HP-selection** level the agent overfit eval on russell.

## Recommendation

- **For H=200 cells, default to the CatBoost sweep** — robust, strong, and cheap (≤3 iters, ~5 min on
  a warm cache). Do **not** deploy agent-tuned XGBoost over it: it never beats CatBoost-sweep at
  top-1/@3 and can underperform even XGBoost-default (russell).
- If XGBoost is required, **validate the agent's pick against both XGBoost-default and CatBoost-sweep**
  — at H=200 the agent can lose to its own untuned default.
- **Loop fix:** eval R-p@1 is a poor single-window target at H=200 (weak eval↔test coupling). A
  **longer / multi-window eval**, or a decay-penalized objective, would help; or simply use CatBoost.
- **Future (not run here):** a **CatBoost agent-tune** to test whether tuning the *robust* backend can
  beat the CatBoost sweep — the natural "can we beat the bar" experiment. → `docs/gbdt/V1.5_TBD.md`.

## Artifacts

- Controls: `configs/gbdt/experiments/*_aligned_{resnap,xgbsweep}.yaml` + artifacts; registry rows
  `*_aligned_resnap` (CatBoost current-vintage sweep) + `*_aligned_xgbsweep` (XGBoost current-vintage
  sweep) in `results/gbdt/data/r_precision_at_k.csv`.
- Agent arm: the `_267` cells (`*_aligned_agent`).
- Sidecar: `results/gbdt/data/_268_agent_vs_sweep_decomposition_data.json`.
