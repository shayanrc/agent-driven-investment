# _269 — CatBoost agent-tune vs the CatBoost sweep on H=200 (V1.5_TBD #1)

**Question (`_268` follow-up):** `_268` found the **backend** (CatBoost) dominates XGBoost at H=200 and
left one cell of the 2×2 unfilled — *CatBoost + full agent tuning*. If XGBoost was the bottleneck,
can agent-tuning the **robust** backend (CatBoost) beat its own cheap ≤3-iter sweep? **Answer:
high-variance — one decisive win (sp500 +50%/200d, test @1 0.930 vs the sweep's 0.630) and two @1
losses (russell, sp500 +40%/200d) where the eval-optimized config did not transfer and lost to the
sweep's plain defaults.** The cheap CatBoost sweep remains the robust H=200 default; tuning CatBoost
does **not** reliably beat it — only sp500/50 broke through, and that one spectacularly.

## Setup

3 specs `configs/gbdt/experiments/*_aligned_cbagent.yaml`: `backend.library: catboost`, V1.3 Option B
scout + fs_prefit + agent_file_protocol, same date-aligned cells/window as `_267`/`_268`
(test 2023-07-27 → 2024-10-03), `--snapshot-end 2026-06-20`. The scout translates XGBoost-vocab HP →
CatBoost (`max_depth→depth`, `eta→learning_rate`, `colsample_bytree→rsm`, `min_child_weight→
min_data_in_leaf`, `alpha→l2_leaf_reg`; `gamma` dropped). Each driven solo. Final HPs were nearly
identical across cells (`learning_rate=0.05, depth=6`, FS-prefit to ~52 feats; only `l2_leaf_reg`
∈ {0.3, 0.5, 3.0} and `rsm` ∈ {0.5, 1.0} differed) — so the win/loss split below is **not**
HP-driven.

## Results — CatBoost-agent vs CatBoost-sweep vs XGBoost-agent (test R-Precision@K + base_rate)

| cell | model | @1 | @3 | @5 | @10 | @20 | base |
|---|---|---|---|---|---|---|---|
| sp500 +50%/200d | CatBoost sweep | 0.630 | 0.532 | 0.494 | 0.491 | 0.448 | 0.135 |
| | **CatBoost agent** | **0.930** | **0.617** | **0.555** | 0.487 | 0.460 | 0.135 |
| | XGBoost agent | 0.623 | 0.454 | 0.467 | 0.449 | 0.445 | 0.135 |
| russell +40%/200d | CatBoost sweep | **0.740** | 0.523 | 0.515 | 0.444 | 0.424 | 0.236 |
| | **CatBoost agent** | 0.567 | 0.548 | 0.515 | 0.455 | 0.422 | 0.236 |
| | XGBoost agent | 0.447 | 0.457 | 0.534 | 0.537 | 0.511 | 0.236 |
| sp500 +40%/200d | CatBoost sweep | **0.727** | 0.556 | 0.521 | 0.450 | 0.451 | 0.219 |
| | **CatBoost agent** | 0.557 | 0.584 | 0.493 | 0.449 | 0.452 | 0.219 |
| | XGBoost agent | 0.697 | 0.468 | 0.429 | 0.445 | 0.466 | 0.219 |

Per cell, CatBoost-agent vs the CatBoost-sweep at @1: sp500/50 **+0.300 (WIN)**, russell **−0.173
(LOSS)**, sp500/40 **−0.170 (LOSS)**. All three won @3 marginally (+0.085, +0.025, +0.028).

## Eval→test decay (the whole story)

| cell | eval@1 | test@1 | decay@1 | eval@3 | test@3 |
|---|---|---|---|---|---|
| sp500 +50%/200d | 0.845 | **0.930** | **−0.085** (improved!) | 0.738 | 0.617 |
| russell +40%/200d | 0.805 | 0.567 | +0.238 | 0.717 | 0.548 |
| sp500 +40%/200d | 0.805 | 0.557 | +0.248 | 0.688 | 0.584 |

**All three had near-identical eval@1 (0.805–0.845), but test@1 ranged 0.557 → 0.930.** eval@1 simply
does not predict test@1 at H=200 — the difference between the win and the losses is **entirely** the
eval→test decay: on sp500/50 the signal *strengthened* into the 2023-H2→2024 test window (test 0.930
> eval 0.845), while on russell/sp500/40 the eval top-1 over-fit and collapsed (−0.24). The agent,
which only sees eval, cannot tell these apart — so optimizing eval@1 is effectively a coin-flip on
test@1 at this horizon.

## Reading

- **Tuning CatBoost is high-variance, not a reliable improvement.** 1/3 cells (sp500/50) cleared the
  sweep decisively; 2/3 lost @1 to the cheap ≤3-iter defaults. The CatBoost-agent did beat the
  *XGBoost*-agent at @1 on 2/3 (sp500/50, russell) — switching to the robust backend helps — but
  still lost to the CatBoost *sweep* on those same 2/3.
- **Same disease as `_268`, milder.** The eval→test-decay variance that sank agent-tuned XGBoost also
  affects agent-tuned CatBoost; CatBoost's decay is generally smaller, but on russell/sp500/40 the
  agent's eval-chosen config still decayed enough to fall below the sweep. The sweep's plain,
  un-FS'd, un-subsampled default held top-of-book better through the test window on those cells.
- **No HP tell.** The winning (sp500/50) and losing (russell, sp500/40) configs are HP-twins
  (`lr=0.05, depth=6, ~52 feats`); the outcome was set by the cell/window, not the knobs. So there is
  no "tune it this way" recipe — the upside is real but unpredictable ex-ante.
- The sp500/50 @1=0.930 is verified (independently recomputed: 279/300 days the top-1 pick is a true
  future +50%/200d mover) — a genuine, large win, not an artifact.

## Verdict

Agent-tuning the robust backend (CatBoost) at H=200 is **high-variance: capable of a large win
(sp500/50) but unreliable (2/3 @1 losses to the cheap sweep).** It is **not** a dependable improvement
over the ≤3-iter CatBoost sweep, which remains the robust H=200 default. This closes the `_267`→`_268`
arc: the H=200 shortfall was never just "XGBoost vs CatBoost" — it is the **eval-R-p@1 objective being
an unreliable single-window proxy at long horizons**, which afflicts *both* backends. Switching to
CatBoost narrows the gap and occasionally wins big, but does not solve it.

## Recommendation

- **Keep the CatBoost ≤3-iter sweep as the H=200 baseline / deployment default.**
- **Agent-tuning CatBoost is opportunistic, not default:** the sp500/50 upside (+0.30 @1) is real, so
  it is worth trying per-cell — but **always validate the agent's pick against the sweep on the test
  window** before trusting it; it can lose @1 to plain defaults (russell, sp500/40).
- **The real fix is the objective, not the backend:** a multi-window / decay-penalized eval (V1.5_TBD
  #2) would address the weak eval↔test coupling for both XGBoost and CatBoost. Promote that before
  any further H=200 agent-tuning push.

## Artifacts

- 3 cbagent cells committed: `704f6bd` (russell), `ce06193` (sp500/50), `0674c26` (sp500/40); registry
  rows `*_aligned_cbagent`.
- Sidecar: `results/gbdt/data/_269_catboost_agent_tune_data.json`.
