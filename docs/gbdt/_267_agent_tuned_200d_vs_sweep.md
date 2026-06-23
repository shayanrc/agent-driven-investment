# _267 — Agent-tuned date-aligned counterparts of the 3 un-tuned top-200d sweep cells

**Headline:** The three H=200 cells that topped the R-Precision@3 registry on **sweep-grade
CatBoost** (≤3 iters) but had **no agent-tuned counterpart** — `russell1000 +40%/200d`,
`sp500 +50%/200d`, `sp500 +40%/200d` — were re-built as fully **agent-tuned**
(`agent_file_protocol`, V1.3 Option B scout+combine, **XGBoost**) date-aligned models on the
**current data vintage**. **Agent-tuning did NOT beat the sweep on the headline test metrics
(R-Precision@1/@3) on any of the three**, despite much higher *eval* R-Precision (≈0.79–0.90 @1).
The driver is a steep, sometimes K-dependent **H=200 eval→test decay**: the agent loop maximizes
*eval* R-p@1, which is sharp on eval but does not survive to the held-out test window. The agent
models are nonetheless the **trustworthy, current-vintage, reproducible** counterparts (the sweep
cells failed `_266`'s forward-OOS self-check). Whether the gap is the **backend** (XGB vs CatBoost),
the **stale-vintage sweep bar**, **eval-overfit**, or **capacity** is the subject of the queued
`_268` investigation.

## Setup

- 3 specs `configs/gbdt/experiments/{russell1000_up_40pct_200d_dd20pct,sp500_up_50pct_200d_dd25pct,sp500_up_40pct_200d_dd20pct}_aligned_agent.yaml`:
  **xgboost**, V1.3 Option B **scout** (35 single-knob fits) + **fs_prefit** cliff-cut (1%) +
  **agent_file_protocol** (max 16 iters, `plateau_threshold 0.0001` per bug #204),
  `conditional_isotonic`. Date-aligned H=200 block (anchor **2018-01-01**, `test_rows 300`,
  test window **2023-07-27 → 2024-10-03**), `--snapshot-end 2026-06-20`.
- Each cell was driven **solo by a dedicated data-scientist sub-agent** (one sub-agent = sole
  driver per cell; concurrent drivers + a duplicate launch OOM'd the box once early on — the
  single-driver + double-launch-guard discipline is in the run log).
- **Comparison bar** = the sweep companion `*_aligned` (CatBoost, ≤3 iters) test R-p from the
  registry. **Caveat (vintage confound):** those sweep numbers are **stale-vintage** — the cells
  failed `_266`'s forward-OOS self-check (their training feature-matrix / data vintage no longer
  reproduces), so the comparison is *current-vintage agent* vs *stale-vintage sweep*. Resolving
  this is the first task of `_268`.

## Results — agent (XGBoost, current vintage) vs sweep (CatBoost, stale vintage)

Test window (identical per cell → base_rate identical within a cell). Raw R-Precision@K + base_rate:

| cell | model | base | @1 | @3 | @5 | @10 | @20 | AUC |
|---|---|---|---|---|---|---|---|---|
| sp500 +50%/200d | sweep | 0.135 | 0.607 | 0.569 | 0.543 | 0.488 | 0.458 | — |
| sp500 +50%/200d | **agent** | 0.135 | 0.623 | 0.454 | 0.467 | 0.449 | 0.445 | 0.72 |
| sp500 +40%/200d | sweep | 0.219 | 0.727 | 0.556 | 0.521 | 0.450 | 0.451 | — |
| sp500 +40%/200d | **agent** | 0.219 | 0.697 | 0.468 | 0.429 | 0.445 | 0.466 | 0.70 |
| russell1000 +40%/200d | sweep | 0.236 | 0.717 | 0.570 | 0.507 | 0.465 | 0.432 | 0.66 |
| russell1000 +40%/200d | **agent** | 0.236 | 0.447 | 0.457 | 0.534 | 0.537 | 0.511 | 0.68 |

Agent eval R-Precision@K (the in-loop optimization target), for the decay comparison:

| cell | base | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|---|
| sp500 +50%/200d | 0.107 | 0.895 | 0.743 | 0.690 | 0.572 | 0.462 |
| sp500 +40%/200d | 0.176 | 0.895 | 0.728 | 0.628 | 0.543 | 0.486 |
| russell1000 +40%/200d | 0.195 | 0.790 | 0.660 | 0.594 | 0.517 | 0.472 |

Final agent HP (non-default knobs) / feature count: sp500/50 `eta0.2 d4 γ0.5 cs0.5` / 143;
sp500/40 `d3 eta0.1 mcw5 α0.1 γ0.1` / 143; russell/40 `cs1.0 γ0.5` / 130. All converged
on the V1.3 combine winner via `agent_should_stop` + the `v14_val_flat_eval_rp1` finalizer
(selects by eval R-p@1 when val_brier is flat within the tie-band — NOT the val_brier argmin).

## Reading

- **Headline R-p@1/@3: the sweep wins or ties all three.** @1: sp500/50 agent +0.016 (a wash),
  sp500/40 sweep +0.030, russell/40 sweep **+0.270**. @3: sweep wins all three (+0.115, +0.088,
  +0.113).
- **Deep tail (K≥5) is mixed-to-agent.** sp500/50 sweep still wins; sp500/40 ≈ tie with the agent
  ahead at @20; **russell/40 the agent wins @5–@20** (+0.03 to +0.08) with higher test AUC.
- **Eval→test decay is the dominant story.** Agent eval @1 ≈ 0.79–0.90 collapses to test @1
  0.45–0.70. Test AUC decays far less than top-1 R-p (e.g. russell eval 0.704 → test 0.681, +
  above the sweep's 0.662) — so the *discrimination* is real, but the *top-of-book ordering* is
  regime-sensitive at this horizon. russell is the cleanest example: uniform decay would have kept
  its eval shape, but instead the top-1 evaporates (0.79→0.45) while K≥5 actually *improves*.
- The agent models are **not overfit** (negative train-val gap every iter; 130–143 low-complexity
  features) — the eval→test gap is a *generalization/regime* effect, not a train-overfit one, and
  more iterations would not close it (the loop already plateaued across ≥2–3 knob families per cell).

## Verdict

For these H=200 cells, agent-tuning XGBoost via the eval-R-p@1 loop produces a **clean, reproducible,
current-vintage** model but **does not beat the (stale) sweep CatBoost on headline top-1/top-3 test
R-Precision**; the eval→test decay dominates. The agent models are preferable for *deployment*
(reproducible + current-vintage, where the sweep cells are not) and are competitive-to-better at
deeper K (K≥5) on 2/3, but the headline metric does not improve. This reaffirms the cross-experiment
**H=200 eval→test-decay prior** and motivates `_268`: decompose whether the shortfall is the
**backend** (XGB vs CatBoost), the **stale-vintage bar**, **eval-overfit**, or the **horizon** itself.

## Artifacts

- 3 agent cells committed: `17e8993` (sp500/50), `d36c45b` (sp500/40), `7820a87` (russell/40);
  registry rows in `results/gbdt/data/r_precision_at_k.csv`.
- Sidecar: `results/gbdt/data/_267_agent_vs_sweep_200d_data.json`.
- Follow-up: `_268` (why agent < sweep — backend / vintage / eval-overfit / capacity decomposition).
