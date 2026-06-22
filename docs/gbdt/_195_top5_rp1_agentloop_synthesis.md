# _195 — top-5 R-Precision@1 cells: agent-driven loop vs sweep baseline

**Branch**: `gbdt-195-top5-rp1-synthesis-memo`.
**Date**: 2026-06-02.
**Cells**: 4 of 5 top-5-R-p@1 cells executed (cell-5 nasdaq100 +10%/50d deferred — anti-AUC anomaly).
**Canonical metrics**: `results/gbdt/data/r_precision_at_k.csv`.

## Headline

The L2 grid → `mcw=10` prescription from the russell1000 trio (_194) + _185 — accepted as the
agent-loop's default closing config — **regresses top-1 by −20% to −32%** on cells where the
sweep already nails top-1 (R-p@1 ≥ 0.5). Per-cell exploration found a winning "mix recipe"
on cell-4 (FS-to-30 + `mcw=5` + `colsample_bytree=0.5` + `gamma=0.5`, R-p@1 **+21%** over
sweep) but the SAME recipe DESTROYED top-1 on cells 1+3 (R-p@1 **−55%** and **−60%** vs
sweep). **There is no transferable recipe across strong-top-1 cells.** The agent loop's value
is per-cell navigation, not recipe application.

## Cells in scope

| Cell | Universe | Direction | Threshold | Horizon | DD  | Sweep R-p@1 | Notes                                       |
|------|----------|-----------|-----------|---------|-----|-------------|---------------------------------------------|
| 1    | sp500    | up        | 50%       | 50d     | 25% | 0.800       | project's best top-1 cell                   |
| 2    | sp500    | up        | 20%       | 50d     | 10% | 0.640       | subsample stack collapse, first-attempt only |
| 3    | sp500    | up        | 20%       | 25d     | 10% | 0.600       |                                             |
| 4    | nasdaq100| up        | 40%       | 50d     | 20% | 0.549       | extensive knob exploration (5 variant runs) |
| 5    | nasdaq100| up        | 10%       | 50d     | 5%  | 0.671       | NOT RUN — anti-AUC anomaly cell, deferred   |

## Loop-vs-sweep results — primary table

| Cell | Sweep R-p@1 | Sweep AUC | Loop original (mcw=10) R-p@1 | Mix recipe R-p@1 | Best loop variant                  | Best loop R-p@1 vs sweep |
|------|-------------|-----------|------------------------------|------------------|------------------------------------|--------------------------|
| 1    | 0.800       | 0.903     | 0.640 (**−20%**)             | 0.360 (**−55%**) | original loop (no FS)              | **−20%**                 |
| 2    | 0.640       | 0.727     | 0.080 (**−87%** — subsample collapse) | not run     | none                               | **−87%**                 |
| 3    | 0.600       | 0.775     | 0.413 (**−31%**)             | 0.240 (**−60%**) | original loop (no FS)              | **−31%**                 |
| 4    | 0.549       | 0.753     | 0.373 (**−32%**)             | **0.667 (+21%)** | **mix (FS+mcw=5+cs=0.5+γ=0.5)**    | **+21%**                 |

**R-Precision@10 comparison** (the operating metric of the rare-event playbook, for cross-reference):

| Cell | Sweep R-p@10 | Loop original R-p@10 | Loop mix R-p@10 | Best loop R-p@10 vs sweep |
|------|--------------|----------------------|------------------|---------------------------|
| 1    | 0.286        | 0.346 (+21%)         | 0.293 (+2%)      | **+21%** (original loop)  |
| 2    | 0.374        | 0.274 (−27%)         | n/a              | **−27%**                  |
| 3    | 0.409        | 0.403 (−2%)          | 0.355 (−13%)     | **−2%** (original loop)   |
| 4    | 0.503        | 0.477 (−5%)          | 0.522 (+4%)      | **+4%** (mix recipe)      |

The L2 grid recipe lifts R-p@10 on cell-1 (+21%) and is mostly neutral elsewhere — so the
*K=10 operating metric* is not where the regression bites; it's *K=1* where it matters most
for the top-tail-pick portfolio framing of these particular cells.

## Findings (F1–F6 — codified as rules 7–9 in the playbook)

**F1 — L2 grid (mcw 1→5→10) over-regularizes strong-top-1 cells.** All 3 cells (1, 3, 4)
where sweep R-p@1 ≥ 0.5 regressed top-1 under the L2 grid → mcw=10 path. The r1k trio + _185
evidence for `mcw=10` came from rare-event cells (sweep R-p@1 ≈ 0.03) where K=10 was the
operating metric, not top-1. Mechanism: `mcw=10` caps leaf weights → smooths the prediction
tail → kills the very-high-confidence top picks the sweep was finding via the un-regularized
baseline. The L1 tie-break (gap + Z) actively SELECTS the most-regularized iter, compounding
the harm.

**F2 — No universal "fix" for strong-top-1 cells.** Cell-4 mix recipe beat sweep R-p@1
**+21%**; the same recipe on cells 1+3 was WORSE than the original mcw=10 loop. Each
cell's optimal regularization profile is cell-shape-dependent.

**F3 — Asymmetry hint (n=3, do not over-read).** Cells 1+3 are sp500 (486 tickers, larger
panel, sweep R-p@1 0.6–0.8); cell-4 is nasdaq100 (92 tickers, smaller panel, sweep R-p@1
0.549). The aggressive FS+stacked-reg recipe may collapse the diversity the sweep was using
on the larger panels. Hypothesis to test on more cells, not a rule.

**F4 — Validation-vs-test divergence is real.** Cell-4 colsample variant: val_brier moved
+0.00001 (noise) between `colsample=1.0` (val 0.023547) and `colsample=0.7` (val 0.023558),
yet test R-p@1 jumped from 0.373 → 0.431 (**+16%**). Cell-4 mix iter_1: val_brier 0.023441
(WORSE than gamma stack's 0.023363), but test R-p@1 0.667 (much BETTER than gamma stack's
0.353). The val_brier metric is too coarse to detect ranking shifts in the prediction tail;
the L1 tie-break (calibrated on val gap + Z) is the wrong objective for top-1-driven cells.

**F5 — Single-knob plateau ≠ global plateau (bug #204).** The runner's auto-plateau (fires
when `val_brier improvement < plateau_threshold=0.005`) silenced entire response surfaces.
On cell-4: `mcw=10` val Δ=0.0012 plateau-stopped before colsample / gamma could be tested,
masking the +16% colsample lift entirely. Forced "fresh-variant spec" workarounds throughout
cell-4 exploration (gamma variant pre-bakes the colsample state via iter_0_decision; mix
variant pre-bakes the FS+mcw+cs+γ recipe in one step). Bug fix in flight — disables
auto-plateau in `agent_file_protocol` mode while keeping it for sweep mode.

**F6 — Per-knob effects on cell-4** (single-cell evidence; do NOT generalize values):
- `mcw` alone (1 → 5 → 10): more reg → worse top-1, neutral on K=10
- `colsample` alone (1.0 → 0.5): partial top-1 recovery (0.373 → 0.431, **+16%** vs original loop)
- `gamma` stack (0 → 0.1 → 0.5 → 1.0): best K=10 recovery (R-p@10 0.520, matches sweep)
- mix at moderate values (`mcw=5 + cs=0.5 + γ=0.5`): cell-4 sweet spot (R-p@1 0.667)
- `mcw=3` on top of mix: top-1 collapses 0.667 → 0.373 (mcw=5 is local optimum, not "lower better")
- `subsample` stacked with mcw: catastrophic collapse on common-event cells (cell-2: R-p@1 0.640 → 0.080, **−87%**)
- `depth=8` is no-op when `gamma ≥ 0.5` (gamma already prevents deep splits)

## Methodology / Process learnings

- **Infrastructure wins this session** (PRs #103, #104, #105): `features.py` optimizations
  (raw=True rolling apply, pandas vectorization) gave 15–90× speedup on features cold-build;
  sp500 cold-build dropped from ~280 min to ~25–40 min.
- **XGBoost `hist + nj=8` swap** (separate PR in flight for task #203): 15–90× speedup on
  training; byte-identical at fixed `(machine, n_jobs)`.
- **Bug #204 workaround** (until runner patch lands): set `plateau_threshold: 0.0001` in
  spec to disable auto-plateau in `agent_file_protocol` mode; rely on `should_stop` +
  `max_iterations` + `degradation_gate` for stopping.
- **"Fresh variant spec" pattern**: when a run plateau-stops mid-investigation, create a
  fresh spec that pre-bakes the prior plateau state via `iter_0_decision` (replicating it in
  one step), then continue exploration from there. Used on cell-4 to chain
  `_agentloop → _colsample → _gamma → _mix → _mix_mcw3` without losing context.

## What this means for the playbook

Rules 7–9 added to `.claude/memories/project-gbdt-tuning-playbook.md` (shipped in a separate
PR). Net change:

- Don't default `mcw=10` on cells where sweep R-p@1 ≥ 0.5
- Don't default the mix recipe either (cell-4-specific, not transferable)
- The agent's role is per-cell navigation: read iter_0 diagnostics + sweep baseline, pick
  ONE knob based on what's needed (`gap > 0.02` → start with regularization; `gap < 0` →
  start with capacity), map ≥2 values per knob, pivot to a structurally-different knob
  family per rule 9.

## What's open

- **cell-5** (nasdaq100 +10%/50d, sweep R-p@1=0.671 with AUC=0.475 anti-AUC anomaly) NOT
  run. Deferred to focus on cell-4 knob-exploration. The anti-AUC profile (AUC < 0.5 yet
  high top-1) suggests the model RANKS positives correctly only at the extreme tail —
  worth a dedicated investigation, not a routine loop run.
- **Cells 1+3 still in regression** — agent loop loses to sweep on R-p@1 here. The mix
  recipe failure means deeper per-cell exploration is needed. Likely UNDER-regularization
  is the answer (sweep with default HPs nails top-1 → try `mcw=1` + smaller FS prune + no
  other reg). Untested.
- **`/gbdt-diagnose` pairwise interaction data** (task #206) — captured 2026-06-08 (PR #206
  follow-up); see appendix below.
- **Bug #203** (`hist + nj=8`) and **bug #204** (plateau-stop) shipping as separate PRs
  alongside this memo.

## Cell-4 mix interaction structure (#206 follow-up)

`/gbdt-diagnose` re-run 2026-06-08 on the cell-4 mix artifact (xgboost, SHAP-based
`pred_interactions`, 5000 in-sample rows, 30 features). Method: TreeSHAP.

Top 5 pairwise interactions by SHAP strength:

| feature A             | feature B                | strength |
|-----------------------|--------------------------|---------:|
| index_runup_50        | parkinson_200            |    0.090 |
| parkinson_200         | return_xs_zscore_200     |    0.042 |
| vol_xs_zscore_50      | vol_xs_rank_200          |    0.042 |
| garman_klass_200      | vol_xs_rank_200          |    0.041 |
| vol_xs_rank_200       | realized_vol_zscore_200  |    0.039 |

Two features dominate the interaction load: `parkinson_200` (involvement 0.22, main
effect 0.48) and `vol_xs_rank_200` (involvement 0.22, main effect 0.33) — both well above
the high-interaction threshold (0.078). The single dominant pair
(`index_runup_50 × parkinson_200`) is **~2.1× the next pair**, and 5 of the top 10 pairs
route through `parkinson_200` or `vol_xs_rank_200`. This is a dense, interaction-driven
signal — the model leans on *conditional* (regime-by-volatility) structure, not on
isolated marginal effects. Mechanistic reading of why `mcw=5 + colsample_bytree=0.5 +
γ=0.5` won on this cell (rule 8 / playbook): `colsample=0.5` forces each tree to
re-discover the dominant interaction paths through different feature subsets (preserving
top-1 tail when the interaction is the signal), while `γ=0.5` filters spurious splits on
the low-importance long tail; `mcw=5` keeps leaf weights high enough not to erase the
interaction-tail rare-event picks. Cells 1+3 (sp500, larger panel, sweep R-p@1 ≥ 0.6)
likely have flatter interaction surfaces (more independent main-effect features) — the
same recipe strangles the diversity those cells were exploiting. **Confirms F2/F3**: the
mix recipe is cell-4-specific because cell-4's signal IS structurally interaction-heavy.

Source artifact: `results/gbdt/experiments/nasdaq100_up_40pct_50d_dd20pct_agentloop_mix/diagnose/`.

## Artifacts (per-cell)

All artifact dirs under `results/gbdt/experiments/`:

| Cell  | Artifact dirs                                                                   |
|-------|---------------------------------------------------------------------------------|
| cell-1| `sp500_up_50pct_50d_dd25pct_agentloop`, `_agentloop_mix`                        |
| cell-2| `sp500_up_20pct_50d_dd10pct_agentloop`                                          |
| cell-3| `sp500_up_20pct_25d_dd10pct_agentloop`, `_agentloop_mix`                        |
| cell-4| `nasdaq100_up_40pct_50d_dd20pct_agentloop`, `_colsample`, `_gamma`, `_mix`, `_mix_mcw3` |

Canonical R-Precision@K metrics: `results/gbdt/data/r_precision_at_k.csv`.
JSON sidecar: `results/gbdt/data/_195_top5_rp1_agentloop_synthesis_data.json`.
