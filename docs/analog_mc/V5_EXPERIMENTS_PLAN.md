# analog_mc v5 — experiments plan

Successor to [`V4_RESULTS.md`](V4_RESULTS.md) (v4 closed: no experiment promoted) and grounded in [`V4_5_RESULTS.md`](V4_5_RESULTS.md) (mechanism inventory) + [`v4.5/_v4_5_8_v5a2_preview.md`](v4.5/_v4_5_8_v5a2_preview.md) + [`_v4_5_9_drawdown_sanity.md`](v4.5/_v4_5_9_drawdown_sanity.md) (de-risk previews).

v5 attacks the failure-anchor problem with a **stacked experiment plan**. The V4.5.8 preview proved no single experiment passes the v5 promotion bar alone — failures-recovered ceiling is 2/5 under any 50/50 ensemble of v2.4 + A2.1. The bar's recovery condition (≥3/5) requires V5.B's drawdown feature on top of V5.A.2's base.

## Starting state

| Item | State |
|---|---|
| Production default | v2.4 Cell-D-s30 (`configs/analog_mc/default.yaml`: `drift_mode=trailing_momentum, momentum_shrinkage=0.30, conditional_block_sampling=true, block_length=10`) |
| Canonical baseline run | `runs/analog_mc/20260520T045525Z` |
| v4 canonical runs | B1: `20260520T155220Z`; A2.1v1: `20260521T061730Z`; B5: `20260521T121025Z` |
| v4.5 diagnostic data | `results/analog_mc/data/v4_5_*.json` |
| Eval anchor set | 15 anchors in `results/analog_mc/data/fat_tail_eval_anchors.json` |
| Promotion bar | ≥3/5 V3.5 failure anchors recovered (90-band ≥45/60) AND ≤2/15 anchors regressing CRPS >5% — UNCHANGED from v4. |

## Scope and non-scope

**In scope.** Four mechanism-targeted experiments. Each gets a canonical walk-forward run (76 folds × 60 origins × 1000 paths) with its own config YAML, fat-tail panel (15 anchors), and per-experiment report. Total compute budget: ~36h (4 × 9h average).

**Out of scope.**
- Re-running v4 experiments (B1, A2.1, B5 already canonical).
- Multi-asset analysis (deferred to V2 module work).
- Position sizing / PnL / transaction costs (downstream).
- Anything not grounded in a specific V4.5 mechanism (M1–M5).

## V5 experiments — mechanism-grounded

In recommended priority/cost order. Each maps to mechanisms identified in [`V4_5_RESULTS.md`](V4_5_RESULTS.md) §"Mechanism inventory."

## Pre-canonical findings (V4.5.8 / V4.5.9 previews)

**V4.5.8 preview** ran V5.A.2 from cached forecasts.npz at α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:

| α | fail_rec/5 | regr/15 | Comment |
|---|---|---|---|
| 0.00 | 0 | 0 | pure v2.4 (reference) |
| 0.25 | 1 | 4 | conservative; loses 2010-04-23 |
| **0.50** | **2** | **6** | **best balance — V5.A.2 base** |
| 0.75 | 2 | 8 | regressions creep up |
| 1.00 | 2 | 10 | pure A2.1 (reference) |

**V5.A.2 cannot pass the bar alone.** Failures-recovered tops at 2/5; the bar requires 3/5. But it cuts regressions cleanly (10 → 6 at α=0.5) and recovers 2008-10-03's catastrophic regression (90-band 7 → 41). **V5.A.2 at α=0.5 is the BASE configuration on which V5.B is layered.**

**V4.5.9 preview** verified the drawdown feature works at 3/5 Cohort-2 anchors (2001-04, 2001-10, 2022-03 — top-20 candidates have 75–85% same-sign forwards). At COVID (2020-03-16) the feature is **bimodal** (pulls both V-recovery and continuation precedents); needs a co-feature. At 2012-03-14 the feature is inert (target at peak, dd=0).

### V5.A.2 — Path-level ensemble v24 ⊕ A2.1 at α=0.5 — **P0 BASE**

**Hypothesis.** v2.4 and A2.1 have complementary tail-selection strengths (V4.5.7). Averaging their forecast paths inherits each anchor's stronger matcher and smooths the worst regressions. Specifically targets M1 (over-concentration: 2018-10-08, 2020-03-16), M2 (bimodal: 2008-10-03), and partial M3 (path-construction). Confirmed effective by V4.5.8 preview.

**Method.**
1. Take v2.4 and A2.1v1 forecasts.npz from existing canonical runs (no new walk-forward).
2. For each (fold × origin), concatenate path arrays at **α = 0.5** (500 paths from v2.4 + 500 from A2.1 = 1000 mixed paths).
3. Recompute all CRPS / coverage / PIT / diagnostics on the mixed path set.
4. Run the 15-anchor fat-tail panel via existing `scripts/analog_mc/compute_fat_tail_eval.py` + `scripts/analog_mc/render_fat_tail_panel.py`.
5. (Optional) Re-do at α=0.6 if 2010-04-23 90-band coverage drops too close to 45 threshold (V4.5.8 showed margin of 48 at α=0.5).

**Cost.** ~1 day. No new walk-forward — repurposes cached forecasts. ~200 LOC for the ensemble-mixing script + fat-tail panel renderer adaptation.

**Decision rule (V5.A.2 isolated).** V5.A.2 alone is NOT expected to pass the bar — V4.5.8 confirmed failures-recovered ceiling of 2/5. Goal of isolated V5.A.2 is: (a) confirm V4.5.8 preview at canonical resolution, (b) lock α and obtain V5.B's required baseline.

**Risk.** None significant. Behavior fully previewed in V4.5.8. The only canonical-resolution uncertainty is whether the 2008-10-03 recovery (7→41 90-band in preview) holds across the full 1000-path mixture — preview used the same path counts, so confidence is high.

**Deliverable.** `results/analog_mc/data/fat_tail_v5_a2.json` + diff vs v2.4 + per-anchor panel charts. (Not a new `runs/` dir since no new walk-forward.)

---

### V5.B — Drawdown-depth feature augmentation — **P0 STACK ON V5.A.2**

**Required to pass the bar.** V5.A.2 alone cannot reach ≥3/5 recoveries (V4.5.8); V5.B must rescue at least one of {2018-10-08, 2020-03-16, 2026-02-19} to clear the recovery condition.

**Hypothesis.** Cohort-2 anchors share **recent extreme drawdown velocity** — a feature missed by z-scores and corrwindow shape. Add `drawdown_60d_norm = log(close[t] / max(close[t-59:t+1])) / std(log_returns[t-59:t+1])`. V4.5.9 sanity confirmed top-20 by drawdown-distance includes 75-85% same-sign forwards at 2001-04, 2001-10, 2022-03. At 2020-03-16 the feature pulls both V-recovery (1987-11) and continuation (2008-10) precedents — bimodal but at least within range.

**Method.**
1. Implement `drawdown_60d_norm` in `src/analog_mc/features.py` (causal — uses log-ratio formula from V4.5.9, validated for unit dimensionality).
2. Extend the composite distance: keep z-scores as is (3-weight grid at resolution 0.1 = 66 combos), add `drawdown_weight` as a separate scalar grid `{0.0, 0.5, 1.0, 2.0, 4.0}` (5 values). Total grid: 66 × 5 = 330 combinations per fold vs current 66. ~5× search cost.
3. Composite distance: `d = sqrt(Σ_h w_h · (z_target_h - z_cand_h)² + drawdown_weight · (dd_target - dd_cand)²)`. Add a 4th feature plumbing through `composite_distance()`.
4. **Stack V5.B on V5.A.2**: the matcher uses (z + drawdown), then path-level ensemble with A2.1 at α=0.5. This is the V5 P0 *stack*.
5. Run canonical walk-forward at the V5.A.2-baseline weight grid configuration.
6. Fat-tail panel against v2.4 baseline.

**Cost.** ~4 days build + ~12h canonical compute (5× search grid). Plus ~1 day for the V5.A.2 stacking on top.

**Decision rule.** Promote if **the V5.A.2 + V5.B stack** passes the bar (≥3/5 V3.5 failures recovered AND ≤2/15 regressions >5%). V5.B alone (without ensemble) is also reported as a diagnostic point — if V5.B alone passes the bar, the ensemble isn't strictly needed.

**Pre-canonical sanity** (already done in V4.5.9): drawdown feature pulls 75-85% same-sign forwards at 2001-04, 2001-10, 2022-03 ✅. At 2020-03-16 it's bimodal ⚠️. At 2012-03-14 it's inert ❌.

**Risk.** 4-D feature search may degenerate to drawdown_weight ≈ ∞ if drawdown is too discriminating; this is the "shape-similar wrong-forward" risk that bit A2.1 (V4_RESULTS). Mitigation: cap drawdown_weight values in the grid as listed above.

**Deliverable.** `runs/analog_mc/<ts>_v5_b_drawdown/`, `docs/analog_mc/experiments/_v5_b_drawdown.md`, fat-tail panel, diff JSON. Plus `_v5_a2_b_stack.md` for the stacked configuration.

---

### V5.B.2 — Drawdown + vol-regime co-feature (stretch) — P2

**Only run if V5.B's COVID coverage (2020-03-16 90-band) is below 30.** V4.5.9 showed drawdown alone is bimodal at COVID's extreme drawdown level; needs a second feature to disambiguate V-recovery from continuation.

**Candidate co-features.**
- `vol_regime_indicator = std(log_returns[t-20:t+1]) / std(log_returns[t-60:t+1])` — high recent vol relative to lookback signals "crisis bottom" vs slow grind.
- `drawdown_duration = days since last 60-day peak`. COVID was ~30 days from peak to trough; 2008 was ~6 months.

**Cost.** +1 week (one new feature + extended weight grid). Same canonical pattern.

**Decision rule.** Only if V5.B alone doesn't lift 2020-03-16 coverage above 30/60.

**Deferral logic.** Initial V5.B canonical reveals the COVID outcome. If V5.B's V5.A.2-stacked COVID coverage is ≥45, no V5.B.2 needed. If 30-44, V5.B.2 may push it over. If <30, V5.B.2 may not help — fall back to V5.A.3.

---

### V5.D — B1 shrinkage parameter — P3

**Hypothesis.** B1's 1990-09-24 catastrophe (+156% CRPS, 90-band 55→11) is magnitude over-correction (V4.5.3): drift +17.9% horizon when realized was +12%. Shrinking the correction by 0.3–0.5 retains B1's mechanism (analog matcher's conditional-mean bias correction) while bounding the worst-case correction magnitude.

**Method.**
1. Add `b1_shrinkage: float = 1.0` to `Config`. Default 1.0 → bit-identical to v4 B1.
2. Modify `simulate.py:277`:
   ```python
   drift_target = drift_target + (config.b1_shrinkage * correction) / config.forecast_horizon
   ```
3. Run canonical at `b1_shrinkage ∈ {0.3, 0.5, 0.7, 1.0}`. Either as 4 separate canonicals (preferred for diagnostic clarity) or as one canonical with `b1_shrinkage` added to the search grid (cheaper but couples B1's correction magnitude to search optimization).
4. Fat-tail panel for the best-shrinkage variant.

**Cost.** ~1 hour implementation + 1 canonical × 4 shrinkage values × ~9h each = 36h compute (or 9h if one canonical with shrinkage in the search grid).

**Decision rule.**
- Promote if any shrinkage value passes the bar (≥3/5 recovered, ≤2/15 regress).
- Diagnostic: at 1990-09-24, the B1 90-band coverage should recover to ≥30/60 (vs v4 B1's 11/60). If not, shrinkage doesn't sufficiently bound the over-correction.

**Risk.** Shrinkage attenuates B1's wins (2010-04-23 −24%, 2012-03-14 −51%, 2001-10-02 −16%) proportionally. At `b1_shrinkage = 0.3`, win magnitudes drop to roughly 30% of v4 B1. Empirical question whether enough win survives to keep the failure-anchor recovery count.

**Stretch variant.** **V5.D.adapt** — make shrinkage anchor-dependent: `b1_shrinkage = clamp(|correction|, 0, max_drift) / |correction|` (cap absolute drift at `max_drift`). Targets the catastrophe directly while preserving small-correction wins. Add only if uniform shrinkage doesn't pass.

**Deliverable.** `runs/analog_mc/<ts>_v5_d_b1shrink_<v>/` per shrinkage value, `docs/analog_mc/experiments/_v5_d_b1_shrinkage.md`, fat-tail panel.

---

### V5.A.3 — Conditional corrwindow re-matching — P4 (stretch)

**Hypothesis.** A2.1's Mode-3 regressions (2017-06-01, 2022-03-01 per V4.5.6) come from path-construction tightness: under corrwindow, conditional block sampling is disabled (`simulate.py:306–309`), so block-0 distances are reused for blocks 1–5 across all paths. Enabling per-path per-block re-matching under corrwindow would diversify paths and widen bands.

**Method.**
1. Implement a per-path simulated **returns window** of length L that slides over the simulated horizon. Each block, the per-path window's last L returns drive a per-path corrwindow distance to the candidate pool.
2. Add `generate_paths_conditional_corrwindow` in `simulate.py` (analog of `generate_paths_conditional` but operating on returns windows).
3. Vectorize the per-path-per-block corrwindow distance: `(n_paths, L) @ (K, L).T / L` after z-scoring rows — analogous to `composite_distance_batched` but for window correlations.
4. Re-route the `use_conditional` gate at `simulate.py:306–309` to accept corrwindow + conditional.
5. Run canonical at the v4 A2.1 settings (corrwindow L=100, n_eff=50) with `conditional_block_sampling=True`.

**Cost.** ~1 week. ~300 LOC + ~10 tests + 1 canonical (~9h compute).

**Decision rule.** Same as V5.A.2.

**Risk.** Subtle correctness bugs (per-path window state management). Mitigation: extensive correctness tests, including the C1 causality test (per-path window must only reference simulated days, not the original returns array except for the initial window).

**Deferral logic.** Only run V5.A.3 if **V5.A.2 + V5.B together** do not pass the bar. If they do, the residual regressions are small enough that V5.A.3's marginal coverage doesn't justify the 1-week cost.

**Deliverable.** `runs/analog_mc/<ts>_v5_a3_cond_corrwindow/`, narrative, fat-tail panel.

---

## Sequencing — revised per V4.5.8 evidence

```
Phase 1 (P0 — minimum stack required to pass the bar):
  Day 1:     V5.A.2 ensemble at α=0.5 from cached forecasts
             → confirms V4.5.8 preview at full resolution
  Days 2-5:  V5.B drawdown feature implementation + tests
  Days 6-7:  V5.B canonical walk-forward + fat-tail panel
  Day 7:     V5.A.2 ⊕ V5.B stack — recompute panel from stacked paths
             → promotion-bar verdict

Phase 2 (P2 — only if Phase 1 fails on COVID):
  Days 8-14: V5.B.2 with vol-regime co-feature
             → COVID coverage attempt

Phase 3 (P3 — cheap independent add):
  Day 15:    V5.D b1_shrinkage canonical at 0.3, 0.5, 0.7
             → rescue 1990-09-24

Phase 4 (P4 — stretch; only if Phases 1-3 don't pass):
  Days 16-22: V5.A.3 conditional corrwindow

Synthesis: V5_RESULTS.md with promotion decision.
```

**Stop conditions.**
- **Phase 1 (V5.A.2 ⊕ V5.B) passes the bar** → no Phase 2/3/4. v5 closes with the stack promoted.
- **Phase 1 partially passes** (e.g., 3/5 recovered but 4/15 regress) → run V5.D (cheap) to see if 1990-09-24 rescue further trims regressions. Decide between V5.B.2 (COVID-focused) or V5.A.3 (Mode-3 focused) based on which residual regression looks more rescuable.
- **Phase 1 fails on the recovery count** (still 2/5) → V5.B is underperforming. Decide between:
   - V5.B.2 with co-feature (if COVID is the unrescued anchor),
   - V5.C delay-coordinate distance (if drawdown formulation seems wrong overall),
   - reformulate V5.B with a different feature.
- **Phase 1 fails on the regression count** (≥3/15) → V5.A.2 isn't damping enough; consider α ∈ {0.4, 0.6} sweep. Should never exceed V4.5.8's α=0.5 baseline of 6/15.

**Hard abort condition.** If V5.A.2 ⊕ V5.B's failure CRPS is *worse* than v2.4's baseline, abort and rebuild the v5 plan — something fundamental about the diagnosis is wrong. Probability low based on V4.5.8 preview.

## V5 promotion bar (unchanged from v4)

> ≥3/5 V3.5 failure anchors recovered (90%-band ≥45/60 days) AND ≤2/15 anchors regressing CRPS >5%.

Conservative interpretation: a candidate must pass BOTH conditions to promote to default. If any candidate passes only the recovery bar OR the regression bar but not both, it remains canonical-only (analyzable but not promoted).

## Refresh the cross-experiment fat-tail comparison

After every V5 canonical run completes, regenerate the cross-experiment fat-tail figures so the comparison panel includes the new run alongside the v4 set. `scripts/analog_mc/render_fat_tail_panel_compare.py` accepts `--experiment LABEL=RUN_DIR` flags (repeatable, `rsplit("=", 1)` so `=` in labels is fine) and assigns colors automatically for unknown labels.

```bash
# Canonical example (extend after V5.A.2 lands):
uv run python scripts/analog_mc/render_fat_tail_panel_compare.py --experiment-grid \
  --experiment "v2.4 baseline (Cell-D-s30)=runs/analog_mc/20260520T045525Z" \
  --experiment "B1 (Platzer local-linear)=runs/analog_mc/20260520T155220Z" \
  --experiment "A2.1 (corrwindow L=100)=runs/analog_mc/20260521T061730Z" \
  --experiment "B5 (joint A2.1 + B1)=runs/analog_mc/20260521T121025Z" \
  --experiment "V5.A.2 ensemble=runs/analog_mc/<v5_a2_ts>"

# Also refresh the per-experiment 15-anchor panel for the new run:
uv run python scripts/analog_mc/render_fat_tail_panel.py \
  --run-dir runs/analog_mc/<v5_a2_ts> \
  --label "V5.A.2 ensemble" \
  --out-dir docs/analog_mc/experiments/figs/v5_a2_ensemble_fat_tail \
  --prefix v5_a2_ensemble
```

Outputs land in `docs/analog_mc/experiments/figs/fat_tail_compare/` (overlay + per-anchor 2×2 grids + `experiment_grid.png` with 15 forecasts + aggregated PIT per experiment) and `docs/analog_mc/experiments/figs/<exp_id>_fat_tail/` (per-experiment 15-anchor gallery). To lock a canonical color across reruns for a new experiment, add it to `EXP_COLORS` at the top of the script. The figures are not currently embedded in any V5 doc — surface them in `V5_RESULTS.md` / per-experiment narratives as needed.

## Deliverables manifest

```
docs/analog_mc/V5_EXPERIMENTS_PLAN.md             # this doc
docs/analog_mc/V5_RESULTS.md                       # synthesis after v5 closes
docs/analog_mc/experiments/_v5_a2_ensemble.md
docs/analog_mc/experiments/_v5_b_drawdown.md
docs/analog_mc/experiments/_v5_d_b1_shrinkage.md
docs/analog_mc/experiments/_v5_a3_cond_corrwindow.md  # if executed

configs/analog_mc/ablation_v5_b_drawdown.yaml
configs/analog_mc/ablation_v5_d_b1shrink_{0.3,0.5,0.7}.yaml
configs/analog_mc/ablation_v5_a3_cond_corrwindow.yaml

scripts/v5/ensemble_paths.py                      # V5.A.2
scripts/v5/compute_v5_a2_fat_tail.py
# (other v5 scripts as needed)

src/analog_mc/features.py                         # extended for V5.B
src/analog_mc/distances.py                        # extended for V5.B
src/analog_mc/config.py                           # b1_shrinkage, drawdown_weight knobs
src/analog_mc/simulate.py                         # b1_shrinkage hook (V5.D), conditional corrwindow (V5.A.3)

tests/analog_mc/test_drawdown_feature.py          # V5.B
tests/analog_mc/test_b1_shrinkage.py              # V5.D
tests/analog_mc/test_conditional_corrwindow.py    # V5.A.3

results/analog_mc/data/fat_tail_v5_{a2,b,d,a3}{,_diff}.json
runs/analog_mc/<ts>_v5_*/                          # canonical run dirs
```

## Read-first checklist for a fresh session

A future Claude session picking this up cold should read, in order:

1. **This file** (`V5_EXPERIMENTS_PLAN.md`).
2. [`V4_5_RESULTS.md`](V4_5_RESULTS.md) — the mechanism inventory that grounds every v5 experiment.
3. [`v4.5/_v4_5_5_mechanism_map.md`](v4.5/_v4_5_5_mechanism_map.md) — the full per-anchor classification matrix.
4. [`V4_RESULTS.md`](V4_RESULTS.md) — v4 outcomes (none promoted; v2.4 still canonical).
5. [`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md) — the 15-anchor eval set and promotion bar.
6. **Existing canonical artifacts**: `runs/analog_mc/{20260520T045525Z, 20260520T155220Z, 20260521T061730Z, 20260521T121025Z}/` — these are V5.A.2's inputs.
7. `src/analog_mc/{simulate.py, distances.py, distances_corrwindow.py, local_linear.py, features.py}` — the implementation surface v5 modifies.

The `CLAUDE.md` project conventions and `.claude/memories/` are auto-loaded; no need to re-read.

## What v5 explicitly does NOT do

- ❌ Add a per-fold gate signal (val_crps or otherwise). v4.5 ruled this out — no clean discriminator.
- ❌ Re-litigate B5 (joint A2.1+B1). v4 already showed joint is not additive.
- ❌ Defer tail inflation to v6. v4.5 showed COVID is matcher-addressable; tail inflation may still surface in v6+ if v5.B doesn't crack it, but it's not the v5 lever.
- ❌ Implement A1 textbook FHS. Originally in v4 backlog; V3.5.3 partial fix evidence is enough to defer until V5's matcher work concludes — only revisit if v5 doesn't pass.
- ❌ Touch the optimizer (B3 Dirichlet, B4 regularized search). v3.5.1 confirmed weight homogeneity is not the failure driver.

These are explicitly out so v5 doesn't lose focus.
