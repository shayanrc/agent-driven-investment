# analog_mc v4 — experiment plan

Companion to [`V3_PLAN.md`](V3_PLAN.md), [`RELATED_WORKS.md`](RELATED_WORKS.md), [`ABLATION_STUDIES_REPORT.md`](ABLATION_STUDIES_REPORT.md). This file is the **spec** for the next round of experiments after v3 closed. Like prior plan docs: every decision below is made for a stated reason — surface deviations rather than silently change.

---

## Starting state (entering v4)

| Item | State |
|---|---|
| Current default | **v2.3 — Cell D** (drift + conditional sampling, shrinkage=0.50, bl=10) |
| **Pending promotion** | **v2.4 — Cell-D-s30** (`momentum_shrinkage=0.30`); canonical confirmation run `20260520T045525Z` in progress. If gates pass (mean CRPS ≤ 0.05056, PIT slope ≤ +0.10), this becomes default before v4 starts. |
| v3 phase closure | E1, E2 (+ ext), E3, E7, E9-A, E10, E11 all done. v3 produced ~9.5% mean-CRPS improvement v2.1 → v2.4 and confirmed the residual `acf_seam_degradation` failure is **structurally unfixable within the analog-block primitive**. |
| Open architectural question | Can we close `acf_seam_degradation` without abandoning analog matching? v4 tests two candidate answers (B1, B2) before considering a full architectural rewrite. |
| Open attribution question | E9-A's small CRPS win is unattributable — no textbook baseline isolates whether analog-block selection adds value over plain GARCH + residual bootstrap (A1). |
| Open primitive question | Is composite-Euclidean z-score the right matcher distance at all, or have we been over-engineering the wrong knob? (A2) |
| **v3.5 diagnostic reshape** | [`V3_5_RESULTS.md`](V3_5_RESULTS.md) (executed 2026-05-20) confirmed the fat-tail failures are **"right-era, wrong-magnitude"** matches: matcher gets direction right (sign-match ≥76% in 3/5 failures) but picks temporally-clustered analogs whose forward windows underestimate the realized move. Pool has 156–194 ≥+30% rallies per failure fold but matcher places <1% mass on them. Reshape applied below: **B1 → sole P0, A2 → joint P0, A1 → P1** (FHS catches only 2/5 in spot-check, not the full failure surface). |

## v4 scope and non-scope

**In scope:** experiments that either
1. Attribute existing wins to specific mechanisms (Category A — baseline isolation), or
2. Structurally improve the analog primitive's calibration without abandoning it (Category B), or
3. Replace heuristic decision rules with formally-grounded calibration tests (Category C).

**Out of scope:** transaction-cost / PnL modeling, downstream allocation logic, deep-learning replacements for the matcher, multi-asset joint forecasting. These remain v5+ rewrites — the analog primitive is still the load-bearing piece.

Mid-v4 the literature-cited "abandon the analog primitive" path (e.g. parametric / diffusion path generation) becomes the live alternative if all three B-experiments fail to close `acf_seam_degradation`. Decision deferred to end-of-v4.

---

## Experiment inventory

Status legend after v4 closure (2026-05-22): ✅ promoted, ⛔ ran-but-did-not-promote, ⏸ deferred, 🚫 cancelled.

| ID | Title | Category | Source | Cost | Priority | **Outcome** |
|---|---|---|---|---|---|---|
| **B1** | Platzer local-linear correction | Structural | Tier 2 — `arXiv 2007.14216` | ~1.5 d impl + ~3 h run | **P0** | ⛔ ran, did not promote — see [`_b1_local_linear.md`](experiments/_b1_local_linear.md) |
| **A2.1** | Correlation-window matcher distance | Attribution | v4 sanity-driven (OFTER substitute) | ~1 d impl + ~6 h run | **P0** | ⛔ ran, did not promote — see [`_a2_corrwindow_L100.md`](experiments/_a2_corrwindow_L100.md) |
| **B5** | A2.1+B1 joint (corrwindow + Platzer) | Structural | v4 sanity evidence — both knobs complementary at the 5 failure anchors | ~0 impl (both knobs ship) + ~5 h run | **P0 (gated)** | ⛔ ran, did not promote — see [`_b5_joint.md`](experiments/_b5_joint.md) |
| A2.2 | OFTER-faithful max-correlation | Attribution | Tier 1 — `arXiv 2304.03877` | unknown (paper-blocked) | P1 | ⏸ deferred — paper PDF not text-extractable |
| A1 | Textbook FHS baseline | Attribution | Tier-4 synthesis (RELATED_WORKS) | ~2-3 h compute | P1 | ⏸ deferred to v5 — V3.5.3 spot-check showed partial fix; canonical worth running formally |
| **C1** | Block-bootstrap KS / PIT GoF | Diagnostic | Tier 3 — `arXiv 2511.05733` | ~half-day impl | **P1** | ⏸ deferred to v5 — diagnostic only, no urgency |
| B2 | Delay-coordinate features | Structural | Tier 2 — `arXiv 2005.06623` (Burov) | ~1 d impl + ~2 h run | P2 | ⏸ deferred |
| B3 | Dirichlet posterior on analog weights | Structural | Tier 2 — `arXiv 1701.04485` (McDermott–Wikle) | ~2 d impl + ~3 h run | P2 | ⏸ deferred |
| B4 | Regularized weight search (candidate) | Structural | v3.5 incidental — `V3_5_RESULTS.md` | ~3 h impl + ~2 h run | P2 (gated) | 🚫 cancelled — V3.5.1 issue no longer load-bearing per v4 results |

P0 = sequenced first. P1 = run after P0 if results don't make them obsolete. P2 = scoped now, defer execution. B4 is gated on B1/A2 outcomes — only runs if val/test gap at failure anchors persists.

### v4 closure (2026-05-22)

**Final synthesis** at [`V4_RESULTS.md`](V4_RESULTS.md). **None of B1/A2.1/B5 promote**; v2.4 Cell-D-s30 remains canonical. Per the V4 promotion bar (≥3/5 failures recovered at 90-band ≥45/60 AND ≤2/15 anchors regressing CRPS >5%):

| | Failures recovered | Anchors regress | Bar? |
|---|---:|---:|---|
| B1 | 1/5 | 5/15 | ❌ |
| A2.1 | 2/5 | 10/15 | ❌ |
| B5 | 2/5 | 10/15 | ❌ |

The strongest mechanistic signal — A2.1's −47% CRPS at two failure anchors (V-recovery + flash crash) — points the next v5 work at **gated A2.1**: keep the corrwindow distance but fall back to weighted-Euclidean when val_crps is anomalously high (the 2008-10-03 collapse pattern). Detail in V4_RESULTS.md §"Strongest signal for v5".

**v3.5 reshape (2026-05-20).** Priorities above reflect the v3.5 diagnostic findings ([`V3_5_RESULTS.md`](V3_5_RESULTS.md)):
- **B1** is the structurally-targeted fix for the "right-era, wrong-magnitude" failure pattern (V3.5.4 evidence) — promoted to sole P0.
- **A2** is the orthogonal test of whether breaking Euclidean-on-features temporal clustering recovers tail-magnitude analogs — promoted to joint P0.
- **A1** retains attribution-baseline value but loses its "ahead of B1" justification — the V3.5.3 spot-check showed it only catches 2/5 failure anchors. Demoted to P1.
- **B4** is a new candidate surfaced by V3.5.1 (failure folds disproportionately at extreme grid corners with bad val/test generalization). P2 / gated.

---

## Category A — Attribution baselines

### A1. Textbook FHS baseline — **P1** (was P0; reshaped by v3.5)

**Motivation.** E9-A (GARCH-conditional resampling) showed −2.2% mean CRPS, −3.8% high-vol vs the EWMA baseline at zero drift. We currently cannot attribute this gain: it could come from (a) σ-conditioning (the GARCH path), or (b) the analog-block selection retaining intra-block dependence structure that residual bootstrap would destroy. Without a textbook-FHS baseline that strips the analog primitive entirely, **E9 wins are unattributable** and we cannot honestly claim the analog primitive does meaningful work in high-vol regimes.

This is the cheapest experiment that can falsify a foundational claim of the pipeline.

**v3.5 reshape.** V3.5.3 ([`v3.5/_v3_5_3_fhs_spotcheck.md`](v3.5/_v3_5_3_fhs_spotcheck.md)) ran a FHS spot-check at the 5 failure anchors and found 2/5 catches at the 90%-band ≥45/60 threshold. FHS wins decisively on COVID-style V-recoveries (2020-03-16, 2026-02-19) by inflating the σ-driven dispersion, but underperforms v2.4 on the 2001 V-recovery (GARCH fits a calmer pre-2001 regime → tight bands) and the bull-momentum→crash failures. Conclusion: A1 only addresses a *sub-pattern* of the failure surface, not the whole. The attribution-baseline motivation still holds; the "ahead of B1" justification does not. **Demoted to P1.**

**Method.** GARCH(1,1) fit per origin (reuse `_garch_fit_cached` from E9). Sample per-step σ trajectories as in E9, but draw return signs from i.i.d. bootstrap of standardised GARCH residuals instead of analog-block selection. No matcher, no k-NN, no weights. Same 76 folds, same metrics.

**Decision rule.**
- If FHS ≈ E9-A (within E3's 0.08% noise floor): **the analog primitive is decorative for high-vol**. E9 should be replaced by textbook FHS in opt-in mode, and we re-scope what the matcher is actually contributing.
- If FHS materially worse than E9-A (mean CRPS gap > 1%): confirms analog primitive adds real value beyond σ-conditioning. Proceed to B1.
- If FHS materially better than E9-A: the analog primitive is *hurting* in high-vol. v5 reframe.

**Acceptance criteria for promoting FHS as a comparator baseline in `RESULTS.md`:** ran successfully on 76 folds, full diagnostic stack rendered (CRPS, per-vol-regime CRPS, PIT, ACF), no implementation bugs (sanity: σ-clip-hit rate < 10%, drift bias ≈ 0).

**Deliverables.** `src/analog_mc/baselines/fhs.py`, `configs/analog_mc/baseline_fhs.yaml`, run dir, `_a1_fhs.md` report. **Plus the mandatory fat-tail panel**: 15 charts in `docs/analog_mc/experiments/figs/a1_fhs_fat_tail/`, coverage table in `_a1_fhs.md`, per-anchor CRPS diff vs v2.4 baseline.

**Cost.** ~3 h impl (GARCH fit + residual bootstrap + integration with walk_forward harness) + ~2 h run + ~30 min fat-tail panel render. Total ~5.5 h.

---

### A2. Matcher-distance ablation — **P0** (was P1; promoted by v3.5; split 2026-05-20 evening)

A2 is now split into A2.1 (correlation-window, implementable now) and A2.2 (OFTER-faithful, deferred). Full design rationale in [`experiments/_a2_design.md`](experiments/_a2_design.md).

- **A2.1 (corrwindow)** — Pearson correlation between L-day pre-origin return windows. Scaffold + sanity landed; canonical at L=100 queued behind B1. Sanity showed −17.15% failure-anchor CRPS at L=100, with material wins on flash-crash and V-recovery anchors.
- **A2.2 (OFTER-faithful)** — deferred until the paper's exact definition is accessible (current WebFetch tooling can't extract the PDF text).

The original A2 motivation below applies to both variants — A2.1 is the implementable approximation, A2.2 the literature-faithful target.

#### Original motivation

**Motivation.** OFTER (`arXiv 2304.03877`) uses a *modified maximal correlation coefficient* as its k-NN distance, instead of weighted Euclidean over hand-picked features. This is the single cleanest test of our matcher-distance design — does our hand-tuned `(w₁, w₂, w₃, n_eff)` grid actually beat a principled correlation-based distance? If max-corr matches or beats us, the entire `weight_grid_resolution` search has been over-engineering the wrong primitive.

**v3.5 reshape.** V3.5.4 ([`v3.5/_v3_5_4_analog_autopsy.md`](v3.5/_v3_5_4_analog_autopsy.md)) showed that the v2.4 matcher's top-20 analogs are heavily **temporally clustered** — once one date matches, its z-score neighbours dominate the top-K. The cluster's forward windows are mutually correlated and systematically mild, even when the global pool has the realized tail magnitude available (V3.5.2: 156–194 ≥+30% rallies per failure pool, but matcher places <1% mass on them). Max-correlation distance attacks this exact pathology: it doesn't share the Euclidean-on-features locality and is plausibly diversity-preserving across regimes. Promoted to joint P0 alongside B1. A2 and B1 are orthogonal code paths and can run in parallel.

**Method.** Replace `compute_distances()` in `src/analog_mc/distances.py` with a max-correlation variant computed over the same input feature vectors. Keep all other knobs at v2.4 (Cell-D-s30) defaults. Single fast-preset run.

**Decision rule.**
- If max-corr mean CRPS ≤ Cell-D-s30 within noise floor: defer the weight grid (huge speed win); investigate whether learned distances beat both (v5 question).
- If max-corr ≥1% worse: our hand-tuned distance is doing real work. Document this and close the question.
- If max-corr ≥1% better: structural win; integrate into the matcher and re-run canonical Cell-D-s30 with the new distance.

**Deliverables.** `src/analog_mc/distances_maxcorr.py`, `configs/analog_mc/ablation_A2_maxcorr.yaml`, `_a2_maxcorr.md`. **Plus the mandatory fat-tail panel**: 15 charts in `docs/analog_mc/experiments/figs/a2_maxcorr_fat_tail/`, coverage table in `_a2_maxcorr.md`, per-anchor CRPS diff vs v2.4 baseline.

**Cost.** ~1 d impl (max-correlation computation has to be causal and vectorised over the 10k-day history — non-trivial) + ~4 h run.

---

## Category B — Structural improvements to the analog primitive

### B1. Platzer local-linear correction — **P0** (single highest-priority experiment after v3.5)

**Motivation.** This is the *only* literature-prescribed fix that is compatible with the analog-block scaffolding and could plausibly close `acf_seam_degradation` from inside the v3 architecture. E9 failed to close the rule because per-step σ scaling preserves the intra-block direction structure that drives the ACF gap. Local-linear regression changes the *conditional mean*, which is the right knob.

**v3.5 evidence.** V3.5.4 ([`v3.5/_v3_5_4_analog_autopsy.md`](v3.5/_v3_5_4_analog_autopsy.md)) made this the highest-info-density v4 experiment. The matcher's *mean placement* misses badly at the 5 failure anchors (e.g. 2020-03-16: realized +43.8%, matcher E[60d] = +5.9%) and places ≤8% mass on the realized-magnitude tail in 3/5 cases. B1's Jacobian-bias correction is the textbook fix for high-Lyapunov regimes — exactly the regime-transition shape of these failures. The motivation is no longer just "close ACF": it's "address the *headline* fat-tail failure pattern surfaced in V3.5.4."

From RELATED_WORKS Tier 2 (Platzer–Yiou, `arXiv 2007.14216`):
> *Forecast error scales N^(−2/d), grows along large-Lyapunov directions, and analog + local linear regression explicitly estimates the Jacobian, eliminating the leading bias.*

Our k-NN matcher degrades worst at high-vol regime onsets (large local Jacobian) — exactly where our per-vol-regime CRPS shows the biggest residual gap (high-vol CRPS is 3× low-vol).

**Method.** For each fold's selected k analog blocks, fit a local linear regression in the *feature space* mapping (z-scores at origin → realized return path). Use the fitted regression to apply a bias correction to each drawn analog block before stitching. Two design choices to nail down before implementing:

1. **Regression target — return path vs feature path.** Regressing on the realized 10-day block of returns is straightforward but high-dim; regressing on a low-dim summary (cumulative return, terminal vol) and then propagating the correction is cheaper. Default to terminal-cumulative-return correction; add a B1-variant for full-path if needed.
2. **Regression weights — uniform vs n_eff-weighted.** Match the matcher's weighting scheme (n_eff-weighted) for consistency.

**Decision rule.**
- If `acf_seam_degradation` stops firing (rule value ≥ −0.30): **structural win**. Closes the v3-residual failure inside the analog primitive. Re-run canonical and promote.
- If rule moves toward zero but doesn't cross threshold (e.g. −0.55): partial mechanism; quantify how much the bias correction explains. Likely still a CRPS win.
- If rule unchanged: confirms the ACF gap is from *intra-block sampling* not *across-block mean bias* — refocuses v5 on within-block primitive change.

Independent of the ACF rule, expect ≥1-2% mean CRPS gain in high-vol regime even if ACF stays put (the Jacobian correction's primary effect).

**Deliverables.** `src/analog_mc/local_linear.py`, sampling.py changes (gated on `local_linear_correction: bool`), `configs/analog_mc/ablation_B1_localreg.yaml`, tests, `_b1_local_linear.md`. **Plus the mandatory fat-tail panel**: 15 charts in `docs/analog_mc/experiments/figs/b1_local_linear_fat_tail/`, coverage table in `_b1_local_linear.md`, per-anchor CRPS diff vs v2.4 baseline. *B1 is the experiment FAT_TAIL_EVAL.md is structurally designed to evaluate — the panel is the headline result, not a supplementary deliverable.*

**Cost.** ~1.5 d impl + ~3 h run.

---

### B2. Delay-coordinate features — **P2**

**Motivation.** Burov–Giannakis (`arXiv 2005.06623`) prove that KAF — and by extension our matcher — recovers the optimal forecast *only* when the prediction variable is Markovian in the observed state. Daily NASDAQ returns are emphatically non-Markovian in our 3-z-score state, so the matcher is biased *exactly where their theory predicts* — and our high-vol CRPS gap is consistent with this prediction.

The cheap fix is delay-coordinate embedding: extend the state from `{z₂₀, z₅₀, z₂₀₀}` to also include lagged returns `r_{t−1}`, `r_{t−5}` (Takens reconstruction).

**Method.** Add `lagged_returns: [1, 5]` knob to config. In `features.py`, append the lagged return components to the feature vector before z-scoring (causal). Re-tune `n_eff` since state dimensionality changes.

**Decision rule.**
- If mean CRPS ≥1% better: structural improvement; ship as v2.5.
- If flat: confirms our 3-z-score state already captures most of the Markovian-recoverable signal; close the question.
- If worse: lag features add noise without information (likely if `r_{t−1}` already correlates with `z₂₀`).

**Why P2 (not P0).** Implementation is cheap but the prediction is *less specific* than B1 — Burov's theory tells us the matcher is biased but doesn't prescribe a specific delay structure. B1 is more targeted; B2 is a softer "follow the theory" hedge.

**Deliverables.** `features.py` lagged-return extension, `configs/analog_mc/ablation_B2_delay.yaml`, `_b2_delay_coords.md`. **Plus the mandatory fat-tail panel**: 15 charts in `docs/analog_mc/experiments/figs/b2_delay_coords_fat_tail/`, coverage table in `_b2_delay_coords.md`, per-anchor CRPS diff vs v2.4 baseline.

**Cost.** ~1 d impl + ~2 h run.

---

### B3. Dirichlet posterior on analog weights — **P2**

**Motivation.** McDermott–Wikle (`arXiv 1701.04485`) show that treating analog weights as a posterior (Dirichlet) rather than point estimates improves probabilistic calibration without re-tuning the matcher. Our `(w, n_eff)` is currently a point estimate per fold; uncertainty in analog selection is not propagated into PIT.

This is the candidate fix for `sloped_global_pit` margin pressure (Cell-D-s30 passes by only 0.0042) — Dirichlet draws would inflate dispersion in under-matched folds where the weight estimate is uncertain, partially correcting the slope without sacrificing CRPS.

**Method.** Per fold, sample K weight vectors from a Dirichlet centred on the optimised `w*` with concentration α (new knob). Generate paths under each, pool. Calibrate α on the validation set against PIT slope.

**Decision rule.**
- If PIT slope improves (closer to zero) without CRPS regression > 1%: structural calibration win, complementary to drift shrinkage; ship as v2.6.
- If CRPS regresses ≥1% with marginal PIT improvement: not worth it.
- If PIT unchanged: weight uncertainty is not the slope's cause — focus on the prior over (w, n_eff) jointly instead.

**Why P2.** More complex implementation than B2, more speculative payoff. Worth scoping but defer.

**Deliverables.** Dirichlet sampling in `search.py`/`simulate.py`, config knob `weight_posterior_alpha: float | None`, `configs/analog_mc/ablation_B3_dirichlet.yaml`, tests, `_b3_dirichlet.md`. **Plus the mandatory fat-tail panel**: 15 charts in `docs/analog_mc/experiments/figs/b3_dirichlet_fat_tail/`, coverage table in `_b3_dirichlet.md`, per-anchor CRPS diff vs v2.4 baseline.

**Cost.** ~2 d impl + ~3 h run.

---

### B5. A2.1+B1 joint — **P0 (gated on B1 + A2.1 ship)**

**Motivation.** Both B1 (Platzer local-linear) and A2.1 (corrwindow distance at L=100) sanity-passed individually with complementary per-anchor strengths (see "Sanity-driven reshape" in §Sequencing). The two effects are **independent code paths**: B1 enters via `drift_target += correction / H` (after distance computation), A2.1 enters via the `matcher_distance` switch (the distance computation itself). Stacking them requires no new code — just a config that flips both knobs.

**Method.** Inherit v2.4 Cell-D-s30 settings; flip `local_linear_correction: true` AND `matcher_distance: corrwindow`, `corrwindow_length: 100`. Same canonical 76-fold harness.

**Decision rule.**
- If failure mean CRPS improves more than `max(B1_solo, A2.1_solo)` by ≥3%: **additive structural win**. Promote as v2.5 if controls don't regress.
- If failure mean CRPS ≈ `max(B1_solo, A2.1_solo)`: knobs are not additive (one dominates). Pick the single-knob winner.
- If failure mean CRPS materially worse than either solo: knobs interact destructively. Investigate the interaction (likely the drift correction's regression coefficients are estimated on a different analog set under corrwindow, producing miscalibrated shifts).

**Gating.** Only execute if both B1 and A2.1 canonical runs ship without major control regression (>5% control CRPS regression vs v2.4 means we're solving the wrong problem).

**Deliverables.** `configs/analog_mc/ablation_B5_joint.yaml`, `_b5_joint.md` report. **Plus the mandatory fat-tail panel**, with a *triple* per-anchor comparison: v2.4 vs B1 vs A2.1 vs B5.

**Cost.** ~0 impl (config-only) + ~3 h run + 30 min diagnostics.

---

### B4. Regularized weight search — **P2 (candidate, gated)**

**Motivation.** Surfaced by V3.5.1 ([`v3.5/_v3_5_1_weights.md`](v3.5/_v3_5_1_weights.md)). 4 of 5 failure folds land at extreme grid corners (max weight > 0.9) with materially worse val_crps than test_crps — e.g. 2001-10-02 (val 0.152, test 0.108), 2010-04-23 (val 0.050, test 0.038). The 60-day val window at regime-transition anchors is short and stressful; the search lands at corner solutions that don't generalize. Plausibly a small but real failure-anchor contributor, orthogonal to the matcher selection / Jacobian-inflation questions.

**Method.** Add a Dirichlet prior on weights centered on the cross-fold median (≈ `[0.10, 0.10, 0.15]`) with small uniform mass. Score candidates as `val_crps − λ · log p(weights | prior)`. Single hyperparameter `λ`. Re-run the 76-fold harness; compare per-anchor coverage on the 15-anchor fat-tail panel.

**Decision rule.**
- If regularized search resolves ≥2 of the 5 failure anchors without regressing controls: ship as a v4 minor improvement.
- Otherwise close as a non-finding.

**Why P2 and gated.** B1 / A2 may already close these failures; if so, B4 isn't needed. Gate execution on B1+A2 fat-tail results.

**Deliverables.** `search.py` regularization hook, `configs/analog_mc/ablation_B4_regweights.yaml`, `_b4_regularized_search.md`. **Plus the mandatory fat-tail panel.**

**Cost.** ~3 h impl + ~2 h run.

---

## Category C — Diagnostic upgrades

### C1. Chandy block-bootstrap KS / PIT GoF — **P1**

**Motivation.** Our four decision rules (`sloped_global_pit`, `u_shaped_high_vol_pit`, `acf_seam_degradation`, `clip_hit_excessive`) use heuristic thresholds picked from V2_PLAN. The v3 Cell-D-s30 verdict — `sloped_global_pit = +0.0958, passes ±0.10` — hinges on a single hand-picked threshold by a 0.0042 margin. A formal goodness-of-fit test would replace the threshold with a p-value with proper Type-I error control.

Chandy et al. (`arXiv 2511.05733`) provide a block-bootstrap KS test that's specifically valid under the dependence structure of daily returns — the textbook KS test assumes i.i.d. and would be over-rejecting on our PIT residuals.

**Method.** Implement the circular block bootstrap KS test (block length `l = ⌈n^{1/3}⌉` per their recommendation; on 76 × 60 = 4560 PIT values this is `l ≈ 17`) in `src/analog_mc/diagnostics.py`. Add as a new diagnostic alongside the existing `sloped_global_pit` heuristic — keep both, deprecate heuristic after one v4 promotion cycle confirms agreement.

**Decision rule.** No promotion decision rides on this experiment directly — it changes how we *evaluate* promotion in future experiments. Acceptance criterion: KS p-values agree with heuristic verdicts on the existing v3 fast-preset runs (E1, E2 family, E9-A) for at least 4 of 5 cases. If they systematically disagree, root-cause before adopting.

**Deliverables.** `diagnostics.py` changes, test cases, retroactive table comparing heuristic vs GoF verdict on v3 results in `_c1_pit_gof.md`.

**Cost.** ~half-day impl. No new compute run needed (uses existing PIT outputs from v3 runs).

---

## Sequencing

Reshaped 2026-05-20 per [`V3_5_RESULTS.md`](V3_5_RESULTS.md). The pre-v3.5 sequencing put A1 ahead of B1 on attribution grounds; v3.5's per-anchor evidence reorders that — see "v3.5 reshape" notes throughout.

### Sanity-driven reshape (2026-05-20 evening)

Both B1 and A2.1 sanity checks at the 5 V3.5 failure anchors completed:

| Anchor | B1 isolated (CRPS Δ) | A2.1 L=100 isolated (CRPS Δ) | Best |
|---|---:|---:|---|
| 2001-10-02 V-recovery | **−22.7%** | **−48.3%** | A2.1 |
| 2010-04-23 flash crash | **−12.4%** | **−36.4%** | A2.1 |
| 2018-10-08 Q4 selloff | +0.7% (flat) | +20% (regress) | B1 |
| 2020-03-16 COVID rally | +5.2% (regress) | −9% | A2.1 |
| 2026-02-19 recent rally | **−13.0%** | flat | B1 |

The two approaches are **complementary**: A2.1 dominates on shape-similar failure anchors (flash crash, V-recovery); B1 dominates where the matcher already gets direction right and just needs magnitude correction. Picking the better of the two per anchor yields a ~20% failure CRPS improvement (oracle bound, isolated). This motivates a joint experiment.

**Reshape:**
- Confirm A2.1 canonical at L=100 (sanity selects this as the winning window) — promote to P0 alongside B1.
- Add **B5 (A2.1+B1 joint)** as a new candidate: stack the drift correction on top of the corrwindow distance. Both are independent knobs and require no new code. Cost: 1 canonical run (~3 h). Decision: gate on B1 + A2.1 canonical results — only run if both ship without regressing controls badly.
- 2020-03-16 and 2018-10-08 remain hard. Neither experiment recovers magnitude on the COVID anchor; if the joint experiment also fails, **B6 (high-vol-regime variance inflation)** becomes a candidate (out of v4 scope; surface at end-of-v4 decision).

1. **Canonical Cell-D-s30 baseline already landed** at `runs/analog_mc/20260520T045525Z` (76 folds, 1000 paths, 66×5 weight grid). This is the v2.4 reference for all v4 comparisons; per-fold artefacts are in place and were the substrate for v3.5's diagnostics.
2. **B1 (Platzer local-linear correction)** — canonical IN PROGRESS at `runs/analog_mc/20260520T155220Z` (started 2026-05-20 21:22, ETA ~7 h). Sanity (`scripts/analog_mc/v4_b1_sanity.py`) showed −6.7% failure CRPS / 4-of-5 sign agreement at isolated correction; canonical includes search-time effect.
3. **A2.1 (correlation-window distance)** — corrwindow scaffolded (replaces OFTER-faithful, which is deferred until paper access). Sanity selects **L=100**. Canonical queued for execution after B1 canonical completes.
4. **B5 (A2.1+B1 joint)** — new candidate from the sanity evidence. Both knobs are orthogonal in code (drift correction + distance change); cost is one extra canonical run. Decision: launch if B1 and A2.1 canonicals both ship without major control regression.
5. **C1 (KS GoF diagnostic)** — runs alongside B1/A2 (no compute). Same role as before: becomes the new promotion decision rule if it agrees with heuristics on retroactive v3 data.
6. **A1 (textbook FHS)** — runs after B1/A2 land. Still required as an attribution baseline (does the analog primitive add value over plain FHS?) and the V3.5.3 spot-check showed it solves a sub-pattern (COVID-style V-recoveries) cleanly — worth running formally, just no longer ahead of B1.
7. **B4 (regularized weight search)** — gated on whether B1+A2+B5 close the failure anchors. If they do, B4 isn't needed. If a residual val/test gap persists, run B4.
8. **B2 / B3** — scoped, execution deferred. Re-evaluate after the top block lands.

End of v4 decision: if all of {B1, A2, B2} fail to close `acf_seam_degradation` (rule still firing at < −0.30) *and* fail to recover the 5 fat-tail failure anchors (V3.5 failure set: 90%-band ≥45/60 days in ≥3 of 5), the analog-block primitive's structural ceiling is confirmed and v5 must consider abandoning intact 10-day blocks (finer-granularity sampling, or parametric / DL path generation). The v3.5 evidence ruled out the "pool is empty" stop condition, so this end-of-v4 decision will be substantive regardless of outcome.

---

## Reading on results

Each v4 experiment gets its own `_<id>_<name>.md` report following the v3 convention (status, setup, headline numbers, mechanistic reading, decision-rule verdict, implication for the roadmap, deliverables). Aggregate findings rolled up into [`V4_RESULTS.md`](V4_RESULTS.md) (closed 2026-05-22).

Per-experiment reports:
- [`experiments/_b1_local_linear.md`](experiments/_b1_local_linear.md) — Platzer local-linear correction
- [`experiments/_a2_corrwindow_L100.md`](experiments/_a2_corrwindow_L100.md) — corrwindow distance (L=100, n_eff=50)
- [`experiments/_b5_joint.md`](experiments/_b5_joint.md) — A2.1+B1 joint
- Auto-generated fat-tail panel summaries: `_<exp>_fat_tail.md` next to each above.

## Mandatory fat-tail evaluation

**Every v4 experiment that produces a forecast must report the [`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md) panel** — 15 anchors (5 extreme-positive z₅₀, 3 extreme-negative z₅₀, 7 hand-curated regime-coverage), 60-day forecast vs realized, coverage table, per-anchor CRPS diff vs v2.4 baseline. The aggregate CRPS / PIT diagnostics are necessary but not sufficient: v3 surfaced that the analog primitive systematically misses regime-transition rallies, and aggregate metrics average those misses away. **An experiment that improves aggregate CRPS but regresses on >2 fat-tail anchors is not promotable without explicit justification.**

The 15-anchor list is pinned at `results/analog_mc/data/fat_tail_eval_anchors.json`. Use `scripts/analog_mc/select_fat_tail_anchors.py` to regenerate after a canonical re-run; `scripts/analog_mc/plot_forecast_from_date.py --date <ISO>` to render per-anchor charts.

Each experiment-producing-a-forecast (A1, A2, B1, B2, B3 below) lists this panel as a required deliverable. The diagnostic-only experiment C1 does not produce forecasts and is exempt.

## What v4 explicitly does not test

Re-iterating to prevent scope creep:
- **No multi-asset experiments.** RELATED_WORKS Tier 4's PC-then-FHS sequencing is interesting but only matters if we move to joint multi-asset forecasting — out of scope here.
- **No CRPS-as-loss / learned-similarity experiments** (RELATED_WORKS borrowables from Tier 5). These require abandoning the analog primitive's interpretability — v5 questions.
- **No transaction costs, position sizing, or PnL.** These belong downstream, not in this pipeline.
- **No optimizer changes** (BayesOpt over grid). V3_PLAN's discipline on this stands.
