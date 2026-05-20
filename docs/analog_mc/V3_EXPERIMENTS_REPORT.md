# analog_mc v3 — experiments synthesis

Closes [`V3_PLAN.md`](V3_PLAN.md). Aggregates the outcomes of E1–E11 (with extensions) and identifies the v3 promotion (v2.4) plus the architectural finding that carries to v4 ([`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md)). Companion to [`RELATED_WORKS.md`](RELATED_WORKS.md).

## Executive summary

**v3 shipped a 9.7% mean-CRPS improvement over v2.1 canonical (v2.1 → v2.4)** on the production NASDAQ100 walk-forward at canonical resolution (76 folds, 66×5 weight grid, 1000 paths).

| Stage | Default config | Mean CRPS | Δ vs prior | Δ vs v2.1 | Promotion driver |
|---|---|---|---|---|---|
| v2.1 canonical | trailing_momentum, shrinkage=0.5 | 0.05265 | — | — | (carried in from V2) |
| v2.3 canonical (Cell D) | + conditional sampling | 0.05056 | −4.0% | −4.0% | E7 (E3+E10 unblocked) |
| **v2.4 canonical (Cell-D-s30)** | shrinkage 0.5 → 0.30 | **0.04755** | **−6.0%** | **−9.7%** | **E2-ext canonical confirmation** |

**What shipped from v3.** (1) Conditional block sampling at test time (E7, v2.3). (2) Momentum-shrinkage retune from 0.5 → 0.30 (v2.4, this report). (3) GARCH-conditional vol path as opt-in `vol_model: "garch"` (E9-A; small high-vol CRPS refinement, not a default).

**What v3 falsified.** (a) Block-length geometry as an ACF-recovery lever (E1). (b) "bl=20 × Cell D" as an additive variance-leak fix (E10 — overlapping mechanism). (c) "drift + bl=20" as a simpler default than Cell D (E11 — drops 2.7% vs Cell D). (d) GARCH-conditional resampling as the v2.2 ACF-degradation fix (E9-A — moves lag-1 ACF from −0.004 to +0.016 vs realized +0.27; structurally insufficient).

**Architectural finding (v4 scope).** `acf_seam_degradation` is **structurally unfixable** within the analog-block primitive — both block-geometry (E1) and σ-conditioning (E9) approaches confirmed the ceiling. v3 reframes this rule from "actionable" to "documented limitation"; closing it requires v4 architecture changes (Platzer-style local-linear correction inside the analog primitive, or replacing intact 10-day blocks).

## Per-experiment table

| ID | Hypothesis | Verdict | CRPS Δ | Rule impact | Detail |
|---|---|---|---|---|---|
| **E1** | Shorter blocks → ACF recovery (v3c carryover) | ❌ Falsified | bl=20: −2.9% vs bl=10 (CRPS direction, not ACF) | `acf_seam` flat across bl ∈ {5,10,20} | [`experiments/_e1_block_length.md`](experiments/_e1_block_length.md) |
| **E2** | Lower shrinkage at Cell B beats s=0.50 | ✅ Confirmed | s=0.25 at Cell B: −4.9% vs s=0.50 | Pareto curve mapped | [`experiments/_e2_momentum.md`](experiments/_e2_momentum.md) |
| **E2-D-s25** | s=0.25 stacks with Cell D | ✅ CRPS, ⚠️ PIT | Cell D s25: −6.1% vs s=0.50; PIT +0.105 🔥 marginal | sloped_pit fires by 0.005 (fast) | same |
| **E2-ext** | Pareto sweet-spot in s∈{0.30,0.35,0.40} | ✅ s=0.30 found | s=0.30 (fast): −5.5%; PIT +0.0958 ✅ | All rules pass | same |
| **E2-canonical** | Confirm s=0.30 at canonical resolution | ✅ **v2.4 PROMOTED** | **−6.0% vs v2.3, −9.7% vs v2.1**; PIT +0.0216 ✅ | All non-structural rules pass | same |
| **E3** | Cell D gain robust to seed noise | ✅ Confirmed (early stop, 2-pt) | Gap 0.08% << 4% Cell D gain | All rules stable to ≤0.013 | [`experiments/_e3_seed_noise.md`](experiments/_e3_seed_noise.md) |
| **E7** | Promote Cell D as v2.3 default | ✅ Shipped | (config flip; gates passed) | same as Cell D fast | (folded into v2.3 commit) |
| **E9-A** | GARCH-conditional σ closes ACF rule | ❌ Falsified | −2.2% mean, −3.8% high-vol | lag-1 ACF: −0.004 → +0.016 (still fails) | [`experiments/_e9_v3b.md`](experiments/_e9_v3b.md) |
| **E9-D** | Cell D + GARCH would stack | ⏭️ Skipped (E9-A result) | — | — | same |
| **E10** | bl=20 stacks with Cell D | ❌ Falsified | E10: +0.3% (flat) vs Cell D | Mechanism overlap with cond. sampling | [`experiments/_e10_celld_bl20.md`](experiments/_e10_celld_bl20.md) |
| **E11** | drift+bl=20 displaces Cell D | ❌ Falsified | E11: +2.7% vs Cell D | bl=20 + drift can't substitute for cond. | [`experiments/_e11_cellB_bl20.md`](experiments/_e11_cellB_bl20.md) |

Aggregate JSON metrics: `results/analog_mc/data/_e{1,2,3,10}_data.json` and `_e2_canonical_Ds30_data.json`.

## Architectural findings

### Finding 1: ACF degradation is structural to the analog-block primitive

**Evidence.** Two independent v3 experiments target the seam-ACF gap from different mechanisms:

| Experiment | Mechanism targeted | Lag-1 ACF (sim vs realized) | Verdict |
|---|---|---|---|
| **E1** (block-length) | Block geometry — shorter blocks → less within-window structure inheritance | −0.002 → −0.004 across bl ∈ {5,10,20} | flat in all cells |
| **E9-A** (GARCH-cond) | Per-step σ replaces block-constant ratio | −0.004 → +0.016 | barely moves |
| (realized target) | — | **+0.271** | unreachable |

Both interventions assume the ACF gap is a *fixable knob* within the analog primitive. Both fail. The Platzer–Yiou theory (Tier-2 in `RELATED_WORKS.md`) explains why: analog blocks drawn intact inherit the *within-window* squared-return ACF (−0.125 from demeaning), not the unconditional ACF (+0.27). Re-scaling magnitudes (σ-conditioning) and varying block geometry both preserve the intra-block direction structure that determines the simulated ACF.

**Carry to v4.** The `acf_seam_degradation` rule is reframed from an *actionable* v3 target to a *documented limitation*. v4 candidates that could close it require structural changes:
- **B1 (Platzer local-linear correction)** — applies a Jacobian bias correction at the conditional-mean level. *In-primitive but novel.*
- v5 alternatives (deferred from v4 scope): single-return-granularity bootstrapping, or replacing analog matching with parametric/DL path generation.

### Finding 2: Variance-leak fixes overlap; only one earns its complexity

E10 (Cell D × bl=20) and E11 (drift + bl=20) together showed the bl=20 CRPS gain at zero drift (−2.9% in E1) is *non-additive* with conditional sampling — both reduce variance leakage at block transitions and share a mechanism. Conditional sampling is the more general fix (works at any block length) and earns its complexity over drift+bl=20 alone by 2.7%.

**Implication for v4.** When evaluating new variance-control mechanisms (e.g. Platzer local-linear, Dirichlet weight posteriors), explicitly check non-additivity with Cell D — a "small but real" win in isolation may collapse to flat when stacked.

### Finding 3: The default's hyperparameters were under-tuned

The shrinkage sweep (E2 + extension) found that the v2.1-inherited default (`momentum_shrinkage=0.5`) was a "half-Kelly" heuristic, not a tuned value. Optimization against actual data found s=0.30 as the Pareto sweet-spot — a **5.5–6.0% CRPS reduction available from one hyperparameter retune.** Generalizes to: other v2.1-era heuristic defaults are candidates for similar sweeps.

**Implication for v4.** A2 (OFTER max-correlation distance) tests the next-largest heuristic: the entire `(w_1, w_2, w_3, n_eff)` grid structure. If that delivers similar gains, the v3 retune pattern repeats.

## Decision-rule trajectory across v2.1 → v2.4

Headline shifts in the four trigger rules across the promotion chain. **Bold** = the rule that *moved most* at each step.

| Rule | Threshold | v2.1 | v2.3 (Cell D) | **v2.4 (Cell-D-s30)** |
|---|---|---|---|---|
| `sloped_global_pit` | ±0.10 | +0.13 🔥 (zero drift) → fixed by trailing_momentum | +0.059 ✅ | **+0.0216 ✅** |
| `u_shaped_high_vol_pit` | +2.5 | +1.91 ✅ | +1.61 ✅ | **+0.963 ✅** |
| `acf_seam_degradation` | −0.30 | −1.05 🔥 | −1.12 🔥 (worse) | −1.12 🔥 (structural) |
| `clip_hit_excessive` | +0.15 | +0.099 ✅ | +0.099 ✅ | +0.106 ✅ |

**v2.4 reduces every PIT-family rule.** The `sloped_global_pit` improvement v2.3 → v2.4 (+0.059 → +0.022) demonstrates that the shrinkage retune *also* tightens calibration on top of the Cell D win — not just CRPS. `u_shaped_high_vol_pit` halves. `acf_seam_degradation` is unchanged and is now documented as structural; `clip_hit_excessive` ticks up marginally but stays well under threshold.

## v3 final state

- **Production default** (`configs/analog_mc/default.yaml`): Cell D + shrinkage=0.30 (v2.4).
- **Opt-in feature**: `vol_model: "garch"` for high-vol-sensitive use cases (E9-A; +1–2% high-vol CRPS at the cost of GARCH fit per fold).
- **Confirmed dead-ends**: block-length geometry (E1, E10, E11) and σ-conditioning (E9) as ACF levers.
- **Confirmed defaults that v3 did NOT change**: `block_length=10`, `n_blocks=6`, `conditional_block_sampling_in_search=false`, `vol_clip_lower=0.5`, `vol_clip_upper=3.0`, the n_eff grid `[15, 30, 50, 80, 150]`, the 3-z-score state `(z₂₀, z₅₀, z₂₀₀)`.

## Carry-overs to v4

See [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md) for the structured plan. Headline carry-overs from v3:

1. **A1 (textbook FHS baseline)** — E9-A's CRPS gain is currently unattributable between σ-conditioning and analog-block selection. Highest-priority attribution gap.
2. **B1 (Platzer local-linear correction)** — only experiment in scope that could close `acf_seam_degradation` inside the analog primitive without abandoning intact blocks.
3. **C1 (block-bootstrap KS/PIT GoF)** — replaces the heuristic ±0.10 PIT-slope threshold (which the v3 Cell-D-s30 fast-preset judgment hinged on by 0.005) with a formal p-value.
4. **Rename `acf_seam_degradation` → `acf_global_degradation`** (V2 carryover-2, deferred again — should land with B1 if B1 reshapes the metric, otherwise as a standalone rename).
5. **Multi-asset robustness (E8)** — V2 carryover-3, still pending. v3 worked NASDAQ-only; defer until v4 architectural questions resolve.

## References

### v3 experiment reports

- [`experiments/_e1_block_length.md`](experiments/_e1_block_length.md) — E1 block-length sweep
- [`experiments/_e2_momentum.md`](experiments/_e2_momentum.md) — E2 momentum shrinkage sweep + extension + canonical confirmation
- [`experiments/_e3_seed_noise.md`](experiments/_e3_seed_noise.md) — E3 seed noise floor
- [`experiments/_e9_v3b.md`](experiments/_e9_v3b.md) — E9-A GARCH-conditional resampling
- [`experiments/_e10_celld_bl20.md`](experiments/_e10_celld_bl20.md) — E10 Cell D × bl=20
- [`experiments/_e11_cellB_bl20.md`](experiments/_e11_cellB_bl20.md) — E11 Cell B × bl=20

### Aggregate metrics (machine-readable)

- `results/analog_mc/data/_e1_data.json`
- `results/analog_mc/data/_e2_data.json`
- `results/analog_mc/data/_e2_canonical_Ds30_data.json` (canonical v2.4 confirmation)
- `results/analog_mc/data/_e3_data.json`
- `results/analog_mc/data/_e10_data.json`

### Companion documents

- [`V3_PLAN.md`](V3_PLAN.md) — v3 spec (sets the questions this report answers)
- [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md) — v4 spec (what comes next)
- [`RELATED_WORKS.md`](RELATED_WORKS.md) — 5-tier literature survey (motivates V4 directions)
- [`ABLATION_STUDIES_REPORT.md`](ABLATION_STUDIES_REPORT.md) — Pre-v3 ablation (Cells A/B/C/D)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — Canonical pipeline spec (C1–C6)
- [`RESULTS.md`](RESULTS.md) — User-facing results summary (to be updated by next session to reflect v2.4 promotion)
