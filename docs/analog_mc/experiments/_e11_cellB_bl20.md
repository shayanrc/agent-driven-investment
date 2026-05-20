# E11 — Cell B × bl=20 (drift + bl=20, no conditional sampling)

V3 experiment 11 (post-E10 follow-up; see [`V3_PLAN.md`](V3_PLAN.md#e11-cell-b--bl20-drift--bl20-no-conditional)). Tests whether bl=20 stacks with **drift alone** (it doesn't stack with conditional sampling per E10) — if yes, Cell B + bl=20 could displace Cell D as a simpler default.

## Setup

Single fast-preset run, `configs/analog_mc/ablation_E11_cellB_bl20.yaml`. Cell B config (drift on, conditional sampling off) with `block_length=20, n_blocks=3`. Run dir: `runs/analog_mc/20260519T144721Z`. Wall **32 min** (no conditional → no slow test eval; ~22 s/fold pace, like E1).

## Headline numbers — four-way comparison

| | Mean CRPS | Low-vol | Mid-vol | High-vol | drift | cond | bl |
|---|---|---|---|---|---|---|---|
| B-fast (drift only) | 0.05313 | 0.0308 | 0.0411 | 0.0862 | ✓ | ✗ | 10 |
| E1-bl20 (zero drift) | 0.05063 | 0.0271 | 0.0386 | 0.0865 | ✗ | ✗ | 20 |
| **E11 (drift, bl=20)** | **0.05175** | 0.0307 | 0.0407 | **0.0843** | ✓ | ✗ | 20 |
| **D-fast (drift+cond)** | **0.05041** | 0.0293 | 0.0398 | 0.0826 | ✓ | ✓ | 10 |

## Verdict

**FAIL all PASS criteria.** E11 (0.05175) does NOT displace Cell D as the default candidate:

| Criterion | Target | E11 | Verdict |
|---|---|---|---|
| Mean CRPS within 1% of Cell D | ≤ 0.0510 | 0.0518 | ✗ FAIL (+2.6%) |
| `sloped_global_pit` ≤ +0.10 | ≤ +0.10 | **+0.045** | ✅ PASS |
| High-vol CRPS ≤ 0.0835 | ≤ 0.0835 | 0.0843 | ✗ FAIL (+1.0%) |
| No rule regression | n/a | none | ✅ PASS |

### Decision-rule details

| Rule | B-fast | E1-bl20 | **E11** | D-fast | threshold |
|---|---|---|---|---|---|
| `sloped_global_pit` | +0.057 ✅ | +0.141 🔥 | **+0.045 ✅** | +0.059 ✅ | ±0.10 |
| `u_shaped_high_vol_pit` | +1.768 ✅ | +1.973 ✅ | **+1.748 ✅** | +1.612 ✅ | +2.50 |
| `acf_seam_degradation` | −1.056 🔥 | −1.115 🔥 | **−1.116 🔥** | −1.121 🔥 | −0.30 |
| `clip_hit_excessive` | +0.101 ✅ | +0.097 ✅ | **+0.095 ✅** | +0.099 ✅ | +0.15 |

E11's PIT calibration is *the tightest* of all four cells. Drift + bl=20 produces a slightly better PIT than drift alone (E11 +0.045 vs B-fast +0.057). But the aggregate-CRPS gap to Cell D is too large to make E11 a viable default candidate.

The aggregate-CRPS criterion fails by 2.6%, larger than the noise floor (E3 ≈ 0.08%). High-vol CRPS also misses target by 1%.

## Mechanism reading

E11 beats B-fast by 2.6% (drift+bl=20 better than drift+bl=10) but loses to D-fast by 2.7%. Compare to:

- E1: bl=20 vs bl=10 at zero drift → −2.9% (bl=20 wins)
- E11: bl=20 vs bl=10 at drift only → −2.6% (bl=20 wins)
- E10: bl=20 vs bl=10 at drift+cond → +0.3% (flat — conditional absorbed it)

**The bl=20 gain stacks with drift but NOT with conditional sampling.** Conditional sampling and bl=20 both reduce variance leakage at block transitions and are largely substitutable. Drift is independent and provides its own (smaller) gain.

Cell D's edge over E11 (2.7%) is purely the conditional-sampling contribution. That contribution is real and not duplicative with bl=20 — confirms Cell D's conditional sampling earns its complexity.

E11 also slightly degrades from E1-bl20 (zero drift, bl=20): drift costs ~2% aggregate CRPS even at bl=20, same trade-off documented in v2.1 acceptance. Likely buys back PIT calibration (which is the whole point of drift).

## What this rules out

1. **No simpler default than Cell D.** The "remove conditional sampling, lean on bl=20 + drift" alternative is now empirically falsified at fast preset.
2. **bl=20's CRPS gain at zero drift was a real but non-additive effect.** It plugs the same variance leak that conditional sampling addresses.
3. **Drift's PIT cost is roughly constant** (~2% aggregate CRPS) regardless of block length. The E2 momentum sweep should target this cost.

## Implication for E2

E2's shrinkage sweep is the right next experiment: if `momentum_shrinkage=0.5` is over-shrunk, a smaller value could buy back the 2% drift cost. With E11 anchoring "drift+bl=10 at 0.5 shrinkage = 0.0531" and "drift+bl=20 at 0.5 shrinkage = 0.0518", E2 will sweep at bl=10 (the production block size) to find the optimal shrinkage on the production matcher.

## Deliverables

- `configs/analog_mc/ablation_E11_cellB_bl20.yaml`
- `runs/analog_mc/20260519T144721Z/`
- This page.
