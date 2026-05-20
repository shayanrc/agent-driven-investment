# E9 — v3b GARCH-conditional resampling

V3 experiment 9 (per [`V3_PLAN.md`](V3_PLAN.md#e9-v3b--garch-conditional-resampling-promoted-by-e1)). Tests whether replacing the block-constant σ ratio with a GARCH(1,1)-simulated per-step σ trajectory closes the `acf_seam_degradation` rule that has fired since v1.

## Status

⚠️ **Negative verdict — partial.** E9-A run (GARCH + zero drift + no conditional) showed:
- **ACF rule does not close.** Simulated within-block lag-1 ACF nudges from −0.004 (E1-bl10) to +0.016 (E9-A) — directionally correct but ~17× below the realized +0.27. Seam lags barely move.
- **Small CRPS win, mainly in high-vol.** Mean CRPS 0.05101 vs E1-bl10's 0.05215 (−2.2%), driven by high-vol regime −3.8%.

E9-D (Cell D + GARCH) was **skipped** — given E9-A's ACF result, the expected outcome on top of Cell D is either flat (mechanism overlap, like E10 with bl=20) or +1–2% (independent GARCH contribution). Either way, less production-relevant than the Pareto s-sweep that's now the next experiment.

## Setup

Single E9-A run launched 2026-05-20 01:45 (after a relaunch — the initial run was 30× slower until per-origin GARCH-fit caching landed in `simulate.py::_garch_fit_cached`). Config `configs/analog_mc/ablation_E9_A_garch.yaml`, run dir `runs/analog_mc/20260519T201515Z`, wall **43 min**.

## Headline numbers

| | Mean CRPS | Low-vol | Mid-vol | High-vol |
|---|---|---|---|---|
| E1-bl10 (EWMA, zero drift) | 0.05215 | 0.0279 | 0.0401 | 0.0888 |
| **E9-A (GARCH, zero drift)** | **0.05101** | 0.0287 | 0.0395 | **0.0854** |
| Δ | **−2.2%** | +2.9% | −1.5% | **−3.8%** |
| Cell D (current default) | 0.05041 | 0.0293 | 0.0398 | 0.0826 |
| Cell-D-s25 (E2 winner) | 0.04731 | 0.0267 | 0.0370 | 0.0787 |

GARCH-conditional helps high-vol calibration (its actual mechanism — vol clustering tracked at the path level) but slightly hurts low-vol (mean-reversion in GARCH may overestimate residual noise during quiet periods). Net mean CRPS improvement is small.

## ACF — the headline failure

| Lag | Type | Realized | E1-bl10 (EWMA) | **E9-A (GARCH)** | Δ |
|---|---|---|---|---|---|
| 1 | within-block | +0.271 | −0.004 | **+0.016** | +0.020 |
| 5 | within-block | +0.256 | +0.008 | +0.016 | +0.008 |
| 10 | **seam** | +0.208 | −0.000 | **+0.004** | +0.004 |
| 15 | within-block | +0.152 | −0.016 | −0.016 | 0.000 |
| 20 | **seam** | +0.170 | −0.004 | −0.006 | −0.002 |
| 50 | **seam** | +0.083 | −0.003 | −0.005 | −0.002 |

GARCH moves within-block lags (1, 5) marginally positive. Seam lags (10, 20, 50) barely move or slightly degrade. The `acf_seam_degradation` rule will continue to fire — its target gap is 0.30 (the threshold) but the realized vs simulated gap at lag 1 is still **0.255** with GARCH (was 0.275 with EWMA).

## Why GARCH doesn't close the rule

V2_PLAN's audit identified the structural ceiling as: 10-day analog blocks drawn intact inherit the *within-window* squared-return ACF (−0.125 from demeaning), not the unconditional ACF (+0.27). E1 confirmed this is independent of block length.

E9's mechanism: replace block-constant σ-ratio with per-step σ-path from GARCH. **This rescales each return's magnitude** but **does not change which days are correlated within a block.** The analog's intra-block direction structure is preserved; only the volatility envelope changes. So:

- Vol clustering (the σ²_t autocorrelation) IS captured by the GARCH σ-path
- Return-squared autocorrelation (the metric the rule reads) is NOT — that requires either (a) sampling at finer granularity than 10-day blocks, or (b) parametric path generation entirely, abandoning the analog primitive

The +0.020 lag-1 improvement is the small contribution from σ-clustering: paths in high-vol windows have correlated-large returns even though the analog's direction is preserved.

## Implication for v3 roadmap

The `acf_seam_degradation` rule is **structurally unfixable within the analog-block primitive.** Both v3a (per-step EWMA σ, E4) and v3b (GARCH-conditional, E9) target σ-scaling and neither can recover the unconditional ACF. The rule should be reframed:

1. **Rename `acf_seam_degradation` → `acf_global_degradation`** (V2_PLAN carryover-2, was noted but not done).
2. **Mark the rule as "informational, not actionable"** within the analog-block architecture. Treat it like a documented limitation, not a v3 target.
3. **Future fixes must abandon intact 10-day blocks** — either by (a) bootstrapping at single-return granularity with a learned dependency structure, or (b) replacing the matcher entirely (parametric / DL path generation). This is a v4 architectural rewrite, not a v3 experiment.

E9 still **ships as an opt-in mode** (`vol_model: "garch"` in config). It's a small CRPS refinement (~2% high-vol) for use cases where vol-clustering calibration matters more than the modest cost. Not a default change.

## Decision-rule verdict (pending diagnostic_report)

Pending render_diagnostics completion. Expected (from manual ACF compute):

| Rule | Cell D | E9-A expected | Verdict |
|---|---|---|---|
| `sloped_global_pit` | +0.059 ✅ | high (zero drift) | 🔥 FIRED (as E1-bl10) |
| `acf_seam_degradation` | −1.121 🔥 | ≈ −1.10 | 🔥 FIRED (marginally less negative; rule remains failing) |
| `u_shaped_high_vol_pit` | +1.612 ✅ | ≤ +1.80 | ✅ ok |
| `clip_hit_excessive` | +0.099 ✅ | ≤ +0.12 | ✅ ok |

## Deliverables

- `src/analog_mc/vol.py` — GARCH wrapper (`fit_garch`, `simulate_garch_sigma_paths`)
- `src/analog_mc/sampling.py` — per-step σ rescaling in `generate_paths` and `generate_paths_conditional` (gated on `sigma_path` kwarg)
- `src/analog_mc/simulate.py::_garch_fit_cached` — per-origin GARCH fit cache (critical for performance)
- `src/analog_mc/config.py` — `vol_model: "ewma" | "garch"` knob
- `tests/analog_mc/test_vol.py` — 8 tests (parameter recovery, determinism, causality, EWMA-equivalence, ACF direction sanity)
- `configs/analog_mc/ablation_E9_{A,D}_garch.yaml` — E9-A used; E9-D config kept for reference but not run
- `runs/analog_mc/20260519T201515Z/` — E9-A run
- This page.

## v3 conclusion

After E1, E10, E3, E11, E2, Cell-D-s25, E9-A — the v3 phase has produced:

- **Confirmed promotion: vanilla Cell D as v2.3 default** (already shipped via E7)
- **Pending promotion: Cell-D-s25 family** (next: s-sweep for PIT-passing variant)
- **GARCH as an opt-in refinement** (small high-vol gain)
- **Architectural finding: `acf_seam_degradation` is unfixable within the analog-block primitive.** Carry to v4.
