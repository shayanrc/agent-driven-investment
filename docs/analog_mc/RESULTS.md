# analog_mc — Results Lookup

Quick-reference dashboard for every walk-forward run that has shaped a v1/v2 decision. Each section: headline numbers, decision-rule verdicts, key plots, and pointers to the persisted artifacts. Update this file when a new acceptance run lands.

| Status legend | meaning |
|---|---|
| ✅ PASS | acceptance criterion satisfied |
| ✗ FAIL | acceptance criterion violated |
| 🔥 FIRED | v2-trigger decision rule fired (action needed) |
| ✅ ok | v2-trigger decision rule did not fire |

---

## TL;DR — current canonical

**v2.1 canonical** (`runs/analog_mc/20260517T145344Z/`, `configs/analog_mc/default.yaml` with `drift_mode: trailing_momentum`) is the current default. It ties v1 on aggregate CRPS, beats v1 high-vol CRPS by 5.2%, and eliminates the sloped-PIT firing. v2.2 conditional block sampling was implemented and tested but **did not address its target rule** — see the v2.2 audit section below.

> **🔬 Ablation update (2026-05-18):** the 2×2 `(drift, conditional)` decomposition in [`ABLATIONS.md`](ABLATIONS.md) finds that **conditional sampling alone (Cell C) dominates aggregate CRPS** (−7.8% vs v1, −9.5% vs v2.1 default, −4.7% vs v2.2). Drift's only validated effect is PIT calibration; on aggregate CRPS it slightly hurts. The freeze of v2.1 as default may be revisited — see ABLATIONS.md "Conclusions" for the three-way decision matrix.

| | v1 canonical (zero) | **v2.1 canonical (DEFAULT)** | v2.2 fast (drift + conditional) |
|---|---|---|---|
| Mean aggregate CRPS | 0.05246 | **0.05265** | 0.05045 |
| High-vol CRPS | 0.0911 | **0.0864** (−5.2%) | 0.0826 |
| `sloped_global_pit` | +0.158 🔥 | **+0.053 ✅** | +0.059 ✅ |
| `acf_seam_degradation` | −1.071 🔥 | −1.073 🔥 (unchanged) | −1.121 🔥 (slightly worse) |
| Wall time | 3h 47m | 3h 43m | 6h 30m (fast preset!) |
| Status | archived | **shipped as default** | implemented, deferred to v3 |

---

## Run index

| Run | Config | Folds | Mean CRPS | Wall time | Purpose |
|---|---|---|---|---|---|
| `20260516T170018Z` | `nasdaq100_fast.yaml` | 76 | 0.0521 | ~50 min | v1 fast proxy — first end-to-end smoke |
| `20260516T180000Z` | `default.yaml` (zero drift) | 76 | 0.05246 | 3h 47m | v1 canonical baseline (archived) |
| `20260517T050831Z` | `nasdaq100_v21.yaml` | 76 | 0.05313 | ~32 min | v2.1 fast acceptance |
| **`20260517T145344Z`** | **`default_v21.yaml`** | **76** | **0.05265** | **3h 43m** | **v2.1 canonical → new default** |
| `20260517T070003Z` | `nasdaq100_v22.yaml` | 76 | 0.05045 | 6h 30m | v2.2 fast acceptance (FAILED on target rule) |

---

## v1 canonical baseline (`runs/analog_mc/20260516T180000Z/`)

**Config:** `configs/analog_mc/default.yaml` (the *original* zero-drift form) — 76 folds, 273,120 origin × step pairs, 1000 paths per forecast, 66-point weight grid × 5 n_eff values, Nelder-Mead local refine.

### Headline numbers

| | Value |
|---|---|
| Mean aggregate OOS CRPS | **0.05246** |
| Median aggregate OOS CRPS | 0.03121 |
| Per-step CRPS (h=1 / 15 / 30 / 60) | 0.0088 / 0.0340 / 0.0525 / 0.0905 |
| Fixed-weight (⅓,⅓,⅓, n_eff=30) baseline | 0.06182 (tuned beats by **+17.84%**) |
| Low-vol regime mean CRPS | 0.0277 |
| Mid-vol regime mean CRPS | 0.0392 |
| **High-vol regime mean CRPS** | **0.0911** ← drove v2 scope |

### v2-trigger decision rules

| Rule | Status | Metric | Threshold | Implication |
|---|---|---|---|---|
| `sloped_global_pit` | 🔥 FIRED | +0.158 | ±0.10 | → v2.1: trailing-momentum drift |
| `acf_seam_degradation` | 🔥 FIRED | −1.071 | −0.30 | → v2.2: conditional block sampling |
| `u_shaped_high_vol_pit` | ✅ ok | +2.190 | 2.50 | tail inflator stays deferred |
| `fixed_weight_close_to_tuned` | ✅ ok | +0.178 | 0.01 | per-fold search earns its keep |
| `clip_hit_excessive` | ✅ ok | +0.100 | 0.15 | vol clip bounds OK |

### Key plots

| Plot | Reading |
|---|---|
| ![Global PIT](figs/v1_canonical_global_pit.png) | **Global PIT histogram.** The left-leaning slope is `sloped_global_pit` firing — realized returns systematically exceed the forecast median. Drift estimator missing. |
| ![Conditional PIT](figs/v1_canonical_conditional_pit.png) | **PIT by vol regime.** Low-vol is well-calibrated; high-vol shows the U-shape that almost (but not quite) trips `u_shaped_high_vol_pit`. |
| ![ACF comparison](figs/v1_canonical_acf_comparison.png) | **Squared-return ACF, simulated vs realized.** Simulated ACF is essentially flat at every lag while realized decays from +0.27 to +0.08. The "seam degradation" rule reads worst-case at seam lags (10/20/30/40/50), but the gap exists everywhere — see the v2.2 audit. |
| ![Weight trajectory](figs/v1_canonical_weight_trajectory.png) | **Per-fold optimal (w0, w1, w2)** over the 76 folds. Healthy: the three weights swap dominance over time. |
| ![Reliability diagram](figs/v1_canonical_reliability.png) | **Reliability.** Predicted vs empirical exceedance for chosen quantiles. Mostly on the diagonal, with the left-tail divergence consistent with the PIT slope. |
| ![Clip-hit summary](figs/v1_canonical_clip_hit_summary.png) | **Vol-clip hit rates.** The 0.5/3.0 clip bounds are rarely binding — `clip_hit_excessive` doesn't fire. |

---

## v2.1 canonical — **current default** (`runs/analog_mc/20260517T145344Z/`)

**Config:** `configs/analog_mc/default_v21.yaml` (also reflected in the live `default.yaml`) — 76 folds, 273,120 origin × step pairs, 1000 paths per forecast, 66-point weight grid × 5 n_eff values, Nelder-Mead local refine. **Only difference from v1 canonical: `drift_mode: trailing_momentum`** with `momentum_lookback=20`, `momentum_shrinkage=0.5`.

### Headline numbers vs v1 canonical

| Metric | v1 canonical (zero) | **v2.1 canonical (trailing_momentum)** | Δ |
|---|---|---|---|
| Mean aggregate CRPS | 0.05246 | **0.05265** | +0.36% (within noise) |
| Median CRPS | 0.03121 | 0.03132 | +0.35% |
| Low-vol CRPS | 0.0277 | 0.0308 | +11.2% |
| Mid-vol CRPS | 0.0392 | 0.0411 | +4.9% |
| **High-vol CRPS** | **0.0911** | **0.0864** | **−5.2% ← the win** |
| h=1 / h=15 / h=30 / h=60 | 0.0088 / 0.0340 / 0.0525 / 0.0905 | 0.0088 / 0.0345 / 0.0529 / 0.0897 | similar |
| Fixed-baseline vs tuned | +17.84% | **+21.33%** | tuning more valuable under v2.1 |
| `sloped_global_pit` | +0.158 🔥 | **+0.053 ✅** | drift eliminated PIT slope |
| `u_shaped_high_vol_pit` | +2.190 ✅ | +1.821 ✅ | further from firing |
| `acf_seam_degradation` | −1.071 🔥 | −1.073 🔥 | unchanged (drift doesn't touch ACF, as expected) |
| `fixed_weight_close_to_tuned` | +0.178 ✅ | +0.213 ✅ | tuning earns more |
| `clip_hit_excessive` | +0.100 ✅ | +0.099 ✅ | flat |

### Acceptance criteria

| Criterion | Target | v2.1 canonical | Verdict |
|---|---|---|---|
| `sloped_global_pit` not firing | within ±0.10 | +0.053 | ✅ PASS |
| Mean CRPS within +5% of v1 canonical | ≤ 0.0551 | 0.05265 | ✅ PASS |
| High-vol CRPS not worse than v1 canonical | ≤ 0.0957 | 0.0864 | ✅ PASS (5.2% improvement) |

### Fast-preset agreement (robustness check)

The fast preset (`nasdaq100_v21.yaml`, 21-point grid × 2 n_eff, 500 paths) is a faithful proxy:

| | Fast | Canonical |
|---|---|---|
| Mean CRPS | 0.05313 | 0.05265 |
| High-vol CRPS | 0.0862 | 0.0864 |
| `sloped_global_pit` | +0.057 ✅ | +0.053 ✅ |
| All other rule verdicts | identical | identical |

### Key plots

| Plot | Reading |
|---|---|
| ![v2.1 canonical global PIT](figs/v2_1_canonical_global_pit.png) | **Global PIT — slope gone.** Histogram is much closer to uniform than v1 canonical. `sloped_global_pit` now +0.053 (well inside ±0.10). |
| ![v2.1 canonical conditional PIT](figs/v2_1_canonical_conditional_pit.png) | **PIT by vol regime — high-vol calibration narrowed.** High-vol U-shape shallower than v1 canonical. |
| ![v2.1 canonical ACF](figs/v2_1_canonical_acf_comparison.png) | **ACF — unchanged from v1.** Drift correction doesn't affect squared-return autocorrelation. The remaining gap is the v2.2/v3 problem. |
| ![v2.1 canonical weight trajectory](figs/v2_1_canonical_weight_trajectory.png) | **Per-fold weights** still diverse — drift didn't make the matcher irrelevant. |
| ![v2.1 canonical reliability](figs/v2_1_canonical_reliability.png) | **Reliability** essentially on the diagonal. |

### Forecast distribution vs realized series

Comparison at three origins from fold 50, chosen at the 10/50/90th percentile of forecast-window vol. Same origins in both rows so the v1-vs-v2.1 difference is the drift, not the data.

![Forecast vs realized — v1 vs v2.1](figs/forecast_vs_realized_v1_vs_v21.png)

Black is realized, coloured line is the forecast median, dark band is 50% credible, light band is 90% credible, thin lines are 30 sample paths. v2.1's median tilts slightly toward recent trailing-mean returns — most visible in the high-vol panel.

Re-render with:
```bash
uv run python scripts/plot_forecast_vs_realized.py \
    --v1-run runs/analog_mc/20260516T180000Z \
    --v2-run runs/analog_mc/20260517T145344Z \
    --out docs/analog_mc/figs/forecast_vs_realized_v1_vs_v21.png \
    --fold-index 50
```

---

## v2.2 acceptance — **deferred to v3** (`runs/analog_mc/20260517T070003Z/`)

**Config:** `configs/analog_mc/nasdaq100_v22.yaml` — fast preset with `drift_mode: trailing_momentum` AND `conditional_block_sampling: true`. Used the test-only contingency (`conditional_block_sampling_in_search: false`) per V2_PLAN open-question-7 because conditional sampling at search-time was intractable (~19 days projected). v2.1 and v2.2 picked **identical (weights, n_eff) at all 76 folds** (deterministic blake2b seeding), so the comparison is a clean A/B isolated to test-time sampling.

### Headline numbers vs v2.1

| Metric | v2.1 fast | v2.2 fast | Δ |
|---|---|---|---|
| Mean aggregate CRPS | 0.05313 | **0.05045** | **−5.0%** ← real gain |
| High-vol CRPS | 0.0862 | 0.0826 | −4.2% |
| `sloped_global_pit` | +0.057 ✅ | +0.059 ✅ | unchanged (drift handles it) |
| **`acf_seam_degradation`** | **−1.056 🔥** | **−1.121 🔥** | **slightly worse — target rule did NOT improve** |
| `u_shaped_high_vol_pit` | +1.768 | +1.612 | further from firing |
| Tuned vs fixed baseline | +20.28% | +21.21% | same magnitude |
| Per-fold weight agreement vs v2.1 | n/a | **76/76 identical** | clean A/B |

### Acceptance criteria

| Criterion | Target | v2.2 result | Verdict |
|---|---|---|---|
| `acf_seam_degradation` not firing | metric ≥ −0.30 | **−1.121 🔥** | **✗ FAIL** |
| Mean CRPS not worse than v2.1 | ≤ 0.0558 | 0.05045 | ✅ PASS |
| High-vol CRPS not worse than v2.1 | ≤ 0.0862 | 0.0826 | ✅ PASS |

### Audit conclusion — why v2.2 can't fix the target rule

Per-lag comparison after the run:

| Lag | Type | v2.1 ACF | v2.2 ACF | Δ |
|---|---|---|---|---|
| 1 | within-block | −0.003 | +0.006 | +0.009 |
| 5 | within-block | +0.007 | +0.015 | +0.008 |
| 10 | **seam** | −0.0002 | **−0.012** | **−0.012** |
| 15 | within-block | −0.016 | −0.014 | +0.002 |
| 20 | **seam** | −0.004 | **−0.016** | **−0.012** |
| 25 | within-block | −0.016 | −0.017 | −0.001 |
| 30 | **seam** | −0.006 | **−0.016** | **−0.010** |
| 40 | **seam** | −0.005 | **−0.012** | −0.007 |
| 50 | **seam** | −0.003 | −0.006 | −0.003 |

Both runs have simulated ACF essentially **flat at every lag**, far from the realized 0.27 → 0.08 curve. The "seam degradation" metric reads the worst seam lag, but the gap is the same magnitude at non-seam lags. v2.2 nudged seam lags more negative (chained similar-vol blocks smooth seam transitions, slightly lowering the squared-return cross-product); within-block lags are essentially unchanged.

**Structural root cause** — direct evidence from real returns:

| | Real data |
|---|---|
| Unconditional lag-1 ACF(r²) | **+0.271** (GARCH-driven, slowly-varying vol) |
| Within-10-day-window lag-1 ACF(r²) | **−0.125** (within-window structure after demeaning) |
| v2.2 simulated lag-1 ACF | −0.011 (closer to within-window) |

Any sampler that draws 10-day analog blocks intact (v1, v2.1, v2.2 all do) inherits the within-window structure and **cannot** reproduce the unconditional ACF. v2.2's per-block re-matching only changes what happens at seams; it cannot put GARCH dynamics inside a block. Fixing this needs per-step σ injection or GARCH-conditional resampling — **v3 scope**.

### Implementation audit verdict

| Check | Result |
|---|---|
| Tail-buffer warm-start (`max_h − block_length` real + `block_length` sim = 200) | ✅ |
| Z-score window slicing matches `causal_zscore` convention | ✅ |
| `composite_distance_batched` per-row equivalence | ✅ unit-tested |
| `distances_to_probs_batched` (vectorized bisection) hits n_eff within 5% | ✅ unit-tested |
| Per-path categorical sampling (cumsum + uniform) | ✅ |
| `mu_origin` constant per forecast (C3) | ✅ |
| `drift_target` constant per forecast (C10) | ✅ |
| Drift added after ratio multiplier (C7) | ✅ |
| EWMA σ recursion (C4) | ✅ same as v1 |
| Candidate set unchanged across blocks (C6) | ✅ open-question-5 resolution documented |
| A/B fairness — same weights as v2.1 | ✅ 76/76 folds identical |

**No bugs found.** The implementation correctly does per-block per-path conditional re-matching as specified. The design hypothesis was wrong: the rule label `acf_seam_degradation` is misleading — the gap is global, not seam-specific.

### What v2.2 ships and what it doesn't

- **Ships:** the implementation (`generate_paths_conditional`), tests (7), config flag (`conditional_block_sampling`), batched solver (`distances_to_probs_batched`), test-only contingency (`conditional_block_sampling_in_search`), preset (`nasdaq100_v22.yaml`). All code remains in the repo as an opt-in mode.
- **Doesn't ship as default:** `default.yaml` keeps `conditional_block_sampling: false` because (a) the target rule still fires, (b) the test-eval cost is ~12× v1, (c) the 5% CRPS gain isn't attributable to fixing volatility clustering.

### Key plots

| Plot | Reading |
|---|---|
| ![v2.2 global PIT](figs/v2_2_global_pit.png) | PIT still uniform (drift correction held up). |
| ![v2.2 conditional PIT](figs/v2_2_conditional_pit.png) | Vol-regime PIT similar to v2.1; no degradation. |
| ![v2.2 ACF](figs/v2_2_acf_comparison.png) | Simulated ACF still flat — visually identical to v1/v2.1. |

---

## How to regenerate

| Artifact | Command |
|---|---|
| Walk-forward run | `uv run python -m analog_mc walk-forward --config configs/analog_mc/<preset>.yaml` |
| Diagnostics + decision rules | `uv run python scripts/render_diagnostics.py runs/analog_mc/<timestamp>` |
| Forecast-vs-realized fans | `uv run python scripts/plot_forecast_vs_realized.py --v1-run <a> --v2-run <b> --out <png>` |

---

## Pointers

- Pipeline spec: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- v2 spec + v3 carryover: [`V2_PLAN.md`](V2_PLAN.md)
- Algorithm math: [`ALGORITHM.MD`](ALGORITHM.MD)
- Decision rules code: `src/analog_mc/diagnostics.py::decision_rules`
- All persisted run artifacts: `runs/analog_mc/<timestamp>/` (per-fold `forecasts.npz`, `summary.json`, `search_grid.parquet`, plus `summary.parquet`, `meta.json`, `diagnostic_report.json`, `figs/`)
</content>
</invoke>