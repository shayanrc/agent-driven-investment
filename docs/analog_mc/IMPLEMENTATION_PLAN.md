# Analog Monte Carlo Forecasting Pipeline — Implementation Plan

## Build status

All 11 v1 stages + v2.1 (trailing-momentum drift) + v2.2 (conditional block sampling) + v2.3 (Cell D promotion) are implemented, unit-tested, and shipped. **The canonical default is now Cell D from the 2×2 ablation: drift + conditional block sampling (test-time only).** Promotion gates (E3 seed noise floor, E10 bl=20 alternative) passed 2026-05-19; see V3_PLAN.md, RESULTS.md, and the v2.3 entry below. Concrete artifacts:

- `src/analog_mc/{config,data,features,distances,sampling,simulate,scoring,search,walk_forward,diagnostics}.py`
- CLI entry: `python -m analog_mc walk-forward --config <yaml>` (used by the run-experiment dashboard view)
- `dashboards/analog_mc/{app.py, views/{config_editor, run_experiment, diagnostics}.py}` + global launcher `dashboards/app.py`
- 5 v2-trigger decision rules in `diagnostics.py` per Stage 9 of this plan
- Crash-resumable per-fold persistence (parquet for search grid, compressed npz for forecast paths + σ ratios + realized, JSON summaries)

**Canonical v1 baseline (archived):** `runs/analog_mc/20260516T180000Z/` (`default.yaml` with the original `drift_mode: zero`, 76 folds, 273,120 origin × step pairs, mean test CRPS 0.05246, 3h 47m). Two v2 triggers fired (`sloped_global_pit`, `acf_seam_degradation`).

**Canonical v2.1 baseline (archived):** `runs/analog_mc/20260517T145344Z/` (`default_v21.yaml`, mean test CRPS 0.05265, 3h 43m). `sloped_global_pit` no longer fires (+0.158 → +0.053); high-vol-regime CRPS improved 5.2% (0.0911 → 0.0864) vs v1. `acf_seam_degradation` still fires unchanged — see RESULTS.md v2.2 audit for why it can't be fixed by per-block re-matching.

**Canonical v2.3 baseline (current default):** `runs/analog_mc/20260518T155800Z/` (`default_v22.yaml`, identical to the live `default.yaml` after v2.3 promotion — 76 folds, 273,120 origin × step pairs, mean test CRPS 0.05056, 7h 36m). Cell D = trailing-momentum drift + conditional block sampling. −4.0% mean CRPS vs v2.1 canonical, −3.9% high-vol CRPS, PIT calibration preserved. `acf_seam_degradation` still fires (structural ceiling, v3 scope).

**Results lookup:** [`RESULTS.md`](RESULTS.md) is the quick-reference dashboard for every walk-forward run that has shaped a v1/v2 decision — headline numbers, decision-rule verdicts, key plots inline, and pointers to persisted artifacts. Check it first before re-deriving anything from raw run directories.

For the end-to-end run command and high-level architecture, see the project `README.md` at the repo root.

## Revision history

- **v1.1** — Revised C3 (per-analog vol scaling). The original per-block demean (`raw_block.mean()`) caused every Monte Carlo path to collapse to a point mass at zero cumulative log return at every block boundary (h = 10, 20, 30, 40, 50, 60), destroying calibration diagnostics at those horizons. Replaced with a single shared baseline per forecast (`mu_origin` = trailing causal mean at the forecast origin over the longest z-score horizon). Each analog block is demeaned against the same constant, preserving its deviation from current regime drift. See C3 for details and the architectural diagram step 2e. Stage 1 also adds `causal_trailing_mean` to `features.py`.

- **v2.1** — Added trailing-momentum drift. `forecast()` reads `drift_mode` from config; when `"trailing_momentum"`, drift is the shrunk recent-mean estimate at the origin, applied per C7 (after σ ratio) and C10 (constant per forecast). `compute_features` adds a second trailing-mean column at `momentum_lookback` when needed. Acceptance: `sloped_global_pit` no longer fires (+0.158 → +0.053), high-vol-regime CRPS improved 5.2%, mean CRPS essentially flat. v2.1 promoted to canonical default — `default.yaml` now sets `drift_mode: trailing_momentum`.

- **v2.2** — Implemented conditional block sampling (`generate_paths_conditional`, batched τ solver, per-path z-score buffer) and the test-only contingency (`conditional_block_sampling_in_search`). Acceptance gate **failed**: the target rule `acf_seam_degradation` did not improve (metric −1.056 → −1.121 on the fast preset). Audit traced the cause to a structural ceiling: any sampler that draws 10-day analog blocks intact inherits the within-window squared-return ACF (real-data within-10-day ACF = −0.125), not the unconditional ACF (+0.27). v2.2 ships as an opt-in mode (`conditional_block_sampling: true` in config). Initially stayed off-by-default. Fixing the ACF rule needs σ-scaling work — v3 scope. See RESULTS.md for the full audit.

- **v2.3** *(2026-05-19, promoted)* — **Default flipped to Cell D from the 2×2 ablation: trailing-momentum drift + conditional block sampling.** The 2×2 `(drift, conditional)` decomposition in ABLATION_STUDIES_REPORT.md reframed the v2.2 deferral: conditional sampling delivers a CRPS gain (~4% canonical) as an independent mechanism, separate from the ACF rule it was originally designed to fix. Cell D canonical confirmation (`runs/analog_mc/20260518T155800Z/`) landed at mean CRPS 0.05056 (−4.0% vs v2.1), high-vol CRPS 0.0831 (−3.9%), with PIT calibration preserved (sloped_global_pit +0.055, well below ±0.10). Promotion gates per V3_PLAN: **E3 seed-noise floor (0.08% << 4% gain → robust)** and **E10 Cell D × bl=20 (does not stack → vanilla Cell D is the right target)**. `acf_seam_degradation` still fires (structural; v3b/E9 GARCH-conditional resampling is the next planned fix). `default_v22.yaml` is kept as archived acceptance config; `default.yaml` now mirrors it.

## Purpose of this document

This is an implementation specification for a probabilistic forecasting pipeline that uses historical analogs (k-NN in multi-horizon z-score space) to generate Monte Carlo simulations of forward price paths. It is intended to be executed by an LLM coding agent in a Claude Code-style harness.

The plan is the output of a design conversation. Every decision documented here was made for a specific reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

The pipeline is **asset-agnostic**. It must work on any time series of price data with sufficient history — broad indices (S&P 500, Nifty 50), single-name equities across markets, sector ETFs, FX, commodities, or other tradable instruments. The forecast horizon and z-score horizons are **specified in the config**, not hardcoded. Different assets and use cases will warrant different horizons (e.g., 20-day for short-term equity forecasts, 60-day for quarterly views, 252-day for annual). All horizon-dependent logic must read from the config and adapt accordingly.

Some design decisions in this plan were originally motivated by single-name Indian equity behaviour (e.g., the σ-independent drift assumption, the asymmetric vol clip bounds). Those decisions are still defensible defaults for broad-market index forecasting, but they are surfaced as **config-driven knobs** so they can be revisited per asset class without code changes.

---

## High-level architecture

```
WALK-FORWARD FOLD LOOP
├── 0. Split data: expanding Train → Val (tuning) → Test (held-out)
├── HYPERPARAMETER SEARCH LOOP (grid + Nelder-Mead, NOT BayesOpt)
│   ├── 1. Compute causal rolling features (returns, EWMA vol, z-scores)
│   ├── 2. For each Val target date:
│   │   ├── 2a. Composite distance to Train dates (weighted multi-horizon z-scores)
│   │   ├── 2b. Restrict candidates to dates strictly before target
│   │   ├── 2c. Distances → probabilities via n_eff parameterization
│   │   ├── 2d. Sample 60-day paths as 6×10-day blocks from analog dates
│   │   └── 2e. Per-analog vol scaling: subtract μ_origin → rescale → re-inject drift
│   └── 3. Evaluate Val CRPS
├── 4. Lock best (w20, w50, w200, n_eff); run on Test; append to global OOS record
├── 4.5. Log per-fold diagnostics
WALK-FORWARD LOOP ENDS
└── 5. Final diagnostics + aggregate metrics (with explicit decision rules for v2)
```

---

## Module structure

Build under a clean package layout. Suggested:

```
repo_root/
├── data/                # Raw OHLC CSV files (gitignored, with .gitkeep)
│   └── .gitkeep
├── dashboards/          # Single Streamlit app with multiple views
│   ├── app.py                  # Entry point: `streamlit run dashboards/app.py`
│   └── views/
│       ├── config_editor.py    # Config editing view
│       ├── run_experiment.py   # Experiment launcher and progress monitor
│       └── diagnostics.py      # Run artifact loader and report renderer
├── src/analog_mc/
│   ├── data.py              # Loading, splitting, walk-forward fold generation
│   ├── features.py          # Causal rolling z-scores, EWMA vol
│   ├── distances.py         # Composite distance, n_eff probability conversion
│   ├── sampling.py          # Block sampling, per-analog vol scaling
│   ├── simulate.py          # Full Monte Carlo path generation
│   ├── scoring.py           # CRPS, PIT, reliability diagrams, ACF diagnostics
│   ├── search.py            # Grid + Nelder-Mead hyperparameter search
│   ├── walk_forward.py      # Orchestrates the fold loop
│   ├── diagnostics.py       # All step-5 reports and plots
│   └── config.py            # Centralized parameters (see Configuration section)
├── tests/                   # Unit tests, especially for causality
├── notebooks/               # Exploration, not production
├── configs/                 # YAML config files per experiment
├── runs/                    # Output artifacts per run (gitignored)
└── .gitignore               # Ignores data/*.csv, runs/*, etc.
```

**Folder details:**

- **`data/`**: Holds raw OHLC CSV files. The directory itself is committed via `.gitkeep`, but contents are gitignored. The data-loading module reads from here by default (overridable via config). Expected CSV schema: at minimum a date column and a close/adjusted-close column; OHLC if available. Document the exact schema in `data/README.md`.

- **`dashboards/`**: Streamlit application providing a UI layer over the pipeline. A single app with multiple views (tabs or a sidebar nav):
  - **Config editor view**: load any YAML config from `configs/`, edit parameters via form widgets (sliders for n_eff candidates, number inputs for horizons, dropdowns for `drift_mode`, etc.), validate against config invariants, and save back to YAML.
  - **Run experiment view**: select a config + (optionally) override ticker, kick off a walk-forward run as a subprocess, stream progress (fold completion, per-fold CRPS) to the UI. Persists results to `runs/<timestamp>/`.
  - **Diagnostics view**: select a completed run from `runs/`, render all step-5 diagnostics (weight trajectory, PIT histograms, CRPS surface, ACF comparison, clip-hit summary, reliability diagram, fixed-weight baseline comparison) with explanatory text and the v2-trigger decision rules highlighted.

  Structure as `dashboards/app.py` (entry point) with view modules under `dashboards/views/` (`config_editor.py`, `run_experiment.py`, `diagnostics.py`) imported and dispatched by the main app. The dashboard imports from `src/analog_mc/` — it is a presentation layer, not a place for new pipeline logic. Any computation lives in the package and is called from the dashboard.

Use type hints. Use `numpy` and `pandas` for core ops, `scipy` for Nelder-Mead, `matplotlib`/`seaborn` for plots, `streamlit` for dashboards. Avoid heavy frameworks. **Do not** use scikit-learn's standardization — implement causal z-scoring directly because sklearn batch-fits across the whole array.

---

## Configuration

Centralize all parameters in `config.py` (or a YAML loaded into a dataclass). All horizon-dependent parameters must be configurable — different assets and forecast use cases will require different settings. Default values shown below are reasonable for a quarterly forecast on a broad equity index; document them as defaults, not hardcoded assumptions.

```python
@dataclass
class Config:
    # Asset
    ticker: str = "^GSPC"              # any ticker the data loader supports
    data_path: str = "data/"           # directory containing OHLC CSV files

    # Horizons (FULLY CONFIGURABLE — different assets/use cases warrant different choices)
    forecast_horizon: int = 60         # total trading days to forecast forward
    block_length: int = 10             # days per sampled block
    n_blocks: int = 6                  # MUST satisfy: forecast_horizon == n_blocks * block_length
    n_paths: int = 1000                # Monte Carlo paths per forecast origin

    # Z-score horizons — choose to span short/medium/long context relevant to the asset
    # Common alternatives: (10, 30, 90) for short-term, (20, 50, 200) for quarterly,
    # (50, 200, 500) for annual macro views. The number of horizons also affects the
    # dimensionality of the weight simplex — keep it at 3 unless you want to rewrite the grid.
    zscore_horizons: tuple = (20, 50, 200)

    # EWMA vol
    ewma_halflife: int = 20            # for trailing causal vol; scale to typical vol-cluster decay

    # Walk-forward
    train_initial_size: int = 1000     # trading days (~4 years) for the first fold
    val_size: int = 60                 # days per Val block
    test_size: int = 60                # days per Test block

    # Hyperparameter search grid
    weight_grid_resolution: float = 0.1     # 10% spacing on simplex → ~66 points
    n_eff_values: tuple = (15, 30, 50, 80, 150)
    local_refine_top_k: int = 5             # take top-k grid points for Nelder-Mead
    nelder_mead_xatol: float = 0.01
    nelder_mead_maxiter: int = 50

    # Volatility scaling (asymmetric clips — defaults reflect equity leverage effect)
    # For symmetric-vol assets like FX, consider clip_lower=0.4, clip_upper=2.5
    vol_clip_lower: float = 0.5
    vol_clip_upper: float = 3.0
    drift_mode: str = "zero"           # "zero" for v1; "trailing_momentum" reserved for v2
    momentum_lookback: int = 20        # only used if drift_mode != "zero"
    momentum_shrinkage: float = 0.5    # only used if drift_mode != "zero"

    # Diagnostics
    pit_n_bins: int = 20
    acf_lags: tuple = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    vol_regime_quantiles: tuple = (0.33, 0.67)   # for conditional PIT terciles

    # Reproducibility
    random_seed: int = 42
```

**Important config invariants** (validate at startup):
- `forecast_horizon == n_blocks * block_length`
- `len(zscore_horizons) == 3` (the grid is built for 3 weights; relaxing requires rewriting `search.py`)
- `max(zscore_horizons) < train_initial_size` (otherwise early Train dates have undefined long-horizon z-scores)
- `vol_clip_lower < 1.0 < vol_clip_upper`

Pass a `Config` object through the pipeline. Do not read from globals.

---

## Critical correctness constraints

These are non-negotiable. Many of them were the subject of long design discussion and silent violation is the most likely way this pipeline can quietly produce nonsense.

### C1. Causal rolling features (zero look-ahead)

For every date *t* in every split (Train, Val, Test), any rolling statistic used at *t* must be computed over data with index strictly < *t* (or ≤ *t*−1 in trading-day terms). The same function must be applied to all three splits — the only difference is the *target* (Val/Test dates have forecasts and are scored; Train dates serve as analog candidates).

The rolling window is allowed to reach **backward** across split boundaries. E.g., the 200-day z-score for the 50th day of Val uses the 150 Train days immediately preceding Val plus the 49 prior Val days. This is **not** leakage; it is the correct causal computation.

Use `pandas.ewm(halflife=..., adjust=False).std()` for EWMA vol — this is causal by default. **Verify with a unit test** that the value at index *t* equals the value computed from `series.iloc[:t+1]` alone.

### C2. n_eff parameterization for distance → probability

Do not use a fixed softmax temperature τ. Instead, for each forecast origin date, solve for τ such that the resulting probability vector has effective sample size n_eff = exp(H(p)), where H is the entropy in nats.

Pseudocode:

```python
def distances_to_probs(distances: np.ndarray, target_n_eff: float) -> np.ndarray:
    """
    Given distances d_i, find tau such that p_i = softmax(-d_i / tau) has
    exp(entropy(p)) ≈ target_n_eff.
    """
    def n_eff_of_tau(tau):
        log_w = -distances / tau
        log_w -= log_w.max()      # numerical stability
        w = np.exp(log_w)
        p = w / w.sum()
        # entropy in nats; guard zeros
        entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
        return np.exp(entropy)

    # Solve via bisection: tau ∈ [tau_low, tau_high]
    # tau_low gives n_eff → 1 (sharp), tau_high gives n_eff → N (uniform)
    # Use scipy.optimize.brentq
```

n_eff is *the* parameter that controls forecast dispersion. It is tuned in the grid search.

### C3. Per-analog volatility scaling with shared-baseline demean

For each sampled analog block, the operation is:

```python
def scale_block(
    raw_block,
    sigma_current,
    sigma_historical_analog,
    mu_origin,            # constant per forecast — see definition below
    drift_target,         # zero in v1
    config,
):
    # raw_block: (block_length,) array of log returns from the historical analog
    # sigma_current: trailing causal EWMA σ at the forecast origin (or running σ for blocks 2+)
    # sigma_historical_analog: trailing causal EWMA σ at the analog's origin date
    # mu_origin: trailing causal mean of returns over the LONGEST z-score horizon,
    #            computed at the forecast origin date. Constant across all analog
    #            blocks and all paths within a single forecast. Same value used to
    #            demean every sampled block.
    # drift_target: per-day expected log return to inject (zero in v1)

    demeaned = raw_block - mu_origin
    ratio = sigma_current / sigma_historical_analog
    ratio = np.clip(ratio, config.vol_clip_lower, config.vol_clip_upper)
    scaled = demeaned * ratio
    return scaled + drift_target
```

`mu_origin` is the trailing causal mean of returns over the longest z-score horizon (`max(config.zscore_horizons)`; default 200 trading days), computed at the forecast origin date. It represents the *current regime's drift baseline*. The same `mu_origin` is used to demean every analog block in the forecast — it is **not** recomputed per-block or per-analog, and it is **not** updated across blocks within a path.

The demean step removes the current regime's baseline drift before the σ-ratio multiplier is applied. The multiplier therefore amplifies only the *deviation from current regime drift*, not the drift itself. For most equity-like instruments — where high-vol regimes typically coincide with drawdowns rather than proportionally higher expected returns — scaling a calm-regime positive drift up by 2.5× would be fundamentally wrong. Subtracting a shared constant (rather than per-block means) preserves the analog's mean *relative* to the current regime: analogs that genuinely outperformed `mu_origin` contribute positively to the path, those that underperformed contribute negatively, and the forecast distribution at horizon end has dispersion driven by analog-selection variation.

**Why a shared constant, not per-block means** (v1.0 → v1.1 fix). An earlier version of this plan used per-block demean (`raw_block.mean()`). That choice makes every analog block sum to zero after demeaning, so every Monte Carlo path collapses to *exactly* zero cumulative log return at h = block_length × k for k = 1..n_blocks. The forecast distribution at block-boundary horizons becomes a point mass at zero, which (a) destroys cumulative-return calibration, (b) makes PIT ranks degenerate (only 0 or 1) at those horizons, and (c) discards the regime-conditioning signal that analog matching is supposed to provide. The shared-constant demean fixes all three.

**Why `max(zscore_horizons)` for the baseline window.** The longest z-score horizon is the longest context window the matcher uses to define "regime." Using the same window for the demean baseline keeps the matching and the drift-baseline semantics aligned: both treat the longest-horizon trailing mean as the canonical current-regime descriptor. If a config uses different `zscore_horizons`, `mu_origin` adapts automatically. The window can be decoupled with a new config knob later if needed; for v1 the coupling keeps the design surface small.

**v2 reservations.** For assets with genuinely σ-proportional drift (rare in practice, more defensible for some commodities or vol-targeted strategies), the demean step can be skipped by setting `drift_mode = "scale_with_vol"` (reserved; not implemented in v1). The default `drift_mode = "zero"` is the most conservative choice and the right starting point regardless of asset class.

In v1, `drift_target = 0.0`. In v2 (if PIT shows directional slope), `drift_target` is a shrunk trailing-momentum estimate added back **on top of** the `mu_origin` removal (so the v2 forecast = `mu_origin`-demeaned + scaled + injected drift).

### C4. Vol scaling timing for block 2+

For blocks 2 through 6 of a path, σ_current is **not** the σ at the forecast origin. It is the trailing σ computed from the *simulated returns up to that point in the path*. Maintain a running EWMA σ inside the path-generation function.

This is what makes the 6×10 block design plausible despite the seam effects. The seam effect from issue 6 is about analog selection conditioning, which is *not* implemented in v1; v1 simply samples each block's analog independently using the same forecast-origin features. The running σ keeps the vol scaling internally consistent.

**Decision flag for v2:** vol-aware conditional block sampling (sampling each block's analog conditioned on the running state at the end of the prior block) is the v2 fix triggered by the squared-return ACF diagnostic.

### C5. Strictly forward sampling

When sampling forward blocks from an analog match at historical date *d*, the sampled block is the actual realized returns on dates *d+1, d+2, ..., d+block_length*. **Not** *d-9, ..., d* (backward), **not** *d-5, ..., d+5* (centered). This is obvious but easy to off-by-one.

### C6. Walk-forward boundary discipline

- Expanding Train: Train always starts from the earliest available data and grows.
- Val and Test are fixed-size windows that march forward.
- Each fold's Test block is appended to the global OOS record (concatenated forecasts + realizations).
- No date appears in more than one Test block across folds.

---

## Step-by-step implementation order

Implement and unit-test in this order. **Do not skip ahead** — the diagnostic infrastructure is what makes this pipeline trustworthy, not the optimizer.

### Stage 1: Data loading and causal features

1. `data.py`: Load OHLC data, compute log returns. Function takes a ticker and a date range.
2. `features.py`:
   - `causal_ewma_vol(returns, halflife)`: returns a Series where index *t* is the EWMA vol using only data ≤ *t*−1.
   - `causal_zscore(returns, horizon)`: returns a Series where index *t* is `(returns[t-horizon:t].mean()) / returns[t-horizon:t].std()` (or similar; choose whether to z-score the return *at t* or the *mean over the past horizon* — pick one and document).
   - `causal_trailing_mean(returns, horizon)`: returns a Series where index *t* is the mean of returns over the same window used by `causal_zscore` (whatever convention was picked there). Required as the source of `mu_origin` in C3.
   - **Unit test**: for a known synthetic series, verify the value at index 100 matches a hand computation using only indices 0..99 (or 0..100 inclusive, per the documented convention). Apply the same test to all three functions.

### Stage 2: Walk-forward fold generation

3. `data.py`: `generate_folds(returns, config) -> list[Fold]` where `Fold` has `train_idx`, `val_idx`, `test_idx`.
4. Verify no overlap between Test blocks across folds.

### Stage 3: Distance and probability

5. `distances.py`:
   - `composite_distance(z_target, z_candidates, weights)`: returns an array of distances (e.g., weighted Euclidean across horizons).
   - `distances_to_probs(distances, n_eff)`: implements C2.
   - **Unit test**: `distances_to_probs` should produce a probability vector that sums to 1 and whose `exp(entropy)` is within 5% of the target n_eff.

### Stage 4: Block sampling

6. `sampling.py`:
   - `sample_analog_blocks(probs, analog_dates, returns, block_length, n_paths, rng)`: for each of n_paths paths, sample with replacement an analog date by `probs`, then return the block of length `block_length` *forward* from that date.
   - `scale_block(...)`: implements C3.
   - `generate_path(...)`: implements C4 — concatenates 6 scaled blocks with a running EWMA σ updated after each block.

### Stage 5: Simulation orchestration

7. `simulate.py`:
   - `forecast(origin_date, train_data, weights, n_eff, config) -> np.ndarray`: returns an (n_paths × horizon) array of simulated log return paths.

### Stage 6: Scoring

8. `scoring.py`:
   - `crps_sample(forecast_paths, realized_path)`: CRPS for a multi-step forecast vs. a single realized path. Use the standard ensemble CRPS estimator. For multi-step, compute per-step CRPS and aggregate (sum or mean — document the choice; mean is more interpretable).
   - **Unit test**: known closed-form case (Gaussian forecast vs. point realization) should match the analytic CRPS.

### Stage 7: Hyperparameter search

9. `search.py`:
   - `generate_weight_grid(resolution)`: returns the list of (w20, w50, w200) points on the simplex.
   - `grid_search(fold, config)`: evaluates Val CRPS at every (grid point, n_eff) combination. Returns a DataFrame with columns `[w20, w50, w200, n_eff, val_crps]`.
   - `local_refine(top_k_points, fold, config)`: runs Nelder-Mead from each of the top k grid points. Returns best (weights, n_eff, val_crps).
   - **Important**: n_eff is discrete in the grid but Nelder-Mead operates on weights only with n_eff fixed at the best grid value. Do not let Nelder-Mead optimize over n_eff (it's discrete and the surface is non-smooth in n_eff).

### Stage 8: Walk-forward orchestration

10. `walk_forward.py`:
    - For each fold: search → lock weights → evaluate on Test → log everything → append to global record.
    - Persist intermediate results to disk after each fold (in case of crash). Use a simple format: parquet or JSON per fold.

### Stage 9: Diagnostics (this is where most of the value is)

11. `diagnostics.py` — implement each in its own function with a clear matplotlib output:

    a. **Weight trajectory plot**: fold index × weight value, three lines.
    b. **CRPS surface contour (fold 1 only)**: ternary plot or 2D projection of the weight simplex with CRPS as color.
    c. **Fixed-weight baseline**: re-run the walk-forward with fixed (1/3, 1/3, 1/3) and a fixed n_eff = 30, compute aggregate OOS CRPS, compare to the tuned version. Report the difference.
    d. **Global PIT histogram**: on concatenated OOS record. For each (forecast origin, horizon step) pair, compute the rank of the realization in the empirical forecast distribution → histogram should be flat.
    e. **Conditional PIT by vol regime**: bucket forecast origins by σ_current terciles, produce 3 PIT histograms.
    f. **Reliability diagram**: for predicted quantiles {0.1, 0.25, 0.5, 0.75, 0.9}, compute the empirical fraction of realizations that fell below each → plot against the nominal quantile.
    g. **ACF comparison**: compute autocorrelation of *squared* returns at lags from config, for simulated paths (averaged over all paths and origins) vs. realized paths. Plot both on the same axes. **Flag if any seam-lag (10, 20, 30, 40, 50) shows simulated ACF >30% below realized.**
    h. **Clip-hit summary**: histogram of σ_current/σ_historical ratios across all analog draws, with vertical lines at the clip bounds. Report the fraction hitting each bound, per fold.

12. **Diagnostic decision rules** (print these as a summary at the end of the walk-forward run):
    - Sloped global PIT → recommend enabling `drift_mode = "trailing_momentum"` in v2.
    - U-shaped high-vol-bucket PIT → recommend raising `vol_clip_upper` and/or increasing `n_eff` candidates in v2.
    - Squared-return ACF degradation >30% at seam lags → recommend implementing vol-aware conditional block sampling in v2.
    - Fixed-weight baseline within 1% of tuned CRPS → recommend dropping the per-fold search and shipping fixed weights.
    - Clip-hit fraction >15% on either bound → recommend revisiting the distance metric (the analog matcher is failing to match the vol regime).

### Stage 10: Final aggregate report

13. After all diagnostics: aggregate OOS CRPS, broken down by:
    - Overall
    - Per-fold
    - Per-horizon-step (1-day-ahead through forecast_horizon-day-ahead)
    - Per-vol regime (using the conditional buckets)

    Only report aggregate CRPS *after* the diagnostics. The aggregate number is meaningful only if PIT is roughly flat and weight trajectories are sane.

### Stage 11: Streamlit dashboard (build LAST, after the pipeline is verified)

14. Build the **diagnostics view first** (`dashboards/views/diagnostics.py`) — it has the least coupling and lets you visually verify completed runs. It should:
    - List runs from `runs/` with metadata (ticker, date, config summary, aggregate CRPS).
    - On selection, render every diagnostic from Stage 9 with matplotlib figures embedded.
    - Surface the v2-trigger decision rules prominently (which ones fired, which didn't).

15. Then the **config editor view** (`dashboards/views/config_editor.py`):
    - Form-based editor for every field in `Config`.
    - Validates invariants (forecast_horizon == n_blocks × block_length, etc.) before allowing save.
    - Save to `configs/<name>.yaml`.

16. Then the **run experiment view** (`dashboards/views/run_experiment.py`):
    - Select a config + (optionally) override ticker.
    - Launches `walk_forward.py` as a subprocess. Captures stdout/stderr to a log file in the run directory.
    - Polls the run directory for per-fold parquet files and displays a progress bar + running CRPS.
    - On completion, offers a button to jump to the diagnostics view for the new run.

    **Do not** run the walk-forward inside the Streamlit process directly — it blocks the event loop and Streamlit's rerun model breaks long-running tasks. Use `subprocess.Popen` and poll filesystem artifacts.

17. Finally, **`dashboards/app.py`** as the entry point: sidebar nav (or tabs) to switch between the three views. Run with `streamlit run dashboards/app.py`.

The dashboard is a thin presentation layer. **No pipeline logic lives in `dashboards/`.** If you find yourself writing computation there, move it to the appropriate module in `src/analog_mc/` and import it.

---

## Reproducibility requirements

- Single config file (YAML or dataclass) defines the entire run.
- All random number generation goes through a single seeded `np.random.Generator` instance, passed explicitly. **No** `np.random.seed()` calls.
- Persist intermediate per-fold results to disk so re-running diagnostics doesn't require re-running the search.
- Log the git commit hash and config hash at the start of each run.

---

## What not to do

- **Do not** swap grid search for BayesOpt without explicit discussion. The grid was chosen for diagnostic interpretability on a low-dim, likely-flat loss surface.
- **Do not** use scikit-learn's `StandardScaler` — it batch-fits.
- **Do not** vectorize feature computation in a way that uses future data. If in doubt, write the slow loop version and a unit test, then optimize.
- **Do not** implement v2 features (trailing-momentum drift, conditional block sampling, tail inflation) in v1. They are gated on specific diagnostic findings. Premature implementation contaminates the diagnostics that decide whether v2 is needed.
- **Do not** report aggregate CRPS as the headline result without the PIT and weight-trajectory diagnostics. CRPS alone cannot distinguish a well-calibrated, sharp forecaster from a miscalibrated one with the same aggregate score.
- **Do not** add transaction costs, position sizing, or PnL calculations to this pipeline. Those belong downstream. This pipeline produces probabilistic *forecasts*, not strategies.

---

## Open questions to raise with the user before starting implementation

1. **Data source(s)**: yfinance, NSE bhav files, AlphaVantage, paid vendor, or local CSVs only? Different sources warrant different loader implementations. The default assumption is "user drops CSVs in `data/`" — confirm whether automated download is also wanted.
2. **Multi-ticker support in v1?** The architecture is single-ticker per run. Multi-ticker is a wrapping concern handled by running the pipeline multiple times. Confirm this is fine.
3. **CSV schema**: what columns are expected in `data/*.csv` files? Minimum needed is date + close. Specify exact column names and date format to write into the loader.
4. **Initial training window**: `train_initial_size = 1000` (~4 years) is reasonable for daily equity data going back a decade or more. For shorter-history assets (recent IPOs, new ETFs), this may need shrinking. Confirm typical data availability per asset class.
5. **Output directory**: default to `./runs/<timestamp>/`. Confirm or override.
6. **Streamlit deployment**: local-only (`streamlit run`) or hosted? This affects how `run_experiment.py` handles concurrent runs and file locking.

---

## Final note for the implementing agent

The temptation will be to start with the optimizer and the analog matching because they're the "interesting" parts. **Resist this.** Build the causal-feature unit tests first, then the walk-forward fold generator, then a single-fold forward pass with hardcoded weights. Verify the diagnostics infrastructure works on that single fold before adding the search loop. The whole pipeline is engineered to make silent failures visible — building it in the wrong order defeats that engineering.

The Streamlit dashboard is explicitly the **last** stage. It is easier to build correctly once the underlying pipeline is solid and the diagnostic functions return predictable artifacts. Do not interleave dashboard work with pipeline work — it tends to bake presentation assumptions into the computation layer.
