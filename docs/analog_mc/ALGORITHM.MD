# Analog Monte Carlo Forecasting Algorithm — Step-by-Step

## Inputs

- Price series for an asset, converted to log returns $r_t$ for $t = 1, \ldots, T$
- Forecast horizon $H$ (trading days)
- Block length $L$ and number of blocks $n_B$ with $H = n_B \cdot L$
- Number of Monte Carlo paths $N_P$
- Z-score horizons $(h_1, h_2, h_3)$, e.g., $(20, 50, 200)$
- EWMA volatility half-life $\lambda$
- Weights $\mathbf{w} = (w_1, w_2, w_3)$ on the 2-simplex
- Target effective sample size $n_\text{eff}^\ast$
- Asymmetric vol clip bounds $(\rho_\text{lo}, \rho_\text{hi})$
- Drift mode (`zero` in v1)
- Shared demean baseline $\mu_\text{origin}$ — trailing causal mean over the longest z-score horizon, computed at the forecast origin (v1.1)

---

## Step 1 — Causal feature construction

For every date $t$, compute strictly using data with index $< t$:

1. Trailing causal EWMA volatility $\sigma_\text{EWMA}(t)$ with half-life $\lambda$.
2. For each horizon $h \in \{h_1, h_2, h_3\}$, the rolling mean $\mu_h(t)$ over the trailing window of length $h$ ending at $t-1$.
3. The z-score $z_h(t) = (\mu_h(t) - \bar{\mu}_h(t)) / \sigma_h(t)$ where $\bar{\mu}_h$ and $\sigma_h$ are causal rolling estimators of $\mu_h$'s mean and standard deviation.

The same function applies to every date in every split. Rolling windows may reach backward across split boundaries — this is causal, not leakage.

---

## Step 2 — Walk-forward fold construction

Partition the time index into a sequence of folds. Each fold $f$ has:

- Expanding training set $\mathcal{T}_f$ (all dates from start to a cutoff)
- Validation block $\mathcal{V}_f$ (next $V$ dates)
- Test block $\mathcal{S}_f$ (next $S$ dates)

with $\max(\mathcal{T}_f) < \min(\mathcal{V}_f) < \max(\mathcal{V}_f) < \min(\mathcal{S}_f)$.

No date appears in more than one $\mathcal{S}_f$ across folds.

---

## Step 3 — Hyperparameter search (per fold)

Within fold $f$, search over weights $\mathbf{w}$ on the simplex and discrete $n_\text{eff}^\ast \in \{15, 30, 50, 80, 150\}$:

1. **Coarse grid**: enumerate $\mathbf{w}$ on the 2-simplex at resolution 0.1 (~66 points). For each $(\mathbf{w}, n_\text{eff}^\ast)$ combination, evaluate validation CRPS via Step 4.
2. **Local refinement**: take the top-$K$ grid points (e.g., $K=5$) by validation CRPS. From each, run Nelder–Mead on $\mathbf{w}$ with $n_\text{eff}^\ast$ fixed, projected onto the simplex.
3. **Lock**: select the global minimum across all refinements as $(\mathbf{w}^\ast, n_\text{eff}^{\ast\ast})$ for this fold.

---

## Step 4 — Forecast generation (for a given $\mathbf{w}, n_\text{eff}^\ast$, origin $t^\ast$)

For each forecast origin $t^\ast$ (in $\mathcal{V}_f$ during tuning, in $\mathcal{S}_f$ at test time):

### 4a. Compute distances

For each candidate analog date $s \in \mathcal{T}_f$ with $s < t^\ast$:

$$d(t^\ast, s) = \sqrt{\sum_{i=1}^{3} w_i (z_{h_i}(t^\ast) - z_{h_i}(s))^2}$$

### 4b. Convert distances to probabilities

Solve for temperature $\tau$ via bisection such that:

$$p_i = \frac{\exp(-d_i / \tau)}{\sum_j \exp(-d_j / \tau)}, \qquad \exp(H(p)) = n_\text{eff}^\ast$$

where $H(p) = -\sum_i p_i \log p_i$.

### 4c. Sample $N_P$ paths

Before path sampling, compute the shared demean baseline once:

$$\mu_\text{origin} = \frac{1}{h_3} \sum_{k=0}^{h_3 - 1} r_{t^\ast - k}$$

(the trailing causal mean over the longest z-score horizon, ending at $t^\ast$ inclusive).

For each path $j = 1, \ldots, N_P$:

1. Initialize $\sigma_\text{cur}^{(1)} \leftarrow \sigma_\text{EWMA}(t^\ast)$.
2. For each block $k = 1, \ldots, n_B$:
   - Sample analog date $s_k \sim p$.
   - Extract forward block $\mathbf{r} = (r_{s_k + 1}, \ldots, r_{s_k + L})$.
   - Form $\tilde{r}_l = r_l - \mu_\text{origin}$ — **v1.1: subtract the shared baseline, not the per-block mean**. The per-block-mean variant (v1.0) caused every path to collapse to zero cumulative return at every block boundary $h = kL$. See `IMPLEMENTATION_PLAN.md` C3 for the rationale.
   - Compute $\rho = \sigma_\text{cur}^{(k)} / \sigma_\text{EWMA}(s_k)$.
   - Clip: $\rho \leftarrow \text{clip}(\rho, \rho_\text{lo}, \rho_\text{hi})$.
   - Emit scaled returns $\hat{r}_l = \rho \tilde{r}_l + \delta$, where $\delta = 0$ in v1.
   - Append $\hat{r}_l$ to path $j$.
   - Update running vol: $\sigma_\text{cur}^{(k+1)} \leftarrow$ trailing EWMA of simulated returns generated so far in path $j$.
3. Path $j$ is the concatenation of all $n_B$ scaled blocks: $H$ simulated log returns.

The result is an $N_P \times H$ matrix of simulated log returns, defining the forecast distribution $\hat{F}_{t^\ast, h}$ at each horizon step $h$.

---

## Step 5 — CRPS evaluation

For each forecast origin $t^\ast$ and each horizon step $h \in \{1, \ldots, H\}$, compute the ensemble CRPS between $\hat{F}_{t^\ast, h}$ (empirical distribution of simulated cumulative returns at step $h$) and the realised cumulative return $r_{t^\ast \to t^\ast + h}$.

Aggregate:

$$\text{CRPS}_\text{val} = \frac{1}{|\mathcal{V}_f|} \sum_{t^\ast \in \mathcal{V}_f} \frac{1}{H} \sum_{h=1}^{H} \text{CRPS}_\text{sample}(\hat{F}_{t^\ast, h}, r_{t^\ast \to t^\ast + h})$$

This is the objective minimised by the Step 3 search.

---

## Step 6 — Out-of-sample test

Using the locked $(\mathbf{w}^\ast, n_\text{eff}^{\ast\ast})$ from Step 3, run Step 4 on each origin $t^\ast \in \mathcal{S}_f$. Append the simulated distributions and realisations to a global out-of-sample record.

---

## Step 7 — Roll the fold

Advance $\mathcal{T}_f$, $\mathcal{V}_f$, $\mathcal{S}_f$ forward by one $\mathcal{S}_f$ width. Repeat Steps 3–6 until the data is exhausted.

Log per-fold: $(\mathbf{w}^\ast, n_\text{eff}^{\ast\ast}, \text{CRPS}_\text{val}, \text{CRPS}_\text{test}, \text{upper-clip-fraction}, \text{lower-clip-fraction})$.

---

## Step 8 — Global diagnostics (on the concatenated OOS record)

Before reporting aggregate CRPS, compute:

1. **Weight trajectory**: $\mathbf{w}^\ast$ and $n_\text{eff}^{\ast\ast}$ vs. fold index. Check for smooth drift vs. chaos.
2. **CRPS surface (fold 1)**: contour over the simplex from the grid output. Check flatness.
3. **Fixed-weight baseline**: re-run Steps 3–7 with $\mathbf{w} = (1/3, 1/3, 1/3)$, $n_\text{eff}^\ast = 30$. Compare aggregate OOS CRPS.
4. **Global PIT histogram**: rank of each realisation in its forecast distribution; expect uniform.
5. **Conditional PIT by vol regime**: bucket origins by $\sigma_\text{EWMA}(t^\ast)$ terciles; three histograms.
6. **Reliability diagram**: predicted vs. empirical quantile exceedance frequencies at $\{0.1, 0.25, 0.5, 0.75, 0.9\}$.
7. **Squared-return ACF comparison**: simulated vs. realised at lags $\{1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50\}$.
8. **Clip-hit fractions**: per fold and aggregate.

---

## Step 9 — Decision rules (v2 triggers)

Inspect the diagnostics and apply:

| Diagnostic finding | v2 action |
|---|---|
| Sloped global PIT | Enable trailing-momentum drift $\delta$ |
| U-shape in high-vol-bucket PIT | Raise $\rho_\text{hi}$, increase $n_\text{eff}^\ast$ candidates, or add tail inflator |
| Squared-return ACF degrades >30% at seam lags | Implement vol-aware conditional block sampling |
| Fixed-weight baseline within ~1% of tuned CRPS | Drop per-fold optimization; ship fixed weights |
| Clip-hit fraction >15% on either bound | Revise distance metric (add vol-regime feature); do not widen clips |

---

## Step 10 — Aggregate reporting

Only after Steps 8–9 are satisfactory, report aggregate OOS CRPS, broken down by:

- Overall
- Per-fold
- Per-horizon-step ($h = 1, \ldots, H$)
- Per-vol regime (low / mid / high terciles)

The aggregate CRPS is meaningful only when the diagnostic suite confirms calibration and weight-stability.