# analog_mc — Related Work

Catalogues the most directly comparable literature for analog_mc and identifies (a) which papers are head-to-head benchmark candidates, (b) which expose theoretical gaps in our framing, and (c) borrowable ideas. Cross-referenced from [`V3_PLAN.md`](V3_PLAN.md) where individual papers motivate specific experiments.

**Method.** Five sub-agents read papers in parallel (one per tier below) and reported back against a fixed schema. Each agent used arxiv abstract pages + HTML mirrors; where PDFs returned binary blobs, agents flagged that paper-level method/theory detail was drawn from abstracts plus the established literature. Treat tier-2 method descriptions in particular as best-effort summaries rather than verbatim re-reads.

---

## Tier 1 — direct comparators (analog/retrieval on financial series)

### History Rhymes: Macro-Contextual Retrieval for Robust Financial Forecasting
**arXiv 2511.09754** · Khanna, Berger et al. · IEEE BigData 2025
- **Problem.** Robust equity forecasting under macro regime shifts.
- **Method.** Joint embedding of macro indicators (CPI, unemployment, yield spread, GDP) + news sentiment as the retrieval key; similarity in learned embedding space; head outputs trading signals.
- **Eval.** S&P 500 train 2007–2023, OOD test on AAPL/XOM 2024; Profit Factor + Sharpe.
- **Headline.** AAPL PF=1.18, Sharpe=0.95 OOD — only method with positive OOD PnL.
- **Comparable?** Partial — same "retrieve analogs" primitive; conditioning signal (macro+news) and output (PnL, not distribution) differ.
- **H2H learning.** Whether enriching the analog key with macro state lowers regime-conditional CRPS in our pipeline.

### ContraSim: Contrastive Similarity Learning for Market Forecasting
**arXiv 2502.16023** · Vinden, Saqur et al. · 2025
- **Problem.** Daily market direction prediction from news headlines.
- **Method.** Weighted self-supervised contrastive learning on augmented WSJ headlines; distance = learned semantic similarity; head = binary classifier.
- **Headline.** +7% classification accuracy.
- **Comparable?** No — text-modality classification, no probabilistic path.
- **H2H learning.** Whether learned contrastive embeddings beat our hand-tuned z-score weights as the analog key (the v4 "learned similarity" question V3_PLAN explicitly defers).

### StockMem: Event-Reflection Memory Framework
**arXiv 2512.02720** · Wang, Xiao et al. · 2025
- **Problem.** Next-day up/down stock movement under event-driven volatility.
- **Method.** Dual-layer LLM memory (temporal event KB + reflective causal-experience KB); retrieves analogous historical scenarios via LLM reasoning.
- **Eval.** CSMAR China, 4 tech stocks, train Q1 2024 / test Q2 2024; ACC, MCC.
- **Headline.** iFlytek ACC=62.53%, MCC=0.169.
- **Comparable?** No — LLM-based event memory, single-day classification.

### OFTER: An Online Pipeline for Time Series Forecasting
**arXiv 2304.03877** · Michael, Cucuringu, Howison · 2023
- **Problem.** Online multivariate mid-size TS forecasting, applied to daily equities.
- **Method.** k-NN + Generalized Regression Neural Networks with a weighted norm based on a *modified maximal correlation coefficient*; non-parametric regression head with dimensionality reduction; online updates.
- **Comparable?** **Yes** — same k-NN-on-equities primitive, same interpretability ethos.
- **H2H learning.** Whether OFTER's maximal-correlation distance beats our composite-Euclidean z-score weighting on the NASDAQ100 universe — the cleanest direct ablation target for our matcher.

**Tier-1 synthesis.** analog_mc is the only paper here doing probabilistic 60-day *path* forecasting with full distributional diagnostics (CRPS, PIT, ACF of r²); the others produce point classifications, trading PnL, or 1-step regressions. The block-bootstrap path-stitching, n_eff-calibrated weighting, walk-forward discipline, and PIT/regime-stratified diagnostics are unique to us in this tier. **Strongest direct comparator: OFTER (2304.03877).**

---

## Tier 2 — analog forecasting theory (meteorology/dynamics roots)

### Analog Forecasting with Dynamics-Adapted Kernels
**arXiv 1412.3831** · Zhao, Giannakis · *Nonlinearity*, 2016
- Kernel-weighted ensembles of analogs over Takens delay-coordinate vectors; anisotropy from the dynamical vector field; Nyström / Laplacian-pyramid out-of-sample extension.
- **Theory.** Kernel regression on the data manifold with spectral convergence under ergodicity.
- **Implication.** Our composite-Euclidean distance over 3 z-scores is a special case of their isotropic kernel; our `n_eff` temperature is functionally equivalent to their kernel bandwidth, just parameterised via ESS rather than σ.
- **Borrowable.** Delay-coordinate (lag) features would make our state vector dynamically richer than 3 rolling-mean snapshots.

### Kernel Analog Forecasting: Multiscale Test Problems
**arXiv 2005.06623** · Burov, Giannakis, Manohar, Stuart · *SIAM Multiscale Model. Simul.*, 2020
- KAF on slow variables in fast-slow systems with/without scale separation.
- **Theory.** KAF recovers the optimal forecast (conditional expectation) *only* when the prediction variable is Markovian in the observed coordinates; without scale separation, the target is non-Markovian and KAF is biased unless memory (delays) is included.
- **Implication.** Daily NASDAQ returns are emphatically non-Markovian in a 3-z-score state — our matcher will be systematically biased exactly where their theory predicts. Bias should concentrate in regime-transition folds (consistent with our high-vol-regime CRPS being 3× low-vol).

### Using Local Dynamics to Explain Analog Forecasting
**arXiv 2007.14216** · Platzer, Yiou, Naveau, Tandeo · *J. Atmos. Sci.* 78(7), 2021
- Compares constant-weight, kernel-weight, and locally-linear (analog + local linear regression) analog variants on Lorenz-63/96.
- **Theory.** Derives forecast error as a function of the *local Jacobian*: error scales N^(−2/d), grows along large-Lyapunov directions, and *analog + local linear regression* explicitly estimates the Jacobian, eliminating the leading bias.
- **Implication.** Our k-NN matcher degrades worst at high-vol regime onsets (large local Jacobian) — exactly what our per-vol-regime CRPS breakdown is sensitive to.
- **Borrowable.** A local-linear-regression correction on top of the k-NN block draws would absorb the Jacobian bias the raw analog inherits. Direct v3/v4 lift.

### Hierarchical Spatio-Temporal Analog Forecasting for Count Data
**arXiv 1701.04485** · McDermott, Wikle, Millspaugh · *Environmetrics*, 2017
- Bayesian hierarchical model where analog weights are *latent random variables* with Dirichlet-style priors; Poisson data layer; analogs matched on SST EOFs.
- **Theory contribution.** Random-weight analogs dominate point-estimated weights in calibration.
- **Implication.** Our `n_eff` weights are point estimates per fold; treating the weight vector as a posterior would propagate analog-selection uncertainty into PIT/CRPS naturally — would inflate ensemble dispersion in undermatched folds without re-tuning `n_eff`.

**Tier-2 synthesis.**
- **Inheritance:** analog_mc sits squarely in the Lorenz → Zhao–Giannakis kernel-regression lineage; n_eff/temperature, composite-Euclidean distance and weighted aggregation are recognised special cases with manifold-learning convergence guarantees under ergodicity.
- **Gaps exposed:** (a) Burov et al. — our 3-z-score state is non-Markovian, so the optimal forecast lies outside our hypothesis class without delay embedding; (b) Platzer et al. — our raw-block aggregation inherits an O(N^{−2/d}) Jacobian bias that local-linear correction would remove; (c) McDermott–Wikle — our weights should be random, not point-estimated.
- **Most relevant for v3:** Platzer et al. — its local-Jacobian theory predicts where our matcher fails (high-Lyapunov / vol-onset folds) and prescribes a fix (analog + local linear regression on block returns) that is compatible with our existing k-NN block-bootstrap scaffolding.

---

## Tier 3 — block bootstrap on returns

### Parsimonious NN for Portfolio Optimization
**arXiv 2303.08968** · van Staden, Forsyth, Li · 2023
- **Bootstrap variant.** Stationary block bootstrap (Politis–Romano) of empirical returns; expected block length is a tuned hyperparameter.
- **Eval.** Synthetic + empirical equity/bond returns; multi-year horizons; metric = recovery of analytical optimal policies.
- **Comparable?** Partial — same bootstrap family, but goal is policy learning, not probabilistic path forecasting.
- **Useful to E1?** Weak — stationary (random-length) blocks, no fixed-length sensitivity reported.

### Galerkin-ARIMA / Projection-Based ARIMA
**arXiv 2507.07469** · Liu, Lin · 2025
- **Bootstrap variant.** Block bootstrap for prediction intervals with two-stage generated-regressor bias; variant/length not specified in abstract.
- **Eval.** Quarterly GDP + daily S&P 500 returns; rolling-window forecasts vs. ARIMA/SARIMA.
- **Comparable?** Partial — overlapping domain (S&P 500 daily) and intent (UQ), but parametric backbone.

### Block Bootstrap KS Goodness-of-Fit
**arXiv 2511.05733** · Chandy, Schifano, Yan, Zhang · 2025
- **Bootstrap variant.** Circular block bootstrap (overlapping, wrap-around); block length `l = ⌈n^{1/3}⌉`; Politis–White plug-in tested but rejected — frequently selects block sizes exceeding sample size under strong dependence.
- **Eval.** Simulation size/power + S&P 500 returns; recommends n ≥ 400.
- **Useful to E1?** **Yes.** Cube-root rule on ~6000 daily obs implies `l ≈ 18`, bracketing our 10/20 sweep points and warning against Politis-White auto-selection.
- **Borrowable.** Add a block-bootstrap KS/PIT GoF check as a v3 diagnostic atop existing PIT plots.

**Tier-3 synthesis.**
- **Block-length tradeoffs.** Literature converges on `n^{1/3}`-style rules (≈18 for our data) and warns Politis–White overshoots under strong dependence — supports our fixed 5/10/20 sweep over an auto-selector.
- **State-conditional block draws.** None of these three condition the block draw on the simulated path's current state; analog_mc v2.2's conditional re-matching appears genuinely novel relative to this sample.
- **Most-direct baseline.** Liu & Lin's Galerkin-ARIMA with block-bootstrap PIs on daily S&P 500 is the cleanest head-to-head for analog_mc's CRPS/PIT among block-bootstrap papers.

---

## Tier 4 — filtered historical simulation (v3b / E9 lineage)

### Managing Volatility Risk: KL Decomposition + FHS
**arXiv 1710.00859** · Yao, Laurent, Bénaben · 2017
- **Method.** Karhunen-Loève decomposes the swaption vol cube into orthogonal PCs; FHS bootstraps GARCH-standardised residuals per PC and rescales by current σ; no-arbitrage filter.
- **Eval.** Kupiec/Christoffersen LR + no-arbitrage at 1-day horizon.
- **Headline.** FHS-on-PCs passes both tests where Gaussian/historical fails.
- **Informs E9.** The PC-then-FHS sequencing is portable: residual rescaling is *exactly* what E9 does with simulated GARCH vol. KL/PCA is irrelevant for a single-asset path.
- **Risk.** GARCH-residual non-stationarity (rare-regime residuals dominate the bootstrap pool) — E9 inherits this trap.

### Forecasting VaR with Time-Varying Variance, Skewness, Kurtosis (EWMA)
**arXiv 1206.1380** · Gabrielsen, Zagaglia et al. · 2012
- **Method.** EWMA on first four moments via modified Gram-Charlier density; Cornish-Fisher expansion to quantiles. Parametric, not bootstrap.
- **Eval.** Benchmarked against HS, FHS, GARCH at 1-day and 10-day VaR; Kupiec/Christoffersen + Basel II traffic-light.
- **Headline.** EWMA-CF competitive with FHS at 1-day, weaker at 10-day.
- **Informs E9.** Validates that **EWMA σ alone is insufficient at our 10-day horizon** — supports the v3b move to GARCH. Cornish-Fisher higher-moment EWMA is a cheaper fallback if GARCH proves brittle.
- **Risk.** Cornish-Fisher breaks at deep tails.

### A New Approach for Scenario Generation in Risk Management
**arXiv 0904.0624** · Ortega, Pullirsch, Teichmann, Wergieluk · 2009
- **Method.** SDE (HJM / Black-Scholes) calibrated empirically; Euler increments replaced by bootstrapped *standardised* historical innovations — the continuous-time analogue of FHS, with explicit market-vs-risk-manager input separation.
- **Informs E9.** The "drift + vol from SDE, shape from historical innovation" decomposition is the **cleanest theoretical justification** for E9's sign/shape-from-analog, scale-from-GARCH split.
- **Risk.** Assumes i.i.d. standardised innovations; our analog blocks are 10-day *correlated* sequences — strict generalisation.

**Tier-4 synthesis.**
- **Canonical FHS.** Fit GARCH(1,1) → standardize residuals → bootstrap residuals → rescale by forecasted σ_{t+h} → integrate. Sign/shape from history, scale from conditional vol model.
- **analog_mc's twist.** Replaces single-asset residual bootstrap with **multivariate-distance analog-block selection** (preserves intra-block ACF and cross-feature joint structure that i.i.d. residuals destroy); E9 grafts FHS's σ-conditioning onto this block primitive.
- **Recommendation.** Run textbook FHS as a **baseline**, not just an E9 ingredient — it isolates whether analog-block selection adds value over plain GARCH+residual-bootstrap. Without it, E9 wins are unattributable.

---

## Tier 5 — modern probabilistic forecasting baselines

### Assessing Uncertainty in Stock Returns: GMM-Based Method
**arXiv 2503.06929** · Wang, Xu · 2025
- **Method.** Deep net with a Gaussian-mixture output head; bag-of-words "stock code" embedding for cross-sectional clustering.
- **Eval.** **Chinese equities**, CRPS + MSE + QLIKE. No PIT, no per-regime CRPS. Short horizons.
- **Comparable?** **Partial** — equities + CRPS, but different market, horizon, no path/PIT diagnostics.
- **Borrowable.** Fit a GMM to analog_mc's empirical samples per horizon for a smoothed parametric overlay.

### VAEneu: VAE for Probabilistic Forecasting
**arXiv 2405.04252** · Koochali, Tahaei · 2024
- **Method.** Conditional autoregressive VAE; predictive distribution from latent samples; **CRPS is the training loss**.
- **Eval.** 12 generic datasets vs. 12 baselines; CRPS primary. No financial dataset; no PIT; no regime breakdowns.
- **Borrowable.** CRPS-as-loss is the natural objective if we ever fit a parametric head over analog_mc samples.

### TimeGMM: Single-Pass Probabilistic Forecasting via GMM
**arXiv 2601.12288** · Liu, Liu · 2026
- **Method.** Temporal encoder + conditional temporal-probabilistic GMM decoder; single forward pass for GMM parameters; reversible instance norm (GRIN).
- **Eval.** Electricity/Traffic-class benchmarks; ~22.48% CRPS gain vs prior SOTA. No financial data.
- **Borrowable.** GRIN-style reversible normalisation could replace/augment our causal z-score scaling.

### Channel-Aware Contrastive Conditional Diffusion
**arXiv 2410.02168** · Li, Chen, Xiong · 2024
- **Method.** Conditional denoising-diffusion with channel-centric denoiser and contrastive past↔future MI objective.
- **Eval.** MSE + CRPS; wins on 83.3% of CRPS cases across standard multivariate benchmarks. No equities.
- **Borrowable.** Contrastive past↔future MI as a principled alternative to grid-searching `(w, n_eff)`.

**Tier-5 synthesis.**
- **Playbook.** Modern probabilistic TS forecasting splits into parametric heads (GMM, VAE-latent) trained on NLL/CRPS, and sample-based generators (diffusion). CRPS dominates as the headline metric; **PIT and per-regime CRPS are essentially absent** — analog_mc's diagnostic stack is unusually rigorous.
- **Where analog_mc sits.** Non-parametric and sample-based like diffusion, but driven by k-NN block bootstrap rather than a learned generator. Closest in *output form* to Li et al.; closest in *target domain* to Wang & Xu.
- **Strongest empirical baseline.** **Wang & Xu (2503.06929)** — only paper of the four with an equity benchmark and CRPS; GMM head is cheap enough to retrain at our 60-day horizon over the same 76 walk-forward folds.

---

## Cross-tier synthesis

### What's unique to analog_mc
1. **Probabilistic 60-day path forecasting on equities** with both CRPS and PIT — Tier 1 has no path output, Tier 5 has no PIT, Tier 3 has no calibration evaluation.
2. **State-conditional block re-matching** (v2.2 / Cell D) — not found in the block-bootstrap literature surveyed.
3. **Per-vol-regime CRPS breakdown + decision-rule trigger system** — absent from every paper above. Closest is Tier 4's Basel traffic-light, which is a single-statistic test rather than a regime-stratified diagnostic.

### Where the literature exposes gaps
| Gap | Tier | Fix candidate |
|---|---|---|
| 3-z-score state likely non-Markovian | 2 (Burov) | Add delay-coordinate features (v4) |
| Raw-block aggregation inherits Jacobian bias | 2 (Platzer) | Analog + local linear regression on block returns |
| Point-estimated `n_eff` weights | 2 (McDermott–Wikle) | Dirichlet posterior over weights |
| σ-ratio scaling can't recover GARCH ACF | 4 (all three) | E9 / v3b (already in V3_PLAN) |
| No textbook-FHS baseline | 4 | **Add as new V3 baseline** — isolates analog-block selection from σ-conditioning |

### Concrete head-to-head benchmarks

Ordered by cost / informativeness:

1. **OFTER** (2304.03877) — k-NN on equities with maximal-correlation distance. Re-implement on NASDAQ100; ablation target is our composite-Euclidean z-score distance.
2. **Textbook FHS** — GARCH(1,1) + residual bootstrap on NASDAQ100. Cheap; isolates whether analog-block selection beats plain residual-bootstrap. Pre-requisite for clean E9 attribution.
3. **GMM head (Wang & Xu, 2503.06929)** — only Tier-5 paper with equities + CRPS; cheap to retrain at our 60-day horizon over the same 76 folds.

### Borrowable ideas catalog

- **Delay-coordinate features** (Tier 2: Zhao–Giannakis, Burov) — extends our state beyond 3 rolling means.
- **Analog + local linear regression** (Tier 2: Platzer) — absorbs first-order Jacobian bias.
- **Dirichlet posterior on analog weights** (Tier 2: McDermott–Wikle) — propagates selection uncertainty into PIT.
- **Block-bootstrap KS / PIT GoF check** (Tier 3: Chandy) — formal calibration test atop existing PIT plot.
- **Cube-root block-length rule** (Tier 3: Chandy) — bracket validation for E1 sweep.
- **PC-then-FHS sequencing** (Tier 4: Yao) — only relevant if we move to multi-asset (E8/v4).
- **Cornish-Fisher higher-moment EWMA** (Tier 4: Gabrielsen) — cheaper fallback if GARCH (E9) proves brittle.
- **GRIN reversible normalisation** (Tier 5: TimeGMM) — drop-in alternative to causal z-score scaling.
- **CRPS-as-loss** (Tier 5: VAEneu) — natural objective if we ever fit a parametric head over analog_mc samples.
- **Contrastive past↔future MI** (Tier 5: Li et al.) — alternative to grid-search for `(w, n_eff)`.

---

## References (arXiv IDs)

| Tier | ID | Short name |
|---|---|---|
| 1 | 2511.09754 | History Rhymes |
| 1 | 2502.16023 | ContraSim |
| 1 | 2512.02720 | StockMem |
| 1 | 2304.03877 | OFTER |
| 2 | 1412.3831 | Zhao–Giannakis (KAF) |
| 2 | 2005.06623 | Burov–Giannakis (KAF multiscale) |
| 2 | 2007.14216 | Platzer–Yiou (local dynamics) |
| 2 | 1701.04485 | McDermott–Wikle (hierarchical analog) |
| 3 | 2303.08968 | van Staden NN portfolio |
| 3 | 2507.07469 | Galerkin-ARIMA |
| 3 | 2511.05733 | Chandy block-bootstrap KS |
| 4 | 1710.00859 | Yao KL+FHS |
| 4 | 1206.1380 | Gabrielsen EWMA-CF |
| 4 | 0904.0624 | Ortega SDE-FHS |
| 5 | 2503.06929 | Wang–Xu GMM equities |
| 5 | 2405.04252 | VAEneu |
| 5 | 2601.12288 | TimeGMM |
| 5 | 2410.02168 | Li diffusion |
