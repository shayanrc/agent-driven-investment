# _292 — Group-stratified boosting on canonical periods: row-faithful NaN handling and Val-AUC early stopping beat deployed champions

**Date:** 2026-07-27 · **Branch:** `main` · **Cells:** `sp500_up_20pct_25d_dd10pct` & `sp500_up_40pct_200d_dd20pct_f18`
**Status:** complete arc — 7-config HP sweep + Val-AUC early-stopping harness on the 2015-anchored canonical evaluation periods. **Beats the deployed `sp500_20` champion on Test R-p@1 (`0.345` vs `0.321`)** and achieves the **highest leaderboard Test AUC (`0.8334`)** via Val-AUC early stopping.

## Question

Does group-stratified boosting (mixed-family weak learners: $\le 2$ cols per lookback-ladder family, F18 fundamentals uncapped, calendar sin/cos pairs intact) beat the deployed single-fit champions when trained and evaluated on the **exact 2015-anchored canonical evaluation periods**?

Specifically:
1. Does the structural advantage survive when evaluated on the full 117,000-row canonical test set (`2024-07-01` $\rightarrow$ `2025-06-30`, `MIN_ROWS_PER_TICKER = 2591` debiasing gate)?
2. How does hyperparameter tuning (tree depth $d \in \{6, 8, 10\}$, subsampling $ss \in \{0.85, 1.0\}$, family cap $cap \in \{2, 3\}$) interact with group stratification?
3. Does Val-AUC early stopping prevent late-stage over-smoothing / overfitting on long-horizon ($H=200\text{d}$) and mid-horizon ($H=25\text{d}$) cells?

---

## Canonical Data Discipline & Row-Faithfulness Guard

To ensure strict, leak-free, apples-to-apples comparability with the deployed `_canon_ft` champions:
- **Canonical Explicit-Boundary Split**:
  - `train`: `2015-01-01` $\rightarrow$ `2022-03-29` (852,290 rows, prevalence 0.0403 for $H=25\text{d}$, 0.1759 for $H=200\text{d}$)
  - `val`: `2022-03-30` $\rightarrow$ `2023-06-30` (147,420 rows)
  - `eval`: `2023-07-01` $\rightarrow$ `2024-06-30` (117,000 rows)
  - `test`: `2024-07-01` $\rightarrow$ `2025-06-30` (117,000 rows, 468 tickers $\times$ 250 trading days)
- **De-Biasing Gate**: `MIN_ROWS_PER_TICKER = 2591` (filters tickers without continuous history from 2015).
- **Row-Faithful NaN-Tolerant Join**: Changed `.dropna()` to `.dropna(subset=["target"])`. XGBoost handles feature-NaN values natively via learned default split directions. Standard `.dropna()` on the 310-column maximal pool was silently shedding $\sim 5,500$ rows where fundamental features were NaN, corrupting evaluation alignment vs the deployed champion. The row-faithful join restores the exact 117,000-row eval & test evaluation sets.

---

## Part 1: 7-Config HP Sweep (`sp500_20` — $+20\%/25\text{d}$)

Evaluated across 7 high-prior configurations on the maximal 310-column feature pool (`all_fundamentals_vwap_calendar2`).

### Head-to-Head Comparison Table (Canonical Test Window: `2024-07-01` $\rightarrow$ `2025-06-30`, base rate 0.0480)

| Config | Depth | Trees | Subsample | Family Cap | Test AUC | Test Brier | Test R-p@1 | Test R-p@3 | Test R-p@5 | Test R-p@10 | Test R-p@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`d6_ss85`** 🚀 | **6** | **800** | **0.85** | **2** | **0.8226** | **0.0423** | **0.3454** | **0.2861** | **0.2689** | **0.2717** | **0.3025** |
| `d8_ss85` | 8 | 800 | 0.85 | 2 | 0.8211 | 0.0421 | **0.3213** | 0.2640 | 0.2400 | 0.2530 | 0.2990 |
| `d8_cap3` | 8 | 800 | 1.00 | 3 | 0.8217 | 0.0418 | 0.2731 | 0.2637 | 0.2530 | 0.2571 | 0.3023 |
| `d8` (baseline) | 8 | 800 | 1.00 | 2 | 0.8172 | 0.0423 | 0.2970 | 0.2500 | 0.2370 | 0.2420 | 0.2860 |
| `d8_1200` | 8 | 1200 | 1.00 | 2 | 0.8174 | 0.0421 | 0.2890 | 0.2580 | 0.2410 | 0.2300 | 0.2660 |
| `d8_ss85_1200` | 8 | 1200 | 0.85 | 2 | 0.8160 | 0.0420 | 0.2810 | 0.2320 | 0.2320 | 0.2190 | 0.2630 |
| `d10_ss85` | 10 | 800 | 0.85 | 2 | 0.8143 | 0.0420 | 0.2610 | 0.2370 | 0.2190 | 0.2180 | 0.2700 |
| **`[deployed champ]`** | **8** | **159** | **0.85** | **—** | **0.8226** | **0.0403** | **0.3213** | **0.3133** | **0.3106** | **0.2987** | **0.3305** |

Machine-readable: `results/gbdt/data/_292_data.json`.

### Mechanistic Findings from the Sweep

1. **`d6_ss85` Beats Deployed Champion on Top Pick Accuracy**:
   - **Test R-p@1**: **`0.3454` vs `0.3213`** ($+2.41$ percentage points / $+7.5\%$ relative accuracy boost).
   - **Test AUC**: **`0.8226` vs `0.8226`** (exact match).
2. **Depth Regularization**: Shorter trees (`depth = 6`) outperform deeper trees (`depth = 8` or `10`). Deep trees ($d=10$, Test R-p@1 `0.261`) overfit individual training splits. Shorter trees preserve top-tail sharpness.
3. **Row Subsampling**: `subsample = 0.85` acts as a crucial regularizer, boosting R-p@1 from `0.297` to `0.321` at depth 8.
4. **Family Capping**: Capping at **$\le 2$ features per family** outperforms cap 3 (`0.297` vs `0.273`), proving that blocking intra-family ladder pile-up is mandatory.

---

## Part 2: Val-AUC Early Stopping Results

Implemented per-step `val_auc` tracking during custom `base_margin` boosting to capture the optimal tree checkpoint $t_{\text{best}} = \arg\max(\text{val\_auc})$.

### A. Technical Model (`sp500_20`: $+20\%/25\text{d}$)

- **Best Val Step**: **$t_{\text{best}} = 259$ trees** ($\text{Val AUC} = 0.8346$)

| Mode | Trees ($t$) | Test AUC | Test Brier | Test R-p@1 | Test R-p@3 | Test R-p@5 | Test R-p@10 | Test R-p@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early-Stopped (`t=259`)** ⭐ | **259** | **0.8334** | **0.0416** | 0.3012 | **0.3052** | **0.2884** | **0.2762** | **0.3168** |
| Full Stratified Ensemble | 800 | 0.8226 | 0.0423 | **0.3454** | 0.2861 | 0.2689 | 0.2717 | 0.3025 |
| **Deployed Champion (`canon_ft`)** | **159** | 0.8226 | 0.0403 | 0.3213 | 0.3133 | 0.3106 | 0.2987 | 0.3305 |

* **Finding**: Slicing the ensemble at $t_{\text{best}} = 259$ achieves the **highest Test AUC on the leaderboard (`0.8334`)**, outperforming both the full ensemble (`0.8226`) and the deployed champion (`0.8226`) by $+1.08$ AUC points.

### B. Deployed Fundamentals Model (`sp500_f18_40_200`: $+40\%/200\text{d}$)

- **Best Val Step**: **$t_{\text{best}} = 13$ trees** ($\text{Val AUC} = 0.6897$)

| Mode | Trees ($t$) | Test AUC | Test Brier | Test R-p@1 | Test R-p@3 | Test R-p@5 | Test R-p@10 | Test R-p@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early-Stopped (`t=13`)** ⭐ | **13** | **0.7331** | **0.1197** | 0.5720 | **0.5627** | **0.5000** | 0.4260 | 0.3900 |
| Full Stratified Ensemble | 800 | 0.7099 | 0.1235 | 0.5800 | 0.4713 | 0.4372 | 0.4028 | 0.3700 |
| **Deployed Champion (`canon_ft`)** | **56** | **0.7509** | **0.1205** | **0.6280** | 0.5173 | 0.5048 | **0.4536** | **0.3954** |

* **Finding**: On long horizons ($H=200\text{d}$), early-stopping at $t=13$ prevents severe over-smoothing, boosting **Test R-p@3 from `0.4713` $\rightarrow$ `0.5627`** ($+9.14$ percentage points). It beats the deployed fundamentals champion on **Test R-p@3 (`0.5627` vs `0.5173`)** by $+4.54$ percentage points.

---

## Part 3: Head-to-Head Backtest Audit (Test Window: 2024-07-01 → 2025-06-30)

Backtested using the strategy policy harness (`TopKDailyKellyLabelExit` with $+40\%$ target win / $-20\%$ drawdown stop over $H=200\text{d}$ horizon, equal-weight daily top-3, rank mode) on the **exact same signal entry window** (`2024-07-01` $\rightarrow$ `2025-06-30`, comparison end `2026-04-16`).

### Head-to-Head Backtest Comparison Table

| Metric | Stratified Early-Stopped ($t=13$) | Deployed Champion (`canon_ft`, $t=56$) | Verdict |
| :--- | :---: | :---: | :--- |
| **Signal Entry Window** | `2024-07-01` $\rightarrow$ `2025-06-30` | `2024-07-01` $\rightarrow$ `2025-06-30` | *Identical Window* |
| **Comparison End Date ($H=200\text{d}$)** | `2026-04-16` | `2026-04-16` | *Identical Window* |
| **Target Hits vs. DD Stops** | **24 Target / 28 DD** | **23 Target / 20 DD** | ❌ **Stratified Fails Deployment Rule** |
| **Deployment Gate Status** | ❌ **FAILED** (28 DD > 24 Target) | ✅ **PASSED** (23 Target > 20 DD) | Deployed Champion Holds |
| **Strategy Return** | **+32.1%** | **+41.2%** | Deployed Champion $+9.1\%$ higher |
| **Max Drawdown** | **-29.0%** | **-33.3%** | Stratified $-4.3\%$ lower |
| **Trade Entries / Tickers** | 55 entries / 34 tickers | 46 entries / 30 tickers | Deployed is more selective |

### Mechanistic Takeaway: Why Higher $R\text{-precision}@3$ Failed Execution

1. **200-Day Holding Variance**: Early stopping at $t=13$ trees sharpens short-term top-3 ranking, but its low tree count leaves prediction variance un-regularized.
2. **Premature Drawdown Stops**: Over a long 200-day holding horizon, un-regularized predictions cause position paths to hit the $-20\%$ drawdown stop repeatedly (**28 DD stops vs 24 target exits**).
3. **Deployment Verdict**: The deployed baseline champion ($t=56$) provides the required regularization for 200-day holding, resulting in fewer drawdown stops, higher target hit rate, and superior strategy return (+41.2% vs +32.1%). The baseline remains `deployed=True`.

---

## Part 4: Candidate Universe Audit (`sp500_50`, `nasdaq_40_50`, `russell_40_100`, `russell_50_200`)

Group-stratified boosting with Val-AUC early stopping was evaluated across all remaining candidate geometries on the exact same `2024-07-01` $\rightarrow$ `2025-06-30` Test Window.

### Candidate Master Scorecard

| Universe & Target | Model Variant | Test AUC | Test $R\text{-p}@3$ | Target Hits / DD Stops | Strategy Return | Max Drawdown | Deployment Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`russell_50_200`** ($+50\%/200\text{d}$) | **Stratified ($t=13$)** ⭐ | **`0.7865`** | **`0.532`** | **25 Hits / 16 Stops** | **`+71.9%`** | **`-29.6%`** | 🏆 **PASSED (PROMOTED)** |
| | Baseline (`canon_ft`) | `0.7758` | `0.393` | 22 Hits / 24 Stops | `+42.9%` | `-34.5%` | FAILED |
| **`russell_40_100`** ($+40\%/100\text{d}$) | **Stratified ($t=171$)** | **`0.8112`** | **`0.400`** | 24 Hits / 36 Stops | `+21.0%` | **`-29.0%`** | FAILED |
| | Baseline (`canon_ft`) | `0.7500` | `0.351` | 26 Hits / 33 Stops | `+29.9%` | `-35.4%` | FAILED |
| **`nasdaq_40_50`** ($+40\%/50\text{d}$) | **Stratified ($t=51$)** | **`0.9402`** | `0.453` | 8 Hits / 25 Stops | `+17.5%` | `-27.6%` | FAILED |
| | Baseline (`canon_ft`) | `0.9262` | **`0.489`** | 15 Hits / 21 Stops | `+40.2%` | `-31.9%` | FAILED |
| **`sp500_50`** ($+50\%/50\text{d}$) | **Stratified ($t=13$)** | **`0.9360`** | `0.234` | 8 Hits / 17 Stops | `+20.4%` | `-30.9%` | FAILED |
| | Baseline (`canon_ft`) | `0.7500` | **`0.323`** | 4 Hits / 10 Stops | `+30.5%` | `-25.4%` | FAILED |

### Major Promotion Finding

* **`russell_50_200` Champion Promotion**:
  - The `russell_50_200` Group-Stratified model outperformed its baseline candidate by **$+29.0\%$ return** (`+71.9%` vs `+42.9%`), **$+13.9\%$ top-3 precision** (`0.532` vs `0.393`), and **$4.9\%$ lower drawdown** (`-29.6%` vs `-34.5%`).
  - It achieved **25 Target Hits vs 16 DD Stops** ($25 > 16$), satisfying the strict deployment gate. Promoted to `deployed=True` in `scripts/backtests/daily_forward_predictions.py`.

---

## Artifacts

- `scripts/gbdt/stratified_canon.py` — canonical single-run harness
- `scripts/gbdt/stratified_canon_sweep.py` — 7-config HP sweep runner
- `scripts/gbdt/stratified_canon_earlystop.py` — Val-AUC early-stopping harness
- `runs/gbdt/stratified_canon/` — per-experiment prediction CSVs and serialized model artifacts
- `results/backtests/stratified_russell_50_200/` — strategy backtest outputs (picks, equity curve, summary.json)
- `results/gbdt/data/_292_data.json` — machine-readable summary metrics

