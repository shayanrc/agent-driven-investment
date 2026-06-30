# _029: K-sweep (top-2/4/5) on the top-10-by-AUC cells — does concentration breach the return board?

## TL;DR (mandatory)

The deployed champions run at **K=3** (top-3 daily, equal weight ≈ 33%/name). Swept **K ∈ {2, 4, 5}** (50/25/20% per name) on the **top-10 cells by AUC** (`results/gbdt/data/r_precision_at_k.csv`) under the champion strategy (`TopKDailyKellyLabelExit` rank/equal/c=1.0/mean), test-window backtests, and checked whether any (cell, K) breaches the total-return leaderboard (`results/backtests/data/backtest_summary.csv`, #10 cutoff **+120.5%**). **Result: 0 of 40 breach.** The best is `sp500_up_50pct_50d_dd25pct_macroreal` K=4 at **+35.6%** — under a third of the cutoff. **High AUC does not buy leaderboard return:** the top-AUC cells are rare-event sp500 +40/+50% models (base_rate 0.0018–0.043) whose test-window backtests top out ~+35%, whereas the board's top is `russell1000_up_50pct` (+135.9%) over a long, denser window. Secondary finding: **K=3 is not the per-cell-optimal concentration** — best-K is **K=2 for 5/10 cells** (incl. the deployed `sp500_50` champion: K=2 +24.9% / DD −4.6% / Sharpe 2.97 vs K=3 +12.5%), K=4/5 for 3, K=3 for only 2. **No champion change** (single bull window, thin samples, no regime gate, macro cells are non-deployed F17). **No registry/board rows added.**

## Setup (mandatory)

- **Cells:** top 10 by AUC (0.897–0.923). 9 sp500 + 1 nasdaq; several are F17 macro/dasw experiments (`daswmacro`/`daswbase`/`macroreal`/`macroproxy`/`base_v2`).
- **Strategy:** `TopKDailyKellyLabelExit`, `selection_mode=rank`, `sizing_mode=equal`, `fractional_c=1.0`, `selection_bound=mean`, `rank_by=calibrated` — the deployed champion config, varying **only K**. K=2→50%/name, 3→33%, 4→25%, 5→20%.
- **Window:** each cell's own test window (comparison_end = test_end + horizon business days, clipped to data; late positions marked-to-market); gross of costs, no regime gate — the `backtest_summary.csv` convention.
- **Reuse / no inference:** K is a post-prediction knob, so this reuses each cell's committed `predictions/test.csv` (no feature build, no re-inference); per cell, fit the VAL calibrator + load prices **once**, then sweep K. Whole sweep ~10 min. Driver: `scripts/backtests/k_sweep_topauc.py`.

## Results (mandatory)

Total return by K (board #10 cutoff = +120.5%; **none breach**):

| cell | AUC | base_rate | K=2 | K=3 | K=4 | K=5 | best K |
|---|--:|--:|--:|--:|--:|--:|---|
| nasdaq100_up_40pct_50d_dd20pct_aligned_mixmatch | 0.923 | 0.037 | −7.9% | −8.3% | −6.0% | −4.4% | K5 −4.4% |
| sp500_up_50pct_25d_dd25pct_daswmacro | 0.918 | 0.002 | +24.5% | +11.0% | +8.3% | +20.6% | K2 +24.5% |
| sp500_up_50pct_50d_dd25pct_macroreal | 0.913 | 0.010 | +14.6% | +24.3% | **+35.6%** | +27.3% | K4 +35.6% |
| sp500_up_50pct_50d_dd25pct_base_v2 | 0.907 | 0.010 | +8.2% | +14.9% | +15.7% | +14.8% | K4 +15.7% |
| sp500_up_40pct_25d_dd20pct_daswbase | 0.905 | 0.005 | +16.8% | −7.0% | +14.1% | +12.5% | K2 +16.8% |
| sp500_up_50pct_50d_dd25pct_macroproxy | 0.901 | 0.010 | +34.1% | +18.8% | +31.4% | +32.4% | K2 +34.1% |
| sp500_up_50pct_100d_dd25pct_aligned | 0.901 | 0.043 | +11.9% | +6.1% | +3.2% | +7.4% | K2 +11.9% |
| sp500_up_50pct_25d_dd25pct_daswbase | 0.900 | 0.002 | +30.0% | +4.2% | +18.2% | +25.3% | K2 +30.0% |
| sp500_up_50pct_50d_dd25pct_agentloop | 0.899 | 0.026 | +24.9% | +12.5% | +5.3% | +6.1% | K2 +24.9% |
| sp500_up_20pct_5d_dd10pct_daswmacro | 0.897 | 0.003 | −7.2% | +10.0% | +5.1% | +2.0% | K3 +10.0% |

Full per-K maxDD / Sharpe / entries in `results/backtests/data/_029_k_sweep_topauc.csv`. Best total overall = macroreal K=4 **+35.6%**; board #10 = +120.5%, top = `russell1000_up_50pct` +135.9%.

- **0/40 breach** — the entire top-AUC cohort sits ≤ +35.6%, far under +120.5%. **AUC rank ⊥ total-return rank:** AUC rewards correctly ranking rare positives; the board rewards compounding over a window, where `russell1000_up_50pct` (long window, denser positives) dominates. The rare-event sp500 +50% cells (base_rate ≤ 0.01) don't fire enough trades to compound onto the board.
- **K=3 is not the per-cell optimum.** Best-K: K=2 ×5, K=4 ×2, K=5 ×1, K=3 ×2. More concentration (K=2, 50%/name) wins most often — when AUC is high the top-1/2 picks carry the return. Most striking on the **deployed `sp500_50` champion**: K=2 **+24.9%** (DD −4.6%, Sharpe **2.97**) vs K=3 +12.5% — narrower K ~doubles return at lower DD on this window. But K=2 is also worst for two cells (nasdaq, sp500_20), so it's cell-specific, not universal.

## Caveats (mandatory)

- **Thin samples** — most cells fire only 3–12 entries over their test window (the +20%/5d cell is the exception at 41–100); single window, gross of costs, no regime gate. The agentloop K=2 Sharpe 2.97 rests on 3 entries.
- **Macro/dasw cells are F17** — failed second-window validation, not deployed (`[[project-gbdt-macro-features-f17]]`); high AUC here is **not** a promotion signal.
- **Test-window, not forward-OOS** — each cell's own (often short) test window, not the live forward cadence.

## Verdict (mandatory)

**Null on the headline question: no top-10-AUC cell breaches the return board at any K** — high AUC does not translate into leaderboard return (rare-event cells; AUC ⊥ compounding). The secondary observation — **K=3 is not per-cell optimal and K=2 often dominates** (notably the deployed `sp500_50` champion, ~2× return at lower DD on this window) — is a genuine but **single-window, thin-sample** signal: **parked as a follow-up** (re-test K=2 vs K=3 for the deployed champions on the forward-OOS cadence + a bear window before any change), **NOT a champion change here.** No registry or `backtest_summary.csv` rows added (0 breaches; sizing/concentration sensitivity only). Driver: `scripts/backtests/k_sweep_topauc.py`; data: `results/backtests/data/_029_k_sweep_topauc.csv`.
