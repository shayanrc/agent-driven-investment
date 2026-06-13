# Back-Test Index

One line per back-test memo. Format: `- [_<NNN>: <short-name>](_<NNN>_<short-name>.md) — <one-line hook>`.

Plans (work-in-progress and historical):
- [V1 — Cell-5 Bayesian + Kelly Back-Test](V1_cell5_bayesian_kelly_plan.md) — first cross-module back-test plan; new modules `src/calibration/` + `src/trading_strategies/`

Memos:
- [_001: cell5_bayesian_kelly](_001_cell5_bayesian_kelly.md) — cell-5 (R-p@3=0.7556) half-Kelly back-test: +6.7% vs NDX +23.2%; underperformance dominated by a 132-day forced-cash signal gap, near-match (+7.6% vs +8.2%) while invested
- [_002: cell5b_fresh_oos](_002_cell5b_fresh_oos.md) — fresh-OOS (trained model scored on refreshed data, no retrain): +9.0% vs NDX +21.6%; signal persists (67% realized win rate) but half-Kelly + breakeven exits keep it ~59% cash; no gap confound
- [_003: cell5b_sizing_sweep](_003_cell5b_sizing_sweep.md) — fresh-OOS c×selection sweep: **quarter-Kelly + mean = +25.4% (DD −3.6%), first config to BEAT NDX +21.6%**; underperformance in _001/_002 was a half-Kelly over-concentration artifact; p_low selection HURTS (filters out winners)
- [_004: best_agent_cells_survey](_004_best_agent_cells_survey.md) — surveyed top-5 R-p@K agent cells at headline c=0.25+mean: **only cell-5 trades; 4/5 (rarer-event cells) make 0 trades — calibrated p never clears the 1/3 Kelly breakeven (base-rate gate)**; `_001` cell re-run +6.7%→+11.5% at c=0.25. Motivates rank/base-rate-relative sizing (V1.2)
- [_005: rank_sizing](_005_rank_sizing.md) — **V1.2 rank mode (top-K by rank, no absolute gate) + equal/rank_kelly sizing: the 0-trade cells now trade and 4/5 BEAT their index OOS** (ndx40 +61.5% vs NDX +36.8%, sp500_20 +58.1% vs +8.4%, r1k +49.1% vs +40.2%). The R-p@K ranking IS tradeable; absolute-Kelly was discarding signal. Caveat: single windows, small N — needs rolling validation before any alpha claim

Dependency graph: [figs/deps_graph.png](figs/deps_graph.png) (DOT source `figs/deps_graph.dot`).

See [CONVENTIONS.md](CONVENTIONS.md) for the memo template, numbering rules, registry CSV schema, and naming conventions.
