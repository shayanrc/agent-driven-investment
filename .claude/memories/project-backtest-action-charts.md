---
name: project-backtest-action-charts
description: scripts/backtests/plot_actions.py is the reusable backtest action-chart verb (auto-emitted by run_backtest_cell); _020 settled that capital-metering is a tested dead-end — concentration in the top ranks is the edge
metadata:
  type: project
---

**Reusable tool — don't reinvent ad-hoc chart code.** `scripts/backtests/plot_actions.py`
renders a single-window back-test's **action chart**: strategy equity + universe-index
buy-hold + every buy ▲ / sell ▼ marked on the strategy curve with the ticker as a
**horizontal label stacked vertically** (entries up, exits down), exits suffixed with the
trigger (`·t` target / `·D` drawdown-stop / `·h` horizon / `·b` breakeven). It is **wired
into `run_backtest_cell.py` (non-fatally)**, so every single-window back-test auto-emits
`figs/actions.png` as a standard artifact. Use it (or its `plot_actions(run_dir)` import)
for any back-test equity/action visualization.

- **CLI:** `uv run python -m scripts.backtests.plot_actions <run_dir> [<run_dir> ...]`
  (batch-safe: one bad dir → SKIP, not abort).
- **Schema-robust** across run-dir vintages: geometry-less older summaries (`_001`–`_003`),
  the `ndx_bh` vs `index_bh` benchmark-key split, and `test_*` / `fresh_*` window keys. It
  **recomputes** the index curve via `benchmarks.buy_and_hold` (the index equity series is
  never persisted) — necessary to back-fill old dirs, and the window is identical so the
  drawn curve agrees with the persisted `index_bh` metric.
- **0-trade runs render** as a flat $100K cash line vs the index (the gate finding made
  visual). **Rolling/regime dirs can't be charted** (no per-trade `picks.csv`) — `plot_actions`
  raises a clear `ValueError` on a summary lacking `window`/`strategy`.
- Back-filled across `_001`–`_005` + `_020`; each run dir carries its own `figs/actions.png`.

**Why it exists — the `_020` finding (a tested dead-end; don't re-propose it).** The
deployed sp500 champion is **rank-select, equal-weight, K=3, c=1.0 → ~33%/name**, so it is
~fully invested in 3 names on day 1 and **locked until one exits**. Metering capital to
capture the fresh signals it skips — **5%/name, ≤15%/day**, accumulating ~20 names — was
tested in `_020`: it captures **7× the trades** (sp500_20 13→92 entries, 8→37 names) and
higher exposure, **but lowers return** (+58.1%→+22.3%) and **erases the edge on the rarer
+50% cell** (+12.5%→−3.3%, below SPX). The edge is concentrated in the very top ranks;
diluting into ranks 4–20 buys near-noise (DD-stops 2→34) and ~10× turnover makes it
cost-fragile. **Concentration monetizes top-rank precision** — consistent with `_012`/`_014`
and `[[project-r-precision-methodology]]` (R-Precision@1 is where the signal lives). See
`docs/backtests/_020_capital_metering.md`.
