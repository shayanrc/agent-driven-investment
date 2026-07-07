# scripts/backtests — what's durable vs. what already served its purpose

Runners and analysis scripts for the cross-module backtest harness (see
`docs/backtests/INDEX.md` for the per-experiment memos). Everything stays for
provenance; the first groups are living tooling.

## Daily forward-OOS cadence (load-bearing, systemd-scheduled)

| Script | Role |
|---|---|
| `daily_forward_predictions.py` | The `/daily-predictions` runner (memo `_019`) |
| `infer_fresh_predictions.py` | Incremental re-scoring on the freshest panel |
| `fetch_ticker_names.py` | Incremental company-name cache refresh |
| `systemd/` | User-timer units (+ its own README) |

## Backtest execution + validation harness

`run_backtest_cell.py`, `run_cell5_bayesian_kelly.py`, `run_fresh_oos.py`,
`run_rolling_validation.py`, `calibration_step.py`, `benchmarks.py`

## Regime + consensus machinery

`regime_signals.py`, `regime_conditioning.py`, `bootstrap_regime.py`,
`consensus_backtest.py`, `consensus_variant_grid.py`

## Analysis / aggregation (run on demand)

`analyze_forward_prob.py` (re-emits the gitignored `predicted_prob_*` figs),
`k_sweep_topauc.py`, `k_sweep_run_artifacts.py`, `true_oos_rprecision.py`,
`regenerate_backtest_performance_csv.py`, `plot_actions.py`,
`nse_signal_forensics.py`

## One-shot, completed (kept for provenance only)

| Script(s) | Served |
|---|---|
| `migrate_forward_log_v2.py` | Forward-log schema migration (2026-06-27) |
| `ev_threshold.py` | EV-threshold study |
| `consensus_bear_check.py`, `consensus_june_check.py`, `consensus_grid_cleanleaky.py` | Point-in-time consensus checks behind their memos |
