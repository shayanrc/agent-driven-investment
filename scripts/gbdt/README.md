# scripts/gbdt — what's durable vs. what already served its purpose

Ad-hoc orchestration and analysis scripts for the gbdt module. Everything here
stays for provenance (memos under `docs/gbdt/` cite these paths), but only the
first groups below are living tooling — don't reach for a one-shot when
starting new work.

## Durable tooling (use these)

| Script | Role |
|---|---|
| `diagnose.py` | Runner behind the `/gbdt-diagnose` skill |
| `compute_r_precision.py` | Canonical post-hoc R-Precision@K (the only sanctioned implementation — see CLAUDE.md) |
| `regenerate_r_precision_at_k_csv.py` | Rebuilds the canonical registry CSV |
| `loop_status.py` | Agent-loop progress inspection |
| `run_agent_loop_resumable.sh` | Exit-and-resume agent-loop driver |
| `check_feature_parity.py`, `profile_feature_build.py` | Feature-build correctness/perf gates |
| `backfill_csv_segment_dates.py` | Registry maintenance (V1.4 date columns) |

## Sweep-campaign tooling (reusable pattern, tied to past campaigns)

Spec generators + launchers + aggregators come in matched sets; regenerate a
campaign end-to-end rather than reusing its artifacts.

- Generators: `gen_aligned_champion_specs.py`, `gen_macro_sweep_specs.py`,
  `gen_fund_sweep_specs.py`, `gen_f19_sweep_specs.py`
- Launchers: `run_{sp500,nasdaq100,russell1000,nifty_next_50}_sweep.sh`,
  `run_macro_sweep.sh`, `run_fund_sweep.sh`, `run_f19_sweep.sh`,
  `run_f18_top3_tune.sh`, `run_f18_w2_confirmation.sh`
- Aggregators: `aggregate_{sp500,nasdaq100,russell1000,macro,fund}_sweep.py`,
  `plot_fund_sweep_heatmap.py`

## Reusable diagnostics (analysis passes, run on demand)

`pdp_and_corr.py`, `feature_corr_heatmap.py`, `pruned_feature_investigation.py`,
`interaction_before_after.py`, `interaction_constraints_capability_check.py`,
`monotone_1d_audit.py`, `monotonic_feature_analysis.py`,
`nse_anti_predictive_cross_cell.py`

## One-shot, completed (kept for provenance only)

| Script(s) | Served |
|---|---|
| `v0_opportunity_scan{,_full,_filtered}.py`, `v0_universe_eligibility.py` | V0 EDA; outputs in `results/gbdt/data/` |
| `_214_*.py` (3) | Task #214 manual cell-1/3 tuning |
| `_226_*.py` (2) | Bug #226 cache-drift triage + cleanup |
| `_228_*.sh` (2) | Historical H100/H200 sweep launches |
| `acceptance_check_147.py` | Task #147 acceptance |
| `phase8_interaction_constraints_verify.py` | V1.2 Phase 8 verification |
| `backtest_top10_revalidation_pilot.py`, `rescore_revalidation_memo_window.py` | Revalidation pilots for memos #194–#195 |
