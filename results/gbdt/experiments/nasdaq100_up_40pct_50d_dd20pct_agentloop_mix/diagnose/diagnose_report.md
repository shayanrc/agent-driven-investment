# gbdt-diagnose — nasdaq100 up +40% / 50d / dd0.2

Artifact: `results/gbdt/experiments/nasdaq100_up_40pct_50d_dd20pct_agentloop_mix`  |  model features: 30  |  in-sample rows: 635989  |  prevalence: 0.054

## Tuning guidance (auto-flagged from the playbook)

- **PREVALENCE DRIFT** across segments (spread 0.034) → calibration ceiling likely; the lever is recency / regime-conditional calibration, **out of the FS/HP loop** (rule 5). Per-segment: {'train': 0.04036684782608695, 'val': 0.025815217391304348, 'eval': 0.05983695652173913, 'test': 0.05543478260869565}
- Pruned features: 10 (<0.01 imp); 5 have a real monotone relationship, of which 3 are redundant (collinear with a kept feature), 5 are weak/noise. → importance≈0 usually means redundant, not unrelated (rule 2).

## Top features — importance, monotonicity, interaction, constraint advice

| feature | imp | marg ρ | marg cons | model-PDP monotone? | interaction | monotone-constraint? |
|---|---:|---:|---:|---|---:|---|
| garman_klass_200 | 27.28 | +0.264 | 1.00 | yes | 0.08 | AVOID — high interaction; constraint degrades conditional structure |
| realized_vol_200 | 20.67 | +0.257 | 1.00 | yes | 0.04 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| vol_xs_rank_50 | 3.83 | +0.207 | 1.00 | yes | 0.07 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| parkinson_200 | 3.71 | +0.265 | 1.00 | yes | 0.22 | AVOID — high interaction; constraint degrades conditional structure |
| vol_xs_rank_200 | 2.58 | +0.208 | 1.00 | yes | 0.22 | AVOID — high interaction; constraint degrades conditional structure |
| moy_sin | 1.62 | -0.010 | 0.44 | yes | 0.04 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| index_runup_50 | 1.53 | +0.053 | 0.78 | yes | 0.16 | AVOID — high interaction; constraint degrades conditional structure |
| vol_xs_zscore_50 | 1.37 | +0.206 | 1.00 | NO (dip -83%) | 0.10 | AVOID — high interaction; constraint degrades conditional structure |
| vol_xs_zscore_100 | 1.34 | +0.215 | 1.00 | yes | 0.05 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| beta_100 | 1.22 | +0.095 | 0.78 | yes | 0.01 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| realized_vol_zscore_200 | 1.15 | +0.039 | 0.67 | yes | 0.09 | AVOID — high interaction; constraint degrades conditional structure |
| index_drawdown_200 | 1.14 | -0.074 | 1.00 | NO (dip -65%) | 0.16 | AVOID — high interaction; constraint degrades conditional structure |
| return_xs_zscore_200 | 1.10 | -0.017 | 0.56 | yes | 0.11 | AVOID — high interaction; constraint degrades conditional structure |
| garman_klass_50 | 0.96 | +0.266 | 1.00 | yes | 0.06 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| index_vol_20 | 0.95 | +0.121 | 0.89 | yes | 0.06 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| index_runup_200 | 0.95 | +0.036 | 0.56 | yes | 0.10 | AVOID — high interaction; constraint degrades conditional structure |
| dollar_move_xs_rank | 0.84 | +0.022 | 0.56 | yes | 0.04 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| returns_skew_200 | 0.81 | +0.012 | 0.56 | yes | 0.14 | AVOID — high interaction; constraint degrades conditional structure |
| runup_50 | 0.80 | +0.123 | 0.78 | yes | 0.05 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| index_return_200 | 0.76 | +0.015 | 0.56 | NO (dip -100%) | 0.06 | AVOID — model learned a non-monotone (e.g. inverted-U) shape here |
| index_return_50 | 0.00 | -0.010 | 0.56 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| drawdown_20 | 0.00 | -0.148 | 1.00 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| runup_200 | 0.00 | +0.107 | 0.67 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| parkinson_50 | 0.00 | +0.266 | 1.00 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| garman_klass_100 | 0.00 | +0.266 | 1.00 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| sma_distance_200 | 0.00 | -0.033 | 0.56 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| vol_xs_rank_100 | 0.00 | +0.214 | 1.00 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| vol_xs_zscore_200_outside_band_2 | 0.00 | +0.102 | 0.78 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| stock_return_zscore_100_outside_band_1 | 0.00 | -0.007 | 0.78 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |
| realized_vol_zscore_200_outside_band_1 | 0.00 | +0.040 | 0.67 | yes | 0.00 | NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost) |

## Top pairwise interactions

| feature A | feature B | strength |
|---|---|---:|
| index_runup_50 | parkinson_200 | 0.09 |
| parkinson_200 | return_xs_zscore_200 | 0.04 |
| vol_xs_zscore_50 | vol_xs_rank_200 | 0.04 |
| garman_klass_200 | vol_xs_rank_200 | 0.04 |
| vol_xs_rank_200 | realized_vol_zscore_200 | 0.04 |
| realized_vol_200 | vol_xs_zscore_50 | 0.04 |
| index_drawdown_200 | parkinson_200 | 0.04 |
| returns_skew_200 | vol_xs_rank_50 | 0.04 |
| return_xs_zscore_200 | vol_xs_rank_200 | 0.04 |
| index_vol_20 | index_drawdown_200 | 0.03 |
| index_return_200 | vol_xs_rank_200 | 0.03 |
| index_drawdown_200 | vol_xs_rank_200 | 0.03 |
| vol_xs_zscore_100 | moy_sin | 0.03 |
| vol_xs_rank_50 | realized_vol_zscore_200 | 0.03 |
| index_drawdown_200 | index_runup_50 | 0.03 |

_Figures: `figs/corr_heatmap.png`. Full numerics: `diagnose.json`._

See `.claude/memories/project-gbdt-tuning-playbook.md` for the rules referenced above.
