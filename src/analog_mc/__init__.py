"""analog_mc — analog Monte Carlo forecasting pipeline.

See ``docs/analog_mc/IMPLEMENTATION_PLAN.md`` for the design spec.
"""

from analog_mc.config import Config
from analog_mc.data import Fold, generate_folds, load_close_series, load_returns, log_returns
from analog_mc.distances import composite_distance, distances_to_probs
from analog_mc.features import causal_ewma_vol, causal_trailing_mean, causal_zscore, compute_features
from analog_mc.sampling import generate_paths, sample_analog_blocks, scale_block
from analog_mc.search import (
    SearchResult, evaluate, generate_weight_grid, grid_search, local_refine, run_search,
)
from analog_mc.diagnostics import (
    FoldArtifacts,
    RunArtifacts,
    acf_comparison,
    aggregate_crps_overall,
    aggregate_crps_per_fold,
    aggregate_crps_per_step,
    aggregate_crps_per_vol_regime,
    clip_hit_summary,
    concatenate_oos,
    conditional_pit_by_vol_regime,
    crps_surface_plot,
    decision_rules,
    fixed_weight_baseline_crps,
    generate_report,
    global_pit_histogram,
    load_run,
    per_step_crps,
    pit_ranks,
    reliability_diagram,
    weight_trajectory_plot,
)
from analog_mc.walk_forward import FoldOutcome, create_run_dir, run_walk_forward
from analog_mc.scoring import crps_ensemble, crps_per_step, crps_sample
from analog_mc.simulate import eligible_candidates, forecast

__all__ = [
    "Config",
    "Fold",
    "FoldArtifacts",
    "FoldOutcome",
    "RunArtifacts",
    "SearchResult",
    "acf_comparison",
    "aggregate_crps_overall",
    "aggregate_crps_per_fold",
    "aggregate_crps_per_step",
    "aggregate_crps_per_vol_regime",
    "causal_ewma_vol",
    "causal_trailing_mean",
    "causal_zscore",
    "clip_hit_summary",
    "composite_distance",
    "compute_features",
    "concatenate_oos",
    "conditional_pit_by_vol_regime",
    "create_run_dir",
    "crps_surface_plot",
    "crps_ensemble",
    "crps_per_step",
    "crps_sample",
    "decision_rules",
    "distances_to_probs",
    "eligible_candidates",
    "evaluate",
    "fixed_weight_baseline_crps",
    "forecast",
    "generate_folds",
    "generate_paths",
    "generate_report",
    "generate_weight_grid",
    "global_pit_histogram",
    "grid_search",
    "load_close_series",
    "load_returns",
    "load_run",
    "local_refine",
    "log_returns",
    "per_step_crps",
    "pit_ranks",
    "reliability_diagram",
    "run_search",
    "run_walk_forward",
    "sample_analog_blocks",
    "scale_block",
    "weight_trajectory_plot",
]
