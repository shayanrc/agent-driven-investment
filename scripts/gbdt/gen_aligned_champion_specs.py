"""Generate date-aligned *matched* specs for the trailing-split backtested champions.

For each trailing champion model we emit a spec that reuses its EXACT resolved config
(backend, HP overrides, pruned feature set, calibration) but swaps the split to
`date_aligned` (train_start 2019-01-01) and does a single fit (`max_iterations: 1`).

This isolates the split effect: same recipe, different evaluation window — the clean
"their training on the date-aligned version" comparison (matched-HP A/B methodology,
see docs/gbdt/_262 + project-gbdt-macro-features-f17). Existing trailing models are
untouched; these are NEW `*_aligned_*match` cells.

Usage: uv run python -m scripts.gbdt.gen_aligned_champion_specs
"""
import os
import yaml

EXP = "results/gbdt/experiments"
OUT = "configs/gbdt/experiments"

# source trailing champion dir -> new date-aligned matched experiment name
MODEL_MAP = {
    "sp500_up_20pct_25d_dd10pct_agentloop": "sp500_up_20pct_25d_dd10pct_aligned_champmatch",
    "sp500_up_50pct_50d_dd25pct_agentloop": "sp500_up_50pct_50d_dd25pct_aligned_champmatch",
    "nasdaq100_up_40pct_50d_dd20pct_agentloop_mix": "nasdaq100_up_40pct_50d_dd20pct_aligned_mixmatch",
    "nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen": "nasdaq100_up_10pct_50d_dd5pct_aligned_revalmatch",
    "nasdaq100_up_10pct_50d_dd5pct_b_acceptance_agent": "nasdaq100_up_10pct_50d_dd5pct_aligned_baccmatch",
}


def build(src_dir, new_name):
    d = os.path.join(EXP, src_dir)
    hp = yaml.safe_load(open(os.path.join(d, "hp.yaml")))["hp"]
    feats = yaml.safe_load(open(os.path.join(d, "features.yaml")))["features"]
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    tgt = spec["target"]
    be = spec.get("backend", {})
    # all 16 families = 279 cols; only pin an explicit list when the champion pruned
    # The feature builder takes family tokens / "all", NOT individual column names,
    # and `exclude` is a post-build glob drop. So to pin a PRUNED set we build "all"
    # (hits the universe cache) and exclude the complement (the columns the champion
    # dropped). The full 279-col pool = an all-features champion's feature list.
    full_pool = yaml.safe_load(open(os.path.join(
        EXP, "sp500_up_20pct_25d_dd10pct_agentloop", "features.yaml")))["features"]
    pruned = len(feats) < len(full_pool)
    if pruned:
        missing = set(feats) - set(full_pool)
        assert not missing, f"{new_name}: selected features not in full pool: {sorted(missing)[:5]}"
        exclude = sorted(set(full_pool) - set(feats))
    else:
        exclude = None
    features_block = {"candidates": "all"}
    if exclude:
        features_block["exclude"] = exclude
    # cast numpy-ish HP scalars to plain python for clean YAML
    hp_clean = {k: (int(v) if isinstance(v, float) and v.is_integer() else float(v) if isinstance(v, float) else v)
                for k, v in hp.items()}
    out = {
        "experiment_name": new_name,
        "target": {
            "universe": tgt["universe"],
            "direction": tgt["direction"],
            "threshold_pct": tgt["threshold_pct"],
            "horizon_days": tgt["horizon_days"],
            "max_drawdown": tgt["max_drawdown"],
        },
        "split": {"mode": "date_aligned", "train_start": "2019-01-01"},
        "features": features_block,
        "backend": {
            "library": be.get("library", "xgboost"),
            "calibration_method": be.get("calibration_method", "conditional_isotonic"),
            "hp_starting": hp_clean,
            "fs_hp_loop": {
                "callback_mode": "default",
                "max_iterations": 1,
                "plateau_threshold": 0.005,
                "degradation_gate": 0.01,
            },
        },
        "random_seed": 42,
    }
    header = (
        f"# {new_name}\n"
        f"# Date-aligned MATCHED refit of trailing champion `{src_dir}`.\n"
        f"# Pins its backend + HP overrides + {len(feats)}-feature set"
        + (f" (build all, exclude the {len(exclude)} dropped cols)" if exclude else "")
        + f"; split -> date_aligned\n"
        f"# (train_start 2019-01-01), single fit. Isolates the split effect. --snapshot-end at runtime.\n"
    )
    path = os.path.join(OUT, f"{new_name}.yaml")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    return path, len(feats), hp_clean


if __name__ == "__main__":
    for src, new in MODEL_MAP.items():
        p, nf, hp = build(src, new)
        print(f"wrote {p}  (n_feat={nf}, hp={hp})")
