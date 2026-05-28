"""gbdt-diagnose — full diagnostic bundle for one fitted gbdt cell artifact.

Generalizes the five exploratory scripts from the nifty50 H=25 study
(docs/gbdt/_147) into one parametrized verb that works on ANY artifact dir.
Given a fitted cell (model.cbm + features.yaml + spec.yaml + predictions/),
it builds the in-sample feature matrix once and emits:

  - feature importance (numeric top-N)
  - prevalence drift across train/val/eval/test (the calibration-ceiling flag)
  - marginal monotonicity per feature (Spearman rho + decile consistency)
  - model 1D-PDP monotonicity audit (is the MODEL monotone? — constraint pre-check)
  - pairwise interaction strength + per-feature involvement (high/low interaction)
  - Spearman correlation heatmap (collinearity structure)
  - pruned-feature investigation (real relationship vs redundancy)
  - a "tuning guidance" section that applies the playbook rules automatically

Outputs: <out>/diagnose.json, <out>/diagnose_report.md, <out>/figs/*.png, and a
cached <out>/_insample_matrix.parquet (reused on re-run).

Pure helpers (monotonicity, redundancy, interaction involvement, prevalence
drift, tuning flags) are importable + unit-tested in tests/gbdt/test_diagnose.py.

CLI:
    uv run python -m scripts.gbdt.diagnose <artifact_dir> [--top-n 30]
        [--importance-threshold 0.01] [--out <dir>] [--no-pdp] [--no-figs]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

# Overfit threshold on train_val_gap (= val_brier - train_brier); matches the
# fallback regularization trigger in src/gbdt/train.py.
_OVERFIT_GAP_THR = 0.02

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no I/O)
# ---------------------------------------------------------------------------


def assess_overfit(train_val_gap: float | None, *, threshold: float = _OVERFIT_GAP_THR) -> bool | None:
    """Is the cell free of overfit? Based on the train/val gap ALONE.

    ``gap = val_brier - train_brier``; POSITIVE = val worse than train = overfit.
    No-overfit when ``gap <= threshold`` (default 0.02, matching the fallback
    regularization trigger in src/gbdt/train.py). Early-stopping firing is
    deliberately NOT a factor: it's the healthy mechanism that selects the tree
    count, not an overfit signal — a model can early-stop at tree 67 with a
    deeply negative gap (val better than train), which is the opposite of
    overfit. Returns None when the gap is unavailable.
    """
    if train_val_gap is None:
        return None
    return bool(train_val_gap <= threshold)


def spearman_monotonicity(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Marginal monotonicity of feature ``x`` against target ``y``.

    Returns ``rho`` (signed Spearman), ``consistency`` (fraction of adjacent
    decile steps moving in rho's direction; ~1.0 = cleanly monotone, ~0.5 =
    non-monotone/U-shaped), and the decile-0 / decile-(n-1) positive rates.
    """
    ok = np.isfinite(x)
    if ok.sum() < 50 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return {"rho": float("nan"), "consistency": float("nan"),
                "pr_lo": float("nan"), "pr_hi": float("nan"), "n": int(ok.sum())}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # benign ConstantInputWarning on near-constant cols
        rho, _ = spearmanr(x[ok], y[ok])
    if not np.isfinite(rho):
        return {"rho": float("nan"), "consistency": float("nan"),
                "pr_lo": float("nan"), "pr_hi": float("nan"), "n": int(ok.sum())}
    try:
        q = pd.qcut(pd.Series(x[ok]).rank(method="first"), n_bins, labels=False)
        pr = pd.Series(y[ok]).groupby(q.values).mean().values
    except ValueError:
        return {"rho": float(rho), "consistency": float("nan"),
                "pr_lo": float("nan"), "pr_hi": float("nan"), "n": int(ok.sum())}
    steps = np.diff(pr)
    dom = np.sign(rho) if rho != 0 else 1
    consistency = float(np.mean(np.sign(steps) == dom))
    return {"rho": float(rho), "consistency": consistency,
            "pr_lo": float(pr[0]), "pr_hi": float(pr[-1]), "n": int(ok.sum())}


def pdp_1d(model, base_vals: np.ndarray, feat_idx: int, grid: np.ndarray) -> np.ndarray:
    """1D partial dependence: mean predicted P(+) sweeping one feature over a grid."""
    out = np.zeros(len(grid))
    for k, gv in enumerate(grid):
        g = base_vals.copy()
        g[:, feat_idx] = gv
        out[k] = model.predict_proba(g)[:, 1].mean()
    return out


def pdp_monotonicity(curve: np.ndarray, tol: float = 1e-4) -> dict:
    """Is a 1D-PDP curve monotone? Returns direction + the worst dip/rise
    as a fraction of the curve's range (signed: negative = downward dip in an
    otherwise-increasing curve)."""
    d = np.diff(curve)
    rng = max(curve.max() - curve.min(), 1e-12)
    inc = bool(np.all(d >= -tol))
    dec = bool(np.all(d <= tol))
    worst_dip = float(min(d.min(), 0.0)) / rng
    worst_rise = float(max(d.max(), 0.0)) / rng
    return {"monotone": inc or dec,
            "increasing": inc and not dec,
            "decreasing": dec and not inc,
            "worst_dip_frac": worst_dip,
            "worst_rise_frac": worst_rise}


def interaction_involvement(model) -> dict[str, float]:
    """Total pairwise-interaction strength involving each feature."""
    names = model.feature_names_
    involve: dict[str, float] = {n: 0.0 for n in names}
    for i1, i2, s in model.get_feature_importance(type="Interaction"):
        involve[names[int(i1)]] += s
        involve[names[int(i2)]] += s
    return involve


def top_interaction_pairs(model, k: int = 15) -> list[tuple[str, str, float]]:
    names = model.feature_names_
    out = []
    for i1, i2, s in model.get_feature_importance(type="Interaction")[:k]:
        out.append((names[int(i1)], names[int(i2)], float(s)))
    return out


def prevalence_drift(seg_prevalence: dict[str, float]) -> dict:
    """Given per-segment positive prevalence, flag non-stationarity."""
    vals = [v for v in seg_prevalence.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return {"spread": float("nan"), "drift_flag": False, "monotone_decline": False}
    spread = float(max(vals) - min(vals))
    order = [seg_prevalence.get(s) for s in ("train", "val", "eval", "test")
             if seg_prevalence.get(s) is not None]
    monotone_decline = bool(len(order) >= 3 and all(
        order[i] >= order[i + 1] for i in range(len(order) - 1)))
    # flag if the spread is a large fraction of the mean prevalence
    drift_flag = bool(spread > 0.5 * (np.mean(vals) if np.mean(vals) > 0 else 1))
    return {"spread": spread, "drift_flag": drift_flag,
            "monotone_decline": monotone_decline}


def constraint_advice(marg: dict, model_pdp: dict, involvement: float,
                      involvement_high_thr: float) -> str:
    """Per-feature monotone-constraint guidance from the playbook (rule 3/4)."""
    if involvement >= involvement_high_thr:
        return "AVOID — high interaction; constraint degrades conditional structure"
    if model_pdp is not None and not model_pdp.get("monotone", True):
        return "AVOID — model learned a non-monotone (e.g. inverted-U) shape here"
    if marg is None or not np.isfinite(marg.get("rho", float("nan"))):
        return "n/a"
    # low interaction + model-monotone: still only neutral (fixed constraints-on cost)
    return "NEUTRAL at best — low interaction, but expect no gain (fixed constraints-on cost)"


# ---------------------------------------------------------------------------
# Loading + in-sample build
# ---------------------------------------------------------------------------


def load_cell(artifact_dir: Path) -> dict:
    """Load model, feature list, spec, and per-segment predictions."""
    from catboost import CatBoostClassifier

    artifact_dir = Path(artifact_dir)
    model = CatBoostClassifier()
    model.load_model(str(artifact_dir / "model.cbm"))
    spec = yaml.safe_load((artifact_dir / "spec.yaml").read_text())
    target = spec["target"]
    preds = {}
    for seg in ("train", "val", "eval", "test"):
        p = artifact_dir / "predictions" / f"{seg}.csv"
        if p.exists():
            try:
                preds[seg] = pd.read_csv(p)
            except Exception:
                preds[seg] = None
    iters = []
    itpath = artifact_dir / "iterations.jsonl"
    if itpath.exists():
        for line in itpath.read_text().strip().splitlines():
            if line.strip():
                iters.append(json.loads(line))
    return {"model": model, "spec": spec, "target": target,
            "predictions": preds, "iterations": iters}


def build_insample(target: dict, *, test_tail_rows: int,
                   cache_path: Path | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the full in-sample feature matrix (test tail dropped) + aligned
    target. Returns ``(X_insample, y_insample)`` with all candidate columns;
    callers subset to the model's feature set as needed. Caches the matrix to
    ``cache_path`` (parquet) and reuses it if present.
    """
    from gbdt import data as gbdt_data
    from gbdt import features as gbdt_features
    from gbdt.targets import build_target

    if cache_path is not None and Path(cache_path).exists():
        X = pd.read_parquet(cache_path)
        po = gbdt_data.load_panel(target["universe"], min_rows=1600)
    else:
        po = gbdt_data.load_panel(target["universe"], min_rows=1600)
        Xfull = gbdt_features.build_feature_matrix(
            po.panel, po.index_series, annualization=po.annualization_factor,
        ).dropna(axis=1, how="all")
        keep = []
        for _, g in Xfull.groupby(level="ticker"):
            keep.append(g.iloc[:-test_tail_rows] if len(g) > test_tail_rows else g)
        X = pd.concat(keep)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            X.to_parquet(cache_path)

    y = build_target(po.panel, direction=target["direction"],
                     threshold_pct=target["threshold_pct"],
                     horizon_days=target["horizon_days"],
                     max_drawdown=target.get("max_drawdown"))
    y = y.reindex(X.index)
    ok = y.notna().values
    return X[ok], y[ok].astype(int).values


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def diagnose(artifact_dir: Path, *, top_n: int = 30, importance_threshold: float = 0.01,
             out_dir: Path | None = None, do_pdp: bool = True, do_figs: bool = True,
             pdp_subsample: int = 4000) -> dict:
    artifact_dir = Path(artifact_dir)
    out_dir = Path(out_dir) if out_dir else artifact_dir / "diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

    cell = load_cell(artifact_dir)
    model, target = cell["model"], cell["target"]
    names = list(model.feature_names_)
    imp = dict(zip(names, model.get_feature_importance()))

    # split sizes (for in-sample test-tail)
    split = (cell["spec"].get("split") or {})
    test_tail = int(split.get("test_rows", 100))

    X, y = build_insample(target, test_tail_rows=test_tail,
                          cache_path=out_dir / "_insample_matrix.parquet")
    prevalence = float(y.mean())

    # ---- A. feature importance (numeric top-N) ----
    imp_sorted = sorted(((f, float(imp.get(f, 0.0))) for f in names), key=lambda kv: -kv[1])
    top_feats = [f for f, _ in imp_sorted[:top_n]]

    # ---- B. prevalence drift across segments ----
    seg_prev = {}
    for seg, df in cell["predictions"].items():
        seg_prev[seg] = float(df["y_true"].mean()) if df is not None and len(df) else None
    drift = prevalence_drift(seg_prev)

    # ---- overfit read (from iterations.jsonl iter0) ----
    # no_overfit is driven by the train/val gap ALONE (see assess_overfit).
    # early_stop_iteration + iteration_cap_hit are kept as informational fields,
    # NOT as overfit gates — early-stopping firing is healthy, not overfit.
    overfit = {"train_val_gap": None, "early_stop": None, "iteration_cap_hit": None,
               "no_overfit": None}
    if cell["iterations"]:
        it0 = cell["iterations"][0]
        gap = it0.get("train_val_gap")
        overfit = {"train_val_gap": gap, "early_stop": it0.get("early_stop_iteration"),
                   "iteration_cap_hit": it0.get("iteration_cap_hit"),
                   "no_overfit": assess_overfit(gap)}

    # ---- E. interaction involvement (compute once; used for advice) ----
    involve = interaction_involvement(model)
    inv_vals = sorted((v for v in involve.values() if v > 0), reverse=True)
    inv_high_thr = inv_vals[max(0, len(inv_vals) // 2) - 1] if inv_vals else float("inf")

    # ---- C + D. marginal monotonicity + model 1D-PDP, per top feature ----
    sub = X.sample(min(pdp_subsample, len(X)), random_state=42)[names].values
    feat_rows = []
    for f in top_feats:
        marg = spearman_monotonicity(X[f].values.astype(float), y)
        pdp_info = None
        if do_pdp:
            grid = np.quantile(X[f].dropna(), np.linspace(0.05, 0.95, 12))
            curve = pdp_1d(model, sub, names.index(f), grid)
            pdp_info = pdp_monotonicity(curve)
        feat_rows.append({
            "feature": f, "importance": float(imp.get(f, 0.0)),
            "marginal": marg, "model_pdp": pdp_info,
            "interaction_involvement": float(involve.get(f, 0.0)),
            "constraint_advice": constraint_advice(marg, pdp_info, involve.get(f, 0.0), inv_high_thr),
        })

    # ---- G. pruned-feature investigation ----
    kept = [f for f in names if imp.get(f, 0.0) >= importance_threshold]
    pruned = [f for f in names if imp.get(f, 0.0) < importance_threshold]
    Xkept = X[kept].fillna(X[kept].median()) if kept else pd.DataFrame(index=X.index)
    pruned_rows = []
    for f in pruned:
        marg = spearman_monotonicity(X[f].values.astype(float), y)
        maxc = 0.0
        if kept:
            kc = Xkept.corrwith(X[f], method="spearman").abs()
            maxc = float(kc.max()) if len(kc) else 0.0
        pruned_rows.append({"feature": f, "rho": marg["rho"],
                            "consistency": marg["consistency"], "maxcorr_kept": maxc})
    pdf = pd.DataFrame(pruned_rows)
    if len(pdf):
        real = pdf[(pdf["rho"].abs() >= 0.04) & (pdf["consistency"] >= 0.75)]
        redundant = real[real["maxcorr_kept"] >= 0.7]
        pruned_summary = {"n_pruned": len(pdf), "n_real_relationship": int(len(real)),
                          "n_redundant": int(len(redundant)),
                          "n_noise": int(len(pdf) - len(real))}
    else:
        pruned_summary = {"n_pruned": 0, "n_real_relationship": 0, "n_redundant": 0, "n_noise": 0}

    bundle = {
        "artifact_dir": str(artifact_dir),
        "cell": {k: target.get(k) for k in ("universe", "direction", "threshold_pct",
                                            "horizon_days", "max_drawdown")},
        "n_features_in_model": len(names),
        "insample_rows": int(len(X)), "insample_prevalence": prevalence,
        "overfit": overfit,
        "prevalence_by_segment": seg_prev, "prevalence_drift": drift,
        "top_features": feat_rows,
        "top_interaction_pairs": [list(t) for t in top_interaction_pairs(model, 15)],
        "interaction_high_threshold": float(inv_high_thr) if np.isfinite(inv_high_thr) else None,
        "kept_count": len(kept), "pruned_count": len(pruned),
        "pruned_summary": pruned_summary,
        "importance_threshold": importance_threshold,
    }

    if do_figs:
        _emit_corr_heatmap(X, top_feats, out_dir / "figs" / "corr_heatmap.png")

    (out_dir / "diagnose.json").write_text(json.dumps(bundle, indent=2, default=str))
    (out_dir / "diagnose_report.md").write_text(_render_report(bundle))
    return bundle


def _emit_corr_heatmap(X: pd.DataFrame, feats: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [f for f in feats if f in X.columns][:30]
    corr = X[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(8, len(feats) * 0.5), max(7, len(feats) * 0.45)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6); ax.set_yticklabels(feats, fontsize=6)
    ax.set_title("Spearman correlation — top features (in-sample)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _render_report(b: dict) -> str:
    c = b["cell"]
    L = [f"# gbdt-diagnose — {c['universe']} {c['direction']} +{c['threshold_pct']}% "
         f"/ {c['horizon_days']}d" + (f" / dd{c['max_drawdown']}" if c.get('max_drawdown') else ""),
         f"\nArtifact: `{b['artifact_dir']}`  |  model features: {b['n_features_in_model']}  |  "
         f"in-sample rows: {b['insample_rows']}  |  prevalence: {b['insample_prevalence']:.3f}\n"]

    # Tuning guidance (playbook-driven flags)
    L.append("## Tuning guidance (auto-flagged from the playbook)\n")
    of = b["overfit"]
    if of.get("no_overfit"):
        L.append(f"- **NO OVERFIT** (train/val gap {of['train_val_gap']} ≤ {_OVERFIT_GAP_THR}; "
                 f"early-stop@{of['early_stop']} is orthogonal) → **do NOT prune for "
                 f"regularization** (rule 1). FS will be neutral-to-harmful.")
    elif of.get("no_overfit") is False:
        L.append(f"- **OVERFIT signal** (train/val gap {of['train_val_gap']} > {_OVERFIT_GAP_THR}, "
                 f"val worse than train) → pruning / regularization (raise l2, drop depth) may help.")
    dr = b["prevalence_drift"]
    if dr.get("drift_flag") or dr.get("monotone_decline"):
        L.append(f"- **PREVALENCE DRIFT** across segments (spread {dr['spread']:.3f}"
                 f"{', monotone decline' if dr['monotone_decline'] else ''}) → calibration ceiling likely; "
                 f"the lever is recency / regime-conditional calibration, **out of the FS/HP loop** (rule 5). "
                 f"Per-segment: {b['prevalence_by_segment']}")
    ps = b["pruned_summary"]
    L.append(f"- Pruned features: {ps['n_pruned']} (<{b['importance_threshold']} imp); "
             f"{ps['n_real_relationship']} have a real monotone relationship, of which "
             f"{ps['n_redundant']} are redundant (collinear with a kept feature), {ps['n_noise']} are weak/noise. "
             f"→ importance≈0 usually means redundant, not unrelated (rule 2).")

    L.append("\n## Top features — importance, monotonicity, interaction, constraint advice\n")
    L.append("| feature | imp | marg ρ | marg cons | model-PDP monotone? | interaction | monotone-constraint? |")
    L.append("|---|---:|---:|---:|---|---:|---|")
    for r in b["top_features"]:
        m = r["marginal"]; p = r["model_pdp"]
        pdp_s = ("—" if p is None else ("yes" if p["monotone"] else f"NO (dip {p['worst_dip_frac']:.0%})"))
        L.append(f"| {r['feature']} | {r['importance']:.2f} | {m['rho']:+.3f} | {m['consistency']:.2f} | "
                 f"{pdp_s} | {r['interaction_involvement']:.2f} | {r['constraint_advice']} |")

    L.append("\n## Top pairwise interactions\n")
    L.append("| feature A | feature B | strength |")
    L.append("|---|---|---:|")
    for a, bb, s in b["top_interaction_pairs"]:
        L.append(f"| {a} | {bb} | {s:.2f} |")

    L.append("\n_Figures: `figs/corr_heatmap.png`. Full numerics: `diagnose.json`._")
    L.append("\nSee `.claude/memories/project-gbdt-tuning-playbook.md` for the rules referenced above.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.gbdt.diagnose")
    ap.add_argument("artifact_dir", type=Path, help="Path to a fitted cell artifact dir")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--importance-threshold", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-pdp", action="store_true", help="Skip the 1D-PDP audit (faster)")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args(argv)
    b = diagnose(a.artifact_dir, top_n=a.top_n, importance_threshold=a.importance_threshold,
                 out_dir=a.out, do_pdp=not a.no_pdp, do_figs=not a.no_figs)
    out = Path(a.out) if a.out else a.artifact_dir / "diagnose"
    print(f"[diagnose] wrote {out}/diagnose_report.md + diagnose.json")
    print(f"[diagnose] no_overfit={b['overfit'].get('no_overfit')} "
          f"prevalence_drift={b['prevalence_drift'].get('drift_flag')} "
          f"pruned_redundant={b['pruned_summary']['n_redundant']}/{b['pruned_summary']['n_pruned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
