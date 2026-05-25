"""Report renderer + figure emission for gbdt v1 artifacts.

After ``walk_forward_train`` returns, the orchestrator writes:
- ``iterations.jsonl`` (one bundle per line)
- ``metrics.json`` (headline)
- ``features.yaml`` / ``hp.yaml`` (best-checkpoint config)
- ``predictions/{train,val,eval,test}.csv``
- ``figs/*.png``
- ``report.md``

This module renders the figures + the markdown report. The orchestrator in
``gbdt.__main__`` (Stage 8 CLI atom) ties everything together.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gbdt.diagnostics import DiagnosticBundle


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def emit_figures(experiment_dir: Path, iterations: list[DiagnosticBundle],
                  predictions: dict[str, pd.DataFrame]) -> list[Path]:
    """Emit reliability + per-iteration learning curve + top importance +
    train-val gap history. Returns the list of written paths."""
    figs_dir = experiment_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ---- reliability diagram on eval (or val if eval missing) ----
    target_df = predictions.get("eval")
    if target_df is None or target_df.empty:
        target_df = predictions.get("val")
    if target_df is not None and not target_df.empty:
        p = target_df["p_calibrated"].values
        y = target_df["y_true"].values
        edges = np.linspace(0, 1, 11)
        bins = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys, cnts = [], [], []
        for b in range(10):
            mask = bins == b
            if mask.sum() == 0:
                continue
            xs.append(p[mask].mean())
            ys.append(y[mask].mean())
            cnts.append(int(mask.sum()))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax.plot(xs, ys, "o-", label="model")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("empirical positive frequency")
        ax.set_title("Reliability diagram (eval)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        path = figs_dir / "reliability_diagram.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # ---- learning curves per iteration ----
    for b in iterations:
        if not b.learning_curve:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, vals in b.learning_curve.items():
            ax.plot(vals, label=k, alpha=0.7)
        ax.set_xlabel("boosting round")
        ax.set_ylabel("metric")
        ax.set_title(f"Learning curve — iter {b.iter}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        p = figs_dir / f"learning_curve_iter_{b.iter}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # ---- final top-30 importance ----
    if iterations:
        last = iterations[-1]
        if last.importance_native:
            imp = (pd.Series(last.importance_native)
                    .sort_values(ascending=False).head(30))
            fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(imp))))
            imp.iloc[::-1].plot.barh(ax=ax)
            ax.set_title("Top 30 native feature importance (final iter)")
            ax.set_xlabel("importance")
            p = figs_dir / "feature_importance_final.png"
            fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close(fig)
            written.append(p)

    # ---- train vs val gap history ----
    if len(iterations) >= 1:
        iters = [b.iter for b in iterations]
        train = [b.train_brier for b in iterations]
        val = [b.val_brier for b in iterations]
        gap = [b.train_val_gap for b in iterations]
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(iters, train, "o-", label="train")
        axes[0].plot(iters, val, "o-", label="val")
        axes[0].set_ylabel("Brier")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(iters, gap, "o-")
        axes[1].set_ylabel("val - train")
        axes[1].set_xlabel("iteration")
        axes[1].grid(alpha=0.3)
        fig.suptitle("Train-val gap history")
        p = figs_dir / "train_val_gap_history.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_report(experiment_dir: Path) -> Path:
    """Read the artifact dir and write ``report.md``. Returns the path."""
    spec_path = experiment_dir / "spec.yaml"
    metrics_path = experiment_dir / "metrics.json"
    iters_path = experiment_dir / "iterations.jsonl"

    spec = yaml.safe_load(spec_path.read_text()) if spec_path.exists() else {}
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    iters = []
    if iters_path.exists():
        for line in iters_path.read_text().splitlines():
            if line.strip():
                iters.append(json.loads(line))

    lines: list[str] = []
    name = metrics.get("experiment_name", experiment_dir.name)
    lines.append(f"# gbdt experiment — {name}")
    lines.append("")

    # 1. Spec echo
    lines.append("## Spec")
    lines.append("")
    target = spec.get("target", {})
    lines.append(f"- universe: `{target.get('universe', '?')}`")
    lines.append(f"- direction: `{target.get('direction', '?')}`")
    lines.append(f"- threshold_pct: `{target.get('threshold_pct', '?')}`")
    lines.append(f"- horizon_days: `{target.get('horizon_days', '?')}`")
    if "max_drawdown" in target:
        lines.append(f"- max_drawdown: `{target['max_drawdown']}`")
    lines.append("")

    # 2. Data
    data = metrics.get("data", {})
    lines.append("## Data")
    lines.append("")
    lines.append(f"- tickers in universe: {data.get('n_tickers_in_universe')}")
    lines.append(f"- tickers used: {data.get('n_tickers_used')}")
    excl = data.get("tickers_excluded") or []
    if excl:
        lines.append(f"- tickers excluded: {', '.join(excl)}")
    for seg in ("train", "val", "eval", "test"):
        k = f"n_rows_{seg}"
        if k in data:
            lines.append(f"- {seg} rows: {data[k]}")
    if "positive_prevalence_train" in data:
        lines.append(f"- positive prevalence (train): {data['positive_prevalence_train']:.3f}")
    if "positive_prevalence_eval" in data:
        lines.append(f"- positive prevalence (eval): {data['positive_prevalence_eval']:.3f}")
    lines.append("")

    # 3. Iteration history
    lines.append("## Iteration history")
    lines.append("")
    lines.append("| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |")
    lines.append("|---|---|---|---|---|---|---|")
    for b in iters:
        rb = b.get("rationale", "").replace("|", "/")[:60]
        delta = b.get("delta_attribution", "")
        rb = rb if not delta else f"{rb} :: {delta}"[:80].replace("|", "/")
        signal = ""
        if "inner_stop" in delta:
            signal = delta.split("=")[-1]
        lines.append(
            f"| {b.get('iter')} | {b.get('n_features')} | "
            f"{b.get('train_brier'):.4f} | {b.get('val_brier'):.4f} | "
            f"{b.get('train_val_gap'):.4f} | {rb} | {signal} |"
        )
    lines.append("")

    # 4. Final checkpoint
    loop = metrics.get("loop", {})
    lines.append("## Final checkpoint")
    lines.append("")
    lines.append(f"- best iteration: {loop.get('best_iteration')}")
    lines.append(f"- iterations run: {loop.get('n_iterations_run')}")
    lines.append(f"- inner stop signal: `{loop.get('inner_stop_signal')}`")
    lines.append("")

    # 5. Calibration
    cal = metrics.get("calibration", {})
    lines.append("## Calibration")
    lines.append("")
    lines.append(f"- method requested: `{cal.get('method')}`")
    lines.append(f"- decision: `{cal.get('decision')}`")
    lines.append(f"- Spiegelhalter Z: {cal.get('spiegelhalter_z'):.3f}" if cal.get("spiegelhalter_z") is not None else "- Spiegelhalter Z: n/a")
    if cal.get("spiegelhalter_p") is not None:
        lines.append(f"- Spiegelhalter p: {cal['spiegelhalter_p']:.4f}")
    if (figs_dir := experiment_dir / "figs" / "reliability_diagram.png").exists():
        lines.append("")
        lines.append(f"![reliability](figs/{figs_dir.name})")
    lines.append("")

    # 6. Headline metrics on eval
    h_ev = metrics.get("headline_eval", {})
    h_te = metrics.get("headline_test", {})
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |")
    lines.append("|---|---|---|---|---|---|")
    for label, h in (("eval", h_ev), ("test", h_te)):
        if not h:
            continue
        lines.append(
            f"| {label} | {h.get('brier', float('nan')):.4f} | "
            f"{h.get('brier_baseline_baserate', float('nan')):.4f} | "
            f"{h.get('brier_improvement_vs_baseline', float('nan')):+.4f} | "
            f"{h.get('log_loss', float('nan')):.4f} | "
            f"{h.get('roc_auc', float('nan')):.4f} |"
        )
    lines.append("")

    # 7. Verdict
    lines.append("## Per-experiment verdict (algorithmic readout)")
    lines.append("")
    verdict = _algorithmic_verdict(metrics)
    lines.append(verdict)
    lines.append("")
    lines.append(
        "> NOTE: this verdict is generated from the metrics by a simple rule "
        "(see ``report._algorithmic_verdict``); it is NOT an automated "
        "pass/fail gate. The user reads the artifact and decides whether "
        "the cell ships."
    )
    lines.append("")

    out = experiment_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


def _algorithmic_verdict(metrics: dict) -> str:
    cal = metrics.get("calibration", {})
    h = metrics.get("headline_eval", {})
    z = cal.get("spiegelhalter_z")
    brier_imp = h.get("brier_improvement_vs_baseline")
    decision = cal.get("decision", "?")
    parts = []
    if z is None:
        parts.append("Calibration: missing.")
    elif abs(z) < 2.0:
        parts.append(f"Calibration: native-passable (|z|={abs(z):.2f}<2).")
    else:
        parts.append(f"Calibration: required isotonic (|z|={abs(z):.2f}); shipped as `{decision}`.")
    if brier_imp is None:
        parts.append("Brier vs base-rate: not computed.")
    elif brier_imp > 0:
        parts.append(f"Brier vs base-rate: +{brier_imp:.4f} (model beats baseline).")
    elif brier_imp == 0:
        parts.append("Brier vs base-rate: 0 (matches baseline).")
    else:
        parts.append(f"Brier vs base-rate: {brier_imp:.4f} (worse than baseline).")
    return " ".join(parts)


__all__ = ["emit_figures", "render_report"]
