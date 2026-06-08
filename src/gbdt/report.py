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
from gbdt import topk_diagnostics


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

    # 0. Runner warnings (issue #31 + issue #32). Surfaced at the top so
    # they can't be missed when scanning the report. Both blocks are
    # no-ops when their condition isn't tripped.
    _render_runner_warnings(lines, metrics)

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
    # FS+HP loop flavour (V1.1) — which callback drove the iterations.
    # ``default`` = the algorithmic prune+nudge fallback; ``agent_file_protocol``
    # = the agent-driven exit-and-resume loop (the agent read each iteration's
    # diagnose bundle and wrote the prune/HP decision). Read from the spec
    # snapshot so archived reports are self-describing about what tuned the run.
    loop_cfg = (spec.get("backend", {}) or {}).get("fs_hp_loop", {}) or {}
    callback_mode = loop_cfg.get("callback_mode", "default")
    lines.append(f"- fs_hp_loop callback_mode: `{callback_mode}`")
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
    # Effective sample size table (LdP §4.4 — sample uniqueness). When
    # the spec disables uniqueness weighting, ESS == row count and the
    # inflation ratio is 1.0.
    su = metrics.get("sample_uniqueness", {})
    per_fold = (su.get("effective_sample_size_per_fold") or {}).get("fold_0") or {}
    for seg in ("train", "val", "eval", "test"):
        k = f"n_rows_{seg}"
        n = data.get(k)
        ess_block = per_fold.get(seg) or {}
        sumw = ess_block.get("sum_weights")
        ratio = ess_block.get("overlap_inflation_ratio")
        if n is None:
            continue
        if sumw is not None and ratio is not None:
            lines.append(
                f"- {seg} rows: {n} (independent events ≈ {sumw:.1f}; "
                f"overlap-inflation {ratio:.2f}×)"
            )
        else:
            lines.append(f"- {seg} rows: {n}")
    if su:
        lines.append(
            f"- sample uniqueness weighting: "
            f"`{'on' if su.get('uniqueness_weighting') else 'off'}` "
            f"(horizon_days={su.get('horizon_days')})"
        )
    if "positive_prevalence_train" in data:
        lines.append(f"- positive prevalence (train): {data['positive_prevalence_train']:.3f}")
    if "positive_prevalence_eval" in data:
        lines.append(f"- positive prevalence (eval): {data['positive_prevalence_eval']:.3f}")
    lines.append("")

    # V1.4 — Segment windows (8 calendar dates). Always emitted, regardless
    # of mode: for date_aligned cells these are the universe-calendar
    # window; for trailing cells they are the calendar UNION across
    # tickers (MIN start, MAX end per segment). A backtester can use
    # ``test_end + 1`` as a clean out-of-sample start.
    sd = metrics.get("segment_dates") or {}
    split_mode = metrics.get("split_mode") or "(unknown)"
    split_anchor = metrics.get("split_train_start")
    if sd:
        lines.append("## Segment windows")
        lines.append("")
        lines.append(f"- split mode: `{split_mode}`")
        if split_anchor:
            lines.append(f"- train_start anchor: `{split_anchor}`")
        for seg in ("train", "val", "eval", "test"):
            block = sd.get(seg) or {}
            s, e = block.get("start"), block.get("end")
            if s and e:
                lines.append(f"- {seg}: `{s}` → `{e}`")
            else:
                lines.append(f"- {seg}: (empty)")
        lines.append("")

    # 3. Iteration history
    # The ``rationale``/``delta_attribution`` columns carry each iteration's
    # decision narrative. Under ``callback_mode: agent_file_protocol`` (V1.1)
    # the ``delta_attribution`` is the agent's per-iteration ``rationale`` from
    # ``loop/iter_<N>_decision.json`` (its lab-notebook entry: why those features
    # were pruned / those HPs changed); under ``default`` it is the algorithmic
    # fallback's one-line summary.
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
    lines.append(f"- fs_hp_loop callback_mode: `{callback_mode}`")
    # V1.4 P2 — surface which branch in ``fs_hp_loop.best_checkpoint`` chose
    # the best iter. The label distinguishes "L1 fired but fell back to eval
    # R-p@1-best" (V1.4 P1) from "classic L1 (gap+|z|) winner" — they share
    # the same iter index but the decision rationale is different. Older
    # ``metrics.json`` files predating P2 won't carry ``tiebreak_path``; we
    # silently skip the line in that case so resumed/legacy runs stay
    # readable.
    tb_path = loop.get("tiebreak_path")
    if tb_path:
        lines.append(f"- tie-break path: `{tb_path}` — {_tiebreak_path_description(tb_path)}")
    # ``agent_should_stop`` is the V1.1 agent-driven stop (the agent emitted
    # ``should_stop: true`` in a decision file); ``plateau`` / ``degradation`` /
    # ``cap`` are the runner's built-in inner-stop gates (fire under either mode).
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

    # 6b. Top-K / per-ticker / per-quarter / pred-range diagnostics
    _render_segment_diagnostics(lines, metrics)

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


def compute_segment_diagnostics(
    predictions: dict[str, pd.DataFrame],
    segments: tuple[str, ...] = ("eval", "test"),
    k_values: tuple[int, ...] = (1, 5, 10),
    per_ticker_k: int = 5,
    per_quarter_k: int = 5,
    low_separation_threshold: float = 0.05,
) -> dict[str, dict]:
    """Compute the four post-prediction diagnostics for each requested
    segment. Returns ``{segment_name: bundle}`` where ``bundle`` is the
    dict produced by ``topk_diagnostics.compute_all``. Segments missing
    from ``predictions`` (or empty) yield an empty-shaped bundle.

    This is the surface the runner consumes to populate ``metrics.json``.
    """
    out: dict[str, dict] = {}
    for seg in segments:
        df = predictions.get(seg) if predictions else None
        out[seg] = topk_diagnostics.compute_all(
            df,
            k_values=k_values,
            per_ticker_k=per_ticker_k,
            per_quarter_k=per_quarter_k,
            low_separation_threshold=low_separation_threshold,
        )
    return out


# ---------------------------------------------------------------------------
# Segment-diagnostic rendering (top-K / per-ticker / per-quarter / range)
# ---------------------------------------------------------------------------


def _render_runner_warnings(lines: list[str], metrics: dict) -> None:
    """Surface runner-level warnings at the top of ``report.md``.

    Covers:
    - Issue #31 — ``data.test_split_warning`` (horizon ate the test segment)
    - Issue #32 — ``loop.hp_search_active = false`` (sweep-mode FS-only loop)

    Both are silent no-ops when nothing is wrong; otherwise we emit an
    explicit "Warnings" section so the user can't miss the caveat.
    """
    data = metrics.get("data", {}) or {}
    loop = metrics.get("loop", {}) or {}
    test_warn = data.get("test_split_warning")
    hp_active = loop.get("hp_search_active")
    if not test_warn and hp_active is not False:
        return
    lines.append("## Warnings")
    lines.append("")
    if test_warn:
        lines.append(f"- **test_split**: {test_warn}")
    if hp_active is False:
        max_it = loop.get("max_iterations")
        thr = loop.get("hp_search_iter_threshold")
        lines.append(
            f"- **hp_search**: HP search disabled in sweep mode "
            f"(max_iter={max_it} < threshold={thr}); the FS+HP loop ran "
            f"feature-selection only — see issue #32."
        )
    lines.append("")


def _fmt(x, spec=".4f", na="n/a"):
    if x is None:
        return na
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return na


def _render_segment_diagnostics(lines: list[str], metrics: dict) -> None:
    """Append the four new diagnostic sections to ``lines``. No-ops the
    section if ``segment_diagnostics`` is absent or empty."""
    seg_diag = metrics.get("segment_diagnostics") or {}
    if not seg_diag:
        return

    # --- Top-K per-day + global ---
    lines.append("## Top-K precision (per-day + global)")
    lines.append("")
    lines.append(
        "Per-day: pick the top-K rows by ``p_calibrated`` each date, pool "
        "across days. ``P@k = sum_d(positives_in_top_k(d)) / "
        "sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on "
        "day ``d`` — the ``min(R(d), k)`` denominator is the achievable-"
        "positives count (mandatory; see "
        "``.claude/memories/project-r-precision-methodology.md``). "
        "``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with "
        "fewer than ``k`` positives. Global: top-K by score across the "
        "whole segment, denominator ``min(k, total_positives)``. "
        "``base_rate`` = unweighted segment positive prevalence (compare "
        "P@k to base_rate directly; lift omitted from the table by "
        "project reporting convention)."
    )
    lines.append("")
    for seg in ("eval", "test"):
        block = seg_diag.get(seg, {})
        tk = block.get("top_k_metrics") or {}
        n_rows = tk.get("n_rows", 0)
        base = tk.get("base_rate")
        lines.append(f"### {seg} — n_rows={n_rows}, base_rate={_fmt(base, '.4f')}")
        lines.append("")
        if not n_rows:
            lines.append("_segment empty — no picks._")
            lines.append("")
            continue
        per_day = tk.get("per_day", {})
        lines.append("Per-day:")
        lines.append("")
        lines.append(
            "| k | P@k | base_rate | n_picks | n_positives | n_denom | "
            "days_R<k / days_full_k / days_total |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for k, b in sorted(per_day.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {k} | {_fmt(b.get('p_at_k'))} | "
                f"{_fmt(base, '.4f')} | {b.get('n_picks_total')} | "
                f"{b.get('n_positives_in_picks')} | "
                f"{b.get('n_denom')} | "
                f"{b.get('n_days_R_lt_k')} / "
                f"{b.get('n_days_full_k')} / {b.get('n_days_total')} |"
            )
        lines.append("")
        glb = tk.get("global", {})
        lines.append("Global (top-K across entire segment):")
        lines.append("")
        lines.append("| k | P@k | base_rate | n_picks | n_positives | n_denom |")
        lines.append("|---|---|---|---|---|---|")
        for k, b in sorted(glb.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {k} | {_fmt(b.get('p_at_k'))} | "
                f"{_fmt(base, '.4f')} | {b.get('n_picks')} | "
                f"{b.get('n_positives_in_picks')} | "
                f"{b.get('n_denom')} |"
            )
        lines.append("")

    # --- Canonical R-Precision@K (macro-averaged) ---
    # Matches the registry of record at
    # ``results/gbdt/data/r_precision_at_k.csv``. Distinct from the Top-K
    # block's ``per_day.p_at_k`` (micro), which is preserved for
    # back-compat. See ``.claude/memories/project-r-precision-methodology.md``.
    lines.append("## R-Precision@K (canonical macro)")
    lines.append("")
    lines.append(
        "Per-day fixed K, **macro-averaged** across days with "
        "``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` "
        "where ``R_q`` = positives that day, ``r_q`` = positives caught "
        "in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable "
        "mergesort. This is the cross-cell headline (matches "
        "``results/gbdt/data/r_precision_at_k.csv``) — distinct from the "
        "Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated "
        "(both forms are mathematically valid; macro is canonical for "
        "cross-cell comparison). See "
        "``.claude/memories/project-r-precision-methodology.md``."
    )
    lines.append("")
    for seg in ("eval", "test"):
        block = (seg_diag.get(seg, {}) or {}).get("r_precision_at_k") or {}
        n_rows = block.get("n_rows", 0)
        Q = block.get("Q_days", 0)
        base = block.get("base_rate")
        lines.append(
            f"### {seg} — n_rows={n_rows}, Q_days={Q}, "
            f"base_rate={_fmt(base, '.4f')}"
        )
        lines.append("")
        by_k = block.get("by_k") or {}
        if not by_k or Q == 0:
            lines.append("_segment empty or no day with positives._")
            lines.append("")
            continue
        lines.append("| k | R-Precision@k | base_rate | Q_days |")
        lines.append("|---|---|---|---|")
        for k, b in sorted(by_k.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {k} | {_fmt(b.get('r_precision_at_k'))} | "
                f"{_fmt(base, '.4f')} | {b.get('n_qualifying_days')} |"
            )
        lines.append("")

    # --- Per-ticker hit-rate when picked ---
    lines.append("## Per-ticker hit-rate when picked (k=5)")
    lines.append("")
    lines.append(
        "Aggregates per-day top-5 picks by ticker. Top 10 most-picked + "
        "bottom 5 most-anti-predictive (when picked at least once) shown; "
        "the full table is in ``metrics.json::segment_diagnostics."
        "<seg>.per_ticker_hit_rate.rows``."
    )
    lines.append("")
    for seg in ("eval", "test"):
        block = (seg_diag.get(seg, {}) or {}).get("per_ticker_hit_rate") or {}
        rows = block.get("rows") or []
        lines.append(f"### {seg}")
        lines.append("")
        if not rows:
            lines.append("_no picks._")
            lines.append("")
            continue
        lines.append("Top-10 by n_picks:")
        lines.append("")
        lines.append("| ticker | n_picks | n_positives | hit_rate |")
        lines.append("|---|---|---|---|")
        for r in rows[:10]:
            lines.append(
                f"| {r['ticker']} | {r['n_picks']} | "
                f"{r['n_positives']} | {_fmt(r['hit_rate'])} |"
            )
        lines.append("")
        # Bottom-5 by hit_rate among tickers with the most picks — to
        # surface "model picks them often and they're systematically
        # wrong". Restrict to tickers with at least a few picks.
        eligible = [r for r in rows if r["n_picks"] >= 5]
        if eligible:
            worst = sorted(
                eligible, key=lambda r: (r["hit_rate"], -r["n_picks"], r["ticker"])
            )[:5]
            lines.append("Bottom-5 by hit_rate (n_picks ≥ 5):")
            lines.append("")
            lines.append("| ticker | n_picks | n_positives | hit_rate |")
            lines.append("|---|---|---|---|")
            for r in worst:
                lines.append(
                    f"| {r['ticker']} | {r['n_picks']} | "
                    f"{r['n_positives']} | {_fmt(r['hit_rate'])} |"
                )
            lines.append("")

    # --- Per-quarter stability of P@5 ---
    lines.append("## Per-quarter P@5 stability")
    lines.append("")
    lines.append(
        "P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide "
        "positive prevalence (constant across rows); regime-dependent "
        "collapse shows as a quarter where ``P@5`` falls toward "
        "``base_rate`` or below. ``lift`` omitted from the table by project "
        "reporting convention."
    )
    lines.append("")
    for seg in ("eval", "test"):
        block = (seg_diag.get(seg, {}) or {}).get("per_quarter_p_k") or {}
        rows = block.get("rows") or []
        lines.append(f"### {seg}")
        lines.append("")
        if not rows:
            lines.append("_no picks._")
            lines.append("")
            continue
        lines.append("| quarter | n_picks | n_positives | P@5 | base_rate |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['quarter']} | {r['n_picks']} | {r['n_positives']} | "
                f"{_fmt(r['p_at_k'])} | {_fmt(r['base_rate'])} |"
            )
        lines.append("")

    # --- Prediction range ---
    lines.append("## Prediction-range diagnostics")
    lines.append("")
    lines.append(
        "Distribution of ``p_calibrated`` per segment. "
        "``flag_low_separation = true`` when ``std < "
        "low_separation_threshold`` — a sign the model's predictions "
        "cluster so tightly that ranking is noise."
    )
    lines.append("")
    lines.append("| segment | n_rows | min | max | mean | std | flag_low_separation |")
    lines.append("|---|---|---|---|---|---|---|")
    for seg in ("eval", "test"):
        block = (seg_diag.get(seg, {}) or {}).get("prediction_range") or {}
        lines.append(
            f"| {seg} | {block.get('n_rows', 0)} | "
            f"{_fmt(block.get('min'))} | {_fmt(block.get('max'))} | "
            f"{_fmt(block.get('mean'))} | {_fmt(block.get('std'))} | "
            f"`{block.get('flag_low_separation', False)}` |"
        )
    lines.append("")


def _tiebreak_path_description(label: str) -> str:
    """Human-readable expansion of the V1.4 P2 tie-break path label.

    Maps the internal :data:`fs_hp_loop.TiebreakPath` codes to the sentence
    that appears in ``report.md``. Unknown labels (forward-compat) pass
    through verbatim so a future label added in code doesn't silently get
    swallowed by the report renderer.
    """
    descriptions = {
        "strict_val_brier": "Strict val_brier argmin (no tie-break entered)",
        "anti_auc_eval_rp1": (
            "Anti-AUC fallback: tie set picked by eval R-Precision@1 "
            "(V1.3 Option A)"
        ),
        "v14_val_flat_eval_rp1": (
            "Val_brier flat: tie set picked by eval R-Precision@1 "
            "(V1.4 P1)"
        ),
        "classic_l1": (
            "Classic L1 (train_val_gap + Spiegelhalter |z|)"
        ),
        "l1_fallthrough": (
            "L1 sort-keys collapsed (no L1 metrics present on tied set); "
            "strict val_brier argmin fallback (Bug #216)"
        ),
    }
    return descriptions.get(label, label)


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


__all__ = ["emit_figures", "render_report", "compute_segment_diagnostics"]
