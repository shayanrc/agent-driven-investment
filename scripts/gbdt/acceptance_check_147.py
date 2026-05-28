"""V1.1 Phase 6 — acceptance harness: automated agent-loop vs the `_147` answer key.

Plan: `docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md` § 0.4 + the Phase 6 row in
§ 10. The hand-driven loop documented in
`docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` is the **answer key**: this
script reads a *completed* automated `agent_file_protocol` run's on-disk
artifacts (`iterations.jsonl`, `metrics.json`, `features.yaml`) and asserts the
loop's end-state reproduces the `_147` findings, printing a PASS/FAIL table.

It does NOT run the loop. The full nifty50 H=25 agent-driven run is a multi-hour,
human-in-the-loop process (each iteration: agent reads `loop/iter_<N>_request.json`,
decides FS+HP, writes `loop/iter_<N>_decision.json`, relaunches `--resume`); see
`docs/gbdt/PHASE6_ACCEPTANCE_RUNBOOK.md`. This script is the *verdict* once that
run finishes.

The `_147` findings turned into checks (with tolerances justified from the memo)
-------------------------------------------------------------------------------
The pure comparison logic lives in :func:`evaluate_acceptance`, which takes the
three artifacts as already-parsed Python objects (so it is unit-testable on
synthetic inputs without a real run — see `tests/gbdt/test_acceptance_check_147.py`).
:func:`load_run` reads them from a run directory; :func:`main` wires the CLI.

Each :class:`Check` carries the `_147` source line(s) it encodes, the observed
value(s) extracted from the run, the tolerance, and a pass/fail verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # PyYAML is a project dep; features.yaml is a plain list of strings.
    import yaml
except Exception:  # pragma: no cover - yaml is always present under uv
    yaml = None


# ---------------------------------------------------------------------------
# Answer-key constants — extracted verbatim from `_147` (cited per check)
# ---------------------------------------------------------------------------

# `_147` "Unified conclusion" table + the iteration narratives:
#   iter 0 baseline 0.1642, iter 3 lr 0.02 -> 0.1641 (best), iter 5 monotone 0.1664
#   (worst). The whole explored band:
ANSWER_KEY = {
    # The HP-ceiling band: every depth{4,6,8} x lr{0.02,0.05} config landed in
    # val_brier [0.1641, 0.1661] (`_147` iter 3 belief update + the conclusion
    # table). Monotone iterations push the upper edge to 0.1664. We bound the
    # whole explored band [0.1641, 0.1664] and assert the *spread* is tiny.
    "ceiling_brier_lo": 0.1641,
    "ceiling_brier_hi": 0.1664,
    # The HP-only (non-monotone) ceiling band width: 0.1641..0.1661 = 0.0020.
    # "if val_brier stays in a tiny band across diverse configs, declare the
    # ceiling" (reusable lesson 2). We require the HP-config spread <= this.
    "hp_band_width_max": 0.0020,
    # depth inverted-U (iter 1/2 + iter 0): depth 4/6/8 -> 0.1661/0.1642/0.1652.
    "depth_optimal": 6,
    "depth_curve": {4: 0.1661, 6: 0.1642, 8: 0.1652},
    # Baseline (iter 0, all-279, depth 6, lr 0.05).
    "baseline_brier": 0.1642,
    # The best val_brier any lever achieved (iter 3 lr 0.02 = 0.1641, a +0.0001
    # "win" = noise). No config MEANINGFULLY beats baseline.
    "best_brier_seen": 0.1641,
    # The "meaningfully beats" threshold: a 0.0001 improvement is declared noise
    # in `_147` (iter 3). So "no config beats baseline by more than this" is the
    # ceiling claim.
    "meaningful_improvement": 0.0005,
    # Monotone-contraindicated: every monotone config (iters 5-9) is WORSE than
    # baseline. Best monotone = iter 8 (0.1654); worst = iter 5/9 (0.1664).
    "monotone_best_brier": 0.1654,
    "monotone_min_harm": 0.0010,  # iter 8: 0.1654 - 0.1642 = 0.0012; floor 0.0010.
    # No-overfit signal (iter 0): train/val gap -0.0048 (val below train). Lesson
    # 1: a gap <= +0.02 = no overfit -> FS will hurt, not help.
    "no_overfit_gap_max": 0.02,
    # Prevalence drift (the "fact that shaped everything"): train 0.280 -> val
    # 0.204 -> eval 0.138, monotone declining. The calibration ceiling.
    "prevalence_train": 0.280,
    "prevalence_eval": 0.138,
    "prevalence_min_decline": 0.05,  # train-eval drop >= 5pp (0.280-0.138=0.142).
    # Ranking robust: weighted R-precision ~2.1x on held-out segments throughout.
    "r_precision_lift_min": 1.5,
    # FS: even the gentlest cut (88 of 279) was ~= baseline, never beat it. The
    # final/best model keeps a substantial feature set (all-279 optimal on val;
    # 88-feat the leaner deployment artifact). A loop that pruned to a tiny set
    # contradicts `_147` -> require the final feature count not collapse.
    "final_features_min": 80,
}


# ---------------------------------------------------------------------------
# Check record + result container
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One acceptance assertion, with provenance + verdict."""

    name: str
    finding_147: str          # the `_147` claim this encodes
    passed: bool | None       # None = could-not-evaluate (missing data)
    observed: str             # what the run actually showed
    expected: str             # the answer-key target + tolerance
    note: str = ""            # extra context / why-skipped


@dataclass
class AcceptanceResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.passed is False)

    @property
    def n_skip(self) -> int:
        return sum(1 for c in self.checks if c.passed is None)

    @property
    def overall_pass(self) -> bool:
        """PASS iff no check FAILED. Skips (missing optional data) don't fail
        the run by themselves, but they're surfaced loudly in the table."""
        return self.n_fail == 0

    def add(self, **kw: Any) -> None:
        self.checks.append(Check(**kw))


# ---------------------------------------------------------------------------
# Small extraction helpers (tolerant of partial / missing fields)
# ---------------------------------------------------------------------------


def _f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def _hp_get(rec: dict, key: str) -> Any:
    """Read an HP off an iteration record: prefer `hp`, fall back to top-level."""
    hp = rec.get("hp") or {}
    if key in hp:
        return hp[key]
    return rec.get(key)


def _has_monotone(rec: dict) -> bool:
    """Did this iteration apply monotone_constraints?"""
    mc = _hp_get(rec, "monotone_constraints")
    if mc is None:
        return False
    if isinstance(mc, (list, tuple, dict)):
        return len(mc) > 0
    if isinstance(mc, str):
        return mc.strip() not in ("", "None", "{}", "[]")
    return bool(mc)


# ---------------------------------------------------------------------------
# The pure comparison logic (unit-testable on synthetic inputs)
# ---------------------------------------------------------------------------


def evaluate_acceptance(
    iterations: list[dict],
    metrics: dict,
    final_features: list[str],
    ak: dict | None = None,
) -> AcceptanceResult:
    """Compare a completed run's artifacts against the `_147` answer key.

    Parameters
    ----------
    iterations:
        Parsed `iterations.jsonl` rows (one dict per iteration). Each is expected
        to carry at least `iter`, `n_features`, `val_brier`, `train_val_gap`, and
        an `hp` dict (with `depth`, `learning_rate`, optionally
        `monotone_constraints`).
    metrics:
        Parsed `metrics.json`. Read for `loop` (best_iteration, inner_stop_signal),
        `data` (segment prevalences), and optionally `headline_*` R-precision /
        ranking fields if present.
    final_features:
        Parsed `features.yaml` (the final/best-checkpoint feature list).
    ak:
        Answer-key constants (defaults to :data:`ANSWER_KEY`); injectable for
        tests.

    Returns
    -------
    AcceptanceResult
        The PASS/FAIL/SKIP verdicts.
    """
    ak = ak or ANSWER_KEY
    res = AcceptanceResult()
    iters = [r for r in iterations if isinstance(r, dict)]

    val_briers = {r.get("iter"): _f(r.get("val_brier")) for r in iters}
    val_briers = {k: v for k, v in val_briers.items() if v is not None}

    # --- Check 1: HP ceiling — all explored val_briers inside the `_147` band.
    if val_briers:
        lo, hi = min(val_briers.values()), max(val_briers.values())
        inside = ak["ceiling_brier_lo"] <= lo and hi <= ak["ceiling_brier_hi"]
        res.add(
            name="hp_ceiling_band",
            finding_147=(
                "Across all explored configs val_brier stayed in a tiny band "
                f"~[{ak['ceiling_brier_lo']}, {ak['ceiling_brier_hi']}] "
                "(`_147` conclusion table / iter-3 belief update)."
            ),
            passed=inside,
            observed=f"explored val_brier range [{lo:.4f}, {hi:.4f}]",
            expected=(
                f"all within [{ak['ceiling_brier_lo']}, {ak['ceiling_brier_hi']}]"
            ),
        )
    else:
        res.add(
            name="hp_ceiling_band", finding_147="HP ceiling band",
            passed=None, observed="no val_brier in iterations.jsonl",
            expected=f"[{ak['ceiling_brier_lo']}, {ak['ceiling_brier_hi']}]",
            note="no iteration val_brier values found",
        )

    # --- Check 2: HP-only ceiling spread is tiny (the "declare the ceiling" test).
    # Use the non-monotone iterations only (monotone widens the band on purpose).
    hp_only = [val_briers[r["iter"]] for r in iters
               if r.get("iter") in val_briers and not _has_monotone(r)]
    if len(hp_only) >= 2:
        spread = max(hp_only) - min(hp_only)
        ok = spread <= ak["hp_band_width_max"] + 1e-9
        res.add(
            name="hp_ceiling_spread",
            finding_147=(
                "HP-only configs (depth x lr, no monotone) span a band "
                f"<= {ak['hp_band_width_max']:.4f} wide => declare the HP ceiling "
                "(reusable lesson 2)."
            ),
            passed=ok,
            observed=f"HP-only val_brier spread {spread:.4f} over {len(hp_only)} iters",
            expected=f"<= {ak['hp_band_width_max']:.4f}",
        )
    else:
        res.add(
            name="hp_ceiling_spread", finding_147="HP-only ceiling spread",
            passed=None,
            observed=f"only {len(hp_only)} non-monotone iters with val_brier",
            expected=f"spread <= {ak['hp_band_width_max']:.4f}",
            note="need >= 2 non-monotone iterations to measure the HP-only spread",
        )

    # --- Check 3: depth inverted-U with depth 6 optimal.
    # Group HP-only iterations by depth, take the best (lowest) val_brier per depth.
    by_depth: dict[int, float] = {}
    for r in iters:
        if r.get("iter") not in val_briers or _has_monotone(r):
            continue
        d = _hp_get(r, "depth")
        try:
            d = int(d)
        except (TypeError, ValueError):
            continue
        vb = val_briers[r["iter"]]
        if d not in by_depth or vb < by_depth[d]:
            by_depth[d] = vb
    if by_depth:
        best_depth = min(by_depth, key=lambda d: by_depth[d])
        # If multiple depths explored, require the optimum be depth_optimal AND
        # the curve be an inverted-U (6 below both 4 and 8 when present).
        ok = best_depth == ak["depth_optimal"]
        curve_note = ""
        if {4, 6, 8}.issubset(by_depth):
            inv_u = by_depth[6] <= by_depth[4] and by_depth[6] <= by_depth[8]
            ok = ok and inv_u
            curve_note = (
                f" curve d4/d6/d8={by_depth[4]:.4f}/{by_depth[6]:.4f}/"
                f"{by_depth[8]:.4f}"
            )
        res.add(
            name="depth_optimal",
            finding_147=(
                "Clean inverted-U on depth: depth 4/6/8 -> 0.1661/0.1642/0.1652; "
                "depth 6 is the sweet spot (`_147` iter 2 belief update)."
            ),
            passed=ok,
            observed=(
                f"best depth = {best_depth} (val_brier {by_depth[best_depth]:.4f});"
                f" depths explored {sorted(by_depth)}" + curve_note
            ),
            expected=f"depth {ak['depth_optimal']} optimal (inverted-U if 4/6/8 all seen)",
            note=("" if len(by_depth) > 1
                  else "only one depth explored — cannot confirm inverted-U shape"),
        )
    else:
        res.add(
            name="depth_optimal", finding_147="depth 6 optimal",
            passed=None, observed="no depth recorded on HP-only iterations",
            expected=f"depth {ak['depth_optimal']}",
            note="no usable depth values in iterations.jsonl",
        )

    # --- Check 4: no HP/FS config meaningfully beats baseline (the ceiling claim).
    baseline = val_briers.get(0)
    if baseline is not None and len(val_briers) >= 2:
        best_seen = min(val_briers.values())
        improvement = baseline - best_seen
        ok = improvement <= ak["meaningful_improvement"] + 1e-9
        res.add(
            name="no_meaningful_improvement",
            finding_147=(
                "No HP/FS config MEANINGFULLY beats baseline: the best win was "
                "+0.0001 (noise) at lr 0.02 (`_147` iter 3 belief update)."
            ),
            passed=ok,
            observed=(
                f"baseline {baseline:.4f}, best-seen {best_seen:.4f}, "
                f"improvement {improvement:+.4f}"
            ),
            expected=f"improvement over baseline <= {ak['meaningful_improvement']:.4f}",
        )
    else:
        res.add(
            name="no_meaningful_improvement", finding_147="HP ceiling (no real win)",
            passed=None,
            observed=f"baseline={baseline}, n_iters_with_brier={len(val_briers)}",
            expected=f"improvement <= {ak['meaningful_improvement']:.4f}",
            note="need iter-0 baseline + >= 2 iterations to compare",
        )

    # --- Check 5: monotone constraints contraindicated (every monotone > baseline).
    mono = [(r.get("iter"), val_briers[r["iter"]]) for r in iters
            if r.get("iter") in val_briers and _has_monotone(r)]
    if mono and baseline is not None:
        best_mono = min(v for _, v in mono)
        worse_than_baseline = all(v > baseline for _, v in mono)
        min_harm_ok = (best_mono - baseline) >= ak["monotone_min_harm"] - 1e-9
        ok = worse_than_baseline and min_harm_ok
        res.add(
            name="monotone_contraindicated",
            finding_147=(
                "Monotone constraints contraindicated: EVERY monotone config "
                "(iters 5-9) is worse than baseline; best monotone 0.1654 "
                "(+0.0012). No safe subset (`_147` iters 5-9 + lesson 3)."
            ),
            passed=ok,
            observed=(
                f"{len(mono)} monotone iters; best monotone {best_mono:.4f} "
                f"(+{best_mono - baseline:.4f} vs baseline {baseline:.4f}); "
                f"all worse than baseline = {worse_than_baseline}"
            ),
            expected=(
                "all monotone > baseline AND best monotone harm >= "
                f"{ak['monotone_min_harm']:.4f}"
            ),
        )
    else:
        res.add(
            name="monotone_contraindicated", finding_147="monotone contraindicated",
            passed=None,
            observed=(f"{len(mono)} monotone iters explored"
                      if mono else "no monotone-constraint iterations explored"),
            expected="all monotone configs > baseline",
            note=("the automated loop never tried a monotone constraint — the "
                  "agent must explore at least one to verify it's contraindicated"),
        )

    # --- Check 6: no overfit (negative / small train/val gap) at baseline.
    gap0 = _f((iters[0].get("train_val_gap") if iters else None))
    if gap0 is None:
        gap0 = _f(metrics.get("data", {}).get("train_val_gap"))
    if gap0 is not None:
        ok = gap0 <= ak["no_overfit_gap_max"]
        res.add(
            name="no_overfit_baseline",
            finding_147=(
                "Iter-0 train/val gap -0.0048 (val below train) => NO overfit; "
                "FS will hurt not help (`_147` iter 0 + lesson 1)."
            ),
            passed=ok,
            observed=f"iter-0 train/val gap {gap0:+.4f}",
            expected=f"<= {ak['no_overfit_gap_max']} (no-overfit threshold)",
        )
    else:
        res.add(
            name="no_overfit_baseline", finding_147="no overfit at baseline",
            passed=None, observed="no train_val_gap on iter 0",
            expected=f"<= {ak['no_overfit_gap_max']}",
            note="train_val_gap missing from iter-0 record",
        )

    # --- Check 7: prevalence-drift ceiling (declining train -> eval prevalence).
    data = metrics.get("data", {}) if isinstance(metrics, dict) else {}
    prev_train = _f(data.get("positive_prevalence_train"))
    prev_eval = _f(data.get("positive_prevalence_eval"))
    if prev_train is not None and prev_eval is not None:
        decline = prev_train - prev_eval
        ok = decline >= ak["prevalence_min_decline"]
        res.add(
            name="prevalence_drift_ceiling",
            finding_147=(
                "Prevalence non-stationary + declining (train 0.280 -> eval "
                "0.138): the calibration ceiling no FS/HP lever can touch "
                "(`_147` Setup + Unified conclusion)."
            ),
            passed=ok,
            observed=(
                f"train prevalence {prev_train:.3f} -> eval {prev_eval:.3f} "
                f"(decline {decline:+.3f})"
            ),
            expected=f"train-eval decline >= {ak['prevalence_min_decline']:.3f}",
        )
    else:
        res.add(
            name="prevalence_drift_ceiling", finding_147="prevalence drift ceiling",
            passed=None,
            observed=f"train_prev={prev_train}, eval_prev={prev_eval}",
            expected=f"decline >= {ak['prevalence_min_decline']:.3f}",
            note="segment prevalences missing from metrics.json::data",
        )

    # --- Check 8: ranking robust (weighted R-precision lift well above 1.0).
    # `metrics.json` does not always carry R-precision; read it where present,
    # else SKIP with a pointer to scripts/gbdt/compute_r_precision.py.
    rprec_lift = _extract_rprecision_lift(metrics)
    if rprec_lift is not None:
        ok = rprec_lift >= ak["r_precision_lift_min"]
        res.add(
            name="ranking_robust",
            finding_147=(
                "Ranking strong + robust throughout: weighted R-precision ~2.1x "
                "base rate on held-out segments (`_147` iter 0 + Unified "
                "conclusion)."
            ),
            passed=ok,
            observed=f"weighted R-precision lift {rprec_lift:.2f}x",
            expected=f">= {ak['r_precision_lift_min']:.1f}x base rate",
        )
    else:
        res.add(
            name="ranking_robust", finding_147="ranking robust (R-precision ~2.1x)",
            passed=None,
            observed="weighted R-precision not present in metrics.json",
            expected=f">= {ak['r_precision_lift_min']:.1f}x base rate",
            note=("run scripts/gbdt/compute_r_precision.py on the run's "
                  "predictions/ to compute weighted R-precision, then re-check"),
        )

    # --- Check 9: final feature set did not collapse (FS neutral, not a hard cut).
    nfeat = len(final_features or [])
    if nfeat > 0:
        ok = nfeat >= ak["final_features_min"]
        res.add(
            name="final_features_not_collapsed",
            finding_147=(
                "FS is neutral-to-harmful (no overfit): the best model keeps a "
                "substantial feature set — all-279 optimal on val, 88-feat the "
                "leaner deployment artifact; never a tiny hard cut (`_147` iter "
                "4 + investigation)."
            ),
            passed=ok,
            observed=f"final feature set has {nfeat} features",
            expected=f">= {ak['final_features_min']} features (no aggressive prune)",
        )
    else:
        res.add(
            name="final_features_not_collapsed", finding_147="FS did not collapse",
            passed=None, observed="features.yaml empty / missing",
            expected=f">= {ak['final_features_min']} features",
            note="features.yaml not readable",
        )

    return res


def _extract_rprecision_lift(metrics: dict) -> float | None:
    """Pull a weighted R-precision lift out of metrics.json if present.

    The runner's standard metrics.json does not always carry R-precision; the
    diagnostics layer / compute_r_precision.py CLI does. We probe a few plausible
    locations (so a run that DOES carry it is checked) and return None otherwise
    (=> SKIP with a pointer to the CLI).
    """
    if not isinstance(metrics, dict):
        return None
    for seg in ("headline_test", "headline_eval", "r_precision", "ranking"):
        blk = metrics.get(seg)
        if not isinstance(blk, dict):
            continue
        rp = blk.get("r_precision") if isinstance(blk.get("r_precision"), dict) else blk
        wtd = _f(rp.get("weighted") if isinstance(rp, dict) else None)
        base = _f(rp.get("base_rate_weighted") if isinstance(rp, dict) else None)
        lift = _f(rp.get("lift") if isinstance(rp, dict) else None)
        if lift is not None:
            return lift
        if wtd is not None and base not in (None, 0.0):
            return wtd / base
    return None


# ---------------------------------------------------------------------------
# Artifact loading + CLI
# ---------------------------------------------------------------------------


def load_run(run_dir: str | Path) -> tuple[list[dict], dict, list[str]]:
    """Read (iterations, metrics, features) from a completed run directory."""
    run_dir = Path(run_dir)
    it_path = run_dir / "iterations.jsonl"
    m_path = run_dir / "metrics.json"
    f_path = run_dir / "features.yaml"

    iterations: list[dict] = []
    if it_path.exists():
        for line in it_path.read_text().splitlines():
            line = line.strip()
            if line:
                iterations.append(json.loads(line))
    metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
    features: list[str] = []
    if f_path.exists():
        if yaml is not None:
            loaded = yaml.safe_load(f_path.read_text()) or []
            features = list(loaded) if isinstance(loaded, list) else []
        else:  # pragma: no cover - fallback parse for a plain "- name" list
            features = [
                ln.strip()[2:].strip()
                for ln in f_path.read_text().splitlines()
                if ln.strip().startswith("- ")
            ]
    return iterations, metrics, features


def format_table(result: AcceptanceResult) -> str:
    """Render the PASS/FAIL/SKIP table + a one-line overall verdict."""
    sym = {True: "PASS", False: "FAIL", None: "SKIP"}
    rows = [
        "ACCEPTANCE CHECK — automated nifty50 H=25 agent-loop vs `_147`",
        "=" * 78,
    ]
    for c in result.checks:
        rows.append(f"[{sym[c.passed]}] {c.name}")
        rows.append(f"       observed : {c.observed}")
        rows.append(f"       expected : {c.expected}")
        if c.note:
            rows.append(f"       note     : {c.note}")
    rows.append("-" * 78)
    verdict = "PASS" if result.overall_pass else "FAIL"
    rows.append(
        f"OVERALL: {verdict}  "
        f"({result.n_pass} pass, {result.n_fail} fail, {result.n_skip} skip)"
    )
    if result.n_skip:
        rows.append(
            "  NOTE: SKIPs are checks whose data wasn't present in the artifacts "
            "(e.g. no monotone iteration explored, or R-precision not in "
            "metrics.json). They do NOT fail the run, but a full acceptance "
            "should resolve them — see each SKIP note."
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acceptance_check_147",
        description=(
            "Assert a completed automated agent-loop run on nifty50 H=25 "
            "reproduces the `_147` hand-driven findings (V1.1 Phase 6)."
        ),
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Path to the completed run's artifact dir "
            "(default: results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance)."
        ),
        nargs="?",
        default=Path(
            "results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance"
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the result as JSON instead of the text table.",
    )
    args = parser.parse_args(argv)

    if not Path(args.run_dir).exists():
        print(
            f"[acceptance] run dir not found: {args.run_dir}\n"
            "Run the full agent-driven loop first (see "
            "docs/gbdt/PHASE6_ACCEPTANCE_RUNBOOK.md), then point this script at "
            "the finished run's artifact dir.",
            file=sys.stderr,
        )
        return 2

    iterations, metrics, features = load_run(args.run_dir)
    result = evaluate_acceptance(iterations, metrics, features)

    if args.json:
        print(json.dumps(
            {
                "overall_pass": result.overall_pass,
                "n_pass": result.n_pass,
                "n_fail": result.n_fail,
                "n_skip": result.n_skip,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "observed": c.observed,
                        "expected": c.expected,
                        "finding_147": c.finding_147,
                        "note": c.note,
                    }
                    for c in result.checks
                ],
            },
            indent=2,
        ))
    else:
        print(format_table(result))

    return 0 if result.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
