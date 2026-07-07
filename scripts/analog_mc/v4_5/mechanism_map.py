"""V4.5.5 — Cross-experiment mechanism map.

Synthesizes V4.5.1–4 + 6/7 into a single classifier matrix:
   anchor × model → primary failure mechanism (M1–M5) + addressable-by tags.

Mechanisms:
   M1: Mode-1 over-concentration (top-1 prob ≥ 0.4)
   M2: Mode-2 bimodal mis-match (top-2 ≥ 0.4 AND top-2 forwards opposite sign to realized)
   M3: Path-construction dispersion deficit (A2 cum-σ-growth < 0.7 × v24's)
   M4: Tail under-selection (A2 lift < 1 AND v24 lift < 1)
   M5: B1 drift over-correction (B1 horizon drift > 10% magnitude)
   M0: No primary mechanism (within-noise or trivial)

V5 candidate tags:
   V5.A.1  Tikhonov-mixed distance (addresses M1 partial, M3 partial)
   V5.A.2  Path-level ensemble (addresses M1, M2, M3 partial)
   V5.A.3  Conditional corrwindow (addresses M3 directly)
   V5.B    Feature augmentation (drawdown depth) (addresses M4)
   V5.C    Delay-coordinate distance (addresses M4 alternative)
   V5.D    B1 shrinkage (addresses M5)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def load(p: Path) -> dict | list:
    return json.loads((REPO / p).read_text())


def main() -> None:
    autopsy = load(Path("results/analog_mc/data/v4_5_2_analog_autopsy.json"))
    tail = load(Path("results/analog_mc/data/v4_5_7_tail_selection_scan.json"))
    path = load(Path("results/analog_mc/data/v4_5_6_path_construction.json"))
    b1 = load(Path("results/analog_mc/data/v4_5_3_b1_beta_autopsy.json"))
    diff_a2 = load(Path("results/analog_mc/data/fat_tail_a2_corrwindow_L100_diff.json"))
    diff_b1 = load(Path("results/analog_mc/data/fat_tail_b1_local_linear_diff.json"))

    # Build per-anchor records keyed by date.
    by_date: dict[str, dict] = {}
    for a in autopsy["anchors"]:
        by_date.setdefault(a["anchor_date"], {})["autopsy"] = a
    for a in tail["anchors"]:
        by_date.setdefault(a["anchor_date"], {})["tail"] = a
    for a in path["anchors"]:
        by_date.setdefault(a["anchor_date"], {})["path"] = a
    for a in b1["anchors"]:
        by_date.setdefault(a["anchor_date"], {})["b1"] = a
    for a in diff_a2["per_anchor"]:
        by_date.setdefault(a["anchor_date"], {})["diff_a2"] = a
    for a in diff_b1["per_anchor"]:
        by_date.setdefault(a["anchor_date"], {})["diff_b1"] = a

    # Classification.
    rows = []
    for date in sorted(by_date.keys()):
        rec = by_date[date]
        au = rec.get("autopsy", {})
        tl = rec.get("tail", {})
        pa = rec.get("path", {})
        b1r = rec.get("b1", {})
        d_a2 = rec.get("diff_a2", {})
        d_b1 = rec.get("diff_b1", {})

        if not au:
            continue

        realized = au["realized_60d_return_pct"]
        a2_d_rel = d_a2.get("delta_mean_crps_rel", 0.0)
        b1_d_rel = d_b1.get("delta_mean_crps_rel", 0.0)
        a2_top1 = au["a2"]["top_probs"][0] if au["a2"].get("top_probs") else 0.0
        a2_top2 = sum(au["a2"]["top_probs"][:2]) if au["a2"].get("top_probs") else 0.0
        a2_top2_fwds = au["a2"]["top_forward_pct"][:2] if au["a2"].get("top_forward_pct") else []
        a2_wfwd = au["a2"]["weighted_pct_forward"]
        v24_wfwd = au["v24"]["weighted_pct_forward"]

        # Mechanism checks (per model).
        # M1 A2
        m1_a2 = a2_top1 >= 0.4
        # M2 A2: top-2 ≥ 0.4 AND both forwards same-sign opposite to realized.
        m2_a2 = (
            a2_top2 >= 0.4
            and len(a2_top2_fwds) == 2
            and (a2_top2_fwds[0] * a2_top2_fwds[1] > 0)
            and (a2_top2_fwds[0] * realized < 0)
        )
        # M3: cum-σ-growth A2 < 0.7 × v24's. Only for anchors we have path data for.
        m3_a2 = False
        if pa:
            ratio = (pa.get("a2_cum_std_growth_actual") or 0) / (pa.get("v24_cum_std_growth_actual") or 1e-9)
            m3_a2 = ratio < 0.7
        # M4: both v24 lift < 1 AND A2 lift < 1.
        v24_lift = tl.get("lift_v24", float("nan")) if tl else float("nan")
        a2_lift = tl.get("lift_a2", float("nan")) if tl else float("nan")
        try:
            m4 = (v24_lift < 1.0) and (a2_lift < 1.0)
        except TypeError:
            m4 = False
        # M5: B1 horizon drift > 10% magnitude.
        m5_b1 = False
        b1_drift_pct = None
        if b1r:
            b1_drift_pct = b1r.get("original_b1", {}).get("horizon_drift_pct")
            if b1_drift_pct is not None:
                m5_b1 = abs(b1_drift_pct) > 10.0

        # Aggregate mechanism flags.
        a2_mechanisms = []
        if m1_a2: a2_mechanisms.append("M1")
        if m2_a2: a2_mechanisms.append("M2")
        if m3_a2: a2_mechanisms.append("M3")
        if m4: a2_mechanisms.append("M4")

        b1_mechanisms = []
        if m5_b1: b1_mechanisms.append("M5")
        if m4: b1_mechanisms.append("M4")  # tail under-selection affects v24 base too

        # V5 candidate coverage.
        v5_candidates_for_a2 = []
        if m1_a2 or m2_a2:
            v5_candidates_for_a2.extend(["V5.A.2", "V5.A.1"])
        if m3_a2:
            v5_candidates_for_a2.append("V5.A.3")
        if m4:
            v5_candidates_for_a2.extend(["V5.B", "V5.C"])
        v5_candidates_for_b1 = []
        if m5_b1:
            v5_candidates_for_b1.append("V5.D")
        if m4:
            v5_candidates_for_b1.extend(["V5.B", "V5.C"])

        rows.append({
            "anchor_date": date,
            "realized_60d_return_pct": realized,
            "a2_delta_crps_rel_pct": a2_d_rel * 100.0,
            "b1_delta_crps_rel_pct": b1_d_rel * 100.0,
            "a2_top1": a2_top1,
            "a2_top2": a2_top2,
            "a2_weighted_fwd_pct": a2_wfwd,
            "v24_weighted_fwd_pct": v24_wfwd,
            "v24_tail_lift": v24_lift,
            "a2_tail_lift": a2_lift,
            "a2_cum_sigma_ratio_vs_v24": (pa.get("a2_cum_std_growth_actual") / pa.get("v24_cum_std_growth_actual"))
                if pa and pa.get("v24_cum_std_growth_actual") else None,
            "b1_horizon_drift_pct": b1_drift_pct,
            "mechanisms_a2": a2_mechanisms,
            "mechanisms_b1": b1_mechanisms,
            "v5_candidates_a2": v5_candidates_for_a2,
            "v5_candidates_b1": v5_candidates_for_b1,
        })

    # Print matrix.
    print(f"\n{'anchor':<14} {'real%':>7} {'A2.Δ%':>7} {'B1.Δ%':>7} {'a2_top1':>8} {'a2_lift':>8} "
          f"{'v24_lift':>9} {'σ-ratio':>8} {'b1_drift%':>9} {'A2 mech':<14} {'B1 mech':<10}")
    print("-" * 130)
    for r in rows:
        sr = r["a2_cum_sigma_ratio_vs_v24"]
        sr_s = f"{sr:>.2f}" if sr is not None else "  -"
        bd = r["b1_horizon_drift_pct"]
        bd_s = f"{bd:>+.2f}" if bd is not None else "  -"
        v24l = f"{r['v24_tail_lift']:>9.2f}" if isinstance(r["v24_tail_lift"], (int, float)) and r["v24_tail_lift"] == r["v24_tail_lift"] else "        -"
        a2l = f"{r['a2_tail_lift']:>8.2f}" if isinstance(r["a2_tail_lift"], (int, float)) and r["a2_tail_lift"] == r["a2_tail_lift"] else "       -"
        print(f"{r['anchor_date']:<14} {r['realized_60d_return_pct']:>+7.1f} "
              f"{r['a2_delta_crps_rel_pct']:>+7.1f} {r['b1_delta_crps_rel_pct']:>+7.1f} "
              f"{r['a2_top1']:>8.3f} {a2l} {v24l} {sr_s:>8} {bd_s:>9} "
              f"{','.join(r['mechanisms_a2']):<14} {','.join(r['mechanisms_b1']):<10}")

    # Per-candidate coverage summary.
    print("\n=== V5 candidate coverage ===")
    candidates = ["V5.A.1", "V5.A.2", "V5.A.3", "V5.B", "V5.C", "V5.D"]
    a2_regressions = [r for r in rows if r["a2_delta_crps_rel_pct"] > 5]
    print(f"\nA2.1 regressions ({len(a2_regressions)} anchors):")
    for c in candidates:
        covered = [r["anchor_date"] for r in a2_regressions if c in r["v5_candidates_a2"]]
        print(f"  {c}: {len(covered)} anchors → {covered}")
    a2_uncovered = [r["anchor_date"] for r in a2_regressions if not r["v5_candidates_a2"]]
    print(f"  UNCOVERED A2.1 regressions: {a2_uncovered}")

    b1_regressions = [r for r in rows if r["b1_delta_crps_rel_pct"] > 5]
    print(f"\nB1 regressions ({len(b1_regressions)} anchors):")
    for c in candidates:
        covered = [r["anchor_date"] for r in b1_regressions if c in r["v5_candidates_b1"]]
        print(f"  {c}: {len(covered)} anchors → {covered}")
    b1_uncovered = [r["anchor_date"] for r in b1_regressions if not r["v5_candidates_b1"]]
    print(f"  UNCOVERED B1 regressions: {b1_uncovered}")

    OUT = REPO / "results/analog_mc/data/v4_5_5_mechanism_map.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": {
            "description": "V4.5.5 mechanism map — per-anchor classification using "
                          "thresholds calibrated from V4.5.2/3/4/6/7.",
            "mechanisms": {
                "M1": "Mode-1 over-concentration (A2.1 top-1 prob ≥ 0.4)",
                "M2": "Mode-2 bimodal mis-match (top-2 ≥ 0.4, forwards opposite-sign to realized)",
                "M3": "Path-construction dispersion deficit (A2 cum-σ-growth < 0.7× v24's)",
                "M4": "Tail under-selection (BOTH v24 lift < 1 AND A2 lift < 1)",
                "M5": "B1 drift over-correction (|horizon drift| > 10%)",
            },
            "v5_candidates": {
                "V5.A.1": "Tikhonov-mixed distance d = (1-α)·d_eu + α·d_cw",
                "V5.A.2": "Path-level ensemble (half-half v24 + A2.1 forecasts)",
                "V5.A.3": "Conditional corrwindow re-matching",
                "V5.B": "Feature augmentation (drawdown depth)",
                "V5.C": "Delay-coordinate (Takens) distance",
                "V5.D": "B1 with shrinkage parameter",
            },
        },
        "anchors": rows,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
