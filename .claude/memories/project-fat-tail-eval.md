---
name: project-fat-tail-eval
description: Mandatory 8-anchor fat-tail eval set for every v4+ forecasting experiment; pinned in FAT_TAIL_EVAL.md
metadata:
  type: project
---

Every v4+ experiment that produces a forecast must report the fat-tail evaluation panel defined in [`docs/analog_mc/FAT_TAIL_EVAL.md`](../../docs/analog_mc/FAT_TAIL_EVAL.md) — 8 anchors (5 extreme-positive z₅₀, 3 extreme-negative), 60-day forecast vs realized, coverage table, per-anchor CRPS diff vs the v2.4 Cell-D-s30 baseline.

**Why:** Aggregate CRPS and PIT diagnostics passed for v2.4 (which is well-calibrated overall), but anchor-by-anchor inspection revealed the analog primitive systematically misses bear-bottom-rally regimes (2001-10, 2020-03 COVID, 2018-Q4, 2026 — realized +30-40% in 60 days, model expected sideways/down). Aggregate metrics average those misses away. The 8-anchor panel pins them down so a v4 experiment that improves aggregate but regresses on >2 fat-tail anchors gets flagged before promotion.

**How to apply:**
- For any new modeling change that produces forecasts, render the 8-anchor panel using `scripts/analog_mc/plot_forecast_from_date.py` and produce the coverage table per `FAT_TAIL_EVAL.md` §"v4 mandatory deliverable".
- Save figures under `docs/analog_mc/experiments/figs/<exp_id>_fat_tail/`.
- Anchor list is pinned at `results/analog_mc/data/fat_tail_eval_anchors.json`; regenerate via `scripts/analog_mc/select_fat_tail_anchors.py` only after a new canonical run shifts walk-forward fold boundaries.
- Compare against the v2.4 baseline coverage table embedded in `FAT_TAIL_EVAL.md`. The headline V4 question is whether B1 (Platzer local-linear) closes the bear-bottom-rally misses without regressing the well-calibrated bull-momentum anchors.
