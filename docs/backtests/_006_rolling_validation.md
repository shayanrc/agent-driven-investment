# _006: rolling multi-window validation of rank/equal (V1.2 gate)

## TL;DR

`_005` showed rank/equal beating each cell's index on its **single** OOS window
(ndx40 +61.5%, sp500_20 +58.1%, …). This is the gate that asks whether that
survives **multiple windows** — and the answer, for the cells we can rigorously
test, is **no consistent edge**. Two hard limits shaped the result:

1. **Only the cell-5 family can be rolled.** A cell needs OOS length ≫ horizon to
   yield multiple windows; the published test windows are only 1–3× horizon. To
   extend the OOS I score the trained model forward by inference — but that path
   is only **faithful** (self-check ≤ 1e-4 vs published test.csv) for the plain
   cell-5 models. **ndx40 (`agentloop_mix`) FAILED the self-check** (max_abs_diff
   3.3e-2 — a V1.3 Option-B *combined-feature* cell whose composed overlay
   `build_feature_matrix` doesn't reproduce), and sp500/russell1000 need a
   per-universe cache refresh. So the two most exciting `_005` results
   (ndx40, sp500_20) **cannot be validated yet**.
2. **On the cell-5 family that we CAN roll, there is no consistent edge vs NDX.**

| Model (full OOS) | Strat total | NDX | Strat maxDD | 50-day windows: % beat NDX | median excess | p25 / p75 |
|---|---|---|---|---|---|---|
| cell5_revalreg (plain loop) | +36.5% | +48.8% | −8.9% | **46%** (52 win) | **−0.6%** | −7.6% / +5.5% |
| b_acceptance (Option B) | +39.6% | +37.5% | −8.8% | **38%** (42 win) | **−4.4%** | −8.3% / +6.0% |

Over the **full extended OOS** (to 2026-06-12), cell5_revalreg **trails** NDX and
b_acceptance only **edges** it on total return — but the rolling distribution shows
b_acceptance beats NDX in just **38% of 50-day holding periods** with a **−4.4%
median** excess; its small total-return lead is carried by a few outlier windows
(max +45%, mean +4.0% vs median −4.4% → strongly right-skewed). cell5_revalreg is
~coin-flip (46%, median −0.6%). **The single-window `_005` cell-5 number does not
generalize to a rolling edge.**

## Reading this honestly

- **This does NOT refute the rare-event `_005` results** (ndx40, sp500_20, r1k) — it
  can't test them (faithful-inference + refresh gap). It DOES show that for the one
  family we can roll (cell-5, high base rate, AUC ≈ 0.52), rank/equal has **no
  durable edge** over buy-and-hold NDX once you look across windows rather than at one.
- **It vindicates the `_005` caveat.** `_005` explicitly flagged "single windows, not
  an alpha claim; rolling validation is the gate." The gate fired: the cell-5 edge was
  largely window selection. The rare-event cells stay **unproven**, not disproven.
- **The promising cells are exactly the ones we can't yet validate.** That asymmetry is
  the actionable finding: the next work is *enabling* their validation (below), not
  declaring victory or defeat.

## Methodology

One clean back-test per cell over its **full** clean-OOS region (published test.csv +
inference-scored fresh region, faithfulness self-checked), rank/equal (c=1.0). From the
equity curve + index buy-hold on the same calendar, compute **rolling H-day (50d) excess
returns** at a 5-day stride; report the distribution. Rolling returns off one curve
(not fresh-capital sub-windows) because short fresh-capital windows starve the strategy
(2–3 trades) and per-window feed slicing is brittle. Overlapping windows are
autocorrelated — `frac_beat` and percentiles are descriptive, not iid significance.

## Caveats

- **C1: cell-5 family only.** N=2 models, same target, same universe (nasdaq100), same
  market regime (2025–26 bull). Not a cross-cell claim.
- **C2: overlapping windows** (stride 5 < window 50) → autocorrelated; treat the
  distribution as descriptive.
- **C3: the rare-event cells are untested**, and they are the ones `_005` made look
  strong. No conclusion about them either way from `_006`.
- **C4/C5:** zero costs, DD-not-bounded — as prior memos.

## Reproducibility

- Branch `backtests-v12-rolling-validation`. Per-window CSVs + summaries under
  `results/backtests/_006_rolling/{cell5_revalreg,b_acceptance}/`.
- `uv run python -m scripts.backtests.run_rolling_validation --cell <cell> --fresh <fresh.csv> --out <dir> --name <n>`
- Fresh OOS for cell5_revalreg via `infer_fresh_predictions` (self-check PASSED, 1.5e-8);
  ndx40 inference ABORTED on the faithfulness self-check (see TL;DR).

## Open questions / follow-ups (the real next work)

- **Faithful inference for Option-B / combined-feature cells** (`*_mix`, scout-combine):
  reproduce the composed feature overlay so ndx40 / similar cells pass the self-check and
  can be rolled. This unblocks validating the strongest `_005` results.
- **Per-universe cache refresh + generalized inference** (sp500, russell1000) so those
  cells can be extended + rolled.
- **Once unblocked**, roll ndx40 / sp500_20 / r1k — the actual test of whether the
  rare-event rank/equal edge is real or single-window.
- **Costs + sector-neutralization** sensitivity on any cell that survives rolling.
