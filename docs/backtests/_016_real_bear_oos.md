# _016: the real-bear OOS test — the champion's edge does NOT survive a sustained bear

## TL;DR

`_015` flagged two honest gaps in the equal@K=3 champion: a small effective-N and **no
real bear** (the worst sub-windows were only −9% to −19%). This memo closes the bear gap
the only valid way — by **retraining** the two sp500 champions with a pre-bear cutoff so
the **test segment is the 2022 bear** (the model never sees 2022), then rolling that
genuinely-OOS test vs ^SPX.

**The result is a clean negative: the consistent edge collapses to coin-flip in a real
bear, and concentration amplifies the drawdown.**

| Cell | regime | full-OOS strat | index | strat DD | % windows beat | median excess |
|---|---|---|---|---|---|---|
| sp500 +50%/50d | bull 2023–26 | +34.2% | +7.8% | −10.1% | **92%** | **+7.3%** |
| sp500 +50%/50d | **bear 2022** | **−22.2%** | −20.5% | **−32.8%** | **53%** | **+0.5%** |
| sp500 +20%/25d | bull 2023–26 | +73.7% | +7.8% | −8.3% | **100%** | **+9.5%** |
| sp500 +20%/25d | **bear 2022** | **−2.3%** | −13.4% | **−32.3%** | **50%** | **+0.3%** |

(Bull rows = the production champions on 2023–26 fresh data, from `_015`. Bear rows = the
pre-bear-cutoff retrains on 2022. Same cell spec, different regime — the comparison is
regime, not model.)

Two things break at once in the bear:

1. **The consistent rolling edge vanishes** — 92–100% of windows beat in the bull → **50–53%
   (coin-flip)** in the bear, with median excess collapsing from +7–10% to **~0%**.
2. **Drawdowns amplify** — both cells draw down **−32% to −35%** vs the index's ~−25%
   peak-to-trough. The concentration (K=3) that monetized precision in the bull *adds* risk
   in the bear: the model's high-p names are high-beta stocks that fall *harder* than the
   index.

## Why it breaks (mechanistic)

The target event is a **large up-move in a fixed window** (+50% in 50d, +20% in 25d). In a
sustained bear that event **structurally almost stops happening** — positive prevalence on
the sp500 +50% cell fell to **1.6%** in 2022 (vs 2.6% normally). The model still has to rank
*something* into the top-3 each day, so it picks the highest-beta / highest-momentum names —
which in a bear are exactly the ones that **fall fastest**. There is no skill to extract
because the thing the model detects (incipient explosive up-moves) isn't occurring.

sp500 +20%/25d ends the full window *less negative than the index* (−2.3% vs −13.4%) only
because the shorter horizon / lower threshold lets it **catch the two sharp 2022 bear
rallies** (summer + October) — but the rolling distribution (50% beat, +0.3% median) shows
that's **episodic luck, not a consistent edge**, and it still carries the −32% drawdown.

## This corrects the `_010`/`_015` "regime-neutral" read

`_010` concluded the US edge was *regime-neutral* (ρ≈0; it did fine in down windows), and
`_015`'s shallow bear sub-windows (−9% to −19%) showed the strategy beating the index. Both
were measuring **short-term down-moves *within* the 2023–26 bull** — dips, not a sustained
bear. **A dip is not a bear.** The 2022 test (a −25% peak-to-trough grind over ten months) is
a different regime, and the edge does not survive it. The honest synthesis:

> The equal@K=3 champion is a **bull / range-bound-market strategy**. It tolerates dips, but
> in a sustained bear its edge is coin-flip and its drawdown is *worse* than passive. It is
> **not all-weather**, and it should not be run naked through a bear.

## What this means for deployment

- **A regime overlay is now a requirement, not a nicety.** Gate the strategy off (to cash /
  index) when the universe index is in a confirmed downtrend; the edge lives in
  bull/range-bound tape. (The strategy has no built-in regime filter — this is downstream.)
- The bull-market numbers (`_008`/`_015`) remain valid **conditional on a non-bear regime** —
  they are not overturned, they are **scoped**.
- Do not read sp500_20's −2.3% as "bear-resilient" — it is high-variance rally-catching with
  a −32% drawdown; on a rolling basis it has no edge.

## Caveats

- **One bear (2022).** 2022 was a rates-driven, orderly bear; a 2008-style credit crash or a
  2020-style vol spike could differ. One bear is one data point — but it is the data point
  the arc was missing, and the sign is unambiguous.
- **Retrained models, not the production champions.** The bear cells were trained on
  2016–2019 (train) / 2019–2021 (val/eval) — a different (smaller, older) training window
  than the production champions. The comparison isolates *regime*, but the two models are not
  identical artifacts. The mechanism (the target event vanishes in a bear) is model-agnostic,
  so this is not a confound for the conclusion.
- **Survivorship**: the 2022 test uses the *current* sp500 membership (cached universe), a
  mild forward-looking bias toward survivors — which if anything **flatters** the bear result,
  so the true picture is no better than reported.
- **Forward-OOS (the other `_015` gap) is not run here** — the US cache is already current
  (2026-06-12), so there is no new data to extend into; it grows only as wall-clock time
  passes (the recurring `ROLLING` node).

## The arc, re-settled

`_005`→`_016`: the tradeable edge is **high-R-p@1 cells, traded concentrated (equal-weight
top-3)** — cost-robust, bootstrap-significant on 3/4 cells, **and conditional on a
non-bear regime**. The bear test is the boundary of the claim: outside a sustained bear the
edge is real and monetizable; inside one it is coin-flip with amplified drawdown. **The
strategy stands, now correctly scoped; the missing piece for deployment is a regime gate, not
more sizing work.**

## Reproducibility

- Branch `backtests-v20-bear-oos`. Specs
  `configs/gbdt/experiments/sp500_up_{50pct_50d_dd25pct,20pct_25d_dd10pct}_bear2022.yaml`
  (date_aligned split, `train_start: 2016-06-01`, 800/400/200/250 train/val/eval/test rows,
  `--snapshot-end 2022-12-31` → test = the 2022 bear). Train:
  `uv run python -m gbdt experiment <spec> --callback-mode default --snapshot-end 2022-12-31`.
  Roll: `run_rolling_validation --cell <bear-artifact> --k 3 --sizing-mode equal --cost-bps {0,25}`
  (no `--fresh` — the test.csv *is* the 2022 OOS). Artifacts `results/backtests/_016_bear/`;
  registry rows 080–083. Figure `figs/bull_vs_bear.png`.

## Open questions / follow-ups

- **A regime gate** (e.g. index > 200d MA, or a vol/trend filter) wrapped around the strategy
  — does gating off in confirmed downtrends recover the bull-only edge with controlled
  drawdown? This is the natural V1.x+1 backtest.
- **A second bear** (2020 COVID, or a 2008 sweep if the panel reaches back) to see whether the
  collapse generalizes across bear *types*.
