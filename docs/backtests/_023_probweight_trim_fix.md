# _023: the prob_weight trim bug — _013/_014's "whole-universe spreading" was an artifact

## TL;DR

`_022` flagged a latent bug: the ratchet-down **trim** (rebalance pass) calls `_notional_f()`,
which only branches `equal`/`rank_kelly` and otherwise falls through to the **Kelly** target.
For `prob_weight` (and `inverse_vol`) that Kelly target is ≈0 on sub-breakeven rare-event
cells, so the trim **drives every position to ~0 the day after entry** → frees room →
re-enters K names daily → the book churns across the **entire universe**. This is exactly the
"prob_weight spreads to 486/890 names / becomes a de-risker" mechanism `_013`/`_014` reported —
**it was a bug, not a property of prob_weight.**

Fixed (skip the ratchet-down trim for the daily-normalized weight modes). On the same cell the
churn collapses: **sp500_50 prob_weight K=20: 406 → 24 entries, 394 → 0 trims.** Re-checking the
`_013`/`_014` conclusions on the corrected code gives a **mixed** result:

- **equal-K=3 still beats prob_weight on sp500_50** (+12.5% vs +1.6%) and **nifty +50%/50d**
  (−4.3% vs −14.7%) — the `_012` K-dilution story holds there.
- **prob_weight K=20 BEATS equal-K=3 on r1k** (+63.5% / 94% of windows vs +17.6% / 60%) —
  **reversing `_013`'s headline "r1k 90% → 4%, edge destroyed."**

So `_013`/`_014`'s blanket conclusion (*"no weighting scheme beats equal-weight top-3"*) **does
not survive the fix** — at least r1k reverses. Most important caveat: single-window, on the
**V1.4 date-aligned cells** (which differ from the `_013` originals), gross of costs — the r1k
reversal needs the full rolling re-run with fresh inference to confirm.

## The bug

```
rebalance pass, per open position:
  ... DD / target / horizon / breakeven exits ...
  TRIM (ratchet-down):  new_f = _notional_f(p_today[tk])      # <-- here
                        if new_f < cur_f: trim shares down to new_f
```

`_notional_f` has no `prob_weight`/`inverse_vol` branch (their per-name weight is normalized
over the **day's selected set** at entry — there is no per-position closed form). So it returns
the Kelly fraction, which for a calibrated `p` below the 1/3 breakeven clamps to **0**. Result:
`new_f ≈ 0 < cur_f` every day → trim to ~0 shares (but the name stays "open" with `f≈0`) →
`exposure = Σf ≈ 0` → `room ≈ gross_cap` → the entry pass adds K fresh names **every day**.
Over the OOS the book accumulates the whole universe. The tell is the **trim count**: 394 trims
on a 50-signal-day sp500_50 run.

**Fix:** skip the ratchet-down trim when `sizing_mode ∈ {prob_weight, inverse_vol}` (a
set-at-entry, daily-normalized weight has no per-position `_notional_f` target). `equal` is
unaffected (it *does* have a `_notional_f` form = the equal slice, and its ratchet-to-equal is
intended). `inverse_vol` was already fixed in `_022`; this extends the same guard to
`prob_weight`.

## Proof it was the bug (same cell, sp500_50 prob_weight K=20)

| | entries | unique | trims | gross | return | max DD |
|---|---|---|---|---|---|---|
| **bug-on** (pre-fix) | 406 | 406 | 394 | 0.42 | −2.7% | −6.0% |
| **bug-off** (fixed) | 24 | 23 | 0 | 0.48 | +1.6% | −9.5% |

The "whole-universe spreading" (406 names) and the "strongest de-risker" (DD −6.0%, an ultra-low
-vol book of 406 near-zero positions) both vanish. Fixed, prob_weight K=20 holds ~23 names — a
genuine K=20 book with multi-day-hold accumulation. Figure:
`results/backtests/_023_probweight_trim_fix/figs/probweight_fix.png`.

## Re-checking _013/_014 (corrected, single-window on the aligned cells)

K=3 = equal champion; K=20 = prob_weight (α=1, and α=4 where shown). `%beat` = fraction of
rolling H-day windows (stride 5) beating the index off the OOS equity curve; `n` = window count.

| Cell | config | entries | unique | return | max DD | %beat | med exc | n |
|---|---|---|---|---|---|---|---|---|
| sp500 +50%/50d | equal K=3 | 4 | 4 | **+12.5%** | −9.3% | **58%** | +1.8% | 12 |
| sp500 +50%/50d | pw K=20 | 24 | 23 | +1.6% | −9.5% | 33% | −2.7% | 12 |
| sp500 +50%/50d | pw K=20 α=4 | 24 | 23 | +1.6% | −9.5% | 33% | −2.7% | 12 |
| r1k +50%/200d | equal K=3 | 10 | 8 | +17.6% | −25.1% | 60% | +1.9% | 62 |
| r1k +50%/200d | pw K=20 | 84 | 42 | **+63.5%** | −25.2% | **94%** | +21.5% | 62 |
| r1k +50%/200d | pw K=20 α=4 | 76 | 42 | **+67.2%** | −25.3% | **98%** | +21.9% | 62 |
| nifty +50%/50d | equal K=3 | 7 | 7 | −4.3% | −18.8% | **73%** | +3.1% | 22 |
| nifty +50%/50d | pw K=20 | 43 | 39 | −14.7% | −15.9% | 55% | +0.5% | 22 |

**What holds, what reverses:**
- **sp500_50 / nifty_50:** equal-K=3 still beats prob_weight — consistent with `_012` (K=3
  concentration > K=20 dilution on the 50-day cells). `_013`/`_014`'s *direction* survives here,
  but the *mechanism* ("spreads to the whole universe") was wrong; it's ordinary K=20 dilution
  over ~23–39 names.
- **r1k_50:** prob_weight K=20 (and α=4) **beats** equal-K=3 by a wide margin — the opposite of
  `_013`'s "4% of windows, edge destroyed." On a 200-day horizon a wider book (~42 names) catches
  far more of the +50% winners than a 3-name book; the bug had hidden this by churning r1k into a
  whole-universe closet-index.

So **prob_weight is not a universal loser** — `_013`/`_014` overstated. On the long-horizon r1k
cell it is the better deployment; on the 50-day cells equal-K=3 remains best.

## Caveats

- **C1: single window + aligned cells.** The `_013`/`_014` originals were pre-V1.4 cells; only
  the date-aligned variants exist now, so this is a *corrected re-evaluation on the current
  cells*, not a byte-reproduction. The %beat is rolling off one OOS equity curve (12–62
  overlapping, autocorrelated windows; sp500's 12 is thin). **The r1k reversal needs the full
  rolling re-run with fresh post-test inference to firm up** — flagged as the follow-up.
- **C2: gross of costs.** prob_weight K=20's turnover (84 entries) is higher than equal-K=3's
  (10) but **nothing like** the pre-fix 890 — so the cost penalty `_013`/`_014` emphasized is far
  smaller than they claimed (that too was inflated by the churn).
- **C3: this does not touch equal/kelly/rank_kelly.** Their `_notional_f` targets are correct;
  `_008`/`_012`'s equal-mode K-sweeps are unaffected by this bug.

## Corrections filed

Added a correction banner to `_013` and `_014` pointing here: their "whole-universe / 486–890
entries / de-risker / r1k 4%" numbers are **bug-contaminated**; the corrected reading is above.

## Reproducibility

- Branch `backtests-probweight-trim-fix`. Fix in `topk_daily_kelly_label_exit.py` (trim guard now
  excludes `prob_weight` as well as `inverse_vol`); `run_backtest_cell.py` gains `prob_weight` +
  `--prob-weight-alpha` (so single-window prob_weight is runnable). **+2 regression tests**
  (`test_prob_weight_not_ratchet_trimmed_to_zero` + the `inverse_vol` analog) lock in the fix;
  38 strategy tests pass.
- Runs: `run_backtest_cell --cell <aligned cell> --c 1.0 --selection-mode rank --sizing-mode {equal --k 3 | prob_weight --k 20 [--prob-weight-alpha 4]}` → `results/backtests/_023_probweight_trim_fix/runs/`.
- Data: `results/backtests/data/_023_data.json` (incl. the sp500_50 bug-on vs bug-off block).

## Follow-ups

- **Full rolling re-run with fresh inference** (`run_rolling_validation` + regenerated `--fresh`
  CSVs) on all 5 cells × {K=10, K=20, α∈{2,4,8}} to confirm the r1k reversal and refresh the
  exact `_013`/`_014` %beat numbers on the aligned cells. (Also fix `run_rolling_validation`'s
  empty-`r.excess` crash when the OOS is shorter than one H-day window — surfaced during this work.)
- **Re-open the K question for long-horizon cells:** if wide-K wins on r1k (200d) but loses on the
  50-day cells, optimal K may track the horizon, not just R-p@1 (`_012`).
