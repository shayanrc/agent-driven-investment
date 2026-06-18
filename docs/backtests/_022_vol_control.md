# _022: volatility control on raw ranking — does risk parity rescue it?

## TL;DR

`_021` showed that ranking the entry top-K by the raw model score (instead of the quantized
calibrated `p`) **hits more targets but worsens drawdown and risk-adjusted return** — the
highest-raw names are the highest-**beta** names, which reach the target more often *and* crash
hardest. The obvious follow-up: add a **volatility control** that keeps raw *selection* but
damps the beta via **inverse-volatility (risk-parity) sizing** — slice ∝ 1/vol, normalized so
the book targets `c·gross_cap`. The hypothesis: keep the extra targets without the extra
drawdown.

**It doesn't pay.** Across the full 2×2 (selection × sizing) on both cells and regimes,
inverse-vol sizing **never beats the equal-weight champion** and does **not** consistently
rescue raw ranking: it softens the sp500_20 bear blowup (−28.8% → −21.3%) but **lowers return
everywhere** and **worsens sp500_50** (bull +14.2% → +4.9%; bear −12.8% → **−22.3%**). On these
rare-event *up-move* cells the high-vol names **are** the return drivers, so down-weighting them
sacrifices the winners faster than it saves on the losers — **the volatility you penalize is the
signal.** The single most important caveat: single bull + single bear OOS window per cell, gross
of costs.

## The 2×2 (selection × sizing), K=3, c=1.0

Return % (max DD %, ret/|DD|). Equal-sizing columns are the `_021` runs (reused); inverse-vol is
new. **Champion = calibrated + equal.**

| Cell · regime (SPX) | calib + equal | calib + inv_vol | raw + equal | raw + inv_vol |
|---|---|---|---|---|
| sp500_20 bull (+8.4%) | **+58.1** (−7.3, 8.0) | +35.5 (−9.1, 3.9) | +56.6 (−13.8, 4.1) | +50.2 (−12.6, 4.0) |
| sp500_20 bear (−17.4%) | **−5.8** (−32.3, −0.18) | −10.4 (−32.5, −0.32) | −28.8 (−31.4, −0.92) | −21.3 (−31.1, −0.69) |
| sp500_50 bull (+8.4%) | **+12.5** (−9.3, 1.34) | +4.9 (−6.3, 0.78) | +14.2 (−15.0, 0.94) | +4.9 (−12.8, 0.38) |
| sp500_50 bear (−17.4%) | −9.2 (−32.8, −0.28) | **−4.8** (−24.7, −0.19) | −12.8 (−29.6, −0.43) | −22.3 (−36.8, −0.61) |

Figure: `results/backtests/_022_vol_control/figs/vol_control_comparison.png`.

## Reading it

- **Does inverse-vol rescue raw?** Only partially and inconsistently. sp500_20 bear improves
  (−28.8% → −21.3%, DD ~flat) and sp500_20 bull DD tightens slightly — but sp500_50 gets
  **worse** in both regimes (bull return halved-and-more; bear return *and* DD deteriorate). No
  consistent rescue.
- **The equal-weight champion still wins.** `calib + equal` beats `raw + inv_vol` on **both
  return and risk-adjusted return in all four cells**. The vol control creates no new winner.
- **Why — the vol you penalize is the signal.** These are rare-event *up-move* cells; the names
  that reach +20%/+50% are disproportionately the high-vol names. Inverse-vol sizing shrinks
  exactly those positions, so it gives back upside faster than it avoids drawdown. (Lone clean
  risk-reduction: `calib + inv_vol` cut the sp500_50 *bear* DD −32.8% → −24.7% — but the same
  control hurt sp500_50 *bull*, so it isn't a reliable risk-reducer either.)
- This closes the `_021` follow-up and matches the whole sizing arc (`_012`/`_013`/`_014`):
  **no sizing or weighting scheme beats equal-weight top-3.**

## A bug caught mid-experiment (fixed for `inverse_vol`)

The first inverse-vol run produced nonsense — **147–467 entries** (vs ~13–54 for equal) and
depressed gross. Cause: the ratchet-down **trim** in the rebalance pass calls `_notional_f()`,
which has no `inverse_vol` branch, so it fell through to the **Kelly** target (≈0 on these
sub-breakeven cells) and trimmed every position to ~0 the day after entry → room freed →
3 new names entered daily (a churn artifact, not a strategy). **Fixed** by skipping the
ratchet-down trim for `inverse_vol` — a set-at-entry risk-parity weight has no per-position
`_notional_f` form. Re-ran clean (entries back to 4–80, gross 0.43–0.72). The numbers in this
memo are all post-fix.

**Flag (not fixed here):** `prob_weight` almost certainly shares the same `_notional_f`→Kelly
fall-through in the trim (`_013` reports r1k prob_weight at *890 entries* — consistent with the
same churn). It is **not** touched in this branch: fixing it would change the merged `_013`/`_014`
results, so per the repo's "surface, don't silently change" rule it is left flagged. A dedicated
fix + a re-run of `_013`/`_014` is the right way to resolve it.

## Implementation

- **Strategy** (`src/trading_strategies/topk_daily_kelly_label_exit.py`): `sizing_mode=
  "inverse_vol"` + optional `vol_scores: dict[Timestamp, dict[ticker, float]]`. The day's
  selected names get slices ∝ 1/vol normalized to `c·gross_cap` (risk parity); missing/≤0 vol →
  mean-of-present fallback (≈ equal for that name); gentle 0.1·equal-slice dust floor (a 0.5·
  floor would clip exactly the high-vol names). Ratchet-down trim skipped for this mode (see
  above). Selection is untouched, so it composes with `rank_by`.
- **Runner** (`scripts/backtests/run_backtest_cell.py`): `--sizing-mode inverse_vol` +
  `--vol-window` (default 20). Builds `vol_scores` as trailing realized vol (causal, returns
  through the signal day) from a wider pre-buffered close fetch. `vol_window` recorded in
  `summary.json::config`.
- **Tests** (`tests/trading_strategies/test_topk_daily_kelly.py`): +3 — slice ∝ 1/vol,
  missing-vol fallback, and the `vol_scores` requirement. 36 pass.

## Caveats

- **C1: single windows, small N, gross of costs** — one bull + one bear OOS per cell. The
  verdict is "inverse-vol is not an improvement here," not a universal law. Rolling-origin per-
  sizing distribution is the rigorous extension (not run).
- **C2: 20-day realized vol** is one reasonable choice; a longer window or beta-to-index would be
  alternative controls (untested) — but the mechanism (winners *are* the high-vol names) suggests
  any vol-penalizing control faces the same headwind on up-move cells.
- **C3: bear cells are leak-free** (`_016`/`_020`/`_021`): train 2016→2019-08, test = held-out
  2022 bear.

## Reproducibility

- Branch `backtests-rank-by-vol-control`.
- Inverse-vol runs (per cell × regime × selection):
  `uv run python -m scripts.backtests.run_backtest_cell --cell results/gbdt/experiments/<cell> --out results/backtests/_022_vol_control/runs/<name> --name <…> --c 1.0 --k 3 --selection-mode rank --sizing-mode inverse_vol --rank-by {calibrated|raw}`
- Equal-sizing baselines reused from `_021` (`results/backtests/_021_entry_p_threshold/rank_by/`).
- Headline metrics (full 2×2): `results/backtests/data/_022_data.json`. Figure under
  `results/backtests/_022_vol_control/figs/`.

## Follow-ups

- **`prob_weight` trim fall-through** — dedicated fix + `_013`/`_014` re-run to confirm those
  conclusions hold (the qualitative finding — prob_weight spreads and destroys the edge — likely
  survives, but the entry counts are suspect).
- **Closes the sizing/selection thread for now.** The standing champion remains rank-select,
  equal-weight, K=3 on calibrated `p`, gated by SMA200 (`_017`). The open frontier is regime/
  cost robustness, not sizing.
