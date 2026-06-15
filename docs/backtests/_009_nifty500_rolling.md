# _009: rolling nifty500 — the US AUC→edge pattern does NOT transfer to NSE

## TL;DR

We rolled five nifty500 cells — four high/moderate-AUC + one low-AUC control — to
test whether `_008`'s clean finding (**AUC tracks the rolling edge**) replicates on
the NSE universe. **It does not.** Over a flat-to-**down** NIFTY500 tape (index
−4.8% across the OOS), the AUC→edge ordering breaks: the **highest-AUC cells suffered
the worst drawdowns** and only a **moderate-AUC** cell showed a mild positive edge.

| Cell | AUC | Full-OOS strat (DD) | NIFTY500 | Rolling: % windows beat | median excess | entries |
|---|---|---|---|---|---|---|
| nifty500 +50%/50d | **0.89** | **−9.2%** (DD **−37.0%**) | −4.8% | 44% | −1.1% | 30 |
| nifty500 +50%/25d | 0.83 | **−55.6%** (DD **−60.2%**) | −4.8% | 35% | −3.0% | 40 |
| nifty500 +30%/25d | 0.81 | −0.9% (DD −27.5%) | −4.8% | 52% | +0.2% | 50 |
| nifty500 +20%/25d | 0.72 | **+34.3%** (DD −26.4%) | −4.8% | **57%** | **+1.6%** | 65 |
| nifty500 +10%/25d (control) | 0.60 | −13.7% (DD −24.2%) | −4.8% | 41% | −1.6% | 48 |

OOS: 2024-09-04 → 2026-05-21 (425 signal days, ~16 mo), 75–80 rolling windows/cell.

## What this means

- **AUC did not order the edge on NSE.** The AUC-0.89 standout (+50%/50d) lost money
  with a −37% drawdown; the AUC-0.83 cell (+50%/25d) was a **−55.6% / −60% DD blow-up**.
  The best cell was the *moderate*-AUC +20%/25d (+1.6% median, 57% of windows). The
  monotone AUC→edge ranking from `_008` is absent here.
- **The regime is the prime suspect.** Every US window in `_005`–`_008` was a bull tape
  (NDX/SPX +8% to +64%); this NSE window is **flat-to-down (NIFTY500 −4.8%)**. The
  rare-event +50% labels need a strong up-move to pay off; in a down/choppy market they
  fire, fail to reach target, and ride to the drawdown stop — hence the −37% / −60% strat
  DDs. This is `_008`'s caveat **C3 (favorable regimes), confirmed by counterexample**:
  the US edge was at least partly a bull-market artifact, not pure ranking skill.
- **The drawdowns are the real story.** NSE strat DDs (−24% to −60%) dwarf the US cells
  (−8% to −22%). On the high-threshold cells the strategy is effectively long-vol on a
  market that didn't cooperate.
- **The one mild bright spot is regime-robust-ish.** +20%/25d (lower threshold → fires
  more often, 65 entries; nearer-term target) cleared the index in 57% of windows with a
  positive median **even in a down market** — the only cell that did. Lower thresholds may
  travel better across regimes than the +50% "lottery" cells.

## Why this result is *more* credible than the US positives (in one direction)

Unlike `_008` (9–22 entries), these cells have **30–65 entries** and **75–80 windows** over
a longer, *adverse* regime. A negative result on a larger sample in a hostile tape is strong
evidence; it does **not** prove the cells are worthless (a single NSE down-regime is still one
regime), but it decisively refutes "high AUC ⇒ rolling edge, universe-independent."

## Methodology

Identical to `_008`: faithful inference (gap-fill aligned; all 5 self-checked PASSED at
≤1e-4) over test.csv + fresh predictions to the index's coverage end (NIFTY:500 cached to
2026-05-22), one full-OOS rank/equal (c=1.0) back-test, then rolling H-day excess vs
NIFTY:500 at stride 5. Calibrator fit on each cell's VAL.

Code: added `nifty500`/`nifty50` to `INDEX_BY_UNIVERSE` (NSE indices use the `NIFTY:`
prefix, routed to the nse_equities table by `gbdt.data._cache_read`). `NIFTY:500` is the
Nifty-500 total-market benchmark.

## Caveats

- **C1: one NSE regime, and an adverse one.** −4.8% index over the window. The *opposite*
  of the US bull windows. The honest read across `_008`+`_009`: **the edge is
  regime-dependent**, positive in bull tapes, negative-to-flat in this down tape. Neither
  window proves the steady-state.
- **C2: data as-of 2026-05-21.** The live NSE refresh failed (the `nselib` provider was
  broken — date-conversion errors on nearly every symbol), so we used the cached panel
  (fresh to 2026-06-12 for stocks; index to 2026-05-22, which capped the OOS). No fresh
  data was lost — only the last ~3 weeks of extension.
- **C3: overlapping windows** (stride 5 ≪ 25/50) → autocorrelated, as prior memos.
- **C4: zero costs; NSE costs (STT, higher spreads/impact) are materially larger than US** —
  the +20%/25d edge (+1.6% median) is thin enough that realistic NSE costs could erase it.
- **C5: DD not bounded on gap-downs** — especially punishing here given the −37%/−60% DDs.

## The cross-market picture (what `_005`–`_009` now say together)

The rank/equal deployment of high-AUC rare-event gbdt cells **beat the index in US bull
windows (`_007`/`_008`: 57–100% of windows) and failed in an NSE down window (`_009`: 35–57%,
mostly < 50%)**. AUC predicts ranking quality on the *label*, but whether that ranking earns
excess return is **gated by the market regime** — the missing variable. The next step isn't
more cells; it's **regime-conditioning**: re-run the US cells across a *bear* sub-window and
the NSE cells across a *bull* sub-window before any cross-market alpha claim.

## Reproducibility

- Branch `backtests-v13-nifty`.
- Per cell: `infer_fresh_predictions --cell <cell> --out <fresh.csv> --end 2026-05-22` →
  `run_rolling_validation --cell <cell> --fresh <fresh.csv> --out <dir> --name <n>`.
- Artifacts under `results/backtests/_009_nifty/`; registry rows 016–020.

## Open questions / follow-ups

- **Regime-conditioning** (the headline follow-up): split each OOS into up/down sub-windows
  and recompute the edge; the US-vs-NSE contrast predicts the edge flips with the tape.
- **Lower-threshold NSE cells** (+10%/+20% across horizons) look more regime-robust than the
  +50% lottery cells — worth a focused NSE sweep.
- **NSE transaction-cost model** (STT + impact) before any NSE edge is taken seriously.
- **Fix/replace the NSE OHLCV provider** (`nselib` is broken) so NSE caches can be refreshed
  to today — see `[[project-nse-data-quirks]]`.
