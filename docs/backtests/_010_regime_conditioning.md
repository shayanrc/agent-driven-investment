# _010: regime-conditioning — the `_009` regime hypothesis is REFUTED

## TL;DR

`_009` speculated that the rank/equal edge "is gated by the market regime" — that
the US cells beat their index only because they were sampled in bull tapes, and the
NSE cells failed only because NIFTY500 was down. We tested that directly by labelling
every rolling window (667 across 11 cells, 4 universes) with the **index's own return
over that window** and conditioning excess on it. **The hypothesis does not survive:**

1. **The edge is not long-beta.** Within most cells, excess is *negatively* or
   *un*-correlated with the index return — the strategy does **relatively better in
   down windows**, the opposite of a closet bull-market bet.
2. **Conditioning on regime does NOT close the US–NSE gap.** In down-index windows the
   US cells earn **+10.0% median excess (98% win)** while NSE earns **+0.2% (53%)**; in
   up-index windows US **+5.6% (64%)** vs NSE **−1.9% (38%)**. The market-quality /
   universe difference dominates; the bull/bear tape does not explain it.

So `_009`'s headline ("regime is the gating variable") was wrong. The real picture is
**better for the US result and worse for NSE** than `_009` implied: the US edge is
regime-*neutral* (not a bull artifact), and NSE is weak in *both* regimes (not just
unlucky in a down tape).

## The evidence

### Pooled by market × regime (idx_ret sign)

| Market | Regime | n windows | median idx_ret | median strat_ret | median excess | win-rate (excess>0) |
|---|---|---|---|---|---|---|
| US | down | 47 | −3.1% | **+7.3%** | **+10.0%** | **98%** |
| US | up | 225 | +10.5% | +15.8% | +5.6% | 64% |
| NSE | down | 215 | −3.5% | −3.5% | +0.2% | 53% |
| NSE | up | 180 | +3.2% | +0.9% | −1.9% | 38% |

The US strategy beats the index **more** when the index falls (+10.0% vs +5.6%); NSE
is flat-to-negative in both. If the edge were regime-gated, US-down and NSE-down would
look alike — they are 50 points of win-rate apart.

### Within-cell excess-vs-regime correlation (horizon fixed → clean)

Pooled, after standardizing `idx_ret`/`excess` within each cell (so the three horizons
are comparable):

| Market | n windows | n cells | Spearman(excess, idx_ret) | p |
|---|---|---|---|---|
| US | 272 | 6 | **−0.05** | 0.40 (n.s.) |
| NSE | 395 | 5 | **−0.25** | <1e-3 |
| ALL | 667 | 11 | −0.15 | 7e-5 |

**US excess is statistically independent of the regime** (ρ≈0, p=0.40) — a regime-neutral
edge, not a bull artifact. **NSE excess is significantly *negatively* related** to the
index return (does better when the market falls) — still no long-beta, and the level is
~0 regardless. Per-cell, the only strongly positive ρ is **r1k_50 (+0.64)** — but H=200
windows are almost all up (4/105 down), so it barely samples the down regime; the
shorter-horizon cells (which see both regimes) are flat-to-negative.

See `figs/regime_scatter.png`: the US (blue) cloud sits above zero across the whole
x-axis; the NSE (orange) cloud hugs zero everywhere. Neither slopes up-and-to-the-right.

## What actually separates US from NSE (the redirected question)

Regime is out. After conditioning, the gap is intrinsic to the universe/signal. Candidate
explanations, now the next investigation:

- **Label-AUC ≠ tradeable excess on NSE.** The NSE cells have high *label* AUC (0.60–0.89,
  measured on each cell's own test split) yet near-zero per-window excess in both regimes.
  Either the AUC is inflated/over-fit on the NSE panel, or the ranked names can't be
  *captured* (gap-prone small/mid-caps, the +50% moves happen in untradeable jumps).
- **Breadth / dispersion.** US large-cap rare-event winners trend; NSE +50% moves may be
  one-day gaps the next_open fill can't catch.
- **Costs (not modelled here)** are materially larger on NSE and would only widen the gap.

## Caveats

- **C1: US down-window N is small (47), and lumpy.** It is dominated by a handful of NDX
  windows (`b_acceptance` med excess in down windows +37.7%!) where the index fell but the
  strategy's few rare-event winners kept climbing. The +10.0% US-down figure rests on few
  windows / few names — directionally clear, not precise.
- **C2: rolling-window excess ≠ full-OOS compounding.** A cell with only mildly-negative
  per-window medians (nifty +50%/25d: −2 to −4%) still compounded to `_009`'s −55.6% full-OOS
  because the loss is concentrated in a few catastrophic windows (min excess −28.5%) and the
  full path rides concentrated positions. The two metrics answer different questions; this
  memo is about the *typical-window* edge, not the path.
- **C3: overlapping windows** (stride 5) → autocorrelated; the n's overstate independent trials.
- **C4: one NSE universe / one ~16-mo NSE window.** "NSE is weak in both regimes" is shown
  for nifty500 over this window only; it is not a claim about NSE in general.

## Net

The `_009` regime story is replaced by a sharper one: **the rank/equal edge is
regime-neutral where it exists (US: positive excess in both up and down windows, ρ≈0),
and absent where it doesn't (NSE: ~0 excess in both).** This *strengthens* the US result
against `_008`'s "bull-market artifact" worry, and reframes the NSE failure as a
signal/universe problem, not bad-luck timing. The next step is **NSE signal forensics**
(is the label-AUC real and capturable?), not regime work.

## Reproducibility

- Branch `backtests-v14-regime`. Pure post-hoc: `uv run python -m scripts.backtests.regime_conditioning`.
- Reads the committed `rolling_windows.csv` for all 11 rolled cells (`_006`–`_009`).
- Outputs: `results/backtests/_010_regime/{per_cell_regime.csv, pooled_regime.csv, summary.json, figs/regime_scatter.png}`.

## Open questions / follow-ups

- **NSE signal forensics**: re-examine the NSE cells' test-split AUC vs a permutation null;
  inspect whether the top-ranked names' +50% moves are next-open-capturable or jump-gaps.
- **Re-state `_008`/`_009` headlines** to "regime-neutral US edge / absent NSE edge" (done
  in this memo; the earlier memos stand as the record of how the read evolved).
- **More US down-regime windows** (a genuine US bear window) to firm up the +10.0% figure.
