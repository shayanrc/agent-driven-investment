# _017: the regime gate — a risk-reducer, not an alpha-restorer

## TL;DR

`_016` showed the equal@K=3 champion is bull/range-bound-only and proposed a **regime gate**
(go to cash in confirmed downtrends) as the deployment fix. This memo builds it and tests the
hypothesis "gating to cash in a bear recovers the bull-only edge with controlled drawdown."

**Verdict: half-true. The gate reliably preserves the bull edge at near-zero cost — but it
cannot manufacture a bear edge.** It is a *risk-reducer*, not an *alpha-restorer* (the same
shape as `_012`'s wider-K finding).

- **Bull (2025–26): edge preserved.** A classic SMA200 trend gate masks only ~11% of bull
  days (minor dips) and leaves the edge intact-to-better: sp500_50 **92% → 100%** of windows
  beat (+34.2% → +34.4%), sp500_20 stays **100%** (+73.7% → **+81.0%**, DD −8.3% → −7.5%). The
  gate does **no harm** in a bull — you can run it in production for free.
- **Bear (2022): no gate speed rescues returns.** The SMA200 gate correctly sits out ~80% of
  the bear, but its rolling edge stays **at/below coin-flip** (sp500_50 38%, sp500_20 50%) and
  full-OOS returns stay negative (sp500_50 −25.9%, even *worse* than ungated −22.2%). Faster
  MAs (100/50d) whipsaw and are noisier still. The **one robust bear benefit** is exposure
  reduction: SMA200 cuts sp500_20's drawdown **−32.3% → −22.2%**.

The gate is worth deploying (cheap bull insurance + bear-drawdown reduction) but it does **not**
make the strategy all-weather. `_016`'s verdict stands.

## Why the gate can't fix the bear (mechanistic)

The 43 entry-allowed (risk-ON) days inside the 2022 test cluster in **2021-12-21 → 2022-04-07**
— the *topping process*. On 2022-01-03 (the all-time high, right before the −25% decline) the
SMA200 was still **rising** (4387 vs 4306 twenty days earlier), trailing the 2021 bull. **No
causal trend filter can avoid entering right before a major top breaks** — that lag is
irreducible. After April the gate *did* sit out the steep April→October leg; the damage is the
handful of top entries it couldn't avoid. A slope condition (require SMA also rising) changed
nothing (identical 166/209 gated) precisely because those entries occur while the MA is still
rising. Faster MAs break down sooner but then **whipsaw** — flipping risk-ON during bear
rallies and letting *more* mistimed entries through.

Underneath it all is `_016`'s mechanism: the target event (a big up-move in N days) structurally
**stops occurring** in a bear, so there is no edge for any overlay to gate *toward*.

## Results

### sp500 +50%/50d

| regime | gate | full-OOS | maxDD | % windows beat | median excess | entries |
|---|---|---|---|---|---|---|
| bull 2025–26 | ungated | +34.2% | −10.1% | 92% | +7.3% | 9 |
| bull 2025–26 | **SMA200** | +34.4% | −10.1% | **100%** | +7.3% | 9 |
| bear 2022 | ungated | −22.2% | −32.8% | 53% | +0.5% | 15 |
| bear 2022 | **SMA200** | −25.9% | −33.9% | 38% | −5.4% | 6 |
| bear 2022 | SMA100 | −30.3% | −34.7% | 41% | −6.5% | 9 |
| bear 2022 | SMA50 | −30.2% | −33.9% | 31% | −3.5% | 9 |

### sp500 +20%/25d

| regime | gate | full-OOS | maxDD | % windows beat | median excess | entries |
|---|---|---|---|---|---|---|
| bull 2025–26 | ungated | +73.7% | −8.3% | 100% | +9.5% | 19 |
| bull 2025–26 | **SMA200** | +81.0% | −7.5% | 100% | +9.5% | 19 |
| bear 2022 | ungated | −2.3% | −32.3% | 50% | +0.3% | 42 |
| bear 2022 | **SMA200** | −8.3% | **−22.2%** | 50% | +0.6% | 11 |
| bear 2022 | SMA100 | −28.2% | −40.6% | 40% | −2.1% | 21 |
| bear 2022 | SMA50 | +0.8% | −28.0% | 52% | +0.4% | 24 |

(Bull = production champions on 2023–26 fresh data; bear = the `_016` pre-bear-cutoff retrains
on 2022. SMA200 is the recommended default; faster MAs shown to demonstrate that speed does not
fix the bear.) See `figs/regime_gate.png`.

## What this means for deployment

- **Ship the SMA200 gate.** It preserves the bull edge (the source of all the alpha) and caps
  bear participation/drawdown — cheap insurance with no bull cost. Implemented as harness
  preprocessing (`run_rolling_validation --regime-ma 200 [--regime-slope D]`), masking
  predictions on risk-off days; the strategy stays backend-agnostic per
  `docs/trading_strategies/goal.md` (universe-market logic does **not** live in the strategy).
- **Do not expect a bear edge from it.** The honest deployment story is: *bull/range-bound
  alpha, with a regime gate that limits — but does not eliminate — bear losses, and cannot turn
  them into gains.*
- **Don't tune the MA on this one bear** — the bear sample is tiny and the MA-speed results are
  noisy; SMA200 is the least-whipsaw default, not an optimized choice.

## Caveats

- **One bear (2022)**, as in `_016` — rates-driven and orderly; a 2020/2008-type bear could
  differ in both the signal collapse and the gate's timing.
- **Soft gate** (no new entries on risk-off days; open positions exit via the normal
  DD/target/horizon rules). A hard flatten-on-flip was not needed — no-new-entries plus ≤H-day
  natural exits reaches cash quickly — and would not help the core problem (mistimed entries
  *before* the flip).
- **Short bull window** (2025-12-30 → 2026-06-12, 13–18 rolling windows): the bull faster-MA
  improvements (sp500_50 g50 +70%) are likely the gate sitting out one early-2026 correction and
  should not be read as a robust MA-speed effect. The robust bull claim is only "SMA200 does no
  harm."
- Survivorship in the 2022 retrains (current membership) flatters the bear, as in `_016`.

## The arc, with the deployment wrapper attached

`_005`→`_017`: the tradeable edge is **high-R-p@1 cells, traded concentrated (equal-weight
top-3)** — cost-robust, bootstrap-significant on 3/4 cells, conditional on a non-bear regime,
**and now packaged with an SMA200 regime gate** that preserves the bull edge and limits bear
drawdown. The strategy is a **bull/range-bound alpha with bounded bear risk** — not all-weather,
and honestly so.

## Reproducibility

- Branch `backtests-v21-regime-gate`. New `run_rolling_validation` flags `--regime-ma N`
  (price > causal N-day SMA) and `--regime-slope D` (also require the SMA rising over D days),
  implemented via the pure helper `compute_risk_on(idx_close, ma, slope)` (5 unit tests in
  `tests/backtests/test_regime_gate.py` — causality, price>MA, slope veto, no-lookahead,
  warmup→NaN). Runs under `results/backtests/_017_regime/`; registry rows 084–095. Figure
  `figs/regime_gate.png`.

## Open questions / follow-ups

- **A second bear** (2020/2008) to see whether the gate's timing failure (entering the top)
  generalizes, or whether a sharper crash (2020) flips risk-off fast enough to help.
- **A faster regime signal** that is forward-looking rather than trend-lagging (e.g. realized-vol
  spike, credit spread, breadth) — could it catch the top the SMA200 structurally cannot? This is
  a research question, not a sizing one.
- The strategy + sizing + gate are settled; further work is regime-signal research, not the
  decision policy.
