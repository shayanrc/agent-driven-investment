# _031 — nifty500 post-pop event study: fade the strength, don't buy the dip

**Headline:** Across **11,753** nifty500 events (2010–2026) where a stock's intraday
high cleared the previous close by **>10%**, the forward tape is a *fade-and-drift*
pattern, not a momentum-continuation one. The immediate few days mean-revert
(sub-46% positive); a positive edge only emerges by +50d and is **entirely
tail-driven** (a minority of runaway winners). Because the edge lives in the tail,
**every rule that caps the upside destroys the mean** — buying the dip
*underperforms* buy-and-hold (the dip is a weakness filter that excludes the
moonshots), stop-losses strictly hurt (they whipsaw out of drawdown-enduring
winners), and take-profit targets cut drawdown + lift the win rate only by
trading away the mean. The one lever with real value is **harvesting the
transient run-up** (median run-up ≫ median endpoint at every horizon), i.e.
exiting *into strength*, not out of weakness.

Reproduce: `uv run python -m scripts.backtests.nifty500_pop_study`
(writes `results/backtests/data/nifty500_pop_events.parquet`).

---

## 1. Motivation & definitions

A recurring trader intuition is that a big, volume-backed up-day signals
continuation ("buy the breakout") or, conversely, an exhaustion top to fade. This
study characterizes the *actual* forward distribution on the nifty500 universe,
using the deep-history nse_equities cache built for the F18-IN work (back-extended
to 2010–2015 for most names).

- **Event:** intraday `adj_high` clears the previous `adj_close` by **>10%**
  (`(adj_high[t] − adj_close[t−1]) / adj_close[t−1] > 0.10`). All prices
  split-adjusted (factor `adj_close/close` applied to O/H/L). **11,753 events**
  (each requires ≥21 prior bars to form the volume baseline; the first
  interactive pass, which skipped that filter, counted 11,976 — immaterial).
- **Returns** measured from a stated entry (event **close**, or the **next day's
  open**), to the close N trading days later.
- **Max drawdown (MDD):** running-peak-relative on the close path, entry = initial
  peak (so it is ≤ 0; the standard "worst peak-to-trough" figure).
- **Max run-up (RU):** peak favorable excursion vs entry (≥ 0).
- **Volume spike:** event-day volume ÷ **trailing-20d median** volume.

Data: 1,109,656 bars, 500 tickers, 2010-01-04 → 2026-07-10.

---

## 2. Unconditional forward performance (from event close)

| Horizon | N | Mean ret | Median ret | % positive | Mean MDD | Median MDD |
|---|--:|--:|--:|--:|--:|--:|
| +1d | 11,753 | +0.1% | −0.3% | 46.3% | −1.8% | −0.3% |
| +5d | 11,743 | +0.2% | −0.8% | 45.1% | −6.4% | −5.0% |
| +10d | 11,727 | +1.0% | −0.5% | 47.8% | −9.5% | −7.8% |
| +20d | 11,672 | +2.5% | +0.1% | 50.4% | −13.4% | −11.2% |
| +50d | 11,508 | +8.4% | +2.3% | 54.3% | −20.3% | −17.4% |
| +200d | 11,161 | +39.4% | +14.9% | 61.7% | −35.5% | −31.8% |

- **Short-term mean reversion:** +1d/+5d medians are negative and <46% positive —
  the typical pop gives back over the next few days.
- **Heavily right-skewed:** mean ≫ median everywhere. A minority of continuation
  runners carry the average while the median event is roughly flat.
- **Drawdowns grow with horizon** — median −11% (20d), −17% (50d), with worst
  cases near −98% (distressed names that popped then collapsed).
- **Bigger pops are worse:** splitting the 20d horizon by pop size, the >30%
  bucket has a **−9.45% mean** (blow-up tail) vs mildly-positive 10–15% pops. The
  largest single-day spikes carry the worst negative skew.

---

## 3. Volume conditioning — a defining feature, not a signal

Volume spikes are nearly ubiquitous on these pops (median pop day = **6.76×** its
trailing-20d median volume):

| Threshold | Share of events |
|---|--:|
| ≥ 1.5× | 90.7% |
| ≥ 2× | 85.6% |
| ≥ 3× | 76.6% |
| ≥ 5× | 60.9% |
| ≥ 10× | 36.4% |

Because ~86% already clear 2×, "high volume" barely discriminates. Conditioning
the forward return on the volume bucket is **non-monotonic** — the best 50d
outcome sits in the *modest* 1.5–3× bucket, not the extreme >5× bucket (giant
volume looks more like a climax). Filtering to **>5× volume** (7,147 events)
leaves the aggregate essentially unchanged vs the full sample (+50d mean +8.9% vs
+8.4%), confirming volume is a *property* of the move, not a selective edge. The
rest of the study uses the >5×-volume subset for a cleaner "conviction" cohort.

---

## 4. Path shape: run-up, drawdown, and their ordering (>5× vol)

| Horizon | N | Mean ret | Median ret | Mean RU | Median RU | Mean MDD | Median MDD |
|---|--:|--:|--:|--:|--:|--:|--:|
| +5d | 7,137 | +0.54% | −0.96% | +5.18% | +2.02% | −5.91% | −4.96% |
| +20d | 7,071 | +2.97% | −0.11% | +12.20% | +6.63% | −12.38% | −10.79% |
| +50d | 6,951 | +8.92% | +1.94% | +23.96% | +13.11% | −19.11% | −16.74% |

**The endpoint understates both the opportunity and the risk.** Over 50 days the
median event *ends* +1.9% but *touches* +13.1% (median RU) and sinks −16.7%
(median MDD). Run-up and drawdown are roughly symmetric in magnitude at every
horizon, so a naive hold captures little of the excursion while eating the full
drawdown — **the gain is transient and must be harvested.**

**Ordering of the trough vs the peak** (which comes first within the window):

| Horizon | N | DD before RU | RU before DD | Median day DD | Median day RU |
|---|--:|--:|--:|--:|--:|
| +5d | 7,137 | 32.5% | 67.5% | 4 | 2 |
| +20d | 7,071 | 32.2% | 67.8% | 15 | 9 |
| +50d | 6,951 | 34.1% | 65.9% | 36 | 26 |

In ~**two-thirds** of events the run-up peak arrives *before* the drawdown trough
("pump-then-dump"): the favorable excursion is front-loaded, the risk back-loaded.
This is the actionable asymmetry — the stock tends to give you its best price
early, then bleed.

---

## 5. Entry timing: event close vs next-day open

Entering at the **next day's open** is marginally *worse* than at the event close
across horizons (50d: mean +8.2% / median +1.4% / 52.7% pos vs close-entry +8.9% /
+1.9% / 53.6%), and clearly worse short-term (5d median −1.4% vs −1.0%) — the
overnight gap and continued fade mean you buy after some of the pop has already
reverted. The long tail is real but expensive: at **+200d** the next-open trade is
mean **+36.9%** / median **+13.5%** / **61.4%** positive — but the median event
draws down **−31%** and reaches only ~⅓ of its +37.5% median peak. Multi-month
positive drift (partly nifty500 survivorship, §8), bought with a ~30% typical
mid-trade loss.

---

## 6. Strategy battery (>5× vol, 200d window, 6,652 events, next-open entry)

### 6a. Buy-the-dip GTC ladder vs buy-and-hold

| Strategy | Participation | Mean ret | Median ret | Median RU | Median MDD | Uncond. mean |
|---|--:|--:|--:|--:|--:|--:|
| **Buy next-open, hold** | 100.0% | **+36.9%** | +13.5% | +37.6% | −31.0% | **+36.9%** |
| Dip ladder −5% | 86.2% | +33.6% | +11.7% | +35.1% | −30.1% | +28.9% |
| Dip ladder −10% | 72.7% | +29.5% | +9.3% | +31.2% | −29.3% | +21.4% |
| Dip ladder −15% | 59.9% | +26.1% | +7.0% | +28.6% | −28.5% | +15.6% |
| Dip ladder −20% | 48.3% | +22.9% | +6.4% | +25.3% | −27.7% | +11.1% |

*(cond. return shown; uncond. = fill-rate × cond. mean, non-fills = no trade.)*

**Buy-and-hold beats the dip ladder on every metric, and the smoking gun is
adverse selection.** The 13.8% of events that *never* dip 5% — exactly what the
ladder sits out — return **+104.5% mean / +66.0% median / 98.8% positive** over
200d. The events that *do* dip 5% (what the ladder trades) return only +26.1% /
+5.7%. **The pullback is a weakness filter:** requiring a discount systematically
excludes the strongest names — the moonshots that pop and never look back.
Deeper limits do not help (adverse selection worsens + fill rate collapses), and
the ladder barely reduces drawdown (−30% vs −31%). Fill rates are high and early:
86% dip ≥5% (median day 4), 73% dip ≥10% (median day 14). A next-day-only −2%
limit fills 60.1% of the time — but filling the dip doesn't improve the trade.

### 6b. Stop-losses — strictly harmful

| Stop | Stopped % | Mean ret | Median ret | % positive | Worst |
|---|--:|--:|--:|--:|--:|
| **none (hold)** | 0% | **+36.9%** | +13.5% | 61.4% | −98.2% |
| hard −10% | 74.2% | +16.4% | −10.0% | 25.2% | −10.0% |
| hard −20% | 49.7% | +27.0% | −15.5% | 45.4% | −20.0% |
| hard −30% | 30.7% | +32.6% | +8.3% | 55.7% | −30.0% |
| trail −10% | 100.0% | +0.8% | −3.8% | 35.0% | −10.0% |
| trail −20% | 94.8% | +10.9% | −4.8% | 41.8% | −20.0% |
| trail −30% | 66.6% | +24.1% | +0.3% | 50.4% | −30.0% |

Every stop underperforms holding; tighter = worse. These are high-volatility names
whose *natural* drawdown is ~30% (§4), so any stop tighter than that catches
routine noise — a −10% hard stop fires on 74% of trades near the local bottom
(25% win rate). Trailing stops whipsaw worse: a −10% trail exits **100%** of
trades by day 7. Stops *do* cap the −98% tail, but the edge lives in
drawdown-enduring winners, so ejecting mid-dip forfeits far more upside than the
tail it saves.

### 6c. Take-profit targets — cut drawdown, cap the mean

| Target | Hit % | Mean ret | Median ret | % positive | Median MDD |
|---|--:|--:|--:|--:|--:|
| **none (hold)** | — | **+36.9%** | +13.5% | 61.4% | −31.0% |
| +10% | 85.0% | +4.2% | +10.0% | 85.3% | −8.8% |
| +20% | 72.6% | +8.1% | +20.0% | 75.5% | −15.7% |
| +30% | 61.1% | +11.2% | +30.0% | 68.9% | −20.3% |
| +50% | 43.9% | +16.4% | +21.9% | 64.1% | −25.3% |

Targets are the mirror image of stops: they **improve the median trade and slash
drawdown** but **cap the mean** (the tail is capped). The verdict splits by what
you optimize — on median-return-per-drawdown a +30% target scores **1.48** vs
**0.44** for hold; on the *mean* (what compounds across a portfolio), hold wins
outright because capping the upside is arithmetically unable to raise a
tail-driven mean. **No single target beats hold on both return and drawdown.**

---

## 7. Synthesis — the through-line

The return distribution is **tail-dominated**: a minority of runaway winners
(the +100%+ non-dippers) carry the entire mean. Every conclusion follows from
this one fact:

1. **Anything that caps the upside sacrifices the mean** — dips (miss the
   moonshots), stops (eject mid-dip), tight targets (cap the runners).
2. **The transient run-up is the only edge worth harvesting.** Median run-up ≫
   median endpoint at every horizon, and the peak precedes the trough ~⅔ of the
   time. Value is created by **exiting into strength**, not out of weakness.
3. **"Conviction" filters (volume, bigger pop) don't help** — high volume is a
   defining feature of the move (86% clear 2×), and bigger pops carry *worse*
   negative skew.
4. **The dual objective (lower DD *and* higher return than hold) is unachievable
   with a single exit rule.** The only structure that can approach it is a
   **partial scale-out** — book a slice at a +20–30% target (locks the median
   gain, cuts realized drawdown on that slice) while a **runner rides uncapped**
   to retain tail exposure. This is the standing follow-up.

---

## 8. Caveats & limitations

- **Not a backtest.** No transaction costs, position sizing, capital efficiency,
  or overlap handling (events overlap; GTC capital is tied up waiting). Per
  project convention these stay out of the study. Treat all figures as a
  *distributional characterization*, not a tradeable P&L.
- **Survivorship.** The nse_equities universe is roughly *current* nifty500
  membership, so the multi-month drift — especially the +104% non-dip cohort — is
  flattered by excluding names that popped then delisted / fell out of the index.
  Treat the absolute long-horizon and moonshot numbers as optimistic; the
  *relative* verdicts (hold > ladder; stops hurt; targets trade mean for
  consistency) are robust since every arm faces the identical universe.
- **Fill assumptions.** Limit/stop fills assumed *at* the level (conservative on
  gap-throughs); target fills at the target on first intraday touch.
- **Regime pooling.** Events are pooled across 2010–2026 (multiple regimes); the
  tail-driven mean will be regime-dependent (cf. the F18/regime work in
  `_016`–`_018` — up-move edges collapse in sustained bears).

---

## 9. Follow-ups

1. **Partial scale-out** (sell ⅔ at +25%, trail the remainder) vs hold — the only
   structure that can plausibly lower drawdown without surrendering the mean (§7.4).
2. **Profit-side exits generally** — "exit on first close above +X%", time-boxed
   harvest at the median run-up day (~day 9 at 20d) — quantify the run-up capture.
3. **Survivorship-corrected universe** — re-run on point-in-time index membership
   (or a broader all-listed set) to de-bias the long-horizon drift.
4. **Regime split** — condition on the SMA200 index regime at the event date; the
   tail-driven drift almost certainly concentrates in bull windows.
5. **Liquidity floor** — require a ₹-turnover minimum to drop the illiquid names
   inflating the volume-ratio mean (418× artifact) and sharpening the volume cut.
