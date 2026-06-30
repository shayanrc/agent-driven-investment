# _028: cross-model consensus strategy — bull-window winner, genuinely-OOS bear-regime failure

## TL;DR (mandatory)

A cross-model **consensus** strategy: each trading day pool the five `_019` forward-log models' **top-5** picks, buy the **single most-voted** stock (tie → highest summed `p_raw`), **one stock/day**, size each new position at `max_alloc` of equity, hold until **+target** or **−stop** on the close (path-honest, no horizon). Swept a 2×2 of (alloc 25%/33%) × (barriers +30%/−15%, +40%/−20%); **A = 25/+30/−15** and **C = 33/+30/−15** (the tight-target corners) were carried forward. **In a bull it looks excellent and is the best *risk-adjusted* selector**: on the 48-day all-5 window **C +46.5%** beats all five constituents (A/C beat all five whenever the target is tight); over the full Jan→Jun-2026 forward window **C +39.5%** (Sharpe 1.60, Calmar 7.5) vs SPX +7.1%; over a 1-year **clean-OOS** June'25→June'26 breadth-ramp **C +142%** (Sharpe 2.12) vs SPX +23.7%, **with stops finally firing** (win-rate ~60%). The edge is **loser-avoidance** — the vote never takes the idiosyncratic blow-ups that give each solo model 1–5 stop-outs. **But the decisive test kills it.** On the only **genuinely-out-of-sample bear** (2022, reachable solely via the two `bear2022` retrains whose training ends pre-2021-12-21 → a 2-model sp500-only consensus), the consensus **loses A −52.9% / C −57.5%** (SPX −20.5%), max DD −55%/−60%, Sharpe ≈ −2, **win-rate ~11% with 23–25 stop-outs** — **2.5–2.9× worse than the index, and worse than sp500_50 solo (+15.9%)**. The bull-winning tight-target config is the **worst** in the bear (wide-target B/D lose "only" −30%/−33%). **Verdict: the consensus is a bull-regime momentum amplifier whose edge inverts catastrophically in a bear; ungated it is undeployable. NOT promoted.** Single most important caveat: the bull windows are short and single-regime, and the bear test is a 2-model sp500-only consensus with no regime gate (the live SMA200 gate was ~80% risk-off in 2022 → mostly cash).

## Spec (mandatory)

```yaml
strategy: cross-model daily consensus (a backtests-area prototype; NOT a trading_strategies class)
  selection: each day pool the models-present top-5; winner = max #models voting (tie -> highest Σ p_raw);
             one stock/day; skip if already held
  sizing:    max_alloc of current equity per new position (cap; one add/day), capped by available cash
  exits:     +target (take-profit) / -stop (stop-loss) on CLOSE (path-honest), hold-to-barrier (NO horizon);
             open positions marked-to-market at end
  anchor:    signal-day close ; capital $100k ; gross of costs ; benchmark ^SPX buy-hold
variants:    A 25%/+30%/-15% · B 33%/+40%/-20% · C 33%/+30%/-15% · D 25%/+40%/-20%
models (forward log): sp500_50, sp500_20 (deployed) + russell_50_200, russell_40_100, nasdaq_40_50 (candidates)
prediction sources:
  bull 48d all-5 + 2x2 + constituent A/B : results/backtests/data/forward_predictions_log.csv (committed OOS log)
  complete A/C (full forward Jan->Jun)   : same log, breadth = models present each day
  june->june 1yr clean OOS               : faithful re-inference (build_scores_multi, 2016 warmup), each model OOS-masked at test_end+1
  2022 bear OOS                          : the two bear2022 retrains' predictions/test.csv (train ends < 2021-12-21 => genuinely OOS)
runners: scripts/backtests/consensus_backtest.py (+ consensus_june_check.py, consensus_bear_check.py)
```

## Pipeline (mandatory)

`per-model top-5 (models present) → daily cross-model vote (tie: Σ p_raw) → 1/day buy (max_alloc, if not held) → triple-barrier +target/−stop on close (hold) → equity / Sharpe / Calmar / maxDD vs ^SPX`. OOS-validity gate: a model contributes on date `d` only if `d ≥ its test_end+1` (clean-OOS); for the bull 48-day window all five are clean.

## Methodology (mandatory)

- **OOS-validity drives everything.** Per the GBDT registry, the five cells go out-of-sample at very different dates — russell_50_200 **2024-10-04**, russell_40_100 **2025-05-14**, sp500_50 & nasdaq_40_50 **2026-03-13**, sp500_20 **2026-04-18** — so a *clean* all-5 consensus only exists from 2026-04-18. The 48-day all-5 window (Apr 20–Jun 26) and the breadth-ramped longer windows (each model masked to ≥ its `test_end+1`) are clean; a naive all-5 run over any earlier window is leaky.
- **Leakage quantified.** Over June'25→June'26, the OOS-clean breadth-ramp gives A +116% / C +142%; the *same* window with all 5 voting every day (sp500/nasdaq scoring their own in-sample data) inflates to **+257% / +329%** — ~2× — which is exactly why a naive 5-model run over the 2024-10 window (which 3 of 5 models would be ~90% in-sample for) is invalid and was not run.
- **The bear test is necessarily 2-model.** The only bear outside all models' training is 2022; only the two sp500 `bear2022` retrains (train 2016→pre-2021-12-21, test = the 2022 bear) make it OOS. The russell/nasdaq candidates have no pre-2022 retrain, so the bear consensus is sp500-only, 2 models — a downside stress test, not the 5-way vote. No regime gate is applied (the raw strategy).
- **Why the inversion.** The strategy is a breakout/momentum amplifier (buy the agreed high-conviction name, target +30%). In a bull, breakouts hit target and the vote avoids idiosyncratic losers. In a sustained bear, breakouts fail, the tight −15% stop triggers repeatedly (25 stops, win-rate 11%), and the strategy bleeds — death by a thousand cuts. The wider +40%/−20% (B/D) churns less (16–18 entries) and bleeds less, so the tight-target config that *wins* the bull is *worst* in the bear.

## Results (mandatory)

**Bull — 48-day all-5 window (2026-04-20→06-26), consensus 2×2 + does it beat all 5 constituents** (SPX +3.3%):

| variant | total | max DD | beats all 5 constituents? |
|---|--:|--:|---|
| A 25/+30/−15 | +34.4% | −8.8% | **yes** |
| C 33/+30/−15 | **+46.5%** | −10.9% | **yes** |
| B 33/+40/−20 | +38.7% | −11.4% | no (sp500_20 solo +56%) |
| D 25/+40/−20 | +29.0% | −9.1% | no (sp500_20 solo +50%) |

(Tight target → consensus beats every constituent; wide target → a single hot model out-earns the diversified vote.)

**Bull — complete forward window (2026-01-02→06-26, 121d) & 1-year clean-OOS (2025-06-05→2026-06-26, 266d):**

| window | variant | total | CAGR | Sharpe | max DD | win-rate | SPX |
|---|---|--:|--:|--:|--:|--:|--:|
| Jan→Jun 2026 (121d) | A | +25.0% | +59% | 1.29 | −11.1% | 55% | +7.1% |
| Jan→Jun 2026 (121d) | C | +39.5% | +100% | 1.60 | −13.3% | 55% | +7.1% |
| Jun'25→Jun'26 clean (266d) | A | +116.0% | +107% | 2.11 | −15.0% | 60% | +23.7% |
| Jun'25→Jun'26 clean (266d) | C | +142.1% | +131% | 2.12 | −19.5% | 62% | +23.7% |

(Leaky all-5 sensitivity over the 266d window: **A +257% / C +329%** — ~2× the clean result, the in-sample inflation. The clean window is russell-dominated until 2026-03; stops fire from the Jan–Mar russell period, incl. an ASTS whipsaw and a SMMT −34% gap through a −15% close-stop.)

**BEAR — genuinely-OOS 2022 (2021-12-21→2022-10-19, 209d, SPX −20.5%), 2-model sp500 consensus + constituents:**

| selector | variant | total | max DD | Sharpe | win-rate | entries (tgt/stop) |
|---|---|--:|--:|--:|--:|--:|
| **consensus** | A 25/+30/−15 | **−52.9%** | −55.2% | −1.98 | 11% | 33 (3/25) |
| **consensus** | C 33/+30/−15 | **−57.5%** | −59.5% | −2.20 | 12% | 30 (3/23) |
| **consensus** | B 33/+40/−20 | −29.9% | −40.2% | −0.65 | 9% | 16 (1/10) |
| **consensus** | D 25/+40/−20 | −33.3% | −43.0% | −0.82 | 8% | 18 (1/11) |
| sp500_50 solo | A | **+15.9%** | −20.5% | 0.66 | 38% | 21 (6/10) |
| sp500_50 solo | C | +12.3% | −21.5% | 0.55 | 29% | 21 (5/12) |
| sp500_20 solo | A | −30.7% | −36.2% | −0.82 | 18% | 23 (3/14) |
| sp500_20 solo | C | −21.3% | −29.6% | −0.44 | 26% | 23 (5/14) |

The bull ranking **inverts**: A/C (bull winners) are the worst (−53%/−58%); B/D (wide target) lose less; and the **consensus is far worse than sp500_50 solo (+15.9%)** — the 2-model agreement over-concentrated in high-beta names that cratered, churning through 25 stop-outs at ~11% win-rate. Figure: `../../results/backtests/_028_consensus_backtest/figs/_028_consensus_bear2022.png`.

## Caveats (mandatory)

- **Bull windows are short and single-regime** (2026 H1 / mid-2025→2026 bull); the CAGR figures annualize a bull and must not be read literally.
- **The bear test is a 2-model, sp500-only consensus**, not the 5-way cross-universe vote (no pre-2022 russell/nasdaq retrains exist). It is the only genuinely-OOS bear available; the qualitative failure is robust but the exact magnitude is specific to the 2 sp500 cells.
- **No regime gate in any of these runs.** The deployed champions use an SMA200 gate that was ~79–82% risk-off across the 2022 bear (`_017`/`_018`) — i.e. mostly cash — which would have avoided most of this loss. These numbers are the *ungated* strategy.
- **Gross of costs.** The bear's high churn (33 entries, 25 stops) would incur real transaction costs, worsening it further (cf `_015`).
- **Concentration** — 7–12 distinct names in the bull; small effective N.

## Verdict (mandatory)

**The cross-model consensus is a bull-regime momentum amplifier, not an all-weather strategy.** When the target is tight it is the best *risk-adjusted* selector in a bull — it beats every constituent and the index (C +46.5% on the matched 48d, +142% on the 1-year clean OOS, Sharpe ~2.1) via genuine **loser-avoidance**. But on the **only genuinely-out-of-sample bear** it **fails catastrophically** (A −52.9% / C −57.5% vs SPX −20.5%, Sharpe ≈ −2, ~11% win-rate), the tight-target config that won the bull being the *worst*, and the consensus underperforming even sp500_50 solo. This is the same bull-only / regime-fragility pattern as `_016`/`_026`; the consensus does not change it. **NOT promoted; documented as a negative robustness result.** Any future deployment would require the SMA200 regime gate (a risk-reducer that in 2022 sat ~80% in cash, `_017`/`_018`) plus transaction-cost accounting — and even then the consensus shows no demonstrated advantage over the gated single-model champions. The harness (`consensus_backtest.py` + june/bear sidecars) is retained as reusable infra; **no rows added to the backtest registry** (prototype windows, vintage- and regime-specific).

## Follow-up (2026-06-30): ≥50%-of-panel vote gate

A reviewer question — *was anything traded unless ≥50% of the models voted for it?* — exposed that every run above trades the **plurality** winner (`argmax` votes, tie → Σ`p`), with **no minimum-vote floor**. Audit over the forward-log inputs (122 trading days): the traded winner had **≥3 of 5 votes on only 68/122 days (56%)**; on the other 54 days (44%) it had just **2/5** — a plurality, not a majority. It was always ≥50% of the models *present* that day, and the strict 48-day all-5 window happened to be ≥3/5 throughout, but that was emergent, not enforced.

Added an explicit `--min-votes N` floor to `consensus_june_check.py`: a day trades only if its most-voted name clears **N of the 5 models**, else no entry (hold). Re-ran A & C on a clean same-window comparison (**2025-06-02→2026-06-01, 251d, SPX +28.0%**), ungated (`min_vote 1`) vs gated **≥3** (≥50% of the panel), masked + leaky:

| version | variant | total | CAGR | Sharpe | max DD | win-rate | entries (tgt/stop) |
|---|---|--:|--:|--:|--:|--:|--:|
| clean · ungated | A 25/30/15 | +89.6% | +90% | 1.98 | −15.0% | 54% | 29 (14/12) |
| clean · ungated | C 33/30/15 | +104.7% | +105% | 1.85 | −19.5% | 56% | 28 (14/11) |
| clean · gate ≥3 | A 25/30/15 | +40.6% | +41% | 2.12 | −5.7% | 100% | 7 (4/0) |
| clean · gate ≥3 | C 33/30/15 | +53.7% | +54% | 2.43 | −7.1% | 100% | 7 (4/0) |
| leaky · ungated | A 25/30/15 | +225.1% | +227% | 3.36 | −15.4% | 67% | 39 (24/12) |
| leaky · ungated | C 33/30/15 | +275.9% | +278% | 3.12 | −19.6% | 65% | 34 (20/11) |
| leaky · gate ≥3 | A 25/30/15 | +203.0% | +204% | 3.41 | −15.4% | 66% | 32 (19/10) |
| leaky · gate ≥3 | C 33/30/15 | +228.6% | +230% | 2.94 | −24.0% | 64% | 31 (18/10) |

**Reading.** (1) **Masked + gate is not a fairer strategy — it is a much shorter one.** Trade-days collapse 251→48, entries 28→7: because only 2 russell models are OOS before 2026-03-13, a 3-vote majority is *impossible* until then, so the gated masked arm sits in cash ~9 months and trades only Mar→Jun 2026 — a clean bull stretch (4/4 targets, 0 stops, 100% win, DD −6/7%). The pristine risk stats are a 7-trade single-quarter artifact; return roughly halves (mostly out of market). (2) **Leaky is the only arm where the gate is active all year** (235/251 days clear ≥3 votes); there it barely moves anything (A 225%→203%, C 276%→229%), and C's max DD actually *worsens* (−19.6%→−24.0%) from name concentration (12→10 names). (3) **The gate reduces participation; it does not add edge.** On the honest (masked) arm it merely defers all trading to the final bull quarter — the same bull-only fragility the verdict above already flagged. The real risk control remains the SMA200 regime gate, not a vote threshold.

Tooling: `consensus_june_check.py --min-votes 1,3 --start 2025-06-01 --end 2026-06-01` (one shared inference pass across both gates; all 5 self-checks PASSED ~1e-8). Results: `results/backtests/_028_consensus_backtest/june_minvote_1and3_2025-06-01_2026-06-01.csv`.
