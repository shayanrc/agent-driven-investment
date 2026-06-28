# _027: expected-value (breakeven) entry threshold — does an EV-derived `p` cutoff help, or break the leaderboard top-10?

## TL;DR (mandatory)

Derived a **principled** entry threshold for every leaderboard model from its own confusion matrix — *trade a name only when the bet is positive expected value* — and back-tested it. Per model, a true positive earns `+target%` and a false positive loses `min(max_dd, 20%)`, so a trade is +EV iff its predicted-positive **precision ≥ `loss/(target+loss)` =: p\***. For each model I found the lowest **raw-`p` cutoff τ** on the **held-out eval split** whose precision clears p\*, then back-tested the **test/OOS** window gating entries at `p_raw ≥ τ` (champion rank/equal top-3 otherwise). **Two clean failure modes, one verdict.** (1) **8 of 22 models can't clear breakeven at any τ** — the EV rule **declines to trade them**, including the **deployed sp500_20 champion** (its +20%/25d/dd10 payoff needs precision ≥ 0.333, never reached) and **both bear2022 cells** — forgoing their real profit. (2) For the **14 with a τ, the gate is a NO-OP for 13** (`base_ret == gate_ret`, identical entries/DD): the champion's **top-3 picks already sit above the breakeven τ**, so "trade only +EV names" never removes a top-3 — *the highest-conviction names already are the highest-EV names*. The lone model where τ bites into the top-3 is **sp500_50**, and there it **hurts** (+12.5% → **−5.5%**, DD −9% → −18%). Gate beat baseline in **2/14 by noise** (+0.5, +1.1 pts); the 2 models above the top-10 bar were already there at baseline — **the gate created no new top-10 entry.** **An EV entry threshold adds nothing to a top-K strategy and disables profitable models — keep the champion ungated.** Single most important caveat: the EV model assumes a fixed FP loss; the back-test uses the real engine, and these are single per-cell windows.

## Spec (mandatory)

```yaml
models: 22 distinct gbdt cells on backtest_summary.csv (the 23rd is a malformed path, skipped)
ev_payoff_model:                 # used ONLY to derive tau (not the back-test)
  true_positive:  +threshold_pct           # hit the target barrier
  false_positive: -min(max_drawdown, 0.20) # "max DD / 20% whichever lower"
  breakeven_precision p*: loss/(target+loss)
threshold:
  scale: raw p_raw (continuous; calibrated p is plateaued — _021/_026)
  rule: lowest raw-p cutoff tau on the HELD-OUT eval split with precision(p_raw>=tau) >= p*
        (>=20 predicted-positive support guard); if none clears p* -> "no trade"
  fit_split: eval (leak-free)   |   backtest_window: test/OOS
strategy:                        # champion; ONLY the gate differs
  class: TopKDailyKellyLabelExit ; K: 3 ; selection_mode: rank ; sizing_mode: equal ; c: 1.0
  gate: --min-entry-p-raw tau    # candidate filter before top-3 selection
runners: scripts/backtests/ev_threshold.py (derive) + run_backtest_cell.py (--min-entry-p-raw, _026 infra)
benchmark: per-universe index (SPX / NDX / NIFTY500)
```

## Pipeline (mandatory)

`eval.csv → [ROC: precision vs raw-p] → breakeven τ (precision ≥ p*) → test.csv → [gate p_raw ≥ τ] → champion TopK=3 → engine → metrics`. τ is fit on eval; the gate is applied on the disjoint test/OOS window.

## Methodology (mandatory)

- **Why the gate is a no-op for a top-K strategy.** The champion already selects the **top-3 by `p` each day** — i.e. the highest-conviction = highest-`p_raw` names. The breakeven τ (a *precision* cutoff) lands *below* those names' `p_raw` for 13/14 models, so gating at τ never excludes a pick the strategy was going to make → identical fills, return, DD. A gate can only change a top-K strategy when τ is high enough to cut *into* the top-K — which happened only for **sp500_50** (50d, τ=0.145 straddles its top-3 `p_raw`), and removing/replacing those picks **lowered** return and doubled DD.
- **The 8 "no-trade" models.** Their predicted-positive precision never reaches p\* at any τ (rare-event payoffs: +20%/dd10 needs 0.333, +50%/dd25 needs 0.286). The EV rule therefore says *don't trade*, which **forgoes the baseline's real profit** (e.g. the sp500_20 champion is a deployed money-maker). For the bear2022 cells the refusal is protective, but for the live champions it is pure opportunity cost.
- **Leak-free**: τ fit on eval, scored on test. The +target/−loss payoff is a *threshold-derivation device only*; the back-test uses actual price paths + horizon/DD exits.

## Results (mandatory)

Base vs EV-gate, 14 tradable models (`backtest_results.csv`; figure `figs/_027_base_vs_gate.png`):

| model | τ | base ret | gate ret | Δ | base DD | gate DD |
|---|--:|--:|--:|--:|--:|--:|
| sp500_up_50pct_50d_dd25pct_agentloop | 0.145 | +12.5% | **−5.5%** | **−18.0** | −9.3% | −17.5% |
| sp500_up_50pct_200d_dd25pct_aligned_cbagent | 0.236 | +92.5% | +93.5% | +1.1 | −13.5% | −13.5% |
| sp500_up_50pct_200d_dd25pct_aligned_agent | 0.257 | +77.9% | +78.4% | +0.5 | −13.5% | −13.5% |
| sp500_up_40pct_200d_dd20pct_aligned_cbagent | 0.383 | +177.7% | +177.7% | 0.0 | −19.6% | −19.6% |
| sp500_up_40pct_200d_dd20pct_aligned_agent | 0.356 | +30.6% | +30.3% | −0.3 | −18.3% | −18.3% |
| russell…40pct_200d_cbagent | 0.304 | +129.1% | +129.1% | −0.1 | −16.4% | −16.4% |
| russell…40pct_200d_agent | 0.291 | +79.8% | +79.8% | −0.0 | −24.3% | −24.3% |
| russell…40pct_100d_v14p1 | 0.129 | +69.6% | +69.6% | 0.0 | −31.4% | −31.4% |
| russell…50pct_200d_cbagent | 0.274 | +106.5% | +106.0% | −0.4 | −17.6% | −17.6% |
| russell…50pct_200d_agent | 0.251 | +96.9% | +96.5% | −0.4 | −19.4% | −19.4% |
| russell…50pct_200d_v14p1 | 0.261 | +49.1% | +49.0% | −0.2 | −21.6% | −21.6% |
| nasdaq…10pct_50d_b_acceptance | 0.274 | +37.3% | +37.3% | 0.0 | −5.9% | −5.9% |
| nasdaq…10pct_50d_revalidation | 0.288 | +12.3% | +12.3% | 0.0 | −5.2% | −5.2% |
| nifty…10pct_25d | 0.003 | −7.5% | −7.5% | 0.0 | −15.9% | −15.9% |

**Gate beat baseline in 2/14** (both by ≤1.1 pts, no DD change — noise). **No-op in 11/14.** **Materially negative in 1/14 (sp500_50).** Mean Δ −1.3%, median 0.0%. Two models clear the top-10 total-return bar (1.205) — sp500 +40%/200d cbagent, russell +40%/200d cbagent — but **at identical base and gate values** (the gate is a no-op there), so the placement is the *baseline model's*, not the threshold's. **8 further models** (nasdaq +40%/50d; 4 nifty +20/30/50%; sp500_20 champion; sp500_20 & sp500_50 bear2022) were **declined by the EV rule** (no τ clears breakeven → flat).

## Caveats (mandatory)

- The EV model assumes every FP loses exactly `min(max_dd,20%)` and every TP earns `+target` — a derivation simplification; the back-test uses real paths.
- eval→test regime shift can move the `p_raw` distribution, so a τ tuned on eval may gate differently OOS (here it mostly didn't bite).
- Leaderboard placement is **vintage-indicative** (these are fresh re-runs on the current cache; committed rows are older vintages).
- The 8 "no-trade" models' forgone profit is read from their (positive) committed/baseline returns, not separately re-back-tested as flat.
- Single per-cell windows; gross of costs.

## Verdict (mandatory)

**An expected-value entry threshold neither improves the models nor breaks the leaderboard top-10.** It is **structurally a no-op for a top-K strategy** — the top-K picks already are the highest-EV names, so a +EV gate rarely removes one (13/14 unchanged) — and it is **actively harmful where it bites** (sp500_50: +12.5% → −5.5%) or **where it declines a profitable model** (8 cells, including the deployed sp500_20 champion). **Keep the champions ungated.** Combined with `_026` (a raw-`p` gate is a bull-only high-beta amplifier), this closes the entry-threshold line: **thresholds on `p` add no alpha to top-K selection** — the selection already encodes the conviction the threshold is trying to impose. The `ev_threshold.py` derivation is retained as reusable infra; **no rows added to the canonical registry** (the gate is a no-op / vintage-indicative).
