# analog_mc v4.5 — investigation results and v5 plan inputs

Companion to [`V4_5_INVESTIGATION_PLAN.md`](V4_5_INVESTIGATION_PLAN.md). Captures the nine diagnostic investigations that ran on existing v4 canonical artifacts (five pre-registered, four added during execution), and the v5 experiment shortlist they motivate. **No new walk-forward runs.**

Session: 2026-05-22. ~3h compute (mostly Python analysis on cached run dirs).

## Headline

| # | Investigation | Verdict |
|---|---|---|
| V4.5.1 | A2.1 gate-signal validation | **val_crps is NOT a viable gate signal** — overlaps wins and losses. V5.1 (val_crps-gated A2.1) is dropped. |
| V4.5.2 | A2.1 analog autopsy | **Three distinct mechanisms (M1/M2/M3) discovered**, not one. Year-Herfindahl doesn't cleanly discriminate. |
| V4.5.3 | B1 β autopsy | **Leverage outliers are not the mechanism.** 1990-09-24 catastrophe is **magnitude over-correction** (+18% drift vs +12% realized). Fix: `b1_shrinkage` parameter. |
| V4.5.4 | COVID pool sufficiency | **Pool has 179 candidates with +30%/60d**, but BOTH matchers assign less-than-uniform mass to them. COVID is matcher-addressable, not tail-inflation-bound. |
| V4.5.5 | Cross-experiment mechanism map | Synthesis matrix: 5 of 10 A2.1 regressions have clear mechanisms; 4 are diffuse/noise. **V5.A.2 + V5.B** is the cheapest path that plausibly passes the bar. |
| V4.5.6 | A2.1 path-construction (added) | **Confirmed** Mode-3 mechanism: A2.1 paths converge at 40% the rate of v24 at 2022-03-01 (cum-σ-growth 3.56 vs 8.87). Fix: conditional corrwindow re-matching. |
| V4.5.7 | Tail-positive generality (added) | **Confirmed complementarity**: v24 and A2.1 strengths are anti-correlated by anchor. 10/15 anchors have at least one matcher finding the tail. **Strongest argument for ensemble.** |
| V4.5.8 | V5.A.2 ensemble preview (added) | **V5.A.2 alone FAILS the bar at every α** (failures-recovered stuck at 2/5). At α=0.5: regressions cut 10→6, 2008-10-03 catastrophe 7→41 in 90-band. V5.A.2 + V5.B must be stacked. |
| V4.5.9 | Drawdown-feature sanity (added) | V5.B drawdown feature works cleanly at 3/5 Cohort-2 anchors. **COVID is bimodal** — pulls both recovery and continuation precedents at extreme drawdown levels; needs a co-feature. 2012-03-14 is inert under drawdown (target at peak). |

## What changed vs the V4_RESULTS recommendation

V4_RESULTS recommended:
1. **Gated A2.1** (val_crps > 1.5× median) — **invalidated by V4.5.1**. No threshold separates wins/losses; the gate has no signal.
2. **Tikhonov-mixed A2.1** — survives as V5.A.1 but **demoted** to a stretch goal. V4.5.6/7 show path-level ensemble (V5.A.2) is cheaper and covers the same mechanisms.
3. **B5 (joint A2.1+B1)** — already promotion-tested and failed in v4; no v4.5 follow-up.
4. **Defer tail inflation to v5+** — **invalidated by V4.5.4**. Tail analogs exist in the pool; matchers fail to select them. Fix is matcher-side, not tail-inflation.

What survived: the qualitative recommendation to focus on matcher-level changes for the failure-anchor problem. What changed: the specific levers, sequencing, and the addition of three new candidate v5 experiments (V5.A.2, V5.B, V5.D) that didn't exist in the V4_RESULTS recommendation.

## Reading order

For someone picking this up cold:

1. **This file** — overview and v5 shortlist.
2. [`v4.5/_v4_5_1_gate_signal.md`](v4.5/_v4_5_1_gate_signal.md) — why val_crps doesn't work.
3. [`v4.5/_v4_5_2_analog_autopsy.md`](v4.5/_v4_5_2_analog_autopsy.md) — the three failure modes M1/M2/M3.
4. [`v4.5/_v4_5_7_tail_selection_scan.md`](v4.5/_v4_5_7_tail_selection_scan.md) — the v24/A2.1 complementarity argument.
5. [`v4.5/_v4_5_5_mechanism_map.md`](v4.5/_v4_5_5_mechanism_map.md) — full per-anchor classification and v5 candidate coverage.
6. **V5_EXPERIMENTS_PLAN.md** — formal v5 plan.
7. [`v4.5/_v4_5_3_b1_beta_autopsy.md`](v4.5/_v4_5_3_b1_beta_autopsy.md), [`_v4_5_4_covid_pool.md`](v4.5/_v4_5_4_covid_pool.md), [`_v4_5_6_path_construction.md`](v4.5/_v4_5_6_path_construction.md) — mechanism-specific details.

## V5 experiment shortlist (revised after V4.5.8/9)

V5.A.2 alone cannot pass the bar (V4.5.8 preview). V5.A.2 at α=0.5 is the **base configuration** that subsequent experiments stack on top of. The minimum stack to potentially pass: **V5.A.2 + V5.B**.

| # | Experiment | Cost | Addresses | Risk |
|---|---|---|---|---|
| **V5.A.2** | **Path ensemble v24 ⊕ A2.1 at α=0.5** (base) | **~1 day** (post-process existing npz) | M1, M2, M3 partial; 2008-10-03 catastrophe directly | Doesn't add failure recoveries; wins narrow slightly |
| **V5.B** | **Drawdown feature** stacked on V5.A.2 | ~4 days + 1 canonical | M4 at 2001-04, 2001-10, 2022-03 (sanity ✅); COVID bimodal (sanity ⚠️); 2012-03-14 inert (sanity ❌) | 4-D weight grid; COVID may need co-feature |
| **V5.D** | **B1 shrinkage** stacked on V5.A.2 | ~1 hour + canonicals at 4 shrinkage values | M5 (1990-09-24 over-correction) | Erodes B1's modest wins |
| **V5.B.2** | **Drawdown + vol-regime co-feature** (stretch) | +1 week if V5.B's COVID is weak | COVID disambiguation | Adds 5th feature, search blowup |
| **V5.A.3** | **Conditional corrwindow re-matching** (stretch) | ~1 week + 1 canonical | M3 directly at 2022-03-01, 2017-06-01 | New code path; risk of subtle bugs |
| V5.A.1 | Tikhonov mix `(1−α)d_eu + α d_cw` | 1 canonical w/ α-grid | M1/M2 partial | Subsumed by V5.A.2 |
| V5.C | Delay-coordinate Takens distance | 1 week + 1 canonical | M4 alternative to V5.B | Less interpretable than V5.B |

## Mechanism inventory — anchor by anchor

From [`v4.5/_v4_5_5_mechanism_map.md`](v4.5/_v4_5_5_mechanism_map.md):

| Anchor | A2.Δ% | B1.Δ% | Primary mech | V5 fix |
|---|---:|---:|---|---|
| 1990-09-24 | +7 | **+156** | B1 over-correction (M5) | **V5.D** |
| 1991-03-26 | +7 | +8 | None (diffuse) | V5.A.2 (averaging) |
| 2000-04-03 | −20 (win) | +3 | (win, no fix needed) | — |
| **2001-04-04** | **+41** | +40 | M4 tail under-select | **V5.B** |
| 2001-10-02 | −48 (win) | −16 (win) | M4 + M5 but wins | — |
| **2008-10-03** | **+122** | +8 | **M2** bimodal mis-match | **V5.A.2** |
| 2010-04-23 | −47 (win) | −24 (win) | (wins) | — |
| 2010-11-10 | +13 | 0 | Diffuse | V5.A.2 (averaging) |
| **2012-03-14** | +27 | −51 | M4 (A2 only) | **V5.B** |
| 2017-06-01 | +6 | +3 | M3 (mild) | V5.A.3 (stretch) |
| **2018-10-08** | +31 | −2 | **M1 + M2** | **V5.A.2** |
| 2020-03-16 | −19 (win*) | +4 | M1 + M4 (coverage collapse) | **V5.A.2 + V5.B** |
| **2022-03-01** | +34 | +1 | **M3 + M4** | **V5.A.3 + V5.B** |
| 2025-07-02 | −19 (win) | +32 | (B1 small) | — |
| 2026-02-19 | +13 | −2 | Diffuse | V5.A.2 (averaging) |

(* 2020-03-16 is a CRPS-win but coverage collapses 38→11.)

## Stop-list for v5

Things investigated and ruled out — v5 does not re-litigate:

- ❌ **val_crps as gate signal** (V4.5.1). Move on.
- ❌ **Year-Herfindahl, top-1, top-3 as gate signals** (V4.5.2). None discriminate cleanly enough.
- ❌ **Leverage-trimmed B1** (V4.5.3). Not the mechanism; shrinkage replaces it.
- ❌ **Tail inflation as primary COVID fix** (V4.5.4). Pool is rich; matcher selects wrong.
- ❌ **Gated A2.1 (any signal)**. Without a valid gate, gating is dropped from v5 entirely. V5 uses blends/ensembles instead.

## Open items not addressed by v5 shortlist

- The 4 "diffuse mechanism" A2.1 regressions (1990-09-24, 1991-03-26, 2010-11-10, 2026-02-19) have no clear lever. V5.A.2 path averaging is the only candidate — empirical question whether averaging clears them.
- B1 small-drift regressions (1991-03-26, 2008-10-03, 2025-07-02 — all with |drift| < 4%) don't cleanly fit M5. Shrinkage reduces them proportionally, marginal benefit.
- The relationship between A2.1's wins and ensemble averaging: V5.A.2 will attenuate 2010-04-23 (−47% CRPS), 2001-10-02 (−48%), 2020-03-16 (−19%) by averaging with v2.4. Need to verify the failure-anchor recovery count (currently 2/5) does not drop below 1/5 post-ensemble.

These risks are baked into the V5 plan as **early-termination conditions** on the canonical experiments.

## Deliverables manifest

```
docs/analog_mc/V4_5_RESULTS.md                   # this synthesis
docs/analog_mc/V4_5_INVESTIGATION_PLAN.md        # the plan that opened v4.5
docs/analog_mc/V5_EXPERIMENTS_PLAN.md            # formal v5 plan (next)
docs/analog_mc/v4.5/_v4_5_1_gate_signal.md
docs/analog_mc/v4.5/_v4_5_2_analog_autopsy.md
docs/analog_mc/v4.5/_v4_5_3_b1_beta_autopsy.md
docs/analog_mc/v4.5/_v4_5_4_covid_pool.md
docs/analog_mc/v4.5/_v4_5_5_mechanism_map.md
docs/analog_mc/v4.5/_v4_5_6_path_construction.md   # added during execution
docs/analog_mc/v4.5/_v4_5_7_tail_selection_scan.md # added during execution
docs/analog_mc/v4.5/_v4_5_8_v5a2_preview.md        # added during execution
docs/analog_mc/v4.5/_v4_5_9_drawdown_sanity.md     # added during execution

scripts/v4_5/validate_gate_signal.py
scripts/v4_5/analog_autopsy_a2.py
scripts/v4_5/b1_beta_autopsy.py
scripts/v4_5/covid_pool_sufficiency.py
scripts/v4_5/path_construction_inspection.py
scripts/v4_5/tail_selection_scan.py
scripts/v4_5/mechanism_map.py
scripts/v4_5/v5_a2_ensemble_preview.py
scripts/v4_5/drawdown_feature_sanity.py

results/analog_mc/data/v4_5_1_gate_signal.json
results/analog_mc/data/v4_5_2_analog_autopsy.json
results/analog_mc/data/v4_5_3_b1_beta_autopsy.json
results/analog_mc/data/v4_5_4_covid_pool.json
results/analog_mc/data/v4_5_5_mechanism_map.json
results/analog_mc/data/v4_5_6_path_construction.json
results/analog_mc/data/v4_5_7_tail_selection_scan.json
results/analog_mc/data/v4_5_8_v5a2_preview.json
results/analog_mc/data/v4_5_9_drawdown_sanity.json
```
