# _290 — nifty500 deploy-shaped fund cells + second-window replication (V1.11)

**Plan:** `docs/gbdt/V1.11_nifty500_fund_cells_plan.md` · **Tasks:** #60, #61 · **Branch:** `nifty500-fund-cells-finetune`
**Follows:** `_289` (V1.10 canonical scan + finetune; picks up `V1.10_TBD` items 1–2).

**One-liner:** The F18-IN edge **replicates on an independent second window** (Task B: the
+50%/200d matched A/B run on test 2023-07→2024-06 has fund beating base at EVERY K) — the
first fundamentals signal in the project's F17/F18 arc to survive independent-window
replication. And the deploy-shaped **+10%/25d ffund finetunes cleanly** (deep+bagging wins
every K on test, backtest +11.6% vs basket +6.2%, raw top-K +37.4%), while **+20%/100d's
baseline stands** (high base @1 already maxed — the recipe's headroom rule called both).

## Task B (#61) — second-window replication of the +50%/200d ffund win

Matched base-vs-fund single fit (default HP, only `features.candidates` differs), independent
EARLIER window: train 2015→2021-06 · val →2022-06 · eval →2023-06 · **test 2023-07-01→2024-06-30**
(snapshot 2025-06-30; 200d labels end ~2025-04, never touching the 2025-07+ backtest window).

| arm | base_rate | R-p@1 | @3 | @5 | @10 | @20 |
|---|--:|--:|--:|--:|--:|--:|
| fbase_w2 | 0.439 | 0.572 | 0.510 | 0.508 | 0.510 | 0.523 |
| **ffund_w2** | 0.439 | **0.650** | **0.582** | **0.587** | **0.585** | **0.581** |
| delta | | +0.078 | +0.071 | +0.079 | +0.075 | +0.057 |

**REPLICATES — fund beats base at every K.** Window-2 prevalence is much hotter (0.439 vs
the canonical test's 0.121, the 2023-24 rally), so levels aren't comparable across windows;
the sign and the every-K consistency are the finding. Combined with `_289` (canonical test
2024-07→2025-06, fund wins every K) this is two independent test windows + the 2025-26
backtest, all agreeing.

## Task A (#60) — finetune the two deploy-shaped fund cells

Canonical recipe: controlled baseline (`all/d6`) → deep+bagging sweep on val+eval →
select on val → confirm on test. Both cells show the eval↔test prevalence inversion →
selected on val.

### +10%/25d ffund — ADOPT `d8 / mcw1 / ss0.7 / cs0.7`
| config | test @1 | @3 | @5 | @10 | @20 |
|---|--:|--:|--:|--:|--:|
| baseline all/d6 (base_rate 0.248) | 0.229 | 0.238 | 0.259 | 0.266 | 0.282 |
| **FT d8 ss0.7 cs0.7 (ADOPTED)** | **0.341** | **0.300** | **0.313** | **0.307** | **0.311** |

Beats the baseline at **every K** (@1 +0.112). The `#50`/`#52` profile exactly: common
event + weak baseline @1 (0.229 < base_rate) → deep+bagging redistributes into a strictly
better book.

### +20%/100d ffund — BASELINE STANDS (`all/d6`)
| config | test @1 | @3 | @5 | @10 | @20 |
|---|--:|--:|--:|--:|--:|
| **baseline all/d6 (KEPT; base_rate 0.245)** | **0.454** | **0.371** | **0.341** | 0.320 | 0.308 |
| FT d10 ss0.7 cs0.7 (val-best) | 0.337 | 0.339 | 0.317 | 0.304 | 0.310 |
| FT d6 ss0.7 cs0.7 | 0.386 | 0.329 | 0.314 | 0.322 | 0.318 |

High base @1 (0.454) = the top is already maxed; both FTs collapse @1 without a book win
(the `#51`/`#54` outcome). The strong val lift did not transfer — val's hotter prevalence
inflated it.

## Backtest (2025-07-01 → 2026-07-14, untouched window, `--sizing-mode equal`)

| cell (adopted config) | strategy | maxDD | EW basket | raw top-K | exits target/DD/horizon |
|---|--:|--:|--:|--:|--:|
| +10%/25d ffund (d8 FT) | **+11.6%** | −18.5% | +6.2% | **+37.4%** | 29 / 49 / 15 |
| +20%/100d ffund (baseline) | **+9.6%** | −12.2% | +6.2% | +6.7% | 26 / 37 / 1 |

Both beat their universe basket. 10/25's raw top-K (+37.4%) is the strongest signal read;
the Kelly-style strategy overlay gives most of it back (25d horizon + 10% target = high
turnover through the DD-stop). Neither meets the strict target>DD deploy cut.

## Verdict + caveats
- **The de-confounded F18-IN edge is real on nifty500**: two independent test windows +
  a forward backtest window, all with fund ≥ base at the book. This closes the
  "single-window mirage" objection that killed every prior F17/F18 claim.
- **Adopted models:** `nifty500_up_10pct_25d_dd5pct_ffund_canon_ft` (d8/ss0.7/cs0.7) and
  `nifty500_up_20pct_100d_dd10pct_ffund_canon_ft` (baseline all/d6);
  `_289`'s +50%/200d d10 stands. NOT promoted — nifty500 has no `/daily-predictions`
  deploy path (a V1.10_TBD item).
- Window-2 caveat: hotter prevalence (2023-24 rally) → deltas, not levels, are the read.
- Backtest benchmark is `^NDX` (US placeholder; no NIFTY index cached — V1.10_TBD §3).

Registry: `*_w2` rows + `*_canon_ft` updates in `results/gbdt/data/r_precision_at_k.csv`.
Per-cell trails: `results/gbdt/experiments/<cell>/hp/EXPLORATION.md`.
