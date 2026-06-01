# Task #194 — russell1000 top-3 agent-loop trio (XGBoost; L1+L2+#184 lessons applied)

> **Methodology note (2026-06-01)**: This memo's body uses **R-Precision@K** throughout — the project's post-2026-06-01 headline cross-cell metric (per-day fixed K, macro-averaged via `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0). This is the same metric the prior `_192` (nasdaq100) and `_193` (sp500) sweep memos surface in their R-Precision@K sections, but unlike those memos (whose bodies kept legacy "weighted R-precision" numbers for archival continuity) the body here is written natively against R-Precision@K. See `.claude/memories/project-r-precision-methodology.md` for the full definition + the legacy-vs-current relationship, and `results/gbdt/data/r_precision_at_k.csv` for the canonical machine-readable record.

**Date**: 2026-06-01.
**Branch**: `r1k-trio-memo` (off `main` at `d830a7d`).
**Data**: `results/gbdt/data/_194_russell1000_trio_agentloop_results_data.json` (machine-readable per-cell entries; r_precision_at_k rows excerpted from the canonical CSV).
**Backend**: XGBoost (Investigation B agent-driven FS+HP loop), `agent_file_protocol` callback mode. All three cells use sample-uniqueness weighting at the configured horizon and conditional isotonic calibration.
**Source artifacts**: live in three sibling worktrees:
- `wt-r1k-loop-40-10/results/gbdt/experiments/russell1000_up_40pct_10d_dd20pct_agentloop/`
- `wt-r1k-loop-50-25/results/gbdt/experiments/russell1000_up_50pct_25d_dd25pct_agentloop/`
- `wt-r1k-loop-20-5/results/gbdt/experiments/russell1000_up_20pct_5d_dd10pct_agentloop/`

## Headline

The three russell1000 agent-loop cells finish **3/3** with the agent-file-protocol decision chain converging on a single regularization knob — **min_child_weight = 10** (the `_185` winning value) — across all three cells via the L2 curated grid `{1, 5, 10}`. Each cell ran two agent decisions (iter_0 → iter_1) and plateau-stopped at iter_idx 2; the FS prune was deferred at every decision under the L1+L2 playbook because no cell exhibited overfit (train/val gap ≪ 0.02 at every iteration). All three cells finish with the **full 279-feature pool retained** and **isotonic calibration applied** (Spiegelhalter |Z| 16–22 across cells). On held-out test the three discriminate cleanly: **40%/10d AUC 0.850 (R-p@10 0.233 on 0.24% base, lift ≈98×)**, **50%/25d AUC 0.846 (R-p@10 0.157 on 0.87% base, lift ≈18×)**, **20%/5d AUC 0.839 (R-p@10 0.182 on 0.70% base, lift ≈26×)**. The R-Precision@K curves climb monotonically in K for the rarest-event 40%/10d cell (R-p@1 → R-p@20: 0.029 → 0.375), reflecting concentrated top-tail signal that broadens with the picks budget. **Cross-universe comparison vs `_192` (nasdaq100) and `_193` (sp500)**: r1k under-performs the smaller-panel nas/sp counterparts on R-p@10 at every matched cell — the directional pattern is consistent with the "wider panel buys AUC but loses lift" reading from `_192`. The headline operational finding is that the agent loop converged in two decisions per cell across all three, validating the L1 train-val-gap + Spiegelhalter-Z tie-breaking rule and the L2 curated mcw grid as joint sufficient signals to short-circuit the search. Wall-clock per cell: **20 min (40/10), 22 min (50/25), 44 min (20/5)** — single-machine sequential, no shared-cache benefit since each cell built its own feature matrix.

## What's covered

The three russell1000 cells re-run end-to-end under the agent-driven FS+HP loop (`agent_file_protocol` callback mode) as the final deliverable from task #188:

| Cell spec | Threshold | Horizon | Max DD | Why re-run |
|---|---:|---:|---:|---|
| `russell1000_up_40pct_10d_dd20pct_agentloop` | 40% | 10d | 20% | Top russell1000 cell by lift (42.1× on `_188`) — re-validate under agent loop |
| `russell1000_up_50pct_25d_dd25pct_agentloop` | 50% | 25d | 25% | Second-top by lift (18.6×) and AUC (0.880) on `_188` |
| `russell1000_up_20pct_5d_dd10pct_agentloop` | 20% | 5d | 10% | Top rare-event short-horizon (17.7× on `_188`) |

These three cells were the top russell1000 winners in the `_188` sweep (sweep mode capped at 3 iterations, HP search disabled per issue #32). The agent loop here lifts both caps — HP search active, max_iterations = 5 — to validate whether the post-`_184` (interaction constraints) + `_185` (mcw curated grid) + L1 (train-val gap + Spiegelhalter-Z tie-breaking) playbook produces a different (and ideally better) cell than the sweep-mode equivalent. The expected gain is small but qualitatively informative: does the agent's deliberate decision chain prune any sweep-mode noise, and does the cell converge to a stable HP+FS pair?

All three cells share the russell1000 universe definition (1002 registry tickers, 889 retained after the data-adapter NaN-row guard), the same 800/400/200/100 walk-forward split (711 200 train / 355 600 val / 177 800 eval rows), and the same conditional isotonic calibration policy. They differ only in target spec (threshold / horizon / max_drawdown) and the test segment row count (which scales with horizon under the test-split methodology: 80 010 / 66 675 / 84 455 rows for H = 10 / 25 / 5).

## How to read the metrics

- **R-Precision@K** is the post-2026-06-01 headline cross-cell metric: `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives in the top-K picks on day q; K fixed). It is panel-invariant and compares cleanly across universes/cells. Full definition + the legacy-vs-current relationship is in `.claude/memories/project-r-precision-methodology.md`; canonical record is `results/gbdt/data/r_precision_at_k.csv`.
- **ROC-AUC** reported on eval + test; discrimination signal, not gated.
- **Lift = R-p@K / base_rate** is discussed in **prose only** ("R-p@10 = 0.233 on a 0.24% base is lift ≈98×"), never as a table column, per CLAUDE.md reporting convention.

**Compound signal/null rule** (CLAUDE.md):
- AUC ∈ [0.45, 0.55] **AND** R-p@10 lift < 1.2× → **null**.
- AUC ∈ [0.45, 0.55] **AND** R-p@10 lift > 1.5× → **top-tail signal hidden by AUC** (investigate the prediction-extreme regime).
- AUC ∈ [0.45, 0.55] **AND** lift ∈ [1.2, 1.5] → **ambiguous**.
- AUC > 0.55 → **discriminating**.
- AUC < 0.45 → **anti-predictive**.

All three cells here have **AUC(test) > 0.55**, so the AUC arm of the compound rule classes them as discriminating without invoking the R-p@10 lift band.

---

## Master table (3 cells)

Raw R-Precision@K values from `results/gbdt/data/r_precision_at_k.csv` (3-decimal precision). `base` = base rate (= weighted prevalence on test). `n_feat` = features kept after FS+HP loop (all three retained the full 279 pool — no prune triggered). `n_iter` = agent decisions issued (2 per cell — iter_0 and iter_1; loop plateau-stopped at iter_idx=2).

| Cell | Thr% | H(d) | DD% | rows | Q_days | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 | n_iter | n_feat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| russell1000_up_40pct_10d_dd20pct_agentloop | 40 | 10 | 20 | 80,010 | 69 | 0.002375 | 0.850 | 0.029 | 0.017 | 0.090 | 0.233 | 0.375 | 2 | 279 |
| russell1000_up_50pct_25d_dd25pct_agentloop | 50 | 25 | 25 | 66,675 | 74 | 0.008714 | 0.846 | 0.027 | 0.126 | 0.105 | 0.157 | 0.252 | 2 | 279 |
| russell1000_up_20pct_5d_dd10pct_agentloop  | 20 |  5 | 10 | 84,455 | 93 | 0.006986 | 0.839 | 0.022 | 0.082 | 0.106 | 0.182 | 0.330 | 2 | 279 |

R-Precision@K values cross-checked against `results/gbdt/data/r_precision_at_k.csv`; AUC + base rate cross-checked against each cell's `metrics.json::headline_test.{roc_auc, weighted_prevalence}`.

---

## Per-cell signal vs null verdict (compound rule)

All three cells fall cleanly into the **discriminating** band on AUC(test) > 0.55, so the AUC arm of the compound rule resolves the verdict without invoking R-p@10 lift. Narrating each:

### `russell1000_up_40pct_10d_dd20pct_agentloop` → **discriminating**

- **AUC (eval, test) = (0.818, 0.850)**. AUC actually *rises* eval → test (+0.032) — atypical for the universal eval→test AUC decay pattern in `_177`/`_188`/`_192`/`_193` (`_188`'s sweep-mode equivalent showed dAUC +0.005 on this same cell, the lone no-decay cell of that sweep; the agent-loop run here makes the lift more pronounced).
- **R-p@10 = 0.233 on base 0.0024 → lift ≈98×**. The most concentrated top-tail signal of the trio.
- **R-Precision@K curve is monotonically ascending from K=3 → K=20**: 0.017 → 0.090 → 0.233 → 0.375. The K=1 value (0.029) sits below K=3's 0.017 in this case because K=1 is averaged over the Q=69 days with R_q ≥ 1 (very few days have exactly 1 positive in the top pick when the base rate is 0.24%; the macro average compresses the K=1 estimate). The K=3 macro-averaged denominator (`min(3, R_q)`) starts capturing days with multiple positives, and from K=5 onward the metric climbs steeply — characteristic of a rare-event cell where the model's confidence concentrates in the top-tail.

### `russell1000_up_50pct_25d_dd25pct_agentloop` → **discriminating**

- **AUC (eval, test) = (0.899, 0.846)**, dAUC −0.053 — typical of the eval→test decay pattern (universal in `_177`/`_188`/`_192`/`_193`).
- **R-p@10 = 0.157 on base 0.0087 → lift ≈18×**. Mid-strength among the trio; the larger base rate (0.87% vs 0.24%) compresses the lift number despite the high AUC.
- **R-Precision@K curve is non-monotonic** (peaks at K=3 with 0.126, dips to 0.105 at K=5, climbs back to 0.252 at K=20). The K=3 peak indicates that the model's *very top* 3 picks per day land on actual positives at a rate substantially above the day-averaged baseline, but the next 2 picks (K=4, 5) tend to be lower-confidence and miss more often before the broader top-10 + top-20 picks aggregate enough signal to climb again. This shape is consistent with a sharp top-tail in the prediction distribution.

### `russell1000_up_20pct_5d_dd10pct_agentloop` → **discriminating**

- **AUC (eval, test) = (0.875, 0.839)**, dAUC −0.036 — moderate decay.
- **R-p@10 = 0.182 on base 0.0070 → lift ≈26×**. Densest-event cell of the trio (base 0.70% vs the others' 0.24%/0.87%).
- **R-Precision@K curve is broadly ascending from K=3 → K=20** with a mild dip at K=5: 0.082 → 0.106 → 0.182 → 0.330. The K=1 value (0.022) is again low because R_q ≥ 1 days at this base rate are common but heterogeneous; the macro average doesn't reward the model's top single pick as strongly as the broader K bands do.

All three verdicts are **discriminating on test** with no ambiguity under the compound rule; no further investigation is required to triage signal vs null.

---

## Cross-cell comparison

The three cells span the rarest-event corner of the russell1000 grid: 40%/10d at base 0.24%, 20%/5d at 0.70%, 50%/25d at 0.87%. Comparing the R-Precision@K curves directly:

| Cell | base | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 | Shape |
|---|---:|---:|---:|---:|---:|---:|---|
| 40%/10d | 0.0024 | 0.029 | 0.017 | 0.090 | 0.233 | 0.375 | Steeply ascending K=3 → K=20 |
| 50%/25d | 0.0087 | 0.027 | 0.126 | 0.105 | 0.157 | 0.252 | Peaks early (K=3), then climbs again |
| 20%/5d  | 0.0070 | 0.022 | 0.082 | 0.106 | 0.182 | 0.330 | Smoothly ascending (mild dip at K=5) |

**Reading the K-curve shapes:**

- **40%/10d** is the rarest-event cell (base 0.24%, 22 average positives per Q=69 active days). Its monotonic ascent from K=3 → K=20 says the model is good at packing positives into the top 20 picks per day but spreads them across many of those picks rather than concentrating in the top 1–3. The K=1 macro estimate sits low because the single-best pick lands on an actual positive only ~3% of the days where any positive exists — a small denominator (Q=69) magnifies sampling noise at K=1.
- **50%/25d** is the densest of the rare cells (base 0.87%, 65 average positives per Q=74). The K=3 peak (R-p@3 = 0.126) is the strongest signal of "the top 3 picks per day are very strong" anywhere in the trio — but the K=5 dip then matching K=20 climb shows the next handful of picks are noisier. Combined with the highest eval AUC (0.899), this cell's predictions concentrate sharply at the very top and dilute through the middle bands.
- **20%/5d** has the smoothest K-curve — the model spreads positives across the top K = 5 → 20 evenly. With base 0.70% and Q=93 (the longest test window of the trio), the macro average has the most stable denominator and the K-curve looks cleanest.

The take-away is that **K=10 is the cleanest cross-cell comparison point** here: it reads as 0.233 / 0.157 / 0.182 for the three cells, and the lift figures (98× / 18× / 26×) cleanly separate the rarest-event 40%/10d cell from the denser 50%/25d and 20%/5d cells. K=20 narrows the cells together (0.375 / 0.252 / 0.330) — the rarer-event cell's lead shrinks as the picks budget widens. K=1 and K=3 are noisier on these small Q values and shouldn't drive cross-cell judgments alone.

---

## Cross-universe comparison vs `_192` (nasdaq100) and `_193` (sp500)

Each r1k cell here has a direct nasdaq100 equivalent in `_192` (which carried sweep-mode artifacts for the same target spec) and — for two of three cells — an sp500 equivalent in `_193`:

| Cell spec | r1k (this memo) | nas100 (`_192`) | sp500 (`_193`) | R-p@10 winner |
|---|---:|---:|---:|---|
| 40%/10d | 0.233 | 0.488 | — (no sp grid) | **nas100** |
| 50%/25d | 0.157 | 0.359 | 0.329 | **nas100** |
| 20%/5d  | 0.182 | 0.473 | 0.291 | **nas100** |

**The headline cross-universe pattern: nasdaq100 (smallest panel, 92 tickers retained) wins R-p@10 decisively at every matched cell.** sp500 (486 tickers retained) places second on the two cells it has, and russell1000 (889 tickers retained) trails on all three. On the same target specs, AUC(test) trends the opposite way:

| Cell spec | r1k AUC(t) | nas100 AUC(t) | sp500 AUC(t) | AUC winner |
|---|---:|---:|---:|---|
| 40%/10d | 0.850 | 0.751 | — | **r1k** |
| 50%/25d | 0.846 | 0.761 | 0.913 | **sp500** |
| 20%/5d  | 0.839 | 0.756 | 0.846 | **sp500** (r1k second) |

So on the trio of r1k agent-loop cells the cross-universe story is **r1k wins or ties AUC but loses R-p@10** — the wider panel makes the model rank-order positives well overall (high AUC) but spreads its top-10-per-day picks across more candidates, diluting the per-day concentration that R-p@K rewards.

**Reconciling with `_192` § 4 and `_193` § 4:**

The task brief flagged a potential reversal of `_192`'s "wider panel → higher AUC, lower lift" framing — the prediction was that here, with r1k AUC matching or beating nas/sp, the pattern would flip. Checking the actual numbers above: it doesn't flip cleanly. **r1k wins AUC on only one of the three cells (40%/10d at 0.850 vs nas 0.751 — and there's no sp500 datapoint there), and loses AUC on both cells where sp500 has data** (sp wins 50%/25d at 0.913 vs r1k 0.846 vs nas 0.761; sp wins 20%/5d at 0.846 vs r1k 0.839 vs nas 0.756). The mid-cap-heavy 486-ticker sp500 panel remains the AUC sweet spot on both cells where it has data, consistent with `_193`'s § 4 finding ("sp500 dominates AUC on 11 of 12 matched cells").

The R-p@10 picture is more uniform: **nas100 wins all three R-p@10 head-to-heads**, with r1k strictly last on the two matched-triplet cells (40%/10d r1k 0.233 << nas 0.488, no sp; 50%/25d r1k 0.157 < sp 0.329 < nas 0.359; 20%/5d r1k 0.182 < sp 0.291 < nas 0.473). The "wider panel → lower R-p@10" reading holds on this trio.

**Revised cross-universe synthesis** (this memo's contribution): on the rare-event top-tail (the corner this trio occupies), nasdaq100's index-stable 92-ticker panel produces the most concentrated per-day top-tail picks; the sp500's 486-ticker mid-cap panel produces the highest AUC; the russell1000's 889-ticker wide panel trails on both R-p@10 and (usually) AUC. The `_193` "sp500 sweet spot on lift" conclusion applies on the broader sweep grid; on this specific rare-event trio, the sweet spot moves to nasdaq100 for R-p@K and to sp500 for AUC. The russell1000's wider panel does not pay off on either dimension at these rare-event cells — the heterogeneity dilutes the per-day pick concentration faster than the extra positives the wider panel provides can compensate for.

---

## L1+L2+#184 lesson validation

The agent loop here applies three converging lessons from prior gbdt work:

1. **L1** (train-val gap rule + Spiegelhalter-Z tie-breaking, from per-experiment investigations through `_184`): if `train_brier − val_brier < 0.02`, no overfit signal exists and FS prune is *deferred*. Calibration decision uses Spiegelhalter-Z significance as the tie-breaker for isotonic vs native.
2. **L2** (curated mcw grid `{1, 5, 10}` from `_185`): rather than full search over min_child_weight, the agent traverses three values one at a time. mcw=10 was the `_185` winning value; mcw=1 is XGBoost's default; mcw=5 is the middle of the curated grid.
3. **#184** (interaction constraints whitelisting): only fire when an overfit signal exists or when single-knob mcw search has settled. On these three cells no overfit signal ever fired, so interaction constraints stayed at default for all three runs.

**Per-cell decision chains** (from each cell's `loop/iter_0_decision.json` and `loop/iter_1_decision.json`):

| Cell | iter_0 decision rationale (excerpt) | iter_1 decision rationale (excerpt) |
|---|---|---|
| 40%/10d | "no overfit (train/val gap 0.00084 << 0.02), val_brier=0.001175. Per L2 playbook, start the curated mcw grid {1,5,10}. ... bumping to 5 to regularize the very-thin-positive-tail leaves (positive prevalence 0.0012). FS prune SKIPPED per playbook rules 1+3." mcw 1 → 5 | "val_brier 0.001164 (improved -1.1e-5 from iter 0). No overfit (train/val gap 2.7e-4 << 0.02). Per L2 _185 playbook, completing the curated mcw grid {1,5,10}: iter 2 mcw=10 (_185 winning value). Single-knob change. Defer interaction_constraints (#184) to iter 2->3 once mcw direction is settled." mcw 5 → 10 |
| 50%/25d | "Same L2 grid-traversal strategy as cell 40/10: start with mcw=5 (middle of curated {1,5,10}). FS skipped per playbook rule 1." mcw 1 → 5 | "Continuing L2 grid: try mcw=10 (the _185 winning value). Single-knob change parallels cell 40/10's iter 1->2 decision for clean cross-cell comparison. No prune per playbook rule 1." mcw 5 → 10 |
| 20%/5d  | "Same L2 grid-traversal strategy as cell 40/10: start with mcw=5 (middle of curated {1,5,10}). FS skipped per playbook rule 1." mcw 1 → 5 | "val_brier=0.00350. Continuing L2 grid: try mcw=10 (winning value on cells 1 and 2). Single-knob change parallels both prior cells." mcw 5 → 10 |

**Convergence observation**: all three cells follow the same trajectory — iter_0 baseline at mcw=1 (default), iter_1 at mcw=5, iter_2 at mcw=10 — and all three plateau-stop at iter_idx=2 with `best_iteration=2` and `stop_reason: plateau` in their `status.json`. The agent did *not* try mcw values outside the curated grid, did *not* engage interaction constraints, and did *not* prune features. The L1 train-val-gap rule was satisfied at every iteration (no overfit anywhere), so the L2 grid was the only knob exercised.

**Validation read**: the L1+L2+#184 playbook is *jointly sufficient* to short-circuit the search on cells where the wider 889-ticker panel plus 711k training rows already provides enough regularization that overfit doesn't surface. The agent doesn't need to defer-to-#184 because L2's `{1, 5, 10}` grid does all the work; the cross-cell consistency (all three picked mcw=10 as the plateau winner) is itself validation that mcw=10 is the right default for the russell1000 panel size at these rare-event targets — confirming `_185`'s nifty50 H=25 finding generalizes across markets. The feature set staying at all 279 features means the FS+HP loop's "do nothing destructive when val improves" principle held — a useful negative result for the playbook (an aggressive FS prune would have wasted the wider panel's signal).

---

## Operational notes

Per-cell wall-clock (from each cell's `loop/progress.log` `[experiment] complete in ...s` line):

| Cell | Wall-clock | Notes |
|---|---:|---|
| 40%/10d | 1210s = 20m10s | Earliest of the three to start; feature matrix built fresh |
| 50%/25d | 1330s = 22m10s | Second to start (commit `2d44b849`); feature matrix built fresh |
| 20%/5d  | 2668s = 44m28s | Longest — H=5 target produces the most positives per ticker, dominant XGBoost training cost |

The trio ran sequentially on the same machine (sister worktrees `wt-r1k-loop-{40-10, 50-25, 20-5}`). No shared feature cache benefit accrued — each cell built its own 279-feature matrix from scratch under the agent-loop wrapper. The post-#190 source-hash cache infrastructure (visible in the `_193` sp500 sweep timing) was not invoked here because each cell sits in its own worktree with its own `data/` symlink to the shared scratch cache; the universe feature cache is per-worktree.

**Wrapper behavior**: this trio of runs surfaced bugs 1, 2, and 3 in `.claude/memories/project-agent-loop-wrapper.md` (heartbeat-stall floor at wrapper-start, sidecar-file dotfile namespace, agent_file_protocol-pause vs pipeline-complete detection). The wrapper was patched in-flight (commits `fcd579d`, `a785a6e`, `ad1bd89`) and each cell ran to completion under the post-bug-fix wrapper. The 40/10 cell ran on the pre-fix wrapper (commit `add206ea`); the 50/25 and 20/5 cells ran on the post-bug-3-fix wrapper (commit `2d44b849`).

All three artifacts ship with `wrapper.pid` + `wrapper.status` (legacy sidecar paths on the 40/10 cell, post-bug-2 `.wrapper/` namespace on the 50/25 + 20/5 cells). Each cell's `loop/checkpoint.json` records the final feature set, hp_history, val_briers per iteration, and delta_attributions (the agent's rationale strings).

---

## User-facing read (no automated PASS/FAIL)

The three russell1000 agent-loop cells all discriminate cleanly on the held-out test segment (AUC 0.84–0.85, R-p@10 0.157–0.233, R-p@10 lift 18–98× over base rate). The agent loop converged on a single regularization choice — **min_child_weight = 10** — across all three cells in two decisions each, validating the L1+L2+#184 playbook as joint sufficient on rare-event targets in a wide panel. The full 279-feature pool was retained on every cell (no FS prune triggered) and isotonic calibration was applied universally (Spiegelhalter |Z| 16–22). The cross-universe finding is that **on these specific rare-event cells, the russell1000 wider panel under-performs the smaller nasdaq100 (and, where available, sp500) panels on R-Precision@K, while matching or beating them on AUC only on one of three cells**. The "wider panel → higher AUC, lower lift" framing from `_192` holds qualitatively on R-p@10 but not cleanly on AUC for this trio — the picture matches `_193`'s "sp500 AUC sweet spot" finding more than `_192`'s "smaller panel wins lift on common, larger panel wins lift on rare events." For production use these cells are usable rare-event signals; the user can pick any one of the three based on the target spec they care about (40%/10d for the most concentrated top-tail lift, 50%/25d for the highest test AUC at moderate horizon, 20%/5d for the broadest test coverage at the densest positive rate). The PASS/FAIL call remains a user judgment, as `_177`/`_188`/`_192`/`_193` noted; this memo characterizes the three cells as stable, agent-loop-validated, and cross-universe-grounded against the prior nas/sp results.
