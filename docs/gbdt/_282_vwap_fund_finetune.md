# _282 — VWAP / fund agent-finetune: FS-first, and the noise-dominated top-of-book boundary

**Date:** 2026-07-08 · **Branch:** `gbdt-vwap-features` · **Universe:** nasdaq100 · **Backend:** xgboost
**Depends on:** `_281` (VWAP sweep — the single-fit baselines this compares against)

## Question

The `_281` sweep left 11 candidate `(cell, feature-arm)` finetune targets (top-3 R-p@3 ∪ top-3 AUC across the vwap / fund / fvwap arms). Does driving the agent FS+HP loop beat the untuned **single-fit** (default-HP xgboost) on the operating metric — per-day top-K R-Precision — for any of them?

## Protocol (one fixed recipe, no per-cell test-fishing)

Every cell gets the **identical FS-first recipe**, so the 11-cell table is a clean protocol comparison, not a search over each cell's test score:

1. **iter 0** — default-HP fit (= the single-fit baseline).
2. **iter 1** — **cliff-cut FS** (drop every feature with gain importance < 0.01) **+ a slow-learn/ES schedule** (`eta 0.05`, `early_stopping_rounds 100`, `max_depth 6`).
3. **finalize** on the loop's eval-best iter; compare TEST R-p@{1,3,5,10,20} + AUC to the single-fit.

`split.mode: date_aligned`, `train_start 2019-01-01`, `--snapshot-end 2026-07-06`, **test window 2024-07-26 → 2024-12-16** (Q = 64 days). FS-first is applied per **`[[project-gbdt-tuning-playbook]]` rule 14** (a smoothing HP change on the full pool anti-selects the top book).

## Result — 11 cells (TEST, single-fit → FS-tune; base_rate = prevalence)

Grouped by regime. `gap` = train/val Brier gap iter0→iter1 (overfit closed?); `eval` = loop eval R-p@1 iter0→iter1.

### Rare event (prevalence ≤ 2.6%) — single-fit stands decisively

| cell · arm | base_rate | gap 0→1 | eval 0→1 | R-p@1 | R-p@3 | R-p@10 | AUC |
|---|--:|--|--|--:|--:|--:|--:|
| 50%/25d · vwap | 0.0102 | 0.005→0.001 | 0.10→0.49 | 0.719→0.703 | 0.716→0.685 | 0.870→0.917 | 0.937→0.961 |
| 50%/25d · fund | 0.0102 | 0.005→0.001 | 0.33→0.67 | 0.672→**0.297** | 0.607→**0.242** | 0.794→0.755 | 0.922→0.922 |
| 50%/25d · fvwap | 0.0102 | 0.005→0.000 | 0.24→0.75 | 0.641→**0.234** | 0.669→**0.195** | 0.792→0.857 | 0.911→0.929 |
| 50%/50d · fund | 0.0257 | 0.025→0.000 | 0.22→0.41 | 0.444→0.358 | 0.584→**0.374** | 0.765→0.769 | 0.950→0.946 |
| 50%/50d · fvwap | 0.0257 | 0.025→−0.001 | 0.26→0.39 | 0.519→**0.074** | 0.543→**0.210** | 0.797→0.659 | 0.950→0.939 |
| 40%/25d · fund | 0.0152 | 0.015→0.002 | 0.33→0.37 | 0.597→**0.167** | 0.516→**0.188** | 0.676→0.829 | 0.904→0.940 |
| 40%/25d · fvwap | 0.0152 | 0.013→0.001 | 0.13→0.45 | 0.583→**0.194** | 0.507→0.377 | 0.779→0.773 | 0.912→0.943 |

### Common event, long horizon (H = 200) — single-fit stands on top

| cell · arm | base_rate | gap 0→1 | eval 0→1 | R-p@1 | R-p@3 | R-p@10 | AUC |
|---|--:|--|--|--:|--:|--:|--:|
| 40%/200d · fund | 0.1566 | 0.160→0.028 | 0.36→**0.29** | 0.670→0.510 | 0.543→0.503 | 0.428→0.540 | 0.737→0.710 |
| 40%/200d · fvwap | 0.1566 | 0.170→0.044 | 0.24→0.31 | 0.640→0.290 | 0.540→0.407 | 0.374→0.475 | 0.710→0.747 |

### Common event, mid horizon (H = 50) — the sweet spot

| cell · arm | base_rate | gap 0→1 | eval 0→1 | R-p@1 | R-p@3 | R-p@10 | AUC |
|---|--:|--|--|--:|--:|--:|--:|
| 20%/50d · vwap | 0.1700 | 0.145→0.016 | 0.26→0.44 | 0.550→0.230 | 0.513→0.510 | 0.400→0.552 | 0.724→0.759 |
| **20%/50d · fund** | 0.1700 | 0.143→0.018 | 0.24→0.39 | 0.610→**0.690** | 0.510→**0.570** | 0.423→0.539 | 0.738→0.763 |

## Reading

**One clean win in eleven.** `20%/50d · fund` is the only cell where the finetune beats the single-fit on *every* K (@1 +0.08, @3 +0.06, @5 +0.10, @10 +0.12, @20 +0.04, AUC +0.03). Its same-cell vwap sibling ties @3 and lifts the broad book but loses @1. Everything else loses the sharp top.

**Universal shape: the finetune trades top-of-book for bulk ranking.** Across the 11: R-p@1 down on **10/11**, R-p@3 down-or-tie on **10/11**, while R-p@10 up on 8/11, R-p@20 up on 9/11, AUC up on 7/11. The smoothing schedule improves aggregate ranking (AUC / @10 / @20) by spreading probability mass off the single most-confident pick — the exact mass top-K trading consumes. This is the `_276` anti-selection lesson reproduced on a fresh arm set.

**FS-first bounds the catastrophe but does not prevent the degradation.** The same slow-learn schedule *without* FS collapsed `50%/25d vwap` R-p@3 to **0.055**; FS-first recovered it to **0.685** (≈ the 0.716 single-fit). FS is inert at default HP (redundant features ignored by the greedy fit) but protective under smoothing — it removes the noise dimensions the schedule would otherwise dilute the top tail into. **But** on the fund / fvwap rare arms, even FS-first leaves @3 at 0.19–0.24 (−0.33 to −0.47): FS bounds the *worst* case, it is not a reliable rescue. (Codified as rule 14; see `[[project-gbdt-tuning-playbook]]`.)

**The boundary — why some cells tune and most don't.** Read the `eval 0→1` column against the outcome:
- **Rare cells** (prev ≤ 2.6%): eval R-p@1 *doubles-to-triples* while test @1/@3 collapse — the eval oracle is **inverted / uninformative**. With ~1 positive/day over 64 test days the top-of-book ranking is variance, not signal; the loop maximizes eval and walks straight into a test-terrible config. No in-loop signal can steer this. **Single-fit stands.**
- **Common long-horizon** (H=200): the overfit only half-closes (gap 0.16→0.03–0.04) and eval moves *with* test — but the H≥100 eval→test decay (`[[project-gbdt-tuning-playbook]]` rule 13) still drops @1/@3. **Single-fit stands on top.**
- **Common mid-horizon** (H=50, prev 17%): overfit **fully** closes (gap → 0.016–0.018) and eval **tracks** test (both up). This is the only regime where the loop is functional — and it delivered the one clean win (fund arm) + a broad-book win (vwap arm).

So: **the agent-finetune helps only on common event + mid horizon + a genuine overfit the schedule can fully close — and even there the feature arm decides whether the sharp top survives.** Rare-event or long-horizon top-of-book is not tunable; the single-fit's overconfident top pick is the thing to keep.

## Verdict

- **Single-fits remain the champions** on the operating metric for 10 of 11 cells. No champion swap.
- **`nasdaq100 +20%/50d · fund` (all_fundamentals) is a finetune candidate** — the FS-first tune beats the single-fit at every K. **One window only**; per the F17/F18/F20 precedent it needs an independent second-window replication (the `_272`→`_273` pattern) before adoption. `deployed=False`, not promoted, `/daily-predictions` unchanged.
- **russell1000 finetunes: TODO** — parked with the russell1000 sweep (`_281`); the same 11-cell protocol should run once that sweep completes.

## What this produced

- Reusable finding → **playbook rule 14** (FS-first mandatory before any smoothing HP change) + the `gbdt-experiment` SKILL diagnostic-first-gate exception + `fs_prefit`/`scout` separation + a CLAUDE.md anti-pattern bullet (commit `015a3782`).
- Driver: `/tmp/ft_fs_all11.sh` (uniform recipe); the 11 finetune specs live at `configs/gbdt/experiments/nasdaq100_up_*_{vwaptune,fundtune,fvwaptune}.yaml`.
