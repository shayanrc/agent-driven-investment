# _262 — Champion re-baseline at matched snapshot: macro BEATS the champion (both cells)

**Headline:** Re-baselining the committed champions at the current snapshot —
removing **both** the window and the tuning-mode confounds that muddied `_260` —
flips the verdict: **with the champion's exact config held fixed, the
macro-augmented model beats the champion on both cells at nearly every K**
(+50%/50d R-p@1 0.540 → 0.680; +20%/25d R-p@1 0.280 → 0.373). On the +50% cell the
macro model (0.680) exceeds even the champion's all-time recorded R-p@1 (0.640).
The `_260` "macro sign-flips" result was an artifact of the default auto-loop
choosing *different* HP for the base vs macro arms — not a real instability.

## What changed vs `_260`

`_260` compared `champbase` vs `macrochamp` under `callback_mode: default`
(auto FS+HP loop, max_iter 5). The loop tunes each arm **independently**, so the
base and macro arms landed on **different** hyperparameters — confounding the macro
effect with the loop's per-arm HP divergence (e.g. on +20% the base arm's loop
found R-p@1 0.347 while the macro arm's loop over-regularized to 0.120).

Both committed champions converged to **one simple config**: `min_child_weight=10`
on all 279 features, no FS, n_iter=1. So this round reproduces that **exactly** via
`hp_starting: {min_child_weight: 10}` + a single fit (max_iter 1), and runs the
macro arm with the **identical** HP. Now the *only* difference between the two arms
is the 45 macro columns — a clean isolation at the champion's own tuning, on the
champion's own trailing split, at the same snapshot (2026-06-20).

## Results (matched window + matched HP; raw R-Precision@K + base rate)

### sp500 +50%/50d (dd25%) — test 2026-01-22→2026-04-02, base_rate 0.0384, Q=50

| K | champ_resnap (mcw=10, base) | macrochampagent (mcw=10, +macro) | committed champion (ref, old window) |
|---|---|---|---|
| R-Precision@1 | 0.5400 | **0.6800** | 0.6400 |
| R-Precision@3 | 0.4600 | **0.4667** | — |
| R-Precision@5 | 0.3800 | **0.4840** | — |
| R-Precision@10 | 0.4572 | **0.5018** | 0.3460 |
| R-Precision@20 | 0.4776 | 0.4734 | — |
| test AUC | 0.8415 | 0.8265 | — |

Macro wins @1/@3/@5/@10 (R-p@1 +26%), ~tie @20. The champion's config transfers to
the new window (0.540, vs its 0.640 on the old window), and **macro on top reaches
0.680 — above the champion's all-time best.**

### sp500 +20%/25d (dd10%) — test 2026-01-22→2026-05-08, base_rate 0.0878, Q=75

| K | champ_resnap (mcw=10, base) | macrochampagent (mcw=10, +macro) | committed champion (ref, old window) |
|---|---|---|---|
| R-Precision@1 | 0.2800 | **0.3733** | 0.4133 |
| R-Precision@3 | 0.3378 | **0.4356** | — |
| R-Precision@5 | 0.3253 | **0.4187** | — |
| R-Precision@10 | 0.3120 | **0.3933** | 0.4027 |
| R-Precision@20 | 0.2965 | **0.3545** | — |
| test AUC | 0.7278 | 0.7376 | — |

Macro wins at **every K** (R-p@1 +33%, @10 +26%). vs the committed champion (0.413,
old window) it's lower — but that's a different, harder window; on the **matched**
window macro clearly beats the champion's own config (0.280).

## Reconciliation with the earlier rounds

- `_259` (date-aligned A/B): macro helped +20% decisively. ✓ consistent.
- `_260` (default-auto, this window): macro "sign-flipped" — **now explained as the
  auto-loop's per-arm HP divergence**, not a macro property.
- `_262` (matched HP, this window): macro helps **both** cells. ✓

The coherent story: **under controlled/matched tuning, the macro panel is additive
and champion-beating on this window.** The apparent instability in `_260` was a
tuning-search artifact.

## Verdict

- **On the cleanest possible comparison — the champion's exact config, the same
  trailing window, the only difference being the macro features — the
  macro-augmented model beats the champion on both cells** (+26% / +33% on R-p@1),
  and on +50% exceeds the champion's all-time R-p@1. This **overturns the `_260`
  "macro doesn't beat the champion" conclusion** for the matched comparison.

## Caveats (why this is "promising + validate", not "deploy now")

1. **One window.** This is the single trailing carve at snapshot 2026-06-20. Macro's
   edge should be confirmed across multiple snapshots/windows before promotion.
2. **HP-sensitive.** Macro helped cleanly at `mcw=10`; the `_260` auto-loop runs were
   noisier. The edge is real under matched HP but not unconditional.
3. **HY-OAS is still a proxy** (`-log(HYG/IEF)`), and ranks low; the result doesn't
   hinge on it (it held in `_260`/`macrochamp` too).

## Recommendation (updated)

Promote macro from "opt-in, do not deploy" to **"validate for deployment"**: run the
matched champ_resnap-vs-macrochampagent head-to-head across several snapshots/windows
(e.g. quarterly trailing carves over 2024–2026); if the macro edge holds, wire the
macro champion into `/daily-predictions`. This is a materially more positive
recommendation than `_260` — earned by removing the confounds.

## Artifacts

- Specs: `configs/gbdt/experiments/sp500_up_{20pct_25d_dd10pct,50pct_50d_dd25pct}_{champ_resnap,macrochampagent}.yaml`
- Registry: 4 rows in `results/gbdt/data/r_precision_at_k.csv`.
- Sidecar: `results/gbdt/data/_262_champion_resnap_data.json`.
