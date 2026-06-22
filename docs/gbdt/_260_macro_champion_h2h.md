# _260 — Phase 3: macro vs the champion head-to-head (the macro effect sign-flips)

**Headline:** Under the **champion regime** (trailing split + auto FS+HP loop), the
macro panel's effect **sign-flips across the two cells** — it *hurts* +20%/25d
(R-p@1 0.347→0.120) and *helps* +50%/50d (R-p@1 0.120→0.360) — and this flips
*again* relative to the date-aligned A/B (`_259`), where macro decisively helped
+20%. **Macro is not a robust improvement, and does not beat the deployed
champions.** Recommendation: keep F17 as an opt-in feature; **do NOT promote a
macro model to the champion / `/daily-predictions`.** The "outperform the
existing models" objective is **not achieved**.

## Setup

Matched A/B under the champions' own regime: **trailing split** (default, not
date_aligned) + `callback_mode: default` auto FS+HP loop (`max_iterations 5`,
converged at n_iter=3 both cells), xgboost, `--snapshot-end 2026-06-20`.
`champbase` (F1–F16) vs `macrochamp` (F1–F16 + F17, the 8-series real panel,
40 macro cols). Within each cell the two arms share an identical trailing test
window + base rate (matched). Committed champions shown for reference only —
they were scored on an **earlier, different window** (see caveat).

## Results (test segment; raw R-Precision@K + base rate)

### sp500 +20%/25d (dd10%) — trailing test 2026-01-22→2026-05-08, base_rate 0.0878, Q=75

| K | champbase | macrochamp | committed champion (ref) |
|---|---|---|---|
| R-Precision@1 | **0.3467** | 0.1200 | 0.4133 |
| R-Precision@3 | **0.2978** | 0.1867 | — |
| R-Precision@5 | **0.2827** | 0.2000 | — |
| R-Precision@10 | **0.3067** | 0.2693 | 0.4027 |
| R-Precision@20 | **0.3003** | 0.2766 | — |
| test AUC | 0.7393 | 0.7389 | — |

Macro **hurts** at every K (R-p@1 −65%, R-p@10 −12%) despite consuming 22.5% of
model gain (top: `macro_VIXCLS_chg_60`). In-sample macro structure that did not
generalize to this window.

### sp500 +50%/50d (dd25%) — trailing test 2026-01-22→2026-04-02, base_rate 0.0384, Q=50

| K | champbase | macrochamp | committed champion (ref) |
|---|---|---|---|
| R-Precision@1 | 0.1200 | **0.3600** | 0.6400 |
| R-Precision@3 | 0.1600 | **0.3133** | — |
| R-Precision@5 | 0.2040 | **0.3360** | — |
| R-Precision@10 | 0.2165 | **0.3050** | 0.3460 |
| R-Precision@20 | 0.3132 | **0.4153** | — |
| test AUC | 0.8562 | 0.8644 | — |

Macro **helps** at every K (R-p@1 +200%, R-p@10 +41%; top: `macro_T10YIE_level`,
the breakeven series, + `DGS10_chg_60`). But R-p@1 0.360 is still well below the
committed champion's 0.640.

## Two findings

1. **The macro effect is regime-dependent and sign-unstable — the decisive
   result.** Same 8-series panel, same cells:
   - `_259` date_aligned A/B: macro **helped +20%** (every K) and was mixed on +50%.
   - `_260` trailing/champion regime: macro **hurt +20%** and **helped +50%**.

   A feature whose sign flips with the split scheme / evaluation window is not a
   reliable edge. This mirrors the playbook's "no universal recipe for strong-top-1
   cells" (`[[project-gbdt-tuning-playbook]]`) — now extended to the macro family.

2. **The matched `champbase` is far below the agent-tuned committed champion**
   (+50%: 0.120 vs 0.640; +20%: 0.347 vs 0.413). The default auto-loop
   over-regularizes the strong-top-1 prediction tail (exactly the playbook's
   warning), so the default-auto arms cannot reach the champion's hand-tuned skill.
   **The vs-committed-champion comparison is therefore confounded on two axes:**
   the window differs (base_rate 0.0384 here vs 0.0257 committed on the +50% cell —
   a 50% richer positive rate, a different holdout) AND the tuning mode differs
   (default-auto vs agent_file_protocol). Read the macro effect off the **matched
   `champbase` arm**, not the recorded champion numbers.

## Verdict

- **Macro does NOT robustly beat the baseline, and does not beat the deployed
  champions.** It helps in some (cell, regime) combinations and hurts in others —
  net not a dependable improvement. The objective "outperform the existing models"
  is **not met**.
- **Keep F17 as a merged, opt-in feature** (it's causal, cheap, and genuinely
  additive on the +50% cell under trailing split / the +20% cell under
  date_aligned). **Do not promote a macro model to the champion or wire it into
  `/daily-predictions`** — the sign-instability makes it unsafe as a default.

## What would settle it definitively (not pursued — low expected value)

An agent-tuned (`agent_file_protocol`) `macrochamp` on a window matched to a
re-run committed champion would remove the tuning-mode + window confounds. But
given the sign-flip across `_259`/`_260`, a *robust* macro win is unlikely;
chasing it would be over-fitting to the +50% cell. If revisited, also close the
`BAMLH0A0HYM2` (HY credit OAS) gap first — credit spreads are the one macro axis
still missing.

## Artifacts

- Specs: `configs/gbdt/experiments/sp500_up_{20pct_25d_dd10pct,50pct_50d_dd25pct}_{champbase,macrochamp}.yaml`
- Registry: 4 rows in `results/gbdt/data/r_precision_at_k.csv`.
- Sidecar: `results/gbdt/data/_260_macro_champion_h2h_data.json`.
