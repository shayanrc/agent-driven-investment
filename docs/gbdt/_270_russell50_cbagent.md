# _270 — CatBoost agent-tune of the #1 cell (russell1000 +50%/200d) vs its sweep

**Question (`_269` follow-up):** `russell1000_up_50pct_200d_dd25pct_aligned` is the **#1 cell on the
date-aligned R-Precision@3 board** (CatBoost sweep, test @3 0.642). Two XGBoost agent-tunes already
tried and **both lost** to it. No CatBoost agent-tune existed. Motivated by the sibling sp500 +50%/200d
cell — where the CatBoost agent-tune *crushed* its sweep (`_269`: test @1 0.930 vs 0.630) — can a
CatBoost agent-tune dethrone #1? **Answer: a narrow, mixed result — the cbagent edges the sweep at
4 of 5 K (@1/@5/@10/@20) but loses @3 (−0.055), the board's headline metric. It does not dethrone #1
at @3; it nudges ahead at @1. Against a now-*verified* bar (see below), this is a marginal win, not the
sp500/50-style breakout.**

## The bar was stale — so we resnapped it first

The #1 sweep had **failed `_266`'s inference faithfulness self-check** (`max_abs_diff` 0.165), leaving
its 0.737/0.642 numbers unverified. Per the `_268` discipline we re-ran the sweep on the current vintage
(`russell1000_up_50pct_200d_dd25pct_aligned_resnap`, `--snapshot-end 2026-06-20`, byte-identical spec).

**The resnap reproduces the stale bar to the decimal** (test @1 0.737, @3 0.642, @5 0.539, @10 0.466,
@20 0.396; AUC 0.697). So the `_266` failure was a **reproduction-path artifact** (the `infer_fresh`
feature-build path diverging from the training path), **not performance drift** — the #1 bar is
trustworthy, and the fair-bar verdict equals the stale-bar verdict. Same outcome as the three `_268`
resnaps. (This also retires task #23.)

## Results — CatBoost-agent vs the (verified) sweep vs XGBoost agents (test R-Precision@K + base_rate)

| model | @1 | @3 | @5 | @10 | @20 | base | AUC |
|---|---|---|---|---|---|---|---|
| **CatBoost agent** | **0.743** | 0.587 | **0.567** | **0.471** | **0.401** | 0.152 | 0.703 |
| CatBoost sweep (resnap = #1, verified) | 0.737 | **0.642** | 0.539 | 0.466 | 0.396 | 0.152 | 0.697 |
| XGBoost agent (v14p1) | 0.647 | 0.559 | 0.547 | 0.509 | 0.460 | 0.152 | — |
| XGBoost agent (plain) | 0.577 | 0.467 | 0.491 | 0.473 | 0.424 | 0.152 | — |

CatBoost-agent vs the sweep per K: **@1 +0.007, @3 −0.055, @5 +0.028, @10 +0.005, @20 +0.005**. It
**beats both XGBoost agents at @1/@3** (@1 +0.096 vs v14p1, +0.166 vs plain) — switching to the robust
backend is unambiguously the best agent-tune of this cell — but still cannot clear the CatBoost *sweep*
at @3. Test window 2023-07-27→2024-10-03 (test_end 2024-10-03), base 0.152, 300 days.

## Eval→test decay (again the whole story)

| metric | eval | test | decay |
|---|---|---|---|
| @1 | 0.900 | 0.743 | −0.157 |
| @3 | 0.687 | 0.587 | −0.100 |

The agent's combine pick had a *spectacular* eval (@1 0.900) that decayed −0.157 into the test window —
exactly the `_269` long-horizon disease (eval R-p is an unreliable single-window proxy at H=200). The
sub-agent correctly chose the **broad-not-@1-spiky** combine config per the `_269` rule (it passed over
`lr01_cs04_l2_1`, eval @1 0.910, for the broader `lr01_d4_cs04_l2_1`), which is why the model held up
across @1/@5/@10/@20 rather than collapsing — but the @3 band still decayed below the sweep's plain
defaults.

## Reading

- **The sweep's un-FS'd, un-subsampled defaults hold the @2–@3 band better.** The cbagent FS-prefit to
  53 features (from 279) and subsampled (`rsm=0.4`); it gained at @1 and the broad tail (@5–@20) but
  gave up @3, where the sweep's full-feature model ranks the 2nd/3rd pick better. This is the prime
  suspect for the @3 loss (parked as investigation #26).
- **No HP/backend story dethrones #1 at @3.** Three agent-tunes (2× XGBoost, 1× CatBoost) have now all
  failed to beat the cheap ≤3-iter CatBoost sweep at @3 on this cell. The sweep is robust here.
- **Consistent with `_269`'s high-variance verdict** — but milder: this is neither the sp500/50 breakout
  nor the russell/40 & sp500/40 @1 *losses*; it's a near-tie that tips slightly to the agent at @1 and
  to the sweep at @3.
- **The #1 bar is real and reproducible** — the resnap confirms it, closing the `_266` "is the top cell
  even trustworthy?" question for this cell (yes).

## Verdict

**The CatBoost ≤3-iter sweep remains the #1 cell's robust default.** The CatBoost agent-tune is the best
agent variant (beats both XGBoost agents) and edges the sweep across most of the curve, but it does
**not** dethrone #1 at the headline @3 — so there is no reason to replace the sweep here. As in `_269`,
agent-tuning CatBoost at H=200 is opportunistic, not a dependable improvement; the real fix is the
eval objective (multi-window / decay-penalized, V1.5_TBD #2), not the backend.

## Recommendation

- **Keep the CatBoost sweep as the #1-cell baseline.** Do not promote the cbagent (it's a @1-only edge
  against a @3 loss; H=200 isn't deployed in `/daily-predictions` anyway).
- **Investigate the @3 regression (#26)** — chiefly the FS-prefit effect — against this verified bar.
- **Promote V1.5_TBD #2 before any further H=200 agent-tuning push.**

## Artifacts

- Cells committed: `25bb679` (cbagent), `4b50015` (sweep resnap); registry rows
  `russell1000_up_50pct_200d_dd25pct_aligned_{cbagent,resnap}`.
- Sidecar: `results/gbdt/data/_270_russell50_cbagent_data.json`.
