# _280 — F19 agent-protocol finetune of the top-6 sweep candidates

**Question.** Does the agent-driven FS+HP loop (`callback_mode: agent_file_protocol`)
extract a better **test top-book** from the F19 (`all_fundamentals2`) models than the
untuned single-fit did in the `_279` sweep? Six cells, the union of **top-3-by-AUC** and
**top-3-by-R-p@3** among the `_279` window-1 F19 arms.

**Setup.** Each cell tuned on the **same window** as the `_279` sweep (date-aligned
`train_start: 2019-01-01`, test 2024-07-26→2024-12-16, `--snapshot-end 2026-07-06` to
reuse the warm `all_fundamentals2` universe cache), so iter 0 == the `_279` `f19{cb,xgb}`
single-fit and every delta is a clean tuned-vs-untuned comparison. The agent (this session)
drove each loop with a disciplined 1–2-probe screen (winner's-curse discipline per
`[[project-gbdt-tuning-playbook]]`), reading the eval R-p@K book + train/val gap each
iteration. Backends per the arm; `tie_band: 0.0` (strict val argmin at finalization).

## The six cells (top-3 by each metric among `_279` F19 arms)

| selected via | cell (sp500 up) | backend | base | single-fit AUC | single-fit R-p@3 |
|---|---|---|---:|---:|---:|
| R-p@3 | 20pct_100d | cb | 0.231 | 0.656 | 0.597 |
| R-p@3 | 50pct_200d | xgb | 0.071 | 0.723 | 0.483 |
| R-p@3 | 10pct_50d | cb | 0.335 | 0.579 | 0.477 |
| AUC | 50pct_50d | cb | 0.010 | 0.929 | 0.158 |
| AUC | 50pct_25d | cb | 0.002 | 0.918 | 0.069 |
| AUC | 50pct_25d | xgb | 0.002 | 0.905 | 0.069 |

The two lists are disjoint → 6 arms. The top-3-by-AUC are all **ultra-rare** 50pct cells
(base 0.002–0.010) where R-p@1/@3 is tiny-count noise and the high AUC is rare-event
inflation.

## Result — tuned vs single-fit test book (window 1)

Raw R-Precision@K, test window 2024-07-26→2024-12-16. "tuned" = the finalized `_f19*agent`
artifact (the loop's val-argmin pick).

| cell | backend | base | arm | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 20pct_100d | cb | 0.231 | single-fit | 0.656 | 0.220 | 0.597 | 0.530 | 0.458 | 0.418 |
| | | | tuned (depth 4) | 0.657 | 0.410 | 0.567 | 0.604 | 0.467 | 0.411 |
| | | | eval-leader (l2 1) | 0.658 | 0.330 | 0.530 | 0.458 | 0.431 | 0.438 |
| 50pct_200d | xgb | 0.071 | single-fit | 0.723 | 0.610 | 0.483 | 0.408 | 0.288 | 0.264 |
| | | | tuned (lambda 5) | 0.742 | 0.200 | 0.213 | 0.230 | 0.223 | 0.197 |
| 10pct_50d | cb | 0.335 | single-fit | 0.579 | 0.540 | 0.477 | 0.476 | 0.445 | 0.427 |
| | | | tuned (= defaults) | 0.579 | 0.540 | 0.477 | 0.476 | 0.445 | 0.427 |
| 50pct_50d | cb | 0.010 | single-fit | 0.929 | 0.209 | 0.158 | 0.180 | 0.209 | 0.394 |
| | | | tuned (= defaults) | 0.929 | 0.209 | 0.158 | 0.180 | 0.209 | 0.394 |
| 50pct_25d | cb | 0.002 | single-fit | 0.918 | 0.150 | 0.069 | 0.108 | 0.322 | 0.522 |
| | | | tuned (= defaults) | 0.918 | 0.150 | 0.069 | 0.108 | 0.322 | 0.522 |
| 50pct_25d | xgb | 0.002 | single-fit | 0.905 | 0.150 | 0.069 | 0.136 | 0.322 | 0.478 |
| | | | tuned (= defaults) | 0.905 | 0.150 | 0.069 | 0.136 | 0.322 | 0.478 |

## Reading

**4 of 6 cells: the loop finds nothing better than defaults.** On the three ultra-rare
50pct cells (base 0.002–0.010) the train/val gap is ≤ 0 (val Brier *below* train — no
overfit to exploit; the model sits at the noise floor) and on the high-base `10pct_50d`
the gap is a well-regularized +0.017. Every probe (l2↓ tail-sharpen, gentle L2↑) is a val
wash, so the val argmin stays iter 0 and the finalized artifact **is** the single-fit,
byte-for-byte on the test book.

**`20pct_100d` cb: the book shuffles within noise — no arm dominates, and the in-loop
pick is not the test-best.** depth-4 (the val argmin) wins @1/@5/@10 (0.22→0.41,
0.53→0.60, 0.458→0.467) but loses @3 (0.597→0.567); the l2-1 config that led the **eval**
book at every K lands only middling on test (@3 0.53, @5 0.458 — *below* the single-fit).
The crown trades at every K across the three arms. You cannot pick the test-best arm from
any in-loop signal — the eval-book leader is not the test leader, and the val argmin is a
different arm again.

**`50pct_200d` xgb: tuning actively collapses the test book — the sharpest `_276`
confirmation yet.** The lambda-5 arm improved *val* (0.0944→0.0937), the *eval* book (@3
0.485→0.61, @5 0.44→0.594), **and** *test AUC* (0.723→0.742) — and its test top-book fell
off a cliff: @1 0.610→0.200, @3 0.483→0.213, @10 0.288→0.223. Three independent in-loop
signals *and* a held-out bulk-ranking metric (test AUC) all moved opposite to the test
top-book. L2 regularization flattens the extreme prediction tail that top-K consumes while
every average-case metric improves.

## Verdict

**Agent-protocol finetuning does not robustly beat the F19 single-fit on any of the six
cells.** Four finalize back onto defaults; one shuffles the book within noise with no
in-loop-selectable winner; one has its book destroyed by a config that improves val + eval
+ test-AUC. This is a clean **negative** result that *strengthens* `_276`/`_279`: on these
top-K cells the in-loop signals (val Brier, eval R-p@K) — and even held-out **test AUC** —
anti-select the test top-book, so there is no reliable lever for the agent to pull beyond
the untuned fit. It also re-confirms the `_279` headline: **F19 unlocks no tunable edge**;
the durable finding of the fundamentals arc remains the **backend** (CatBoost, `_277`/`_278`),
not more fundamentals features or more tuning.

**No window-2 pass is warranted.** The only cell with a window-1 book gain (`20pct_100d`,
depth-4 @1/@5/@10) is (a) **not selectable** from in-loop signals — the loop's own pick
(l2-1 eval-leader) is worse on test — and (b) a **partial-K shuffle**, not a clean win. An
un-selectable, partial gain does not clear the bar for a second-window confirmation
(parked as optional in `V1.8_TBD` §). Nothing here changes the champion set or
`/daily-predictions`.

## Byproduct — FS drops the single-quarter QoQ (answers the `_279` design question)

Across the catboost cells, F19 importance concentrates in **`fund_rev_ttm_qoq_xs_zscore`**
(3.02 on `20pct_100d`, 1.46 on `10pct_50d`), with `fund_rev_ttm_yoy_pct_xs_zscore` and
`fund_rev_q_yoy_xs_zscore` a distant second. The **single-quarter measures**
(`fund_rev_q_yoy`, `fund_rev_q_qoq`) sit at **~zero importance** — CatBoost's
importance-FS drops the single-quarter QoQ on its own, directly confirming the "if it's
bad FS should drop it" hypothesis behind adding it in `_279`. The TTM-QoQ sequential
z-score is the only F19 column the models lean on.

## Artifacts

Specs `configs/gbdt/experiments/sp500_up_*_f19{cb,xgb}agent.yaml` (+ the `_f19cbagentEL`
eval-leader 1-shot). Registry: 7 rows in `results/gbdt/data/r_precision_at_k.csv`
(`mode=agent_loop` for the six finalized arms, `single_fit` for the eval-leader). Chain:
`_279` (F19 sweep) → **`_280`** (agent-tune of its top-6). Related: `_276` (F18 top-3
agent-tune — same anti-selection), `_277`/`_278` (CatBoost the robust finding).
