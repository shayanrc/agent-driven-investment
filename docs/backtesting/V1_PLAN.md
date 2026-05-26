# backtesting — V1 Implementation Plan

## Build status

- **v1.0 spec-lock:** `docs/backtesting/{goal,spec}.md` merged via PR #12 (2026-05-26). Subsequently amended (this PR) to lock the engine-level `fill_mode`, drop limit orders + per-order execution overrides, defer multi-frequency support to v1.1, redefine `advance_time()` as no-mutate-when-done, and introduce a configurable `gap_policy`. Locks the step-loop API, the single-phase execution lifecycle, the single-frequency timeline, the lot-size contract, the asset-class-agnostic stance, and the explicit non-goals (no margin, no rewards, no Gym subclass, no live broker).
- **v1.0 implementation plan:** Translates the (amended) spec into a concrete 8-stage build order with critical-correctness constraints, cross-module integration points, resolved questions, and test strategy. **All 8 stages implemented**:
  - Stage 1 — `Portfolio` ✅
  - Stage 2 — `DataHandler` + timeline correctness harness ✅
  - Stage 3 — `ExecutionBroker` ✅
  - Stage 4 — `_parse_action` + `_snap_to_lot` + action-validation utility ✅
  - Stage 5 — `Backtest` orchestrator ✅
  - Stage 6 — Strategy protocol + run-loop helper ✅ *(scope re-shaped from the original Stage 6 `from_identifiers` convenience — the data-fetch hook is parked in `V1.1_TBD.md`)*
  - Stage 7 — Result aggregation + info-schema lock validator ✅
  - Stage 8 — CLI entry point + smoketest config ✅ *(scope re-shaped from the original Stage 8 example-against-real-cache — pure synthetic smoketest only; live-cache example parked in `V1.1_TBD.md`)*

For *what success looks like* (the look-ahead-bias structural-elimination rule, B1–B7, scope limits), see [`goal.md`](goal.md). For the architectural decisions (D1 step-loop, D2 engine-level `fill_mode`, D3 single-frequency timeline with activity masks), the action / state schemas, the component APIs, and the lifecycle walkthrough, see [`spec.md`](spec.md). Parked v1.1 follow-ups are in [`V1.1_TBD.md`](V1.1_TBD.md).

---

## Purpose

This is an implementation specification for a **correct, configurable, multi-asset backtesting engine** that simulates order execution against historical data via a step-based loop. v1 ships the four core components (`DataHandler`, `Portfolio`, `ExecutionBroker`, `Backtest`), the structural look-ahead-elimination machinery (engine-level `fill_mode` + single-frequency timeline + per-asset activity masks), and the test infrastructure that verifies the B1–B7 correctness constraints cannot be violated by valid inputs.

The plan is the output of the design conversations reflected in `goal.md` and `spec.md`. Decisions documented here were made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating. Architectural decisions (D1 / D2 / D3 in `spec.md`) are out of scope for "silent change" — they shaped the entire design and should be re-litigated at the goal/spec level, not in implementation.

---

## Source of truth

This document is the **authoritative v1 implementation spec for the backtesting pipeline**. It is the build-order contract between `goal.md` (the why), `spec.md` (the what), and the code that lands in `src/backtesting/` (the how). When `goal.md` / `spec.md` and this plan disagree, escalate — don't silently re-decide.

Specifically:
- The 8-stage build order in § "Stage breakdown" is strict. Diagnostic infrastructure (Stages 2 and 7) is what makes the engine trustworthy, not the orchestrator (Stage 5). Do not skip ahead.
- The B1–B7 correctness constraints in § "Critical correctness constraints" are non-negotiable. Each constraint has a dedicated test surface in `tests/backtesting/test_correctness.py` (Stage 7). Any change that makes a constraint harder to verify, or any failed test on these constraints, blocks the PR.
- The cross-module dependencies in § "Cross-module dependencies" are the only integration points between `backtesting` and the rest of the project. Adding a new dependency on `analog_mc`, `gbdt`, or `forecasters` requires updating this section first.
- The architectural decisions in § "Architectural decisions" — and the resolved-design-decisions block in `spec.md` § 9 — together form the locked decision set. The "Resolved questions" section below records the 10 decisions made during plan review; everything in this plan should be treated as locked.

---

## Critical correctness constraints

These are non-negotiable invariants for v1, equivalent in role to `analog_mc`'s C1–C6 and `gbdt`'s leakage-harness gate. They map directly onto B1–B7 in `spec.md` § 7 and are reproduced here with the implementation-side test surface attached. Every constraint must hold under all valid input combinations, not just the ones we happened to test.

### B1. No look-ahead in state

`DataHandler.get_window()` returns data with indices `[current_step − lookback + 1, current_step]` inclusive. No row at index `> current_step` is ever included in the returned state, by integer slicing on the timeline.

**Test surface:** `tests/backtesting/test_correctness.py::test_b1_no_lookahead_in_state` — drive the engine across a synthetic dataset where every value at row `t` equals `t` (a "row-index sentinel"). Assert that every state returned at step `T` contains only sentinel values `≤ T` in every feed.

### B2. Engine-mode-correct execution

All orders submitted in a given engine fill at the bar selected by the engine's configured `fill_mode`. `fill_mode="current_close"` ⇒ T's close; `fill_mode="next_open"` ⇒ T+1's open. The state returned from `step()` reflects T+1 data (or T if `done=True`).

**Test surface:** `tests/backtesting/test_correctness.py::test_b2_fill_mode_execution` — drive a synthetic dataset where each bar's prices are a known function of the step index. Run the same scripted scenario twice, once with each `fill_mode`. Assert that every recorded `fill_price` in `info["fills"]` equals the bar / field pair implied by the engine's configured `fill_mode`. Assert that the returned `state["step"]` equals `T+1` (non-terminal) or remains parked (terminal).

### B3. Causal data access (forward-fill on internal gaps only)

With `gap_policy="ffill_zero_volume"` (the default), internal date gaps within an asset's active range are forward-filled from the previous bar; volume is zeroed. No backward-fill, no interpolation, no centering. At step T, every cell in `state["market_data"]` reflects either a true observation at or before T or a value carried forward from such an observation. Mid-series-start assets carry NaN before their first observation. Multi-frequency forward-fill across feeds is deferred to v1.1 (see Resolved Questions Q3).

**Test surface:** `tests/backtesting/test_correctness.py::test_b3_causal_data_access` — build a single-frequency feed with a deliberately injected internal gap; assert that the post-gap state's gap row equals the previous bar's price columns with zero volume, and that no future row leaked back. Also assert that **mid-series-start assets** carry NaN before their first observation (no backfill) and are marked untradeable (per D3 in `spec.md`).

### B4. Portfolio consistency

After every `step()`, `portfolio.equity == portfolio.cash + sum(position_qty * mark_price)` for all positions held. Cash is never negative. The overdraw guard in `Portfolio.execute_trade()` and the post-trade reconciliation in `Portfolio.update_valuations()` enforce this.

**Test surface:** `tests/backtesting/test_correctness.py::test_b4_portfolio_consistency` — run a 200-step random-action stress test, sampling buys / sells / weight-rebalances at every step against multi-asset data. After every `step()`, assert (a) `portfolio.cash >= 0`, (b) `portfolio.equity == portfolio.cash + Σ position * mark_price` within float tolerance.

### B5. Deterministic replay

Same `data_feeds` + same construction arguments + same sequence of `action` dicts ⇒ identical sequence of `(state, done, info)` tuples. No internal randomness exists in v1 (no fill-noise, no slippage model, no random tie-breakers).

**Test surface:** `tests/backtesting/test_correctness.py::test_b5_deterministic_replay` — run the same scripted scenario twice; compare each `(state, done, info)` tuple element-wise. Floating-point equality is required, not approximate — there are no operations in v1 that should introduce nondeterminism.

### B6. Lot-size integrity

Every filled order quantity is a valid multiple of the instrument's lot size, per the `lot_sizes` dict / `default_lot_size`. The engine never fills a fractional quantity for a whole-share instrument, or a non-multiple for a round-lot instrument.

**Test surface:** `tests/backtesting/test_correctness.py::test_b6_lot_size_integrity` — drive weight-based actions that produce fractional ideal quantities. Assert every executed fill in `info` satisfies `qty % lot_size == 0` (with `lot_size == 0` treated as the "fractional, no check" case). Cover all three lot-size regimes: `0` (fractional), `1` (whole shares), `N` (round lots of N).

### B7. Terminal-step contract (no-mutate-when-done)

`DataHandler.advance_time()` is no-mutate-when-done — at the terminal bar it returns `True` without incrementing `current_step`. The handler stays parked at `max_steps - 1`; the invariant `current_step ∈ [lookback, max_steps - 1]` holds for the entire engine lifetime. For `fill_mode="next_open"`, any orders still pending when `done=True` are reported under `info["rejected_untradeable"]` (the T+1 bar that would have filled them does not exist).

**Test surface:** `tests/backtesting/test_correctness.py::test_b7_terminal_step_contract` — run the engine to the last bar with a pending `next_open` order. Assert (a) the order is in `info["rejected_untradeable"]`, (b) `done == True`, (c) the state's `timestamp` equals the last bar's date, (d) `data_handler.current_step == max_steps - 1` post-done (no off-by-one drift), (e) `portfolio.equity` reflects the last bar's close-price mark. Also assert that a second `step(None)` after done keeps `current_step` at `max_steps - 1` (idempotent terminal state).

---

## Cross-module dependencies

Backtesting is a **consumer** of upstream modules' outputs and a **producer** of trade-execution artifacts that downstream evaluation code can read. v1 keeps the consumption surface as narrow as possible — DataFrames in, fills out — so the engine stays asset-class-agnostic and not coupled to any specific forecaster.

### What backtesting CONSUMES

#### `data_pipelines` (optional convenience wrapper)

- **Function:** `data_pipelines.fetch(identifier, start, end) → pd.DataFrame` (the canonical OHLCV schema in `src/data_pipelines/schema.py`: `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`).
- **Where:** an optional convenience constructor `Backtest.from_identifiers(identifiers, start, end, **kwargs)` that wraps `data_pipelines.fetch()` per identifier and assembles the `data_feeds` dict. The core `Backtest.__init__` continues to accept raw DataFrames — making `data_pipelines` a soft dependency, not a hard one (per `spec.md` § 9 #1 and `goal.md` § "How to apply this").
- **Cache contract:** `data_pipelines.fetch()` is cache-served on hit (sub-second), cold pull on miss. The convenience constructor respects the cache; no separate cache layer in `backtesting`.
- **Domain coverage in v1:** `us_equities` + `nse_equities` ship in `data_pipelines`. Macro feeds (FRED-style `(date, value)` series) are a separate domain — not built yet in `data_pipelines`. v1 is single-frequency-per-engine (resolved Q3): the multi-frequency master-timeline + slower-feed forward-fill machinery is deferred to v1.1, removing the macro-adapter dependency from the v1 critical path.

#### `forecasters` (zero v1 dependency; v1.1 hook only)

- v1 has **no direct import** of `forecasters`. The engine consumes trading signals as `action` dicts, not as forecaster outputs. A caller may build the action dict from a `forecasters` result, but that translation is the caller's responsibility, not the engine's.
- v1.1 may add a thin `backtesting.signal_adapters` module that translates `forecasters` outputs into actions — parked in `V1.1_TBD.md`.

#### `gbdt` (zero v1 dependency; v1.1 hook only)

- v1 has **no direct import** of `gbdt`. The engine has no concept of "probability of event"; it sees only orders.
- `gbdt` produces calibrated probabilities at `results/gbdt/experiments/<name>/predictions/*.csv` with schema `(date, ticker, p_raw, p_calibrated, y_true)`. A separate v1.1 strategy module can read these and emit actions; the engine doesn't see the predictions directly.
- **Integration risk:** none for v1, because the contract is action-shaped, not probability-shaped. The engine's design correctly punts the probability-to-action translation to the caller.

#### `analog_mc` (zero v1 dependency; v1.1 hook only)

- v1 has **no direct import** of `analog_mc`. Same logic as `gbdt`: paths-to-actions translation is the caller's job.
- `analog_mc` produces path samples + summary quantiles via `analog_mc.simulate.forecast()`. A path-based strategy (e.g., "sell when quantile-90 of next-20-day return < −0.05") is downstream of the engine.

### What backtesting PRODUCES

- **In-memory state stream** — the `(state, done, info)` 3-tuple per `step()`. This is the engine's primary output; everything else is derived from it. See `spec.md` § 3.2 for the schema.
- **Diagnostic `info` payload per step** — fills executed (asset, qty, price, commission, execution mode), rejected orders, lot-size rounding adjustments, weight-rebalancing shortfalls, cancelled-at-terminal orders. Sufficient detail that a caller can reconstruct trade history without instrumenting the engine.
- **No artifact directory in v1.** The engine is a Python class returning Python objects; persistence (trade ledgers, equity curves, attribution) is a v1.1 concern (see `V1.1_TBD.md` § "Trade-ledger persistence"). Strategy authors that want to persist results compose the state stream into their own files outside the engine.

The asymmetry — rich consumption story, minimal production story — is intentional. The engine is a primitive; it executes orders. What gets logged, plotted, and aggregated is the caller's concern. This mirrors `analog_mc`'s "the module produces probabilities, downstream owns evaluation" stance.

---

## Stage breakdown

The build order is strict. Each stage ends with a passing test suite and a commit. Don't skip ahead. As with `analog_mc`'s plan, the **diagnostic infrastructure (Stage 2: timeline + sentinel + gap-policy harness; Stage 7: B1–B7 constraint suite) is what makes the engine trustworthy**, not the orchestrator (Stage 5: `Backtest`).

Effort sizing: S = <1 day, M = 1–2 days, L = 3–5 days, XL = >1 week.

### Stage 1 — `Portfolio` (S)

**Goal:** ship the ledger primitive. The simplest component; trivial to test in isolation.

**Files added:**
- `src/backtesting/__init__.py` — package skeleton.
- `src/backtesting/portfolio.py` — `Portfolio` class per `spec.md` § 4.2.

**Tasks:**
- Implement `__init__(initial_cash: float)`, `execute_trade(asset, qty, price)`, `update_valuations(current_prices)`, `get_state()`, `reset()`.
- `qty` is `float` per spec (lot enforcement is upstream).
- `execute_trade` raises if a buy would make cash negative (the direction-aware overdraw check is in the broker per `spec.md` § 4.2; the portfolio's own guard is the last-line-of-defense `raise`).
- `get_state()` returns `{cash, equity, positions, unrealized_pnl}`.
- Positions stored as `dict[str, float]`; absent key ≡ zero position.

**Tests added:**
- `tests/backtesting/test_portfolio.py`:
  - Construction → state matches `initial_cash`.
  - Single buy / single sell → cash + position delta correct.
  - Multiple trades on same asset → position accumulates (signed).
  - `update_valuations()` with known prices → equity == cash + Σ pos*price.
  - Overdraw → raises.
  - Short sale (`qty < 0`) → cash increases, position negative, no overdraw raise.
  - `reset()` → restores to `initial_cash`, clears positions.

**Diagnostic infrastructure:** none yet (Stage 1 is too small).

**Done when:** `pytest tests/backtesting/test_portfolio.py` passes; `portfolio.py` is the only file under `src/backtesting/`.

### Stage 2 — `DataHandler` + timeline correctness harness (M)

**Goal:** ship the timekeeping primitive. This is where D3 (single-frequency timeline + per-asset activity masks) and Q9's `gap_policy` live. Get this right and B1 (no look-ahead) becomes structurally enforced; get it wrong and the engine silently leaks future data into the state.

**Files added:**
- `src/backtesting/data_handler.py` — `DataHandler` class per `spec.md` § 4.1.
- `src/backtesting/utils.py` — shared helpers (timeline construction, NaN-aware reindex, gap detection). Created here, extended by later stages.

**Tasks:**
- Implement `__init__(data_feeds, lookback, gap_policy)`, `_align`, `advance_time()`, `get_current_bar()`, `get_window()`, `get_price(asset, field)`, `reset()`.
- **Single-frequency timeline construction (D3):**
  - The timeline is the **union** of all dates across all assets across all feeds. All feeds share the same trading calendar in v1; cross-frequency forward-fill is deferred to v1.1 per Q3.
  - For each asset, reindex its DataFrame to the timeline.
  - Mid-series-start assets → NaN before first observation (do NOT backfill); mark untradeable on those dates.
  - Mid-series-end assets → forward-fill last known value; mark untradeable after last date.
  - Internal-gap handling per `gap_policy` (Q9):
    - `"raise"` → any missing date within an asset's active range fails construction with a clear error.
    - `"ffill_zero_volume"` (default) → forward-fill price columns from the previous bar; set the `volume` column to zero. The gap day is then a valid bar with zero traded volume.
  - Untradeable marker → boolean mask stored alongside the panel, queried by the broker in Stage 4.
- `current_step` starts at `lookback`, not 0 (so `get_window()` always returns a full-length window).
- `get_window()` returns dict `{feed_name: {asset_name: ndarray of shape (lookback, n_columns)}}`. Half-open Python indexing: `data[step − lookback + 1 : step + 1]`.
- `advance_time()` is **no-mutate-when-done** (Q8): if `current_step + 1 >= max_steps`, return `True` without incrementing. Otherwise increment and return `False`.

**Tests added:**
- `tests/backtesting/test_data_handler.py`:
  - Single-feed single-asset construction → timeline length correct, `current_step` = lookback.
  - Multi-asset same-frequency construction → timeline is union of asset dates; each asset reindexed; untradeable masks correct.
  - Mid-series-start asset → NaN before, untradeable flag asserted.
  - Mid-series-end asset → forward-fill after, untradeable flag asserted.
  - `get_window()` at first valid step → full-length window, no nulls in the active range.
  - `advance_time()` at last bar → returns `True`; `current_step` unchanged (Q8 no-mutate).
  - Repeated `advance_time()` after done → still returns `True`; `current_step` still pinned to `max_steps - 1`.
  - `reset()` → `current_step` back to `lookback`.
  - **Q9 internal-gap policy:**
    - `test_internal_gap_raises_when_gap_policy_is_raise` — build a feed with a deliberately-injected missing date inside an asset's active range; assert `DataHandler(..., gap_policy="raise")` raises with a clear error.
    - `test_internal_gap_ffills_when_gap_policy_is_default` — same input; assert `DataHandler(..., gap_policy="ffill_zero_volume")` constructs cleanly, the gap-day row equals the previous bar's price columns, and the gap-day volume is 0.

**Diagnostic infrastructure (the trustworthiness layer):**
- `tests/backtesting/test_data_handler.py::test_row_index_sentinel_no_lookahead` — the "row-index sentinel" harness referenced in B1. Build a synthetic feed where every value at row `t` equals `t`. Step through the timeline. Assert that at step `T`, no value in `get_window()` exceeds `T`. This is the canonical look-ahead-leak detector — every later stage that touches data access must keep this test green.

**Done when:** single-frequency timeline construction works on synthetic inputs; the row-index sentinel harness passes; mid-series start/end edge cases are tested in both directions; both `gap_policy` modes are tested; `advance_time()` no-mutate-when-done is asserted (and idempotent under repeated calls).

### Stage 3 — `ExecutionBroker` (S)

**Goal:** ship the order-queue primitive. v1's broker is dramatically simpler than the original two-phase design: a single fill path against whatever bar the orchestrator hands in. ~30 LOC of fill logic, not ~150 — the engine-level `fill_mode` (Q6) collapses the per-order branching out of the broker entirely.

**Files added:**
- `src/backtesting/broker.py` — `ExecutionBroker` per `spec.md` § 4.3.

**Tasks:**
- Implement `__init__(commission_fn)`, `submit_orders(orders)`, `process_queue(current_data, portfolio)`, `get_pending_count()`, `reset()`.
- **Order validation in `submit_orders`:** exactly the two required fields `{asset: str, qty: float}`; `qty != 0`; `asset` is a known instrument. Unknown fields (`execution`, `limit_price`, `time_in_force`) are rejected with details captured for `info["rejected_invalid"]`. The strict schema is intentional — it forces v1.1 to make a deliberate API extension when those fields come back.
- **`process_queue`** is fill_mode-agnostic. The orchestrator decides which bar to hand in (T's bar for `current_close`, T+1's bar for `next_open`); the broker fills all pending orders against whatever is provided. No phase argument, no per-order branching, no limit-price evaluation.
- **Sells before buys.** Stable sort by `qty` sign (negative first) before processing. Cash freed by sells funds subsequent buys.
- **Direction-aware overdraw guard** (per `spec.md` § 4.2):
  - Buys: `qty * price + commission <= cash` before filling.
  - Sells: `commission <= cash + abs(qty) * price` (post-fill cash covers commission).
  - Orders that fail the guard are skipped, recorded in the returned fill log under `rejected_overdraw`.
- **Untradeable assets** (from Stage 2): broker rejects buys for untradeable assets, permits sells (filled at last-known price). Captured under `rejected_untradeable`. In `fill_mode="next_open"`, when the engine is done and no T+1 bar exists, every still-pending order is reported under `rejected_untradeable` by the orchestrator (the broker itself just doesn't run).
- **No partial fills in v1** — order either fills completely or not at all.
- `process_queue` returns a fill log: `{filled: [...], rejected_overdraw: [...], rejected_untradeable: [...], rejected_invalid: [...]}`. This becomes the spine of `info` in `Backtest.step()`.

**Tests added:**
- `tests/backtesting/test_broker.py`:
  - `submit_orders` validates and rejects malformed orders (missing `asset`, missing `qty`, `qty == 0`, extra fields like `execution` / `limit_price`).
  - `process_queue` against a known bar → fill prices match the bar's relevant column (close or open depending on what the caller hands in).
  - Sells process before buys: scenario with cash-funding dependency demonstrates this.
  - Buy overdraw → order rejected, cash unchanged, no exception.
  - Sell of untradeable (delisted) asset → permitted at last-known price.
  - Buy of untradeable asset → rejected.
  - `commission_fn` applied to each fill; cash deducted post-trade.

**Diagnostic infrastructure:** order-log schema is itself the diagnostic — every fill / rejection must be traceable to the bar that produced it. Add a `tests/backtesting/test_broker.py::test_fill_log_audit_trail` that runs a multi-order scenario and asserts every submitted order ID appears exactly once in the union of `filled` and `rejected_*` lists.

**Done when:** the single-phase fill path is tested; sells-before-buys tested; direction-aware overdraw guard tested; untradeable handling tested; commission application tested; audit-trail invariant holds. The whole file should fit on one screen.

### Stage 4 — `_parse_action` + `_snap_to_lot` + action-validation utility (M)

**Goal:** the action-translation layer that sits between the caller's `action` dict and the broker's `submit_orders`. Pulled out as its own stage because the **weight-based action translation has the most subtle correctness obligations** in the engine (uses pre-fill equity, snaps to lot, sequences sells before buys) — easier to verify in isolation than wrapped in the `Backtest` orchestrator.

**Files added:**
- `src/backtesting/utils.py` (extended) — `parse_action(action, portfolio, data_handler, lot_sizes, default_lot_size) → list[dict]` and `snap_to_lot(qty, lot_size) → float`.

**Tasks:**
- `snap_to_lot(qty, lot_size)`:
  - `lot_size == 0` → return `qty` unchanged.
  - `lot_size == 1` → `int(qty)` (truncate toward zero: `2.7 → 2`, `−2.7 → −2`).
  - `lot_size == N` → `floor(abs(qty) / N) * N * sign(qty)`.
- `parse_action`:
  - `action is None` → return `[]` (no-op).
  - `action["type"] == "order"` → validate each order has exactly `{asset, qty}` (Q6 — no `execution`, no `limit_price`), snap qty to lot, record `lot_size_audit[asset] = {"requested_qty": original_qty, "filled_qty": snapped_qty}` for every order where the snap changed the value (including snap-to-zero, per Q10's unified key), drop orders with snapped qty == 0, return list.
  - `action["type"] == "weight"`:
    - Validate `sum(target_weights.values()) <= 1.0` (Q1) — raise `ValueError` on over-allocation.
    - Compute current pre-fill equity from `portfolio.cash + Σ pos * current_close_price` (close is hard-coded per Q4).
    - For each asset in `target_weights`:
      - Target position = `target_weight * equity / current_close_price`.
      - Delta = target_position − current_position.
      - Snap delta to lot; record in `lot_size_audit` if snap changed it.
      - If snapped delta != 0, emit an order `{asset, qty: snapped_delta}` (no `execution` field — engine `fill_mode` decides at fill time).
    - For assets in `positions` but NOT in `target_weights`: their target weight is 0 → emit a sell-to-zero order.
    - Sequence: sells first (qty < 0), then buys.
- Validation errors raise `ValueError` with the offending field. The caller is responsible for catching; the engine does NOT silently drop malformed actions.

**Tests added:**
- `tests/backtesting/test_utils.py`:
  - `snap_to_lot` for the three lot regimes (0, 1, N) with positive + negative + fractional inputs.
  - `parse_action` with `None` → empty list.
  - `parse_action` with order-type action → orders pass through, qty snapped, `lot_size_audit` populated for snapped-or-zeroed entries only.
  - `parse_action` with order containing `execution` field → `ValueError` (extra-field rejection per Q6).
  - `parse_action` with weight-type action against known portfolio + prices → emitted orders math out by hand.
  - Weight-type action drops asset from previous positions → sell-to-zero emitted.
  - Weight-type action with `sum(target_weights) > 1.0` → `ValueError` raised (Q1).
  - Order with snapped qty == 0 → entry in `lot_size_audit` with `filled_qty: 0`, order not emitted (Q10 unified key).

**Diagnostic infrastructure:** the `lot_size_audit` payload (Q10) — every difference between requested and actual qty is logged under one key. This is what makes B6 visible to the caller, not just enforced internally.

**Done when:** `parse_action` covers all three action variants (None, order, weight); lot snapping covers all three lot regimes; `lot_size_audit` records exist for every quantity adjustment (rounding or zeroing); weight-action sells-before-buys preserved; over-allocation raises per Q1; extra fields on orders are rejected per Q6.

### Stage 5 — `Backtest` orchestrator (M)

**Goal:** the single-phase `step()` lifecycle that wires Stages 1–4 together. This is the public API. The engine's `fill_mode` (Q6) selects which of two near-identical paths runs — there is no per-order branching anywhere in the orchestrator.

**Files added:**
- `src/backtesting/backtest.py` — `Backtest` per `spec.md` § 4.4.

**Tasks:**
- Implement `__init__(data_feeds, initial_cash, lookback, lot_sizes, default_lot_size, commission_fn, fill_mode, gap_policy)`. Construct internal `DataHandler` (passing `gap_policy`), `Portfolio`, `ExecutionBroker`. Store `fill_mode` and validate it's one of `{"current_close", "next_open"}` at construction time.
- Implement `reset() → (state, done, info)`. Reset all three components; return state at the first valid step; `done=False`, `info={}`.
- Implement `step(action) → (state, done, info)`:
  - **PARSE** — `orders, parse_audit = parse_action(action, self.portfolio, self.data_handler, ...)`. Submit valid orders to broker.
  - Branch on `fill_mode`:
    - `"current_close"`:
      1. **FILL** — `fill_log = broker.process_queue(data_handler.get_current_bar(), portfolio)`. Fills against T's close.
      2. **ADVANCE** — `done = data_handler.advance_time()`. T → T+1, or pinned at T if done (Q8 no-mutate).
      3. **MARK** — `portfolio.update_valuations(close_prices_at_current_step)`.
    - `"next_open"`:
      1. **ADVANCE** — `done = data_handler.advance_time()`. T → T+1, or pinned at T if done.
      2. **FILL** — if not done, `fill_log = broker.process_queue(data_handler.get_current_bar(), portfolio)`. If done, all pending orders are routed to `fill_log["rejected_untradeable"]` (the T+1 bar that would have filled them does not exist); broker is not invoked.
      3. **MARK** — `portfolio.update_valuations(close_prices_at_current_step)`.
  - **RETURN** — assemble state from `data_handler.get_window()` + `portfolio.get_state()` + step index + timestamp; pack `info` from `parse_audit` + `fill_log` using the locked Q10 key names (omit any key whose payload is empty / all-zero); return `(state, done, info)`.
- State assembly: build the dict shape from `spec.md` § 3.2 exactly. Include `step`, `timestamp` (ISO string from data_handler's current date), `market_data`, `portfolio` (cash, equity, positions, pending_orders).
- **Weight-drift accounting:** after MARK, compute realized weights from `positions * close_price / equity`; compare to the action's `target_weights` (if any); populate `info["weight_drift"]` per Q10 (omit assets where drift is zero).

**Tests added:**
- `tests/backtesting/test_backtest.py`:
  - `reset()` produces valid state at first step.
  - `step(None)` advances time, no orders placed.
  - `step({"type": "order", ...})` with `fill_mode="current_close"` fills at T's close.
  - `step({"type": "order", ...})` with `fill_mode="next_open"` fills at T+1's open.
  - `step({"type": "weight", ...})` rebalances correctly given known prices; `info["weight_drift"]` populated only for non-zero drifts.
  - **Terminal step (Q8 no-mutate):** at the last bar, `done=True`, `data_handler.current_step == max_steps - 1`; a second `step(None)` after done keeps `current_step` pinned.
  - **Terminal step (next_open):** pending order at terminal step appears in `info["rejected_untradeable"]`, last state returned, equity reflects last close.
  - State schema matches spec (keys present, types correct).
  - `info` keys are emitted only when non-empty (Q10).
  - Multi-step scenario (e.g., 50 steps with mixed orders) executes without exception.

**Diagnostic infrastructure:** the `info` payload is the diagnostic — every observable side-effect of the step lands in `info` under the locked Q10 key names. Add `tests/backtesting/test_backtest.py::test_info_completeness` — run a scenario with at least one of every side-effect (fill, rejection, lot-zero, weight-drift, rebalance-shortfall, terminal-untradeable) and assert each appears in the corresponding `info` key.

**Done when:** both `fill_mode` paths execute correctly; Q8 no-mutate terminal-path tested in both directions (single-step and repeated); state schema matches spec; Q10 info-key emission rules respected; multi-step scenarios run clean.

### Stage 6 — `Backtest.from_identifiers` convenience constructor (S)

**Goal:** the `data_pipelines.fetch()`-wrapping constructor. Soft dependency only; the core API stays DataFrame-shaped.

**Files added:**
- `src/backtesting/backtest.py` (extended) — `Backtest.from_identifiers(identifiers, start, end, feed_name="equities", lookback=20, **kwargs) → Backtest`.

**Tasks:**
- For each identifier in `identifiers` (single string OR list), call `data_pipelines.fetch(identifier, start, end)`.
- Assemble `data_feeds = {feed_name: {identifier: df_for_identifier}}`.
- Forward remaining kwargs to `Backtest.__init__`.
- If `data_pipelines` import fails (the soft-dependency case), raise `ImportError` with a clear message: "Install data_pipelines or pass `data_feeds` directly."

**Tests added:**
- `tests/backtesting/test_backtest_from_identifiers.py`:
  - Single-identifier cold path mocked → constructor produces working `Backtest`.
  - Multi-identifier → assembled `data_feeds` has correct shape.
  - Missing identifier raises through `data_pipelines.AllProvidersFailed`.
  - Real-cache integration test (skipped if no cache present) for a known identifier.

**Done when:** the convenience constructor exists, is tested, but is not the path the core engine depends on.

### Stage 7 — B1–B7 correctness suite + leakage harness (L)

**Goal:** the trustworthiness layer. Pull all correctness tests into one place, parameterized over many input shapes, so a regression on any B-constraint is immediately visible. This is the analog to `analog_mc`'s Stage 6/9 diagnostic infrastructure and `gbdt`'s leakage harness.

**Files added:**
- `tests/backtesting/test_correctness.py` — the B1–B7 suite.
- `src/backtesting/leakage_harness.py` — synthetic-data builders for B1 / B2 / B3 stress tests.

**Tasks:**
- Implement `leakage_harness.row_index_sentinel_feed(n_dates, n_assets, n_columns)` — produces a dataset where every value at row `t` equals `t` (the canonical B1 test substrate).
- Implement `leakage_harness.step_function_price_feed(...)` — produces a feed where bar T's open/close/high/low are known deterministic functions of T (the B2 test substrate).
- Implement `leakage_harness.internal_gap_feed(...)` — produces a single-frequency feed with a deliberately-injected missing date inside an asset's active range, for the B3 forward-fill / gap-policy test.
- Implement `leakage_harness.random_action_stream(n_steps, n_assets, seed)` — produces a reproducible random action sequence (order-type and weight-type, no `execution` field — Q6) for the B4 / B5 stress tests.
- Each of B1–B7 has at least one dedicated test (as enumerated in § "Critical correctness constraints" above) plus one randomized stress test driven by the harness.
- The B2 test runs the same scripted scenario twice (once per `fill_mode`) and asserts every fill price matches the engine's configured phase.
- The B5 deterministic-replay test runs the same scripted scenario **twice** and compares every step's output element-wise.
- The B7 test asserts `current_step ∈ [lookback, max_steps - 1]` at every step (including post-done), driven by the random-action stress harness.

**Tests added:** the B1–B7 suite in `test_correctness.py` — one test function per constraint per harness substrate.

**Diagnostic infrastructure:** the leakage harness itself **IS** the diagnostic infrastructure. Each later change to the engine must keep the B1–B7 suite green; the harness provides synthetic substrates that no realistic dataset can match. This is what makes "look-ahead-bias elimination is structural, not conventional" (the goal.md rule) verifiable.

**Done when:** B1–B7 all have dedicated tests + stress tests; the leakage harness produces the four substrate types; running `pytest tests/backtesting/test_correctness.py` is the single command that proves the engine's correctness contract.

### Stage 8 — Example notebook + README + PR merge gate (S)

**Goal:** an end-to-end runnable example that exercises every action type and demonstrates the engine on real data. PR merge gate.

**Files added:**
- `docs/backtesting/example_buy_and_hold.md` — narrative walkthrough: load NIFTY 50 ticker via `Backtest.from_identifiers`, run a buy-and-hold strategy for 100 steps, assert equity curve length, dump fills.
- `scripts/backtesting/example_buy_and_hold.py` — the runnable script the doc references.
- `README.md` (root) — small section under "Modules" linking `docs/backtesting/goal.md` + `spec.md` + this `V1_PLAN.md`.

**Tasks:**
- The example loads one identifier (e.g., `NSE:RELIANCE`), runs a strategy that buys 10 shares at step 5 and holds, steps to done.
- Print the equity curve, the trade list (from `info` logs across steps), and the final portfolio state.
- The script is callable as `uv run python scripts/backtesting/example_buy_and_hold.py`.
- The PR merge gate is: (a) the example script runs end-to-end against the live `data_pipelines` cache, (b) all of `tests/backtesting/` passes, (c) `tests/backtesting/test_correctness.py` covers all of B1–B7.

**Tests added:**
- `tests/backtesting/test_example.py` — invokes the example script as a subprocess against a mock or seeded data feed; asserts the exit code and the existence of expected stdout sections.

**Done when:** the example script runs clean against real cached data; all of `tests/backtesting/` passes; `pytest tests/backtesting/` is the single command that proves v1 ships.

---

## Architectural decisions

The three locked-in decisions live in `spec.md` § 2 (D1 step-loop, D2 engine-level `fill_mode`, D3 single-frequency timeline with activity masks). The decisions below are **implementation-side** choices that follow from those three but warrant explicit defense.

### IA1. Components are concrete classes, not protocols / ABCs

`DataHandler`, `Portfolio`, `ExecutionBroker`, `Backtest` are concrete classes. No `DataHandlerProtocol`, no `BrokerABC`, no plugin registry in v1.

**Why not the alternative:** A plugin architecture (e.g., a `Broker` ABC with `LiveBroker` and `BacktestBroker` implementations) would make sense if we were shipping multiple broker implementations in v1. We're not — `ExecutionBroker` is the only broker. Introducing an ABC for a single implementation is over-engineering; it crystallizes a contract before we know the actual flex points. The right time to extract a protocol is when the second implementation lands (paper-trading-against-live-data is the obvious v2 candidate).

### IA2. `Portfolio.positions` is `dict[str, float]`, not a fully-typed `Position` dataclass

Positions are stored as `{asset_name: signed_quantity}`. No per-position cost basis, no entry timestamp, no per-position PnL.

**Why not the alternative:** A `Position` dataclass with cost basis would enable per-position realized/unrealized PnL — useful for tax-lot reporting, FIFO/LIFO accounting, etc. None of these are in v1 scope (no PnL is one of `analog_mc`'s anti-rules and `goal.md` § "What this module is *not*" excludes margin / risk). The minimal `dict[str, float]` shape is sufficient for v1 and trivially upgradable when a real downstream consumer demands tax-lot accounting.

### IA3. `info` is a free-form dict; no `StepInfo` dataclass

`info` is a `dict[str, list[dict]]` whose schema is documented in this plan and `spec.md` but not enforced by a typed dataclass.

**Why not the alternative:** A typed `StepInfo` dataclass would catch consumer-side bugs at construction time. But it would also tightly couple the engine to its current diagnostic shape — every new `info` field becomes a contract change. v1's diagnostic surface is still being discovered (every B-constraint test adds something we'll want to surface). Free-form dict now; promote to a dataclass after v1 ships and the schema stabilizes.

### IA4. No persistence layer in v1

The engine returns Python objects per step. It does not write any files. There is no `backtest.save_results(path)` method.

**Why not the alternative:** A persistence layer would standardize the output and make backtests comparable across runs. But "what to persist" is a strategy-dependent question (equity curve? trade ledger? per-step state? per-asset attribution?) and v1 should let the caller decide. Mirrors `analog_mc`'s "this module produces probabilities, downstream owns evaluation" stance. v1.1 may add a `backtesting.results.TradeLedger` writer behind a flag.

### IA5. Sells-before-buys is sequenced inside both `_parse_action` and `process_queue`

The ordering happens twice: once when weight-actions are converted into orders, once when the broker processes the pending queue.

**Why not the alternative:** A single sort point (broker-only) would be simpler. But weight-action translation needs the ordering to compute correct deltas (so the sell of asset A frees cash to compute the buy of asset B against post-sell equity). And the broker needs the ordering to actually fund the buys. Both points are real; consolidating them is a v1.1 refactor candidate once the patterns stabilize.

### IA6. `Backtest.from_identifiers` lives in `backtest.py`, not a separate `loaders.py`

The convenience constructor that depends on `data_pipelines` sits next to the core constructor it wraps.

**Why not the alternative:** A separate `loaders.py` would isolate the optional dependency. But it would scatter the public surface across two files. Keeping it in `backtest.py` and lazy-importing `data_pipelines` inside the method body achieves the same isolation without splitting the API.

### IA7. `gap_policy` is engine-level, not per-feed

`gap_policy` is a single `Backtest.__init__` arg (`"raise"` or `"ffill_zero_volume"`, default the latter), applied uniformly across every asset in every feed. Per-feed overrides (e.g., one policy for equities, another for an FX feed) are deferred to v1.1.

**Why not the alternative:** A per-feed override dict (`gap_policy_per_feed: dict[str, str]`) would be ergonomically nicer for mixed portfolios but requires the multi-frequency machinery to be useful — and Q3 deferred that to v1.1. Until there are multiple genuinely-different feed types in scope, a single engine-level knob captures every real use case. Once v1.1's multi-frequency support lands, promoting the knob to per-feed is a backward-compatible extension (the engine-level value becomes the per-feed default).

---

## Resolved questions

The 10 plan-review questions are now closed. Decisions below are the locked v1 contract; everything follows from them. Items that turned into spec amendments are noted; the four amendments together are recorded in the footer of `spec.md` § 6.

| # | Question | Decision | Spec impact |
|---|---|---|---|
| Q1 | Weight-action over-allocation: raise vs accept-and-skip? | **Raise** on `sum(target_weights.values()) > 1.0`. Caller must normalize. | `spec.md` § 3.1 amended. |
| Q2 | Staggered-history `wait_for_all_active` knob? | **No knob.** Keep spec-default NaN/untradeable per D3 — caller handles NaN columns for not-yet-started assets. | None (already in D3). |
| Q3 | Macro feed in v1? | **Defer multi-frequency entirely to v1.1.** v1 is single-frequency-per-engine; the master-timeline + cross-frequency forward-fill machinery moves to v1.1, gated on a `data_pipelines` macro domain. | Spec amendment: D3 rewritten, §3.2 macro example removed, §4.1 single-frequency. |
| Q4 | `close` vs `adj_close` as trading price? | **Hard-code `close` (raw)** in v1. Per-feed knob with `adj_close` option deferred to v1.1. | `spec.md` § 6 notes the hard-code. |
| Q5 | Commission signature? | **Ship `(asset, qty, price) → float`** as spec'd. The plan's recommendation to expand to `(order_dict, fill_price)` is rejected. | None (matches existing spec). |
| Q6 | MOC vs MOO per-order semantics? | **Engine-level `fill_mode: Literal["current_close", "next_open"] = "next_open"`** constructor arg. Not per-order. LIMIT orders DROPPED entirely from v1. Order schema is just `{asset, qty}` — no `execution`, no `limit_price`, no time-in-force. | Spec amendment: §3.1, §4.3, §4.4, §5, §6, B2, B7 all rewritten. |
| Q7 | Per-feed default lot sizes? | **Ship spec as-is.** `lot_sizes: dict[str, int]` overrides + `default_lot_size: int = 1` global fallback. No per-feed default in v1. | None (matches existing spec). |
| Q8 | `advance_time()` at terminal step: mutate or no-mutate? | **No-mutate-when-done.** `advance_time()` checks if `current_step + 1 >= max_steps`; if so returns `done=True` WITHOUT incrementing. Invariant: `current_step ∈ [lookback, max_steps - 1]` for the entire engine lifetime. | Spec amendment: §4.1, B7. |
| Q9 | Internal-gap policy? | **Configurable `gap_policy: Literal["raise", "ffill_zero_volume"] = "ffill_zero_volume"`** constructor arg. Default forward-fills price columns + zeros volume (treats gap day as "no trading happened"). Strict mode opts into raise. | Spec amendment: D3, §4.1, §6, B3. |
| Q10 | `info` schema key names? | Locked 3 keys: `weight_drift: dict[asset, float]` (realized − target, omit if zero); `rebalance_shortfall: dict[asset, float]` (per-asset shortfall, omit if zero); `lot_size_audit: dict[asset, {"requested_qty": float, "filled_qty": int}]` (unified key for rounding + zeroing, omit if requested == filled). | Spec amendment: §3.2. |

Notes:
- Q3 and Q6 are load-bearing simplifications: they remove ~50% of v1's Stage-3 broker complexity and ~30% of Stage-2 DataHandler complexity by deferring features that are not on any near-term consumer's critical path.
- Q5 (rejected expansion) and Q7 (rejected per-feed default) are conservative choices — they hold the v1 API surface area down so v1.1 is free to extend without breaking callers.
- The four spec amendments (Q3 / Q6 / Q8 / Q9) are noted in a footer in `spec.md` § 6.

---

## Testing strategy

Five layers, each enforced by `pytest tests/backtesting/`:

| Layer | What gets tested | Where |
|---|---|---|
| Unit (per component) | Each of Portfolio / DataHandler / ExecutionBroker / utils tested in isolation against scripted inputs | `test_portfolio.py`, `test_data_handler.py`, `test_broker.py`, `test_utils.py` |
| Integration (orchestrator) | `Backtest.step()` lifecycle across all action types; terminal-step path; multi-step scenarios | `test_backtest.py` |
| Convenience constructor | `Backtest.from_identifiers` against mock + real cache | `test_backtest_from_identifiers.py` |
| Correctness (B1–B7) | Each B-constraint has a dedicated test using the leakage harness's synthetic substrates | `test_correctness.py` |
| End-to-end | Example script runs against live cache; equity curve + trade log produced | `test_example.py` |

**Test data philosophy:**
- Unit tests use hand-written synthetic DataFrames (small, deterministic, hand-computable).
- Correctness tests use the `leakage_harness.py` synthetic substrates (row-index sentinel, step-function prices, macro-release feed, random action streams). These are the analog of `gbdt`'s leakage harness and `analog_mc`'s causal-test substrates — synthetic by design, so a "passes on real data" result is meaningless if the substrate test fails.
- The end-to-end test uses real `data_pipelines` cache data; it asserts shape and lifecycle, not specific dollar values.

**No regression in the diagnostic infrastructure.** Once Stage 7's B1–B7 suite passes, every later commit must keep it green. CI fails the suite → PR blocked. The B-constraints are the contract; tests are how the contract is enforced.

**Stress tests, not just unit tests.** B4 and B5 each include a randomized stress test that runs 200+ steps of mixed actions on multi-asset data, asserting the invariants per step. Catching B-violations only on the scripted-scenario tests would miss exactly the kind of interaction bugs (e.g., partial sell-then-buy in same step under untradeable transitions) that randomized scenarios surface.

---

## v1 vs v1.1 scope

v1 explicitly ships **the four core components + the structural correctness machinery**. Anything that adds modeling realism (margin, borrow costs, slippage, partial fills) or downstream surface area (persistence, signal adapters, dashboards) is deferred. See [`V1.1_TBD.md`](V1.1_TBD.md) for the full parking lot.

**In v1:**
- `Portfolio`, `DataHandler`, `ExecutionBroker`, `Backtest` per `spec.md` §§ 4.1–4.4.
- The single-phase execution lifecycle with engine-level `fill_mode` (Q6) per `spec.md` § 5.
- Single-frequency timeline + per-asset activity masks per D3.
- Order (schema `{asset, qty}` only) and weight and None action types per `spec.md` § 3.1.
- Per-instrument lot sizes (`0`, `1`, `N`) per `spec.md` § 4.5.
- Commission function hook with signature `(asset, qty, price) → float` (Q5) per `spec.md` § 6.
- Configurable `gap_policy` (Q9) per `spec.md` § 4.1.
- Hard-coded `close` (raw) as the fill price column (Q4).
- `advance_time()` no-mutate-when-done semantics (Q8).
- Locked `info` schema (Q10): `fills`, `rejected_overdraw`, `rejected_untradeable`, `rejected_invalid`, `weight_drift`, `rebalance_shortfall`, `lot_size_audit`.
- Weight-action over-allocation raises (Q1).
- Untradeable-asset handling (mid-series start, delisted, etc.) per D3.
- B1–B7 correctness test suite + leakage harness.
- Convenience constructor wrapping `data_pipelines.fetch()`.
- One example script demonstrating end-to-end usage.

**Deferred to v1.1+ (in `V1.1_TBD.md`):**
- Margin requirements, borrow costs, locate requirements (short selling stays unconstrained in v1 per `spec.md` § 4.2 design note).
- Partial fills.
- Limit orders, per-order `execution` override, time-in-force (GFD / GTC / GTD / IOC / FOK) (Q6).
- Slippage model.
- YAML-driven config layer (v1 is constructor-args-only per `spec.md` § 6).
- Multi-frequency feeds (master-timeline-driven slower-feed forward-fill) (Q3).
- Macro domain (FRED-style `(date, value)`) in `data_pipelines` (Q3 dependency).
- `adj_close` price-column knob (Q4).
- Per-feed default lot sizes (Q7).
- Per-feed `gap_policy` overrides (Q9).
- Expanded commission signature `(order_dict, fill_price) → float` for maker/taker/tiered models (Q5).
- Intrabar / multi-resolution data (v1 is single-frequency-per-feed).
- Cross-calendar alignment (v1 assumes all feeds share trading calendar; D3 design note).
- Trade-ledger / equity-curve persistence.
- Signal adapters (forecaster output → action dict translation).
- Multi-broker plugin architecture.
- `gymnasium.Env` adapter wrapper.
- Streamlit dashboard.
- Typed dataclasses for `state` / `info` (currently dicts).

---

## File layout

```
docs/backtesting/
  goal.md                              # done (PR #12)
  spec.md                              # done (PR #12)
  V1_PLAN.md                           # this PR
  V1.1_TBD.md                          # this PR

src/backtesting/
  __init__.py                          # Stage 1
  portfolio.py                         # Stage 1
  data_handler.py                      # Stage 2
  utils.py                             # Stages 2 + 4 (extended)
  broker.py                            # Stage 3
  backtest.py                          # Stages 5 + 6 (extended)
  leakage_harness.py                   # Stage 7

tests/backtesting/
  __init__.py
  test_portfolio.py                    # Stage 1
  test_data_handler.py                 # Stage 2
  test_broker.py                       # Stage 3
  test_utils.py                        # Stage 4
  test_backtest.py                     # Stage 5
  test_backtest_from_identifiers.py    # Stage 6
  test_correctness.py                  # Stage 7 (B1–B7 + leakage harness)
  test_example.py                      # Stage 8

scripts/backtesting/
  example_buy_and_hold.py              # Stage 8

configs/backtesting/                   # reserved (empty in v1; YAML config deferred to v1.1)
```

---

## Branch + PR plan

- This work lives on the `backtesting-v1-plan` branch (this PR; doc-only).
- Implementation lands on subsequent branches: each stage in the build (1–8) is a logical commit on an implementation branch (`backtesting-v1-impl` or similar). The implementation PR opens against `main` once Stage 8's example is green and the B1–B7 suite passes.
- Per `[[feedback-branch-retention]]`, branches stay after merge.
- Per `CLAUDE.md`, no AI attribution in commits or PR text.

---

## Dependencies (additions expected)

- v1 has **no new third-party dependencies**. The engine uses `numpy`, `pandas`, and the standard library — all already in `pyproject.toml` for `analog_mc` / `gbdt`.
- The `Backtest.from_identifiers` convenience constructor depends on `data_pipelines` (already an in-repo editable install).
- Tests use `pytest` (already present).

---

## Compute envelope

The engine is CPU-bound and single-threaded. v1 has no parallelization (per-step is sequential; a strategy that runs many backtests in parallel is the caller's responsibility).

- **`step()` overhead per call:** dominated by `DataHandler.get_window()` (a small slice operation on cached arrays) and `broker.process_queue()` (linear in pending order count). On daily-bar data with ~10 assets and ~5 pending orders, expect microsecond-to-millisecond step times.
- **End-to-end backtest (10 years daily, single asset, buy-and-hold):** ~2,500 steps × <1 ms = sub-second.
- **End-to-end backtest (10 years daily, 50-asset universe, daily rebalance via weight-action):** ~2,500 steps × ~10 ms = tens of seconds.
- **Memory:** `O(n_dates × n_assets × n_columns)` for the panel, plus `O(lookback × n_assets × n_columns)` returned per step. On a typical NIFTY-50 / 10-year setup: ~50 assets × ~2,500 dates × ~6 columns × 8 bytes = ~6 MB panel; lookback windows are dwarfed.

The engine is intentionally not optimized in v1. If profile-driven optimization becomes a concern, the natural target is `get_window()` (currently a per-step Python-level slice) — vectorizing into a single pre-allocated buffer is a v1.1 micro-optimization.

---

## References

- [`docs/backtesting/goal.md`](goal.md) — what success looks like; the structural-elimination-of-look-ahead-bias rule that shapes the design.
- [`docs/backtesting/spec.md`](spec.md) — the full system specification: D1–D3 architectural decisions, action / state schemas, component APIs, the 6-step lifecycle, B1–B6 correctness constraints, the resolved design decisions block.
- [`docs/backtesting/V1.1_TBD.md`](V1.1_TBD.md) — parked v1.1 follow-ups.
- [`docs/analog_mc/IMPLEMENTATION_PLAN.md`](../analog_mc/IMPLEMENTATION_PLAN.md) — the analog plan-style reference (11-stage build, C1–C6 constraints, diagnostic-first Stage 6/9).
- [`docs/gbdt/V1_PLAN.md`](../gbdt/V1_PLAN.md) — the categorical-outcome plan reference (9-stage build, decisions log, leakage-harness gate).
- [`docs/data_pipelines/goal.md`](../data_pipelines/goal.md) — the data-fetch surface the convenience constructor wraps.
- [`CLAUDE.md`](../../CLAUDE.md) — repo-wide conventions: module namespacing, plans-on-branches rule, what-not-to-do per module, environment setup.
