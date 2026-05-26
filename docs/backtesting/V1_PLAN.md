# backtesting — V1 Implementation Plan

## Build status

- **v1.0 spec-lock:** `docs/backtesting/{goal,spec}.md` merged via PR #12 (2026-05-26). Locks the step-loop API, the two-phase execution lifecycle, the master-timeline construction, the lot-size contract, the asset-class-agnostic stance, and the explicit non-goals (no margin, no rewards, no Gym subclass, no live broker).
- **v1.0 implementation plan:** this PR. Translates the spec into a concrete 8-stage build order with critical-correctness constraints, cross-module integration points, open questions, and test strategy. Stages 1–8 below are pending implementation.

For *what success looks like* (the look-ahead-bias structural-elimination rule, B1–B6, scope limits), see [`goal.md`](goal.md). For the architectural decisions (D1 step-loop, D2 two-phase lifecycle, D3 master timeline + forward-fill), the action / state schemas, the component APIs, and the lifecycle walkthrough, see [`spec.md`](spec.md). Parked v1.1 follow-ups are in [`V1.1_TBD.md`](V1.1_TBD.md).

---

## Purpose

This is an implementation specification for a **correct, configurable, multi-asset backtesting engine** that simulates order execution against historical data via a step-based loop. v1 ships the four core components (`DataHandler`, `Portfolio`, `ExecutionBroker`, `Backtest`), the structural look-ahead-elimination machinery (two-phase lifecycle + master timeline), and the test infrastructure that verifies the B1–B6 correctness constraints cannot be violated by valid inputs.

The plan is the output of the design conversations reflected in `goal.md` and `spec.md`. Decisions documented here were made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating. Architectural decisions (D1 / D2 / D3 in `spec.md`) are out of scope for "silent change" — they shaped the entire design and should be re-litigated at the goal/spec level, not in implementation.

---

## Source of truth

This document is the **authoritative v1 implementation spec for the backtesting pipeline**. It is the build-order contract between `goal.md` (the why), `spec.md` (the what), and the code that lands in `src/backtesting/` (the how). When `goal.md` / `spec.md` and this plan disagree, escalate — don't silently re-decide.

Specifically:
- The 8-stage build order in § "Stage breakdown" is strict. Diagnostic infrastructure (Stages 2 and 7) is what makes the engine trustworthy, not the orchestrator (Stage 5). Do not skip ahead.
- The B1–B6 correctness constraints in § "Critical correctness constraints" are non-negotiable. Each constraint has a dedicated test surface in `tests/backtesting/test_correctness.py` (Stage 7). Any change that makes a constraint harder to verify, or any failed test on these constraints, blocks the PR.
- The cross-module dependencies in § "Cross-module dependencies" are the only integration points between `backtesting` and the rest of the project. Adding a new dependency on `analog_mc`, `gbdt`, or `forecasters` requires updating this section first.
- The architectural decisions in § "Architectural decisions" — and the resolved-design-decisions block in `spec.md` § 9 — together form the locked decision set. The "Open questions" section below lists what is *not* yet decided; everything else should be treated as locked.

---

## Critical correctness constraints

These are non-negotiable invariants for v1, equivalent in role to `analog_mc`'s C1–C6 and `gbdt`'s leakage-harness gate. They map directly onto B1–B6 in `spec.md` § 7 and are reproduced here with the implementation-side test surface attached. Every constraint must hold under all valid input combinations, not just the ones we happened to test.

### B1. No look-ahead in state

`DataHandler.get_window()` returns data with indices `[current_step − lookback + 1, current_step]` inclusive. No row at index `> current_step` is ever included in the returned state, by integer slicing on the master timeline.

**Test surface:** `tests/backtesting/test_correctness.py::test_b1_no_lookahead_in_state` — drive the engine across a synthetic dataset where every value at row `t` equals `t` (a "row-index sentinel"). Assert that every state returned at step `T` contains only sentinel values `≤ T` in every feed.

### B2. Phase-ordered execution

`current_close` orders fill in Phase 1 (before `advance_time()`) at bar T's close. `next_open` and `limit` orders fill in Phase 2 (after `advance_time()`) at bar T+1's open / OHLC bar. The state returned from `step()` reflects T+1 data.

**Test surface:** `tests/backtesting/test_correctness.py::test_b2_phase_ordered_execution` — drive a synthetic dataset where each bar's prices are a known function of the step index. Submit one order per execution mode. Assert that the recorded fill price in `info["fills"]` for the `current_close` order equals bar-T's close, and the fill price for the `next_open` order equals bar-(T+1)'s open. Assert that the returned `state["step"]` equals `T+1`.

### B3. Causal forward-fill

Lower-frequency feeds are forward-filled from past observations only. No backward-fill, no interpolation, no centering. At step T, every cell in `state["market_data"]` for slow-updating data reflects the most recent observation at or before T.

**Test surface:** `tests/backtesting/test_correctness.py::test_b3_causal_forward_fill` — build a daily-equities feed plus a quarterly-macro feed with known release dates. Assert that the macro value visible at every step `T` equals the most recent release at or before T's date. Also assert that **mid-series-start assets** carry NaN before their first observation (no backfill) and are marked untradeable (per D3 in `spec.md`).

### B4. Portfolio consistency

After every `step()`, `portfolio.equity == portfolio.cash + sum(position_qty * mark_price)` for all positions held. Cash is never negative. The overdraw guard in `Portfolio.execute_trade()` and the post-trade reconciliation in `Portfolio.update_valuations()` enforce this.

**Test surface:** `tests/backtesting/test_correctness.py::test_b4_portfolio_consistency` — run a 200-step random-action stress test, sampling buys / sells / weight-rebalances at every step against multi-asset data. After every `step()`, assert (a) `portfolio.cash >= 0`, (b) `portfolio.equity == portfolio.cash + Σ position * mark_price` within float tolerance.

### B5. Deterministic replay

Same `data_feeds` + same construction arguments + same sequence of `action` dicts ⇒ identical sequence of `(state, done, info)` tuples. No internal randomness exists in v1 (no fill-noise, no slippage model, no random tie-breakers).

**Test surface:** `tests/backtesting/test_correctness.py::test_b5_deterministic_replay` — run the same scripted scenario twice; compare each `(state, done, info)` tuple element-wise. Floating-point equality is required, not approximate — there are no operations in v1 that should introduce nondeterminism.

### B6. Lot-size integrity

Every filled order quantity is a valid multiple of the instrument's lot size, per the `lot_sizes` dict / `default_lot_size`. The engine never fills a fractional quantity for a whole-share instrument, or a non-multiple for a round-lot instrument.

**Test surface:** `tests/backtesting/test_correctness.py::test_b6_lot_size_integrity` — drive weight-based actions that produce fractional ideal quantities. Assert every executed fill in `info` satisfies `qty % lot_size == 0` (with `lot_size == 0` treated as the "fractional, no check" case). Cover all three lot-size regimes: `0` (fractional), `1` (whole shares), `N` (round lots of N).

### B7. Terminal-step contract

When `advance_time()` returns done at Phase 1 → Phase 2 transition, the engine **must not** attempt to fill Phase-2 orders against a nonexistent T+1 bar. Pending `next_open` / `limit` orders are cancelled and reported in `info["cancelled"]`. The returned state reflects T (the last available bar), marked to T's close prices.

**Test surface:** `tests/backtesting/test_correctness.py::test_b7_terminal_step_contract` — run the engine to the last bar with a pending `next_open` order. Assert (a) the order is in `info["cancelled"]`, (b) `done == True`, (c) the state's `timestamp` equals the last bar's date, (d) `portfolio.equity` reflects the last bar's close-price mark.

> B7 isn't in `spec.md` § 7's B-list but the terminal-path behavior **is** in `spec.md` § 5's lifecycle walkthrough. We promote it to a constraint here because it's the place a naive implementation will silently produce NaN fills or off-by-one state, and the cost of testing it is low.

---

## Cross-module dependencies

Backtesting is a **consumer** of upstream modules' outputs and a **producer** of trade-execution artifacts that downstream evaluation code can read. v1 keeps the consumption surface as narrow as possible — DataFrames in, fills out — so the engine stays asset-class-agnostic and not coupled to any specific forecaster.

### What backtesting CONSUMES

#### `data_pipelines` (optional convenience wrapper)

- **Function:** `data_pipelines.fetch(identifier, start, end) → pd.DataFrame` (the canonical OHLCV schema in `src/data_pipelines/schema.py`: `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`).
- **Where:** an optional convenience constructor `Backtest.from_identifiers(identifiers, start, end, **kwargs)` that wraps `data_pipelines.fetch()` per identifier and assembles the `data_feeds` dict. The core `Backtest.__init__` continues to accept raw DataFrames — making `data_pipelines` a soft dependency, not a hard one (per `spec.md` § 9 #1 and `goal.md` § "How to apply this").
- **Cache contract:** `data_pipelines.fetch()` is cache-served on hit (sub-second), cold pull on miss. The convenience constructor respects the cache; no separate cache layer in `backtesting`.
- **Domain coverage in v1:** `us_equities` + `nse_equities` ship in `data_pipelines`. Macro feeds (FRED-style `(date, value)` series) are a separate domain — not built yet in `data_pipelines`. **Risk:** if v1 wants to demonstrate the master-timeline forward-fill on real macro data (e.g., quarterly GDP), the macro adapter must exist first. See OQ-3 in § "Open questions".

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

The build order is strict. Each stage ends with a passing test suite and a commit. Don't skip ahead. As with `analog_mc`'s plan, the **diagnostic infrastructure (Stage 2: master timeline + leakage harness; Stage 7: B1–B7 constraint suite) is what makes the engine trustworthy**, not the orchestrator (Stage 5: `Backtest`).

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

### Stage 2 — `DataHandler` + master-timeline correctness harness (M)

**Goal:** ship the timekeeping primitive. This is where D3 (master timeline + forward-fill) lives. Get this right and B1 (no look-ahead) becomes structurally enforced; get it wrong and the engine silently leaks future data into the state.

**Files added:**
- `src/backtesting/data_handler.py` — `DataHandler` class per `spec.md` § 4.1.
- `src/backtesting/utils.py` — shared helpers (master-timeline construction, NaN-aware reindex). Created here, extended by later stages.

**Tasks:**
- Implement `__init__(data_feeds, lookback)`, `_align_and_fill`, `advance_time()`, `get_current_bar()`, `get_window()`, `get_price(asset, field)`, `reset()`.
- **Master timeline construction (D3):**
  - Determine the highest-frequency feed = the feed whose constituent DataFrames have the greatest number of **unique dates** across the union of all assets in that feed.
  - The master timeline is the **union** of all dates across all assets in the highest-frequency feed.
  - For every other feed, reindex each asset's DataFrame to the master timeline.
  - Higher-frequency feeds → no fill needed (they define the timeline).
  - Lower-frequency feeds → forward-fill (`pandas.DataFrame.ffill()`) per asset.
  - Mid-series-start assets → NaN before first observation (do NOT backfill); mark untradeable on those dates.
  - Mid-series-end assets → forward-fill last known value; mark untradeable after last date.
  - Untradeable marker → boolean mask stored alongside the panel, queried by the broker in Stage 4.
- `current_step` starts at `lookback`, not 0 (so `get_window()` always returns a full-length window).
- `get_window()` returns dict `{feed_name: {asset_name: ndarray of shape (lookback, n_columns)}}`. Half-open Python indexing: `data[step − lookback + 1 : step + 1]`.
- `advance_time()` returns `True` when the increment would move `current_step` past `max_steps − 1` (no T+1 bar exists).

**Tests added:**
- `tests/backtesting/test_data_handler.py`:
  - Single-feed single-asset construction → master timeline length correct, `current_step` = lookback.
  - Two feeds different frequencies → forward-fill applied to slower; timeline driven by faster.
  - Mid-series-start asset → NaN before, untradeable flag asserted.
  - Mid-series-end asset → forward-fill after, untradeable flag asserted.
  - `get_window()` at first valid step → full-length window, no nulls in the active range.
  - `advance_time()` at last bar → returns `True`.
  - `reset()` → `current_step` back to `lookback`.

**Diagnostic infrastructure (the trustworthiness layer):**
- `tests/backtesting/test_data_handler.py::test_row_index_sentinel_no_lookahead` — the "row-index sentinel" harness referenced in B1. Build a synthetic feed where every value at row `t` equals `t`. Step through the timeline. Assert that at step `T`, no value in `get_window()` exceeds `T`. This is the canonical look-ahead-leak detector — every later stage that touches data access must keep this test green.

**Done when:** master timeline construction works on synthetic mixed-frequency inputs; the row-index sentinel harness passes; mid-series start/end edge cases are tested in both directions.

### Stage 3 — `ExecutionBroker` (M)

**Goal:** ship the order-queue primitive. This is where the two-phase lifecycle (D2) is mechanically enforced.

**Files added:**
- `src/backtesting/broker.py` — `ExecutionBroker` per `spec.md` § 4.3.

**Tasks:**
- Implement `__init__(commission_fn)`, `submit_orders(orders)`, `process_queue(current_data, portfolio, phase)`, `get_pending_count()`, `reset()`.
- **Order validation in `submit_orders`:** required fields (`asset`, `qty`, `execution`); `qty != 0`; `limit` orders must have `limit_price`; `execution` must be one of `{"current_close", "next_open", "limit"}`. Invalid orders are rejected with details captured for `info`.
- **Phase 1** (`phase=1`): process only `execution == "current_close"` orders against the current bar's close.
- **Phase 2** (`phase=2`): process `execution == "next_open"` orders against the current bar's open; process `execution == "limit"` orders against the current bar's OHLC (buy limit fills if low ≤ limit_price; sell limit fills if high ≥ limit_price; fill at limit_price).
- **Within each phase, sells before buys.** Stable sort by `qty` sign (negative first).
- **Direction-aware overdraw guard** (per `spec.md` § 4.2):
  - Buys: `qty * price + commission <= cash` before filling.
  - Sells: `commission <= cash + abs(qty) * price` (post-fill cash covers commission).
  - Orders that fail the guard are skipped, recorded in the returned fill log under `rejected_overdraw`.
- **Untradeable assets** (from Stage 2): broker rejects buys for untradeable assets, permits sells (filled at last-known price). Captured under `rejected_untradeable`.
- **Limit-order expiry:** unfilled limit orders after Phase 2 are removed and recorded under `expired_limits`.
- **No partial fills in v1** — order either fills completely or not at all.
- `process_queue` returns a fill log: `{filled: [...], rejected_overdraw: [...], rejected_untradeable: [...], rejected_invalid: [...], expired_limits: [...]}`. This becomes part of `info` in `Backtest.step()`.

**Tests added:**
- `tests/backtesting/test_broker.py`:
  - `submit_orders` validates and rejects malformed orders.
  - Phase 1 fills `current_close` order; Phase 2 leaves it alone.
  - Phase 2 fills `next_open` and `limit` orders; Phase 1 leaves them alone.
  - Buy limit at price P fills when bar's low ≤ P; doesn't fill otherwise.
  - Sell limit at price P fills when bar's high ≥ P; doesn't fill otherwise.
  - Unfilled limit after Phase 2 expires (good-for-day).
  - Sells process before buys within a phase: scenario with cash-funding dependency demonstrates this.
  - Buy overdraw → order rejected, cash unchanged, no exception.
  - Sell of untradeable (delisted) asset → permitted at last-known price.
  - Buy of untradeable asset → rejected.
  - `commission_fn` applied to each fill; cash deducted post-trade.

**Diagnostic infrastructure:** order-log schema is itself the diagnostic — every fill / rejection / expiry must be traceable to the bar that produced it. Add a `tests/backtesting/test_broker.py::test_fill_log_audit_trail` that runs a multi-order scenario and asserts every order ID appears exactly once in the union of `filled`, `rejected_*`, and `expired_limits` lists.

**Done when:** all execution modes fill at their spec'd prices in their spec'd phases; sells-before-buys within phase tested; direction-aware overdraw guard tested; untradeable handling tested; commission application tested; audit-trail invariant holds.

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
  - `action["type"] == "order"` → validate each order, snap qty to lot, drop orders with snapped qty == 0 (with `info["lot_size_zeroed"]` audit record), return list.
  - `action["type"] == "weight"`:
    - Compute current pre-fill equity from `portfolio.cash + Σ pos * current_close_price`.
    - For each asset in `target_weights`:
      - Target position = `target_weight * equity / current_close_price`.
      - Delta = target_position − current_position.
      - Snap delta to lot.
      - If delta != 0 after snap, emit an order with `qty = delta` and the shared `execution` mode.
    - For assets in `positions` but NOT in `target_weights`: their target weight is 0 → emit a sell-to-zero order.
    - Sequence: sells first (qty < 0), then buys.
- Validation errors raise `ValueError` with the offending field. The caller is responsible for catching; the engine does NOT silently drop malformed actions.

**Tests added:**
- `tests/backtesting/test_utils.py`:
  - `snap_to_lot` for the three lot regimes (0, 1, N) with positive + negative + fractional inputs.
  - `parse_action` with `None` → empty list.
  - `parse_action` with order-type action → orders pass through, qty snapped.
  - `parse_action` with weight-type action against known portfolio + prices → emitted orders math out by hand.
  - Weight-type action drops asset from previous positions → sell-to-zero emitted.
  - Weight-type action with target sum > 1.0 → either validation raises OR the engine fills what fits (decide per OQ-1 below).
  - Order with snapped qty == 0 → audit record in returned info, order not emitted.

**Diagnostic infrastructure:** the lot-size audit (`info["lot_size_adjustments"]`) — every difference between requested and actual qty is logged. This is what makes B6 visible to the caller, not just enforced internally.

**Done when:** `parse_action` covers all three action variants (None, order, weight); lot snapping covers all three lot regimes; audit records exist for every quantity adjustment; weight-action sells-before-buys preserved.

### Stage 5 — `Backtest` orchestrator (L)

**Goal:** the 6-step `step()` lifecycle that wires Stages 1–4 together. This is the public API.

**Files added:**
- `src/backtesting/backtest.py` — `Backtest` per `spec.md` § 4.4.

**Tasks:**
- Implement `__init__(data_feeds, initial_cash, lookback, lot_sizes, default_lot_size, commission_fn)`. Construct internal `DataHandler`, `Portfolio`, `ExecutionBroker`.
- Implement `reset() → (state, done, info)`. Reset all three components; return state at the first valid step; `done=False`, `info={}`.
- Implement `step(action) → (state, done, info)`:
  1. **PARSE** — `orders = parse_action(action, self.portfolio, self.data_handler, ...)`. Submit to broker.
  2. **PHASE 1** — `phase1_log = broker.process_queue(data_handler.get_current_bar(), portfolio, phase=1)`.
  3. **ADVANCE** — `done = data_handler.advance_time()`. If `done`, take the terminal path (below).
  4. **PHASE 2** — `phase2_log = broker.process_queue(data_handler.get_current_bar(), portfolio, phase=2)`.
  5. **MARK** — `portfolio.update_valuations(current_close_prices_at_T+1)`.
  6. **RETURN** — assemble state from `data_handler.get_window()` + `portfolio.get_state()` + step index + timestamp; pack `info` with phase1/phase2 logs + parse-time adjustments; return `(state, done=False, info)`.
- **Terminal path** (done=True at step 3):
  - Skip Phases 2 and 5 (no T+1 bar to fill against).
  - `portfolio.update_valuations(close_prices_at_T)` (mark to T's close; T+1 doesn't exist).
  - Cancel all pending broker orders; append to `info["cancelled_at_terminal"]`.
  - Build state from `data_handler.get_window()` (last full window at T).
  - Return `(state, done=True, info)`.
- State assembly: build the dict shape from `spec.md` § 3.2 exactly. Include `step`, `timestamp` (ISO string from data_handler's current date), `market_data`, `portfolio` (cash, equity, positions, pending_orders).

**Tests added:**
- `tests/backtesting/test_backtest.py`:
  - `reset()` produces valid state at first step.
  - `step(None)` advances time, no orders placed.
  - `step({"type": "order", ...})` with `current_close` fills in Phase 1.
  - `step({"type": "order", ...})` with `next_open` fills in Phase 2.
  - `step({"type": "weight", ...})` rebalances correctly given known prices.
  - Terminal step: pending order is cancelled, `done=True`, last state returned.
  - State schema matches spec (keys present, types correct).
  - Multi-step scenario (e.g., 50 steps with mixed orders) executes without exception.

**Diagnostic infrastructure:** the `info` payload is the diagnostic — every observable side-effect of the step lands in `info`. Add `tests/backtesting/test_backtest.py::test_info_completeness` — run a scenario with at least one of every side-effect (fill, rejection, lot-zero, terminal cancel) and assert each appears in the corresponding `info` key.

**Done when:** all 6 lifecycle steps execute in order; terminal path tested; state schema matches spec; multi-step scenarios run clean.

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
- Implement `leakage_harness.row_index_sentinel_feed(n_dates, n_assets, n_columns)` — produces a multi-feed dataset where every value at row `t` equals `t` (the canonical B1 test substrate).
- Implement `leakage_harness.step_function_price_feed(...)` — produces a feed where bar T's open/close/high/low are known deterministic functions of T (the B2 test substrate).
- Implement `leakage_harness.macro_release_feed(release_dates, release_values, master_dates)` — produces a slow-frequency feed aligned to known release dates, for the B3 forward-fill test.
- Implement `leakage_harness.random_action_stream(n_steps, n_assets, seed)` — produces a reproducible random action sequence for the B4 / B5 stress tests.
- Each of B1–B7 has at least one dedicated test (as enumerated in § "Critical correctness constraints" above) plus one randomized stress test driven by the harness.
- The B5 deterministic-replay test runs the same scripted scenario **twice** and compares every step's output element-wise.

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

The three locked-in decisions live in `spec.md` § 2 (D1 step-loop, D2 two-phase lifecycle, D3 master timeline). The decisions below are **implementation-side** choices that follow from those three but warrant explicit defense.

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

The ordering happens twice: once when weight-actions are converted into orders, once when the broker processes each phase's queue.

**Why not the alternative:** A single sort point (broker-only) would be simpler. But weight-action translation needs the ordering to compute correct deltas (so the sell of asset A frees cash to compute the buy of asset B against post-sell equity). And the broker needs the ordering to actually fund the buys. Both points are real; consolidating them is a v1.1 refactor candidate once the patterns stabilize.

### IA6. `Backtest.from_identifiers` lives in `backtest.py`, not a separate `loaders.py`

The convenience constructor that depends on `data_pipelines` sits next to the core constructor it wraps.

**Why not the alternative:** A separate `loaders.py` would isolate the optional dependency. But it would scatter the public surface across two files. Keeping it in `backtest.py` and lazy-importing `data_pipelines` inside the method body achieves the same isolation without splitting the API.

---

## Open questions

These are points where `spec.md` is genuinely ambiguous or silent and where implementation requires a user-level decision. They are NOT items the implementer should silently resolve.

### OQ-1. Weight-action over-allocation handling

`spec.md` § 3.1 says target_weights sum to "at most 1.0; the remainder stays in cash." But it does not specify what happens when target_weights sum to **more than 1.0** (e.g., a leveraged target). Two options:

- **(a)** Validate and raise — the caller must normalize before submitting.
- **(b)** Accept the action and let the per-asset overdraw guards kick in (some buys will be skipped; the realized weights will diverge from target).

Recommendation: **(a)** — fail loudly. Silent over-allocation followed by per-asset skipping is a debugging nightmare. But this needs user confirmation since (b) is closer to the spec's "engine fills what it can and reports the shortfall" pattern for the legal-but-cash-short case.

### OQ-2. Multi-asset state when assets have different lookback availabilities

If asset A has 1,000 days of history and asset B has 500 days, and `lookback=200`, then steps 200–699 have B's full window but A's; steps 700+ have both. `DataHandler.current_step` starts at `lookback` — does that mean step 200 (when A is the only tradeable asset), or does it mean "first step where every asset has lookback bars"?

The spec implies the former (B is "marked untradeable" before its first observation, but the engine still steps). But the consequence is that the state at step 200 has NaN columns for B in its market_data window — and the caller must handle NaN. Confirm this is the intended behavior.

### OQ-3. Macro feed support in v1

`spec.md` § 3.2's example state shape includes a `macro` feed with `GDP`. D3 (master timeline + forward-fill) is designed for the multi-frequency case. But `data_pipelines` v1 ships `us_equities` + `nse_equities` domains only — no macro domain.

Options:
- **(a)** v1 demonstrates the multi-frequency case using synthetic macro data only (the leakage harness's `macro_release_feed`); real macro support is gated on `data_pipelines` adding a FRED-like domain.
- **(b)** Ship a minimal `macro` domain in `data_pipelines` as part of this PR (scope creep).
- **(c)** Defer multi-frequency real-data support entirely to v1.1; v1 ships single-frequency equities only.

Recommendation: **(a)** — the engine code handles multi-frequency, but the v1 demonstration uses synthetic macro. Real macro is unblocked the moment `data_pipelines` ships the domain.

### OQ-4. Adjusted-close vs raw-close as the trading price

`data_pipelines` schema includes both `close` and `adj_close`. Splits and dividends are baked into `adj_close`. For backtests:

- Using `close` produces fills at the "real" historical prices but introduces gap-down artifacts on ex-dividend days that the strategy didn't actually pay.
- Using `adj_close` produces dividend-and-split-clean fills but inflates historical prices to today's basis (every fill in 2010 is at a 2026-adjusted price).

Spec says nothing on this. Strategy-by-strategy choice? Engine-level config? Recommendation: per-feed config knob (default `adj_close` for equities) with a one-time documented warning when switching.

### OQ-5. Commission function signature

`spec.md` § 6 specifies `commission_fn: (asset, qty, price) → float`. But many commission models depend on **trade direction** (e.g., short-sale-only fees) or **trade time-of-day** (intraday commissions vs MOC). The current signature can't express these.

Options:
- **(a)** Ship the spec'd 3-arg signature; flexible commission models are v1.1.
- **(b)** Expand to `(order_dict, fill_price) → float` — the entire order dict + actual fill price are passed, giving the commission function access to everything.

Recommendation: **(b)** — costs almost nothing and unblocks a broader class of commission models without an API break later.

### OQ-6. Should `step()` return the state from before or after `MARK`?

Step 5 (MARK) updates `portfolio.equity` using T+1's close. Step 6 (RETURN) builds the state. So the returned state's `portfolio.equity` reflects T+1's close. But the returned state's `market_data` window also includes T+1's close as the most-recent value. This is **consistent** (state at step T+1 reflects bar T+1) but means the caller's next action is decided against T+1's close already visible in the state — and if the next action's `execution == "current_close"`, it fills at T+1's close (the close the caller saw in this state).

This is what `current_close` is **designed** to do (MOC: see close, trade at close). But it's worth a confirmatory check that this is intended and not a subtle off-by-one. Recommendation: keep as designed; document the MOC semantics explicitly in the example notebook (Stage 8) so callers internalize it.

### OQ-7. Lot-size lookup for assets not in `lot_sizes` but with non-default needs

`default_lot_size=1` (whole shares) is the default. But a portfolio mixing crypto (`lot_size=0`) and equities (`lot_size=1`) would need every crypto asset enumerated in `lot_sizes` even if there are dozens. Is per-feed default (`feed_default_lot_sizes={"crypto": 0, "equities": 1}`) worth adding to v1, or is the per-asset dict sufficient?

Recommendation: per-asset dict in v1; per-feed default is a v1.1 ergonomics improvement.

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
- The two-phase execution lifecycle per `spec.md` § 5.
- Master timeline + forward-fill per D3.
- Order, weight, and None action types per `spec.md` § 3.1.
- `current_close`, `next_open`, `limit` execution modes per `spec.md` § 3.1.
- Per-instrument lot sizes (`0`, `1`, `N`) per `spec.md` § 4.5.
- Commission function hook per `spec.md` § 6.
- Untradeable-asset handling (mid-series start, delisted, etc.) per D3.
- B1–B7 correctness test suite + leakage harness.
- Convenience constructor wrapping `data_pipelines.fetch()`.
- One example script demonstrating end-to-end usage.

**Deferred to v1.1+ (in `V1.1_TBD.md`):**
- Margin requirements, borrow costs, locate requirements (short selling stays unconstrained in v1 per `spec.md` § 4.2 design note).
- Partial fills.
- Limit-order GTC semantics (v1 is good-for-day only).
- Slippage model.
- YAML-driven config layer (v1 is constructor-args-only per `spec.md` § 6).
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
