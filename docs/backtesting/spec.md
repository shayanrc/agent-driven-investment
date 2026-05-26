# Backtesting Engine — System Specification

## 1. Purpose

`backtesting` is a configurable, multi-asset backtesting environment that exposes a step-based execution loop for programmatic trading strategies.

**Core capabilities:**

- Consume and align multiple multidimensional time series (e.g., OHLCV equities, macroeconomic indicators) onto a single master timeline.
- Operate on an iterative `while not done: bt.step(action)` loop, producing a `(state, done, info)` tuple at each step.
- Output structured state vectors containing both historical data windows (configurable lookback) and current portfolio status.
- Accept diverse action formats: absolute order quantities and target portfolio weightages.
- Support configurable order execution timing (Market-on-Close, Market-on-Open, Limit) with structural look-ahead-bias elimination via a two-phase execution lifecycle.
- Enforce per-instrument lot sizing: whole shares (default), fractional, or round lots.

**What this module is *not*:**

- Not a forecasting engine. Forecasts are upstream (`analog_mc`, `gbdt`); this module consumes signals and simulates their execution.
- Not an alpha-research tool. It evaluates strategy performance; it does not discover strategies.
- Not a live-trading gateway. The `ExecutionBroker` simulates fills against historical data; it does not connect to any brokerage API.
- Not an RL environment. The step-loop design is naturally compatible with RL frameworks, but the engine does not subclass `gymnasium.Env`, compute rewards, or define action/observation spaces. A thin Gym adapter wrapper can be built on top if needed.

---

## 2. Architectural Decisions

Three structural choices shape the entire design. Each was chosen for a specific reason; do not change them without surfacing the deviation and asking first.

### D1: Step-Loop Execution (not event-driven callbacks)

The system uses a `while not done: bt.step(action)` loop rather than an event-driven callback architecture.

**Rationale:** The step loop makes state transitions explicit. The caller observes a state, makes a decision, and passes it back to the engine. The engine never calls the caller — the caller always calls the engine. This provides ultimate control over execution flow and allows the same environment to serve hand-coded strategies, optimization loops, and RL agents without adapter glue.

### D2: Decoupled Decision and Execution (two-phase order lifecycle)

User actions are routed into an `ExecutionBroker` order queue rather than executed immediately against the `Portfolio`.

**Rationale:** Instant execution can only simulate Market-on-Close (MOC) orders — the caller sees price at time T and trades at price T, which is look-ahead bias when the decision is made intraday. By decoupling, a caller issues an order at time T, and the broker holds it until time T+1 to simulate Market-on-Open (MOO) or evaluates it against T+1 prices for limit fills. The two-phase lifecycle (Phase 1: current-step fills; Phase 2: next-step fills) is the structural mechanism that eliminates look-ahead bias — it makes bias impossible rather than relying on user discipline.

### D3: Master Timeline with Forward-Fill Alignment

The `DataHandler` establishes a single integer-indexed master timeline driven by the highest-frequency data feed (e.g., daily equities). Lower-frequency data (e.g., quarterly GDP) is forward-filled to match.

**Rationale:** At every step, the caller must receive a valid, non-null state matrix. Forward-filling is the correct causal operation for slow-updating data: quarterly GDP is genuinely "still the last reported value" until the next release. The master timeline is the single source of truth for "what step are we on?" — all components reference it rather than maintaining independent clocks.

**Master timeline construction:** The highest-frequency feed is determined by the feed with the greatest number of unique dates. The master timeline is the **union** of all dates across all assets within that feed. Assets that start mid-series (e.g., a newly-IPO'd stock) have NaN values before their first observation; these are **not** forward-filled (there is no prior value to fill from). Instead, the asset is marked as untradeable on those dates — orders for it are rejected by the broker, and its columns in the state window are NaN. The caller is responsible for handling NaN in the state (e.g., by not trading assets that haven't started yet). Assets that end mid-series (e.g., delisted) are forward-filled with their last known values and marked as untradeable after their final date. The broker rejects **buy** orders for untradeable assets but permits **sell** orders (filled at the last known price) so that existing positions can be closed. Without this, capital would be permanently locked in phantom positions.

---

## 3. Data Schemas

### 3.1 Action Schema (input to `step()`)

The `step()` function accepts an action dictionary or `None`. Two action types are supported: `"order"` (absolute quantities) and `"weight"` (target portfolio percentages). The engine translates weight-based actions into order-based actions internally before submitting to the broker.

**No-op (hold):** passing `action=None` advances time without placing any orders. The engine skips PARSE and proceeds directly to the execution phases (which have nothing to process). This is the canonical way to express "do nothing this bar."

**Order-based action:**

```json
{
  "type": "order",
  "orders": [
    {"asset": "AAPL", "qty": 100, "execution": "next_open"},
    {"asset": "MSFT", "qty": -50, "execution": "current_close"},
    {"asset": "TSLA", "qty": 10, "execution": "limit", "limit_price": 200.0}
  ]
}
```

**Weight-based action:**

```json
{
  "type": "weight",
  "target_weights": {
    "AAPL": 0.40,
    "MSFT": 0.35,
    "TSLA": 0.25
  },
  "execution": "next_open"
}
```

Weight-based actions specify target portfolio allocation percentages (summing to at most 1.0; the remainder stays in cash). The engine calculates the required trades based on **pre-fill** portfolio equity and positions (the snapshot before any orders execute this step), snaps quantities to each instrument's lot size, then submits the resulting orders to the broker. Because Phase 1 fills may change equity before Phase 2 buys execute, the realized allocation may deviate slightly from the target; this drift is accepted and reported in `info`. Sells are sequenced before buys to free up cash; if sell proceeds are insufficient to fund all buys, the engine fills what it can and reports the shortfall in `info`. All orders generated from a weight-based action share the same `execution` timing.

**Execution modes:**

| Mode | Fill price | When processed |
|------|-----------|----------------|
| `current_close` | Close price at time T (the step when the order is submitted). **Assumption:** the state represents end-of-bar data, so the caller has already observed this price. Valid for daily bars where the state is the completed bar. If the engine is later extended to intrabar resolution, this mode must be re-evaluated. | Phase 1 (before time advances) |
| `next_open` | Open price at time T+1 | Phase 2 (after time advances) |
| `limit` | The `limit_price` specified on the order | Phase 2; filled only if T+1 bar's low ≤ limit_price (buy) or high ≥ limit_price (sell). Unfilled limits expire after one bar (good-for-day semantics). |

### 3.2 State Schema (output from `step()`)

`step()` returns a 3-tuple: `(state, done, info)`.

**`state`** — the environment observation at the current step:

```json
{
  "market_data": {
    "equities": {
      "AAPL": "<array of shape (lookback, num_columns)>",
      "MSFT": "<array of shape (lookback, num_columns)>"
    },
    "macro": {
      "GDP": "<array of shape (lookback, 1)>"
    }
  },
  "portfolio": {
    "cash": 50000.0,
    "equity": 105000.0,
    "positions": {"AAPL": 100, "MSFT": -50},
    "pending_orders": 1
  },
  "step": 42,
  "timestamp": "2024-03-15"
}
```

- `market_data`: the lookback window (length = `lookback` config parameter) ending at the current step, keyed by feed name and asset. Each asset's data is the full column set from its source feed (e.g., OHLCV for equities).
- `portfolio`: current portfolio snapshot — cash, total equity (cash + mark-to-market positions), position quantities, and count of pending orders in the broker queue.
- `step`: integer index on the master timeline.
- `timestamp`: the calendar date/datetime corresponding to the current step.

**`done`** — boolean. `True` when the data handler has exhausted all bars.

**`info`** — dictionary of auxiliary diagnostics for the current step (fills executed this step, slippage, rejected orders, lot-size rounding adjustments, rebalancing shortfalls, etc.). Used for logging and debugging.

---

## 4. Core Components

### 4.1 DataHandler

Responsible for timekeeping, multi-feed alignment, and window slicing. The single source of truth for "what time is it?"

```python
class DataHandler:
    def __init__(self, data_feeds: dict[str, dict[str, pd.DataFrame]], lookback: int):
        """
        Args:
            data_feeds: nested dict of {feed_name: {asset_name: DataFrame}}.
                        Each DataFrame is indexed by date with columns per the
                        feed's schema (e.g., OHLCV for equities).
            lookback:   number of historical bars to include in the state window.
        """
        self.data = self._align_and_fill(data_feeds)
        self.lookback = lookback
        self.current_step = lookback  # first valid step after full window available
        self.max_steps = ...          # length of master timeline

    def _align_and_fill(self, data_feeds: dict) -> dict:
        """Build the master timeline from the highest-frequency feed (the feed
        with the most unique dates). The timeline is the union of dates across
        all assets in that feed. Reindex all feeds to the master timeline.
        Forward-fill lower-frequency feeds. Assets starting mid-series have
        NaN before their first observation (not forward-filled). Assets ending
        mid-series are forward-filled with last known values. Raise on NaNs
        that indicate a data gap within an asset's active range."""

    def advance_time(self) -> bool:
        """Increment current_step by 1. Return True (done) if current_step >= max_steps
        after the increment, meaning T+1 has no data. The last bar in the dataset
        (index max_steps - 1) is the final bar that appears in a state; no step
        can advance past it."""

    def get_current_bar(self) -> dict:
        """Return data for exactly the current step. Used by the broker for
        fill-price lookup."""

    def get_window(self) -> dict:
        """Return the data slice [current_step - lookback + 1, current_step] inclusive.
        This is exactly `lookback` bars ending at the current step.
        Equivalently: data[current_step - lookback + 1 : current_step + 1] in
        Python half-open indexing. This is the market_data portion of the state."""

    def get_price(self, asset: str, field: str = "close") -> float:
        """Convenience: return a single price field for one asset at current_step."""

    def reset(self):
        """Reset current_step to lookback (the first valid step)."""
```

**Design notes:**

- The master timeline is derived from the union of all dates across all assets in the highest-frequency feed (determined by greatest count of unique dates). No synthetic dates are inserted. Assets that start or end mid-series are handled per D3 (NaN before first observation, forward-fill after last).
- `current_step` starts at `lookback` (not 0) so that `get_window()` always returns a full-length array from the first call.
- v1 scope: all feeds share the same trading calendar. Cross-calendar alignment (e.g., US equities vs EU macro with different holidays) is deferred.

### 4.2 Portfolio

The ledger. Tracks cash, positions, and valuations. Knows nothing about time, data, or execution logic.

```python
class Portfolio:
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}
        self.equity = initial_cash

    def execute_trade(self, asset: str, qty: float, price: float):
        """Update cash and position for a single fill.
        cash -= qty * price (positive qty = buy = cash outflow).
        Positions track net quantity (positive = long, negative = short).
        Raises if the trade would make cash negative."""

    def update_valuations(self, current_prices: dict[str, float]):
        """Mark all positions to market. Compute equity = cash + sum(pos * price).
        Called once per step after all fills are processed."""

    def get_state(self) -> dict:
        """Return snapshot: cash, equity, positions dict, unrealized PnL."""

    def reset(self):
        """Restore to initial_cash, clear all positions."""
```

**Design notes:**

- `qty` is float at the `Portfolio` level to keep the ledger simple. Lot-size enforcement happens upstream in `Backtest._parse_action()`, not inside the portfolio.
- No commission or slippage model inside `Portfolio`. Commissions are handled entirely by the broker. The overdraw check is direction-aware:
  - **Buys** (qty > 0): broker checks `qty * price + commission <= portfolio.cash` before filling. If it passes, calls `portfolio.execute_trade(asset, qty, price)` then `portfolio.cash -= commission`.
  - **Sells** (qty < 0): broker checks `commission <= portfolio.cash + abs(qty) * price` (i.e., post-fill cash can cover the commission). If it passes, calls `portfolio.execute_trade(asset, qty, price)` then `portfolio.cash -= commission`.
  - The portfolio itself only sees the market price; the commission is a post-fill cash adjustment. This keeps the ledger simple and testable.
- Short selling is supported: a negative position is a short. **v1 limitation:** shorts are unconstrained — no margin requirement, no borrow cost, no locate requirement. A short sale increases cash (`cash -= negative_qty * price` = cash inflow), so the cash-negative guard does not limit short exposure. Strategy authors should be aware that unlimited shorting produces unrealistic results. Margin/borrow mechanics are a v2 concern.
- Cash cannot go negative. `execute_trade` raises if `cash - qty * price < 0`. The caller (broker / action parser) is responsible for sequencing sells before buys and skipping orders that would overdraw.

### 4.3 ExecutionBroker

The order queue. Sits between the caller's desired actions and the portfolio's ledger. Responsible for fill logic and order lifecycle.

```python
class ExecutionBroker:
    def __init__(self, commission_fn: Callable | None = None):
        """
        Args:
            commission_fn: optional callable (asset, qty, price) -> commission_amount.
                           If provided, commission is deducted from cash on each fill.
        """
        self.pending_orders: list[dict] = []

    def submit_orders(self, orders: list[dict]):
        """Validate and enqueue orders. Validation: required fields present,
        qty != 0, limit orders have limit_price, execution mode is recognized."""

    def process_queue(self, current_data: dict, portfolio: Portfolio, phase: int):
        """Process pending orders against current market data.

        Args:
            current_data: output of DataHandler.get_current_bar().
            portfolio:    the Portfolio to route fills into.
            phase:        1 = pre-advance (current_close fills),
                          2 = post-advance (next_open and limit fills).

        Within each phase, sells are processed before buys to free up cash.
        Fills are routed to portfolio.execute_trade().
        Filled orders are removed from the queue.
        Unfilled limit orders are expired (good-for-day: if phase 2 doesn't
        fill them, they're dropped).
        Orders that would overdraw cash are skipped and reported in the
        returned fill log."""

    def get_pending_count(self) -> int:
        """Number of orders still in the queue."""

    def reset(self):
        """Clear all pending orders."""
```

**Design notes:**

- The `phase` parameter replaces the original `is_new_step` boolean for clarity. Phase 1 = before time advances (fills `current_close` orders). Phase 2 = after time advances (fills `next_open` and `limit` orders).
- Limit order fill logic: for a buy limit at price P, the order fills if the current bar's low ≤ P (fill at P). For a sell limit at price P, the order fills if the current bar's high ≥ P (fill at P). This is a simplification — real limit fills depend on intrabar sequencing, which daily bars cannot resolve.
- Limit orders use good-for-day semantics: if not filled in Phase 2 of the step after submission, they are expired and removed. Good-til-cancelled (GTC) is a v2 extension.
- Partial fills are not supported in v1. An order either fills completely or not at all.
- Within each phase, sells are processed before buys. This ensures cash freed by liquidations is available for new purchases without ever allowing negative cash.

### 4.4 Backtest

The orchestrator. Manages the two-phase execution lifecycle and routes data between components. This is the class the caller interacts with.

```python
class Backtest:
    def __init__(
        self,
        data_feeds: dict,
        initial_cash: float = 100_000.0,
        lookback: int = 20,
        lot_sizes: dict[str, int] | None = None,
        default_lot_size: int = 1,
        commission_fn: Callable | None = None,
    ):
        self.data_handler = DataHandler(data_feeds, lookback)
        self.portfolio = Portfolio(initial_cash)
        self.broker = ExecutionBroker(commission_fn)
        self.lot_sizes = lot_sizes or {}
        self.default_lot_size = default_lot_size

    def step(self, action: dict | None) -> tuple[dict, bool, dict]:
        """The core progression lifecycle.

        Execution order (6 steps, strictly sequential):

        1. PARSE     — Convert action to order list (or no-op if action is
                       None). Weight-based actions are translated to absolute
                       orders using current (pre-fill) portfolio equity and
                       positions, with quantities snapped to each instrument's
                       lot size. Sells are sequenced before buys. Validated
                       orders are submitted to the broker.

        2. PHASE 1   — broker.process_queue(phase=1). Executes current_close
                       orders against the current bar's close price.

        3. ADVANCE   — data_handler.advance_time(). The time pointer moves
                       from T to T+1. If done, go to terminal path (below).

        4. PHASE 2   — broker.process_queue(phase=2). Executes next_open
                       orders against T+1's open price. Evaluates and
                       fills/expires limit orders against T+1's bar.

        5. MARK      — portfolio.update_valuations() using T+1's close prices.

        6. RETURN    — Assemble and return (state, done, info).

        Terminal step (done=True at step 3):
            Skip steps 4–5. Mark portfolio to T's close prices (last
            available). Cancel all pending orders (next_open, limit) and
            report them in info. Return state at T with done=True.
        """

    def reset(self) -> tuple[dict, bool, dict]:
        """Reset all components. Return (state, done, info) at the first valid
        step (after the lookback window is available), with done=False and
        info={}. Same return shape as step() for a uniform caller loop."""

    def _parse_action(self, action: dict) -> list[dict]:
        """Translate action dict into a list of normalized order dicts.
        For weight-based actions: compute target position for each asset
        from (target_weight * current_equity / current_price), diff against
        current position, snap to lot size, and emit the delta as orders.
        Sells are placed before buys in the returned list."""

    def _snap_to_lot(self, asset: str, qty: float) -> float:
        """Truncate qty to the nearest valid lot toward zero.
        Lot size lookup: self.lot_sizes.get(asset, self.default_lot_size).
        - lot_size == 0: no rounding (fractional).
        - lot_size == 1: truncate toward zero — int(qty). (2.7 → 2, -2.7 → -2).
        - lot_size == N: truncate toward zero to nearest multiple of N.
            floor(abs(qty) / N) * N * sign(qty)."""
```

### 4.5 Lot Size Specification

Per-instrument lot sizes control the minimum tradeable quantity. Configured via the `lot_sizes` dict and `default_lot_size` on `Backtest.__init__()`.

| `lot_size` value | Meaning | Rounding behavior |
|------------------|---------|-------------------|
| `1` (default) | Whole shares | Truncate toward zero: `int(qty)`. `2.7 → 2`, `-2.7 → -2` |
| `0` | Fractional | No rounding; `qty` used as-is |
| `N` (any positive int) | Round lots of N | Truncate toward zero to nearest multiple of N: `floor(abs(qty)/N)*N*sign(qty)` |

Examples:
- `lot_sizes={"BTC-USD": 0, "RELIANCE.NS": 1}` — BTC trades fractionally, RELIANCE trades in whole shares.
- `default_lot_size=1` — any instrument not in `lot_sizes` trades in whole shares.
- `lot_sizes={"NIFTY_FUT": 25}` — NIFTY futures trade in lots of 25.

When a weight-based action produces an ideal fractional quantity, the engine snaps it to the instrument's lot size (rounding toward zero). The residual cash stays in cash. The rounding adjustment is reported in `info` so the caller can see the difference between requested and actual allocation.

---

## 5. Step Lifecycle — Detailed Walkthrough

This is the most important section of the spec. The lifecycle in `Backtest.step()` is the mechanism that makes look-ahead-bias elimination structural rather than conventional.

```
Time:  ──────────── T ──────────────┃──────────── T+1 ────────────────
                                    ┃
Caller sees state at T              ┃
Caller calls step(action)           ┃
                                    ┃
  1. PARSE action → orders          ┃
     (snap to lot sizes,            ┃
      sells before buys)            ┃
  2. PHASE 1: fill current_close    ┃
     orders at T's close price      ┃
                                    ┃
  3. ADVANCE time pointer ──────────╋───
     (if done → terminal path)      ┃
                                    ┃
  4. PHASE 2: fill next_open        ┃
     orders at T+1's open price;    ┃
     evaluate limit orders          ┃
     against T+1's bar              ┃
  5. MARK portfolio to T+1 close    ┃
  6. Return (state_T+1, done, info) ┃

Terminal path (done=True at step 3):
  Skip 4–5. Mark to T's close.
  Cancel pending orders → info.
  Return (state_T, done=True, info).
```

**Why this order matters:**

- `current_close` orders fill at T's close *before* time advances. The caller has already observed T's close price in the state, so this is not look-ahead — it is "I see the close, I trade at the close" (Market-on-Close).
- `next_open` orders fill at T+1's open *after* time advances. The caller has not seen T+1's prices when it issues the order — the broker holds it and fills at the next available open. This is the standard "decide today, execute tomorrow" pattern.
- Limit orders are evaluated against T+1's full bar (OHLC). This is a simplification of intrabar dynamics but is standard for daily-bar backtesting.
- Portfolio is marked to market at T+1's close, which is the price the caller will see in the next state.

---

## 6. Configuration

`Backtest` is configured at construction time via explicit arguments. A YAML-driven config layer may be added later but is not part of v1 — construction arguments are the contract.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_feeds` | `dict` | required | Nested dict of `{feed_name: {asset: DataFrame}}` |
| `initial_cash` | `float` | `100_000.0` | Starting cash balance |
| `lookback` | `int` | `20` | Number of historical bars in the state window |
| `lot_sizes` | `dict[str, int]` | `{}` | Per-instrument lot sizes. `0` = fractional, `1` = whole shares, `N` = round lots of N |
| `default_lot_size` | `int` | `1` | Lot size for instruments not in `lot_sizes` |
| `commission_fn` | `Callable` | `None` (no commission) | `(asset, qty, price) -> float` |

---

## 7. Correctness Constraints

These are non-negotiable invariants, analogous to analog_mc's C1–C6.

**B1: No look-ahead in state.** `get_window()` returns data up to and including `current_step`. No future data is ever included in the state. Enforced by integer indexing on the master timeline.

**B2: Phase-ordered execution.** `current_close` orders fill in Phase 1 (before `advance_time()`); `next_open` and `limit` orders fill in Phase 2 (after `advance_time()`). `current_close` fill prices come from bar T (the bar the caller observed before calling `step()`). `next_open` and `limit` fill prices come from bar T+1 (which the caller had not seen when it submitted the action). The state returned from `step()` reflects T+1 data. **Testable assertion:** for every fill in `info`, verify that `current_close` fills use T's close price and `next_open` fills use T+1's open price.

**B3: Causal forward-fill.** Lower-frequency data is forward-filled from past observations only. No backward-fill, no interpolation, no centering. At step T, macro data reflects the most recent observation at or before T.

**B4: Portfolio consistency.** After every `step()`, `portfolio.equity == portfolio.cash + sum(position_qty * mark_price)` for all positions. Cash is never negative. These are verified by `update_valuations()` and enforced by `execute_trade()`.

**B5: Deterministic replay.** Same `data_feeds` + same sequence of `action` dicts → identical sequence of `(state, done, info)` tuples. No internal randomness.

**B6: Lot-size integrity.** Every filled order quantity is a valid multiple of the instrument's lot size. The engine never fills a fractional quantity for a whole-share instrument, or a non-multiple for a round-lot instrument.

---

## 8. Module Layout

Following project conventions (`CLAUDE.md` § "Module namespacing"):

```
src/backtesting/
├── __init__.py
├── data_handler.py      # DataHandler
├── portfolio.py         # Portfolio
├── broker.py            # ExecutionBroker
├── backtest.py          # Backtest
└── utils.py             # shared helpers (action validation, lot-size snapping)

docs/backtesting/
├── spec.md              # this document
└── goal.md              # what the module optimizes for

tests/backtesting/
├── test_data_handler.py
├── test_portfolio.py
├── test_broker.py
├── test_backtest.py
└── test_correctness.py  # B1–B6 constraint tests

configs/backtesting/     # reserved for future YAML-driven configs
```

---

## 9. Resolved Design Decisions

These questions were resolved during the spec review. Recorded here for traceability.

1. **Data source integration.** `Backtest` accepts raw DataFrames at the core. A convenience constructor (`Backtest.from_identifiers(...)`) wrapping `data_pipelines.fetch()` can be added later without changing the core contract.

2. **Lot sizing.** Per-instrument via `lot_sizes` dict + `default_lot_size`. Values: `0` = fractional (no rounding), `1` = whole shares (default), `N` = round lots of N. Quantities are snapped toward zero. Rounding adjustments reported in `info`.

3. **Reward function.** Removed. The engine returns `(state, done, info)`. The caller computes whatever performance metrics it needs from the state trajectory. RL reward shaping is the caller's responsibility, not the engine's.

4. **Weight-based rebalancing.** The engine sequences sells before buys to free up cash. Cash is never allowed to go negative. If sell proceeds are insufficient to fund all buys, the engine fills what it can and reports the shortfall in `info`.

5. **Gym compatibility.** No `gymnasium.Env` subclass. The step-loop design is naturally Gym-compatible; a thin `BacktestGymWrapper` adapter can be built on top if needed, without adding `gymnasium` as a dependency.
