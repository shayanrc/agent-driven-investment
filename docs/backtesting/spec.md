# Backtesting Engine — System Specification

## 1. Purpose

`backtesting` is a configurable, multi-asset backtesting environment that exposes a step-based execution loop for programmatic trading strategies.

**Core capabilities:**

- Consume one or more single-frequency time series feeds (e.g., OHLCV equities) on a shared trading calendar. Multi-frequency alignment (forward-fill of slower feeds onto a master timeline) is deferred to v1.1.
- Operate on an iterative `while not done: bt.step(action)` loop, producing a `(state, done, info)` tuple at each step.
- Output structured state vectors containing both historical data windows (configurable lookback) and current portfolio status.
- Accept diverse action formats: absolute order quantities and target portfolio weightages.
- Eliminate look-ahead bias structurally via an engine-level `fill_mode` (Market-on-Close or Market-on-Open) that pins all orders submitted in a given engine to the same execution phase.
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

### D2: Decoupled Decision and Execution (engine-level `fill_mode`)

User actions are routed into an `ExecutionBroker` order queue rather than executed immediately against the `Portfolio`. The engine is constructed with a `fill_mode` (`"current_close"` for MOC or `"next_open"` for MOO, default `"next_open"`) that pins **every** order submitted in that engine to the same execution phase.

**Rationale:** Instant execution can only simulate Market-on-Close (MOC) orders — the caller sees price at time T and trades at price T, which is look-ahead bias when the decision is made intraday. By decoupling, a caller issues an order at time T, and the broker holds it until time T+1 to simulate Market-on-Open (MOO). The engine-level `fill_mode` makes the choice once at construction time rather than per-order: it eliminates a class of caller-side bugs (mixed-mode strategies that accidentally observe a price before they trade at it) and keeps the lifecycle a single linear path rather than a switch-on-each-order. Limit orders and per-order `execution` overrides are deferred to v1.1.

### D3: Single-Frequency Timeline with Per-Asset Activity Masks

The `DataHandler` establishes a single integer-indexed timeline from the union of all dates across all assets in the (single, daily-frequency) feed. Assets that start or end mid-series are tracked via per-asset activity masks rather than being dropped from the panel.

**Rationale:** At every step, the caller receives a state matrix whose row count equals the lookback. The timeline is the single source of truth for "what step are we on?" — all components reference it rather than maintaining independent clocks. v1 is single-frequency-per-engine: every feed shares the trading calendar of the equity feed. Multi-frequency support (master-timeline-driven forward-fill of slower feeds like quarterly macro) is deferred to v1.1 because the only ready data domain in `data_pipelines` v1 is equities (`us_equities` + `nse_equities`); a real macro feed needs a new `data_pipelines` domain first.

**Timeline construction:** the timeline is the **union** of all dates across all assets in the feed. Assets that start mid-series (e.g., a newly-IPO'd stock) have NaN values before their first observation; these are **not** forward-filled (there is no prior value to fill from). Instead, the asset is marked as untradeable on those dates — orders for it are rejected by the broker, and its columns in the state window are NaN. The caller is responsible for handling NaN in the state (e.g., by not trading assets that haven't started yet). Assets that end mid-series (e.g., delisted) are forward-filled with their last known values and marked as untradeable after their final date. The broker rejects **buy** orders for untradeable assets but permits **sell** orders (filled at the last known price) so that existing positions can be closed. Without this, capital would be permanently locked in phantom positions.

**Internal-gap policy:** when an asset has a date gap *within* its active range (e.g., a missing trading day between two valid observations), the engine's `gap_policy` constructor arg controls behavior. Default `"ffill_zero_volume"` forward-fills price columns from the previous bar and sets volume to zero on the gap day (treating it as "no trading happened"). Setting `"raise"` opts into strict mode where any internal gap is a configuration error and construction fails. Per-feed `gap_policy` overrides are a v1.1 ergonomics improvement.

---

## 3. Data Schemas

### 3.1 Action Schema (input to `step()`)

The `step()` function accepts an action dictionary or `None`. Two action types are supported: `"order"` (absolute quantities) and `"weight"` (target portfolio percentages). The engine translates weight-based actions into order-based actions internally before submitting to the broker.

All submitted orders fill at the engine's configured `fill_mode` — there is no per-order execution override in v1. Limit orders, time-in-force, and per-order `execution` fields are deferred to v1.1.

**No-op (hold):** passing `action=None` advances time without placing any orders. The engine skips PARSE and proceeds directly to the execution phase (which has nothing to process). This is the canonical way to express "do nothing this bar."

**Order-based action:**

```json
{
  "type": "order",
  "orders": [
    {"asset": "AAPL", "qty": 100},
    {"asset": "MSFT", "qty": -50}
  ]
}
```

Each order is a dict with exactly two required fields: `asset: str` and `qty: float` (positive = buy, negative = sell). No `execution`, no `limit_price`, no `time_in_force`. Extra fields are rejected by the validator.

**Weight-based action:**

```json
{
  "type": "weight",
  "target_weights": {
    "AAPL": 0.40,
    "MSFT": 0.35,
    "TSLA": 0.25
  }
}
```

Weight-based actions specify target portfolio allocation percentages. Sum must be `<= 1.0` — the engine **raises** on over-allocation (e.g., target_weights summing to 1.05) rather than silently truncating; the caller must normalize. The remainder up to 1.0 stays in cash. The engine calculates the required trades based on **pre-fill** portfolio equity and positions (the snapshot before any orders execute this step), snaps quantities to each instrument's lot size, then submits the resulting orders to the broker. Realized allocations may drift from targets because of lot-size rounding and per-asset overdraw skips; drift is accepted and reported in `info["weight_drift"]`. Sells are sequenced before buys to free up cash; if sell proceeds are insufficient to fund all buys, the engine fills what it can and reports the per-asset shortfall in `info["rebalance_shortfall"]`.

### 3.2 State Schema (output from `step()`)

`step()` returns a 3-tuple: `(state, done, info)`.

**`state`** — the environment observation at the current step:

```json
{
  "market_data": {
    "equities": {
      "AAPL": "<array of shape (lookback, num_columns)>",
      "MSFT": "<array of shape (lookback, num_columns)>"
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

- `market_data`: the lookback window (length = `lookback` config parameter) ending at the current step, keyed by feed name and asset. Each asset's data is the full column set from its source feed (e.g., OHLCV for equities). v1 is single-frequency-per-engine — multi-frequency feeds (e.g., a `macro` block forward-filled from quarterly releases) are deferred to v1.1.
- `portfolio`: current portfolio snapshot — cash, total equity (cash + mark-to-market positions), position quantities, and count of pending orders in the broker queue.
- `step`: integer index on the timeline.
- `timestamp`: the calendar date/datetime corresponding to the current step.

**`done`** — boolean. `True` when the data handler has exhausted all bars.

**`info`** — dictionary of auxiliary diagnostics for the current step. The following keys are locked for v1; the engine emits each key only when its payload is non-empty (a key whose contents would be all-zero / all-equal is omitted):

- `fills: list[dict]` — `{asset, qty, fill_price, commission}` per executed fill.
- `rejected_overdraw: list[dict]` — orders skipped because the direction-aware cash guard failed.
- `rejected_untradeable: list[dict]` — orders skipped because the asset was inactive (pre-IPO or post-delisting) on the fill bar.
- `rejected_invalid: list[dict]` — orders dropped by the parse-time validator (malformed schema, qty == 0, etc.).
- `weight_drift: dict[asset, float]` — realized minus target weight for assets where the realized allocation differs from the target after fills. Asset omitted if the drift is zero.
- `rebalance_shortfall: dict[asset, float]` — per-asset shortfall (target qty minus filled qty) when insufficient cash forced a partial-rebalance. Asset omitted if the shortfall is zero.
- `lot_size_audit: dict[asset, {"requested_qty": float, "filled_qty": int}]` — single unified key covering both lot-size rounding and snap-to-zero (an order whose snapped qty is zero appears with `filled_qty: 0`). Asset omitted if requested == filled.

---

## 4. Core Components

### 4.1 DataHandler

Responsible for timekeeping, per-asset activity-mask construction, and window slicing. The single source of truth for "what time is it?" v1 is single-frequency — every asset in every feed shares the same trading calendar. Multi-frequency master-timeline construction with cross-frequency forward-fill is deferred to v1.1.

```python
class DataHandler:
    def __init__(
        self,
        data_feeds: dict[str, dict[str, pd.DataFrame]],
        lookback: int,
        gap_policy: Literal["raise", "ffill_zero_volume"] = "ffill_zero_volume",
    ):
        """
        Args:
            data_feeds: nested dict of {feed_name: {asset_name: DataFrame}}.
                        Each DataFrame is indexed by date with columns per the
                        feed's schema (e.g., OHLCV for equities). All feeds must
                        share the same trading calendar.
            lookback:   number of historical bars to include in the state window.
            gap_policy: how to handle missing dates within an asset's active
                        range. "raise" fails construction on any internal gap;
                        "ffill_zero_volume" (default) forward-fills price columns
                        from the previous bar and zeroes the volume column.
        """
        self.data = self._align(data_feeds, gap_policy)
        self.lookback = lookback
        self.current_step = lookback  # first valid step after full window available
        self.max_steps = ...          # length of timeline

    def _align(self, data_feeds: dict, gap_policy: str) -> dict:
        """Build the timeline as the union of dates across all assets in the
        feed. Reindex each asset's DataFrame to the timeline. Assets starting
        mid-series have NaN before their first observation (not forward-filled)
        and are marked untradeable on those dates. Assets ending mid-series are
        forward-filled with last known values and marked untradeable after their
        final date. Internal gaps within an asset's active range are handled per
        gap_policy."""

    def advance_time(self) -> bool:
        """Move to the next bar. Semantics:
        - If current_step + 1 >= max_steps (the next step would exceed the
          timeline), return True (done) WITHOUT mutating current_step. The
          handler stays parked at the last valid bar (max_steps - 1).
        - Otherwise, increment current_step by 1 and return False.
        Invariant: current_step ∈ [lookback, max_steps - 1] for the entire
        engine lifetime."""

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

- The timeline is the union of all dates across all assets in the feed. No synthetic dates are inserted. Assets that start or end mid-series are handled per D3 (NaN before first observation, forward-fill after last).
- `current_step` starts at `lookback` (not 0) so that `get_window()` always returns a full-length array from the first call.
- `advance_time()` is no-mutate-when-done: at the terminal bar it reports done=True but leaves `current_step` unchanged, so a re-entrant call doesn't drift off the end. The terminal state observed by the caller is always the bar at index `max_steps - 1`.
- v1 scope: all feeds share the same trading calendar. Cross-calendar alignment (e.g., US equities vs EU macro with different holidays) is deferred to v1.1. So is multi-frequency forward-fill (slower feeds aligned onto a daily master timeline).

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

The order queue. Sits between the caller's desired actions and the portfolio's ledger. Responsible for fill logic. v1 has a single fill path — every order fills at the engine's configured `fill_mode` (set at `Backtest.__init__`). No per-order execution branching, no time-in-force, no limit-price logic.

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
        """Validate and enqueue orders. Validation: exactly the two required
        fields {asset: str, qty: float}, qty != 0, asset is a known instrument.
        Unknown fields (execution, limit_price, time_in_force) are rejected."""

    def process_queue(self, current_data: dict, portfolio: Portfolio):
        """Process all pending orders against current market data.

        Args:
            current_data: output of DataHandler.get_current_bar() at the bar
                          chosen by the engine's fill_mode (T for current_close,
                          T+1 for next_open). The broker is fill_mode-agnostic
                          — it fills against whatever bar is handed in.
            portfolio:    the Portfolio to route fills into.

        Sells are processed before buys to free up cash.
        Fills are routed to portfolio.execute_trade().
        All processed orders are removed from the queue.
        Orders that would overdraw cash are skipped and reported in the
        returned fill log."""

    def get_pending_count(self) -> int:
        """Number of orders still in the queue."""

    def reset(self):
        """Clear all pending orders."""
```

**Design notes:**

- `process_queue` has no `phase` argument and no internal branching on execution mode. The orchestrator picks which bar to hand in based on `fill_mode`; the broker just fills.
- Partial fills are not supported in v1. An order either fills completely or is skipped (overdraw / untradeable).
- Sells are processed before buys. This ensures cash freed by liquidations is available for new purchases without ever allowing negative cash.
- Limit orders, time-in-force semantics (GFD / GTC / IOC / FOK), and per-order `execution` overrides are deferred to v1.1.

### 4.4 Backtest

The orchestrator. Drives the single-phase execution lifecycle and routes data between components. This is the class the caller interacts with.

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
        fill_mode: Literal["current_close", "next_open"] = "next_open",
        gap_policy: Literal["raise", "ffill_zero_volume"] = "ffill_zero_volume",
    ):
        self.data_handler = DataHandler(data_feeds, lookback, gap_policy=gap_policy)
        self.portfolio = Portfolio(initial_cash)
        self.broker = ExecutionBroker(commission_fn)
        self.lot_sizes = lot_sizes or {}
        self.default_lot_size = default_lot_size
        self.fill_mode = fill_mode

    def step(self, action: dict | None) -> tuple[dict, bool, dict]:
        """The core progression lifecycle.

        Single-phase lifecycle, chosen by self.fill_mode:

        fill_mode == "current_close":
          1. PARSE     — Convert action to order list, submit to broker.
          2. FILL      — broker.process_queue(bar=T). Fills at T's close price.
          3. ADVANCE   — data_handler.advance_time(). T → T+1 (or done).
          4. MARK      — portfolio.update_valuations() using T+1's close
                         (or T's close if done).
          5. RETURN    — Assemble and return (state, done, info).

        fill_mode == "next_open":
          1. PARSE     — Convert action to order list, submit to broker.
          2. ADVANCE   — data_handler.advance_time(). T → T+1 (or done).
                         If done, skip step 3 (no T+1 bar to fill against);
                         pending orders remain in the queue and are reported
                         in info["rejected_untradeable"].
          3. FILL      — broker.process_queue(bar=T+1). Fills at T+1's open.
          4. MARK      — portfolio.update_valuations() using T+1's close
                         (or T's close if done).
          5. RETURN    — Assemble and return (state, done, info).

        The two modes share PARSE / ADVANCE / MARK / RETURN; only the FILL bar
        differs. Look-ahead-bias elimination is structural: the engine never
        fills against a bar the caller hasn't yet observed, regardless of mode.
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
        Sells are placed before buys in the returned list. Raises ValueError
        if target_weights.values() sum to more than 1.0."""

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

This is the most important section of the spec. The lifecycle in `Backtest.step()` is the mechanism that makes look-ahead-bias elimination structural rather than conventional. The engine's `fill_mode` selects between the two single-phase variants below; all orders submitted in a given engine use the same variant.

### `fill_mode="current_close"` (MOC)

```
Time:  ──────────── T ──────────────┃──────────── T+1 ────────────────
                                    ┃
Caller sees state at T              ┃
Caller calls step(action)           ┃
                                    ┃
  1. PARSE action → orders          ┃
     (snap to lot sizes,            ┃
      sells before buys)            ┃
  2. FILL at T's close price        ┃
                                    ┃
  3. ADVANCE time pointer ──────────╋───
     (no-mutate when done)          ┃
  4. MARK portfolio to T+1 close    ┃
     (or T's close if done)         ┃
  5. Return (state, done, info)     ┃
```

### `fill_mode="next_open"` (MOO, default)

```
Time:  ──────────── T ──────────────┃──────────── T+1 ────────────────
                                    ┃
Caller sees state at T              ┃
Caller calls step(action)           ┃
                                    ┃
  1. PARSE action → orders          ┃
     (snap to lot sizes,            ┃
      sells before buys)            ┃
                                    ┃
  2. ADVANCE time pointer ──────────╋───
     (no-mutate when done)          ┃
                                    ┃
  3. FILL at T+1's open price       ┃
     (skipped if done — orders      ┃
      reported as untradeable)      ┃
  4. MARK portfolio to T+1 close    ┃
     (or T's close if done)         ┃
  5. Return (state, done, info)     ┃
```

**Why this order matters:**

- In `current_close` mode: orders fill at T's close *before* time advances. The caller has already observed T's close price in the state, so this is not look-ahead — it is "I see the close, I trade at the close" (Market-on-Close).
- In `next_open` mode: orders fill at T+1's open *after* time advances. The caller has not seen T+1's prices when it issues the order — the broker holds the queue across the time pointer and fills at the next available open. This is the standard "decide today, execute tomorrow" pattern.
- The choice is engine-wide and locked at construction time. A caller cannot mix MOC and MOO in the same engine; the entire backtest commits to one phase, eliminating an entire class of caller-side timing bugs.
- Portfolio is marked to market at T+1's close (the price the caller will see in the next state), or at T's close on the terminal step where T+1 doesn't exist.

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
| `fill_mode` | `Literal["current_close", "next_open"]` | `"next_open"` | Engine-level execution phase. Locked at construction time and applies to every order. |
| `gap_policy` | `Literal["raise", "ffill_zero_volume"]` | `"ffill_zero_volume"` | How to handle internal date gaps within an asset's active range. |

**Price column:** v1 hard-codes `close` (raw, not split/dividend-adjusted) as the fill-price column. A `price_column` knob exposing `close` / `adj_close` is deferred to v1.1.

> **This spec was amended on 2026-05-26 based on V1_PLAN review.** Q3 deferred multi-frequency support (master timeline, slower-feed forward-fill, macro domain). Q6 removed per-order execution configuration and limit orders (`fill_mode` is now engine-level). Q8 redefined `advance_time()` to no-mutate-when-done. Q9 added the configurable `gap_policy`. See `V1.1_TBD.md` for what was moved out.

---

## 7. Correctness Constraints

These are non-negotiable invariants, analogous to analog_mc's C1–C6.

**B1: No look-ahead in state.** `get_window()` returns data up to and including `current_step`. No future data is ever included in the state. Enforced by integer indexing on the timeline.

**B2: Phase-ordered execution.** All orders submitted in a given engine fill at the bar selected by the engine's configured `fill_mode`: `"current_close"` ⇒ T's close (the bar the caller observed before calling `step()`); `"next_open"` ⇒ T+1's open (a bar the caller had not seen when it submitted the action). The state returned from `step()` reflects T+1 data (or T's data when `done=True` and the timeline cannot advance). **Testable assertion:** for every fill in `info["fills"]`, verify that the recorded `fill_price` equals the engine's configured `fill_mode` bar's price field (close for `current_close`, open for `next_open`).

**B3: Causal data access (forward-fill on internal gaps only).** When `gap_policy="ffill_zero_volume"`, the engine forward-fills only from past observations within an asset's active range — never from a future observation. No backward-fill, no interpolation, no centering. At step T, every cell in `state["market_data"]` reflects either a true observation at or before T or a forward-filled value carried over from such an observation. Mid-series-start assets carry NaN (no backfill) before their first observation.

**B4: Portfolio consistency.** After every `step()`, `portfolio.equity == portfolio.cash + sum(position_qty * mark_price)` for all positions. Cash is never negative. These are verified by `update_valuations()` and enforced by `execute_trade()`.

**B5: Deterministic replay.** Same `data_feeds` + same sequence of `action` dicts → identical sequence of `(state, done, info)` tuples. No internal randomness.

**B6: Lot-size integrity.** Every filled order quantity is a valid multiple of the instrument's lot size. The engine never fills a fractional quantity for a whole-share instrument, or a non-multiple for a round-lot instrument.

**B7: Terminal-step contract.** `DataHandler.advance_time()` is no-mutate-when-done — it returns `True` at the terminal bar without incrementing `current_step`. Therefore `current_step` is always in `[lookback, max_steps - 1]` for the entire engine lifetime; the post-done state's `timestamp` equals the last bar's date; the post-done `portfolio.equity` reflects the last bar's close-price mark. For `fill_mode="next_open"`, any orders still pending when `done=True` are reported under `info["rejected_untradeable"]` (the T+1 bar that would have filled them does not exist).

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
└── test_correctness.py  # B1–B7 constraint tests

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
