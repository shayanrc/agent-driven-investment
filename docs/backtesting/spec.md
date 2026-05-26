# Backtesting Engine — System Specification

## 1. Purpose

`backtesting` is a configurable, multi-asset backtesting environment that exposes a step-based execution loop compatible with Reinforcement Learning agents (OpenAI Gym interface) and programmatic trading strategies.

**Core capabilities:**

- Consume and align multiple multidimensional time series (e.g., OHLCV equities, macroeconomic indicators) onto a single master timeline.
- Operate on an iterative `while not done: env.step(action)` loop, producing a `(state, reward, done, info)` tuple at each step.
- Output structured state vectors containing both historical data windows (configurable lookback) and current portfolio status.
- Accept diverse action formats: absolute order quantities and target portfolio weightages.
- Support configurable order execution timing (Market-on-Close, Market-on-Open, Limit) with structural look-ahead-bias elimination via a two-phase execution lifecycle.

**What this module is *not*:**

- Not a forecasting engine. Forecasts are upstream (`analog_mc`, `gbdt`); this module consumes signals and simulates their execution.
- Not an alpha-research tool. It evaluates strategy performance; it does not discover strategies.
- Not a live-trading gateway. The `ExecutionBroker` simulates fills against historical data; it does not connect to any brokerage API.

---

## 2. Architectural Decisions

Three structural choices shape the entire design. Each was chosen for a specific reason; do not change them without surfacing the deviation and asking first.

### D1: Step-Loop Environment (not event-driven callbacks)

The system uses a `while not done: env.step(action)` loop rather than an event-driven callback architecture.

**Rationale:** This maps directly to the Markov Decision Process formulation used in RL. The agent observes a state, selects an action, and passes it back to the engine. The engine never calls the agent — the agent always calls the engine. This provides ultimate control over execution flow, makes state transitions explicit, and allows the same environment to serve both hand-coded strategies and RL training loops without adapter glue.

### D2: Decoupled Decision and Execution (two-phase order lifecycle)

User actions are routed into an `ExecutionBroker` order queue rather than executed immediately against the `Portfolio`.

**Rationale:** Instant execution can only simulate Market-on-Close (MOC) orders — the agent sees price at time T and trades at price T, which is look-ahead bias when the decision is made intraday. By decoupling, an agent issues an order at time T, and the broker holds it until time T+1 to simulate Market-on-Open (MOO) or evaluates it against T+1 prices for limit fills. The two-phase lifecycle (Phase 1: current-step fills; Phase 2: next-step fills) is the structural mechanism that eliminates look-ahead bias — it makes bias impossible rather than relying on user discipline.

### D3: Master Timeline with Forward-Fill Alignment

The `DataHandler` establishes a single integer-indexed master timeline driven by the highest-frequency data feed (e.g., daily equities). Lower-frequency data (e.g., quarterly GDP) is forward-filled to match.

**Rationale:** At every step, the agent must receive a valid, non-null state matrix. Forward-filling is the correct causal operation for slow-updating data: quarterly GDP is genuinely "still the last reported value" until the next release. The master timeline is the single source of truth for "what step are we on?" — all components reference it rather than maintaining independent clocks.

---

## 3. Data Schemas

### 3.1 Action Schema (input to `step()`)

The `step()` function accepts an action dictionary. Two action types are supported: `"order"` (absolute quantities) and `"weight"` (target portfolio percentages). The engine translates weight-based actions into order-based actions internally before submitting to the broker.

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

Weight-based actions specify target portfolio allocation percentages (summing to at most 1.0; the remainder stays in cash). The engine calculates the required trades based on current portfolio equity and positions, then submits the resulting orders to the broker. All orders generated from a weight-based action share the same `execution` timing.

**Execution modes:**

| Mode | Fill price | When processed |
|------|-----------|----------------|
| `current_close` | Close price at time T (the step when the order is submitted) | Phase 1 (before time advances) |
| `next_open` | Open price at time T+1 | Phase 2 (after time advances) |
| `limit` | The `limit_price` specified on the order | Phase 2; filled only if T+1 bar's low ≤ limit_price (buy) or high ≥ limit_price (sell). Unfilled limits expire after one bar (good-for-day semantics). |

### 3.2 State Schema (output from `step()`)

`step()` returns a 4-tuple: `(state, reward, done, info)`.

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

**`reward`** — a float. Default: single-step portfolio return `(equity_t - equity_{t-1}) / equity_{t-1}`. Configurable via a reward function passed to `BacktestEnv.__init__()`.

**`done`** — boolean. `True` when the data handler has exhausted all bars.

**`info`** — dictionary of auxiliary diagnostics for the current step (fills executed this step, slippage, rejected orders, etc.). Not part of the RL state; used for logging and debugging.

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
        """Build the master timeline from the highest-frequency feed's date index.
        Reindex all feeds to the master timeline. Forward-fill lower-frequency
        feeds. Raise on any remaining NaNs after fill (indicates data gap, not
        frequency mismatch)."""

    def advance_time(self) -> bool:
        """Increment current_step by 1. Return True if done (past max_steps)."""

    def get_current_bar(self) -> dict:
        """Return data for exactly the current step. Used by the broker for
        fill-price lookup."""

    def get_window(self) -> dict:
        """Return the data slice [current_step - lookback, current_step] inclusive.
        This is the market_data portion of the state."""

    def get_price(self, asset: str, field: str = "close") -> float:
        """Convenience: return a single price field for one asset at current_step."""

    def reset(self):
        """Reset current_step to lookback (the first valid step)."""
```

**Design notes:**

- The master timeline is derived from the union of all dates in the highest-frequency feed. No synthetic dates are inserted.
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
        Positions track net quantity (positive = long, negative = short)."""

    def update_valuations(self, current_prices: dict[str, float]):
        """Mark all positions to market. Compute equity = cash + sum(pos * price).
        Called once per step after all fills are processed."""

    def get_state(self) -> dict:
        """Return snapshot: cash, equity, positions dict, unrealized PnL."""

    def reset(self):
        """Restore to initial_cash, clear all positions."""
```

**Design notes:**

- `qty` is float to support fractional shares. Whether fractional trading is permitted is a policy decision in `BacktestEnv`, not `Portfolio`.
- No commission or slippage model inside `Portfolio`. Commissions, if added, are applied by the broker before calling `execute_trade` (the portfolio sees the effective price). This keeps the ledger simple and testable.
- Short selling is supported: a negative position is a short. Margin requirements are out of scope for v1.

### 4.3 ExecutionBroker

The order queue. Sits between the agent's desired actions and the portfolio's ledger. Responsible for fill logic and order lifecycle.

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

        Fills are routed to portfolio.execute_trade().
        Filled orders are removed from the queue.
        Unfilled limit orders are expired (good-for-day: if phase 2 doesn't
        fill them, they're dropped)."""

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

### 4.4 BacktestEnv

The orchestrator. Manages the two-phase execution lifecycle and routes data between components. This is the class the agent interacts with.

```python
class BacktestEnv:
    def __init__(
        self,
        data_feeds: dict,
        initial_cash: float = 100_000.0,
        lookback: int = 20,
        reward_fn: Callable | None = None,
        commission_fn: Callable | None = None,
    ):
        self.data_handler = DataHandler(data_feeds, lookback)
        self.portfolio = Portfolio(initial_cash)
        self.broker = ExecutionBroker(commission_fn)
        self.reward_fn = reward_fn or self._default_reward
        self._prev_equity = initial_cash

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """The core progression lifecycle.

        Execution order (6 phases, strictly sequential):

        1. PARSE     — Convert action to order list. Weight-based actions are
                       translated to absolute orders using current portfolio
                       equity and positions. Validated orders are submitted to
                       the broker.

        2. PHASE 1   — broker.process_queue(phase=1). Executes current_close
                       orders against the current bar's close price.

        3. ADVANCE   — data_handler.advance_time(). The time pointer moves
                       from T to T+1. If done, skip to step 6.

        4. PHASE 2   — broker.process_queue(phase=2). Executes next_open
                       orders against T+1's open price. Evaluates and
                       fills/expires limit orders against T+1's bar.

        5. MARK      — portfolio.update_valuations() using T+1's close prices.
                       Compute reward from equity change.

        6. RETURN     — Assemble and return (state, reward, done, info).
        """

    def reset(self) -> dict:
        """Reset all components. Return the initial state (at the first valid
        step, after the lookback window is available)."""

    def _parse_action(self, action: dict) -> list[dict]:
        """Translate action dict into a list of normalized order dicts.
        For weight-based actions: compute target position for each asset
        from (target_weight * current_equity / current_price), diff against
        current position, and emit the delta as orders."""

    def _default_reward(self, prev_equity: float, curr_equity: float) -> float:
        """Default reward: single-step portfolio return."""
        return (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
```

---

## 5. Step Lifecycle — Detailed Walkthrough

This is the most important section of the spec. The six-phase lifecycle in `BacktestEnv.step()` is the mechanism that makes look-ahead-bias elimination structural rather than conventional.

```
Time:  ──────────── T ──────────────┃──────────── T+1 ────────────────
                                    ┃
Agent sees state at T               ┃
Agent calls step(action)            ┃
                                    ┃
  1. PARSE action → orders          ┃
  2. PHASE 1: fill current_close    ┃
     orders at T's close price      ┃
                                    ┃
  ─── time pointer advances ────────╋───
                                    ┃
  3. PHASE 2: fill next_open        ┃
     orders at T+1's open price;    ┃
     evaluate limit orders          ┃
     against T+1's bar              ┃
  4. MARK portfolio to T+1 close    ┃
  5. Compute reward                 ┃
  6. Return (state_T+1, reward,     ┃
     done, info)                    ┃
```

**Why this order matters:**

- `current_close` orders fill at T's close *before* time advances. The agent has already observed T's close price in the state, so this is not look-ahead — it is "I see the close, I trade at the close" (Market-on-Close).
- `next_open` orders fill at T+1's open *after* time advances. The agent has not seen T+1's prices when it issues the order — the broker holds it and fills at the next available open. This is the standard "decide today, execute tomorrow" pattern.
- Limit orders are evaluated against T+1's full bar (OHLC). This is a simplification of intrabar dynamics but is standard for daily-bar backtesting.
- Portfolio is marked to market at T+1's close, which is the price the agent will see in the next state.

---

## 6. Configuration

`BacktestEnv` is configured at construction time via explicit arguments. A YAML-driven config layer may be added later but is not part of v1 — construction arguments are the contract.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_feeds` | `dict` | required | Nested dict of `{feed_name: {asset: DataFrame}}` |
| `initial_cash` | `float` | `100_000.0` | Starting cash balance |
| `lookback` | `int` | `20` | Number of historical bars in the state window |
| `reward_fn` | `Callable` | single-step return | `(prev_equity, curr_equity) -> float` |
| `commission_fn` | `Callable` | `None` (no commission) | `(asset, qty, price) -> float` |

---

## 7. Correctness Constraints

These are non-negotiable invariants, analogous to analog_mc's C1–C6.

**B1: No look-ahead in state.** `get_window()` returns data up to and including `current_step`. No future data is ever included in the state. Enforced by integer indexing on the master timeline.

**B2: Two-phase execution eliminates look-ahead in fills.** `current_close` orders fill before time advances (agent has already seen the price). `next_open` and `limit` orders fill after time advances (agent has not seen the execution price when it submitted the order). No execution mode allows the agent to trade at a price it hasn't yet observed in state and have that fill reflected in the same state observation.

**B3: Causal forward-fill.** Lower-frequency data is forward-filled from past observations only. No backward-fill, no interpolation, no centering. At step T, macro data reflects the most recent observation at or before T.

**B4: Portfolio consistency.** After every `step()`, `portfolio.equity == portfolio.cash + sum(position_qty * mark_price)` for all positions. This is verified by `update_valuations()` at the end of every step.

**B5: Deterministic replay.** Same `data_feeds` + same sequence of `action` dicts → identical sequence of `(state, reward, done, info)` tuples. No internal randomness. (RL agents may be stochastic, but the environment is not.)

---

## 8. Module Layout

Following project conventions (`CLAUDE.md` § "Module namespacing"):

```
src/backtesting/
├── __init__.py
├── data_handler.py      # DataHandler
├── portfolio.py         # Portfolio
├── broker.py            # ExecutionBroker
├── env.py               # BacktestEnv
└── utils.py             # shared helpers (action validation, schema constants)

docs/backtesting/
├── spec.md              # this document
└── goal.md              # what the module optimizes for (to be written)

tests/backtesting/
├── test_data_handler.py
├── test_portfolio.py
├── test_broker.py
├── test_env.py
└── test_correctness.py  # B1–B5 constraint tests

configs/backtesting/     # reserved for future YAML-driven configs
```

---

## 9. Open Questions for v1

These are design decisions that need resolution before or during implementation. They do not block the spec but will shape the implementation plan.

1. **Data source integration.** Should `BacktestEnv` accept raw DataFrames (as specified above), or should it accept `data_pipelines` identifiers and fetch internally? The current spec keeps it DataFrame-in for maximum flexibility; wiring to `data_pipelines` can be a convenience layer on top.

2. **Fractional shares.** `Portfolio.execute_trade` accepts `float` quantities. Should `BacktestEnv` enforce integer-only quantities by default (with a `fractional=True` opt-in), or leave it unconstrained?

3. **Reward function scope.** The default reward is single-step return. RL agents often need shaped rewards (Sharpe ratio over a window, drawdown penalty, etc.). Should the reward function receive the full portfolio history, or just `(prev_equity, curr_equity)`? Richer signatures are more flexible but couple the reward function to internal state.

4. **Multi-asset weight rebalancing.** When a weight-based action is parsed into orders, should the engine handle the sequencing of sells-before-buys to free up cash? Or should it assume sufficient cash/margin and let the portfolio go negative on cash temporarily within a step?

5. **Gym compatibility.** Should `BacktestEnv` formally subclass `gymnasium.Env` and implement `observation_space` / `action_space`? This would make it plug-and-play with Stable Baselines, RLlib, etc., but adds a dependency and constrains the state/action shapes.
