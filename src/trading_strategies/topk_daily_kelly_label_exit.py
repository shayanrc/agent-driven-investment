"""TopKDailyKellyLabelExit — top-K daily-Kelly strategy (plan D10 / §6.5).

A ``backtesting.Strategy``-conformant callable. Each signal day:

1. **Rebalance pass** (existing positions, runs first to release room).
   Per open position, in priority order (D11):
     1. drawdown floor ``close ≤ (1−stop_dd)·anchor`` → full exit
     2. target ``close ≥ (1+target)·anchor`` → full exit
     3. horizon ``bd_held ≥ horizon_days`` → full exit
     4. (D22) ticker absent from today's predictions → skip breakeven+trim
     5. ``p_today ≤ breakeven_p`` → full exit (model dropped conviction)
     6. ``new_kelly_f < cur_f`` → trim down to ``new_kelly_f`` (ratchet-down
        only; never add up)
2. **Entry pass** (new picks, uses freed room). Candidates are tickers in
   today's predictions with ``p_mean > breakeven_p`` not already held,
   sorted ``(p_mean desc, ticker asc)`` — the D21 tie-break, identical to
   ``src/gbdt/topk_diagnostics.py``. Top-K candidates each take
   ``min(intended_f, room)`` notional; dropped if below the D9 floor.

**Anchor convention (D12):** the anchor is the SIGNAL-DAY close — the close
the strategy observes at the decision step (the last row of the lookback
window), NOT the next-open fill price. The anchor is set once at entry and
never updated by trims; a full exit + re-entry gets a fresh anchor.

**Engine contract notes (plan-vs-spec deltas, documented):**

- The plan §6.5 reads ``info["last_close"][ticker]``; the engine's ``info``
  has no such key (spec §3.2). Current close is read from
  ``state["market_data"]["equities"][ticker][-1, close_col]`` instead.
- The strategy emits ``{"type": "order", ...}`` (delta quantities), NOT
  weight actions: weight actions liquidate any held position omitted from
  ``target_weights`` and continuously rebalance held shares to a target
  weight, which conflicts with "hold shares fixed until a trigger fires."
  Order actions give "hold = emit nothing."
- Notional sizing: ``notional_f = fractional_c · f_risk / payoff_loss``,
  where ``f_risk`` is the sizer's fraction-at-risk. This conversion lives
  here (the plan inlines it in §6.5); the Kelly formula stays in the sizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_strategies import PerPredictionSizer, PortfolioSizer

PredictionRow = tuple[str, float, float, float]  # (ticker, p_mean, p_low, p_high)


@dataclass
class StrategyEvent:
    """One recorded action for the memo's turnover / per-pick tables."""

    step: int
    date: pd.Timestamp
    ticker: str
    kind: str  # "entry" | "exit" | "trim"
    p_today: float | None
    anchor_close: float | None
    close: float | None
    shares_before: float
    shares_after: float
    notional_f: float | None
    trigger: str | None = None  # for exits: DD|target|horizon|breakeven


class TopKDailyKellyLabelExit:
    """Top-K daily-Kelly strategy with ratchet-down trims + label exits."""

    def __init__(
        self,
        predictions: dict[pd.Timestamp, list[PredictionRow]],
        K: int,
        target_return: float,
        stop_drawdown: float,
        horizon_days: int,
        sizer: PerPredictionSizer | PortfolioSizer,
        sizer_payoffs: tuple[float, float],  # (win, loss)
        breakeven_p: float,
        fractional_c: float = 0.5,
        gross_cap: float = 1.0,
        floor_pct_equity: float = 0.05,
        floor_pct_room: float = 0.10,
        close_col: int = 3,  # OHLCV → close is index 3
        enable_rebalance: bool = True,  # §7 "daily rebalance OFF" counterfactual
        cash_buffer: float = 0.02,
        selection_bound: str = "mean",  # "mean" | "low" (plan §10 Q10a)
        selection_mode: str = "breakeven",  # "breakeven" | "rank" (V1.2)
        sizing_mode: str = "kelly",  # "kelly"|"equal"|"rank_kelly"|"prob_weight"
        rank_kelly_p: float | None = None,  # eval hit-rate for rank_kelly sizing
        prob_weight_alpha: float = 1.0,  # prob_weight sharpness: weight ∝ p**alpha (_014)
        enable_breakeven_exit: bool | None = None,  # None → auto (off in rank mode)
        rank_scores: dict[pd.Timestamp, dict[str, float]] | None = None,
        vol_scores: dict[pd.Timestamp, dict[str, float]] | None = None,
    ) -> None:
        self._preds = {pd.Timestamp(k): v for k, v in predictions.items()}
        # Optional per-(date, ticker) ranking score that overrides p_mean for the
        # entry top-K sort ONLY (sizing + breakeven gate stay on p_mean). Used to
        # rank by the finest-resolution raw model score instead of the quantized
        # calibrated p (rank_by="raw"); see docs/backtests/_020. A missing ticker
        # for a day falls back to that row's p_mean.
        self._rank_scores = (
            {pd.Timestamp(k): dict(v) for k, v in rank_scores.items()}
            if rank_scores is not None else None
        )
        # Optional per-(date, ticker) realized-volatility used by sizing_mode=
        # "inverse_vol": each of the day's selected names gets a slice ∝ 1/vol,
        # normalized so the book targets fractional_c·gross_cap (risk parity).
        # Keeps raw SELECTION but damps the high-beta names that rank_by="raw"
        # surfaces (see docs/backtests/_021). Missing/≤0 vol → mean-of-present
        # fallback (≈ equal weight for that name).
        self._vol_scores = (
            {pd.Timestamp(k): dict(v) for k, v in vol_scores.items()}
            if vol_scores is not None else None
        )
        self.K = K
        self.target_return = target_return
        self.stop_drawdown = stop_drawdown
        self.horizon_days = horizon_days
        self.sizer = sizer
        self.payoff_win, self.payoff_loss = sizer_payoffs
        self.breakeven_p = breakeven_p
        self.fractional_c = fractional_c
        self.gross_cap = gross_cap
        self.floor_pct_equity = floor_pct_equity
        self.floor_pct_room = floor_pct_room
        self.close_col = close_col
        self.enable_rebalance = enable_rebalance
        if selection_bound not in ("mean", "low"):
            raise ValueError(
                f"selection_bound must be 'mean' or 'low'; got {selection_bound!r}"
            )
        # Which probability the ENTRY filter clears against breakeven (D2/§10
        # Q10a). "mean" (default) filters on p_mean; "low" filters on the 2.5%
        # credible bound p_low (conservative — rejects picks whose lower band
        # doesn't clear breakeven). Ranking + sizing stay on p_mean either way.
        self.selection_bound = selection_bound

        # --- V1.2 rank-sizing modes (rare-event cells whose calibrated p never
        # clears the absolute Kelly breakeven; see _004 memo). ---------------
        if selection_mode not in ("breakeven", "rank"):
            raise ValueError(
                f"selection_mode must be 'breakeven' or 'rank'; got {selection_mode!r}"
            )
        if sizing_mode not in ("kelly", "equal", "rank_kelly", "prob_weight", "inverse_vol"):
            raise ValueError(
                "sizing_mode must be 'kelly', 'equal', 'rank_kelly', "
                f"'prob_weight' or 'inverse_vol'; got {sizing_mode!r}"
            )
        if sizing_mode == "rank_kelly" and rank_kelly_p is None:
            raise ValueError("sizing_mode='rank_kelly' requires rank_kelly_p (eval hit-rate)")
        if sizing_mode == "inverse_vol" and vol_scores is None:
            raise ValueError("sizing_mode='inverse_vol' requires vol_scores (per-date,ticker realized vol)")
        # "rank" selection: take the day's top-K by p_mean WITHOUT the absolute
        # p > breakeven gate, so a strongly-ranked rare-event cell still trades.
        self.selection_mode = selection_mode
        # "equal": each position = fractional_c · gross_cap / K (deploy the rank,
        # don't size by absolute p). "rank_kelly": Kelly on the rank bucket's
        # empirical hit-rate rank_kelly_p instead of the per-row calibrated p.
        self.sizing_mode = sizing_mode
        self.rank_kelly_p = rank_kelly_p
        # "prob_weight" sharpness (_014): weight ∝ p**alpha among the day's top-K.
        # alpha=1 → raw p (flat → spreads, the _013 failure); alpha>1 amplifies the
        # ranking's relative gaps so the book concentrates on the highest-p picks
        # even when calibrated p is nearly flat across candidates.
        if prob_weight_alpha <= 0:
            raise ValueError(f"prob_weight_alpha must be > 0; got {prob_weight_alpha}")
        self.prob_weight_alpha = prob_weight_alpha
        # In rank mode the per-row p is (by construction) below breakeven, so the
        # breakeven EXIT would fire on every position immediately — disable it
        # there by default. Caller can override explicitly.
        self.enable_breakeven_exit = (
            (selection_mode != "rank") if enable_breakeven_exit is None
            else enable_breakeven_exit
        )
        # next_open fills above the signal-day close in an uptrend; an order
        # sized to consume ~all cash at the close price overdraws at the open
        # and the engine rejects the WHOLE order. cash_buffer reserves a slice
        # of cash so near-cap entries still fill. Realized gross exposure is
        # therefore capped at ~(1 - cash_buffer); occasional overnight gaps
        # larger than the buffer still produce a rejected entry (surfaced in
        # info["rejected_overdraw"]). Plan §6.4 sizes off equity; this is the
        # cash-constraint reality under next_open (documented Stage-7 delta).
        if not 0.0 <= cash_buffer < 1.0:
            raise ValueError(f"cash_buffer must be in [0, 1); got {cash_buffer}")
        self.cash_buffer = cash_buffer

        # ticker -> {entry_step, anchor_close, f (booked notional fraction)}
        self._open: dict[str, dict[str, Any]] = {}
        self.events: list[StrategyEvent] = []

    def reset(self) -> None:
        self._open.clear()
        self.events.clear()

    # -- sizing --------------------------------------------------------------
    def _f_risk(self, p: float) -> float:
        if isinstance(self.sizer, PortfolioSizer):
            return self.sizer.per_position_fraction_at_risk
        return self.sizer.fraction_at_risk(
            p, payoff_win=self.payoff_win, payoff_loss=self.payoff_loss
        )

    def _notional_f(self, p: float) -> float:
        """Per-position notional fraction of equity, by sizing_mode.

        - ``kelly`` (default, plan §6.5): ``fractional_c · f_risk(p) / payoff_loss``.
        - ``equal``: ``fractional_c · gross_cap / K`` — equal slices, p-independent
          (deploy the rank, not the absolute probability).
        - ``rank_kelly``: Kelly on the rank bucket's empirical hit-rate
          ``rank_kelly_p`` instead of the per-row calibrated ``p``.
        """
        if self.sizing_mode == "equal":
            return self.fractional_c * self.gross_cap / self.K
        if self.sizing_mode == "rank_kelly":
            return self.fractional_c * self._f_risk(self.rank_kelly_p) / self.payoff_loss
        return self.fractional_c * self._f_risk(p) / self.payoff_loss

    # -- helpers -------------------------------------------------------------
    def _close(self, state: dict, ticker: str) -> float | None:
        md = state.get("market_data", {}).get("equities", {})
        arr = md.get(ticker)
        if arr is None or len(arr) == 0:
            return None
        c = float(arr[-1, self.close_col])
        if c != c or c <= 0.0:  # NaN or non-positive
            return None
        return c

    # -- the callback --------------------------------------------------------
    def __call__(self, state: dict, info: dict) -> dict | None:
        ts = pd.Timestamp(state["timestamp"])
        step = int(state["step"])
        equity = float(state["portfolio"]["equity"])
        positions = state["portfolio"]["positions"]
        rows = self._preds.get(ts, [])
        p_today = {tk: pm for (tk, pm, _lo, _hi) in rows}

        orders: list[dict[str, Any]] = []
        exited_today: set[str] = set()  # D14: no same-day re-entry after exit

        # === REBALANCE PASS (existing positions) =========================
        if self.enable_rebalance:
            for tk in list(self._open):
                pos = self._open[tk]
                shares = float(positions.get(tk, 0.0))
                if shares == 0.0:
                    # Entry submitted on a prior step not yet filled (or already
                    # liquidated by the engine). Nothing to rebalance yet.
                    continue
                c = self._close(state, tk)
                if c is None:
                    # No price today (delisting gap) — engine's gap_policy owns
                    # liquidation; we keep tracking and skip exit math (D22-like
                    # for prices). DD/target/horizon can't be evaluated without
                    # a close.
                    continue
                anchor = pos["anchor_close"]
                bd_held = step - pos["entry_step"]

                trigger: str | None = None
                if c <= (1.0 - self.stop_drawdown) * anchor:
                    trigger = "DD"
                elif c >= (1.0 + self.target_return) * anchor:
                    trigger = "target"
                elif bd_held >= self.horizon_days:
                    trigger = "horizon"
                elif tk not in p_today:
                    trigger = None  # D22: skip breakeven + trim this day
                elif self.enable_breakeven_exit and p_today[tk] <= self.breakeven_p:
                    trigger = "breakeven"

                if trigger is not None:
                    orders.append({"asset": tk, "qty": -shares})
                    self.events.append(
                        StrategyEvent(
                            step, ts, tk, "exit", p_today.get(tk), anchor, c,
                            shares, 0.0, None, trigger=trigger,
                        )
                    )
                    del self._open[tk]
                    exited_today.add(tk)
                    continue

                # TRIM pass — ratchet-down only (D13). Only if p_today present.
                # Skip for the daily-normalized weight modes (prob_weight, inverse_vol):
                # their per-name weight is set over the day's selected set at entry and
                # has no per-position _notional_f form, so _notional_f would fall through
                # to the Kelly target (≈0 on sub-breakeven cells) and trim every position
                # to ~0 — freeing room and re-entering K names daily (a churn artifact,
                # not a strategy: it spuriously spread prob_weight across the whole
                # universe in _013/_014; corrected per docs/backtests/_023).
                if tk in p_today and self.sizing_mode not in ("inverse_vol", "prob_weight"):
                    new_f = self._notional_f(p_today[tk])
                    cur_f = (shares * c) / equity if equity > 0 else 0.0
                    if new_f < cur_f:
                        target_shares = (new_f * equity) / c
                        delta = target_shares - shares  # negative → sell
                        if delta < 0:
                            orders.append({"asset": tk, "qty": delta})
                            self.events.append(
                                StrategyEvent(
                                    step, ts, tk, "trim", p_today[tk],
                                    anchor, c, shares, target_shares, new_f,
                                )
                            )
                            pos["f"] = new_f

        # === ENTRY PASS (new picks) ======================================
        exposure = sum(p["f"] for p in self._open.values())
        room = self.gross_cap - exposure
        # Cash actually spendable on new buys this step (the engine rejects
        # overdraw on the whole order). Decremented across same-day entries.
        avail_cash = float(state["portfolio"].get("cash", equity)) * (
            1.0 - self.cash_buffer
        )

        # D21 ordering. The RANK key defaults to p_mean, but can be overridden by
        # an external per-(date,ticker) score (rank_by="raw" → the finest-resolution
        # raw model score). Conditional-isotonic calibration quantizes p_mean into
        # wide tied plateaus, so ranking on it degenerates to the alphabetical
        # tie-break; ranking on the raw score recovers the within-plateau ordering
        # (see docs/backtests/_020). Sizing + the breakeven gate stay on p_mean.
        day_scores = self._rank_scores.get(ts) if self._rank_scores is not None else None

        def _rank_key(tk: str, pm: float) -> float:
            return day_scores.get(tk, pm) if day_scores is not None else pm

        candidates = sorted(
            [
                (tk, pm)
                for (tk, pm, lo, _hi) in rows
                # "rank" mode: no absolute gate — take the day's top-K by p_mean
                # regardless of whether p clears breakeven (V1.2). "breakeven"
                # mode: filter on the selected bound (p_mean or p_low).
                if (self.selection_mode == "rank"
                    or (lo if self.selection_bound == "low" else pm) > self.breakeven_p)
                and tk not in self._open
                and tk not in exited_today  # D14: not the same day
            ],
            key=lambda x: (-_rank_key(x[0], x[1]), x[0]),  # rank desc, ticker asc
        )
        selected = candidates[: self.K]
        # "prob_weight" (_013): size each of the day's top-K ∝ its calibrated p,
        # normalized so the book targets fractional_c·gross_cap. Unlike Kelly, it
        # never zeros a sub-breakeven pick (so rare-event cells still deploy), and
        # it AUTO-adapts concentration to precision: peaked p (high-precision day)
        # → concentrated weights; flat p → near-equal spread. Normalizer needs the
        # whole day's selected set, so it's computed here, not in _notional_f.
        prob_w: dict[str, float] = {}
        if self.sizing_mode == "prob_weight":
            a = self.prob_weight_alpha
            wsum = sum(pm ** a for _, pm in selected)
            if wsum > 0.0:
                prob_w = {tk: self.fractional_c * self.gross_cap * (pm ** a / wsum)
                          for tk, pm in selected}
        # "inverse_vol" (_022): risk-parity — slice ∝ 1/realized-vol, normalized so
        # the book targets fractional_c·gross_cap. Selection is unchanged (so
        # rank_by="raw" still picks the high-hit-rate names) but the high-vol/high-
        # beta names get smaller slices. Missing/≤0 vol → mean-of-present fallback
        # (that name lands ≈ equal weight). Normalizer needs the whole selected set.
        vol_w: dict[str, float] = {}
        if self.sizing_mode == "inverse_vol":
            day_vol = self._vol_scores.get(ts, {}) if self._vol_scores is not None else {}
            present = [day_vol[tk] for tk, _ in selected
                       if day_vol.get(tk, 0.0) and day_vol[tk] > 0.0]
            fallback = (sum(present) / len(present)) if present else 1.0
            inv = {tk: 1.0 / (day_vol[tk] if day_vol.get(tk, 0.0) and day_vol[tk] > 0.0
                              else fallback)
                   for tk, _ in selected}
            isum = sum(inv.values())
            if isum > 0.0:
                vol_w = {tk: self.fractional_c * self.gross_cap * (inv[tk] / isum)
                         for tk, _ in selected}
        equal_slice = self.fractional_c * self.gross_cap / self.K
        for tk, pm in selected:
            if room <= 0.0:
                break
            if self.sizing_mode == "prob_weight":
                intended_f = prob_w.get(tk, 0.0)
                # dust cut at half the equal slice: drop a pick whose p-share is
                # below half the average allocation (genuine tail), keep the tilt.
                floor = 0.5 * equal_slice
            elif self.sizing_mode == "inverse_vol":
                intended_f = vol_w.get(tk, 0.0)
                # gentle floor (10% of equal) — risk parity intentionally makes the
                # high-vol names small; a 0.5·equal floor would clip exactly them.
                floor = 0.1 * equal_slice
            else:
                intended_f = self._notional_f(pm)
                if self.sizing_mode == "equal":
                    # Equal slices are intentionally ~gross_cap/K. The K-independent
                    # dust floor (max(5% equity, 10% room)) wrongly rejects every
                    # legitimate slice at wide K — e.g. K=20 → 5% slices < 10%-of-room
                    # floor → nothing enters, room never shrinks, 0 entries forever.
                    # Floor at half the intended slice instead: a full 1/K slice
                    # clears, a room/cash-squeezed slice is still dropped as dust.
                    floor = 0.5 * intended_f
                else:
                    floor = max(self.floor_pct_equity, self.floor_pct_room * room)
            actual_f = min(intended_f, room)
            if actual_f < floor:
                continue  # drop this entry, try the next candidate
            c = self._close(state, tk)
            if c is None:
                continue
            # Cap the order cost at spendable cash so the next_open fill
            # doesn't overdraw (which would reject the whole order).
            notional = min(actual_f * equity, avail_cash)
            if notional <= 0.0:
                continue
            shares = notional / c
            orders.append({"asset": tk, "qty": shares})
            self._open[tk] = {
                "entry_step": step,
                "anchor_close": c,  # D12: signal-day close
                "f": actual_f,  # booked intended fraction for room accounting
            }
            self.events.append(
                StrategyEvent(
                    step, ts, tk, "entry", pm, c, c, 0.0, shares, actual_f
                )
            )
            room -= actual_f
            avail_cash -= notional

        if not orders:
            return None
        return {"type": "order", "orders": orders}
