"""Signal logic.

Trend regime filter + RSI mean-reversion entry + RSI/ATR-trail/trend-break exit.
Long only in v1 — no shorting.

The same ``evaluate`` function is used by the backtest, the paper run and the
live run. There is intentionally no separate "live" or "backtest" code path, so
the two cannot diverge. All thresholds come from ``StrategyConfig`` (the YAML
config), never from constants here.

Look-ahead safety: ``evaluate`` only ever inspects the *last row* of the bars it
is handed. Callers must pass bars that end at the last *completed* daily bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from .config import StrategyConfig
from .indicators import atr, rsi, sma


class SignalType(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class PositionState:
    """Minimal position context the strategy needs to evaluate exits."""

    symbol: str
    quantity: float
    entry_price: float
    entry_date: date
    highest_high: float  # highest *high* reached since entry (for the ATR trail)


@dataclass
class SignalResult:
    symbol: str
    signal: SignalType
    reason: str
    price: float  # reference price = close of the last completed bar
    asof: date
    indicators: dict = field(default_factory=dict)
    # Updated trailing-stop high (max of prior high and this bar's high). Callers
    # persist this so the trail ratchets correctly across runs.
    new_highest_high: float | None = None


def compute_indicators(df: pd.DataFrame, p: StrategyConfig) -> pd.DataFrame:
    """Return a copy of ``df`` with sma/rsi/atr/momentum columns appended.

    ``df`` must have columns: open, high, low, close (DatetimeIndex sorted asc).
    ``momentum`` is the classic 12-1 style return: the change from ``mom_lookback``
    bars ago to ``mom_skip`` bars ago (skipping the most recent ``mom_skip`` days to
    avoid short-term reversal). It is unused in mean-reversion mode.
    """
    out = df.copy()
    out["sma_trend"] = sma(out["close"], p.sma_trend)
    out["rsi"] = rsi(out["close"], p.rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_period)
    out["momentum"] = out["close"].shift(p.mom_skip) / out["close"].shift(p.mom_lookback) - 1.0
    return out


def compute_regime(close: pd.Series, sma_period: int) -> pd.Series:
    """Risk-on (True) when ``close`` is above its SMA(sma_period). Index-aligned
    to ``close``; the warm-up span is risk-off (NaN comparison -> False)."""
    return close > sma(close, sma_period)


def _min_bars(p: StrategyConfig) -> int:
    """Bars required before the mode's indicators are fully warmed up."""
    if p.mode == "trend_momentum":
        return max(p.sma_trend, p.mom_lookback, p.atr_period) + 1
    return max(p.sma_trend, p.rsi_period, p.atr_period) + 1


def evaluate(
    df: pd.DataFrame,
    position: PositionState | None,
    p: StrategyConfig,
    market_risk_on: bool | None = None,
) -> SignalResult:
    """Evaluate the strategy on the last completed bar of ``df``.

    ``df`` may be raw bars (indicators are computed on the fly) or already carry
    sma_trend/rsi/atr columns. Pass ``position=None`` when flat.

    ``market_risk_on`` is the regime-filter input (the market index above its SMA):
    only honoured when ``p.use_regime_filter`` is set, and only an explicit
    ``False`` is treated as risk-off (``None`` leaves the filter inert).
    """
    if any(c not in df.columns for c in ("sma_trend", "rsi", "atr")):
        df = compute_indicators(df, p)  # also adds momentum
    elif "momentum" not in df.columns:
        # Indicators were supplied but momentum wasn't — add only that column so
        # caller-provided values are not clobbered.
        df = df.copy()
        df["momentum"] = df["close"].shift(p.mom_skip) / df["close"].shift(p.mom_lookback) - 1.0

    symbol = position.symbol if position else (df.attrs.get("symbol") or "?")
    asof = _row_date(df.index[-1])

    if len(df) < _min_bars(p):
        return SignalResult(symbol, SignalType.HOLD, "insufficient_history", float("nan"), asof)

    last = df.iloc[-1]
    close = float(last["close"])
    high = float(last["high"])
    sma_v = float(last["sma_trend"])
    rsi_v = float(last["rsi"])
    atr_v = float(last["atr"])
    mom_v = float(last["momentum"])

    momentum_mode = p.mode == "trend_momentum"
    snapshot = {"close": close, "sma_trend": sma_v, "atr": atr_v}
    snapshot["momentum" if momentum_mode else "rsi"] = mom_v if momentum_mode else rsi_v

    timing = mom_v if momentum_mode else rsi_v
    if pd.isna(sma_v) or pd.isna(atr_v) or pd.isna(timing):
        return SignalResult(symbol, SignalType.HOLD, "indicators_not_ready", close, asof, snapshot)

    risk_off = p.use_regime_filter and market_risk_on is False
    snapshot["market_risk_on"] = market_risk_on

    # ── In a position: check exits (any one triggers) ────────────────────────
    if position is not None:
        new_high = max(position.highest_high, high)
        snapshot["highest_high"] = new_high

        # Defensive regime exit: market risk-off flattens the position first.
        if risk_off and p.regime_exit:
            return SignalResult(symbol, SignalType.EXIT, "regime_exit", close, asof, snapshot, new_high)

        # RSI mean-reversion exit only applies in mean-reversion mode; momentum
        # mode rides the trend and exits on the trail or a trend break.
        if not momentum_mode and rsi_v > p.rsi_exit:
            return SignalResult(symbol, SignalType.EXIT, "rsi_exit", close, asof, snapshot, new_high)

        trail_stop = new_high - p.atr_mult * atr_v
        snapshot["trail_stop"] = trail_stop
        if close < trail_stop:
            return SignalResult(symbol, SignalType.EXIT, "atr_trailing_stop", close, asof, snapshot, new_high)

        if close < sma_v:
            return SignalResult(symbol, SignalType.EXIT, "trend_break", close, asof, snapshot, new_high)

        return SignalResult(symbol, SignalType.HOLD, "in_position", close, asof, snapshot, new_high)

    # ── Flat: check entry ─────────────────────────────────────────────────────
    if risk_off:
        return SignalResult(symbol, SignalType.HOLD, "regime_risk_off", close, asof, snapshot)

    uptrend = close > sma_v
    if momentum_mode:
        if uptrend and mom_v > p.mom_threshold:
            return SignalResult(symbol, SignalType.ENTER_LONG, "trend_momentum", close, asof, snapshot)
        reason = "no_uptrend" if not uptrend else "momentum_not_positive"
        return SignalResult(symbol, SignalType.HOLD, reason, close, asof, snapshot)

    oversold = rsi_v < p.rsi_entry
    if uptrend and oversold:
        return SignalResult(symbol, SignalType.ENTER_LONG, "trend_up_rsi_oversold", close, asof, snapshot)
    if not uptrend:
        reason = "no_uptrend"
    elif not oversold:
        reason = "rsi_not_oversold"
    else:
        reason = "no_entry"
    return SignalResult(symbol, SignalType.HOLD, reason, close, asof, snapshot)


def _row_date(idx) -> date:
    if isinstance(idx, pd.Timestamp):
        return idx.date()
    if hasattr(idx, "date"):
        return idx.date()
    return idx
