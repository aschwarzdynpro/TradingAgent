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
    """Return a copy of ``df`` with sma/rsi/atr columns appended.

    ``df`` must have columns: open, high, low, close (DatetimeIndex sorted asc).
    """
    out = df.copy()
    out["sma_trend"] = sma(out["close"], p.sma_trend)
    out["rsi"] = rsi(out["close"], p.rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_period)
    return out


def _min_bars(p: StrategyConfig) -> int:
    """Bars required before any indicator is fully warmed up."""
    return max(p.sma_trend, p.rsi_period, p.atr_period) + 1


def evaluate(
    df: pd.DataFrame,
    position: PositionState | None,
    p: StrategyConfig,
) -> SignalResult:
    """Evaluate the strategy on the last completed bar of ``df``.

    ``df`` may be raw bars (indicators are computed on the fly) or already carry
    sma_trend/rsi/atr columns. Pass ``position=None`` when flat.
    """
    if "rsi" not in df.columns or "sma_trend" not in df.columns or "atr" not in df.columns:
        df = compute_indicators(df, p)

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

    snapshot = {"close": close, "sma_trend": sma_v, "rsi": rsi_v, "atr": atr_v}

    if pd.isna(sma_v) or pd.isna(rsi_v) or pd.isna(atr_v):
        return SignalResult(symbol, SignalType.HOLD, "indicators_not_ready", close, asof, snapshot)

    # ── In a position: check exits (any one triggers) ────────────────────────
    if position is not None:
        new_high = max(position.highest_high, high)
        snapshot["highest_high"] = new_high

        if rsi_v > p.rsi_exit:
            return SignalResult(symbol, SignalType.EXIT, "rsi_exit", close, asof, snapshot, new_high)

        trail_stop = new_high - p.atr_mult * atr_v
        snapshot["trail_stop"] = trail_stop
        if close < trail_stop:
            return SignalResult(symbol, SignalType.EXIT, "atr_trailing_stop", close, asof, snapshot, new_high)

        if close < sma_v:
            return SignalResult(symbol, SignalType.EXIT, "trend_break", close, asof, snapshot, new_high)

        return SignalResult(symbol, SignalType.HOLD, "in_position", close, asof, snapshot, new_high)

    # ── Flat: check entry (all conditions required) ──────────────────────────
    uptrend = close > sma_v
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
