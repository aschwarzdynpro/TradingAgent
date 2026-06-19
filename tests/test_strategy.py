"""Strategy entry/exit logic tests with synthetic, fully-controlled bars."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.config import StrategyConfig
from src.strategy import PositionState, SignalType, evaluate


# Small periods so a short synthetic frame is enough to pass the warm-up gate.
P = StrategyConfig(
    sma_trend=5, rsi_period=3, rsi_entry=30, rsi_exit=55, atr_period=3, atr_mult=3.0
)


def make_df(close, high, low, sma_trend, rsi, atr, n=10):
    """Build an indicator-augmented frame whose LAST row carries the given
    values (earlier rows are filler — evaluate only reads the last row)."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "sma_trend": [100.0] * n,
            "rsi": [50.0] * n,
            "atr": [1.0] * n,
        },
        index=idx,
    )
    df.iloc[-1, df.columns.get_loc("close")] = close
    df.iloc[-1, df.columns.get_loc("high")] = high
    df.iloc[-1, df.columns.get_loc("low")] = low
    df.iloc[-1, df.columns.get_loc("sma_trend")] = sma_trend
    df.iloc[-1, df.columns.get_loc("rsi")] = rsi
    df.iloc[-1, df.columns.get_loc("atr")] = atr
    df.attrs["symbol"] = "TEST"
    return df


def pos(highest_high=110.0, entry_price=105.0):
    return PositionState(
        symbol="TEST",
        quantity=10,
        entry_price=entry_price,
        entry_date=date(2024, 1, 1),
        highest_high=highest_high,
    )


# ── Entry ────────────────────────────────────────────────────────────────────
def test_enter_long_when_uptrend_and_oversold():
    df = make_df(close=105, high=105, low=104, sma_trend=100, rsi=25, atr=1.0)
    r = evaluate(df, None, P)
    assert r.signal is SignalType.ENTER_LONG
    assert r.reason == "trend_up_rsi_oversold"


def test_no_entry_when_rsi_exactly_at_threshold():
    # Entry requires RSI strictly < rsi_entry. RSI == 30 must NOT trigger.
    df = make_df(close=105, high=105, low=104, sma_trend=100, rsi=30.0, atr=1.0)
    r = evaluate(df, None, P)
    assert r.signal is SignalType.HOLD
    assert r.reason == "rsi_not_oversold"


def test_no_entry_when_not_uptrend():
    df = make_df(close=95, high=95, low=94, sma_trend=100, rsi=25, atr=1.0)
    r = evaluate(df, None, P)
    assert r.signal is SignalType.HOLD
    assert r.reason == "no_uptrend"


def test_no_entry_when_not_oversold():
    df = make_df(close=105, high=105, low=104, sma_trend=100, rsi=45, atr=1.0)
    r = evaluate(df, None, P)
    assert r.signal is SignalType.HOLD
    assert r.reason == "rsi_not_oversold"


# ── Exit ─────────────────────────────────────────────────────────────────────
def test_exit_on_rsi_target():
    df = make_df(close=112, high=112, low=111, sma_trend=100, rsi=60, atr=1.0)
    r = evaluate(df, pos(highest_high=112), P)
    assert r.signal is SignalType.EXIT
    assert r.reason == "rsi_exit"


def test_exit_on_atr_trailing_stop():
    # highest_high=120, atr=2, k=3 -> stop at 120 - 6 = 114. close 113 < 114.
    df = make_df(close=113, high=113, low=112, sma_trend=100, rsi=50, atr=2.0)
    r = evaluate(df, pos(highest_high=120), P)
    assert r.signal is SignalType.EXIT
    assert r.reason == "atr_trailing_stop"


def test_no_exit_when_above_trailing_stop():
    # stop at 120 - 6 = 114, close 116 > 114, rsi neutral, close > sma -> HOLD
    df = make_df(close=116, high=116, low=115, sma_trend=100, rsi=50, atr=2.0)
    r = evaluate(df, pos(highest_high=120), P)
    assert r.signal is SignalType.HOLD
    assert r.reason == "in_position"


def test_exit_on_trend_break_even_at_entry_day():
    # Trend break: close < sma. rsi below exit, close above trail -> only trend break.
    df = make_df(close=99, high=99, low=98, sma_trend=100, rsi=50, atr=2.0)
    p = PositionState("TEST", 10, 100.0, date(2024, 1, 1), highest_high=100.0)
    r = evaluate(df, p, P)
    assert r.signal is SignalType.EXIT
    assert r.reason == "trend_break"


def test_trailing_high_ratchets_with_new_bar_high():
    df = make_df(close=130, high=135, low=129, sma_trend=100, rsi=50, atr=2.0)
    r = evaluate(df, pos(highest_high=120), P)
    # New bar high 135 > stored 120 -> highest_high updates to 135.
    assert r.new_highest_high == pytest.approx(135.0)


# ── Guards ───────────────────────────────────────────────────────────────────
def test_insufficient_history():
    df = make_df(close=105, high=105, low=104, sma_trend=100, rsi=25, atr=1.0, n=3)
    r = evaluate(df, None, P)
    assert r.signal is SignalType.HOLD
    assert r.reason == "insufficient_history"


def test_nan_indicators_hold():
    df = make_df(close=105, high=105, low=104, sma_trend=float("nan"), rsi=25, atr=1.0)
    r = evaluate(df, None, P)
    assert r.signal is SignalType.HOLD
    assert r.reason == "indicators_not_ready"
